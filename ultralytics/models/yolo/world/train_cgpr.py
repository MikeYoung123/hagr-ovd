# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Coarse-guided partial-label ranking (CGPR) training for YOLO-World."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.nn.modules.cgpr import attach_cgpr_head
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
from ultralytics.utils.torch_utils import unwrap_model


def normalize_phrase(value: Any) -> str:
    """Normalize cache keys while retaining word order."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower()).strip(" .,:;|\"'")


class CGPRLoss(v8DetectionLoss):
    """Detection plus objectness assignment, category prior and one-RoI-per-GT ranking."""

    def __init__(self, model: nn.Module, cache: dict[str, Any]) -> None:
        super().__init__(model)
        self.head = model.model[-1]
        self.object_assigner = TaskAlignedAssigner(
            topk=10, num_classes=1, alpha=0.5, beta=6.0, stride=self.stride.tolist()
        )
        self.phrase_prototypes = F.normalize(cache["phrase_embeddings"].to(self.device).float(), dim=-1)
        self.category_prototypes = F.normalize(cache["category_embeddings"].to(self.device).float(), dim=-1)
        self.assignment_objectness_weight = 0.0
        self.object_gain = float(self.hyp.cgpr_object_gain)
        self.category_gain = float(self.hyp.cgpr_category_gain)
        self.fine_gain = float(self.hyp.cgpr_fine_gain)
        self.fine_margin = float(self.hyp.cgpr_fine_margin)
        self.fine_topk = int(self.hyp.cgpr_fine_hard_topk)
        self.object_gamma = float(self.hyp.cgpr_object_gamma)
        self.object_negative_weight = float(self.hyp.cgpr_object_negative_weight)

    def _group(self, values: torch.Tensor, batch_idx: torch.Tensor, batch_size: int, fill: int) -> torch.Tensor:
        """Group flat instance values into the assigner's padded GT layout."""
        counts = torch.bincount(batch_idx, minlength=batch_size)
        max_boxes = int(counts.max().item()) if counts.numel() else 0
        output = torch.full(
            (batch_size, max_boxes, *values.shape[1:]), fill, dtype=values.dtype, device=self.device
        )
        if values.numel() == 0 or max_boxes == 0:
            return output
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets[1:] = counts.cumsum(0)
        within = torch.arange(values.shape[0], device=self.device) - offsets[batch_idx]
        output[batch_idx, within] = values
        return output

    def _assign(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple:
        pred_dist = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_raw = preds["scores"].permute(0, 2, 1).contiguous()
        pred_obj = preds["cgpr_objectness"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_raw.shape[0]
        dtype = pred_raw.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_dist)

        object_scores = pred_obj.detach().sigmoid().pow(float(self.assignment_objectness_weight))
        object_labels = torch.zeros_like(gt_labels)
        _, target_bboxes, target_obj, fg_mask, target_gt_idx = self.object_assigner(
            object_scores,
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            object_labels,
            gt_bboxes,
            mask_gt,
        )
        return (
            pred_dist,
            pred_raw,
            pred_obj,
            pred_bboxes,
            anchor_points,
            stride_tensor,
            imgsz,
            gt_labels,
            gt_bboxes,
            target_bboxes,
            target_obj,
            fg_mask,
            target_gt_idx,
        )

    def _raw_targets(
        self,
        pred_raw: torch.Tensor,
        gt_labels: torch.Tensor,
        target_obj: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, anchors, _ = pred_raw.shape
        raw_targets = torch.zeros_like(pred_raw)
        if fg_mask.any() and gt_labels.shape[1]:
            image = torch.arange(batch_size, device=self.device)[:, None]
            labels = gt_labels[image, target_gt_idx.clamp(0, gt_labels.shape[1] - 1), 0].long()
            raw_targets.scatter_(2, labels.clamp(0, self.nc - 1).unsqueeze(-1), target_obj)
            raw_targets *= fg_mask[..., None]
            raw_targets = self.apply_multi_positive_targets(
                raw_targets, fg_mask, target_gt_idx, batch, batch_size, gt_labels.shape[1]
            )
        return raw_targets

    def _objectness_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = logits.sigmoid()
        p_t = target * probability + (1 - target) * (1 - probability)
        weights = torch.where(target > 0, target.clamp_min(0.05), logits.new_full((), self.object_negative_weight))
        return (bce * weights * (1 - p_t).pow(self.object_gamma)).sum() / target.sum().clamp_min(1)

    def _category_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        category_ids = batch.get("cgpr_category_id")
        if category_ids is None or not fg_mask.any():
            return preds["scores"].new_zeros(())
        batch_idx = batch["batch_idx"].to(self.device).long()
        grouped = self._group(category_ids.to(self.device).long().view(-1), batch_idx, fg_mask.shape[0], -1)
        if grouped.shape[1] == 0:
            return preds["scores"].new_zeros(())
        image = torch.arange(fg_mask.shape[0], device=self.device)[:, None]
        labels = grouped[image, target_gt_idx.clamp(0, grouped.shape[1] - 1)][fg_mask]
        valid = (labels >= 0) & (labels < self.category_prototypes.shape[0])
        if not valid.any():
            return preds["scores"].new_zeros(())
        visual = torch.cat(
            [feature.flatten(2).transpose(1, 2) for feature in preds["cgpr_category_base_feats"]], dim=1
        )[fg_mask][valid]
        visual = self.head.cgpr_head.category_adapter(visual)
        visual = F.normalize(visual.float(), dim=-1)
        labels = labels[valid]
        # A shared batch candidate set preserves hard cross-category competition without a 20k dense prompt tensor.
        positive = torch.unique(labels)
        max_candidates = int(self.hyp.cgpr_category_candidates)
        all_ids = torch.arange(self.category_prototypes.shape[0], device=self.device)
        available = all_ids[~torch.isin(all_ids, positive)]
        remaining = max(max_candidates - positive.numel(), 0)
        if available.numel() > remaining:
            available = available[torch.randperm(available.numel(), device=self.device)[:remaining]]
        candidates = torch.cat((positive, available))
        target = torch.searchsorted(positive, labels)
        logits = visual @ self.category_prototypes[candidates].t() / 0.07
        return F.cross_entropy(logits, target)

    def _fine_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], imgsz: torch.Tensor) -> torch.Tensor:
        positive = batch.get("cgpr_phrase_id")
        negative = batch.get("cgpr_negative_ids")
        if positive is None or negative is None or positive.numel() == 0:
            return preds["scores"].new_zeros(())
        positive = positive.to(self.device).long().view(-1)
        negative = negative.to(self.device).long()
        valid_positive = (positive >= 0) & (positive < self.phrase_prototypes.shape[0])
        valid_negative = (negative >= 0) & (negative < self.phrase_prototypes.shape[0])
        valid_negative &= negative.ne(positive[:, None]) & valid_positive[:, None]
        if not valid_negative.any():
            return preds["scores"].new_zeros(())

        boxes = xywh2xyxy(batch["bboxes"].to(self.device).float())
        boxes[:, [0, 2]] *= imgsz[1]
        boxes[:, [1, 3]] *= imgsz[0]
        rois = torch.cat((batch["batch_idx"].to(self.device).float().view(-1, 1), boxes), dim=1)
        visual = self.head.cgpr_head.fine_projector(
            preds["feats"], rois, (int(imgsz[0].item()), int(imgsz[1].item()))
        )
        positive_proto = self.phrase_prototypes[positive.clamp(0, self.phrase_prototypes.shape[0] - 1)]
        negative_proto = self.phrase_prototypes[negative.clamp(0, self.phrase_prototypes.shape[0] - 1)]
        positive_logits = (visual * positive_proto).sum(-1) / 0.07
        negative_logits = torch.einsum("nd,nkd->nk", visual, negative_proto) / 0.07
        pairs = F.softplus(self.fine_margin + negative_logits - positive_logits[:, None])
        pairs = pairs.masked_fill(~valid_negative, float("-inf"))
        k = min(self.fine_topk, pairs.shape[1])
        top = pairs.topk(k, dim=1).values
        finite = torch.isfinite(top)
        return top[finite].mean() if finite.any() else visual.new_zeros(())

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute six independent and interpretable CGPR objectives."""
        values = self._assign(preds, batch)
        (
            pred_dist, pred_raw, pred_obj, pred_bboxes, anchor_points, stride_tensor, imgsz, gt_labels, _,
            target_bboxes, target_obj, fg_mask, target_gt_idx,
        ) = values
        raw_targets = self._raw_targets(pred_raw, gt_labels, target_obj, fg_mask, target_gt_idx, batch)
        normalizer = target_obj.sum().clamp_min(1)
        raw_bce = self.bce(pred_raw, raw_targets.to(pred_raw.dtype))
        if self.class_weights is not None:
            raw_bce *= self.class_weights
        cls_loss = raw_bce.sum() / normalizer
        box_loss = pred_raw.new_zeros(())
        dfl_loss = pred_raw.new_zeros(())
        if fg_mask.any():
            box_loss, dfl_loss = self.bbox_loss(
                pred_dist, pred_bboxes, anchor_points, target_bboxes / stride_tensor, target_obj, normalizer,
                fg_mask, imgsz, stride_tensor,
            )
        losses = torch.stack(
            (
                box_loss * self.hyp.box,
                cls_loss * self.hyp.cls,
                dfl_loss * self.hyp.dfl,
                self._objectness_loss(pred_obj, target_obj) * self.object_gain,
                self._category_loss(preds, batch, fg_mask, target_gt_idx) * self.category_gain,
                self._fine_loss(preds, batch, imgsz) * self.fine_gain,
            )
        )
        return losses * pred_raw.shape[0], losses.detach()


class CGPRWorldTrainer(WorldTrainerFromScratch):
    """Train CGPR branches and a small shared neck tail from a proven raw-phrase checkpoint."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        self.cgpr_cache: dict[str, Any] = {}
        self.phrase_to_id: dict[str, int] = {}
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        attach_cgpr_head(
            model.model[-1],
            object_hidden=int(self.args.cgpr_object_hidden),
            category_hidden=int(self.args.cgpr_category_hidden),
            roi_channels=int(self.args.cgpr_roi_channels),
            fine_hidden=int(self.args.cgpr_fine_hidden),
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self) -> None:
        super().set_model_attributes()
        path = Path(self.args.cgpr_cache).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CGPR cache not found: {path}")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("kind") != "bridge_cgpr_cache" or int(cache.get("version", 0)) != 1:
            raise ValueError(f"Unsupported CGPR cache: {path}")
        if not cache.get("base_annotations_unchanged") or not cache.get("missing_slots_are_unknown"):
            raise ValueError("CGPR cache violates baseline/partial-label invariants.")
        model = unwrap_model(self.model)
        embed_dim = model.model[-1].cgpr_head.fine_projector.embed_dim
        if cache["phrase_embeddings"].shape[1] != embed_dim:
            raise ValueError(f"Cache/model embedding mismatch: {cache['phrase_embeddings'].shape[1]} != {embed_dim}")
        self.cgpr_cache = cache
        self.phrase_to_id = {normalize_phrase(k): int(v) for k, v in cache["phrase_to_id"].items()}
        model.criterion = CGPRLoss(model, cache)
        LOGGER.info(
            f"CGPR cache: {len(cache['phrases']):,} phrases, {len(cache['categories']):,} categories, "
            f"K={cache['negative_pools'].shape[1]}; missing structure is masked, not negative"
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Resolve each transformed primary phrase to immutable prototype/category/negative ids."""
        classes = batch["cls"].view(-1).long().tolist()
        images = batch["batch_idx"].view(-1).long().tolist()
        texts = batch["texts"]
        phrase_ids = torch.full((len(classes),), -1, dtype=torch.long)
        category_ids = torch.full((len(classes),), -1, dtype=torch.long)
        width = int(self.args.cgpr_negatives_per_gt)
        negative_ids = torch.full((len(classes), width), -1, dtype=torch.long)
        pool = self.cgpr_cache["negative_pools"].long()
        phrase_categories = self.cgpr_cache["phrase_category_ids"].long()
        for row, (image_id, cls_id) in enumerate(zip(images, classes)):
            if image_id >= len(texts) or cls_id >= len(texts[image_id]):
                raise IndexError(f"Invalid transformed text index image={image_id}, cls={cls_id}.")
            phrase = normalize_phrase(texts[image_id][cls_id])
            pid = self.phrase_to_id.get(phrase, -1)
            if pid < 0:
                continue
            phrase_ids[row] = pid
            category_ids[row] = phrase_categories[pid]
            choices = pool[pid]
            choices = choices[choices >= 0]
            if choices.numel() > width:
                choices = choices[torch.randperm(choices.numel())[:width]]
            negative_ids[row, : choices.numel()] = choices
        batch["cgpr_phrase_id"] = phrase_ids
        batch["cgpr_category_id"] = category_ids
        batch["cgpr_negative_ids"] = negative_ids
        return super().preprocess_batch(batch)

    def configure_trainable_parameters(self) -> None:
        """Train CGPR plus only the last shared feature-producing layers at a lower LR."""
        model = unwrap_model(self.model)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.model[-1].cgpr_head.parameters():
            parameter.requires_grad = True
        count = int(self.args.cgpr_unfreeze_layers)
        if count > 0:
            for layer in list(model.model[:-1])[-count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        branch = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "cgpr_head" in n)
        shared = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "cgpr_head" not in n)
        if branch == 0:
            raise RuntimeError("No CGPR parameters are trainable.")
        LOGGER.info(f"CGPR optimization: branch={branch:,}, shared_tail={shared:,}, unfreeze_layers={count}")

    def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-4, iterations=1e5):
        if str(name).lower() != "adamw":
            raise ValueError("CGPR supports optimizer=AdamW only.")
        groups = []
        for scope, multiplier in (("branch", 1.0), ("shared", float(self.args.cgpr_shared_lr_mult))):
            named = [
                (name, parameter)
                for name, parameter in unwrap_model(model).named_parameters()
                if parameter.requires_grad and (("cgpr_head" in name) == (scope == "branch"))
            ]
            decay_params = [p for _, p in named if p.ndim > 1]
            no_decay_params = [p for _, p in named if p.ndim <= 1]
            if decay_params:
                groups.append({"params": decay_params, "lr": lr * multiplier, "weight_decay": decay, "scope": scope})
            if no_decay_params:
                groups.append({"params": no_decay_params, "lr": lr * multiplier, "weight_decay": 0.0, "scope": scope})
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=(momentum, 0.999))
        LOGGER.info(f"optimizer: AdamW branch_lr={lr:g}, shared_lr={lr * float(self.args.cgpr_shared_lr_mult):g}")
        return optimizer

    def _model_train(self) -> None:
        self.model.train()
        model = unwrap_model(self.model)
        criterion = model.criterion
        warmup = max(int(self.args.cgpr_assignment_warmup), 1)
        criterion.assignment_objectness_weight = min(max((self.epoch + 1) / warmup, 0.0), 1.0)
        for name, module in self.model.named_modules():
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(parameter.requires_grad for parameter in parameters) and isinstance(
                module, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)
            ):
                module.eval()

    def _setup_train(self) -> None:
        """Avoid an online AMP probe and remove the large criterion cache from EMA checkpoints."""
        import ultralytics.engine.trainer as trainer_module

        original_check_amp = trainer_module.check_amp
        trainer_module.check_amp = lambda _: True
        try:
            super()._setup_train()
        finally:
            trainer_module.check_amp = original_check_amp
        if self.ema is not None and hasattr(unwrap_model(self.ema.ema), "criterion"):
            delattr(unwrap_model(self.ema.ema), "criterion")

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "obj_loss", "cat_loss", "fine_loss"
        return validator
