# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Asymmetric category-attribute-affordance training for YOLO-World.

The detector architecture is unchanged.  The criterion routes prompt logits by
their semantic role, uses full task-aligned quality targets for every known
positive, masks unknown prompt/GT pairs only on assigned foreground anchors,
and aligns the per-GT spatial response of category and affordance prompts to a
stop-gradient attribute teacher.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model


ATTRIBUTE_CODE_MIN = 2.5
CATEGORY_CODE_MIN = 1.5
OWN_AFFORDANCE_CODE_MIN = 0.75


class TriGranularLoss(v8DetectionLoss):
    """Role-routed selective BCE plus cross-granularity spatial consistency."""

    def __init__(self, model) -> None:
        super().__init__(model)
        self.category_gain = float(model.args.tri_category_gain)
        self.affordance_gain = float(model.args.tri_affordance_gain)
        self.consistency_base_gain = float(model.args.tri_consistency_gain)
        self.consistency_gain = 0.0
        self.temperature = float(model.args.tri_consistency_temperature)
        self.use_unknown_mask = bool(model.args.tri_use_unknown_mask)
        self.use_compatible_affordance = bool(model.args.tri_use_compatible_affordance)
        if min(self.category_gain, self.affordance_gain, self.consistency_base_gain) < 0:
            raise ValueError("Tri-granular loss gains must be nonnegative")
        if self.temperature <= 0:
            raise ValueError("tri_consistency_temperature must be positive")

    def _state_role(self, states: torch.Tensor, role: str) -> torch.Tensor:
        """Return the positive-state mask for one semantic role."""
        if role == "attribute":
            return states >= ATTRIBUTE_CODE_MIN
        if role == "category":
            return (states >= CATEGORY_CODE_MIN) & (states < ATTRIBUTE_CODE_MIN)
        if role == "affordance":
            minimum = 0.0 if self.use_compatible_affordance else OWN_AFFORDANCE_CODE_MIN
            return (states > minimum) & (states < CATEGORY_CODE_MIN)
        raise KeyError(role)

    def _prepare_supervision(
        self,
        batch: dict[str, torch.Tensor],
        batch_size: int,
        max_boxes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded per-GT supervision and one role id per sampled prompt."""
        multi = batch.get("multi_positive")
        if multi is None or multi.ndim != 3 or multi.shape[-1] != 2:
            raise RuntimeError(
                "Tri-granular training requires multi_positive[N,K,2]. Rebuild data with "
                "tools/build_bridge_tri_granular.py and remove stale *.cache files."
            )
        if multi.shape[0] != batch["batch_idx"].shape[0]:
            raise RuntimeError("multi_positive and GT instance counts differ")
        multi = multi.to(self.device, non_blocking=True).float()
        batch_idx = batch["batch_idx"].to(self.device, non_blocking=True).long()
        grouped = self.preprocess_multi_positive(multi, batch_idx, batch_size, max_boxes)

        # A prompt can be shared by several boxes.  Assign one global role per
        # image/prompt with attribute > category > affordance precedence.  This
        # partitions BCE elements instead of counting the same negative three
        # times in separate task losses.
        prompt_roles = torch.zeros(batch_size, self.nc, dtype=torch.long, device=self.device)
        labels = multi[..., 0].long()
        states = multi[..., 1]
        images = batch_idx[:, None].expand_as(labels)
        for role_id, valid in (
            (1, (states > 0) & (states < CATEGORY_CODE_MIN)),
            (2, (states >= CATEGORY_CODE_MIN) & (states < ATTRIBUTE_CODE_MIN)),
            (3, states >= ATTRIBUTE_CODE_MIN),
        ):
            valid &= (labels >= 0) & (labels < self.nc)
            if valid.any():
                prompt_roles[images[valid], labels[valid]] = role_id
        if not (prompt_roles == 3).any():
            raise RuntimeError("Tri-granular batch has no attribute role codes; stale/wrong dataset cache is likely")
        return grouped, prompt_roles

    def _role_targets(
        self,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        grouped: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build full-quality targets and an unknown-only foreground mask."""
        batch_size, anchors, _ = target_scores.shape
        max_boxes = grouped.shape[1]
        if max_boxes == 0:
            targets = {
                "attribute": target_scores.clone(),
                "category": torch.zeros_like(target_scores),
                "affordance": torch.zeros_like(target_scores),
            }
            return targets, torch.ones_like(target_scores, dtype=torch.bool)
        image = torch.arange(batch_size, device=self.device)[:, None]
        assigned = grouped[image, target_gt_idx.clamp(0, max_boxes - 1)]
        labels, states = assigned[..., 0].long(), assigned[..., 1]
        quality = target_scores.sum(-1, keepdim=True)
        targets = {
            "attribute": target_scores.clone(),
            "category": torch.zeros_like(target_scores),
            "affordance": torch.zeros_like(target_scores),
        }
        for role in ("attribute", "category", "affordance"):
            valid = (
                self._state_role(states, role)
                & (labels >= 0)
                & (labels < self.nc)
                & fg_mask[..., None]
            )
            if not valid.any():
                continue
            extra = torch.zeros_like(target_scores)
            extra.scatter_add_(
                2,
                labels.clamp(0, self.nc - 1),
                quality * valid.to(quality.dtype),
            )
            targets[role] = torch.maximum(targets[role], extra.clamp_max(1.0))

        unknown = (
            (states == -1)
            & (labels >= 0)
            & (labels < self.nc)
            & fg_mask[..., None]
        )
        known_mask = torch.ones(batch_size, anchors, self.nc, dtype=torch.bool, device=self.device)
        if self.use_unknown_mask and unknown.any():
            counts = torch.zeros_like(target_scores)
            counts.scatter_add_(2, labels.clamp(0, self.nc - 1), unknown.to(target_scores.dtype))
            known_mask = counts.eq(0)
            # Positive wins defensively if malformed input marks both states.
            union_positive = targets["attribute"] + targets["category"] + targets["affordance"]
            known_mask |= union_positive > 0
        return targets, known_mask

    def _selective_bce(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        prompt_mask: torch.Tensor,
        known_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one role loss with independent positive-quality normalization."""
        if not prompt_mask.any() or not (targets > 0).any():
            return pred_scores.sum() * 0.0
        element_mask = known_mask & prompt_mask[:, None, :]
        bce = self.bce(pred_scores, targets.to(pred_scores.dtype))
        if self.class_weights is not None:
            bce *= self.class_weights
        bce *= element_mask.to(bce.dtype)
        return bce.sum() / targets.sum().clamp_min(1.0)

    def _spatial_consistency(
        self,
        pred_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        grouped: torch.Tensor,
    ) -> torch.Tensor:
        """Align category/affordance anchor distributions to the raw attribute teacher."""
        zero = pred_scores.sum() * 0.0
        if self.consistency_gain <= 0 or not fg_mask.any() or grouped.shape[1] == 0:
            return zero
        temperature = self.temperature
        terms = []
        for batch_id in range(pred_scores.shape[0]):
            gt_ids = target_gt_idx[batch_id][fg_mask[batch_id]].unique()
            for gt_id_tensor in gt_ids:
                gt_id = int(gt_id_tensor.item())
                if gt_id < 0 or gt_id >= grouped.shape[1]:
                    continue
                anchor_mask = fg_mask[batch_id] & target_gt_idx[batch_id].eq(gt_id)
                if int(anchor_mask.sum()) < 2:
                    continue
                supervision = grouped[batch_id, gt_id]
                labels, states = supervision[:, 0].long(), supervision[:, 1]

                def first_label(mask: torch.Tensor) -> int:
                    values = labels[mask & (labels >= 0) & (labels < self.nc)]
                    return int(values[0].item()) if values.numel() else -1

                attribute_id = first_label(states >= ATTRIBUTE_CODE_MIN)
                category_id = first_label((states >= CATEGORY_CODE_MIN) & (states < ATTRIBUTE_CODE_MIN))
                affordance_id = first_label((states >= OWN_AFFORDANCE_CODE_MIN) & (states < CATEGORY_CODE_MIN))
                if attribute_id < 0:
                    continue
                teacher = F.softmax(pred_scores[batch_id, anchor_mask, attribute_id].detach() / temperature, dim=0)
                for student_id in (category_id, affordance_id):
                    if student_id < 0 or student_id == attribute_id:
                        continue
                    student_log = F.log_softmax(pred_scores[batch_id, anchor_mask, student_id] / temperature, dim=0)
                    terms.append(F.kl_div(student_log, teacher, reduction="sum") * (temperature**2))
        return torch.stack(terms).mean() if terms else zero

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return box, three role losses, consistency, and DFL."""
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size, dtype = pred_scores.shape[0], pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        raw_targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        raw_targets = self.preprocess(raw_targets.to(self.device), batch_size, imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = raw_targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_min(1.0)
        grouped, prompt_roles = self._prepare_supervision(batch, batch_size, gt_bboxes.shape[1])
        role_targets, known_mask = self._role_targets(target_scores, fg_mask, target_gt_idx, grouped)
        attr_loss = self._selective_bce(pred_scores, role_targets["attribute"], prompt_roles.eq(3), known_mask)
        cat_loss = self._selective_bce(pred_scores, role_targets["category"], prompt_roles.eq(2), known_mask)
        aff_loss = self._selective_bce(pred_scores, role_targets["affordance"], prompt_roles.eq(1), known_mask)
        consistency = self._spatial_consistency(pred_scores, fg_mask, target_gt_idx, grouped)

        box_loss = pred_scores.sum() * 0.0
        dfl_loss = pred_scores.sum() * 0.0
        if fg_mask.any():
            box_loss, dfl_loss = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        losses = torch.stack(
            (
                box_loss * self.hyp.box,
                attr_loss * self.hyp.cls,
                cat_loss * self.hyp.cls * self.category_gain,
                aff_loss * self.hyp.cls * self.affordance_gain,
                consistency * self.hyp.cls * self.consistency_gain,
                dfl_loss * self.hyp.dfl,
            )
        )
        return losses.sum() * batch_size, losses.detach()


class TriGranularWorldTrainer(WorldTrainerFromScratch):
    """World trainer that installs the tri-granular criterion without changing the model head."""

    def set_model_attributes(self) -> None:
        super().set_model_attributes()
        model = unwrap_model(self.model)
        model.criterion = TriGranularLoss(model)
        LOGGER.info(
            "Tri-granular loss: attribute=1, category=%g, affordance=%g, consistency=%g, temperature=%g, "
            "unknown_mask=%s, compatible_affordance=%s",
            float(self.args.tri_category_gain),
            float(self.args.tri_affordance_gain),
            float(self.args.tri_consistency_gain),
            float(self.args.tri_consistency_temperature),
            bool(self.args.tri_use_unknown_mask),
            bool(self.args.tri_use_compatible_affordance),
        )

    def _model_train(self) -> None:
        super()._model_train()
        criterion = unwrap_model(self.model).criterion
        warmup = int(self.args.tri_consistency_warmup_epochs)
        scale = 1.0 if warmup <= 0 else min(max((self.epoch + 1) / warmup, 0.0), 1.0)
        criterion.consistency_gain = criterion.consistency_base_gain * scale

    def _setup_train(self) -> None:
        super()._setup_train()
        if self.ema is not None and hasattr(unwrap_model(self.ema.ema), "criterion"):
            delattr(unwrap_model(self.ema.ema), "criterion")

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = (
            "box_loss",
            "attr_loss",
            "cat_loss",
            "aff_loss",
            "cgc_loss",
            "dfl_loss",
        )
        return validator
