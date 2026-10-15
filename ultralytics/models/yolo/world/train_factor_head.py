# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Frozen-baseline structured factor-head training for YOLO-World."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.nn.modules.structured_adapter import attach_factor_adapter
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model


def normalize_phrase(value: Any) -> str:
    """Normalize phrase keys while preserving their word order."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower()).strip(" .,:;|\"'")


class FactorHeadLoss(v8DetectionLoss):
    """Use frozen raw assignments to train independent masked slot classifiers."""

    def __init__(
        self,
        model: nn.Module,
        prototypes: torch.Tensor,
        component_slot_ids: torch.Tensor,
        gain: float,
        negative_gain: float,
        max_candidates_per_slot: int,
    ) -> None:
        super().__init__(model)
        self.head = model.model[-1]
        self.prototypes = F.normalize(prototypes.to(self.device).float(), dim=-1)
        self.component_slot_ids = component_slot_ids.to(self.device).long()
        self.gain = float(gain)
        self.negative_gain = float(negative_gain)
        self.max_candidates_per_slot = int(max_candidates_per_slot)
        if self.gain <= 0 or self.negative_gain < 0 or self.max_candidates_per_slot < 1:
            raise ValueError("Invalid factor-head loss configuration.")
        if self.component_slot_ids.shape != (self.prototypes.shape[0],):
            raise ValueError("component_slot_ids must have one entry per prototype.")

    def _group_components(
        self, flat_ids: torch.Tensor, batch_idx: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Group flat instance component ids into TaskAlignedAssigner's padded GT layout."""
        count = flat_ids.shape[1]
        counts = torch.bincount(batch_idx, minlength=batch_size)
        max_boxes = int(counts.max().item()) if counts.numel() else 0
        grouped = torch.full((batch_size, max_boxes, count), -1, dtype=torch.long, device=self.device)
        if not flat_ids.numel() or max_boxes == 0:
            return grouped
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets[1:] = counts.cumsum(0)
        within = torch.arange(flat_ids.shape[0], device=self.device) - offsets[batch_idx]
        grouped[batch_idx, within] = flat_ids
        return grouped

    def _flatten_factor_features(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the frozen contrastive normalization/scale and flatten in detector anchor order."""
        visual, scales, biases = [], [], []
        for feature, contrastive_head in zip(features, self.head.cv4):
            norm = getattr(contrastive_head, "norm", None)
            feature = norm(feature) if norm is not None else F.normalize(feature, dim=1)
            feature = feature.float().flatten(2).transpose(1, 2).contiguous()
            visual.append(feature)
            scale = getattr(contrastive_head, "logit_scale", None)
            bias = getattr(contrastive_head, "bias", None)
            scale_value = float(scale.detach().float().exp()) if scale is not None else 1.0
            bias_value = float(bias.detach().float()) if bias is not None else 0.0
            scales.append(feature.new_full((feature.shape[1],), scale_value))
            biases.append(feature.new_full((feature.shape[1],), bias_value))
        return torch.cat(visual, 1), torch.cat(scales), torch.cat(biases)

    def factor_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute balanced multi-label sigmoid losses only for slots explicitly present in each GT phrase."""
        flat_ids = batch.get("factor_component_ids")
        factor_feats = preds.get("factor_feats")
        if flat_ids is None or factor_feats is None or not fg_mask.any():
            return preds["scores"].new_zeros(())

        flat_ids = flat_ids.to(self.device, non_blocking=True).long()
        batch_idx = batch["batch_idx"].to(self.device, non_blocking=True).long()
        batch_size = int(fg_mask.shape[0])
        grouped = self._group_components(flat_ids, batch_idx, batch_size)
        if grouped.shape[1] == 0:
            return preds["scores"].new_zeros(())
        image_index = torch.arange(batch_size, device=self.device)[:, None]
        assigned_ids = grouped[image_index, target_gt_idx.clamp(0, grouped.shape[1] - 1)][fg_mask]
        valid_ids = (assigned_ids >= 0) & (assigned_ids < self.prototypes.shape[0])
        if not valid_ids.any():
            return preds["scores"].new_zeros(())

        assigned_slots = torch.full_like(assigned_ids, -1)
        assigned_slots[valid_ids] = self.component_slot_ids[assigned_ids[valid_ids]]
        visual, anchor_scales, anchor_biases = self._flatten_factor_features(factor_feats)
        visual = visual[fg_mask]
        scales = anchor_scales[None].expand(batch_size, -1)[fg_mask]
        biases = anchor_biases[None].expand(batch_size, -1)[fg_mask]

        slot_losses = []
        for slot_id in torch.unique(assigned_slots[assigned_slots >= 0]).tolist():
            positive = valid_ids & assigned_slots.eq(slot_id)
            active_rows = positive.any(1)
            if not active_rows.any():
                continue
            positive_ids = torch.unique(assigned_ids[positive], sorted=True)
            slot_pool = (self.component_slot_ids == slot_id).nonzero(as_tuple=False).squeeze(1)
            available = slot_pool[~torch.isin(slot_pool, positive_ids)]
            remaining = max(self.max_candidates_per_slot - positive_ids.numel(), 0)
            if remaining and available.numel() > remaining:
                available = available[torch.randperm(available.numel(), device=self.device)[:remaining]]
            elif remaining == 0:
                available = available[:0]
            candidate_ids = torch.cat((positive_ids, available))

            row_visual = visual[active_rows]
            logits = row_visual @ self.prototypes[candidate_ids].t()
            logits = logits * scales[active_rows, None] + biases[active_rows, None]
            row_ids = assigned_ids[active_rows]
            row_positive = row_ids[:, :, None].eq(candidate_ids[None, None, :]).any(1)
            positive_loss = F.softplus(-logits)[row_positive].mean()
            negative_mask = ~row_positive
            negative_loss = (
                F.softplus(logits)[negative_mask].mean() if negative_mask.any() else logits.new_zeros(())
            )
            slot_losses.append(positive_loss + self.negative_gain * negative_loss)

        if not slot_losses:
            return preds["scores"].new_zeros(())
        return torch.stack(slot_losses).mean().to(preds["scores"].dtype)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return frozen raw diagnostics and the only trainable factor loss."""
        assigned, detection_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        fg_mask, target_gt_idx = assigned[:2]
        factor = self.factor_loss(preds, batch, fg_mask, target_gt_idx) * self.gain
        losses = torch.cat((detection_loss.detach(), factor.reshape(1)))
        return losses * preds["boxes"].shape[0], losses.detach()


class FactorHeadWorldTrainer(WorldTrainerFromScratch):
    """Freeze the proven detector and train one shared structured visual factor head."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        self.phrase_to_component_ids: dict[str, list[int]] = {}
        self.max_components = 0
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Attach the factor module before loading either a baseline or factor checkpoint."""
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        attach_factor_adapter(model.model[-1], hidden_dim=int(self.args.factor_head_hidden))
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self) -> None:
        """Validate the immutable factor cache and install the factor criterion."""
        super().set_model_attributes()
        path = Path(self.args.factor_head_cache).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Factor-head cache not found: {path}")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(cache, dict)
            or cache.get("kind") != "bridge_factor_training_cache"
            or int(cache.get("version", 0)) != 1
        ):
            raise ValueError(f"Unsupported factor-head cache: {path}")
        if cache.get("base_annotations_unchanged") is not True:
            raise ValueError("Factor cache must certify that the original MP annotations are unchanged.")
        prototypes = cache.get("prototypes")
        slot_ids = cache.get("component_slot_ids")
        mapping = cache.get("phrase_to_component_ids")
        if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
            raise ValueError("Factor cache contains invalid prototypes.")
        if not isinstance(slot_ids, torch.Tensor) or not isinstance(mapping, dict):
            raise ValueError("Factor cache contains invalid slot ids or phrase mapping.")

        model = unwrap_model(self.model)
        expected_dim = int(model.model[-1].cv3[0][-1].out_channels)
        if prototypes.shape[1] != expected_dim:
            raise ValueError(f"Factor prototype width {prototypes.shape[1]} != detector width {expected_dim}.")
        self.phrase_to_component_ids = {
            normalize_phrase(key): [int(value) for value in values]
            for key, values in mapping.items()
            if normalize_phrase(key) and isinstance(values, list) and values
        }
        self.max_components = max(map(len, self.phrase_to_component_ids.values()), default=0)
        if self.max_components < 1:
            raise ValueError("Factor cache contains no usable phrase assignments.")
        model.criterion = FactorHeadLoss(
            model=model,
            prototypes=prototypes,
            component_slot_ids=slot_ids,
            gain=float(self.args.factor_head_gain),
            negative_gain=float(self.args.factor_head_negative_gain),
            max_candidates_per_slot=int(self.args.factor_head_max_candidates),
        )
        LOGGER.info(
            f"Factor cache: {len(self.phrase_to_component_ids):,} phrases, {prototypes.shape[0]:,} components, "
            f"slots={cache.get('slot_names', [])}; raw MP detector remains frozen"
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Resolve transformed primary phrases to masked structured component targets."""
        classes = batch["cls"].view(-1).long().tolist()
        image_ids = batch["batch_idx"].view(-1).long().tolist()
        texts = batch["texts"]
        ids = torch.full((len(classes), self.max_components), -1, dtype=torch.long)
        for index, (image_id, class_id) in enumerate(zip(image_ids, classes)):
            if image_id >= len(texts) or class_id >= len(texts[image_id]):
                raise IndexError(f"Invalid transformed text index image={image_id}, cls={class_id}.")
            values = self.phrase_to_component_ids.get(normalize_phrase(texts[image_id][class_id]), [])
            if values:
                ids[index, : len(values)] = torch.tensor(values, dtype=torch.long)
        batch["factor_component_ids"] = ids
        return super().preprocess_batch(batch)

    def configure_trainable_parameters(self) -> None:
        """Make the independent factor adapter the only trainable module."""
        model = unwrap_model(self.model)
        trainable = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "factor_adapter" in name
            trainable += parameter.numel() if parameter.requires_grad else 0
        if trainable < 1:
            raise RuntimeError("No factor-adapter parameters were found.")
        LOGGER.info(f"Factor-head-only optimization: {trainable:,} trainable parameters")

    def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-4, iterations=1e5):
        """Build a small optimizer containing only factor tensors."""
        if str(name).lower() != "adamw":
            raise ValueError("Factor-head training intentionally supports only optimizer=AdamW.")
        named = [
            (name, parameter) for name, parameter in unwrap_model(model).named_parameters() if parameter.requires_grad
        ]
        decay_params = [parameter for _, parameter in named if parameter.ndim > 1]
        no_decay_params = [parameter for _, parameter in named if parameter.ndim <= 1]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": decay, "param_group": "weight"},
                {"params": no_decay_params, "weight_decay": 0.0, "param_group": "bias"},
            ],
            lr=lr,
            betas=(momentum, 0.999),
        )
        LOGGER.info(f"optimizer: AdamW(lr={lr}) with {len(decay_params) + len(no_decay_params)} factor tensors")
        return optimizer

    def _model_train(self) -> None:
        """Train the factor adapter while every frozen normalization/dropout layer stays deterministic."""
        self.model.train()
        for name, module in self.model.named_modules():
            if "factor_adapter" not in name and isinstance(
                module, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)
            ):
                module.eval()

    def _setup_train(self) -> None:
        """Avoid the unrelated online AMP probe and keep serialized EMA checkpoints compact."""
        import ultralytics.engine.trainer as trainer_module

        original_check_amp = trainer_module.check_amp
        trainer_module.check_amp = lambda _: True
        try:
            super()._setup_train()
        finally:
            trainer_module.check_amp = original_check_amp
        if self.ema is not None and hasattr(unwrap_model(self.ema.ema), "criterion"):
            delattr(unwrap_model(self.ema.ema), "criterion")
        self._assert_raw_identity()

    def _assert_raw_identity(self) -> None:
        """Prove that merely attaching the independent factor head cannot change raw inference."""
        model = unwrap_model(self.model)
        head = model.model[-1]
        batch = self.preprocess_batch(next(iter(self.train_loader)))
        self.model.eval()
        with torch.inference_mode():
            head.factor_score_alpha = 0.0
            head.factor_adapter_enabled = True
            enabled = self.model(batch["img"], txt_feats=batch["txt_feats"])
            head.factor_adapter_enabled = False
            disabled = self.model(batch["img"], txt_feats=batch["txt_feats"])
            head.factor_adapter_enabled = True
        enabled = enabled[1] if isinstance(enabled, tuple) else enabled
        disabled = disabled[1] if isinstance(disabled, tuple) else disabled
        box_error = float((enabled["boxes"] - disabled["boxes"]).abs().max())
        score_error = float((enabled["scores"] - disabled["scores"]).abs().max())
        if box_error > 1e-7 or score_error > 1e-7:
            raise RuntimeError(f"Factor head changed raw inference: boxes={box_error:.3e}, scores={score_error:.3e}")
        LOGGER.info(f"Raw-identity check passed: boxes={box_error:.1e}, scores={score_error:.1e}")

    def get_validator(self):
        """Validate the unchanged raw detector during training; fused validation is an explicit separate step."""
        validator = super().get_validator()
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "factor_loss"
        return validator
