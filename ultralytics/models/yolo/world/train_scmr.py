# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Baseline-preserving SCMR adapter training for YOLO-World."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.nn.modules.structured_adapter import attach_scmr_adapters
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model


def normalize_phrase(value: Any) -> str:
    """Normalize phrase keys without changing their semantic word order."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower()).strip(" .,:;|\"'")


class SCMRAdapterLoss(v8DetectionLoss):
    """Frozen baseline assignment plus isolated minimal-pair and preservation losses.

    The original box/class/DFL losses are computed from detached baseline scores
    only for diagnostics and stable foreground assignment. They cannot update the
    adapter. Hard negatives are external cached text prototypes, so they never
    alter RandomLoadText's 80-prompt budget or become ordinary BCE negatives.
    """

    def __init__(
        self,
        model: nn.Module,
        prototypes: torch.Tensor,
        gain: float,
        margin: float,
        preserve_gain: float,
    ) -> None:
        super().__init__(model)
        self.head = model.model[-1]
        self.prototypes = F.normalize(prototypes.to(self.device).float(), dim=-1)
        self.gain = float(gain)
        self.margin = float(margin)
        self.preserve_gain = float(preserve_gain)
        if self.gain <= 0 or self.margin < 0 or self.preserve_gain <= 0:
            raise ValueError("SCMR gain/preserve_gain must be > 0 and margin must be >= 0.")

    def _group_instances(
        self, values: torch.Tensor, batch_idx: torch.Tensor, batch_size: int, fill_value: int
    ) -> torch.Tensor:
        """Group flat instance tensors into TaskAlignedAssigner's padded layout."""
        counts = torch.bincount(batch_idx, minlength=batch_size)
        max_boxes = int(counts.max().item()) if counts.numel() else 0
        output = torch.full(
            (batch_size, max_boxes, *values.shape[1:]),
            fill_value,
            dtype=values.dtype,
            device=self.device,
        )
        if not values.numel() or max_boxes == 0:
            return output
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets[1:] = counts.cumsum(0)
        within = torch.arange(values.shape[0], device=self.device) - offsets[batch_idx]
        output[batch_idx, within] = values
        return output

    def _flatten_visual_and_scale(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten adapted maps in anchor order using the exact WorldDetect normalization and scale."""
        visual, scales = [], []
        for feature, contrastive_head in zip(features, self.head.cv4):
            norm = getattr(contrastive_head, "norm", None)
            if norm is not None:
                feature = norm(feature)
            else:
                feature = F.normalize(feature, dim=1)
            flattened = feature.float().flatten(2).transpose(1, 2).contiguous()
            visual.append(flattened)
            scale = getattr(contrastive_head, "logit_scale", None)
            scale = scale.detach().float().exp() if scale is not None else flattened.new_ones(())
            scales.append(flattened.new_full((flattened.shape[1],), float(scale)))
        return torch.cat(visual, dim=1), torch.cat(scales)

    def ranking_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Rank each raw attribute prototype above its strict one-slot counterfactuals."""
        positive_ids = batch.get("scmr_positive_id")
        negative_ids = batch.get("scmr_negative_ids")
        aligned = preds.get("aligned_feats")
        if positive_ids is None or negative_ids is None or aligned is None or not fg_mask.any():
            return preds["scores"].new_zeros(())

        batch_size = int(fg_mask.shape[0])
        batch_idx = batch["batch_idx"].to(self.device, non_blocking=True).long()
        positive_ids = positive_ids.to(self.device, non_blocking=True).long().view(-1)
        negative_ids = negative_ids.to(self.device, non_blocking=True).long()
        if positive_ids.shape[0] != batch_idx.shape[0] or negative_ids.shape[0] != batch_idx.shape[0]:
            raise ValueError("SCMR prototype ids must have one row per GT instance.")

        grouped_positive = self._group_instances(positive_ids, batch_idx, batch_size, -1)
        grouped_negative = self._group_instances(negative_ids, batch_idx, batch_size, -1)
        max_boxes = grouped_positive.shape[1]
        if max_boxes == 0:
            return preds["scores"].new_zeros(())

        image_index = torch.arange(batch_size, device=self.device)[:, None]
        assigned_index = target_gt_idx.clamp(0, max_boxes - 1)
        assigned_positive = grouped_positive[image_index, assigned_index][fg_mask]
        assigned_negative = grouped_negative[image_index, assigned_index][fg_mask]
        valid_positive = (assigned_positive >= 0) & (assigned_positive < self.prototypes.shape[0])
        valid_negative = (assigned_negative >= 0) & (assigned_negative < self.prototypes.shape[0])
        valid_negative &= assigned_negative.ne(assigned_positive[:, None]) & valid_positive[:, None]
        if not valid_negative.any():
            return preds["scores"].new_zeros(())

        visual, anchor_scales = self._flatten_visual_and_scale(aligned)
        visual = visual[fg_mask]
        scale = anchor_scales.to(self.device)[None, :].expand(batch_size, -1)[fg_mask]
        positive_proto = self.prototypes[assigned_positive.clamp(0, self.prototypes.shape[0] - 1)]
        negative_proto = self.prototypes[assigned_negative.clamp(0, self.prototypes.shape[0] - 1)]
        positive_logits = (visual * positive_proto).sum(-1) * scale
        negative_logits = torch.einsum("nc,nkc->nk", visual, negative_proto) * scale[:, None]
        pair_loss = F.softplus(self.margin + negative_logits - positive_logits[:, None])
        return pair_loss[valid_negative].mean().to(preds["scores"].dtype)

    @staticmethod
    def preservation_loss(preds: dict[str, torch.Tensor]) -> torch.Tensor:
        """Keep raw-prompt logits close to the frozen baseline on every anchor."""
        base_scores = preds.get("base_scores")
        if base_scores is None:
            return preds["scores"].new_zeros(())
        return F.smooth_l1_loss(preds["scores"].float(), base_scores.float(), beta=0.25).to(preds["scores"].dtype)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached baseline diagnostics, SCMR ranking, and raw-logit preservation."""
        base_scores = preds.get("base_scores")
        if base_scores is None:
            raise RuntimeError("SCMR training requires WorldDetect to return detached base_scores.")
        base_preds = dict(preds)
        base_preds["scores"] = base_scores
        assigned, detection_loss, _ = self.get_assigned_targets_and_loss(base_preds, batch)
        fg_mask, target_gt_idx = assigned[:2]
        ranking = self.ranking_loss(preds, batch, fg_mask, target_gt_idx) * self.gain
        preservation = self.preservation_loss(preds) * self.preserve_gain
        losses = torch.cat((detection_loss.detach(), ranking.reshape(1), preservation.reshape(1)))
        return losses * preds["boxes"].shape[0], losses.detach()


class SCMRWorldTrainer(WorldTrainerFromScratch):
    """Train only zero-initialized dense classification adapters with offline SCMR pairs."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        self.hard_negative_map: dict[str, list[str]] = {}
        self.phrase_to_prototype_id: dict[str, int] = {}
        self.max_hard_negatives = 0
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Attach adapters before loading weights so both baseline and SCMR checkpoints are supported."""
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        attach_scmr_adapters(model.model[-1], hidden_dim=int(self.args.scmr_adapter_hidden))
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self) -> None:
        """Load and validate the immutable phrase-to-counterfactual sidecar."""
        super().set_model_attributes()
        path = Path(self.args.scmr_hard_negative_map).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SCMR hard-negative sidecar not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = payload.get("phrase_to_hard_negatives") if isinstance(payload, dict) else None
        if int(payload.get("version", 0)) != 1 or not isinstance(mapping, dict):
            raise ValueError(f"Unsupported SCMR sidecar: {path}")
        self.hard_negative_map = {
            normalize_phrase(phrase): list(
                dict.fromkeys(normalize_phrase(value) for value in values if normalize_phrase(value))
            )[: int(self.args.scmr_max_negatives)]
            for phrase, values in mapping.items()
            if normalize_phrase(phrase) and isinstance(values, list)
        }
        self.hard_negative_map = {key: values for key, values in self.hard_negative_map.items() if values}
        self.max_hard_negatives = max(map(len, self.hard_negative_map.values()), default=0)
        if not self.hard_negative_map or self.max_hard_negatives < 1:
            raise ValueError("SCMR sidecar contains no usable hard-negative mappings.")

    def _install_scmr_criterion(self) -> None:
        """Build the prototype bank from the existing offline SigLIP2 text cache."""
        normalized_embeddings = {
            normalize_phrase(text): embedding for text, embedding in (self.text_embeddings or {}).items()
        }
        required = set(self.hard_negative_map)
        required.update(value for values in self.hard_negative_map.values() for value in values)
        missing = sorted(required - normalized_embeddings.keys())
        if missing:
            raise RuntimeError(
                f"{len(missing)} SCMR phrases are absent from the existing training text cache; examples={missing[:10]}"
            )
        phrases = sorted(required)
        self.phrase_to_prototype_id = {phrase: index for index, phrase in enumerate(phrases)}
        prototypes = torch.stack([normalized_embeddings[phrase] for phrase in phrases])
        model = unwrap_model(self.model)
        model.criterion = SCMRAdapterLoss(
            model=model,
            prototypes=prototypes,
            gain=float(self.args.scmr_gain),
            margin=float(self.args.scmr_margin),
            preserve_gain=float(self.args.scmr_preserve_gain),
        )
        LOGGER.info(
            f"SCMR prototype bank: {len(phrases):,} cached phrases, {len(self.hard_negative_map):,} positive keys, "
            f"K={self.max_hard_negatives}; RandomLoadText/BCE unchanged"
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build the original MP dataset and install SCMR without modifying its transforms."""
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        if mode == "train":
            self._install_scmr_criterion()
        return dataset

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Resolve transformed primary phrases to external positive/negative prototype ids."""
        classes = batch["cls"].view(-1).long().tolist()
        image_ids = batch["batch_idx"].view(-1).long().tolist()
        texts = batch["texts"]
        positive = torch.full((len(classes),), -1, dtype=torch.long)
        negative = torch.full((len(classes), self.max_hard_negatives), -1, dtype=torch.long)
        for index, (image_id, cls_id) in enumerate(zip(image_ids, classes)):
            if image_id >= len(texts) or cls_id >= len(texts[image_id]):
                raise IndexError(f"Invalid transformed text index image={image_id}, cls={cls_id}.")
            phrase = normalize_phrase(texts[image_id][cls_id])
            values = self.hard_negative_map.get(phrase)
            if not values:
                continue
            positive[index] = self.phrase_to_prototype_id[phrase]
            ids = [self.phrase_to_prototype_id[value] for value in values]
            negative[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        batch["scmr_positive_id"] = positive
        batch["scmr_negative_ids"] = negative
        return super().preprocess_batch(batch)

    def configure_trainable_parameters(self) -> None:
        """Freeze the complete detector; SCMR adapters are the only trainable parameters."""
        model = unwrap_model(self.model)
        trainable = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "scmr_adapters" in name
            trainable += parameter.numel() if parameter.requires_grad else 0
        if trainable < 1:
            raise RuntimeError("No SCMR adapter parameters were found.")
        LOGGER.info(f"SCMR adapter-only optimization: {trainable:,} trainable parameters")

    def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-4, iterations=1e5):
        """Build an optimizer containing only SCMR adapter tensors."""
        if str(name).lower() != "adamw":
            raise ValueError("SCMR adapter training intentionally supports only optimizer=AdamW.")
        named = [(n, p) for n, p in unwrap_model(model).named_parameters() if p.requires_grad]
        decay_params = [p for _, p in named if p.ndim > 1]
        no_decay_params = [p for _, p in named if p.ndim <= 1]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": decay, "param_group": "weight"},
                {"params": no_decay_params, "weight_decay": 0.0, "param_group": "bias"},
            ],
            lr=lr,
            betas=(momentum, 0.999),
        )
        LOGGER.info(
            f"optimizer: AdamW(lr={lr}) with {len(decay_params)} decay and {len(no_decay_params)} no-decay tensors"
        )
        return optimizer

    def _model_train(self) -> None:
        """Train adapters while keeping every frozen normalization/dropout layer deterministic."""
        self.model.train()
        for name, module in self.model.named_modules():
            if "scmr_adapters" not in name and isinstance(
                module, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)
            ):
                module.eval()

    def _setup_train(self) -> None:
        """Skip online AMP probes, keep EMA lean, and enforce zero-adapter identity."""
        import ultralytics.engine.trainer as trainer_module

        original_check_amp = trainer_module.check_amp
        trainer_module.check_amp = lambda _: True
        try:
            super()._setup_train()
        finally:
            trainer_module.check_amp = original_check_amp
        if self.ema is not None and hasattr(unwrap_model(self.ema.ema), "criterion"):
            delattr(unwrap_model(self.ema.ema), "criterion")
        self._assert_identity_initialization()

    def _assert_identity_initialization(self) -> None:
        """Abort unless a fresh SCMR adapter is an exact no-op on detector logits and boxes."""
        model = unwrap_model(self.model)
        head = model.model[-1]
        if any(adapter.up.weight.detach().abs().max().item() > 0 for adapter in head.scmr_adapters):
            LOGGER.info("Loaded non-zero SCMR adapters; skipping fresh-adapter identity assertion.")
            return
        batch = self.preprocess_batch(next(iter(self.train_loader)))
        self.model.eval()
        with torch.no_grad():
            head.scmr_adapter_enabled = True
            adapted = self.model(batch["img"], txt_feats=batch["txt_feats"])
            head.scmr_adapter_enabled = False
            baseline = self.model(batch["img"], txt_feats=batch["txt_feats"])
            head.scmr_adapter_enabled = True
        adapted = adapted[1] if isinstance(adapted, tuple) else adapted
        baseline = baseline[1] if isinstance(baseline, tuple) else baseline
        box_error = float((adapted["boxes"] - baseline["boxes"]).abs().max())
        score_error = float((adapted["scores"] - baseline["scores"]).abs().max())
        if box_error > 1e-6 or score_error > 1e-6:
            raise RuntimeError(
                f"Zero-SCMR-adapter identity failed: box_error={box_error:.3e}, score_error={score_error:.3e}."
            )
        LOGGER.info(f"Zero-SCMR-adapter identity passed: boxes={box_error:.1e}, scores={score_error:.1e}")

    def get_validator(self):
        """Use unchanged detector validation and expose the two adapter losses."""
        validator = super().get_validator()
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "scmr_loss", "preserve_loss"
        return validator
