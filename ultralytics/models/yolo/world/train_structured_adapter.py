# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Baseline-preserving structured residual adapter training for YOLO-World."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.nn.modules.structured_adapter import attach_structured_adapters
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model


def normalize_phrase(text: Any) -> str:
    """Normalize a grounding phrase for deterministic sidecar lookup."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).strip(" .,:;|\"'")


class StructuredAdapterLoss(v8DetectionLoss):
    """Original YOLO-World detection loss plus foreground structured sigmoid loss."""

    def __init__(
        self,
        model: nn.Module,
        prototypes: torch.Tensor,
        gain: float,
        negative_gain: float,
        temperature: float,
        max_candidates: int,
    ) -> None:
        super().__init__(model)
        self.head = model.model[-1]
        self.prototypes = F.normalize(prototypes.to(self.device).float(), dim=-1)
        self.gain = float(gain)
        self.negative_gain = float(negative_gain)
        self.temperature = float(temperature)
        self.max_candidates = int(max_candidates)
        if self.gain <= 0 or self.negative_gain < 0 or self.temperature <= 0 or self.max_candidates < 1:
            raise ValueError("Invalid structured loss configuration.")

    @staticmethod
    def _flatten_aligned(features: list[torch.Tensor], heads: nn.ModuleList) -> torch.Tensor:
        """Normalize and flatten P3/P4/P5 maps in WorldDetect anchor order."""
        output = []
        for feature, contrastive_head in zip(features, heads):
            norm = getattr(contrastive_head, "norm", None)
            if norm is not None:
                feature = norm(feature)
            feature = F.normalize(feature.float(), dim=1)
            output.append(feature.flatten(2))
        return torch.cat(output, dim=2).transpose(1, 2).contiguous()

    def _group_structured_ids(
        self, flat_ids: torch.Tensor, batch_idx: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Group flat per-instance atom ids into TaskAlignedAssigner's padded GT layout."""
        atom_count = flat_ids.shape[1]
        counts = torch.bincount(batch_idx.long(), minlength=batch_size)
        max_boxes = int(counts.max().item()) if counts.numel() else 0
        grouped = torch.full(
            (batch_size, max_boxes, atom_count), -1, dtype=torch.long, device=self.device
        )
        if not flat_ids.numel() or max_boxes == 0:
            return grouped
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets[1:] = counts.cumsum(0)
        within = torch.arange(flat_ids.shape[0], device=self.device) - offsets[batch_idx.long()]
        grouped[batch_idx.long(), within] = flat_ids.long()
        return grouped

    def structured_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise sigmoid loss on assigned positive anchors and cached structured atoms."""
        flat_ids = batch.get("structured_ids")
        aligned = preds.get("aligned_feats")
        if flat_ids is None or aligned is None or not fg_mask.any():
            return preds["scores"].new_zeros(())

        batch_size = fg_mask.shape[0]
        flat_ids = flat_ids.to(self.device, non_blocking=True).long()
        batch_idx = batch["batch_idx"].to(self.device, non_blocking=True).long()
        grouped = self._group_structured_ids(flat_ids, batch_idx, batch_size)
        if grouped.shape[1] == 0:
            return preds["scores"].new_zeros(())
        assigned_ids = grouped[
            torch.arange(batch_size, device=self.device)[:, None], target_gt_idx.clamp(0, grouped.shape[1] - 1)
        ][fg_mask]
        valid_ids = assigned_ids[(assigned_ids >= 0) & (assigned_ids < self.prototypes.shape[0])]
        if not valid_ids.numel():
            return preds["scores"].new_zeros(())

        positive_ids = torch.unique(valid_ids, sorted=True)
        candidate_ids = positive_ids
        remaining = max(self.max_candidates - positive_ids.numel(), 0)
        if remaining:
            available = torch.ones(self.prototypes.shape[0], dtype=torch.bool, device=self.device)
            available[positive_ids] = False
            negative_ids = available.nonzero(as_tuple=False).squeeze(1)
            if negative_ids.numel() > remaining:
                negative_ids = negative_ids[torch.randperm(negative_ids.numel(), device=self.device)[:remaining]]
            candidate_ids = torch.cat((positive_ids, negative_ids))

        visual = self._flatten_aligned(aligned, self.head.cv4)[fg_mask]
        logits = visual @ self.prototypes[candidate_ids].t() / self.temperature
        positive_mask = assigned_ids[:, :, None].eq(candidate_ids[None, None, :]).any(dim=1)
        valid_rows = positive_mask.any(dim=1)
        if not valid_rows.any():
            return logits.new_zeros(())
        logits, positive_mask = logits[valid_rows], positive_mask[valid_rows]
        positive_loss = F.softplus(-logits)[positive_mask].mean()
        negative_mask = ~positive_mask
        negative_loss = F.softplus(logits)[negative_mask].mean() if negative_mask.any() else logits.new_zeros(())
        return positive_loss + self.negative_gain * negative_loss

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return box, classification, DFL and weighted structured losses."""
        batch_size = preds["boxes"].shape[0]
        assigned, detection_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        fg_mask, target_gt_idx = assigned[:2]
        structured = self.structured_loss(preds, batch, fg_mask, target_gt_idx) * self.gain
        losses = torch.cat((detection_loss, structured.reshape(1)))
        return losses * batch_size, losses.detach()


class StructuredAdapterWorldTrainer(WorldTrainerFromScratch):
    """Train only zero-initialized dense adapters while preserving the base detector."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.phrase_to_atom_ids: dict[str, list[int]] = {}
        self.max_atoms_per_label = 0

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Create WorldModel, attach adapters, then load base or adapter weights without shifting cv3 keys."""
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        attach_structured_adapters(model.model[-1], hidden_dim=int(self.args.structured_adapter_hidden))
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Load the sidecar cache and install the structured criterion."""
        super().set_model_attributes()
        cache_path = Path(self.args.structured_adapter_cache).expanduser()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Structured adapter cache not found: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(cache, dict) or int(cache.get("version", 0)) != 3:
            raise ValueError(f"Unsupported structured adapter cache: {cache_path}")
        if cache.get("base_annotations_unchanged") is not True:
            raise ValueError("Structured cache does not certify unchanged base multi-positive annotations; rebuild it.")
        prototypes = cache.get("prototypes")
        mapping = cache.get("phrase_to_atom_ids")
        if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2 or not isinstance(mapping, dict):
            raise ValueError("Structured cache must contain 2D prototypes and phrase_to_atom_ids.")
        model = unwrap_model(self.model)
        expected_dim = int(model.model[-1].cv3[0][-1].out_channels)
        if prototypes.shape[1] != expected_dim:
            raise ValueError(
                f"Structured prototype width {prototypes.shape[1]} does not match model embedding width {expected_dim}."
            )
        cached_text_model = str(cache.get("text_model", ""))
        active_text_model = str(getattr(model, "text_model", ""))
        if cached_text_model and active_text_model and cached_text_model != active_text_model:
            raise ValueError(
                f"Structured cache text encoder {cached_text_model!r} does not match model encoder "
                f"{active_text_model!r}."
            )
        self.phrase_to_atom_ids = {normalize_phrase(k): [int(x) for x in v] for k, v in mapping.items()}
        self.max_atoms_per_label = max((len(v) for v in self.phrase_to_atom_ids.values()), default=0)
        if self.max_atoms_per_label < 1:
            raise ValueError("Structured cache contains no usable phrase-to-atom assignments.")
        model.criterion = StructuredAdapterLoss(
            model=model,
            prototypes=prototypes,
            gain=float(self.args.structured_adapter_gain),
            negative_gain=float(self.args.structured_adapter_negative_gain),
            temperature=float(self.args.structured_adapter_temperature),
            max_candidates=int(self.args.structured_adapter_max_candidates),
        )
        LOGGER.info(
            f"Structured adapter cache: {len(self.phrase_to_atom_ids):,} phrases, "
            f"{prototypes.shape[0]:,} atoms, {prototypes.shape[1]}-D"
        )

    def configure_trainable_parameters(self):
        """Freeze the complete detector and leave only structured adapters trainable before DDP construction."""
        model = unwrap_model(self.model)
        trainable = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "structured_adapters" in name
            trainable += parameter.numel() if parameter.requires_grad else 0
        if not trainable:
            raise RuntimeError("No structured adapter parameters were found.")
        LOGGER.info(f"Adapter-only optimization: {trainable:,} trainable parameters")

    def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-4, iterations=1e5):
        """Build a compact optimizer containing adapter parameters only."""
        if str(name).lower() != "adamw":
            raise ValueError("Structured adapter training intentionally supports only optimizer=AdamW.")
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
            f"optimizer: AdamW(lr={lr}, betas=({momentum}, 0.999)) with "
            f"{len(decay_params)} decay and {len(no_decay_params)} no-decay tensors"
        )
        return optimizer

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Resolve each transformed primary phrase to fixed structured atom ids before device transfer."""
        classes = batch["cls"].view(-1).long().tolist()
        image_ids = batch["batch_idx"].view(-1).long().tolist()
        texts = batch["texts"]
        structured_ids = torch.full((len(classes), self.max_atoms_per_label), -1, dtype=torch.long)
        for index, (image_id, cls_id) in enumerate(zip(image_ids, classes)):
            if image_id >= len(texts) or cls_id >= len(texts[image_id]):
                raise IndexError(f"Invalid transformed text index image={image_id}, cls={cls_id}.")
            ids = self.phrase_to_atom_ids.get(normalize_phrase(texts[image_id][cls_id]), [])
            if ids:
                structured_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        if structured_ids.shape[0] != batch["cls"].shape[0]:
            raise RuntimeError("Structured sidecar changed the GT instance count.")
        batch["structured_ids"] = structured_ids
        return super().preprocess_batch(batch)

    def _model_train(self):
        """Keep frozen normalization/dropout state fixed while adapters train."""
        self.model.train()
        for name, module in self.model.named_modules():
            if "structured_adapters" not in name and isinstance(
                module, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)
            ):
                module.eval()

    def _setup_train(self):
        """Run standard setup, keep EMA serialization lean, and verify identity initialization."""
        super()._setup_train()
        if self.ema is not None and hasattr(unwrap_model(self.ema.ema), "criterion"):
            delattr(unwrap_model(self.ema.ema), "criterion")
        self._assert_identity_initialization()

    def _assert_identity_initialization(self) -> None:
        """Abort if a newly attached adapter changes checkpoint logits before training."""
        model = unwrap_model(self.model)
        head = model.model[-1]
        if any(adapter.up.weight.detach().abs().max().item() > 0 for adapter in head.structured_adapters):
            LOGGER.info("Loaded non-zero adapter weights; skipping new-adapter identity assertion.")
            return
        batch = self.preprocess_batch(next(iter(self.train_loader)))
        self.model.eval()
        with torch.no_grad():
            head.structured_adapter_enabled = True
            adapted = self.model(batch["img"], txt_feats=batch["txt_feats"])
            head.structured_adapter_enabled = False
            baseline = self.model(batch["img"], txt_feats=batch["txt_feats"])
            head.structured_adapter_enabled = True
        adapted = adapted[1] if isinstance(adapted, tuple) else adapted
        baseline = baseline[1] if isinstance(baseline, tuple) else baseline
        box_error = float((adapted["boxes"] - baseline["boxes"]).abs().max())
        score_error = float((adapted["scores"] - baseline["scores"]).abs().max())
        if box_error > 1e-6 or score_error > 1e-6:
            raise RuntimeError(
                f"Zero-adapter identity check failed: box_error={box_error:.3e}, score_error={score_error:.3e}."
            )
        LOGGER.info(f"Zero-adapter identity check passed: boxes={box_error:.1e}, scores={score_error:.1e}")

    def get_validator(self):
        """Use normal detector validation and expose the structured training loss column."""
        validator = super().get_validator()
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "structured_loss"
        return validator
