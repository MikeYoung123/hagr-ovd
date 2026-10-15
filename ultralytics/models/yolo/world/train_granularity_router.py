# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Primary-preserving hierarchical category/attribute/affordance routing for YOLO-World."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.nn.modules.granularity_router import (
    ROLE_AFFORDANCE,
    ROLE_ATTRIBUTE,
    ROLE_CATEGORY,
    attach_granularity_router,
    set_granularity_inference_role,
)
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model


ATTRIBUTE_CODE_MIN = 2.5
CATEGORY_CODE_MIN = 1.5
GRANULARITY_TRAIN_STAGES = {"gate", "gate_child", "full", "temporal_joint"}


def normalize_phrase(value: Any) -> str:
    """Normalize only formatting for exact train-vocabulary routing."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower()).strip(" .,:;|\"'")


class PrimaryPreservingGranularityLoss(v8DetectionLoss):
    """Train the full hierarchy on immutable base assignment with attribute retention.

    Base scores determine TaskAlignedAssigner foregrounds and are not routed.
    Three independently normalized role losses update their routed adapters,
    while a base-logit trust region limits drift of the strong attribute route.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        raw_gains = {
            "category": float(model.args.granularity_category_gain),
            "attribute": float(model.args.granularity_attribute_gain),
            "affordance": float(model.args.granularity_affordance_gain),
        }
        if any(value <= 0 for value in raw_gains.values()):
            raise ValueError(f"granularity role gain ratios must all be positive, got {raw_gains}")
        gain_sum = sum(raw_gains.values())
        self.category_gain = raw_gains["category"] / gain_sum
        self.attribute_gain = raw_gains["attribute"] / gain_sum
        self.affordance_gain = raw_gains["affordance"] / gain_sum
        self.attribute_preserve_gain = float(model.args.granularity_attribute_preserve_gain)
        if self.attribute_preserve_gain < 0:
            raise ValueError("granularity_attribute_preserve_gain must be non-negative")
        self.train_localization = str(model.args.granularity_train_stage) == "temporal_joint"

    @staticmethod
    def _state_role(states: torch.Tensor, role: str) -> torch.Tensor:
        if role == "attribute":
            return states >= ATTRIBUTE_CODE_MIN
        if role == "category":
            return (states >= CATEGORY_CODE_MIN) & (states < ATTRIBUTE_CODE_MIN)
        if role == "affordance":
            return (states > 0) & (states < CATEGORY_CODE_MIN)
        raise KeyError(role)

    def _group_supervision(self, batch: dict[str, torch.Tensor], batch_size: int, max_boxes: int) -> torch.Tensor:
        """Group signed per-instance prompt supervision into padded GT order."""
        multi = batch.get("multi_positive")
        if multi is None or multi.ndim != 3 or multi.shape[-1] != 2:
            raise RuntimeError(
                "Granularity routing requires multi_positive[N,K,2]. Use "
                "the published tri-granular annotations and remove the stale derived *.cache file."
            )
        if multi.shape[0] != batch["batch_idx"].shape[0]:
            raise RuntimeError("multi_positive and GT instance counts differ")
        return self.preprocess_multi_positive(
            multi.to(self.device, non_blocking=True).float(),
            batch["batch_idx"].to(self.device, non_blocking=True).long(),
            batch_size,
            max_boxes,
        )

    def _role_targets(
        self,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        grouped: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build full-quality role positives and an assigned-foreground unknown mask."""
        batch_size = target_scores.shape[0]
        targets = {
            "attribute": target_scores.clone(),
            "category": torch.zeros_like(target_scores),
            "affordance": torch.zeros_like(target_scores),
        }
        if grouped.shape[1] == 0:
            return targets, torch.ones_like(target_scores, dtype=torch.bool)

        image = torch.arange(batch_size, device=self.device)[:, None]
        assigned = grouped[image, target_gt_idx.clamp(0, grouped.shape[1] - 1)]
        labels, states = assigned[..., 0].long(), assigned[..., 1]
        quality = target_scores.sum(-1, keepdim=True)
        for role in targets:
            valid = (
                self._state_role(states, role)
                & labels.ge(0)
                & labels.lt(self.nc)
                & fg_mask[..., None]
            )
            if not valid.any():
                continue
            extra = torch.zeros_like(target_scores)
            extra.scatter_add_(2, labels.clamp(0, self.nc - 1), quality * valid.to(quality.dtype))
            targets[role] = torch.maximum(targets[role], extra.clamp_max(1.0))

        unknown = states.eq(-1) & labels.ge(0) & labels.lt(self.nc) & fg_mask[..., None]
        known_mask = torch.ones_like(target_scores, dtype=torch.bool)
        if unknown.any():
            counts = torch.zeros_like(target_scores)
            counts.scatter_add_(2, labels.clamp(0, self.nc - 1), unknown.to(target_scores.dtype))
            known_mask = counts.eq(0)
            # Positive wins if a malformed artifact contains both states.
            known_mask |= sum(targets.values()).gt(0)
        return targets, known_mask

    def _role_bce(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        prompt_roles: torch.Tensor,
        role_id: int,
        known_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute independently normalized BCE for exactly one routed role."""
        role_mask = prompt_roles.eq(role_id)
        if not role_mask.any() or not targets.gt(0).any():
            return pred_scores.sum() * 0.0
        element_mask = known_mask & role_mask[:, None, :]
        bce = self.bce(pred_scores, targets.to(pred_scores.dtype))
        if self.class_weights is not None:
            bce *= self.class_weights
        bce *= element_mask.to(bce.dtype)
        return bce.sum() / targets.sum().clamp_min(1.0)

    @staticmethod
    def _attribute_preserve_loss(
        pred_scores: torch.Tensor,
        base_scores: torch.Tensor,
        prompt_roles: torch.Tensor,
    ) -> torch.Tensor:
        """Keep attribute logits in a trust region around the frozen MP checkpoint."""
        mask = prompt_roles.eq(ROLE_ATTRIBUTE)[:, None, :]
        if not mask.any():
            return pred_scores.sum() * 0.0
        distance = F.smooth_l1_loss(
            pred_scores.float(), base_scores.detach().float(), reduction="none", beta=1.0
        )
        expanded = mask.expand_as(distance)
        return distance.masked_select(expanded).mean()

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return localization, three routed role losses, and attribute retention."""
        base_score_maps = preds.get("granularity_base_scores")
        prompt_roles = batch.get("prompt_roles")
        if base_score_maps is None:
            raise RuntimeError("WorldDetect did not return granularity_base_scores; router is not active")
        if prompt_roles is None:
            raise RuntimeError("Training batch has no prompt_roles; use PrimaryPreservingGranularityTrainer")

        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        base_scores = base_score_maps.permute(0, 2, 1).contiguous()
        prompt_roles = prompt_roles.to(self.device, non_blocking=True).long()
        if tuple(prompt_roles.shape) != (pred_scores.shape[0], pred_scores.shape[2]):
            raise RuntimeError(
                f"prompt_roles shape {tuple(prompt_roles.shape)} does not match scores "
                f"{(pred_scores.shape[0], pred_scores.shape[2])}"
            )

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size, dtype = pred_scores.shape[0], pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        raw_targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        raw_targets = self.preprocess(raw_targets.to(self.device), batch_size, imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = raw_targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Unrouted base scores own foreground assignment, keeping semantic
        # target assignment independent of the selected role route.
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            base_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_min(1.0)
        grouped = self._group_supervision(batch, batch_size, gt_bboxes.shape[1])
        role_targets, known_mask = self._role_targets(target_scores, fg_mask, target_gt_idx, grouped)

        attr_loss = self._role_bce(
            pred_scores, role_targets["attribute"], prompt_roles, ROLE_ATTRIBUTE, known_mask
        )
        category_loss = self._role_bce(
            pred_scores, role_targets["category"], prompt_roles, ROLE_CATEGORY, known_mask
        )
        affordance_loss = self._role_bce(
            pred_scores, role_targets["affordance"], prompt_roles, ROLE_AFFORDANCE, known_mask
        )
        attribute_preserve_loss = self._attribute_preserve_loss(pred_scores, base_scores, prompt_roles)

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

        box_term = box_loss * self.hyp.box
        dfl_term = dfl_loss * self.hyp.dfl
        attribute_preserve_term = attribute_preserve_loss * self.hyp.cls * self.attribute_preserve_gain
        diagnostics = torch.stack(
            (
                box_term.detach(),
                attr_loss * self.hyp.cls * self.attribute_gain,
                category_loss * self.hyp.cls * self.category_gain,
                affordance_loss * self.hyp.cls * self.affordance_gain,
                attribute_preserve_loss.detach(),
                dfl_term.detach(),
            )
        )
        trainable = diagnostics[1] + diagnostics[2] + diagnostics[3] + attribute_preserve_term
        if self.train_localization:
            trainable = trainable + box_term + dfl_term
        if not torch.isfinite(trainable):
            values = {
                "box": float(box_loss.detach()),
                "dfl": float(dfl_loss.detach()),
                "attribute": float(attr_loss.detach()),
                "category": float(category_loss.detach()),
                "affordance": float(affordance_loss.detach()),
                "attribute_drift": float(attribute_preserve_loss.detach()),
            }
            raise FloatingPointError(f"Non-finite HAGR loss before optimizer step: {values}")
        return trainable * batch_size, diagnostics.detach()


class PrimaryPreservingGranularityTrainer(WorldTrainerFromScratch):
    """Optimize category-rooted routing, optionally with temporal box-branch fine-tuning."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        self.phrase_roles: dict[str, int] = {}
        self.ambiguous_phrases: set[str] = set()
        self._routing_logged = False
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Attach v5 gates before loading base or legacy-v4 router weights."""
        stage = str(self.args.granularity_train_stage)
        if stage not in GRANULARITY_TRAIN_STAGES:
            raise ValueError(f"Unsupported granularity_train_stage={stage!r}")
        source = weights.get("model") if isinstance(weights, dict) else weights
        source_head = source.model[-1] if source is not None and hasattr(source, "model") else None
        source_router = getattr(source_head, "granularity_router", None)
        if source_router is not None:
            source_stop_gradient = bool(getattr(source_router, "stop_gradient", True))
            requested_stop_gradient = bool(self.args.granularity_stop_gradient)
            if source_stop_gradient != requested_stop_gradient:
                raise RuntimeError(
                    f"Source router stop_gradient={source_stop_gradient} differs from requested "
                    f"{requested_stop_gradient}; explicitly preserve the checkpoint routing semantics"
                )
        if stage in {"gate", "gate_child", "temporal_joint"}:
            if source_router is None:
                raise RuntimeError(
                    f"granularity_train_stage={stage!r} requires a trained v4/v5 router checkpoint; "
                    "use stage='full' when starting from the aligned detector"
                )
            if stage == "temporal_joint" and int(getattr(source_router, "router_version", 0)) != 5:
                raise RuntimeError("temporal_joint requires the trained v5 direct-inheritance HAGR checkpoint")
            source_prior = float(getattr(source_router, "category_prior", 0.0))
            requested_prior = float(self.args.granularity_category_prior)
            if requested_prior <= 0.0:
                raise RuntimeError(f"granularity_train_stage={stage!r} requires category_prior > 0")
            if abs(source_prior - requested_prior) > 1e-12:
                raise RuntimeError(
                    f"Source router category_prior={source_prior} differs from requested {requested_prior}; "
                    "continued routing stages must preserve the source category-rooted route"
                )
            source_max = getattr(source_router, "inheritance_max", None)
            if source_max is not None and abs(float(source_max) - float(self.args.granularity_inheritance_max)) > 1e-12:
                raise RuntimeError(
                    f"Source router inheritance_max={source_max} differs from requested "
                    f"{self.args.granularity_inheritance_max}"
                )
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        attach_granularity_router(
            model.model[-1],
            hidden_dim=int(self.args.granularity_router_hidden),
            category_prior=float(self.args.granularity_category_prior),
            inheritance_max=float(self.args.granularity_inheritance_max),
            stop_gradient=bool(self.args.granularity_stop_gradient),
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self) -> None:
        """Load exact role vocabularies and install the primary-preserving criterion."""
        super().set_model_attributes()
        path = Path(self.args.granularity_router_vocab).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Granularity train vocabulary not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        role_sets = {}
        for role, role_id in (
            ("category", ROLE_CATEGORY),
            ("attribute", ROLE_ATTRIBUTE),
            ("affordance", ROLE_AFFORDANCE),
        ):
            values = {normalize_phrase(value) for value in payload.get(role, []) if normalize_phrase(value)}
            if not values:
                raise ValueError(f"Granularity vocabulary has no {role} phrases: {path}")
            role_sets[role_id] = values
        all_phrases = set().union(*role_sets.values())
        self.ambiguous_phrases = {
            phrase for phrase in all_phrases if sum(phrase in values for values in role_sets.values()) > 1
        }
        # Attribute precedence matches the deterministic dataset's collision
        # rule; positive role codes override this fallback per image below.
        precedence = (ROLE_AFFORDANCE, ROLE_CATEGORY, ROLE_ATTRIBUTE)
        self.phrase_roles = {
            phrase: max(role_id for role_id in precedence if phrase in role_sets[role_id]) for phrase in all_phrases
        }
        model = unwrap_model(self.model)
        model.criterion = PrimaryPreservingGranularityLoss(model)
        LOGGER.info(
            f"Granularity vocab: {len(all_phrases):,} phrases, {len(self.ambiguous_phrases):,} cross-role text "
            f"collisions; attribute precedence with per-GT positive override"
        )

    def configure_trainable_parameters(self) -> None:
        """Select a router stage; temporal_joint also opens only the shared box/DFL branch."""
        model = unwrap_model(self.model)
        stage = str(self.args.granularity_train_stage)
        if stage not in GRANULARITY_TRAIN_STAGES:
            raise ValueError(f"Unsupported granularity_train_stage={stage!r}")
        head = model.model[-1]
        box_parameter_ids = {id(parameter) for parameter in head.cv2.parameters()}

        def selected(name: str, parameter: nn.Parameter) -> bool:
            marker = f".{name}."
            inheritance = ".granularity_router.attribute_inheritance." in marker or (
                ".granularity_router.affordance_inheritance." in marker
            )
            child = ".granularity_router.attribute." in marker or ".granularity_router.affordance." in marker
            if stage == "gate":
                return inheritance
            if stage == "gate_child":
                return inheritance or child
            if stage == "temporal_joint":
                return ".granularity_router." in marker or id(parameter) in box_parameter_ids
            return ".granularity_router." in marker

        trainable = 0
        tensors = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = selected(name, parameter)
            if parameter.requires_grad:
                trainable += parameter.numel()
                tensors += 1
        if trainable < 1:
            raise RuntimeError(f"No granularity-router parameters were selected for stage={stage!r}")
        router = head.granularity_router
        router.train_stage = stage
        if stage in {"gate", "gate_child", "temporal_joint"}:
            category_signal = max(
                max(float(adapter.up.weight.detach().abs().max()), float(adapter.up.bias.detach().abs().max()))
                for adapter in router.category
            )
            if category_signal == 0.0:
                raise RuntimeError(
                    f"stage={stage!r} loaded a zero category residual; start from the trained "
                    "HAGR checkpoint, not the pre-router base checkpoint"
                )
        LOGGER.info(
            f"Direct category-rooted stage={stage}: {trainable:,} parameters in {tensors} tensors; "
            f"category->(attribute,affordance) prior={router.category_prior:g}, "
            f"inheritance_max={router.inheritance_max:g}, "
            f"stop_gradient={bool(getattr(router, 'stop_gradient', True))}, "
            f"shared_box_branch_trainable={stage == 'temporal_joint'}, backbone_neck_trainable=False"
        )

    def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-4, iterations=1e5):
        """Build stage-aware AdamW groups for gates, child routes, and shared residuals."""
        if str(name).lower() != "adamw":
            raise ValueError("Granularity-router training intentionally supports only optimizer=AdamW")
        named = [(n, p) for n, p in unwrap_model(model).named_parameters() if p.requires_grad]
        shared_lr_mult = float(self.args.granularity_shared_lr_mult)
        box_lr_mult = float(self.args.granularity_box_lr_mult)
        if not 0.0 < shared_lr_mult <= 1.0:
            raise ValueError(f"granularity_shared_lr_mult must be in (0, 1], got {shared_lr_mult}")
        if not 0.0 < box_lr_mult <= 1.0:
            raise ValueError(f"granularity_box_lr_mult must be in (0, 1], got {box_lr_mult}")
        head = unwrap_model(model).model[-1]
        box_parameter_ids = {id(parameter) for parameter in head.cv2.parameters()}

        parameter_by_name = dict(named)

        def scope(name: str) -> str:
            parameter = parameter_by_name[name]
            if id(parameter) in box_parameter_ids:
                return "box"
            marker = f".{name}."
            if ".granularity_router.attribute_inheritance." in marker or (
                ".granularity_router.affordance_inheritance." in marker
            ):
                return "inheritance"
            if ".granularity_router.shared." in marker:
                return "shared"
            return "role"

        groups = []
        counts = {}
        for group_scope, selected, group_lr in (
            ("inheritance", [(n, p) for n, p in named if scope(n) == "inheritance"], lr),
            ("role", [(n, p) for n, p in named if scope(n) == "role"], lr),
            ("shared", [(n, p) for n, p in named if scope(n) == "shared"], lr * shared_lr_mult),
            ("box", [(n, p) for n, p in named if scope(n) == "box"], lr * box_lr_mult),
        ):
            # Channel gates are already bounded and should not be biased back
            # toward zero by weight decay during the short gate-only probe.
            decay_params = [p for _, p in selected if p.ndim > 1 and group_scope != "inheritance"]
            no_decay_params = [p for _, p in selected if p.ndim <= 1 or group_scope == "inheritance"]
            if decay_params:
                groups.append(
                    {
                        "params": decay_params,
                        "weight_decay": decay,
                        "lr": group_lr,
                        "param_group": f"{group_scope}_weight",
                    }
                )
            if no_decay_params:
                groups.append(
                    {
                        "params": no_decay_params,
                        "weight_decay": 0.0,
                        "lr": group_lr,
                        "param_group": f"{group_scope}_bias",
                    }
                )
            counts[group_scope] = (len(decay_params), len(no_decay_params))
        if not groups or sum(counts.get(name, (0, 0))[1] for name in ("inheritance",)) < 1:
            raise RuntimeError("Granularity optimizer did not find the direct inheritance gates")
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=(momentum, 0.999))
        LOGGER.info(
            f"optimizer: AdamW role_lr={lr:g}, shared_lr={lr * shared_lr_mult:g}, "
            f"box_lr={lr * box_lr_mult:g}; "
            f"inheritance tensors={counts['inheritance']}, role tensors={counts['role']}, "
            f"shared tensors={counts['shared']}, box tensors={counts['box']}"
        )
        return optimizer

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Install exact per-query roles after text sampling and before model forward."""
        batch = super().preprocess_batch(batch)
        texts = batch["texts"]
        batch_size = len(texts)
        classes = len(texts[0]) if batch_size else 0
        if classes < 1 or any(len(values) != classes for values in texts):
            raise RuntimeError("Granularity routing requires a rectangular per-image prompt batch")
        roles = torch.zeros(batch_size, classes, dtype=torch.long, device=self.device)
        for image_id, values in enumerate(texts):
            for class_id, phrase in enumerate(values):
                roles[image_id, class_id] = self.phrase_roles.get(normalize_phrase(phrase), 0)

        multi = batch.get("multi_positive")
        batch_idx = batch["batch_idx"].long()
        if multi is None or multi.shape[0] != batch_idx.shape[0]:
            raise RuntimeError("Missing or misaligned multi_positive supervision")
        positive_roles = torch.zeros_like(roles)
        labels, states = multi[..., 0].long(), multi[..., 1]
        images = batch_idx[:, None].expand_as(labels)
        for role_id, valid in (
            (ROLE_AFFORDANCE, states.gt(0) & states.lt(CATEGORY_CODE_MIN)),
            (ROLE_CATEGORY, states.ge(CATEGORY_CODE_MIN) & states.lt(ATTRIBUTE_CODE_MIN)),
            (ROLE_ATTRIBUTE, states.ge(ATTRIBUTE_CODE_MIN)),
        ):
            valid &= labels.ge(0) & labels.lt(classes)
            if valid.any():
                # Role order gives attribute > category > affordance precedence
                # when identical text is positive under multiple roles.
                positive_roles[images[valid], labels[valid]] = role_id
        roles = torch.where(positive_roles.gt(0), positive_roles, roles)

        positive = multi[..., 1].gt(0) & multi[..., 0].ge(0) & multi[..., 0].lt(classes)
        if positive.any():
            images = batch_idx[:, None].expand_as(positive)[positive]
            labels = multi[..., 0].long()[positive]
            if roles[images, labels].eq(0).any():
                raise RuntimeError("A sampled positive prompt has no granularity route")

        batch["prompt_roles"] = roles
        head = unwrap_model(self.model).model[-1]
        head.granularity_prompt_roles = roles
        head.granularity_router_enabled = True
        if not self._routing_logged:
            counts = {role: int(roles.eq(role_id).sum()) for role, role_id in (
                ("category", ROLE_CATEGORY),
                ("attribute", ROLE_ATTRIBUTE),
                ("affordance", ROLE_AFFORDANCE),
            )}
            counts["unrouted"] = int(roles.eq(0).sum())
            LOGGER.info(f"First sampled prompt routing counts: {counts}")
            self._routing_logged = True
        return batch

    def _model_train(self) -> None:
        """Keep frozen normalization/dropout state fixed while adapters train."""
        self.model.train()
        for name, module in self.model.named_modules():
            if ".granularity_router." not in f".{name}." and isinstance(
                module, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)
            ):
                module.eval()

    def _setup_train(self) -> None:
        """Keep EMA lean and prove fresh routing is initially an exact no-op."""
        super()._setup_train()
        if self.ema is not None and hasattr(unwrap_model(self.ema.ema), "criterion"):
            delattr(unwrap_model(self.ema.ema), "criterion")
        self._assert_identity_initialization()

    def _assert_identity_initialization(self) -> None:
        """Abort if a freshly attached router changes the base checkpoint."""
        model = unwrap_model(self.model)
        head = model.model[-1]
        router = head.granularity_router
        fresh = all(
            adapter.up.weight.detach().abs().max().item() == 0
            and adapter.up.bias.detach().abs().max().item() == 0
            for adapter in (*router.shared, *router.category, *router.attribute, *router.affordance)
        ) and all(
            gate.logits.detach().abs().max().item() == 0
            for gate in (*router.attribute_inheritance, *router.affordance_inheritance)
        )
        if not fresh:
            LOGGER.info("Loaded non-zero granularity router; skipping fresh-router identity assertion")
            return
        batch = self.preprocess_batch(next(iter(self.train_loader)))
        self.model.eval()
        with torch.no_grad():
            head.granularity_router_enabled = True
            routed = self.model(batch["img"], txt_feats=batch["txt_feats"])
            head.granularity_router_enabled = False
            baseline = self.model(batch["img"], txt_feats=batch["txt_feats"])
            head.granularity_router_enabled = True
        routed = routed[1] if isinstance(routed, tuple) else routed
        baseline = baseline[1] if isinstance(baseline, tuple) else baseline
        box_error = float((routed["boxes"] - baseline["boxes"]).abs().max())
        score_error = float((routed["scores"] - baseline["scores"]).abs().max())
        if box_error > 1e-6 or score_error > 1e-6:
            raise RuntimeError(
                f"Zero-router identity check failed: box_error={box_error:.3e}, score_error={score_error:.3e}"
            )
        LOGGER.info(f"Zero-router identity check passed: boxes={box_error:.1e}, scores={score_error:.1e}")

    def validate(self):
        """Select checkpoints with the role configured for the active training stage."""
        role = str(self.args.granularity_selection_role)
        if role not in {"category", "attribute", "affordance"}:
            raise ValueError(f"Unsupported granularity_selection_role={role!r}")
        set_granularity_inference_role(unwrap_model(self.model), role)
        if self.ema is not None:
            set_granularity_inference_role(unwrap_model(self.ema.ema), role)
        result = super().validate()
        report_model = unwrap_model(self.ema.ema) if self.ema is not None else unwrap_model(self.model)
        router = report_model.model[-1].granularity_router
        LOGGER.info(f"Checkpoint role={role}; direct inheritance gates: {router.inheritance_statistics()}")
        return result

    def get_validator(self):
        """Expose localization diagnostics, three role losses, and attribute retention."""
        validator = super().get_validator()
        if str(self.args.granularity_train_stage) == "temporal_joint":
            self.loss_names = "box_loss", "attr_loss", "cat_loss", "aff_loss", "attr_drift", "dfl_loss"
        else:
            self.loss_names = "box_diag", "attr_loss", "cat_loss", "aff_loss", "attr_drift", "dfl_diag"
        return validator
