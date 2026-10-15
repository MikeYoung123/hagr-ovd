# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Primary-preserving hierarchical semantic-granularity routing for YOLO-World."""

from __future__ import annotations

import torch
import torch.nn as nn


ROLE_AFFORDANCE = 1
ROLE_CATEGORY = 2
ROLE_ATTRIBUTE = 3
ROLE_TO_ID = {
    "affordance": ROLE_AFFORDANCE,
    "category": ROLE_CATEGORY,
    "attribute": ROLE_ATTRIBUTE,
}


class GranularityResidualAdapter(nn.Module):
    """Zero-initialized bounded dense bottleneck adapter.

    The final projection is initialized to zero, making a fresh adapter an
    exact identity residual. This makes the complete router reproduce the
    initialization checkpoint before optimization.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, max_delta: float = 0.5) -> None:
        super().__init__()
        embed_dim, hidden_dim = int(embed_dim), int(hidden_dim)
        if embed_dim < 1 or hidden_dim < 1:
            raise ValueError(f"embed_dim and hidden_dim must be positive, got {embed_dim}, {hidden_dim}")
        if float(max_delta) <= 0:
            raise ValueError(f"max_delta must be positive, got {max_delta}")
        self.max_delta = float(max_delta)
        self.down = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """Return a bounded residual to prevent noisy role gradients from exploding logits."""
        return self.max_delta * torch.tanh(self.up(self.act(self.down(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the learned role residual to a shared dense embedding map."""
        return x + self.residual(x)


class CategoryInheritanceGate(nn.Module):
    """Zero-initialized bounded channel gate for direct category inheritance.

    The signed tanh gate is initialized at zero, so adding it to an existing
    v4 router is an exact no-op. Once optimized, each child can select which
    category-residual channels to inherit. The router normally detaches this
    input; the no-stop-gradient ablation deliberately leaves it connected.
    """

    def __init__(self, embed_dim: int, max_gain: float = 0.5) -> None:
        super().__init__()
        if int(embed_dim) < 1:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if not 0.0 < float(max_gain) <= 1.0:
            raise ValueError(f"max_gain must be in (0, 1], got {max_gain}")
        self.max_gain = float(max_gain)
        self.logits = nn.Parameter(torch.zeros(1, int(embed_dim), 1, 1))

    def coefficients(self) -> torch.Tensor:
        """Return bounded signed per-channel inheritance coefficients."""
        return self.max_gain * torch.tanh(self.logits)

    def forward(self, category_delta: torch.Tensor) -> torch.Tensor:
        """Select a bounded portion of the supplied category residual."""
        return self.coefficients() * category_delta


class GranularityRouter(nn.Module):
    """Route three granularities through a direct category-rooted hierarchy.

    A shared residual first calibrates the frozen base embedding with balanced
    gradients from all roles. Category then learns an object-identity residual.
    Attribute and affordance both consume the detached category embedding and
    directly inherit a separately gated portion of its residual before adding
    their role-specific residual. By default, child labels cannot update the
    category branch; this stop-gradient can be disabled only for a controlled
    ablation. All residual outputs and inheritance gates are zero-initialized, so
    a fresh router reproduces the initialization checkpoint exactly and a v5
    upgrade reproduces a v4 checkpoint until a gate is optimized.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_scales: int,
        category_prior: float = 1.0,
        inheritance_max: float = 0.5,
        stop_gradient: bool = True,
    ) -> None:
        super().__init__()
        if int(num_scales) < 1:
            raise ValueError(f"num_scales must be positive, got {num_scales}")
        if not 0.0 <= float(category_prior) <= 1.0:
            raise ValueError(f"category_prior must be in [0, 1], got {category_prior}")
        if not 0.0 < float(inheritance_max) <= 1.0:
            raise ValueError(f"inheritance_max must be in (0, 1], got {inheritance_max}")
        self.num_scales = int(num_scales)
        self.router_version = 5
        self.category_prior = float(category_prior)
        self.inheritance_max = float(inheritance_max)
        self.stop_gradient = bool(stop_gradient)
        self.shared = nn.ModuleList(
            GranularityResidualAdapter(embed_dim, hidden_dim) for _ in range(self.num_scales)
        )
        self.category = nn.ModuleList(
            GranularityResidualAdapter(embed_dim, hidden_dim) for _ in range(self.num_scales)
        )
        self.attribute = nn.ModuleList(
            GranularityResidualAdapter(embed_dim, hidden_dim) for _ in range(self.num_scales)
        )
        self.affordance = nn.ModuleList(
            GranularityResidualAdapter(embed_dim, hidden_dim) for _ in range(self.num_scales)
        )
        self.attribute_inheritance = nn.ModuleList(
            CategoryInheritanceGate(embed_dim, self.inheritance_max) for _ in range(self.num_scales)
        )
        self.affordance_inheritance = nn.ModuleList(
            CategoryInheritanceGate(embed_dim, self.inheritance_max) for _ in range(self.num_scales)
        )

    def has_direct_inheritance(self) -> bool:
        """Return whether this checkpoint contains the v5 direct gates."""
        return hasattr(self, "attribute_inheritance") and hasattr(self, "affordance_inheritance")

    def inheritance_statistics(self) -> dict[str, dict[str, list[float]]]:
        """Return auditable gate magnitudes for evaluation reports."""
        result = {}
        for role, gates in (
            ("attribute", getattr(self, "attribute_inheritance", None)),
            ("affordance", getattr(self, "affordance_inheritance", None)),
        ):
            if gates is None:
                result[role] = {"mean_abs": [], "max_abs": []}
                continue
            coefficients = [gate.coefficients().detach().float() for gate in gates]
            result[role] = {
                "mean_abs": [float(value.abs().mean()) for value in coefficients],
                "max_abs": [float(value.abs().max()) for value in coefficients],
            }
        return result

    @staticmethod
    def _normalize_roles(
        role_ids: torch.Tensor | None,
        batch_size: int,
        classes: int,
        device: torch.device,
        default_role: str,
    ) -> torch.Tensor:
        """Return one validated role id per image/query pair."""
        if default_role not in ROLE_TO_ID:
            raise ValueError(f"Unsupported default granularity role: {default_role!r}")
        if role_ids is None:
            return torch.full(
                (batch_size, classes), ROLE_TO_ID[default_role], dtype=torch.long, device=device
            )
        roles = role_ids.to(device=device, dtype=torch.long, non_blocking=True)
        if roles.ndim == 1:
            roles = roles.unsqueeze(0)
        if roles.shape[0] == 1 and batch_size > 1:
            roles = roles.expand(batch_size, -1)
        if tuple(roles.shape) != (batch_size, classes):
            raise ValueError(
                f"Granularity roles must have shape {(batch_size, classes)}, got {tuple(roles.shape)}"
            )
        return roles

    def forward_scale(
        self,
        scale_index: int,
        base_embedding: torch.Tensor,
        text: torch.Tensor,
        contrastive_head: nn.Module,
        role_ids: torch.Tensor | None,
        default_role: str,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return routed scores and the untouched base scores for one scale."""
        if not 0 <= int(scale_index) < self.num_scales:
            raise IndexError(f"Invalid granularity-router scale {scale_index}/{self.num_scales}")
        base_scores = contrastive_head(base_embedding, text)
        if not enabled:
            return base_scores, base_scores

        roles = self._normalize_roles(
            role_ids,
            batch_size=base_scores.shape[0],
            classes=base_scores.shape[1],
            device=base_scores.device,
            default_role=default_role,
        )
        routed = base_scores
        category_mask = roles.eq(ROLE_CATEGORY)
        attribute_mask = roles.eq(ROLE_ATTRIBUTE)
        affordance_mask = roles.eq(ROLE_AFFORDANCE)
        active = bool(category_mask.any() or attribute_mask.any() or affordance_mask.any())
        if not active:
            return routed, base_scores

        shared_embedding = self.shared[scale_index](base_embedding)
        category_delta = self.category[scale_index].residual(shared_embedding)

        if category_mask.any():
            category_scores = contrastive_head(shared_embedding + category_delta, text)
            routed = torch.where(category_mask[:, :, None, None], category_scores, routed)

        # category_prior interpolates between the flat control (0) and the full
        # category-rooted hierarchy (1). v5 additionally places a gated portion
        # of the category residual directly on each child route. The default
        # stop-gradient blocks noisy child supervision at this boundary. Old
        # v4 checkpoints have no gate modules and retain their exact behavior.
        inherited_delta = self.category_prior * category_delta
        child_context = shared_embedding + inherited_delta
        if bool(getattr(self, "stop_gradient", True)):
            inherited_delta = inherited_delta.detach()
            child_context = child_context.detach()
        direct_inheritance = self.has_direct_inheritance()
        if attribute_mask.any():
            attribute_root = (
                self.attribute_inheritance[scale_index](inherited_delta) if direct_inheritance else 0.0
            )
            attribute_embedding = (
                shared_embedding + attribute_root + self.attribute[scale_index].residual(child_context)
            )
            attribute_scores = contrastive_head(attribute_embedding, text)
            routed = torch.where(attribute_mask[:, :, None, None], attribute_scores, routed)

        if affordance_mask.any():
            affordance_root = (
                self.affordance_inheritance[scale_index](inherited_delta) if direct_inheritance else 0.0
            )
            affordance_embedding = (
                shared_embedding + affordance_root + self.affordance[scale_index].residual(child_context)
            )
            affordance_scores = contrastive_head(affordance_embedding, text)
            routed = torch.where(affordance_mask[:, :, None, None], affordance_scores, routed)
        return routed, base_scores


def attach_granularity_router(
    head: nn.Module,
    hidden_dim: int = 192,
    category_prior: float = 1.0,
    inheritance_max: float = 0.5,
    stop_gradient: bool = True,
) -> GranularityRouter:
    """Attach or losslessly upgrade the category-rooted two-child router."""
    if not hasattr(head, "cv3") or not hasattr(head, "cv4"):
        raise TypeError("Expected a WorldDetect head with cv3/cv4 modules")
    embed_dims = [int(branch[-1].out_channels) for branch in head.cv3]
    if len(set(embed_dims)) != 1:
        raise ValueError(f"WorldDetect cv3 branches must share one embedding width, got {embed_dims}")
    if hasattr(head, "granularity_router"):
        router = head.granularity_router
        if not isinstance(router, GranularityRouter):
            raise TypeError("Existing granularity_router has an unexpected type")
        if router.num_scales != len(head.cv3):
            raise RuntimeError("Existing granularity router does not match WorldDetect scales")
        compatible = (
            getattr(router, "router_version", 0) in {4, 5}
            and hasattr(router, "shared")
            and hasattr(router, "attribute")
        )
        if not compatible:
            raise RuntimeError(
                "Loaded granularity router predates the bounded stable hierarchy; restart from the requested "
                "base checkpoint instead of resuming an older router checkpoint"
            )
        loaded_prior = float(getattr(router, "category_prior", 0.0))
        if abs(loaded_prior - float(category_prior)) > 1e-12:
            raise RuntimeError(
                f"Loaded granularity router category_prior={loaded_prior} differs from requested "
                f"category_prior={category_prior}"
            )
        router.category_prior = loaded_prior
        loaded_stop_gradient = bool(getattr(router, "stop_gradient", True))
        if loaded_stop_gradient != bool(stop_gradient):
            raise RuntimeError(
                f"Loaded granularity router stop_gradient={loaded_stop_gradient} differs from requested "
                f"stop_gradient={bool(stop_gradient)}"
            )
        router.stop_gradient = loaded_stop_gradient
        if not router.has_direct_inheritance():
            router.inheritance_max = float(inheritance_max)
            reference = next(router.parameters())
            router.attribute_inheritance = nn.ModuleList(
                CategoryInheritanceGate(embed_dims[0], inheritance_max) for _ in range(router.num_scales)
            ).to(device=reference.device, dtype=reference.dtype)
            router.affordance_inheritance = nn.ModuleList(
                CategoryInheritanceGate(embed_dims[0], inheritance_max) for _ in range(router.num_scales)
            ).to(device=reference.device, dtype=reference.dtype)
            router.router_version = 5
        else:
            loaded_max = float(getattr(router, "inheritance_max", inheritance_max))
            if abs(loaded_max - float(inheritance_max)) > 1e-12:
                raise RuntimeError(
                    f"Loaded granularity router inheritance_max={loaded_max} differs from requested "
                    f"inheritance_max={inheritance_max}"
                )
        return router
    conflicts = [
        name
        for name in ("structured_adapters", "scmr_adapters", "factor_adapter")
        if hasattr(head, name)
    ]
    if conflicts:
        raise RuntimeError(f"Granularity routing cannot be combined with legacy adapters: {conflicts}")

    head.granularity_router = GranularityRouter(
        embed_dims[0],
        int(hidden_dim),
        len(embed_dims),
        category_prior=float(category_prior),
        inheritance_max=float(inheritance_max),
        stop_gradient=bool(stop_gradient),
    )
    head.granularity_router_enabled = True
    head.granularity_default_role = "attribute"
    # Ephemeral per-batch routing metadata. It is intentionally not a buffer or
    # parameter and is refreshed by the specialized trainer before every batch.
    head.granularity_prompt_roles = None
    return head.granularity_router


def set_granularity_inference_role(model: nn.Module, role: str) -> bool:
    """Select one role for a complete inference vocabulary if a router exists."""
    if role not in ROLE_TO_ID:
        raise ValueError(f"Unsupported granularity role: {role!r}")
    head = model.model[-1] if hasattr(model, "model") else model
    if not hasattr(head, "granularity_router"):
        return False
    head.granularity_default_role = role
    head.granularity_prompt_roles = None
    head.granularity_router_enabled = True
    return True
