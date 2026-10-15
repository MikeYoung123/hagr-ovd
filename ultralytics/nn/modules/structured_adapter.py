# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Lightweight residual adapters for structured YOLO-World supervision."""

from __future__ import annotations

import torch
import torch.nn as nn


class StructuredResidualAdapter(nn.Module):
    """Add a zero-initialized bottleneck residual to a dense visual embedding map.

    The last projection is exactly zero at construction, so attaching this module
    to a trained YOLO-World head is initially an identity operation.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        embed_dim, hidden_dim = int(embed_dim), int(hidden_dim)
        if embed_dim < 1 or hidden_dim < 1:
            raise ValueError(f"embed_dim and hidden_dim must be positive, got {embed_dim}, {hidden_dim}.")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.down = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input plus a learned dense residual."""
        return x + self.up(self.act(self.down(x)))


def attach_structured_adapters(head: nn.Module, hidden_dim: int = 192) -> nn.ModuleList:
    """Attach one residual adapter to every YOLO-World classification scale."""
    if not hasattr(head, "cv3") or not hasattr(head, "cv4"):
        raise TypeError("Expected a WorldDetect head with cv3/cv4 modules.")
    if hasattr(head, "structured_adapters"):
        adapters = head.structured_adapters
        if len(adapters) != len(head.cv3):
            raise RuntimeError("Existing structured adapter count does not match WorldDetect scales.")
        return adapters

    embed_dims = [int(branch[-1].out_channels) for branch in head.cv3]
    if len(set(embed_dims)) != 1:
        raise ValueError(f"WorldDetect cv3 branches must share one embedding width, got {embed_dims}.")
    head.structured_adapters = nn.ModuleList(
        StructuredResidualAdapter(embed_dim=embed_dim, hidden_dim=hidden_dim) for embed_dim in embed_dims
    )
    head.structured_adapter_enabled = True
    return head.structured_adapters


def attach_scmr_adapters(head: nn.Module, hidden_dim: int = 192) -> nn.ModuleList:
    """Attach identity-initialized residual adapters used exclusively by SCMR.

    SCMR has its own module name so its checkpoints cannot be confused with the
    older structured-atom adapter experiment. Both use the same safe bottleneck
    implementation, but their supervision and inference semantics are different.
    """
    if not hasattr(head, "cv3") or not hasattr(head, "cv4"):
        raise TypeError("Expected a WorldDetect head with cv3/cv4 modules.")
    if hasattr(head, "scmr_adapters"):
        adapters = head.scmr_adapters
        if len(adapters) != len(head.cv3):
            raise RuntimeError("Existing SCMR adapter count does not match WorldDetect scales.")
        return adapters
    if hasattr(head, "structured_adapters"):
        raise RuntimeError("SCMR and structured-atom adapters cannot be attached to the same head.")

    embed_dims = [int(branch[-1].out_channels) for branch in head.cv3]
    if len(set(embed_dims)) != 1:
        raise ValueError(f"WorldDetect cv3 branches must share one embedding width, got {embed_dims}.")
    head.scmr_adapters = nn.ModuleList(
        StructuredResidualAdapter(embed_dim=embed_dim, hidden_dim=hidden_dim) for embed_dim in embed_dims
    )
    head.scmr_adapter_enabled = True
    return head.scmr_adapters


def attach_factor_adapter(head: nn.Module, hidden_dim: int = 128) -> StructuredResidualAdapter:
    """Attach one shared, identity-initialized factor projection without changing the raw detection branch.

    Unlike the older ``structured_adapters`` and ``scmr_adapters``, this module is not inserted into the raw
    YOLO-World classification path. WorldDetect feeds it a detached copy of the dense visual embedding and exposes
    the result as an independent factor feature. Consequently, training the factor head cannot change raw boxes or
    raw phrase scores.
    """
    if not hasattr(head, "cv3") or not hasattr(head, "cv4"):
        raise TypeError("Expected a WorldDetect head with cv3/cv4 modules.")
    if hasattr(head, "factor_adapter"):
        adapter = head.factor_adapter
        if not isinstance(adapter, StructuredResidualAdapter):
            raise TypeError("Existing factor_adapter has an unexpected type.")
        return adapter
    if hasattr(head, "structured_adapters") or hasattr(head, "scmr_adapters"):
        raise RuntimeError("The independent factor head cannot be combined with legacy structured/SCMR adapters.")

    embed_dims = [int(branch[-1].out_channels) for branch in head.cv3]
    if len(set(embed_dims)) != 1:
        raise ValueError(f"WorldDetect cv3 branches must share one embedding width, got {embed_dims}.")
    head.factor_adapter = StructuredResidualAdapter(embed_dim=embed_dims[0], hidden_dim=hidden_dim)
    head.factor_adapter_enabled = True
    head.factor_score_alpha = 0.0
    return head.factor_adapter
