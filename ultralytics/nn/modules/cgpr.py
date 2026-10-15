# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Coarse-guided partial-label ranking modules for YOLO-World."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class ZeroResidual(nn.Module):
    """Identity-initialized residual MLP applied only to selected foreground anchors."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.down = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.up(self.act(self.down(value)))


class FineRoIProjector(nn.Module):
    """Extract one inexpensive P3/P4 region embedding per GT/proposal.

    P3 uses a 5x5 grid and P4 a 3x3 grid. P5 is deliberately excluded because
    the median Bridge box is only about 62 pixels on its short side.
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_dim: int,
        roi_channels: int = 96,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("FineRoIProjector requires at least P3 and P4 features.")
        self.in_channels = [int(value) for value in in_channels[:2]]
        self.embed_dim = int(embed_dim)
        self.roi_channels = int(roi_channels)
        self.projections = nn.ModuleList(Conv(value, self.roi_channels, 1) for value in self.in_channels)
        flattened = self.roi_channels * (5 * 5 + 3 * 3)
        self.mlp = nn.Sequential(
            nn.LayerNorm(flattened),
            nn.Linear(flattened, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.embed_dim),
        )

    @staticmethod
    def _roi_align(feature: torch.Tensor, rois: torch.Tensor, size: int, image_shape: tuple[int, int]) -> torch.Tensor:
        try:
            from torchvision.ops import roi_align
        except ImportError as error:  # pragma: no cover - torchvision is an Ultralytics dependency
            raise ImportError("CGPR fine RoI training requires torchvision.ops.roi_align.") from error
        image_h, image_w = image_shape
        scale_h, scale_w = feature.shape[-2] / image_h, feature.shape[-1] / image_w
        if abs(scale_h - scale_w) > 1e-6:
            raise ValueError(f"Non-uniform feature scale is unsupported: h={scale_h}, w={scale_w}.")
        return roi_align(feature, rois, output_size=size, spatial_scale=scale_h, sampling_ratio=2, aligned=True)

    def forward(
        self, features: list[torch.Tensor], rois: torch.Tensor, image_shape: tuple[int, int]
    ) -> torch.Tensor:
        if rois.numel() == 0:
            return features[0].new_zeros((0, self.embed_dim))
        pooled = []
        for feature, projection, size in zip(features[:2], self.projections, (5, 3)):
            value = self._roi_align(projection(feature), rois, size, image_shape)
            pooled.append(value.flatten(1))
        return F.normalize(self.mlp(torch.cat(pooled, dim=1)).float(), dim=-1)


class CGPRHead(nn.Module):
    """Objectness, category and P3/P4 fine-region branches.

    The module is independent from raw phrase logits. Attaching it therefore
    leaves the proven YOLO-World checkpoint bit-identical at raw inference.
    """

    def __init__(
        self,
        in_channels: list[int],
        embed_dim: int,
        object_hidden: int = 128,
        category_hidden: int = 192,
        roi_channels: int = 96,
        fine_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.objectness = nn.ModuleList(
            nn.Sequential(Conv(channel, min(object_hidden, channel), 3), nn.Conv2d(min(object_hidden, channel), 1, 1))
            for channel in in_channels
        )
        # Start conservatively: about 1% foreground probability before warm-up.
        for branch in self.objectness:
            nn.init.zeros_(branch[-1].weight)
            nn.init.constant_(branch[-1].bias, -4.595)
        # One shared vector adapter is evaluated after foreground gathering. This avoids materializing three extra
        # Bx768xHxW maps and keeps the auxiliary category branch inexpensive.
        self.category_adapter = ZeroResidual(embed_dim, category_hidden)
        self.fine_projector = FineRoIProjector(
            in_channels=in_channels,
            embed_dim=embed_dim,
            roi_channels=roi_channels,
            hidden_dim=fine_hidden,
        )

    def forward_dense(
        self, source_features: list[torch.Tensor], aligned_features: list[torch.Tensor]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        batch = source_features[0].shape[0]
        objectness = torch.cat(
            [branch(feature).view(batch, 1, -1) for branch, feature in zip(self.objectness, source_features)], dim=2
        )
        return {"cgpr_objectness": objectness, "cgpr_category_base_feats": aligned_features}


def attach_cgpr_head(
    head: nn.Module,
    object_hidden: int = 128,
    category_hidden: int = 192,
    roi_channels: int = 96,
    fine_hidden: int = 512,
) -> CGPRHead:
    """Attach the isolated CGPR module to a WorldDetect head."""
    if hasattr(head, "cgpr_head"):
        return head.cgpr_head
    if not hasattr(head, "cv2") or not hasattr(head, "cv3"):
        raise TypeError("CGPR expects a WorldDetect head with cv2/cv3 branches.")
    if any(hasattr(head, name) for name in ("structured_adapters", "scmr_adapters", "factor_adapter")):
        raise RuntimeError("Do not combine CGPR with legacy structured/SCMR/factor experimental heads.")
    in_channels = [int(branch[0].conv.in_channels) for branch in head.cv2]
    embed_dims = [int(branch[-1].out_channels) for branch in head.cv3]
    if len(set(embed_dims)) != 1:
        raise ValueError(f"WorldDetect embedding widths differ across levels: {embed_dims}.")
    head.cgpr_head = CGPRHead(
        in_channels=in_channels,
        embed_dim=embed_dims[0],
        object_hidden=object_hidden,
        category_hidden=category_hidden,
        roi_channels=roi_channels,
        fine_hidden=fine_hidden,
    )
    head.cgpr_enabled = True
    return head.cgpr_head
