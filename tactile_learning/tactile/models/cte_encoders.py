from __future__ import annotations

import torch
from torch import nn

from .encoders import (
    TinyFrameEncoder,
    make_rgb_encoder,
    encode_valid_frames,
)


class CTEInputEncoder(nn.Module):
    """Encode RGB points and optical-flow intervals without temporal pooling."""

    def __init__(
        self,
        feature_dim=128,
        backbone="resnet18",
        pretrained=False,
    ):
        super().__init__()

        if feature_dim < 4 or feature_dim % 4:
            raise ValueError("feature_dim must be >= 4 and divisible by 4")

        self.feature_dim = feature_dim

        self.rgb_encoder = make_rgb_encoder(
            backbone, feature_dim, pretrained
        )

        self.flow_encoder = TinyFrameEncoder(
            in_channels=2,
            feature_dim=feature_dim,
        )

        self.duration_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        self.flow_norm = nn.LayerNorm(feature_dim)

    def train(self, mode: bool = True):
        super().train(mode)

        # Prevent BatchNorm statistics from mixing future frames
        # within the same episode during training.
        for module in self.rgb_encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        return self

    def forward(self, batch):
        rgb = batch["rgb"]
        flow = batch["flow"]

        rgb_mask = batch["rgb_mask"].bool()
        flow_mask = batch["flow_mask"].bool()

        if rgb.ndim != 6 or rgb.shape[1] != 3 or rgb.shape[3] != 3:
            raise ValueError("rgb must be [B,3,Tr,3,H,W]")

        if flow.ndim != 6 or flow.shape[1] != 3 or flow.shape[3] != 2:
            raise ValueError("flow must be [B,3,Tf,2,H,W]")

        B, N, Tr = rgb.shape[:3]
        Bf, Nf, Tf = flow.shape[:3]

        if (B, N) != (Bf, Nf):
            raise ValueError("RGB and Flow batch/finger dimensions differ")

        if rgb_mask.shape != (B, N, Tr):
            raise ValueError("Invalid rgb_mask shape")

        if flow_mask.shape != (B, N, Tf):
            raise ValueError("Invalid flow_mask shape")

        # Encode each valid RGB point independently.
        rgb_features = encode_valid_frames(
            self.rgb_encoder,
            rgb,
            rgb_mask,
            self.feature_dim,
        )

        # Encode each completed optical-flow interval independently.
        flow_features = encode_valid_frames(
            self.flow_encoder,
            flow,
            flow_mask,
            self.feature_dim,
        )

        dt = batch["flow_dt"]

        if dt.shape != (B, Tf):
            raise ValueError("flow_dt must be [B,Tf]")

        if flow_mask.any():
            expanded_dt = dt[:, None, :].expand(B, N, Tf)

            valid_dt = expanded_dt[flow_mask]

            if (
                not torch.isfinite(valid_dt).all()
                or (valid_dt <= 0).any()
            ):
                raise ValueError("Valid Flow intervals need positive finite dt")

            duration = torch.log1p(
                expanded_dt.clamp_min(0) * 50.0
            ).unsqueeze(-1)

            duration_embedding = self.duration_encoder(duration)

            flow_features = self.flow_norm(
                flow_features + duration_embedding
            )

        rgb_features = torch.where(
            rgb_mask.unsqueeze(-1),
            rgb_features,
            torch.zeros_like(rgb_features),
        )

        flow_features = torch.where(
            flow_mask.unsqueeze(-1),
            flow_features,
            torch.zeros_like(flow_features),
        )

        return rgb_features, flow_features
