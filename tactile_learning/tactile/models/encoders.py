"""Shared frame encoders and mask-aware temporal aggregation.

Invalid frames never enter a CNN, including its batch-normalization statistics.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence


class TinyFrameEncoder(nn.Module):
    """Small GroupNorm CNN for debugging and the optical-flow branch."""

    def __init__(self, in_channels: int, feature_dim: int):
        super().__init__()
        layers = []
        channels = in_channels
        for width in (32, 64, 96):
            layers.extend([
                nn.Conv2d(channels, width, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, width), nn.SiLU(inplace=True),
            ])
            channels = width
        self.network = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(channels, feature_dim))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.network(frames)


def make_rgb_encoder(backbone: str, feature_dim: int, pretrained: bool) -> nn.Module:
    if backbone == "tiny":
        if pretrained:
            raise ValueError("The tiny backbone has no pretrained weights.")
        return TinyFrameEncoder(3, feature_dim)
    if backbone == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, feature_dim)
        return model
    raise ValueError(f"Unknown RGB backbone: {backbone!r}")


def encode_valid_frames(encoder: nn.Module, frames: torch.Tensor,
                        mask: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Encode [B,N,T,C,H,W] without evaluating invalid/padded images."""
    if frames.ndim != 6 or tuple(mask.shape) != tuple(frames.shape[:3]):
        raise ValueError("frames must be [B,N,T,C,H,W] with mask [B,N,T].")
    flat = frames.reshape(-1, *frames.shape[-3:])
    valid = mask.reshape(-1).nonzero(as_tuple=False).flatten()
    if valid.numel() == 0:
        return frames.new_zeros(*frames.shape[:3], feature_dim)
    features = encoder(flat.index_select(0, valid))
    output = features.new_zeros(flat.shape[0], feature_dim)
    output = output.index_copy(0, valid, features)
    return output.reshape(*frames.shape[:3], feature_dim)


class StaticEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", feature_dim: int = 128,
                 pretrained: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.frame_encoder = make_rgb_encoder(backbone, feature_dim, pretrained)

    def forward(self, rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = encode_valid_frames(self.frame_encoder, rgb, mask, self.feature_dim)
        weights = mask.unsqueeze(-1).to(features.dtype)
        return (features * weights).sum(2) / weights.sum(2).clamp_min(1)


class DynamicEncoder(nn.Module):
    """Flow CNN + GRU: padded timesteps never update the recurrent state.

    The dataset supplies displacement normalized by image width/height. Its
    duration is a separate input; displacement is not silently called velocity.
    Empty flow histories map to zero.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.frame_encoder = TinyFrameEncoder(2, feature_dim)
        self.temporal_encoder = nn.GRU(feature_dim + 1, feature_dim, batch_first=True)

    def forward(self, flow: torch.Tensor, mask: torch.Tensor,
                dt: torch.Tensor) -> torch.Tensor:
        batch, nodes, steps = flow.shape[:3]
        if dt.shape != (batch, steps):
            raise ValueError("flow_dt must have shape [B,L].")
        features = encode_valid_frames(self.frame_encoder, flow, mask, self.feature_dim)
        flat_features = features.reshape(batch * nodes, steps, self.feature_dim)
        flat_mask = mask.reshape(batch * nodes, steps)
        expanded_dt = dt[:, None, :].expand(batch, nodes, steps).reshape(batch * nodes, steps)
        valid_dt = expanded_dt[flat_mask]
        if valid_dt.numel() and (not torch.isfinite(valid_dt).all() or (valid_dt <= 0).any()):
            raise ValueError("Valid optical-flow intervals must have finite positive flow_dt.")
        sequences = []
        indices = []
        for index in range(batch * nodes):
            selected = flat_mask[index]
            if selected.any():
                # A 20 ms interval becomes log(2); this scale is fixed beforehand.
                duration = torch.log1p(expanded_dt[index, selected] * 50).unsqueeze(-1)
                sequences.append(torch.cat((flat_features[index, selected],
                                            duration.to(features.dtype)), dim=-1))
                indices.append(index)
        output = features.new_zeros(batch * nodes, self.feature_dim)
        if sequences:
            packed = pack_sequence(sequences, enforce_sorted=False)
            _, final_hidden = self.temporal_encoder(packed)
            output = output.index_copy(0, torch.tensor(indices, device=flow.device), final_hidden[-1])
        return output.reshape(batch, nodes, self.feature_dim)
