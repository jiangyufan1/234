"""Pose-conditioned alignment without forcing different local patches to match."""
from __future__ import annotations

import torch
from torch import nn


class PoseConditionedAlignment(nn.Module):
    """Shared residual adapter conditioned on a sensor's gripper-frame pose.

    Pose is [x,y,z,R00,...,R22], with position in metres, obtained from robot
    kinematics. It must not contain ground-truth object position/orientation.
    """

    def __init__(self, feature_dim: int, dropout: float = 0.2):
        super().__init__()
        self.feature_projection = nn.Sequential(nn.LayerNorm(feature_dim),
                                                nn.Linear(feature_dim, feature_dim), nn.GELU())
        self.pose_projection = nn.Sequential(nn.Linear(12, 64), nn.GELU(),
                                             nn.Linear(64, 2 * feature_dim))
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor, sensor_pose: torch.Tensor,
                node_mask: torch.Tensor) -> torch.Tensor:
        if sensor_pose.shape != (*features.shape[:2], 12):
            raise ValueError("sensor_pose must have shape [B,3,12]: position + row-major R.")
        mask = node_mask.unsqueeze(-1)
        pose = torch.where(mask, sensor_pose, torch.zeros_like(sensor_pose))
        if not torch.isfinite(pose).all():
            raise ValueError("Valid sensor poses must be finite.")
        pose = torch.cat((pose[..., :3] / 0.1, pose[..., 3:]), dim=-1)
        scale, bias = self.pose_projection(pose).chunk(2, dim=-1)
        residual = self.feature_projection(features) * (1 + 0.1 * torch.tanh(scale)) + bias
        aligned = self.output_norm(features + self.dropout(residual))
        return torch.where(mask, aligned, torch.zeros_like(aligned))
