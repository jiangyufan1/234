"""Per-finger bidirectional dynamic pseudo-memory routing."""
from __future__ import annotations

import torch
from torch import nn


class DynamicPseudoRouter(nn.Module):
    """Cross-modal pseudo-memory routing without collapsing finger identity.

    Inputs:
        rgb, flow: [B,N,D]
        masks:     [B,N]
        edge_summary: optional [B,N,5]

    Outputs:
        routed RGB and Flow: [B,N,D]
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()

        if dim % heads:
            raise ValueError("dim must be divisible by heads")

        self.dim = dim

        self.rgb_memory = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.flow_memory = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.rgb_input = nn.Linear(dim, dim)
        self.flow_input = nn.Linear(dim, dim)

        self.edge_projection = nn.Sequential(
            nn.Linear(5, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )

        self.rgb_norm1 = nn.LayerNorm(dim)
        self.flow_norm1 = nn.LayerNorm(dim)
        self.rgb_norm2 = nn.LayerNorm(dim)
        self.flow_norm2 = nn.LayerNorm(dim)

        self.rgb_gate = nn.Linear(2 * dim, dim)
        self.flow_gate = nn.Linear(2 * dim, dim)

        self.rgb_dispatch = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.flow_dispatch = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        rgb,
        rgb_mask,
        flow,
        flow_mask,
        edge_summary=None,
    ):
        if rgb.ndim != 3 or flow.ndim != 3:
            raise ValueError("rgb and flow must have shape [B,N,D]")

        if rgb.shape != flow.shape or rgb.shape[-1] != self.dim:
            raise ValueError("RGB and Flow must have matching [B,N,D] shapes")

        B, N, D = rgb.shape

        if rgb_mask.shape != (B, N) or flow_mask.shape != (B, N):
            raise ValueError("Modality masks must have shape [B,N]")

        rm = rgb_mask.bool()
        fm = flow_mask.bool()
        both = (rm & fm).reshape(B * N, 1, 1)

        rgb_safe = torch.where(rm.unsqueeze(-1), rgb, torch.zeros_like(rgb))
        flow_safe = torch.where(fm.unsqueeze(-1), flow, torch.zeros_like(flow))

        if edge_summary is None:
            edge_summary = rgb.new_zeros(B, N, 5)
        elif edge_summary.shape != (B, N, 5):
            raise ValueError("edge_summary must have shape [B,N,5]")

        edge_embedding = self.edge_projection(edge_summary)

        # Each finger gets its own pair of modality memories.
        rgb_memory = (
            self.rgb_memory.expand(B, N, D)
            + self.rgb_input(rgb_safe)
            + edge_embedding
        ).reshape(B * N, 1, D)

        flow_memory = (
            self.flow_memory.expand(B, N, D)
            + self.flow_input(flow_safe)
            + edge_embedding
        ).reshape(B * N, 1, D)

        # Two directions of cross-modal memory exchange.
        rgb_cross, _ = self.cross_attention(
            rgb_memory, flow_memory, flow_memory, need_weights=False
        )
        flow_cross, _ = self.cross_attention(
            flow_memory, rgb_memory, rgb_memory, need_weights=False
        )

        rgb_memory = self.rgb_norm1(
            rgb_memory + self.dropout(rgb_cross) * both
        )
        flow_memory = self.flow_norm1(
            flow_memory + self.dropout(flow_cross) * both
        )

        rgb_memory = rgb_memory.reshape(B, N, D)
        flow_memory = flow_memory.reshape(B, N, D)

        rgb_gate = torch.sigmoid(
            self.rgb_gate(torch.cat([rgb_safe, rgb_memory], dim=-1))
        )
        flow_gate = torch.sigmoid(
            self.flow_gate(torch.cat([flow_safe, flow_memory], dim=-1))
        )

        routed_rgb = self.rgb_norm2(
            rgb_safe + rgb_gate * self.rgb_dispatch(rgb_memory)
        )
        routed_flow = self.flow_norm2(
            flow_safe + flow_gate * self.flow_dispatch(flow_memory)
        )

        routed_rgb = torch.where(
            rm.unsqueeze(-1), routed_rgb, torch.zeros_like(routed_rgb)
        )
        routed_flow = torch.where(
            fm.unsqueeze(-1), routed_flow, torch.zeros_like(routed_flow)
        )

        return routed_rgb, routed_flow, {
            "rgb_gate": rgb_gate,
            "flow_gate": flow_gate,
        }
