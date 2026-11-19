"""Multi-token bidirectional RGB/Flow routing for CTE-HGNN."""
from __future__ import annotations
import torch
from torch import nn


class DynamicPseudoRouter(nn.Module):
    """Cross-modal attention over separate Trend and Event tokens.

    Inputs rgb/flow: [B,N,2,D], masks: [B,N,2].
    Outputs retain the original [B,N,D] CTE interface.
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.rgb_type = nn.Parameter(torch.randn(1, 1, 1, dim) * 0.02)
        self.flow_type = nn.Parameter(torch.randn(1, 1, 1, dim) * 0.02)
        self.edge_projection = nn.Sequential(
            nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim)
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
            nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * dim, dim)
        )
        self.flow_dispatch = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _safe_padding(tokens, mask):
        """Avoid all-masked attention rows for absent modalities."""
        padding = ~mask
        empty = ~mask.any(dim=1)
        if empty.any():
            tokens = tokens.clone()
            padding = padding.clone()
            tokens[empty, 0] = 0
            padding[empty, 0] = False
        return tokens, padding

    def forward(self, rgb, rgb_mask, flow, flow_mask, edge_summary=None):
        if rgb.ndim != 4 or flow.ndim != 4:
            raise ValueError("rgb and flow must be [B,N,2,D]")
        B, N, Kr, D = rgb.shape
        Kf = flow.shape[2]
        if D != self.dim or flow.shape[:2] != (B, N) or flow.shape[-1] != D:
            raise ValueError("RGB and Flow must share [B,N,D]")
        if rgb_mask.shape != (B, N, Kr) or flow_mask.shape != (B, N, Kf):
            raise ValueError("Token masks must match [B,N,K]")
        rm, fm = rgb_mask.bool(), flow_mask.bool()
        rs = torch.where(rm.unsqueeze(-1), rgb, torch.zeros_like(rgb))
        fs = torch.where(fm.unsqueeze(-1), flow, torch.zeros_like(flow))
        if edge_summary is None:
            edge_summary = rgb.new_zeros(B, N, 5)
        if edge_summary.shape != (B, N, 5):
            raise ValueError("edge_summary must be [B,N,5]")
        edge = self.edge_projection(edge_summary).unsqueeze(2)
        r = (rs + self.rgb_type + edge).reshape(B * N, Kr, D)
        f = (fs + self.flow_type + edge).reshape(B * N, Kf, D)
        rmask, fmask = rm.reshape(B * N, Kr), fm.reshape(B * N, Kf)
        r, rpad = self._safe_padding(r, rmask)
        f, fpad = self._safe_padding(f, fmask)

        r_cross, _ = self.cross_attention(
            r, f, f, key_padding_mask=fpad, need_weights=False
        )
        f_cross, _ = self.cross_attention(
            f, r, r, key_padding_mask=rpad, need_weights=False
        )
        r_cross = r_cross.reshape(B, N, Kr, D)
        f_cross = f_cross.reshape(B, N, Kf, D)
        both = (rm.any(-1) & fm.any(-1)).unsqueeze(-1).unsqueeze(-1)
        rmem = self.rgb_norm1(rs + self.dropout(r_cross) * both)
        fmem = self.flow_norm1(fs + self.dropout(f_cross) * both)

        rg = torch.sigmoid(self.rgb_gate(torch.cat((rs, rmem), dim=-1)))
        fg = torch.sigmoid(self.flow_gate(torch.cat((fs, fmem), dim=-1)))
        rout = self.rgb_norm2(rs + rg * self.rgb_dispatch(rmem))
        fout = self.flow_norm2(fs + fg * self.flow_dispatch(fmem))
        rout = torch.where(rm.unsqueeze(-1), rout, torch.zeros_like(rout))
        fout = torch.where(fm.unsqueeze(-1), fout, torch.zeros_like(fout))

        # Aggregate Trend/Event only after cross-modal interaction, preserving
        # the existing [B,N,D] interface expected by CTE-HGNN downstream.
        rden = rm.sum(dim=2, keepdim=True).clamp_min(1).to(rout.dtype)
        fden = fm.sum(dim=2, keepdim=True).clamp_min(1).to(fout.dtype)
        rout = rout.sum(dim=2) / rden
        fout = fout.sum(dim=2) / fden
        return rout, fout, {
            "rgb_gate": rg.mean(dim=2),
            "flow_gate": fg.mean(dim=2),
        }
