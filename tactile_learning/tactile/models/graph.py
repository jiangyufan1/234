"""Three-node edge-aware GATv2-style fusion in ordinary PyTorch."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DenseGraphAttention(nn.Module):
    """Row i receives messages from column j, including its self-loop.

    edge_attr[...,i,j,:] = [p_j-p_i (m), distance (m), normal_i dot normal_j].
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.2,
                 use_edge_geometry: bool = True):
        super().__init__()
        if dim % heads:
            raise ValueError("Node feature dimension must be divisible by attention heads.")
        self.heads, self.head_dim = heads, dim // heads
        self.receiver = nn.Linear(dim, dim, bias=False)
        self.sender = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.edge = nn.Linear(5, dim, bias=False) if use_edge_geometry else None
        self.attention = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.attention)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        batch, nodes, dim = x.shape
        active = node_mask.unsqueeze(-1)
        x = torch.where(active, x, torch.zeros_like(x))
        recv = self.receiver(x).reshape(batch, nodes, self.heads, self.head_dim)
        send = self.sender(x).reshape(batch, nodes, self.heads, self.head_dim)
        pairs = recv[:, :, None] + send[:, None, :]
        if self.edge is not None:
            if edge_attr is None or edge_attr.shape != (batch, nodes, nodes, 5):
                raise ValueError("Graph geometry requires edge_attr [B,3,3,5].")
            edge_mask = (node_mask[:, :, None] & node_mask[:, None, :]).unsqueeze(-1)
            safe_edges = torch.where(edge_mask, edge_attr, torch.zeros_like(edge_attr))
            if not torch.isfinite(safe_edges).all():
                raise ValueError("Geometry between valid sensors must be finite.")
            safe_edges = torch.cat((safe_edges[..., :4] / 0.1,
                                    safe_edges[..., 4:].clamp(-1, 1)), dim=-1)
            pairs = pairs + self.edge(safe_edges).reshape(batch, nodes, nodes,
                                                        self.heads, self.head_dim)
        scores = (F.leaky_relu(pairs, negative_slope=0.2) * self.attention).sum(-1)
        scores = scores.masked_fill(~node_mask[:, None, :, None], float("-inf"))
        weights = self.dropout(scores.softmax(dim=2))
        values = self.value(x).reshape(batch, nodes, self.heads, self.head_dim)
        messages = torch.einsum("bijh,bjhd->bihd", weights, values).reshape(batch, nodes, dim)
        x = self.norm1(x + self.dropout(self.output(messages)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return torch.where(active, x, torch.zeros_like(x))


class LocalModalityGraph(nn.Module):
    # Two-node RGB/Flow graph applied independently to each finger.
    # Node order is fixed as [RGB, Flow]; learned embeddings preserve modality identity.
    # Both updated nodes are retained for the global finger graph.

    def __init__(self, dim: int, dropout: float = 0.2, heads: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError("Local modality feature dimension must be divisible by heads.")
        self.dim = dim
        self.modality_embedding = nn.Parameter(torch.empty(1, 2, dim))
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            DenseGraphAttention(dim, heads, dropout, use_edge_geometry=False)
            for _ in range(2)
        ])

    def forward(self, features: torch.Tensor,
                modality_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (2, self.dim):
            raise ValueError(
                f"Local graph expects [B*3,2,{self.dim}], got {tuple(features.shape)}"
            )
        if modality_mask.shape != features.shape[:2]:
            raise ValueError("modality_mask must have shape [B*3,2].")
        modality_mask = modality_mask.bool()
        if (~modality_mask.any(dim=1)).any():
            raise ValueError("Each valid finger needs at least one valid modality.")

        x = features + self.modality_embedding
        x = torch.where(modality_mask.unsqueeze(-1), x, torch.zeros_like(x))
        for layer in self.layers:
            x = layer(x, modality_mask)
        x = torch.where(modality_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return x.flatten(start_dim=1)


class GraphFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2, use_edge_geometry: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([DenseGraphAttention(dim, 4, dropout, use_edge_geometry)
                                     for _ in range(2)])
        self.pool_score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, node_mask, edge_attr)
        scores = self.pool_score(x).squeeze(-1).masked_fill(~node_mask, float("-inf"))
        return (x * scores.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
