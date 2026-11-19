from __future__ import annotations
import torch
from torch import nn


class LocalSpatialGATLayer(nn.Module):
    """8-neighbour spatial graph augmented with feature-adaptive top-k edges."""
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1,
                 adaptive_neighbors: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError("spatial_dim must be divisible by attention heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.adaptive_neighbors = int(adaptive_neighbors)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * dim, dim)
        )

    @staticmethod
    def build_adjacency(height: int, width: int, device: torch.device):
        count = height * width
        adjacency = torch.eye(count, dtype=torch.bool, device=device)
        for row in range(height):
            for col in range(width):
                source = row * width + col
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = row + dr, col + dc
                        if (dr or dc) and 0 <= rr < height and 0 <= cc < width:
                            adjacency[source, rr * width + cc] = True
        return adjacency

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor):
        batch, points, _ = nodes.shape
        q = self.query(nodes).view(batch, points, self.heads, self.head_dim)
        k = self.key(nodes).view(batch, points, self.heads, self.head_dim)
        v = self.value(nodes).view(batch, points, self.heads, self.head_dim)
        scores = torch.einsum("mphd,mqhd->mpqh", q, k) / (self.head_dim ** 0.5)

        # Add semantic links selected from current node features; local grid edges
        # are always retained, so the learned graph cannot disconnect the grid.
        k_top = min(max(1, self.adaptive_neighbors), points)
        top_indices = scores.mean(dim=-1).topk(k_top, dim=-1).indices
        adaptive = torch.zeros(
            batch, points, points, dtype=torch.bool, device=nodes.device
        )
        adaptive.scatter_(2, top_indices, True)
        allowed = adjacency.unsqueeze(0) | adaptive
        scores = scores.masked_fill(
            ~allowed.unsqueeze(-1), torch.finfo(scores.dtype).min
        )
        attention = self.dropout(torch.softmax(scores, dim=2))
        messages = torch.einsum("mpqh,mqhd->mphd", attention, v)
        messages = messages.reshape(batch, points, self.heads * self.head_dim)
        x = self.norm1(nodes + self.dropout(self.output(messages)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class TactileSpatialGraphBranch(nn.Module):
    """RGB frames -> CNN maps -> adaptive spatial GAT -> temporal attention."""
    def __init__(self, feature_dim=128, spatial_dim=128, dropout=0.1,
                 graph_layers=2, adaptive_neighbors=4):
        super().__init__()
        if graph_layers < 1:
            raise ValueError("spatial_graph_layers must be >= 1")
        self.spatial_dim = spatial_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, spatial_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, spatial_dim), nn.GELU(),
        )
        self.graph_layers = nn.ModuleList([
            LocalSpatialGATLayer(
                spatial_dim, heads=4, dropout=dropout,
                adaptive_neighbors=adaptive_neighbors
            ) for _ in range(graph_layers)
        ])
        self.node_attention = nn.Linear(spatial_dim, 1)
        hidden = max(8, spatial_dim // 2)
        self.temporal_attention = nn.Sequential(
            nn.Linear(spatial_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.output = nn.Sequential(
            nn.Linear(3 * spatial_dim, feature_dim),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(feature_dim)
        )

    def forward(self, rgb, rgb_mask, rgb_times=None, query_time=None):
        if rgb.ndim != 6 or rgb.shape[1] != 3 or rgb.shape[3] != 3:
            raise ValueError("rgb must be [B,3,Tr,3,H,W]")
        batch, fingers, steps, channels, height, width = rgb.shape
        if rgb_mask.shape != (batch, fingers, steps):
            raise ValueError("rgb_mask must be [B,3,Tr]")
        valid_mask = rgb_mask.bool()

        if query_time is not None:
            if rgb_times is None:
                raise ValueError("rgb_times required for causal online spatial inference")
            times = torch.as_tensor(rgb_times, device=rgb.device, dtype=rgb.dtype)
            if times.shape == (batch, steps):
                times = times[:, None, :].expand(batch, fingers, steps)
            elif times.shape != (batch, fingers, steps):
                raise ValueError("rgb_times must be [B,Tr] or [B,3,Tr]")
            query = torch.as_tensor(query_time, device=rgb.device, dtype=rgb.dtype)
            if query.ndim == 0:
                query = query.expand(batch, fingers)
            elif query.shape == (batch,):
                query = query[:, None].expand(batch, fingers)
            elif query.shape != (batch, fingers):
                raise ValueError("query_time must be scalar, [B] or [B,3]")
            valid_mask = valid_mask & (times <= query.unsqueeze(-1) + 1e-9)

        flat = rgb.reshape(batch * fingers * steps, channels, height, width)
        valid_indices = valid_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError("No valid RGB frame at query_time")
        feature_map = self.cnn(flat.index_select(0, valid_indices))
        _, _, map_h, map_w = feature_map.shape
        nodes = feature_map.flatten(2).transpose(1, 2)
        adjacency = LocalSpatialGATLayer.build_adjacency(map_h, map_w, rgb.device)
        for layer in self.graph_layers:
            nodes = layer(nodes, adjacency)
        node_weights = torch.softmax(self.node_attention(nodes).squeeze(-1), dim=-1)
        frame_features = (nodes * node_weights.unsqueeze(-1)).sum(dim=1)
        all_features = rgb.new_zeros(
            batch * fingers * steps, self.spatial_dim
        ).index_copy(0, valid_indices, frame_features)
        all_features = all_features.reshape(batch, fingers, steps, self.spatial_dim)

        # Masked learned temporal pooling replaces uniform averaging.
        scores = self.temporal_attention(all_features).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * valid_mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        per_finger = (all_features * weights.unsqueeze(-1)).sum(dim=2)
        per_finger = torch.where(
            valid_mask.any(dim=2).unsqueeze(-1), per_finger,
            torch.zeros_like(per_finger)
        )
        episode_feature = self.output(per_finger.flatten(1))
        return episode_feature, per_finger
