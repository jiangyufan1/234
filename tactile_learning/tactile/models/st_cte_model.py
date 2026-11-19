from __future__ import annotations
from collections.abc import Mapping
import torch
from torch import nn
from .cte_encoders import CTEInputEncoder
from .cte_hgnn import CTEHGNN
from .spatial_graph import TactileSpatialGraphBranch


class SpatiotemporalTokenFusion(nn.Module):
    """Jointly model CTE temporal context and three finger-specific spatial tokens."""
    def __init__(self, dim: int, dropout: float = 0.1, heads: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError("feature_dim must be divisible by fusion_heads")
        self.temporal_projection = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU()
        )
        self.position = nn.Parameter(torch.randn(1, 4, dim) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=2 * dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=1)
        hidden = max(8, dim // 2)
        self.pool_score = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, temporal, spatial_fingers, finger_mask):
        tokens = torch.cat(
            (self.temporal_projection(temporal).unsqueeze(1), spatial_fingers),
            dim=1
        ) + self.position
        padding = torch.cat((
            torch.zeros(finger_mask.shape[0], 1, dtype=torch.bool,
                        device=finger_mask.device),
            ~finger_mask.bool()
        ), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        scores = self.pool_score(encoded).squeeze(-1)
        scores = scores.masked_fill(padding, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        fused = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return self.norm(fused), weights


class STCTETactileClassifier(nn.Module):
    """Optimized ST-CTE: separate RGB spatial/temporal and optical-flow paths."""
    def __init__(self, num_classes: int, config: Mapping):
        super().__init__()
        if num_classes < 2:
            raise ValueError("At least two classes are required")
        dim = int(config.get("feature_dim", 128))
        spatial_dim = int(config.get("spatial_dim", dim))
        dropout = float(config.get("dropout", 0.2))
        if spatial_dim != dim:
            raise ValueError(
                "Optimized ST-CTE requires spatial_dim == feature_dim "
                "because the three finger tokens share the fusion dimension"
            )
        self.encoder = CTEInputEncoder(
            feature_dim=dim, backbone=str(config.get("backbone", "resnet18")),
            pretrained=bool(config.get("pretrained", False))
        )
        self.temporal = CTEHGNN(
            dim=dim, num_classes=num_classes,
            num_atoms=int(config.get("cte_num_atoms", 8)),
            dropout=dropout,
            sigma_seconds=float(config.get("cte_sigma_seconds", 0.03))
        )
        self.spatial = TactileSpatialGraphBranch(
            feature_dim=dim, spatial_dim=dim, dropout=dropout,
            graph_layers=int(config.get("spatial_graph_layers", 2)),
            adaptive_neighbors=int(config.get("spatial_adaptive_neighbors", 4))
        )
        self.fusion = SpatiotemporalTokenFusion(
            dim, dropout, int(config.get("fusion_heads", 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    @staticmethod
    def _masks(batch):
        required = (
            "rgb", "flow", "rgb_mask", "flow_mask", "flow_dt",
            "rgb_times", "flow_start", "flow_end", "node_mask"
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Missing ST-CTE-HGNN batch fields: {missing}")
        rgb_mask = batch["rgb_mask"].bool()
        flow_mask = batch["flow_mask"].bool()
        node_mask = batch["node_mask"].bool()
        if node_mask.shape != rgb_mask.shape[:2] or node_mask.shape[1] != 3:
            raise ValueError("node_mask must be [B,3]")
        return rgb_mask & node_mask.unsqueeze(-1), flow_mask & node_mask.unsqueeze(-1)

    def forward_with_aux(self, batch, query_time=None):
        rgb_mask, flow_mask = self._masks(batch)
        encoder_batch = {**batch, "rgb_mask": rgb_mask, "flow_mask": flow_mask}
        rgb_features, flow_features = self.encoder(encoder_batch)
        h_time, temporal_aux = self.temporal(
            rgb_features, flow_features, batch["rgb_times"],
            batch["flow_start"], batch["flow_end"], rgb_mask, flow_mask,
            query_time=query_time, return_aux=True, return_features=True
        )
        _, per_finger_space = self.spatial(
            batch["rgb"], rgb_mask, batch["rgb_times"], query_time
        )
        finger_mask = rgb_mask.any(dim=-1)
        h_fused, fusion_weights = self.fusion(
            h_time, per_finger_space, finger_mask
        )
        logits = self.classifier(h_fused)
        aux = {
            **temporal_aux,
            "fusion_token_weights": fusion_weights.detach(),
            "spatial_feature_norm": per_finger_space.detach().norm(dim=-1).mean(),
            "temporal_feature_norm": h_time.detach().norm(dim=-1).mean(),
            "fused_feature_norm": h_fused.detach().norm(dim=-1).mean()
        }
        return logits, aux

    def forward(self, batch):
        return self.forward_with_aux(batch)[0]

    def forward_online(self, accumulated_batch, query_time):
        logits, aux = self.forward_with_aux(
            accumulated_batch, query_time=query_time
        )
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "logits": logits, "probabilities": probabilities,
            "prediction": probabilities.argmax(dim=-1), "aux": aux
        }
