from __future__ import annotations

from collections.abc import Mapping
import torch
from torch import nn

from .cte_encoders import CTEInputEncoder
from .cte_hgnn import CTEHGNN
from .spatial_graph import TactileSpatialGraphBranch


class STCTETactileClassifier(nn.Module):
    # Joint spatial-temporal model. Existing CTE branch is retained.
    def __init__(self, num_classes: int, config: Mapping):
        super().__init__()
        if num_classes < 2:
            raise ValueError("At least two classes are required")
        dim = int(config.get("feature_dim", 128))
        spatial_dim = int(config.get("spatial_dim", dim))
        dropout = float(config.get("dropout", 0.2))

        self.encoder = CTEInputEncoder(
            feature_dim=dim,
            backbone=str(config.get("backbone", "resnet18")),
            pretrained=bool(config.get("pretrained", False)),
        )
        self.temporal = CTEHGNN(
            dim=dim,
            num_classes=num_classes,
            num_atoms=int(config.get("cte_num_atoms", 8)),
            dropout=dropout,
            sigma_seconds=float(config.get("cte_sigma_seconds", 0.03)),
        )
        self.spatial = TactileSpatialGraphBranch(
            feature_dim=dim,
            spatial_dim=spatial_dim,
            dropout=dropout,
            graph_layers=int(config.get("spatial_graph_layers", 2)),
        )

        # CTE returns [B,2D]; spatial branch returns [B,D].
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    @staticmethod
    def _masks(batch):
        required = (
            "rgb", "flow", "rgb_mask", "flow_mask", "flow_dt",
            "rgb_times", "flow_start", "flow_end", "node_mask",
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

        # return_features=True returns CTE's pooled pre-classifier embedding.
        h_time, aux = self.temporal(
            rgb_features,
            flow_features,
            batch["rgb_times"],
            batch["flow_start"],
            batch["flow_end"],
            rgb_mask,
            flow_mask,
            query_time=query_time,
            return_aux=True,
            return_features=True,
        )
        h_space, per_finger_space = self.spatial(
            batch["rgb"], rgb_mask, batch["rgb_times"], query_time
        )
        h_fused = self.fusion(torch.cat((h_time, h_space), dim=-1))
        logits = self.classifier(h_fused)

        aux = {
            **aux,
            "spatial_feature_norm": h_space.detach().norm(dim=-1).mean(),
            "temporal_feature_norm": h_time.detach().norm(dim=-1).mean(),
            "fused_feature_norm": h_fused.detach().norm(dim=-1).mean(),
        }
        return logits, aux

    def forward(self, batch):
        return self.forward_with_aux(batch)[0]

    def forward_online(self, accumulated_batch, query_time):
        # The caller supplies the accumulated prefix; query_time masks future observations.
        logits, aux = self.forward_with_aux(accumulated_batch, query_time=query_time)
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "prediction": probabilities.argmax(dim=-1),
            "aux": aux,
        }
