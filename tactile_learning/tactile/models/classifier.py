"""Configurable tactile classifier and controlled baseline variants."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .alignment import PoseConditionedAlignment
from .encoders import DynamicEncoder, StaticEncoder
from .graph import GraphFusion, LocalModalityGraph


class TactileClassifier(nn.Module):
    def __init__(self, num_classes: int, config: Mapping):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Object classification requires at least two classes.")
        feature_dim = int(config.get("feature_dim", 128))
        self.feature_dim = feature_dim
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        self.fusion = str(config.get("fusion", "graph"))
        if self.fusion not in {"graph", "concat", "mean", "single"}:
            raise ValueError(f"Unknown fusion method: {self.fusion!r}")
        self.single_finger = int(config.get("single_finger", 0))
        if self.single_finger not in (0, 1, 2):
            raise ValueError("single_finger must be 0, 1 or 2.")
        dropout = float(config.get("dropout", 0.2))
        node_dim = 2 * feature_dim
        self.static = StaticEncoder(str(config.get("backbone", "resnet18")), feature_dim,
                                    bool(config.get("pretrained", False)))
        self.dynamic = DynamicEncoder(feature_dim) if config.get("use_flow", True) else None
        self.local_graph = (
            LocalModalityGraph(feature_dim, dropout)
            if config.get("use_local_graph", False) else None
        )
        if self.local_graph is not None and self.dynamic is None:
            raise ValueError("use_local_graph requires use_flow=True.")
        if self.local_graph is not None and self.fusion != "graph":
            raise ValueError("use_local_graph requires fusion='graph'.")
        self.static_projection = nn.Linear(feature_dim, node_dim) if self.dynamic is None else None
        self.alignment = (PoseConditionedAlignment(node_dim, dropout)
                          if config.get("use_alignment", True) else None)
        self.graph = (GraphFusion(node_dim, dropout, bool(config.get("use_edge_geometry", True)))
                      if self.fusion == "graph" else None)
        self.concat_projection = (nn.Sequential(nn.Linear(3 * node_dim, node_dim), nn.GELU(),
                                                nn.Dropout(dropout))
                                  if self.fusion == "concat" else None)
        self.head = nn.Sequential(nn.Linear(node_dim, feature_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return an object embedding; labels and object identity are never read."""
        rgb = batch["rgb"]
        if rgb.ndim != 6 or rgb.shape[1] != 3 or rgb.shape[3] != 3:
            raise ValueError("rgb must have shape [B,3,S,3,H,W].")
        rgb_mask = batch["rgb_mask"].bool()
        if rgb_mask.shape != rgb.shape[:3]:
            raise ValueError("rgb_mask must have shape [B,3,S].")
        node_mask = batch["node_mask"].bool()
        if node_mask.shape != rgb.shape[:2]:
            raise ValueError("node_mask must have shape [B,3].")
        node_mask = node_mask & rgb_mask.any(dim=2)
        if self.fusion == "single":
            selected = torch.zeros_like(node_mask)
            selected[:, self.single_finger] = True
            node_mask = node_mask & selected
        if (~node_mask.any(dim=1)).any():
            raise ValueError("Every episode needs a valid contact with a valid RGB frame; "
                             "the single baseline also requires its selected finger.")
        rgb_mask = rgb_mask & node_mask.unsqueeze(-1)
        static = self.static(rgb, rgb_mask)
        if self.dynamic is not None:
            flow = batch["flow"]
            if flow.ndim != 6 or flow.shape[:2] != rgb.shape[:2] or flow.shape[3] != 2:
                raise ValueError("flow must have shape [B,3,L,2,H,W].")
            flow_mask = batch["flow_mask"].bool()
            if flow_mask.shape != flow.shape[:3]:
                raise ValueError("flow_mask must have shape [B,3,L].")
            valid_flow_mask = flow_mask & node_mask.unsqueeze(-1)
            dynamic = self.dynamic(flow, valid_flow_mask, batch["flow_dt"])

            if self.local_graph is not None:
                batch_size = rgb.shape[0]
                # [B,3,2,F]: modality nodes are ordered RGB then Flow.
                modality_features = torch.stack((static, dynamic), dim=2)
                modality_mask = torch.stack(
                    (rgb_mask.any(dim=2), valid_flow_mask.any(dim=2)), dim=2
                )
                local_nodes = self.local_graph(
                    modality_features.reshape(batch_size * 3, 2, self.feature_dim),
                    modality_mask.reshape(batch_size * 3, 2),
                )
                # Keep updated RGB and Flow node features separately.
                nodes = local_nodes.reshape(batch_size, 3, 2 * self.feature_dim)
            else:
                # Preserve the original Dynamic Concat baseline.
                nodes = torch.cat((static, dynamic), dim=-1)
        else:
            nodes = self.static_projection(static)
        nodes = torch.where(node_mask.unsqueeze(-1), nodes, torch.zeros_like(nodes))
        if self.alignment is not None:
            nodes = self.alignment(nodes, batch["sensor_pose"], node_mask)
        if self.graph is not None:
            return self.graph(nodes, node_mask, batch.get("edge_attr"))
        if self.concat_projection is not None:
            return self.concat_projection(nodes.flatten(1))
        if self.fusion == "single":
            return nodes[:, self.single_finger]
        return nodes.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).to(nodes.dtype)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return unnormalized [B,K] logits for CrossEntropyLoss."""
        return self.head(self.encode(batch))


def create_model(num_classes: int, config: Mapping | None = None) -> nn.Module:
    """Create the selected model while preserving the original baseline."""
    config = {} if config is None else config
    config = config.get("model", config)

    architecture = config.get("architecture", "baseline")

    if architecture == "cte_hgnn":
        from .cte_model import CTETactileClassifier
        return CTETactileClassifier(num_classes, config)

    if architecture in ("st_cte_hgnn", "st_cte_hgnn_optimized"):
        from .st_cte_model import STCTETactileClassifier
        return STCTETactileClassifier(num_classes, config)

    if architecture == "st_cte_hgnn_single":
        from .st_cte_single_model import STCTESingleSensorClassifier
        return STCTESingleSensorClassifier(num_classes, config)

    if architecture != "baseline":
        raise ValueError(f"Unknown model architecture: {architecture!r}")

    return TactileClassifier(num_classes, config)
