from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .cte_encoders import CTEInputEncoder
from .cte_hgnn import CTEHGNN


class CTETactileClassifier(nn.Module):
    """End-to-end raw tactile RGB/Flow classifier."""

    def __init__(self, num_classes: int, config: Mapping):
        super().__init__()

        if num_classes < 2:
            raise ValueError("At least two classes are required")

        dim = int(config.get("feature_dim", 128))
        dropout = float(config.get("dropout", 0.2))

        self.encoder = CTEInputEncoder(
            feature_dim=dim,
            backbone=str(config.get("backbone", "resnet18")),
            pretrained=bool(config.get("pretrained", False)),
        )

        self.network = CTEHGNN(
            dim=dim,
            num_classes=num_classes,
            num_atoms=int(config.get("cte_num_atoms", 8)),
            dropout=dropout,
            sigma_seconds=float(
                config.get("cte_sigma_seconds", 0.03)
            ),
        )

    def _prepare(self, batch):
        rgb_mask = batch["rgb_mask"].bool()
        flow_mask = batch["flow_mask"].bool()

        node_mask = batch["node_mask"].bool()

        if node_mask.shape != rgb_mask.shape[:2]:
            raise ValueError("node_mask must be [B,3]")

        rgb_mask = rgb_mask & node_mask.unsqueeze(-1)
        flow_mask = flow_mask & node_mask.unsqueeze(-1)

        rgb_features, flow_features = self.encoder(batch)

        return (
            rgb_features,
            flow_features,
            batch["rgb_times"],
            batch["flow_start"],
            batch["flow_end"],
            rgb_mask,
            flow_mask,
        )

    def forward_with_aux(self, batch, query_time=None):
        inputs = self._prepare(batch)

        return self.network(
            *inputs,
            query_time=query_time,
            return_aux=True,
        )

    def forward(self, batch):
        logits, _ = self.forward_with_aux(batch)
        return logits
