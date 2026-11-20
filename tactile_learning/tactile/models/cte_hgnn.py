"""CTE-HGNN with bidirectional edge-aware heterogeneous propagation."""
from __future__ import annotations

import math
import torch
from torch import nn

from .trend_event import CausalTrendEventDecomposer
from .temporal_graph import PointIntervalGraphBuilder
from .dynamic_routing import DynamicPseudoRouter


RELATIONS = (
    ("rgb_trend", "flow_trend"),
    ("rgb_event", "flow_event"),
    ("rgb_trend", "flow_event"),
    ("rgb_event", "flow_trend"),
)


def masked_softmax(scores, mask, dim):
    """FP32 masked softmax; all-invalid rows return exact zeros.

    The graph attention logits can be FP16 under autocast. Performing masking,
    softmax and normalization in FP32 avoids half-precision sentinel/epsilon
    underflow and keeps empty rows finite.
    """
    mask = mask.bool()
    scores_fp32 = scores.float()
    scores_fp32 = scores_fp32.masked_fill(
        ~mask, torch.finfo(torch.float32).min
    )
    weights = torch.softmax(scores_fp32, dim=dim)
    weights = weights * mask.to(torch.float32)
    denom = weights.sum(dim=dim, keepdim=True)
    weights = weights / denom.clamp_min(torch.finfo(torch.float32).eps)
    return weights.to(dtype=scores.dtype)


class BidirectionalRelationMessage(nn.Module):
    """One typed RGB-Flow relation with two propagation directions."""

    def __init__(self, dim, edge_dim=5, dropout=0.1):
        super().__init__()

        self.rgb_query = nn.Linear(dim, dim, bias=False)
        self.flow_key = nn.Linear(dim, dim, bias=False)
        self.flow_value = nn.Linear(dim, dim, bias=False)

        self.flow_query = nn.Linear(dim, dim, bias=False)
        self.rgb_key = nn.Linear(dim, dim, bias=False)
        self.rgb_value = nn.Linear(dim, dim, bias=False)

        self.edge_projection = nn.Sequential(
            nn.Linear(edge_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(dim)

    def forward(self, rgb, flow, edge_attr, edge_mask, relation_embedding):
        # rgb: [B,N,Tr,D], flow: [B,N,Tf,D]
        edge_embedding = (
            self.edge_projection(edge_attr)
            + relation_embedding.view(1, 1, 1, 1, -1)
        )

        q_rgb = self.rgb_query(rgb)
        k_flow = self.flow_key(flow)

        scores = torch.einsum(
            "bnid,bnjd->bnij", q_rgb, k_flow
        ) / self.scale

        # Keep attention logits and the compatibility logarithm in FP32.
        # Under autocast, exp(-distance^2 / (2 sigma^2)) in edge_attr can
        # underflow to exact zero in FP16. Clamping to 1e-8 *in FP16* is not
        # reliable because that floor is below the normal half range; log(0)
        # can then inject -inf and NaN gradients into the attention branch.
        scores = scores.float() + self.edge_bias(edge_attr).squeeze(-1).float()
        compatibility = edge_attr[..., 4].float().clamp_min(1e-8)
        scores = scores + compatibility.log()

        # RGB receives messages from Flow.
        weight_rgb = masked_softmax(scores, edge_mask, dim=-1)
        flow_values = self.flow_value(flow)

        rgb_messages = (
            weight_rgb.unsqueeze(-1)
            * (
                flow_values.unsqueeze(2)
                + edge_embedding
            )
        ).sum(dim=3)

        # Flow receives messages from RGB using the transposed relation.
        weight_flow = masked_softmax(
            scores.transpose(-1, -2),
            edge_mask.transpose(-1, -2),
            dim=-1,
        )

        rgb_values = self.rgb_value(rgb)

        flow_messages = (
            weight_flow.unsqueeze(-1)
            * (
                rgb_values.unsqueeze(3)
                + edge_embedding
            ).transpose(2, 3)
        ).sum(dim=3)

        rgb_has_neighbor = edge_mask.any(dim=-1)
        flow_has_neighbor = edge_mask.any(dim=-2)

        return (
            self.dropout(rgb_messages),
            self.dropout(flow_messages),
            rgb_has_neighbor,
            flow_has_neighbor,
        )


class TemporalHeteroMessagePassing(nn.Module):
    """Four independently parameterized, bidirectional relations."""

    def __init__(self, dim, edge_dim=5, dropout=0.1):
        super().__init__()

        self.relations = nn.ModuleList([
            BidirectionalRelationMessage(dim, edge_dim, dropout)
            for _ in range(4)
        ])

        self.relation_embedding = nn.Embedding(4, dim)

        self.updates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
            for name in (
                "rgb_trend", "rgb_event",
                "flow_trend", "flow_event",
            )
        })

        self.norms = nn.ModuleDict({
            name: nn.LayerNorm(dim)
            for name in (
                "rgb_trend", "rgb_event",
                "flow_trend", "flow_event",
            )
        })

    def forward(self, graph):
        B, N, E, _ = graph.edge_index.shape

        rgb = graph.node_x["rgb_trend"]
        flow = graph.node_x["flow_trend"]

        Tr = rgb.shape[2]
        Tf = flow.shape[2]

        expected_edges = 4 * Tr * Tf
        if E != expected_edges:
            raise ValueError(
                f"Expected {expected_edges} graph edges, received {E}"
            )

        messages = {
            name: torch.zeros_like(graph.node_x[name])
            for name in graph.node_x
        }
        counts = {
            name: torch.zeros_like(
                graph.node_mask[name], dtype=rgb.dtype
            )
            for name in graph.node_x
        }

        for relation_id, (rgb_name, flow_name) in enumerate(RELATIONS):
            start = relation_id * Tr * Tf
            stop = (relation_id + 1) * Tr * Tf

            edge_attr = graph.edge_attr[:, :, start:stop].reshape(
                B, N, Tr, Tf, -1
            )
            edge_mask = graph.edge_mask[:, :, start:stop].reshape(
                B, N, Tr, Tf
            )

            rgb_message, flow_message, rgb_valid, flow_valid = (
                self.relations[relation_id](
                    graph.node_x[rgb_name],
                    graph.node_x[flow_name],
                    edge_attr,
                    edge_mask,
                    self.relation_embedding.weight[relation_id],
                )
            )

            messages[rgb_name] = messages[rgb_name] + rgb_message
            messages[flow_name] = messages[flow_name] + flow_message

            counts[rgb_name] = counts[rgb_name] + rgb_valid.to(rgb.dtype)
            counts[flow_name] = counts[flow_name] + flow_valid.to(rgb.dtype)

        output = {}

        for name in graph.node_x:
            mask = graph.node_mask[name].bool()

            aggregated = messages[name] / counts[name].clamp_min(1.0).unsqueeze(-1)

            updated = self.norms[name](
                graph.node_x[name] + self.updates[name](aggregated)
            )

            output[name] = torch.where(
                mask.unsqueeze(-1), updated, torch.zeros_like(updated)
            )

        return output


class CTEHGNN(nn.Module):
    """Latent-sequence CTE-HGNN.

    Inputs are encoded sequences [B,N,T,D].
    Raw RGB/Flow image encoders are integrated in a separate stage.
    """

    def __init__(
        self,
        dim,
        num_classes,
        num_atoms=8,
        dropout=0.1,
        sigma_seconds=0.03,
    ):
        super().__init__()

        self.decomp = nn.ModuleDict({
            "rgb": CausalTrendEventDecomposer(dim, num_atoms),
            "flow": CausalTrendEventDecomposer(dim, num_atoms),
        })

        self.graph = PointIntervalGraphBuilder(
            sigma_seconds=sigma_seconds
        )

        self.mp = TemporalHeteroMessagePassing(
            dim, edge_dim=5, dropout=dropout
        )

        self.router = DynamicPseudoRouter(
            dim, heads=4, dropout=dropout
        )

        self.head = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    @staticmethod
    def _last_valid(x, mask):
        B, N, T, D = x.shape

        indices = torch.arange(
            T, device=x.device
        ).view(1, 1, T).expand(B, N, T)

        indices = indices.masked_fill(~mask.bool(), -1)
        last_index = indices.max(dim=-1).values
        valid = last_index >= 0

        safe_index = last_index.clamp_min(0)
        result = x.gather(
            2,
            safe_index[:, :, None, None].expand(B, N, 1, D),
        ).squeeze(2)

        return torch.where(
            valid.unsqueeze(-1), result, torch.zeros_like(result)
        )

    @staticmethod
    def _edge_summary(graph):
        weights = graph.edge_mask.to(graph.edge_attr.dtype)
        total = (graph.edge_attr * weights.unsqueeze(-1)).sum(dim=2)
        count = weights.sum(dim=2, keepdim=True)
        return total / count.clamp_min(1.0)

    @staticmethod
    def _expand_time(x, B, N, T, name, device, dtype):
        x = torch.as_tensor(x, device=device, dtype=dtype)
        if x.shape == (B, T):
            return x[:, None, :].expand(B, N, T)
        if x.shape == (B, N, T):
            return x
        raise ValueError(
            f"{name} must be [B,T] or [B,N,T], got {tuple(x.shape)}"
        )

    @staticmethod
    def _last_valid(x, mask):
        B, N, T, D = x.shape
        indices = torch.arange(T, device=x.device)
        indices = indices.view(1, 1, T).expand(B, N, T)
        indices = indices.masked_fill(~mask.bool(), -1)
        last_index = indices.max(dim=-1).values
        valid = last_index >= 0
        safe_index = last_index.clamp_min(0)
        result = x.gather(
            2, safe_index[:, :, None, None].expand(B, N, 1, D)
        ).squeeze(2)
        return torch.where(
            valid.unsqueeze(-1), result, torch.zeros_like(result)
        )

    @staticmethod
    def _edge_summary(graph):
        weights = graph.edge_mask.to(graph.edge_attr.dtype)
        total = (graph.edge_attr * weights.unsqueeze(-1)).sum(dim=2)
        count = weights.sum(dim=2, keepdim=True)
        return total / count.clamp_min(1.0)

    def forward(
        self,
        rgb,
        flow,
        rgb_times,
        flow_start,
        flow_end,
        rgb_mask,
        flow_mask,
        query_time=None,
        return_aux=True,
        return_features=False,
    ):
        if rgb.ndim != 4 or flow.ndim != 4:
            raise ValueError("rgb and flow must be [B,N,T,D]")

        B, N, Tr, D = rgb.shape
        Bf, Nf, Tf, Df = flow.shape

        if (B, N, D) != (Bf, Nf, Df):
            raise ValueError("RGB and Flow batch, finger and feature dimensions must match")

        if rgb_mask.shape != (B, N, Tr):
            raise ValueError("rgb_mask must be [B,N,Tr]")
        if flow_mask.shape != (B, N, Tf):
            raise ValueError("flow_mask must be [B,N,Tf]")

        rt = self._expand_time(
            rgb_times, B, N, Tr, "rgb_times", rgb.device, rgb.dtype
        )
        fs = self._expand_time(
            flow_start, B, N, Tf, "flow_start", flow.device, flow.dtype
        )
        fe = self._expand_time(
            flow_end, B, N, Tf, "flow_end", flow.device, flow.dtype
        )

        # Causal filtering MUST happen before trend/event decomposition.
        effective_rgb_mask = rgb_mask.bool()
        effective_flow_mask = flow_mask.bool()

        if query_time is not None:
            q = torch.as_tensor(
                query_time, device=rgb.device, dtype=rgb.dtype
            )

            if q.ndim == 0:
                q = q.expand(B, N)
            elif q.shape == (B,):
                q = q[:, None].expand(B, N)
            elif q.shape != (B, N):
                raise ValueError("query_time must be scalar, [B], or [B,N]")

            q = q.unsqueeze(-1)

            # RGB is a point observation: available at its timestamp.
            effective_rgb_mask = (
                effective_rgb_mask & (rt <= q + 1e-9)
            )

            # Flow is an interval observation: available only at its end.
            effective_flow_mask = (
                effective_flow_mask & (fe <= q + 1e-9)
            )

        if not effective_rgb_mask.any(dim=(1, 2)).all():
            raise ValueError("No RGB observations are available at query_time")

        ro = self.decomp["rgb"](
            rgb, rt, effective_rgb_mask
        )

        flow_center = (fs + fe) / 2.0

        fo = self.decomp["flow"](
            flow, flow_center, effective_flow_mask
        )

        nodes = {
            "rgb_trend": ro.trend,
            "rgb_event": ro.event,
            "flow_trend": fo.trend,
            "flow_event": fo.event,
        }

        masks = {
            "rgb_trend": effective_rgb_mask,
            "rgb_event": effective_rgb_mask,
            "flow_trend": effective_flow_mask,
            "flow_event": effective_flow_mask,
        }

        graph = self.graph(
            nodes,
            masks,
            rt,
            fs,
            fe,
            query_time=query_time,
        )

        propagated = self.mp(graph)

        rt_last = self._last_valid(
            propagated["rgb_trend"], effective_rgb_mask
        )
        re_last = self._last_valid(
            propagated["rgb_event"], effective_rgb_mask
        )
        ft_last = self._last_valid(
            propagated["flow_trend"], effective_flow_mask
        )
        fe_last = self._last_valid(
            propagated["flow_event"], effective_flow_mask
        )

        # Preserve Trend and Event as separate tokens during cross-modal routing.
        rgb_tokens = torch.stack((rt_last, re_last), dim=2)
        flow_tokens = torch.stack((ft_last, fe_last), dim=2)
        rgb_token_mask = effective_rgb_mask.any(dim=-1, keepdim=True).expand(-1, -1, 2)
        flow_token_mask = effective_flow_mask.any(dim=-1, keepdim=True).expand(-1, -1, 2)
        edge_summary = self._edge_summary(graph)

        routed_rgb, routed_flow, route_aux = self.router(
            rgb_tokens, rgb_token_mask,
            flow_tokens, flow_token_mask,
            edge_summary=edge_summary,
        )

        finger_mask = (
            effective_rgb_mask.any(dim=-1)
            | effective_flow_mask.any(dim=-1)
        )

        combined = torch.cat(
            [routed_rgb, routed_flow], dim=-1
        )

        combined = torch.where(
            finger_mask.unsqueeze(-1),
            combined,
            torch.zeros_like(combined),
        )

        pooled = combined.sum(dim=1) / finger_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1).to(combined.dtype)

        logits = self.head(pooled)
        aux = {
            **ro.losses,
            **{f"flow_{k}": v for k, v in fo.losses.items()},
            **route_aux,
        }
        if return_features:
            return (pooled, aux) if return_aux else pooled
        if not return_aux:
            return logits
        return logits, aux
