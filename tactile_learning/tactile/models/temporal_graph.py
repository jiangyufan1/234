"""Physics-constrained RGB point / optical-flow interval graph."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


NODE_TYPES = (
    "rgb_trend",
    "rgb_event",
    "flow_trend",
    "flow_event",
)


@dataclass
class TemporalGraph:
    node_x: dict[str, torch.Tensor]
    node_mask: dict[str, torch.Tensor]
    node_time: dict[str, torch.Tensor]
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_mask: torch.Tensor
    edge_type: torch.Tensor


class PointIntervalGraphBuilder(nn.Module):
    """Build four typed relations between RGB points and Flow intervals.

    RGB features:  [B,N,Tr,D]
    Flow features: [B,N,Tf,D]

    RGB timestamps: [B,Tr] or [B,N,Tr]
    Flow boundaries: [B,Tf] or [B,N,Tf]

    edge_index: [B,N,4*Tr*Tf,2]
      Last dimension stores (rgb_index, flow_index).

    edge_attr: [B,N,4*Tr*Tf,5]
    edge_mask: [B,N,4*Tr*Tf]
    edge_type: [B,N,4*Tr*Tf]

    Relation order:
      0 RGB trend -> Flow trend
      1 RGB event -> Flow event
      2 RGB trend -> Flow event
      3 RGB event -> Flow trend
    """

    PAIR_TYPES = (
        ("rgb_trend", "flow_trend", 0),
        ("rgb_event", "flow_event", 1),
        ("rgb_trend", "flow_event", 2),
        ("rgb_event", "flow_trend", 3),
    )

    def __init__(
        self,
        sigma_seconds: float = 0.03,
        max_edges_per_pair: int = 1,
        causal: bool = True,
    ):
        super().__init__()

        if sigma_seconds <= 0:
            raise ValueError("sigma_seconds must be positive")
        if max_edges_per_pair < 1:
            raise ValueError("max_edges_per_pair must be positive")

        self.sigma_seconds = float(sigma_seconds)
        self.max_edges_per_pair = max_edges_per_pair
        self.causal = bool(causal)

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
    def _expand_query(query_time, B, N, device, dtype):
        q = torch.as_tensor(query_time, device=device, dtype=dtype)

        if q.ndim == 0:
            q = q.expand(B, N)
        elif q.shape == (B,):
            q = q[:, None].expand(B, N)
        elif q.shape != (B, N):
            raise ValueError("query_time must be scalar, [B], or [B,N]")

        return q.unsqueeze(-1)

    @staticmethod
    def _validate_order(times, mask, name):
        if not torch.isfinite(times[mask]).all():
            raise ValueError(f"Valid {name} timestamps must be finite")

        B, N, _ = times.shape
        for b in range(B):
            for n in range(N):
                t = times[b, n, mask[b, n]]
                if t.numel() > 1 and not bool((t[1:] > t[:-1]).all()):
                    raise ValueError(f"Valid {name} timestamps must increase")

    def forward(
        self,
        node_x,
        node_mask,
        rgb_times,
        flow_start,
        flow_end,
        query_time=None,
    ):
        for name in NODE_TYPES:
            if name not in node_x or name not in node_mask:
                raise KeyError(f"Missing graph stream: {name}")

        rgb = node_x["rgb_trend"]
        flow = node_x["flow_trend"]

        if rgb.ndim != 4 or flow.ndim != 4:
            raise ValueError("Node features must have shape [B,N,T,D]")

        B, N, Tr, D = rgb.shape
        Bf, Nf, Tf, Df = flow.shape

        if (B, N, D) != (Bf, Nf, Df):
            raise ValueError("RGB and Flow batch, finger and feature dimensions must match")

        if Tr < 1 or Tf < 1:
            raise ValueError("RGB and Flow sequences must be non-empty")

        device, dtype = rgb.device, rgb.dtype

        for name in ("rgb_trend", "rgb_event"):
            if node_x[name].shape != (B, N, Tr, D):
                raise ValueError(f"Inconsistent RGB shape: {name}")
            if node_mask[name].shape != (B, N, Tr):
                raise ValueError(f"Inconsistent RGB mask: {name}")

        for name in ("flow_trend", "flow_event"):
            if node_x[name].shape != (B, N, Tf, D):
                raise ValueError(f"Inconsistent Flow shape: {name}")
            if node_mask[name].shape != (B, N, Tf):
                raise ValueError(f"Inconsistent Flow mask: {name}")

        masks = {k: node_mask[k].bool() for k in NODE_TYPES}

        rt = self._expand_time(
            rgb_times, B, N, Tr, "rgb_times", device, dtype
        )
        fs = self._expand_time(
            flow_start, B, N, Tf, "flow_start", device, dtype
        )
        fe = self._expand_time(
            flow_end, B, N, Tf, "flow_end", device, dtype
        )

        rgb_valid = masks["rgb_trend"] | masks["rgb_event"]
        flow_valid = masks["flow_trend"] | masks["flow_event"]

        self._validate_order(rt, rgb_valid, "RGB")
        self._validate_order(fs, flow_valid, "Flow start")

        # Validate each interval before validating sequence ordering.
        # This gives zero/negative-duration intervals a precise error.
        duration = fe - fs

        if not (duration[flow_valid] > 0).all():
            raise ValueError("Valid Flow intervals must have positive duration")

        self._validate_order(fe, flow_valid, "Flow end")

        if query_time is None:
            legal_flow = torch.ones(
                B, N, Tf, dtype=torch.bool, device=device
            )
        else:
            q = self._expand_query(
                query_time, B, N, device, dtype
            )
            legal_flow = fe <= q + 1e-9

        center = (fs + fe) / 2.0

        node_time = {
            "rgb_trend": rt,
            "rgb_event": rt,
            "flow_trend": center,
            "flow_event": center,
        }

        edge_indices = []
        edge_attributes = []
        edge_masks = []
        edge_types = []

        # Relation-major, then RGB-time-major, then Flow-time-major.
        for source_name, target_name, relation_id in self.PAIR_TYPES:
            for i in range(Tr):
                for j in range(Tf):
                    point = rt[:, :, i]
                    start = fs[:, :, j]
                    end = fe[:, :, j]
                    c = center[:, :, j]
                    dt = duration[:, :, j]

                    signed_offset = point - c
                    abs_offset = signed_offset.abs()

                    inside = (point >= start) & (point <= end)

                    distance = torch.where(
                        inside,
                        torch.zeros_like(point),
                        torch.minimum(
                            (point - start).abs(),
                            (point - end).abs(),
                        ),
                    )

                    compatibility = torch.exp(
                        -0.5 * (distance / self.sigma_seconds).square()
                    )

                    valid = (
                        masks[source_name][:, :, i]
                        & masks[target_name][:, :, j]
                        & legal_flow[:, :, j]
                    )

                    attr = torch.stack(
                        [
                            abs_offset,
                            signed_offset,
                            dt,
                            inside.to(dtype),
                            compatibility,
                        ],
                        dim=-1,
                    )

                    index = torch.tensor(
                        [i, j], device=device, dtype=torch.long
                    ).view(1, 1, 2).expand(B, N, 2)

                    edge_indices.append(index)
                    edge_attributes.append(attr)
                    edge_masks.append(valid)

                    edge_types.append(
                        torch.full(
                            (B, N),
                            relation_id,
                            device=device,
                            dtype=torch.long,
                        )
                    )

        return TemporalGraph(
            node_x=node_x,
            node_mask=masks,
            node_time=node_time,
            edge_index=torch.stack(edge_indices, dim=2),
            edge_attr=torch.stack(edge_attributes, dim=2),
            edge_mask=torch.stack(edge_masks, dim=2),
            edge_type=torch.stack(edge_types, dim=2),
        )
