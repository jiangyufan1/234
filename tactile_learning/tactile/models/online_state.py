"""Causal streaming state for RGB point observations and completed Flow intervals."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class StreamState:
    last_rgb_time: torch.Tensor | None = None
    last_flow_end: torch.Tensor | None = None
    rgb_history: torch.Tensor | None = None
    flow_history: torch.Tensor | None = None

class CausalStreamBuffer:
    """Per-stream causal buffer.

    A flow interval [t0,t1] becomes observable only at t1.  The buffer therefore
    never exposes an unfinished interval to the online model.
    """
    def __init__(self, max_rgb: int = 32, max_flow: int = 32):
        self.max_rgb, self.max_flow = max_rgb, max_flow
        self.reset()

    def reset(self):
        self.state = StreamState()

    def push_rgb(self, feature: torch.Tensor, timestamp: float):
        if feature.ndim != 1:
            raise ValueError("RGB feature must be [D]")
        t = float(timestamp)
        if self.state.last_rgb_time is not None and t <= float(self.state.last_rgb_time):
            raise ValueError("RGB timestamps must increase")
        item = torch.as_tensor(t, dtype=feature.dtype, device=feature.device)
        self.state.last_rgb_time = item
        self.state.rgb_history = self._append(self.state.rgb_history, feature, self.max_rgb)
        return self.state.rgb_history

    def push_flow(self, feature: torch.Tensor, start: float, end: float):
        if feature.ndim != 1:
            raise ValueError("Flow feature must be [D]")
        if end <= start:
            raise ValueError("Flow interval must satisfy end > start")
        if self.state.last_flow_end is not None and start < float(self.state.last_flow_end):
            raise ValueError("Flow intervals must be ordered/non-overlapping")
        # Completion time is end, so this is legal only after the interval ends.
        self.state.last_flow_end = torch.as_tensor(end, dtype=feature.dtype, device=feature.device)
        self.state.flow_history = self._append(self.state.flow_history, feature, self.max_flow)
        return self.state.flow_history

    @staticmethod
    def _append(history, x, limit):
        x = x.unsqueeze(0)
        if history is None:
            return x
        return torch.cat([history, x], dim=0)[-limit:]

    def available_flow_mask(self, starts, ends, query_time):
        starts, ends = torch.as_tensor(starts), torch.as_tensor(ends)
        return (ends <= float(query_time)) & (ends > starts)

    def snapshot(self):
        return self.state
