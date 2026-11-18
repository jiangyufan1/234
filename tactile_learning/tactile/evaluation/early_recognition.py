"""Causal early-recognition utilities."""
from __future__ import annotations

import torch


@torch.no_grad()
def earliest_confident_time(
    logits_by_time: torch.Tensor,
    times: torch.Tensor,
    confidence: float = 0.90,
    patience: int = 2,
):
    """Return actual decision-confirmation time, not first qualifying time.

    logits_by_time: [T, C]
    times: [T], monotonically increasing timestamps in seconds.
    """
    if logits_by_time.ndim != 2 or times.ndim != 1:
        raise ValueError("logits must be [T,C] and times must be [T]")

    if logits_by_time.shape[0] != times.numel():
        raise ValueError("logits and timestamps must have equal length")

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    if patience < 1:
        raise ValueError("patience must be >= 1")

    if times.numel() == 0:
        return None

    if not torch.isfinite(times).all():
        raise ValueError("timestamps must be finite")

    if times.numel() > 1 and not bool((times[1:] > times[:-1]).all()):
        raise ValueError("timestamps must be strictly increasing")

    probs = logits_by_time.softmax(-1)
    conf, pred = probs.max(-1)

    for i in range(times.numel() - patience + 1):
        stable_conf = (conf[i:i + patience] >= confidence).all()
        stable_class = (pred[i:i + patience] == pred[i]).all()

        if bool(stable_conf and stable_class):
            # Decision can only be confirmed at the final observation.
            return float(times[i + patience - 1])

    return None


@torch.no_grad()
def causal_prefix_logits(model, batch, query_times):
    """Evaluate causal prefixes in chronological order."""
    outputs = []

    for t in query_times:
        out = model(batch, query_time=float(t))
        outputs.append(out)

    return torch.stack(outputs, dim=0)
