"""Timestamp-aware sampling utilities for FPS robustness experiments."""
from __future__ import annotations

import torch


def subsample_by_period(
    times: torch.Tensor,
    period: float,
    tolerance: float = 1e-3,
):
    """Greedy timestamp sampler with configurable timestamp jitter tolerance.

    The tolerance compensates for timestamp quantization/jitter. It does not
    rescale time or alter the timestamps used by downstream models.
    """
    if times.ndim != 1 or times.numel() == 0:
        raise ValueError("times must be a non-empty 1-D tensor")

    if period <= 0 or tolerance < 0 or tolerance >= period:
        raise ValueError("require period > tolerance >= 0")

    if not torch.isfinite(times).all():
        raise ValueError("timestamps must be finite")

    if times.numel() > 1 and not bool((times[1:] > times[:-1]).all()):
        raise ValueError("timestamps must be strictly increasing")

    keep = [0]
    last = times[0]

    for i in range(1, times.numel()):
        elapsed = float(times[i] - last)

        if elapsed >= period - tolerance:
            keep.append(i)
            last = times[i]

    return torch.tensor(keep, device=times.device, dtype=torch.long)


def paired_sampling_indices(
    times_a,
    times_b,
    period_a,
    period_b,
    tolerance=1e-3,
):
    return (
        subsample_by_period(times_a, period_a, tolerance),
        subsample_by_period(times_b, period_b, tolerance),
    )
