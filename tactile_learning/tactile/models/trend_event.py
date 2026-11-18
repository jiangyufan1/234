"""Causal elapsed-time trend/event decomposition for tactile latent features.

The module is inspired by continuous-time trend/event representations, but is
implemented as a causal state-space model suitable for short tactile episodes.
It does not assign a physical label (slip, impact, etc.) to an event residual.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TrendEventOutput:
    trend: torch.Tensor
    residual: torch.Tensor
    event: torch.Tensor
    reconstruction: torch.Tensor
    gates: torch.Tensor
    atom_responses: torch.Tensor
    losses: dict[str, torch.Tensor]


class CausalTrendEventDecomposer(nn.Module):
    """Causal trend/event decomposition with real elapsed time.

    Args:
        feature_dim: latent feature dimension.
        num_atoms: number of transient Gabor atoms.
        tau_min_s/tau_max_s: physically plausible trend time-constant range.
        event_sigma_min_s/event_sigma_max_s: transient time-scale range.
        event_freq_min_hz/event_freq_max_hz: frequency range before the
            sequence-specific Nyquist cap is applied.

    Input:
        x: [B, N, T, D] latent features.
        times: [B, T] or [B, N, T], seconds.
        mask: [B, N, T], valid observations.

    RGB timestamps are point observations. For optical-flow features computed
    from frames (t_i, t_{i+1}), pass the flow feature at the interval midpoint
    (t_i+t_{i+1})/2 when aligning it to this module.
    """

    def __init__(
        self,
        feature_dim: int,
        num_atoms: int = 8,
        tau_min_s: float = 0.02,
        tau_max_s: float = 0.50,
        event_sigma_min_s: float = 0.01,
        event_sigma_max_s: float = 0.20,
        event_freq_min_hz: float = 1.0,
        event_freq_max_hz: float = 8.0,
        gate_target_sparsity: float = 0.20,
        gate_temperature: float = 0.15,
        smoothness_weight: float = 1e-3,
        sparsity_weight: float = 1e-3,
        reconstruction_weight: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or num_atoms < 1:
            raise ValueError("feature_dim and num_atoms must be positive")
        if not (0 < tau_min_s < tau_max_s):
            raise ValueError("invalid trend time-constant range")
        if not (0 < event_sigma_min_s < event_sigma_max_s):
            raise ValueError("invalid event sigma range")
        if not (0 < event_freq_min_hz < event_freq_max_hz):
            raise ValueError("invalid event frequency range")
        if not (0 < gate_target_sparsity < 1):
            raise ValueError("gate_target_sparsity must be in (0,1)")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")

        self.feature_dim = feature_dim
        self.num_atoms = num_atoms
        self.tau_min_s, self.tau_max_s = tau_min_s, tau_max_s
        self.event_sigma_min_s, self.event_sigma_max_s = event_sigma_min_s, event_sigma_max_s
        self.event_freq_min_hz, self.event_freq_max_hz = event_freq_min_hz, event_freq_max_hz
        self.target_sparsity = gate_target_sparsity
        self.gate_temperature = gate_temperature
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.reconstruction_weight = reconstruction_weight
        self.eps = eps

        # Unconstrained parameters mapped into explicit physical time ranges.
        self.tau_logits = nn.Parameter(torch.zeros(feature_dim))
        self.sigma_logits = nn.Parameter(torch.linspace(-2.0, 2.0, num_atoms))
        self.freq_logits = nn.Parameter(torch.linspace(-2.0, 2.0, num_atoms))
        self.phase = nn.Parameter(torch.zeros(num_atoms))
        self.gate_threshold = nn.Parameter(torch.full((num_atoms,), -2.0))
        self.gain = nn.Parameter(torch.ones(num_atoms))

        self.event_decoder = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.eye_(self.event_decoder.weight)

    @staticmethod
    def _bounded(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(raw)

    def _validate(self, x, times, mask):
        if x.ndim != 4 or x.shape[-1] != self.feature_dim:
            raise ValueError(f"x must be [B,N,T,{self.feature_dim}]")
        b, n, t, _ = x.shape
        if mask.shape != (b, n, t):
            raise ValueError("mask must be [B,N,T]")
        if times.shape == (b, t):
            times = times[:, None, :].expand(b, n, t)
        elif times.shape != (b, n, t):
            raise ValueError("times must be [B,T] or [B,N,T]")
        mask = mask.bool()
        if mask.any():
            if not torch.isfinite(x[mask]).all() or not torch.isfinite(times[mask]).all():
                raise ValueError("valid features and timestamps must be finite")
        for bi in range(b):
            for ni in range(n):
                ts = times[bi, ni, mask[bi, ni]]
                if ts.numel() > 1 and not bool((ts[1:] > ts[:-1]).all()):
                    raise ValueError("valid timestamps must be strictly increasing")
        return times.to(x.dtype), mask

    def _trend(self, x, times, mask):
        b, n, t, d = x.shape
        tau = self._bounded(self.tau_logits, self.tau_min_s, self.tau_max_s).to(x.dtype)
        state = x.new_zeros(b, n, d)
        last_time = x.new_zeros(b, n)
        seen = torch.zeros(b, n, dtype=torch.bool, device=x.device)
        out = []
        for i in range(t):
            valid = mask[:, :, i]
            dt = (times[:, :, i] - last_time).clamp_min(0)
            alpha = 1.0 - torch.exp(-dt.unsqueeze(-1) / tau.view(1, 1, d))
            alpha = torch.where(seen.unsqueeze(-1), alpha, torch.ones_like(alpha))
            candidate = state + alpha * (x[:, :, i] - state)
            state = torch.where(valid.unsqueeze(-1), candidate, state)
            last_time = torch.where(valid, times[:, :, i], last_time)
            seen = seen | valid
            out.append(torch.where(valid.unsqueeze(-1), state, torch.zeros_like(state)))
        return torch.stack(out, dim=2)

    def _gabor_bank(self, residual, times, mask):
        b, n, t, d = residual.shape
        k = self.num_atoms

        sigma = self._bounded(
            self.sigma_logits,
            self.event_sigma_min_s,
            self.event_sigma_max_s,
        ).to(residual.dtype)

        base_freq = self._bounded(
            self.freq_logits,
            self.event_freq_min_hz,
            self.event_freq_max_hz,
        ).to(residual.dtype)

        phase = self.phase.to(residual.dtype)

        out = residual.new_zeros(b, n, t, k, d)

        for bi in range(b):
            for ni in range(n):
                ids = mask[bi, ni].nonzero(
                    as_tuple=False
                ).flatten().tolist()

                for qi in range(1, len(ids)):
                    now = times[bi, ni, ids[qi]]

                    # Only intervals completed by the current observation
                    # may determine the effective sampling-rate limit.
                    history_ids = ids[:qi + 1]

                    history_dt = (
                        times[bi, ni, history_ids[1:]]
                        - times[bi, ni, history_ids[:-1]]
                    )

                    dt_ref = history_dt.median().clamp_min(self.eps)
                    max_freq = (0.45 / dt_ref).to(residual.dtype)

                    freq = torch.minimum(base_freq, max_freq)
                    omega = 2.0 * torch.pi * freq

                    acc = residual.new_zeros(k, d)
                    den = residual.new_zeros(k, 1)

                    for p_idx in range(1, qi + 1):
                        a = ids[p_idx - 1]
                        z = ids[p_idx]

                        dt = times[bi, ni, z] - times[bi, ni, a]
                        midpoint = 0.5 * (
                            times[bi, ni, z] + times[bi, ni, a]
                        )
                        lag = (now - midpoint).clamp_min(0)

                        kernel = torch.exp(
                            -0.5 * (lag / sigma).square()
                        )
                        kernel = kernel * torch.cos(
                            omega * lag + phase
                        )

                        weight = kernel * dt

                        interval_residual = 0.5 * (
                            residual[bi, ni, a]
                            + residual[bi, ni, z]
                        )

                        acc = acc + (
                            weight.unsqueeze(-1)
                            * interval_residual.unsqueeze(0)
                        )

                        den = den + weight.abs().unsqueeze(-1)

                    out[bi, ni, ids[qi]] = (
                        acc / den.clamp_min(self.eps)
                    )

        return out

    def forward(self, x, times, mask) -> TrendEventOutput:
        times, mask = self._validate(x, times, mask)
        x_safe = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))

        trend = self._trend(x_safe, times, mask)
        residual = torch.where(
            mask.unsqueeze(-1), x_safe - trend, torch.zeros_like(x_safe)
        )

        atoms = self._gabor_bank(residual, times, mask)
        energy = atoms.square().mean(dim=-1).add(self.eps).sqrt()
        threshold = F.softplus(self.gate_threshold).view(1, 1, 1, -1)
        gates = torch.sigmoid((energy - threshold) / self.gate_temperature)
        gates = gates * mask.unsqueeze(-1).to(gates.dtype)

        event = (
            atoms
            * gates.unsqueeze(-1)
            * self.gain.view(1, 1, 1, -1, 1)
        ).sum(dim=3) / (self.num_atoms ** 0.5)

        reconstruction = trend + self.event_decoder(event)
        reconstruction = torch.where(
            mask.unsqueeze(-1), reconstruction, torch.zeros_like(reconstruction)
        )

        smooth_terms = []
        t = x.shape[2]
        for i in range(1, t):
            valid_pair = mask[:, :, i] & mask[:, :, i - 1]
            dt = (times[:, :, i] - times[:, :, i - 1]).clamp_min(self.eps)
            deriv = (trend[:, :, i] - trend[:, :, i - 1]) / dt.unsqueeze(-1)
            if valid_pair.any():
                smooth_terms.append(deriv[valid_pair].square().mean())
        smoothness = torch.stack(smooth_terms).mean() if smooth_terms else x.new_zeros(())

        active = gates[mask]
        active_mean = active.mean() if active.numel() else x.new_zeros(())
        sparsity = active_mean + 5.0 * (active_mean - self.target_sparsity).clamp_min(0.0)

        valid = mask.unsqueeze(-1).expand_as(x)
        reconstruction_loss = (
            (reconstruction[valid] - x[valid]).square().mean()
            if mask.any() else x.new_zeros(())
        )

        losses = {
            "trend_smoothness": self.smoothness_weight * smoothness,
            "event_sparsity": self.sparsity_weight * sparsity,
            "decomposition_reconstruction": self.reconstruction_weight * reconstruction_loss,
        }
        return TrendEventOutput(
            trend=trend,
            residual=residual,
            event=event,
            reconstruction=reconstruction,
            gates=gates,
            atom_responses=atoms,
            losses=losses,
        )

    @staticmethod
    def total_auxiliary_loss(output: TrendEventOutput) -> torch.Tensor:
        return sum(output.losses.values())
