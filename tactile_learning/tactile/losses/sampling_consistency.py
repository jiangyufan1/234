"""Sampling-rate consistency losses.

The loss is applied to semantic latent representations, not raw RGB/Flow.
Thus it does not force physically different observations to become identical.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def symmetric_kl(p_logits, q_logits, temperature=1.0):
    p = F.log_softmax(p_logits / temperature, dim=-1)
    q = F.log_softmax(q_logits / temperature, dim=-1)
    return 0.5 * (F.kl_div(p, q.exp(), reduction="batchmean") +
                  F.kl_div(q, p.exp(), reduction="batchmean"))

def representation_consistency(z_a, z_b, mask=None):
    if z_a.shape != z_b.shape:
        raise ValueError("paired representations must have equal shape")
    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    sim = (a * b).sum(-1)
    if mask is not None:
        mask = mask.bool()
        if not mask.any():
            return z_a.new_zeros(())
        sim = sim[mask]
    return (1.0 - sim).mean()

def sampling_consistency_loss(z_a, z_b, logits_a=None, logits_b=None,
                              weight_repr=1.0, weight_logits=0.5):
    loss = weight_repr * representation_consistency(z_a, z_b)
    if logits_a is not None and logits_b is not None:
        loss = loss + weight_logits * symmetric_kl(logits_a, logits_b)
    return loss
