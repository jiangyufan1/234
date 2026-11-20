# Causal early-recognition utilities and observation-ratio metrics.
from __future__ import annotations
import numpy as np
import torch
from tactile.metrics import classification_metrics

@torch.no_grad()
def earliest_confident_time(logits_by_time, times, confidence=0.90, patience=2, target=None):
    if logits_by_time.ndim != 2 or times.ndim != 1:
        raise ValueError("logits must be [T,C] and times must be [T]")
    if logits_by_time.shape[0] != times.numel():
        raise ValueError("logits and timestamps must have equal length")
    if not 0 < confidence < 1 or patience < 1:
        raise ValueError("Invalid confidence or patience")
    if times.numel() == 0:
        return None
    if not torch.isfinite(times).all() or (times.numel() > 1 and not bool((times[1:] > times[:-1]).all())):
        raise ValueError("timestamps must be finite and strictly increasing")
    probs = logits_by_time.softmax(-1)
    conf, pred = probs.max(-1)
    for i in range(times.numel() - patience + 1):
        stable = (conf[i:i+patience] >= confidence).all() and (pred[i:i+patience] == pred[i]).all()
        correct = target is None or int(pred[i]) == int(target)
        if bool(stable) and correct:
            return float(times[i + patience - 1])
    return None

@torch.no_grad()
def causal_prefix_logits(model, batch, query_times):
    outputs = []
    for t in query_times:
        if hasattr(model, "forward_with_aux"):
            logits = model.forward_with_aux(batch, query_time=float(t))[0]
        else:
            logits = model(batch, query_time=float(t))
        outputs.append(logits)
    return torch.stack(outputs, dim=0)

def early_recognition_metrics(logits_by_ratio, targets, observation_ratios, classes,
                              logits_by_time=None, times=None, confidence=0.90, patience=2):
    # logits_by_ratio [R,N,C]; optional logits_by_time/times [N,T,C]/[N,T].
    logits = torch.as_tensor(logits_by_ratio).detach().float().cpu()
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    ratios = np.asarray(observation_ratios, dtype=np.float64).reshape(-1)
    if logits.ndim != 3 or logits.shape[:2] != (len(ratios), len(y)):
        raise ValueError("logits_by_ratio must be [R,N,C] and match ratios/targets")
    if len(ratios) < 2 or not np.isfinite(ratios).all() or (ratios <= 0).any() or (ratios > 1).any() or not np.all(np.diff(ratios) > 0):
        raise ValueError("Ratios must increase strictly within (0,1]")
    rows, accuracies = [], []
    for i, ratio in enumerate(ratios):
        m = classification_metrics(y, logits[i].argmax(-1).numpy(), classes)
        accuracies.append(m["accuracy"])
        rows.append({"observation_ratio": float(ratio), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    result = {
        "by_observation_ratio": rows,
        "auc_or_accuracy": float(np.trapezoid(np.asarray(accuracies), ratios)),
        "auc_or_ratio_range": [float(ratios[0]), float(ratios[-1])],
    }
    if (logits_by_time is None) != (times is None):
        raise ValueError("Provide both logits_by_time and times for TTCR")
    if logits_by_time is not None:
        time_logits = torch.as_tensor(logits_by_time).detach().float().cpu()
        time_values = torch.as_tensor(times).detach().float().cpu()
        if time_logits.ndim != 3 or time_logits.shape[0] != len(y) or time_values.shape != time_logits.shape[:2]:
            raise ValueError("Expected logits_by_time [N,T,C] and times [N,T]")
        ttcr = []
        for i in range(len(y)):
            value = earliest_confident_time(time_logits[i], time_values[i], confidence, patience, target=int(y[i]))
            if value is not None:
                ttcr.append(value)
        result["ttcr"] = {
            "mean_seconds": float(np.mean(ttcr)) if ttcr else None,
            "median_seconds": float(np.median(ttcr)) if ttcr else None,
            "successful_samples": len(ttcr),
            "success_rate": float(len(ttcr) / len(y)),
            "confidence_threshold": float(confidence),
            "patience": int(patience),
        }
    return result
