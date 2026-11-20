# Compact classification metrics for reproducible paper reporting.
from __future__ import annotations
import numpy as np

def classification_metrics(targets, predictions, classes):
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    k = len(classes)
    if targets.size == 0:
        raise ValueError("Cannot evaluate an empty partition")
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have equal lengths")
    if k < 2 or (targets < 0).any() or (targets >= k).any():
        raise ValueError("Invalid target class index or class list")
    if (predictions < 0).any() or (predictions >= k).any():
        raise ValueError("Prediction index outside classes range")
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    tp = np.diag(matrix).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    total = int(support.sum())
    return {
        "count": total,
        "accuracy": float(tp.sum() / total),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "balanced_accuracy": float(recall[present].mean()),
        "classes_with_support": int(present.sum()),
        "per_class": [
            {"label": label, "support": int(support[i]),
             "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i])}
            for i, label in enumerate(classes)
        ],
        "confusion_matrix": matrix.tolist(),
    }
