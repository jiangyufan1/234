"""Metrics from integer counts, with no scikit-learn dependency."""
from __future__ import annotations

import numpy as np


def classification_metrics(targets, predictions, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    targets, predictions = np.asarray(targets, dtype=int), np.asarray(predictions, dtype=int)
    if targets.size == 0:
        raise ValueError("Cannot evaluate an empty partition")
    np.add.at(matrix, (targets, predictions), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    return {
        "count": int(support.sum()), "accuracy": float(tp.sum() / support.sum()),
        "macro_f1": float(f1[present].mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "classes_with_support": int(present.sum()),
        "per_class": [{"label": label, "support": int(support[i]),
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i])} for i, label in enumerate(classes)],
        "confusion_matrix": matrix.tolist(),
    }
