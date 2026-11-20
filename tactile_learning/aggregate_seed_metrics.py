#!/usr/bin/env python3
"""Aggregate test metrics from independent random-seed evaluation runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tactile.metrics import classification_metrics


METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def load_run(run_dir: Path):
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"{run_dir} must contain metrics.json and predictions.csv")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    classes = [row["label"] for row in metrics["per_class"]]
    with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    predictions = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    if not len(targets):
        raise ValueError(f"No predictions in {predictions_path}")
    return classes, targets, predictions


def bootstrap_metrics(targets, predictions, classes, rng, repeats):
    n = len(targets)
    samples = {name: [] for name in METRICS}
    for _ in range(repeats):
        indices = rng.integers(0, n, size=n)
        result = classification_metrics(targets[indices], predictions[indices], classes)
        for name in METRICS:
            samples[name].append(result[name])
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Evaluation directories, one per independent seed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("Provide at least two independent seed evaluation directories")
    if args.bootstrap < 100:
        parser.error("--bootstrap must be >=100")

    rng = np.random.default_rng(args.seed)
    per_seed = []
    boot_values = {name: [] for name in METRICS}
    reference_classes = None

    for run in args.runs:
        classes, targets, predictions = load_run(Path(run))
        if reference_classes is None:
            reference_classes = classes
        elif classes != reference_classes:
            raise ValueError("Class order differs across seed runs")
        result = classification_metrics(targets, predictions, classes)
        per_seed.append({"run": str(Path(run).resolve()), **{
            name: result[name] for name in METRICS
        }})
        boot = bootstrap_metrics(targets, predictions, classes, rng, args.bootstrap)
        for name in METRICS:
            boot_values[name].extend(boot[name])

    summary = {"num_seeds": len(per_seed), "bootstrap_repeats_per_seed": args.bootstrap,
               "seed_results": per_seed, "summary": {}}
    for name in METRICS:
        values = np.asarray([row[name] for row in per_seed], dtype=np.float64)
        boots = np.asarray(boot_values[name], dtype=np.float64)
        summary["summary"][name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "mean_plus_minus_std": f"{values.mean():.4f} ± {values.std(ddof=1):.4f}",
            "bootstrap_95ci": [
                float(np.percentile(boots, 2.5)),
                float(np.percentile(boots, 97.5)),
            ],
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
