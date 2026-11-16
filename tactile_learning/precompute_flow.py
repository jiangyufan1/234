"""Precompute optical flow on every adjacent RGB pair; safe to resume."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from tactile.features import FEATURE_VERSION, RaftFlow, compute_episode_flow


def _job(args):
    episode, output, config, force = args
    try:
        return compute_episode_flow(episode, output, config, force)
    except Exception as exc:
        return {"id": episode["id"], "status": "error", "error": str(exc)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Flow cache root (e.g. artifacts/flow); writes <episode id>/flow.npz")
    parser.add_argument("--backend", choices=("farneback", "raft"), default="farneback")
    parser.add_argument("--device", default="cuda", help="RAFT device; Farneback runs on CPU")
    parser.add_argument("--workers", type=int, default=1, help="Farneback process count; RAFT requires 1")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8, help="RAFT image-pair batch size")
    parser.add_argument("--iterations", type=int, default=12, help="RAFT refinement iterations")
    parser.add_argument("--limit", type=int, help="Only first N indexed episodes (smoke check)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or min(args.height, args.width) < 16 or (args.limit is not None and args.limit < 1):
        parser.error("workers/limit must be positive; height and width must be >=16")
    if args.backend == "raft" and args.workers != 1:
        parser.error("RAFT uses one model with --batch-size; set --workers 1")
    index = json.loads(args.index.read_text(encoding="utf-8-sig"))
    if index.get("schema_version") != 1:
        parser.error("Unsupported episode index schema")
    episodes = index["episodes"][:args.limit]
    config = {"version": FEATURE_VERSION, "backend": args.backend, "height": args.height, "width": args.width,
              "normalization": "u/width,v/height", "consecutive_pairs": True}
    if args.backend == "raft":
        config.update(weights="Raft_Small_Weights.DEFAULT", iterations=args.iterations)
    else:
        config["farneback"] = {"pyr_scale": 0.5, "levels": 3, "winsize": 15, "iterations": 3, "poly_n": 5, "poly_sigma": 1.2}
    counts = {"written": 0, "cached": 0, "error": 0}
    def report(result):
        counts[result["status"]] += 1
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.backend == "raft":
        raft = RaftFlow(args.device, args.batch_size, args.iterations)
        for episode in episodes:
            try:
                report(compute_episode_flow(episode, args.output, config, args.force, raft))
            except Exception as exc:
                report({"id": episode["id"], "status": "error", "error": str(exc)})
    else:
        jobs = [(e, args.output, config, args.force) for e in episodes]
        if args.workers == 1:
            for job in jobs:
                report(_job(job))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for result in executor.map(_job, jobs):
                    report(result)
    print(json.dumps({"total": len(episodes), **counts}), flush=True)
    return int(counts["error"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
