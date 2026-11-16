"""Reconstruct encoder-based gripper geometry without running collection or dynamics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tactile.geometry import GripperGeometry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Geometry cache root (e.g. artifacts/geometry); writes <episode id>/geometry.npz")
    parser.add_argument("--xml", required=True, type=Path, help="Scene XML containing the calibrated robot and tactile sites")
    parser.add_argument("--reference-body", default="gripper_mount")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    index = json.loads(args.index.read_text(encoding="utf-8-sig"))
    if index.get("schema_version") != 1:
        parser.error("Unsupported episode index schema")
    episodes = index["episodes"][:args.limit]
    geometry = GripperGeometry(args.xml, args.reference_body)
    counts = {"written": 0, "cached": 0, "error": 0}
    for episode in episodes:
        try:
            result = geometry.compute_episode(episode, args.output, args.force)
        except Exception as exc:
            result = {"id": episode["id"], "status": "error", "error": str(exc)}
        counts[result["status"]] += 1
        print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps({"total": len(episodes), **counts}), flush=True)
    return int(counts["error"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
