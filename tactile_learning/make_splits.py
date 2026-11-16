"""Create reproducible, class-preserving episode groups for evaluation."""
import argparse

from tactile.data.splits import build_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", choices=["grouped", "position", "pose"], default="grouped")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.15)
    parser.add_argument("--test-fraction", type=float, default=.15)
    parser.add_argument("--block-mm", type=float, default=15.0)
    parser.add_argument("--buffer-mm", type=float, default=0.0,
                        help="Position split only: exclude nearby train/val target offsets")
    parser.add_argument("--allow-single-class", action="store_true", help="Debug only; not recognition evaluation")
    args = parser.parse_args()
    result = build_splits(args.index, args.output, protocol=args.protocol, seed=args.seed,
                          val_fraction=args.val_fraction, test_fraction=args.test_fraction,
                          block_mm=args.block_mm, buffer_mm=args.buffer_mm,
                          allow_single_class=args.allow_single_class)
    for label, counts in result["class_counts"].items():
        print(f"{label}: {counts}")
    print(f"Excluded: {len(result['excluded'])}; saved {args.output}")


if __name__ == "__main__":
    main()
