"""Evaluate a frozen checkpoint using the exact saved index and split."""
import argparse
from tactile.engine import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--partition", choices=("train", "val", "test"), default="test")
    p.add_argument("--workers", type=int)
    p.add_argument("--drop-fingers", type=int, nargs="*", choices=(1, 2, 3), default=[])
    args = p.parse_args()
    if len(set(args.drop_fingers)) == 3:
        p.error("Cannot remove all three fingers")
    evaluate(args.checkpoint, args.output, args.device, args.partition,
             args.workers, tuple(i-1 for i in args.drop_fingers))


if __name__ == "__main__":
    main()
