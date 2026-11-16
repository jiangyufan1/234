"""Train a baseline or full three-contact model; see README.md for commands."""
import argparse
from tactile.config import VARIANTS, load_config, resolve_config
from tactile.engine import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--config")
    p.add_argument("--variant", choices=VARIANTS, default="full")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--resume", help="Path to this run's last.pt checkpoint")
    args = p.parse_args()
    config = load_config(args.config, args.variant, args.seed)
    for arg, key in ((args.epochs, "epochs"), (args.batch_size, "batch_size"), (args.workers, "num_workers")):
        if arg is not None:
            config["training"][key] = arg
    train(args.index, args.split, resolve_config(config), args.output,
          args.device, args.resume, args.variant)


if __name__ == "__main__":
    main()
