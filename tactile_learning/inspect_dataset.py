"""Print tensor shapes/masks and optionally forward one sample, without training."""
import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tactile.config import VARIANTS, load_config
from tactile.data.dataset import TactileEpisodeDataset
from tactile.models.classifier import create_model
from tactile.runtime import choose_device, move_batch, read_json, write_json


def inspect(args, split):
    config = load_config(args.config, args.variant)
    dataset = TactileEpisodeDataset(args.index, split, args.partition, config["data"])
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    print("episodes:", len(dataset))
    for name, value in batch.items():
        if torch.is_tensor(value):
            print(name, list(value.shape), str(value.dtype),
                  "finite=" + str(bool(torch.isfinite(value).all())))
        else:
            print(name, value)
    print("rgb valid frames:", batch["rgb_mask"].sum(-1).tolist())
    print("flow valid pairs:", batch["flow_mask"].sum(-1).tolist())
    if args.forward or args.backward:
        torch.set_num_threads(config["training"]["cpu_threads"])
        device = choose_device(args.device)
        # At least two outputs solely to exercise the classification forward path.
        classes = read_json(args.index)["classes"]
        model = create_model(max(2, len(classes)), config["model"]).to(device)
        model.train(args.backward)
        with torch.enable_grad() if args.backward else torch.inference_mode():
            logits = model(move_batch(batch, device))
            if args.backward:
                logits.square().mean().backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Missing or nonfinite gradients")
                print("backward: finite gradients in", len(gradients), "parameter tensors")
        print("forward logits:", list(logits.shape), "finite:", bool(torch.isfinite(logits).all()))
        print("This is a shape/forward check, not a recognition accuracy result.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--split")
    p.add_argument("--partition", choices=("train", "val", "test"), default="train")
    p.add_argument("--config")
    p.add_argument("--variant", choices=VARIANTS, default="static_concat")
    p.add_argument("--forward", action="store_true")
    p.add_argument("--backward", action="store_true", help="Also check backward gradients; no optimizer step")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.split:
        inspect(args, args.split)
    else:
        if args.partition != "train":
            p.error("Without --split only the debug train partition is available")
        index = read_json(args.index)
        with tempfile.TemporaryDirectory(prefix="tactile_inspect_") as folder:
            split = Path(folder) / "debug_split.json"
            write_json(split, {"schema_version": 1, "index_path": str(Path(args.index).resolve()),
                "protocol": "debug", "seed": 0, "train": [e["id"] for e in index["episodes"]],
                "val": [], "test": [], "excluded": []})
            inspect(args, split)


if __name__ == "__main__":
    main()
