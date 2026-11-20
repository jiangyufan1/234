"""Training/evaluation loops kept separate from CLI, model and data code."""
from __future__ import annotations

import csv
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from tactile.data.dataset import TactileEpisodeDataset
from tactile.metrics import classification_metrics
from tactile.models.classifier import create_model
from tactile.runtime import (choose_device, file_digest, move_batch, read_json,
    save_checkpoint, seed_worker, set_seed, write_json)


def make_loader(dataset, config, device, training=False, generator=None):
    t = config["training"]
    sampler = None
    if training and t["balanced_sampling"]:
        counts = np.bincount(dataset.labels)
        weights = [1.0 / counts[y] for y in dataset.labels]
        sampler = WeightedRandomSampler(weights, len(weights), generator=generator)
    options = dict(batch_size=t["batch_size"], shuffle=training and sampler is None,
        sampler=sampler, num_workers=t["num_workers"], pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=generator, drop_last=False)
    if t["num_workers"]:
        options.update(multiprocessing_context="spawn", persistent_workers=False)
    return DataLoader(dataset, **options)


def perturb_batch(batch, shuffle_time=False, finger_dropout=0.0, drop_fingers=(), shuffle_seed=None):
    """Ablations preserve per-flow dt and keep at least one valid contact."""
    result = dict(batch)
    if shuffle_time:
        flow, mask, dt = batch["flow"], batch["flow_mask"], batch["flow_dt"]
        permutations = []
        for row in range(flow.shape[0]):
            generator = None
            if shuffle_seed is not None:
                key = f"{shuffle_seed}:{batch['episode_id'][row]}".encode()
                seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63)
                generator = torch.Generator(device=flow.device).manual_seed(seed)
            permutations.append(torch.randperm(flow.shape[2], device=flow.device, generator=generator))
        permutations = torch.stack(permutations)
        result["flow"] = flow.gather(2, permutations[:, None, :, None, None, None].expand_as(flow))
        result["flow_mask"] = mask.gather(2, permutations[:, None, :].expand_as(mask))
        result["flow_dt"] = dt.gather(1, permutations)
    node_mask = batch["node_mask"].clone()
    if finger_dropout:
        keep = torch.rand(node_mask.shape, device=node_mask.device) >= finger_dropout
        keep &= node_mask
        # Preserve a randomly chosen originally valid finger for each empty row.
        for row in (~keep.any(1)).nonzero().flatten().tolist():
            valid = node_mask[row].nonzero().flatten()
            if valid.numel():
                keep[row, valid[torch.randint(valid.numel(), (), device=valid.device)]] = True
        node_mask = keep
    for finger in drop_fingers:
        node_mask[:, finger] = False
    if not node_mask.any(1).all():
        raise ValueError("A sample has no valid contact after masking; use a compatible evaluation subset.")
    result["node_mask"] = node_mask
    result["rgb_mask"] = batch["rgb_mask"] & node_mask[:, :, None]
    result["flow_mask"] = result["flow_mask"] & node_mask[:, :, None]
    return result


def run_epoch(model, loader, device, classes, config, optimizer=None, scaler=None,
              drop_fingers=()):
    training = optimizer is not None
    model.train(training)
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")
    targets, predictions, records = [], [], []
    total_loss, total_count = 0.0, 0
    started = time.monotonic()
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for step, cpu_batch in enumerate(loader, 1):
            batch = move_batch(cpu_batch, device)
            batch = perturb_batch(batch,
                shuffle_time=config["training"]["temporal_shuffle"],
                finger_dropout=config["training"]["finger_dropout"] if training else 0,
                drop_fingers=drop_fingers,
                shuffle_seed=None if training else config["seed"])
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if hasattr(model, "forward_with_aux"):
                    logits, aux = model.forward_with_aux(batch)

                    classification_loss = torch.nn.functional.cross_entropy(
                        logits, batch["label"]
                    )

                    aux_names = (
                        "trend_smoothness",
                        "event_sparsity",
                        "decomposition_reconstruction",
                    )

                    aux_terms = [
                        value for key, value in aux.items()
                        if key in aux_names
                        or (
                            key.startswith("flow_")
                            and key[5:] in aux_names
                        )
                    ]

                    auxiliary_loss = (
                        torch.stack(aux_terms).sum()
                        if aux_terms
                        else classification_loss.new_zeros(())
                    )

                    loss = classification_loss + (
                        config["training"].get("cte_aux_weight", 1.0)
                        * auxiliary_loss
                    )
                else:
                    logits = model(batch)
                    loss = torch.nn.functional.cross_entropy(
                        logits, batch["label"]
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at episodes {batch['episode_id']}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config["training"]["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            truth = batch["label"].detach().cpu().tolist()
            probabilities = logits.detach().float().softmax(-1)
            confidence, prediction = probabilities.max(-1)
            prediction, confidence = prediction.cpu().tolist(), confidence.cpu().tolist()
            targets.extend(truth)
            predictions.extend(prediction)
            total_count += len(truth)
            total_loss += float(loss.detach()) * len(truth)
            if not training:
                probability_rows = probabilities.cpu().tolist()
                records.extend(
                    {"episode_id": eid, "target": y, "prediction": p, "confidence": c,
                     "probabilities": prob}
                    for eid, y, p, c, prob in zip(
                        batch["episode_id"], truth, prediction, confidence, probability_rows
                    )
                )
            if training and step % config["training"]["log_every"] == 0:
                print(f"  batch {step}/{len(loader)} loss={total_loss/total_count:.5f}", flush=True)
    metrics = classification_metrics(targets, predictions, classes)
    metrics.update(loss=total_loss/total_count, seconds=time.monotonic()-started)
    return metrics, records


def measure_model_efficiency(model, loader, device, config, warmup=5, repeats=30):
    # Model-only latency, parameter count and peak CUDA memory.
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    sample = move_batch(next(iter(loader)), device)
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")

    def infer():
        with torch.autocast(device_type=device.type, enabled=use_amp):
            return model(sample)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    with torch.inference_mode():
        for _ in range(warmup):
            infer()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            infer()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    batch_size = int(sample["label"].shape[0])
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "parameter_millions": float(total_params / 1e6),
        "batch_size": batch_size,
        "warmup_iterations": int(warmup),
        "measured_iterations": int(repeats),
        "inference_batch_latency_ms": float(elapsed * 1000 / repeats),
        "inference_sample_latency_ms": float(elapsed * 1000 / repeats / batch_size),
        "precision": "amp" if use_amp else "fp32",
        "peak_gpu_memory_gb": float(peak / 1024**3) if peak is not None else None,
        "incremental_peak_gpu_memory_gb": float(max(0, peak - baseline) / 1024**3) if peak is not None else None,
        "flops": None,
        "flops_note": "Not estimated: no validated FLOPs profiler is configured.",
    }


def dataset_signature(index_path, split_path):
    return {"index_sha256": file_digest(index_path), "split_sha256": file_digest(split_path)}


def train(index_path, split_path, config, output, device_name="auto", resume=None, variant="full"):
    index_path, split_path, output = Path(index_path).resolve(), Path(split_path).resolve(), Path(output).resolve()
    index, split = read_json(index_path), read_json(split_path)
    classes = index["classes"]
    if len(classes) < 2:
        raise ValueError("Classification training needs at least two object classes. Use inspect_dataset.py for one-class data.")
    if split.get("protocol") == "debug":
        raise ValueError("Debug splits are only for inspect_dataset.py, not training")
    if resume and (Path(resume).resolve().parent != output or Path(resume).name != "last.pt"):
        raise ValueError("Resume requires last.pt from the same --output directory; this preserves best.pt and run history")
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Run directory is not empty: {output}. Use a new directory or --resume.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["training"]["cpu_threads"])
    set_seed(config["seed"])
    device = choose_device(device_name)
    generator = torch.Generator().manual_seed(config["seed"])
    datasets = {name: TactileEpisodeDataset(index_path, split_path, name, config["data"])
                for name in ("train", "val")}
    for name, dataset in datasets.items():
        if set(dataset.labels) != set(range(len(classes))):
            raise ValueError(f"{name} partition does not contain all classes")
    loaders = {name: make_loader(ds, config, device, name == "train", generator)
               for name, ds in datasets.items()}
    # Loading a checkpoint must not require downloading pretrained weights.
    model_config = dict(config["model"])
    if resume:
        model_config["pretrained"] = False
    model = create_model(len(classes), model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"]["amp"] and device.type == "cuda")
    signature = dataset_signature(index_path, split_path)
    start_epoch, best, stale = 0, -1.0, 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["classes"] != classes or state["dataset_signature"] != signature:
            raise ValueError("Checkpoint belongs to a different class mapping/index/split")
        saved_config = state["config"]
        # Extending the total epoch budget is allowed; other changes create a new experiment.
        compare_config = copy.deepcopy(config)
        compare_config["training"]["epochs"] = saved_config["training"]["epochs"]
        if compare_config != saved_config:
            raise ValueError("Resume configuration differs (only total epochs may change)")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, stale = state["epoch"] + 1, state["best_macro_f1"], state["stale_epochs"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if device.type == "cuda" and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        generator.set_state(state["rng"]["loader"])
        if start_epoch >= config["training"]["epochs"]:
            raise ValueError("Checkpoint already reached requested epoch budget")
    run_info = {"config": config, "variant": variant, "classes": classes,
        "index_path": str(index_path), "split_path": str(split_path),
        "dataset_signature": signature, "device": str(device),
        "torch_version": str(torch.__version__),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "partition_sizes": {key: len(split[key]) for key in ("train", "val", "test")}}
    write_json(output / "run.json", run_info)
    print(f"device={device} classes={len(classes)} samples={run_info['partition_sizes']} parameters={run_info['trainable_parameters']:,}", flush=True)
    print("Selection uses validation macro-F1; the test partition is not evaluated during training.", flush=True)
    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}", flush=True)
        train_metrics, _ = run_epoch(model, loaders["train"], device, classes, config, optimizer, scaler)
        val_metrics, _ = run_epoch(model, loaders["val"], device, classes, config)
        improved = val_metrics["macro_f1"] > best
        best, stale = (val_metrics["macro_f1"], 0) if improved else (best, stale+1)
        record = {"epoch": epoch+1, "train": train_metrics, "val": val_metrics}
        with (output / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        state = {**run_info, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "best_macro_f1": best, "stale_epochs": stale,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "loader": generator.get_state()}}
        save_checkpoint(output / "last.pt", state)
        if improved:
            save_checkpoint(output / "best.pt", state)
        print(f"  train loss={train_metrics['loss']:.4f} val acc={val_metrics['accuracy']:.4f} macro-F1={val_metrics['macro_f1']:.4f} best={best:.4f}", flush=True)
        if stale >= config["training"]["patience"]:
            print("Early stopping: validation macro-F1 did not improve.", flush=True)
            break
    return run_info


def evaluate(checkpoint, output, device_name="auto", partition="test", num_workers=None,
             drop_fingers=()):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    index_path, split_path = state["index_path"], state["split_path"]
    if state["dataset_signature"] != dataset_signature(index_path, split_path):
        raise ValueError("Index or split changed since this checkpoint was trained")
    if num_workers is not None:
        config["training"]["num_workers"] = num_workers
    set_seed(config["seed"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = choose_device(device_name)
    dataset = TactileEpisodeDataset(index_path, split_path, partition, config["data"])
    loader = make_loader(dataset, config, device)
    model_config = dict(config["model"], pretrained=False)
    model = create_model(len(state["classes"]), model_config).to(device)
    model.load_state_dict(state["model"])
    metrics, records = run_epoch(model, loader, device, state["classes"], config, drop_fingers=drop_fingers)
    efficiency = measure_model_efficiency(model, loader, device, config)
    metrics.update(partition=partition, checkpoint=str(Path(checkpoint).resolve()),
                   drop_fingers=[i+1 for i in drop_fingers], seed=config["seed"],
                   efficiency=efficiency)
    # Pose/position annotations are used only for reporting, never model inputs.
    episodes = {e["id"]: e for e in read_json(index_path)["episodes"]}
    by_pose = {}
    for record in records:
        entry = episodes[record["episode_id"]]
        pose = entry.get("metadata", {}).get("pose_config", {}).get("pose_group", "unknown")
        by_pose.setdefault(str(pose), []).append(record)
    metrics["by_pose_group"] = {key: classification_metrics(
        [r["target"] for r in value], [r["prediction"] for r in value], state["classes"])
        for key, value in by_pose.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode_id", "target", "prediction", "confidence", "probabilities"]
        )
        writer.writeheader()
        csv_records = []
        for record in records:
            row = dict(record)
            row["probabilities"] = json.dumps(row["probabilities"], separators=(",", ":"))
            csv_records.append(row)
        writer.writerows(csv_records)
    np.savez_compressed(
        output / "probabilities.npz",
        probabilities=np.asarray([r["probabilities"] for r in records], dtype=np.float32),
        targets=np.asarray([r["target"] for r in records], dtype=np.int64),
        classes=np.asarray(state["classes"]),
    )
    write_json(output / "efficiency.json", efficiency)
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true / predicted", *state["classes"]])
        for label, row in zip(state["classes"], metrics["confusion_matrix"]):
            writer.writerow([label, *row])
    print(f"{partition}: count={metrics['count']} accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}", flush=True)
    return metrics
