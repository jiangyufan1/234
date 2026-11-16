"""JSON configuration and named, reproducible ablation presets."""
from __future__ import annotations

import copy
import json
from pathlib import Path


DEFAULTS = {
    "seed": 42,
    "data": {
        "height": 128, "width": 96, "static_frames": 4,
        "max_flow_frames": 31, "normalize_rgb": True,
        "contact_source": "signals", "flow_cache": "artifacts/flow",
        "geometry_cache": "artifacts/geometry",
        "flow_backend": "farneback", "geometry_reference_body": "gripper_mount",
    },
    "model": {
        "backbone": "resnet18", "pretrained": False,
        "use_flow": True, "use_alignment": True,
        "fusion": "graph", "single_finger": 0,
        "use_edge_geometry": True, "feature_dim": 128, "dropout": 0.2,
    },
    "training": {
        "epochs": 60, "batch_size": 4, "num_workers": 2,
        "learning_rate": 0.0001, "weight_decay": 0.0001,
        "amp": True, "grad_clip": 1.0, "patience": 12,
        "balanced_sampling": False, "finger_dropout": 0.0,
        "temporal_shuffle": False, "log_every": 10, "cpu_threads": 4,
    },
}

VARIANTS = {
    "full": {},
    "static_single1": {"model": {"fusion": "single", "single_finger": 0,
        "use_flow": False, "use_alignment": False, "use_edge_geometry": False}},
    "static_single2": {"model": {"fusion": "single", "single_finger": 1,
        "use_flow": False, "use_alignment": False, "use_edge_geometry": False}},
    "static_single3": {"model": {"fusion": "single", "single_finger": 2,
        "use_flow": False, "use_alignment": False, "use_edge_geometry": False}},
    "static_concat": {"model": {"fusion": "concat", "use_flow": False,
        "use_alignment": False, "use_edge_geometry": False}},
    "dynamic_concat": {"model": {"fusion": "concat", "use_flow": True,
        "use_alignment": False, "use_edge_geometry": False}},
    "no_flow": {"model": {"use_flow": False}},
    "no_alignment": {"model": {"use_alignment": False}},
    "no_geometry": {"model": {"use_edge_geometry": False}},
    "no_edge_geometry": {"model": {"use_edge_geometry": False}},
    "no_geometry_input": {"model": {"use_edge_geometry": False, "use_alignment": False}},
    "mean": {"model": {"fusion": "mean", "use_edge_geometry": False}},
    "concat": {"model": {"fusion": "concat", "use_edge_geometry": False}},
    "temporal_shuffle": {"training": {"temporal_shuffle": True}},
}


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key not in result:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None, variant="full", seed=None):
    config = copy.deepcopy(DEFAULTS)
    if path:
        with Path(path).open(encoding="utf-8-sig") as f:
            config = merge(config, json.load(f))
    config = merge(config, VARIANTS[variant])
    if seed is not None:
        config["seed"] = seed
    return resolve_config(config)


def resolve_config(config):
    """Derive cache needs from the model rather than independent switches."""
    config = copy.deepcopy(config)
    data, model, training = config["data"], config["model"], config["training"]
    data["use_flow"] = bool(model["use_flow"])
    data["use_geometry"] = bool(model["use_alignment"] or
        (model["fusion"] == "graph" and model["use_edge_geometry"]))
    for key in ("flow_cache", "geometry_cache"):
        data[key] = str(Path(data[key]).resolve())
    for key in ("height", "width", "static_frames", "max_flow_frames"):
        if not isinstance(data[key], int) or data[key] <= 0:
            raise ValueError(f"data.{key} must be a positive integer")
    if data["contact_source"] not in ("signals", "all"):
        raise ValueError("contact_source must be signals or all")
    if data["flow_backend"] not in ("farneback", "raft"):
        raise ValueError("flow_backend must be farneback or raft")
    for key in ("epochs", "batch_size", "patience", "log_every", "cpu_threads"):
        if not isinstance(training[key], int) or training[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    if not isinstance(training["num_workers"], int) or training["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if not 0 <= training["finger_dropout"] < 1:
        raise ValueError("finger_dropout must be in [0, 1)")
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if training["finger_dropout"] and model["fusion"] == "single":
        raise ValueError("Do not enable finger_dropout for single-finger baselines")
    return config
