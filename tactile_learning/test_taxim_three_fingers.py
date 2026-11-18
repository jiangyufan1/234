from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(r"D:\robotics\234\tactile_learning")
COLLECTION = PROJECT / "scripts" / "collection"
XML = PROJECT / "simulation" / "mcb_xml" / "clamp_train_00015149_v2.xml"
BASE_COLLECTOR = COLLECTION / "collect_underactuated_tactile_dataset_pose_sweep_v3.py"
MCB_ADAPTER = COLLECTION / "collect_mcb_gelsight_dataset.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    for p in (XML, BASE_COLLECTOR, MCB_ADAPTER):
        if not p.is_file():
            raise FileNotFoundError(p)

    print("[1/5] Compile MuJoCo model")
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data,
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    )
    mujoco.mj_forward(model, data)
    print(f"      nq={model.nq}, nv={model.nv}, nu={model.nu}, dt={model.opt.timestep}")

    print("[2/5] Load existing collector and MCB Taxim adapter")
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    if str(COLLECTION) not in sys.path:
        sys.path.insert(0, str(COLLECTION))
    adapter = load_module("mcb_gelsight_adapter_test", MCB_ADAPTER)
    collector = adapter.load_collector(BASE_COLLECTOR, skip_taxim=False)

    print("[3/5] Configure MCB renderer adapter")
    adapter.configure_collector(
        collector=collector,
        xml_path=XML,
        label="clamp_train_00015149_v2_taxim_test",
        output=PROJECT / "runs" / "taxim_three_finger_test",
        configs=[{
            "grasp_offset_object_m": [0.0, 0.0, 0.0],
            "approach_azimuth_deg": 0.0,
            "approach_tilt_deg": 0.0,
            "gripper_roll_deg": 0.0,
        }],
        repeats=1,
        seed=20261118,
        show_viewer=False,
        save_images=True,
        pin_object=False,
        sensor_clearance_m=0.001,
    )

    print("[4/5] Initialize all three Taxim sensors")
    collector.init_tactile_renderers(model, data)
    names = ("finger1", "finger2", "finger3")
    if set(collector.TACTILE_RENDERERS) != set(names):
        raise RuntimeError(f"Unexpected renderer keys: {list(collector.TACTILE_RENDERERS)}")
    print("      renderers:", list(collector.TACTILE_RENDERERS))

    print("[5/5] Render two synchronized frames and validate RGB/Depth")
    first = collector.capture_gelsight_images(model, data)
    mujoco.mj_step(model, data)
    second = collector.capture_gelsight_images(model, data)

    for finger in names:
        rgb1 = np.asarray(first["rgb"][finger])
        dep1 = np.asarray(first["depth"][finger])
        rgb2 = np.asarray(second["rgb"][finger])
        dep2 = np.asarray(second["depth"][finger])

        assert rgb1.dtype == np.uint8, (finger, rgb1.dtype)
        assert rgb1.ndim == 3 and rgb1.shape[-1] == 3, (finger, rgb1.shape)
        assert dep1.dtype == np.float32, (finger, dep1.dtype)
        assert dep1.shape == rgb1.shape[:2], (finger, dep1.shape, rgb1.shape)
        assert np.isfinite(dep1).all(), f"{finger}: depth contains NaN/Inf"
        assert rgb2.shape == rgb1.shape
        assert dep2.shape == dep1.shape
        assert np.isfinite(dep2).all(), f"{finger}: second depth contains NaN/Inf"

        print(
            f"      {finger}: RGB={rgb1.shape}/{rgb1.dtype}, "
            f"Depth={dep1.shape}/{dep1.dtype}, "
            f"depth_range=({float(dep1.min()):.6g}, {float(dep1.max()):.6g}), "
            f"frame_changed={bool(np.any(rgb1 != rgb2) or np.any(dep1 != dep2))}"
        )

    print("\nTAXIM THREE-FINGER RGB/DEPTH TEST: PASS")
    print("Note: this is a renderer smoke test, not a physical contact/indentation validation.")


if __name__ == "__main__":
    main()
