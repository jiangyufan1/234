#!/usr/bin/env python
"""MCB OBJ selection, category inventory, and centered robot-scene helpers."""

from __future__ import annotations

import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
MCB_ROOT = PROJECT / "data" / "raw" / "MCB_A"
DATASET_ROOT = MCB_ROOT / "dataset_org_norm"
TEMPLATE_XML = PROJECT / "simulation" / "mcb_xml" / "articulated_joint_train_00011996.xml"
TEMP_ROOT = Path(tempfile.gettempdir()) / "mcb_any_object_models"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "mcb_object"


def select_representatives(dataset_root=None) -> list[tuple[str, Path]]:
    """Pick the first train OBJ per category, falling back to the first test OBJ."""
    dataset_root = DATASET_ROOT if dataset_root is None else Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"MCB_A 数据集目录不存在：{dataset_root}")
    categories: dict[str, dict[str, Path]] = {}
    for split in ("train", "test"):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for category_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            candidates = sorted(category_dir.glob("*.obj"))
            if candidates:
                categories.setdefault(category_dir.name, {})[split] = candidates[0]
    return [(category, paths.get("train", paths.get("test")))
            for category, paths in sorted(categories.items())]


def choose_obj(arg_path: str | None) -> Path:
    if arg_path:
        path = Path(arg_path).expanduser().resolve()
    else:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="选择 MCB_A 零件 OBJ（train/test 任意类别）",
            initialdir=str(DATASET_ROOT if DATASET_ROOT.is_dir() else MCB_ROOT),
            filetypes=(("MCB OBJ", "*.obj"), ("All files", "*.*")),
        )
        root.destroy()
        if not selected:
            raise SystemExit("未选择 OBJ，已退出。")
        path = Path(selected).resolve()
    if path.suffix.lower() != ".obj" or not path.is_file():
        raise FileNotFoundError(f"OBJ 文件不存在或扩展名不正确：{path}")
    return path


def object_label(obj_path: Path) -> str:
    try:
        relative = obj_path.relative_to(DATASET_ROOT)
        parts = relative.parts
        split = parts[0] if len(parts) >= 3 else "custom"
        category = parts[1] if len(parts) >= 3 else obj_path.parent.name
    except ValueError:
        split, category = "custom", obj_path.parent.name
    return slug(f"{split}_{category}_{obj_path.stem}")


def make_centered_xml(obj_path: Path, label: str) -> Path:
    if not TEMPLATE_XML.is_file():
        raise FileNotFoundError(f"找不到通用 UR3/三指 MuJoCo 模板：{TEMPLATE_XML}")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    target = TEMP_ROOT / f"{label}.xml"
    tree = ET.parse(TEMPLATE_XML)
    root = tree.getroot()

    # Resolve robot mesh directory before compiling XML from string.
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("Template XML has no compiler element")

    meshdir = Path(compiler.get("meshdir", "."))
    if not meshdir.is_absolute():
        meshdir = (TEMPLATE_XML.parent / meshdir).resolve()

    if not meshdir.is_dir():
        raise FileNotFoundError(
            f"Robot mesh directory missing: {meshdir}"
        )

    compiler.set("meshdir", meshdir.as_posix())

    visual = root.find(".//geom[@name='mcb_part_visual']")
    collision = root.find(".//geom[@name='mcb_part_collision']")
    if visual is None or collision is None or not visual.get("mesh"):
        raise RuntimeError("模板缺少 MCB 视觉/碰撞网格")
    old_mesh_name = visual.attrib["mesh"]
    asset = root.find(f".//mesh[@name='{old_mesh_name}']")
    if asset is None:
        raise RuntimeError(f"模板缺少 mesh asset：{old_mesh_name}")
    asset.set("name", "mcb_selected_object_mesh")
    asset.set("file", obj_path.as_posix())
    visual.set("mesh", "mcb_selected_object_mesh")
    collision.set("mesh", "mcb_selected_object_mesh")

    # Recenter the source OBJ's nonzero 0..1 vertex range at the body's origin.
    original_model = mujoco.MjModel.from_xml_path(str(TEMPLATE_XML))
    mesh_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    _ = original_model  # validates that the base robot template itself compiles
    geom_ids = [
        mujoco.mj_name2id(mesh_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("mcb_part_visual", "mcb_part_collision")
    ]
    if min(geom_ids) < 0:
        raise RuntimeError("所选模型编译后缺少 MCB 视觉/碰撞 geom")
    offsets = [mesh_model.geom_pos[g].copy() for g in geom_ids]
    if not np.allclose(offsets[0], offsets[1], atol=1e-8):
        raise RuntimeError(f"视觉/碰撞网格中心偏移不同：{offsets}")
    for name in ("mcb_part_visual", "mcb_part_collision"):
        geom = root.find(f".//geom[@name='{name}']")
        old_pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
        geom.set("pos", " ".join(f"{v:.12g}" for v in old_pos - offsets[0]))
    tree.write(target, encoding="utf-8", xml_declaration=True)

    centered = mujoco.MjModel.from_xml_path(str(target))
    if not np.allclose(centered.opt.gravity, 0.0, atol=1e-12):
        raise RuntimeError(f"模板重力不是零：{centered.opt.gravity}")
    for name in ("mcb_part_visual", "mcb_part_collision"):
        gid = mujoco.mj_name2id(centered, mujoco.mjtObj.mjOBJ_GEOM, name)
        if not np.allclose(centered.geom_pos[gid], 0.0, atol=1e-7):
            raise RuntimeError(f"网格中心校正失败：{name}={centered.geom_pos[gid]}")
    print(f"[OK] OBJ={obj_path}")
    print(f"[OK] center correction={offsets[0].tolist()} m; gravity={centered.opt.gravity.tolist()}")
    return target


