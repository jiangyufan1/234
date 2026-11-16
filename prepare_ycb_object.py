# -*- coding: utf-8 -*-
r"""
prepare_ycb_object.py

当前对象目录：
E:\project\p1\deepgelsight\object\ycb\cracker_box

功能：
- 读取 textured.obj
- 检查尺寸与单位
- 自动转成米（仅内存中，不改原文件）
- 表面均匀采样 2048 点
- 生成对应法向
- 保存 pointcloud_2048.npy / normals_2048.npy / object_meta.json / preview.png
- 检查 textured_coacd_*.stl 碰撞模型
"""

from pathlib import Path
import json
import sys
import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBJECT_DIR = Path(r"E:\project\p1\deepgelsight\object\ycb\cracker_box")
VISUAL_MESH_NAME = "textured.obj"
N_POINTS = 2048

# None=自动判断；1.0=原始单位为米；0.001=原始单位为毫米
FORCE_SCALE_TO_METERS = None


def load_mesh(path: Path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"无法读取为 Trimesh: {path}")
    return mesh


def choose_scale(extents):
    max_extent = float(np.max(extents))

    if FORCE_SCALE_TO_METERS is not None:
        return float(FORCE_SCALE_TO_METERS), "forced"

    if max_extent <= 1.0:
        return 1.0, "auto: likely meters"

    if max_extent >= 10.0:
        return 0.001, "auto: likely millimeters"

    raise RuntimeError(
        f"模型最大原始尺寸为 {max_extent:.6f}，处于 1~10 的模糊区间。\n"
        "请在脚本顶部手动设置 FORCE_SCALE_TO_METERS：\n"
        "1.0    -> 原始单位为米\n"
        "0.001  -> 原始单位为毫米"
    )


def save_preview(points, path):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("YCB CAD point cloud")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    half = max(float(np.max(maxs - mins)) / 2.0, 1e-3)

    ax.set_xlim(center[0]-half, center[0]+half)
    ax.set_ylim(center[1]-half, center[1]+half)
    ax.set_zlim(center[2]-half, center[2]+half)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    print("=" * 72)
    print("YCB object preparation")
    print("=" * 72)
    print("OBJECT_DIR =", OBJECT_DIR)

    if not OBJECT_DIR.exists():
        raise FileNotFoundError(f"找不到目录：{OBJECT_DIR}")

    visual_path = OBJECT_DIR / VISUAL_MESH_NAME
    if not visual_path.exists():
        objs = list(OBJECT_DIR.glob("*.obj"))
        if len(objs) == 1:
            visual_path = objs[0]
        else:
            raise FileNotFoundError(
                f"找不到 {VISUAL_MESH_NAME}，当前 OBJ: {[p.name for p in objs]}"
            )

    mesh_raw = load_mesh(visual_path)

    extents_raw = np.asarray(mesh_raw.extents, dtype=float)
    bounds_raw = np.asarray(mesh_raw.bounds, dtype=float)
    center_raw = np.asarray(mesh_raw.bounding_box.centroid, dtype=float)

    print("\n[Visual mesh]")
    print("file      =", visual_path.name)
    print("vertices  =", len(mesh_raw.vertices))
    print("faces     =", len(mesh_raw.faces))
    print("extents   =", extents_raw)
    print("center    =", center_raw)
    print("watertight=", bool(mesh_raw.is_watertight))

    scale, reason = choose_scale(extents_raw)
    print("\n[Scale]")
    print("scale_to_meters =", scale)
    print("reason          =", reason)

    mesh_m = mesh_raw.copy()
    mesh_m.apply_scale(scale)

    extents_m = np.asarray(mesh_m.extents, dtype=float)
    bounds_m = np.asarray(mesh_m.bounds, dtype=float)
    center_m = np.asarray(mesh_m.bounding_box.centroid, dtype=float)

    print("\n[Size in meters]")
    print("extents_m  =", extents_m)
    print(
        "extents_mm =",
        np.round(extents_m * 1000.0, 3)
    )

    points, face_idx = trimesh.sample.sample_surface(mesh_m, N_POINTS)
    normals = np.asarray(mesh_m.face_normals[face_idx], dtype=float)
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)

    points = points.astype(np.float32)
    normals = normals.astype(np.float32)

    pc_path = OBJECT_DIR / f"pointcloud_{N_POINTS}.npy"
    normal_path = OBJECT_DIR / f"normals_{N_POINTS}.npy"
    preview_path = OBJECT_DIR / "pointcloud_preview.png"
    meta_path = OBJECT_DIR / "object_meta.json"

    np.save(pc_path, points)
    np.save(normal_path, normals)
    save_preview(points, preview_path)

    collision_files = sorted(OBJECT_DIR.glob("textured_coacd_*.stl"))
    collision_info = []

    print("\n[Collision meshes]")
    if not collision_files:
        print("未发现 textured_coacd_*.stl")
    else:
        for p in collision_files:
            cm = load_mesh(p)
            ex_raw = np.asarray(cm.extents, dtype=float)
            ex_m = ex_raw * scale
            collision_info.append({
                "file": p.name,
                "faces": int(len(cm.faces)),
                "extents_raw": ex_raw.tolist(),
                "extents_m": ex_m.tolist(),
                "watertight": bool(cm.is_watertight),
            })
            print(
                p.name,
                "size(mm)=",
                np.round(ex_m * 1000.0, 2),
                "faces=",
                len(cm.faces)
            )

    meta = {
        "object_name": OBJECT_DIR.name,
        "visual_mesh": visual_path.name,
        "collision_meshes": [p.name for p in collision_files],
        "scale_to_meters": scale,
        "scale_reason": reason,
        "recentered": False,
        "coordinate_system_preserved": True,
        "bounds_raw": bounds_raw.tolist(),
        "extents_raw": extents_raw.tolist(),
        "bbox_center_raw": center_raw.tolist(),
        "bounds_m": bounds_m.tolist(),
        "extents_m": extents_m.tolist(),
        "bbox_center_m": center_m.tolist(),
        "point_count": N_POINTS,
        "pointcloud_file": pc_path.name,
        "normals_file": normal_path.name,
        "collision_info": collision_info,
    }

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n[Output]")
    print(pc_path)
    print(normal_path)
    print(meta_path)
    print(preview_path)
    print("pointcloud shape =", points.shape)
    print("normals shape    =", normals.shape)

    print("\n注意：未对 textured.obj 做二次居中或旋转。")
    print("标准点云保持与 YCB mesh 相同的物体坐标系。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[ERROR]", exc)
        sys.exit(1)
