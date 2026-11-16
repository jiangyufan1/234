#!/usr/bin/env python
"""Audit YCB MuJoCo assets and build a tactile-collection pose/budget catalog.

This is a read-only model audit except for the JSON/CSV files it writes to
--output. It does not launch MuJoCo collection or modify the YCB source files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

_deps = Path(os.environ.get(
    "TACTILE_PYTHON_DEPS",
    r"E:\project\p1\rl\新建文件夹\output\tactile_collection\_deps",
))
if _deps.is_dir():
    sys.path.insert(0, str(_deps))


DEFAULT_ROOT = Path(r"E:\project\p1\deepgelsight\object\ycb")
DEFAULT_OUTPUT = Path(
    r"E:\project\p1\deepgelsight\dataset_underactuated\plans\ycb_tactile_catalog"
)
DEFAULT_SPACING_M = 0.005
DEFAULT_RADIAL_CAP_M = 0.015
DEFAULT_AXIAL_CAP_M = 0.030
DEFAULT_REPEATS = 3
DEFAULT_ROLLS_DEG = (0.0, 30.0, 60.0, 90.0)
DEFAULT_TABLE_TOP_M = 0.75
DEFAULT_DROP_GAP_M = 0.04
DEFAULT_EPISODE_MB = 20.0


def quat_wxyz_from_matrix(rotation: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to MuJoCo's wxyz quaternion order."""
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError("Stable-pose cataloging requires scipy.") from exc
    q_xyzw = Rotation.from_matrix(rotation).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    if q_wxyz[0] < 0:
        q_wxyz *= -1
    return q_wxyz.astype(float).tolist()


def load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Could not load a triangle mesh: {path}")
    return mesh


def read_body_center_and_compile(xml_path: Path, object_name: str) -> tuple[list[float], str | None]:
    """Read the inertial center and ask MuJoCo to compile the standalone object."""
    root = ET.parse(xml_path).getroot()
    body = root.find(f".//body[@name='{object_name}']")
    if body is None:
        body = root.find(".//worldbody/body")
    if body is None:
        return [0.0, 0.0, 0.0], "XML has no object body"

    inertial = body.find("inertial")
    center = [0.0, 0.0, 0.0]
    if inertial is not None and inertial.get("pos"):
        center = [float(v) for v in inertial.get("pos").split()]

    compile_error = None
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.get("name", object_name))
        if body_id >= 0:
            center = np.asarray(model.body_ipos[body_id], dtype=float).tolist()
    except Exception as exc:  # retain useful geometry info even if XML compile fails
        compile_error = f"{type(exc).__name__}: {exc}"
    return center, compile_error


def referenced_collision_meshes(object_dir: Path, xml_path: Path, object_name: str) -> list[Path]:
    root = ET.parse(xml_path).getroot()
    body = root.find(f".//body[@name='{object_name}']")
    if body is None:
        body = root.find(".//worldbody/body")
    assets = root.find("asset")
    if body is None or assets is None:
        return []
    mesh_files = {mesh.get("name"): mesh.get("file", "") for mesh in assets.findall("mesh")}
    resolved = []
    for geom in body.findall("geom"):
        if "visual" in geom.get("name", ""):
            continue
        filename = mesh_files.get(geom.get("mesh"), "")
        if "coacd" not in filename:
            continue
        path = (object_dir.parent / filename).resolve()
        if not path.is_file():
            path = (object_dir / Path(filename).name).resolve()
        if path.is_file():
            resolved.append(path)
    return resolved


def stable_pose_records(mesh, top_n: int, table_top_m: float, drop_gap_m: float):
    """Return upright plus diverse high-probability resting rotations."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    identity_lowest_z = float(np.min(vertices[:, 2]))
    records = [{
        "pose_id": "upright",
        "source": "identity",
        "probability": None,
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "initial_body_z_m": table_top_m + drop_gap_m - identity_lowest_z,
    }]
    try:
        transforms, probabilities = mesh.compute_stable_poses(
            n_samples=max(48, top_n * 24),
            sigma=0.0,
            threshold=0.0,
        )
    except Exception as exc:
        records[0]["stable_pose_error"] = f"{type(exc).__name__}: {exc}"
        return records

    accepted: list[np.ndarray] = [np.eye(3)]
    chosen = 0
    for rank, (transform, probability) in enumerate(zip(transforms, probabilities)):
        rotation = np.asarray(transform[:3, :3], dtype=float)
        if any(abs(float(np.dot(quat_wxyz_from_matrix(rotation), quat_wxyz_from_matrix(prev)))) > 0.995
               for prev in accepted):
            continue
        accepted.append(rotation)
        q = quat_wxyz_from_matrix(rotation)
        lowest_z = float(np.min((rotation @ vertices.T)[2]))
        chosen += 1
        records.append({
            "pose_id": f"stable_{chosen:02d}",
            "source": "trimesh_compute_stable_poses",
            "rank": int(rank),
            "probability": float(probability),
            "quat_wxyz": q,
            "initial_body_z_m": table_top_m + drop_gap_m - lowest_z,
        })
        if chosen >= top_n:
            break
    return records


def count_grid(extents: np.ndarray, spacing: float, radial_cap: float, axial_cap: float) -> dict:
    # Match the current dense sweep's proportions while adapting to each object.
    radial = min(radial_cap, 0.44 * float(min(extents[0], extents[1])))
    axial = min(axial_cap, 0.30 * float(extents[2]))
    xy = np.arange(-radial, radial + spacing / 2, spacing)
    zz = np.arange(-axial, axial + spacing / 2, spacing)
    xy_count = sum(x * x + y * y <= radial * radial + 1e-12 for x in xy for y in xy)
    return {
        "radial_limit_m": radial,
        "axial_limit_m": axial,
        "xy_center_count": int(xy_count),
        "z_center_count": int(len(zz)),
        "center_count_per_pose": int(xy_count * len(zz)),
    }


def audit_object(object_dir: Path, args) -> dict:
    name = object_dir.name
    xml_path = object_dir.parent / f"{name}.xml"
    visual_path = object_dir / "textured.obj"
    collision_paths = (
        referenced_collision_meshes(object_dir, xml_path, name)
        if xml_path.exists() else []
    )
    row = {
        "object": name,
        "object_dir": str(object_dir.resolve()),
        "xml": str(xml_path.resolve()) if xml_path.exists() else None,
        "visual_mesh": str(visual_path.resolve()) if visual_path.exists() else None,
        "collision_meshes": [str(p.resolve()) for p in collision_paths],
        "valid": False,
        "errors": [],
    }
    if not xml_path.exists():
        row["errors"].append("missing object XML")
    if not visual_path.exists():
        row["errors"].append("missing textured.obj")
    if not collision_paths:
        row["errors"].append("object XML has no resolvable CoACD collision mesh references")
    if row["errors"]:
        return row

    try:
        visual = load_mesh(visual_path)
        collisions = [load_mesh(p) for p in collision_paths]
        import trimesh

        collision = trimesh.util.concatenate(collisions)
        extents = np.asarray(collision.extents, dtype=float)
        bounds = np.asarray(collision.bounds, dtype=float)
        center, compile_error = read_body_center_and_compile(xml_path, name)
        row.update({
            "visual_vertices": int(len(visual.vertices)),
            "visual_faces": int(len(visual.faces)),
            "collision_part_count": len(collision_paths),
            "collision_extents_m": extents.tolist(),
            "collision_extents_mm": (extents * 1000).tolist(),
            "collision_bounds_body_m": bounds.tolist(),
            "center_local_m": center,
            "center_source": "compiled MuJoCo body inertial center",
            "compile_error": compile_error,
            "grid": count_grid(extents, args.spacing, args.radial_cap, args.axial_cap),
            "stable_poses": stable_pose_records(
                collision, args.stable_poses, args.table_top, args.drop_gap
            ),
            "valid": True,
        })
    except Exception as exc:
        row["errors"].append(f"{type(exc).__name__}: {exc}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Directory containing per-object folders and <object>.xml files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stable-poses", type=int, default=3,
                        help="Number of non-identity stable poses to keep per object")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M)
    parser.add_argument("--radial-cap", type=float, default=DEFAULT_RADIAL_CAP_M)
    parser.add_argument("--axial-cap", type=float, default=DEFAULT_AXIAL_CAP_M)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--rolls", type=float, nargs="+", default=DEFAULT_ROLLS_DEG)
    parser.add_argument("--table-top", type=float, default=DEFAULT_TABLE_TOP_M)
    parser.add_argument("--drop-gap", type=float, default=DEFAULT_DROP_GAP_M)
    parser.add_argument("--episode-mb", type=float, default=DEFAULT_EPISODE_MB,
                        help="Storage estimate based on sampled episode files")
    parser.add_argument("--max-centers-per-pose", type=int, default=0,
                        help="0 keeps full grid; positive value caps the plan estimate")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"YCB object root does not exist: {args.root}")
    if args.spacing <= 0 or args.repeats <= 0 or args.stable_poses < 0:
        parser.error("spacing/repeats must be positive and stable-poses nonnegative")

    object_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
    records = [audit_object(p, args) for p in object_dirs]
    valid = [r for r in records if r.get("valid")]
    invalid = [r for r in records if not r.get("valid")]

    dense_episodes = 0
    for row in valid:
        poses_n = len(row["stable_poses"])
        centers = row["grid"]["center_count_per_pose"]
        if args.max_centers_per_pose:
            centers = min(centers, args.max_centers_per_pose)
        row["planned_centers_per_pose"] = centers
        row["planned_episodes"] = int(poses_n * centers * len(args.rolls) * args.repeats)
        dense_episodes += row["planned_episodes"]

    summary = {
        "ycb_root": str(args.root.resolve()),
        "object_count_seen": len(object_dirs),
        "valid_object_count": len(valid),
        "invalid_object_count": len(invalid),
        "spacing_m": args.spacing,
        "rolls_deg": list(args.rolls),
        "repeats": args.repeats,
        "stable_pose_count_includes_identity": True,
        "max_centers_per_pose": args.max_centers_per_pose or None,
        "estimated_episode_count": dense_episodes,
        "estimated_storage_gb_decimal": round(dense_episodes * args.episode_mb / 1000, 2),
        "episode_size_assumption_mb": args.episode_mb,
        "storage_estimate_note": "Use actual files from the first completed batch to refine this estimate.",
        "objects": records,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "ycb_tactile_catalog.json"
    csv_path = args.output / "ycb_tactile_catalog.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["object", "valid", "collision_part_count", "collision_extents_mm",
               "center_local_m", "stable_pose_count", "center_count_per_pose",
               "planned_episodes", "compile_error", "errors"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "object": row["object"],
                "valid": row["valid"],
                "collision_part_count": row.get("collision_part_count", 0),
                "collision_extents_mm": json.dumps(row.get("collision_extents_mm")),
                "center_local_m": json.dumps(row.get("center_local_m")),
                "stable_pose_count": len(row.get("stable_poses", [])),
                "center_count_per_pose": row.get("grid", {}).get("center_count_per_pose"),
                "planned_episodes": row.get("planned_episodes", 0),
                "compile_error": row.get("compile_error"),
                "errors": "; ".join(row.get("errors", [])),
            })

    print(f"YCB objects: {len(object_dirs)}; valid: {len(valid)}; invalid: {len(invalid)}")
    print(f"Planned episodes: {dense_episodes:,}")
    print(f"Estimated storage: {summary['estimated_storage_gb_decimal']:,.2f} GB "
          f"({args.episode_mb:g} MB/episode)")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    if invalid:
        print("Invalid models:", ", ".join(row["object"] for row in invalid))
    compile_failures = [row for row in valid if row.get("compile_error")]
    if compile_failures:
        print(f"MuJoCo compile warnings: {len(compile_failures)}; see CSV/JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
