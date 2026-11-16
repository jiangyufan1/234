# -*- coding: utf-8 -*-
r"""
generate_grasp_candidates_from_cad.py

第一阶段：只做几何候选生成，不碰 UR3、不做 MuJoCo 闭合、不做 Taxim。

输入：
    1) gripper_envelope/envelope.npz
       —— 由 analyze_gripper_tactile_envelope_v2.py 生成
    2) YCB CAD：
       object/ycb/tomato_soup_can/textured.obj

输出：
    grasp_candidates/tomato_soup_can/
        candidates_top50.npz
        candidates_top50.csv
        candidates_top50.json
        candidate_001.png
        ...
        top_candidates_overview.png
        cad_surface_cache.npz

核心思想：
    对每个有效夹爪模板 alpha：
        T_alpha = {(s_i, n_i)} i=1..3

    任选一根 GelSight 作为 anchor，
    将其法向对齐到某个 CAD 表面点的“内法向” -n_cad，
    再绕该法向旋转 psi。

    这样得到一个候选夹爪姿态：
        ^O T_G

    然后检查另外两根 GelSight：
        - 位置离 CAD 表面是否足够近
        - 法向是否和 CAD 外法向相对

    几何匹配通过的候选再排序、SE(3) 去重，保存 Top-N。

注意：
    这里得到的是 G_c（geometric grasp candidate），不是 G*。
    后面还必须经过：
        IK -> MuJoCo真实闭合 -> three-contact -> GWS -> lift test
    才能成为有效目标抓取 G*。

运行：
    conda activate mujoco
    cd E:\project\p1\deepgelsight
    python generate_grasp_candidates_from_cad.py

也支持：
    python generate_grasp_candidates_from_cad.py --object tomato_soup_can
"""

from pathlib import Path
import argparse
import csv
import json
import math
import sys

import numpy as np

try:
    import trimesh
except ImportError:
    raise ImportError(
        "缺少 trimesh。请在 mujoco 环境安装：\n"
        "pip install trimesh"
    )

try:
    from scipy.spatial import cKDTree
except ImportError:
    raise ImportError(
        "缺少 scipy。请在 mujoco 环境安装：\n"
        "pip install scipy"
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# 默认参数
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_OBJECT = "tomato_soup_can"

ENVELOPE_PATH = PROJECT_ROOT / "gripper_envelope" / "envelope.npz"

# YCB object path:
# E:\project\p1\deepgelsight\object\ycb\<object>\textured.obj
YCB_ROOT = PROJECT_ROOT / "object" / "ycb"

OUTPUT_ROOT = PROJECT_ROOT / "grasp_candidates"

# ----- CAD sampling -----
# 用较密的 CAD surface cloud 做最近表面匹配。
MATCH_SURFACE_POINTS = 30000

# 从密集表面点中做 farthest-point sampling，作为 anchor seeds。
SEED_COUNT = 240

# ----- gripper template sampling -----
# envelope 原始步长 0.5°，候选阶段不需要每一个都用。
# 每隔约 2.5° 取一个模板。
ALPHA_TEMPLATE_STEP_DEG = 2.5

# 绕 anchor 法向的 roll / psi 采样。
PSI_STEP_DEG = 15.0

# ----- 几何候选阈值 -----
# 第一版故意偏“高召回率”，后面还有真实物理验证。
MAX_SURFACE_DISTANCE_MM = 5.0
MAX_NORMAL_ERROR_DEG = 40.0

# anchor 本身严格对齐，因此主要约束另外两指；
# 但代码仍统一检查三指。
MIN_MATCHED_CONTACT_SEPARATION_MM = 8.0

# ----- candidate score -----
# score 越小越好。
# 位置误差和法向误差分别按阈值归一化。
WEIGHT_MEAN_DIST = 0.40
WEIGHT_MAX_DIST = 0.25
WEIGHT_MEAN_ANGLE = 0.20
WEIGHT_MAX_ANGLE = 0.15

# ----- output -----
TOP_N = 50
VISUALIZE_TOP_N = 12

# SE(3) greedy 去重阈值。
DEDUP_TRANSLATION_MM = 5.0
DEDUP_ROTATION_DEG = 8.0
DEDUP_ALPHA_DEG = 2.5

# 画图时最多显示多少 CAD 点。
VIS_CAD_POINTS = 6000

RANDOM_SEED = 20260909


# ============================================================================
# 数学工具
# ============================================================================

def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("遇到零长度向量。")
    return v / n


def skew(v):
    x, y, z = v
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def axis_angle_rotation(axis, angle_rad):
    axis = normalize(axis)
    K = skew(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def rotation_align_vector(a, b):
    """
    求 R，使 R @ a = b。
    """
    a = normalize(a)
    b = normalize(b)

    c = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if c > 1.0 - 1e-10:
        return np.eye(3)

    if c < -1.0 + 1e-10:
        # 180°：找一个与 a 正交的稳定旋转轴。
        basis = np.eye(3)
        k = int(np.argmin(np.abs(basis @ a)))
        axis = normalize(np.cross(a, basis[k]))
        return axis_angle_rotation(axis, math.pi)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = skew(v)

    return (
        np.eye(3)
        + K
        + K @ K * ((1.0 - c) / (s * s))
    )


def rotation_distance_deg(R1, R2):
    R = R1.T @ R2
    c = float(
        np.clip(
            (np.trace(R) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
    )
    return math.degrees(math.acos(c))


def pose_matrix(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def min_pairwise_distance(P):
    d01 = np.linalg.norm(P[0] - P[1])
    d02 = np.linalg.norm(P[0] - P[2])
    d12 = np.linalg.norm(P[1] - P[2])
    return float(min(d01, d02, d12))


# ============================================================================
# CAD
# ============================================================================

def load_mesh(mesh_path):
    loaded = trimesh.load(
        str(mesh_path),
        force="mesh",
        process=False,
    )

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError(f"CAD Scene 中没有 geometry：{mesh_path}")
        loaded = trimesh.util.concatenate(
            tuple(g for g in loaded.geometry.values())
        )

    mesh = loaded.copy()
    mesh.remove_unreferenced_vertices()

    # 尽量修正 mesh 面朝向；不会改变顶点坐标。
    try:
        mesh.fix_normals(multibody=True)
    except TypeError:
        try:
            mesh.fix_normals()
        except Exception:
            pass
    except Exception:
        pass

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"CAD mesh 为空：{mesh_path}")

    return mesh


def sample_surface_with_normals(mesh, count, rng):
    """
    trimesh.sample.sample_surface 使用 numpy 全局 RNG。
    为可复现，临时设置 numpy seed。
    """
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))

    try:
        points, face_idx = trimesh.sample.sample_surface(
            mesh,
            count,
        )
    finally:
        np.random.set_state(state)

    normals = mesh.face_normals[face_idx].copy()

    # 避免异常零法向。
    n = np.linalg.norm(normals, axis=1)
    good = n > 1e-12
    points = points[good]
    normals = normals[good]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return points.astype(np.float64), normals.astype(np.float64)


def farthest_point_sampling(points, n_samples, rng):
    """
    O(N*K) 的简单 FPS。
    N≈30000, K≈240，第一次运行几秒是正常的。
    """
    points = np.asarray(points, dtype=np.float64)

    n_samples = min(int(n_samples), len(points))

    selected = np.empty(n_samples, dtype=np.int64)

    # 第一个 seed 随机。
    selected[0] = int(rng.integers(0, len(points)))

    min_dist2 = np.sum(
        (points - points[selected[0]]) ** 2,
        axis=1,
    )

    for k in range(1, n_samples):
        idx = int(np.argmax(min_dist2))
        selected[k] = idx

        d2 = np.sum(
            (points - points[idx]) ** 2,
            axis=1,
        )

        min_dist2 = np.minimum(min_dist2, d2)

    return selected


def load_or_build_cad_cache(
    mesh,
    output_dir,
    rng,
):
    cache_path = output_dir / "cad_surface_cache.npz"

    if cache_path.exists():
        data = np.load(cache_path)

        if (
            "surface_points" in data
            and "surface_normals" in data
            and "seed_indices" in data
            and len(data["surface_points"]) == MATCH_SURFACE_POINTS
        ):
            print("[CAD] 使用已有 surface cache：", cache_path)
            return (
                data["surface_points"],
                data["surface_normals"],
                data["seed_indices"],
            )

    print(
        f"[CAD] 从 textured.obj 采样 {MATCH_SURFACE_POINTS} 个表面点 ..."
    )

    P, N = sample_surface_with_normals(
        mesh,
        MATCH_SURFACE_POINTS,
        rng,
    )

    print(
        f"[CAD] Farthest-point sampling {SEED_COUNT} 个 anchor seeds ..."
    )

    seed_idx = farthest_point_sampling(
        P,
        SEED_COUNT,
        rng,
    )

    np.savez_compressed(
        cache_path,
        surface_points=P,
        surface_normals=N,
        seed_indices=seed_idx,
    )

    return P, N, seed_idx


# ============================================================================
# Envelope templates
# ============================================================================

def select_template_indices(alpha_deg, valid_mask):
    valid_idx = np.flatnonzero(valid_mask)

    if len(valid_idx) == 0:
        raise RuntimeError(
            "envelope.npz 中没有有效 alpha。"
            "请确认使用的是修正后的 envelope v2。"
        )

    chosen = []
    last_alpha = None

    for idx in valid_idx:
        a = float(alpha_deg[idx])

        if (
            last_alpha is None
            or a - last_alpha >= ALPHA_TEMPLATE_STEP_DEG - 1e-9
        ):
            chosen.append(int(idx))
            last_alpha = a

    # 确保最后一个有效 alpha 也被覆盖。
    if chosen[-1] != int(valid_idx[-1]):
        chosen.append(int(valid_idx[-1]))

    return np.asarray(chosen, dtype=np.int64)


def load_templates(envelope_path):
    env = np.load(envelope_path)

    required = [
        "alpha_deg",
        "surface_center_tool",
        "normal_tool",
    ]

    for key in required:
        if key not in env:
            raise RuntimeError(
                f"envelope.npz 缺少字段：{key}"
            )

    alpha = env["alpha_deg"].astype(np.float64)
    centers = env["surface_center_tool"].astype(np.float64)
    normals = env["normal_tool"].astype(np.float64)

    # 优先使用 safe_clearance；没有则使用 collision_free。
    if "safe_clearance" in env and np.any(env["safe_clearance"]):
        valid_mask = env["safe_clearance"].astype(bool)
        valid_source = "safe_clearance"
    elif "collision_free" in env:
        valid_mask = env["collision_free"].astype(bool)
        valid_source = "collision_free"
    else:
        valid_mask = np.ones(len(alpha), dtype=bool)
        valid_source = "all"

    idx = select_template_indices(
        alpha,
        valid_mask,
    )

    print(
        f"[Envelope] 有效来源：{valid_source}"
    )
    print(
        f"[Envelope] 使用 {len(idx)} 个 alpha templates："
    )
    print(
        " ",
        [float(x) for x in alpha[idx]],
    )

    return {
        "alpha_deg": alpha[idx],
        "centers_tool": centers[idx],
        "normals_tool": normals[idx],
        "source_indices": idx,
        "valid_source": valid_source,
    }


# ============================================================================
# Candidate generation
# ============================================================================

def score_candidates(
    dist_m,
    angle_deg,
):
    dist_norm = dist_m / (MAX_SURFACE_DISTANCE_MM / 1000.0)
    angle_norm = angle_deg / MAX_NORMAL_ERROR_DEG

    mean_dist = np.mean(dist_norm, axis=1)
    max_dist = np.max(dist_norm, axis=1)

    mean_ang = np.mean(angle_norm, axis=1)
    max_ang = np.max(angle_norm, axis=1)

    return (
        WEIGHT_MEAN_DIST * mean_dist
        + WEIGHT_MAX_DIST * max_dist
        + WEIGHT_MEAN_ANGLE * mean_ang
        + WEIGHT_MAX_ANGLE * max_ang
    )


def generate_candidates(
    templates,
    surface_points,
    surface_normals,
    seed_indices,
):
    tree = cKDTree(surface_points)

    psi_deg = np.arange(
        0.0,
        360.0,
        PSI_STEP_DEG,
        dtype=np.float64,
    )
    psi_rad = np.radians(psi_deg)

    candidates = []

    total_blocks = (
        len(templates["alpha_deg"])
        * 3
        * len(seed_indices)
    )
    done_blocks = 0

    print("\n开始 Envelope-CAD 三点模板匹配：")
    print(
        f"  alpha templates = {len(templates['alpha_deg'])}"
    )
    print(
        f"  anchor seeds     = {len(seed_indices)}"
    )
    print("  anchor fingers   = 3")
    print(
        f"  psi samples      = {len(psi_deg)} "
        f"(step={PSI_STEP_DEG:.1f}°)"
    )
    print(
        f"  raw hypotheses   ≈ "
        f"{len(templates['alpha_deg']) * len(seed_indices) * 3 * len(psi_deg):,}"
    )
    print(
        f"  distance threshold = {MAX_SURFACE_DISTANCE_MM:.1f} mm"
    )
    print(
        f"  normal threshold   = {MAX_NORMAL_ERROR_DEG:.1f}°"
    )

    for ti, alpha_deg in enumerate(
        templates["alpha_deg"]
    ):
        S = templates["centers_tool"][ti]   # (3,3)
        N = templates["normals_tool"][ti]   # (3,3)

        for anchor in range(3):
            s_anchor = S[anchor]
            n_anchor = N[anchor]

            for seed_rank, seed_idx in enumerate(seed_indices):
                q_anchor = surface_points[seed_idx]
                n_cad_anchor = surface_normals[seed_idx]

                # GelSight active normal 应与 CAD 外法向相反。
                target_n = -n_cad_anchor

                R_align = rotation_align_vector(
                    n_anchor,
                    target_n,
                )

                # 一次性生成所有 psi 候选。
                Rs = np.empty(
                    (len(psi_rad), 3, 3),
                    dtype=np.float64,
                )

                for pi, psi in enumerate(psi_rad):
                    Rs[pi] = (
                        axis_angle_rotation(target_n, float(psi))
                        @ R_align
                    )

                # anchor 精确落到 q_anchor：
                # t = q_anchor - R s_anchor
                anchor_rot = np.einsum(
                    "kij,j->ki",
                    Rs,
                    s_anchor,
                )

                ts = q_anchor[None, :] - anchor_rot

                # 预测的三个 GelSight surface centers
                pred_P = (
                    np.einsum(
                        "kij,fj->kfi",
                        Rs,
                        S,
                    )
                    + ts[:, None, :]
                )

                pred_N = np.einsum(
                    "kij,fj->kfi",
                    Rs,
                    N,
                )

                # 查询 CAD 最近表面点
                flat = pred_P.reshape(-1, 3)

                d_flat, nn_idx_flat = tree.query(
                    flat,
                    k=1,
                )

                d = d_flat.reshape(
                    len(psi_rad),
                    3,
                )

                nn_idx = nn_idx_flat.reshape(
                    len(psi_rad),
                    3,
                )

                match_P = surface_points[nn_idx]
                cad_N = surface_normals[nn_idx]

                # 理想：pred_N = -cad_N
                dots = np.sum(
                    pred_N * (-cad_N),
                    axis=2,
                )
                dots = np.clip(dots, -1.0, 1.0)

                angles = np.degrees(np.arccos(dots))

                valid = (
                    np.max(d, axis=1)
                    <= MAX_SURFACE_DISTANCE_MM / 1000.0
                ) & (
                    np.max(angles, axis=1)
                    <= MAX_NORMAL_ERROR_DEG
                )

                if np.any(valid):
                    # 最近 CAD 接触点不能退化成几乎同一点。
                    valid_idx = np.flatnonzero(valid)

                    keep = []

                    for pi in valid_idx:
                        sep = min_pairwise_distance(
                            match_P[pi]
                        )

                        if (
                            sep
                            >= MIN_MATCHED_CONTACT_SEPARATION_MM / 1000.0
                        ):
                            keep.append(int(pi))

                    if keep:
                        keep = np.asarray(keep, dtype=np.int64)

                        scores = score_candidates(
                            d[keep],
                            angles[keep],
                        )

                        for local_j, pi in enumerate(keep):
                            candidates.append(
                                {
                                    "score": float(scores[local_j]),
                                    "alpha_deg": float(alpha_deg),
                                    "anchor_finger": int(anchor),
                                    "seed_rank": int(seed_rank),
                                    "seed_surface_index": int(seed_idx),
                                    "psi_deg": float(psi_deg[pi]),
                                    "R_object_tool": Rs[pi].copy(),
                                    "t_object_tool": ts[pi].copy(),
                                    "pred_contact_object": pred_P[pi].copy(),
                                    "matched_contact_object": match_P[pi].copy(),
                                    "gel_normal_object": pred_N[pi].copy(),
                                    "cad_normal_object": cad_N[pi].copy(),
                                    "distance_m": d[pi].copy(),
                                    "angle_deg": angles[pi].copy(),
                                    "min_contact_separation_m": float(
                                        min_pairwise_distance(match_P[pi])
                                    ),
                                }
                            )

                done_blocks += 1

                if (
                    done_blocks == 1
                    or done_blocks == total_blocks
                    or done_blocks % 1000 == 0
                ):
                    print(
                        f"  progress "
                        f"{done_blocks:6d}/{total_blocks:6d} | "
                        f"passing={len(candidates):6d}"
                    )

    print(
        f"\n几何阈值通过的 raw candidates：{len(candidates)}"
    )

    return candidates


# ============================================================================
# SE(3) dedup
# ============================================================================

def greedy_deduplicate(candidates, top_n):
    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
    )

    selected = []

    trans_thr = DEDUP_TRANSLATION_MM / 1000.0

    for cand in candidates:
        duplicate = False

        for keep in selected:
            dt = np.linalg.norm(
                cand["t_object_tool"]
                - keep["t_object_tool"]
            )

            if dt > trans_thr:
                continue

            da = abs(
                cand["alpha_deg"]
                - keep["alpha_deg"]
            )

            if da > DEDUP_ALPHA_DEG:
                continue

            dr = rotation_distance_deg(
                cand["R_object_tool"],
                keep["R_object_tool"],
            )

            if dr <= DEDUP_ROTATION_DEG:
                duplicate = True
                break

        if not duplicate:
            selected.append(cand)

        if len(selected) >= top_n:
            break

    return selected


# ============================================================================
# 保存
# ============================================================================

def save_candidates(candidates, output_dir, object_name):
    if not candidates:
        raise RuntimeError(
            "没有候选可保存。请先看控制台的匹配阈值和 CAD 法向。"
        )

    N = len(candidates)

    T = np.stack(
        [
            pose_matrix(
                c["R_object_tool"],
                c["t_object_tool"],
            )
            for c in candidates
        ]
    )

    npz_path = output_dir / f"candidates_top{N}.npz"

    np.savez_compressed(
        npz_path,
        object_name=np.array(object_name),
        T_object_tool=T,
        score=np.asarray(
            [c["score"] for c in candidates]
        ),
        alpha_deg=np.asarray(
            [c["alpha_deg"] for c in candidates]
        ),
        anchor_finger=np.asarray(
            [c["anchor_finger"] for c in candidates],
            dtype=np.int32,
        ),
        psi_deg=np.asarray(
            [c["psi_deg"] for c in candidates]
        ),
        distance_m=np.stack(
            [c["distance_m"] for c in candidates]
        ),
        normal_error_deg=np.stack(
            [c["angle_deg"] for c in candidates]
        ),
        pred_contact_object=np.stack(
            [c["pred_contact_object"] for c in candidates]
        ),
        matched_contact_object=np.stack(
            [c["matched_contact_object"] for c in candidates]
        ),
        gel_normal_object=np.stack(
            [c["gel_normal_object"] for c in candidates]
        ),
        cad_normal_object=np.stack(
            [c["cad_normal_object"] for c in candidates]
        ),
        min_contact_separation_m=np.asarray(
            [c["min_contact_separation_m"] for c in candidates]
        ),
    )

    csv_path = output_dir / f"candidates_top{N}.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "rank",
                "score",
                "alpha_deg",
                "anchor_finger",
                "psi_deg",
                "mean_distance_mm",
                "max_distance_mm",
                "mean_normal_error_deg",
                "max_normal_error_deg",
                "min_contact_separation_mm",
                "tool_x_m",
                "tool_y_m",
                "tool_z_m",
            ]
        )

        for rank, c in enumerate(candidates, start=1):
            writer.writerow(
                [
                    rank,
                    c["score"],
                    c["alpha_deg"],
                    c["anchor_finger"] + 1,
                    c["psi_deg"],
                    float(np.mean(c["distance_m"]) * 1000.0),
                    float(np.max(c["distance_m"]) * 1000.0),
                    float(np.mean(c["angle_deg"])),
                    float(np.max(c["angle_deg"])),
                    c["min_contact_separation_m"] * 1000.0,
                    c["t_object_tool"][0],
                    c["t_object_tool"][1],
                    c["t_object_tool"][2],
                ]
            )

    json_path = output_dir / f"candidates_top{N}.json"

    serial = {
        "object": object_name,
        "num_candidates": N,
        "candidate_definition": "geometric G_c, NOT validated G*",
        "thresholds": {
            "max_surface_distance_mm": MAX_SURFACE_DISTANCE_MM,
            "max_normal_error_deg": MAX_NORMAL_ERROR_DEG,
            "min_matched_contact_separation_mm": (
                MIN_MATCHED_CONTACT_SEPARATION_MM
            ),
            "dedup_translation_mm": DEDUP_TRANSLATION_MM,
            "dedup_rotation_deg": DEDUP_ROTATION_DEG,
            "dedup_alpha_deg": DEDUP_ALPHA_DEG,
        },
        "candidates": [],
    }

    for rank, c in enumerate(candidates, start=1):
        serial["candidates"].append(
            {
                "rank": rank,
                "score": c["score"],
                "alpha_deg": c["alpha_deg"],
                "anchor_finger": c["anchor_finger"] + 1,
                "psi_deg": c["psi_deg"],
                "T_object_tool": pose_matrix(
                    c["R_object_tool"],
                    c["t_object_tool"],
                ).tolist(),
                "distance_mm": (
                    c["distance_m"] * 1000.0
                ).tolist(),
                "normal_error_deg": c["angle_deg"].tolist(),
                "pred_contact_object": (
                    c["pred_contact_object"].tolist()
                ),
                "matched_contact_object": (
                    c["matched_contact_object"].tolist()
                ),
            }
        )

    json_path.write_text(
        json.dumps(
            serial,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return npz_path, csv_path, json_path


# ============================================================================
# 可视化
# ============================================================================

def set_equal_3d(ax, points):
    P = np.asarray(points, dtype=np.float64)

    lo = P.min(axis=0)
    hi = P.max(axis=0)

    center = 0.5 * (lo + hi)
    half = max(
        0.5 * float(np.max(hi - lo)),
        1e-4,
    )

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def choose_visual_cad_points(surface_points, rng):
    if len(surface_points) <= VIS_CAD_POINTS:
        return surface_points

    idx = rng.choice(
        len(surface_points),
        size=VIS_CAD_POINTS,
        replace=False,
    )

    return surface_points[idx]


def draw_tool_axes(ax, R, t, axis_len):
    origin = t

    for j in range(3):
        v = R[:, j]

        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            v[0],
            v[1],
            v[2],
            length=axis_len,
            normalize=True,
        )


def plot_one_candidate(
    cand,
    cad_vis,
    rank,
    output_dir,
):
    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    # 单位统一为 mm，便于看误差。
    CAD = cad_vis * 1000.0
    P = cand["pred_contact_object"] * 1000.0
    Q = cand["matched_contact_object"] * 1000.0
    Ng = cand["gel_normal_object"]
    Nc = cand["cad_normal_object"]

    ax.scatter(
        CAD[:, 0],
        CAD[:, 1],
        CAD[:, 2],
        s=1,
        alpha=0.10,
    )

    # 预测的 GelSight surface center
    ax.scatter(
        P[:, 0],
        P[:, 1],
        P[:, 2],
        s=55,
        marker="o",
        label="GelSight centers",
    )

    # 最近 CAD surface points
    ax.scatter(
        Q[:, 0],
        Q[:, 1],
        Q[:, 2],
        s=55,
        marker="x",
        label="matched CAD",
    )

    # 匹配误差连线
    for i in range(3):
        ax.plot(
            [P[i, 0], Q[i, 0]],
            [P[i, 1], Q[i, 1]],
            [P[i, 2], Q[i, 2]],
            linewidth=1.5,
        )

    # GelSight normal
    ax.quiver(
        P[:, 0],
        P[:, 1],
        P[:, 2],
        Ng[:, 0],
        Ng[:, 1],
        Ng[:, 2],
        length=14.0,
        normalize=True,
    )

    # CAD outward normal
    ax.quiver(
        Q[:, 0],
        Q[:, 1],
        Q[:, 2],
        Nc[:, 0],
        Nc[:, 1],
        Nc[:, 2],
        length=10.0,
        normalize=True,
        alpha=0.6,
    )

    draw_tool_axes(
        ax,
        cand["R_object_tool"],
        cand["t_object_tool"] * 1000.0,
        axis_len=22.0,
    )

    all_pts = np.concatenate(
        [
            CAD,
            P,
            Q,
            cand["t_object_tool"][None, :] * 1000.0,
        ],
        axis=0,
    )

    set_equal_3d(ax, all_pts)

    ax.set_xlabel("Object X (mm)")
    ax.set_ylabel("Object Y (mm)")
    ax.set_zlabel("Object Z (mm)")

    ax.set_title(
        f"Candidate #{rank} | "
        f"score={cand['score']:.3f} | "
        f"alpha={cand['alpha_deg']:.1f}° | "
        f"psi={cand['psi_deg']:.0f}° | "
        f"anchor=F{cand['anchor_finger']+1}"
    )

    ax.legend(loc="upper right")

    fig.tight_layout()

    path = output_dir / f"candidate_{rank:03d}.png"

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(fig)


def plot_overview(
    candidates,
    cad_vis,
    output_dir,
):
    show = candidates[: min(12, len(candidates))]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    CAD = cad_vis * 1000.0

    ax.scatter(
        CAD[:, 0],
        CAD[:, 1],
        CAD[:, 2],
        s=1,
        alpha=0.08,
    )

    all_pts = [CAD]

    for c in show:
        P = c["matched_contact_object"] * 1000.0
        all_pts.append(P)

        # 三点组成一个三角形
        cyc = np.vstack([P, P[0]])

        ax.plot(
            cyc[:, 0],
            cyc[:, 1],
            cyc[:, 2],
            linewidth=1.2,
            alpha=0.8,
        )

        ax.scatter(
            P[:, 0],
            P[:, 1],
            P[:, 2],
            s=18,
        )

    all_pts = np.concatenate(all_pts, axis=0)

    set_equal_3d(ax, all_pts)

    ax.set_xlabel("Object X (mm)")
    ax.set_ylabel("Object Y (mm)")
    ax.set_zlabel("Object Z (mm)")
    ax.set_title(
        f"Top {len(show)} geometric grasp candidates"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "top_candidates_overview.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================================
# summary
# ============================================================================

def print_top_summary(candidates):
    print("\nTop candidates:")
    print(
        " rank | score | alpha | psi | anchor | "
        "mean d(mm) | max d(mm) | mean ang | max ang"
    )

    for rank, c in enumerate(
        candidates[: min(15, len(candidates))],
        start=1,
    ):
        print(
            f" {rank:4d} | "
            f"{c['score']:5.3f} | "
            f"{c['alpha_deg']:+5.1f} | "
            f"{c['psi_deg']:5.0f} | "
            f"F{c['anchor_finger']+1} | "
            f"{np.mean(c['distance_m'])*1000:9.3f} | "
            f"{np.max(c['distance_m'])*1000:8.3f} | "
            f"{np.mean(c['angle_deg']):8.2f} | "
            f"{np.max(c['angle_deg']):7.2f}"
        )


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--object",
        default=DEFAULT_OBJECT,
        help="YCB object folder name",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=TOP_N,
        help="Top-N candidates to save",
    )

    args = parser.parse_args()

    object_name = args.object
    top_n = int(args.top)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    object_dir = YCB_ROOT / object_name
    mesh_path = object_dir / "textured.obj"

    output_dir = OUTPUT_ROOT / object_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("Envelope-CAD Three-Point Geometric Grasp Candidate Generator")
    print("=" * 88)

    print("OBJECT:")
    print("  ", object_name)

    print("CAD:")
    print("  ", mesh_path)

    print("ENVELOPE:")
    print("  ", ENVELOPE_PATH)

    print("OUTPUT:")
    print("  ", output_dir)

    if not ENVELOPE_PATH.exists():
        raise FileNotFoundError(
            "找不到 envelope.npz：\n"
            f"{ENVELOPE_PATH}\n"
            "请先运行 analyze_gripper_tactile_envelope_v2.py。"
        )

    if not mesh_path.exists():
        raise FileNotFoundError(
            "找不到 YCB textured.obj：\n"
            f"{mesh_path}"
        )

    # 1. Envelope
    templates = load_templates(
        ENVELOPE_PATH
    )

    # 2. CAD
    mesh = load_mesh(
        mesh_path
    )

    print("\n[CAD mesh]")
    print(
        f"  vertices = {len(mesh.vertices)}"
    )
    print(
        f"  faces    = {len(mesh.faces)}"
    )
    print(
        "  extents(mm) =",
        np.round(mesh.extents * 1000.0, 3),
    )
    print(
        "  bounds(mm)  =\n",
        np.round(mesh.bounds * 1000.0, 3),
    )

    surface_points, surface_normals, seed_idx = (
        load_or_build_cad_cache(
            mesh,
            output_dir,
            rng,
        )
    )

    # 3. candidate generation
    raw_candidates = generate_candidates(
        templates,
        surface_points,
        surface_normals,
        seed_idx,
    )

    if not raw_candidates:
        raise RuntimeError(
            "\n没有候选通过第一轮几何筛选。\n"
            "先不要改程序结构。请把控制台输出发给我，"
            "我们检查 CAD 法向和阈值。"
        )

    # 4. score + SE(3) dedup
    print("\n执行 score 排序 + SE(3) 去重 ...")

    selected = greedy_deduplicate(
        raw_candidates,
        top_n,
    )

    print(
        f"去重后保存 Top {len(selected)} 个 G_c。"
    )

    print_top_summary(
        selected
    )

    # 5. save
    npz_path, csv_path, json_path = save_candidates(
        selected,
        output_dir,
        object_name,
    )

    # 6. visualization
    cad_vis = choose_visual_cad_points(
        surface_points,
        rng,
    )

    print("\n生成 3D 可视化 ...")

    plot_overview(
        selected,
        cad_vis,
        output_dir,
    )

    for rank, cand in enumerate(
        selected[: min(VISUALIZE_TOP_N, len(selected))],
        start=1,
    ):
        plot_one_candidate(
            cand,
            cad_vis,
            rank,
            output_dir,
        )

    print("\n" + "=" * 88)
    print("完成")
    print("=" * 88)

    print("\n保存：")
    print("  ", npz_path)
    print("  ", csv_path)
    print("  ", json_path)
    print(
        "  ",
        output_dir / "top_candidates_overview.png",
    )

    print(
        f"  candidate_001.png ~ "
        f"candidate_{min(VISUALIZE_TOP_N, len(selected)):03d}.png"
    )

    print(
        "\n注意：这些仍然只是几何候选 G_c，不是 G*。"
    )

    print(
        "\n下一步先只检查 candidate_001~012 的三点位置和法向。"
        "\n如果几何可视化正确，再接 MuJoCo 真实闭合。"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户终止。")
    except Exception as exc:
        print("\n[ERROR]")
        print(repr(exc))
        print(
            "\n请把完整控制台输出发给我。"
        )
        sys.exit(1)
