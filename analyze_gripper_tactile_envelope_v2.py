# -*- coding: utf-8 -*-
r"""
analyze_gripper_tactile_envelope.py

目标：
    对当前三指 GelSight 夹爪做一次“数字化运动学标定”，自动得到：

    1) 三根 GelSight 中心轨迹
    2) 三块 GelSight 有效接触面的扫掠区域
    3) 三根接触面法向随关节角 alpha 的变化
    4) 真正的 finger-finger 自碰撞角度
       （finger-base 仅作为诊断量，不参与 collision-free 判定）
    5) collision-free tactile reachable envelope
    6) 等效抓取直径 D(alpha)
    7) 供后续 YCB CAD -> G_c 快速筛选使用的 envelope.npz

运行：
    conda activate mujoco
    cd E:\project\p1\deepgelsight
    python analyze_gripper_tactile_envelope.py

默认输出：
    E:\project\p1\deepgelsight\gripper_envelope\

重要：
    - 不需要物体
    - 不需要 Taxim
    - 不做动力学仿真，只做夹爪几何/运动学扫描
    - 不修改你的原始 XML
    - 自碰撞采用 mujoco.mj_geomDistance() 直接测几何最短距离，
      因此即使 finger*_vis 的 contype/conaffinity=0，也可以分析几何碰撞
"""

from pathlib import Path
import csv
import json
import math
import sys

import numpy as np
import mujoco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# 项目配置
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
XML_DIR = PROJECT_ROOT / "xml"
OUTPUT_DIR = PROJECT_ROOT / "gripper_envelope"

# 如果设为 None，就自动在 xml/ 中寻找合适的 UR3 + threefinger XML。
XML_FILE = None

PREFERRED_XML_FILES = [
    "ur3_with_threefinger_gelsight (2)(1).xml",
    "ur3_with_threefinger_gelsight (2).xml",
    "ur3_with_threefinger_gelsight.xml",
    "ur3_with_threefinger_gripper.xml",
    "ur3_with_threefinger_gelsight_table_cracker.xml",
    "ur3_with_threefinger_gelsight_table_tomato.xml",
]

# 扫描范围：当前 XML 的三个 finger hinge 是 ±30°
ALPHA_MIN_DEG = -30.0
ALPHA_MAX_DEG = +30.0
ALPHA_STEP_DEG = 0.5

# 每个 GelSight 有效面内部采样网格。
# 11×11 足够用于 envelope 可视化和后续几何匹配。
SURFACE_GRID_N = 11

# mj_geomDistance 搜索最大距离。
GEOM_DISTANCE_MAX_M = 0.05

# 实际相交：distance <= COLLISION_TOLERANCE_M
# 允许一点数值误差。
COLLISION_TOLERANCE_M = 1e-6

# 另外报告一个“安全净空”版本。
# 不代表物理碰撞，只用于保守筛选 G_c。
SAFE_CLEARANCE_M = 0.0005  # 0.5 mm

# 画法向箭头时，每隔多少个 alpha 画一次。
NORMAL_PLOT_STRIDE = 10

# ACTIVE_FACE_SIGNS:
#   None -> 程序自动判断每根 GelSight 的 ±local-Z 哪一侧朝向三指抓取中心
#   也可以手工指定，例如 [-1, -1, -1] 或 [1, 1, 1]
ACTIVE_FACE_SIGNS = None


# ============================================================================
# MuJoCo 名称
# ============================================================================

TOOL_SITE = "tool0"

FINGER_JOINTS = [
    "finger1_hinge",
    "finger2_hinge",
    "finger3_hinge",
]

GEL_SITES = [
    "finger1_gelsight_site",
    "finger2_gelsight_site",
    "finger3_gelsight_site",
]

GEL_GEOMS = [
    "finger1_gel",
    "finger2_gel",
    "finger3_gel",
]

FINGER_GEOM_GROUPS = {
    "finger1": [
        "finger1_vis",
        "finger1_gelsight_housing",
        "finger1_gel",
    ],
    "finger2": [
        "finger2_vis",
        "finger2_gelsight_housing",
        "finger2_gel",
    ],
    "finger3": [
        "finger3_vis",
        "finger3_gelsight_housing",
        "finger3_gel",
    ],
}

BASE_GEOMS = [
    "gripper_base_vis",
]

# 只比较“不同刚体组”之间的距离。
# 不比较同一根手指内部的 finger mesh / housing / gel，
# 因为这些部件设计上可能本来就互相重叠。
# 真正用于判定“夹爪是否自碰撞”的几何对。
# 注意：finger ↔ base 在 hinge 安装区域存在设计性的固定重叠，
# 不能用于判定 collision-free。
VALIDATING_COLLISION_GROUP_PAIRS = [
    ("finger1", "finger2"),
    ("finger1", "finger3"),
    ("finger2", "finger3"),
]

# 仅记录、仅诊断，不参与 collision-free / safe-clearance 判定。
DIAGNOSTIC_COLLISION_GROUP_PAIRS = [
    ("finger1", "base"),
    ("finger2", "base"),
    ("finger3", "base"),
]

ALL_COLLISION_GROUP_PAIRS = (
    VALIDATING_COLLISION_GROUP_PAIRS
    + DIAGNOSTIC_COLLISION_GROUP_PAIRS
)


# ============================================================================
# 基础工具
# ============================================================================

def find_source_xml():
    if not XML_DIR.exists():
        raise FileNotFoundError(f"找不到 XML 目录：{XML_DIR}")

    if XML_FILE is not None:
        path = XML_DIR / XML_FILE
        if not path.exists():
            raise FileNotFoundError(f"指定 XML 不存在：{path}")
        return path

    for name in PREFERRED_XML_FILES:
        path = XML_DIR / name
        if path.exists():
            return path

    candidates = sorted(XML_DIR.glob("*.xml"))

    # 只选明显包含 threefinger 的文件。
    for path in candidates:
        lname = path.name.lower()
        if "threefinger" in lname or "three_finger" in lname:
            return path

    raise FileNotFoundError(
        "xml/ 中没有找到 UR3 + three-finger XML。\n"
        "请在脚本顶部设置 XML_FILE。"
    )


def require_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"MuJoCo 模型中找不到：{name}")
    return idx


def optional_geom_id(model, name):
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return idx if idx >= 0 else None


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise RuntimeError("遇到零长度向量。")
    return v / n


def reset_to_home(model, data):
    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)


def set_finger_alpha(model, data, joint_ids, alpha_rad):
    for jid in joint_ids:
        qadr = int(model.jnt_qposadr[jid])
        value = alpha_rad

        if bool(model.jnt_limited[jid]):
            lo, hi = model.jnt_range[jid]
            value = float(np.clip(value, lo, hi))

        data.qpos[qadr] = value

    mujoco.mj_forward(model, data)


def to_tool_point(p_world, tool_pos, tool_R):
    return tool_R.T @ (p_world - tool_pos)


def to_tool_rotation(R_world, tool_R):
    return tool_R.T @ R_world


def contiguous_intervals(alpha_deg, mask):
    """
    将 bool mask 转为连续角度区间。
    """
    intervals = []
    start = None
    prev = None

    for a, valid in zip(alpha_deg, mask):
        if valid and start is None:
            start = float(a)

        if not valid and start is not None:
            intervals.append([start, float(prev)])
            start = None

        prev = a

    if start is not None:
        intervals.append([start, float(prev)])

    return intervals


# ============================================================================
# GelSight 有效面
# ============================================================================

def infer_active_face_signs(model, data, ids):
    """
    在 alpha=0 时自动判断每个 GelSight 的 ±local-Z 中，
    哪一个方向更朝向三指中心。

    sign=+1 -> active face normal = local +Z
    sign=-1 -> active face normal = local -Z
    """
    if ACTIVE_FACE_SIGNS is not None:
        signs = np.asarray(ACTIVE_FACE_SIGNS, dtype=np.int32)
        if signs.shape != (3,):
            raise ValueError("ACTIVE_FACE_SIGNS 必须是长度为3的列表。")
        return signs

    set_finger_alpha(
        model,
        data,
        ids["finger_joint_ids"],
        0.0,
    )

    centers = np.stack(
        [
            data.geom_xpos[gid].copy()
            for gid in ids["gel_geom_ids"]
        ],
        axis=0,
    )

    centroid = centers.mean(axis=0)

    signs = []

    for i, gid in enumerate(ids["gel_geom_ids"]):
        R = data.geom_xmat[gid].reshape(3, 3).copy()

        n_plus = R @ np.array([0.0, 0.0, 1.0])
        toward_center = centroid - centers[i]

        if np.dot(n_plus, toward_center) >= 0:
            signs.append(+1)
        else:
            signs.append(-1)

    return np.asarray(signs, dtype=np.int32)


def gel_surface_geometry_tool(
    model,
    data,
    gel_geom_id,
    active_sign,
    tool_pos,
    tool_R,
):
    """
    返回当前 alpha 下一个 GelSight 的：
      geom center
      active face center
      active face normal
      4 corners
      NxN surface grid
    全部表达在 tool0 frame。
    """
    geom_pos_w = data.geom_xpos[gel_geom_id].copy()
    geom_R_w = data.geom_xmat[gel_geom_id].reshape(3, 3).copy()

    # box geom_size = half extents [hx, hy, hz]
    hx, hy, hz = model.geom_size[gel_geom_id][:3]

    z_face = float(active_sign) * float(hz)

    face_center_local = np.array(
        [0.0, 0.0, z_face],
        dtype=np.float64,
    )

    corners_local = np.array(
        [
            [+hx, +hy, z_face],
            [+hx, -hy, z_face],
            [-hx, -hy, z_face],
            [-hx, +hy, z_face],
        ],
        dtype=np.float64,
    )

    normal_local = np.array(
        [0.0, 0.0, float(active_sign)],
        dtype=np.float64,
    )

    # surface grid
    xs = np.linspace(-hx, +hx, SURFACE_GRID_N)
    ys = np.linspace(-hy, +hy, SURFACE_GRID_N)
    grid_local = np.zeros(
        (SURFACE_GRID_N, SURFACE_GRID_N, 3),
        dtype=np.float64,
    )

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            grid_local[ix, iy] = [x, y, z_face]

    face_center_w = geom_pos_w + geom_R_w @ face_center_local

    corners_w = (
        geom_pos_w[None, :]
        + (geom_R_w @ corners_local.T).T
    )

    grid_flat = grid_local.reshape(-1, 3)
    grid_w = (
        geom_pos_w[None, :]
        + (geom_R_w @ grid_flat.T).T
    ).reshape(SURFACE_GRID_N, SURFACE_GRID_N, 3)

    normal_w = normalize(geom_R_w @ normal_local)

    geom_center_tool = to_tool_point(
        geom_pos_w,
        tool_pos,
        tool_R,
    )

    face_center_tool = to_tool_point(
        face_center_w,
        tool_pos,
        tool_R,
    )

    corners_tool = (
        tool_R.T
        @ (corners_w - tool_pos[None, :]).T
    ).T

    grid_tool = (
        tool_R.T
        @ (grid_w.reshape(-1, 3) - tool_pos[None, :]).T
    ).T.reshape(
        SURFACE_GRID_N,
        SURFACE_GRID_N,
        3,
    )

    normal_tool = normalize(tool_R.T @ normal_w)

    return {
        "geom_center_tool": geom_center_tool,
        "surface_center_tool": face_center_tool,
        "normal_tool": normal_tool,
        "corners_tool": corners_tool,
        "grid_tool": grid_tool,
    }


# ============================================================================
# 自碰撞
# ============================================================================

def resolve_collision_groups(model):
    groups = {}

    for group_name, geom_names in FINGER_GEOM_GROUPS.items():
        ids = []
        existing_names = []

        for name in geom_names:
            gid = optional_geom_id(model, name)

            if gid is not None:
                ids.append(gid)
                existing_names.append(name)

        if not ids:
            raise RuntimeError(
                f"碰撞分析组 {group_name} 中没有找到任何 geom。"
            )

        groups[group_name] = {
            "ids": ids,
            "names": existing_names,
        }

    base_ids = []
    base_names = []

    for name in BASE_GEOMS:
        gid = optional_geom_id(model, name)

        if gid is not None:
            base_ids.append(gid)
            base_names.append(name)

    if not base_ids:
        raise RuntimeError("找不到 gripper base collision-analysis geom。")

    groups["base"] = {
        "ids": base_ids,
        "names": base_names,
    }

    return groups


def geom_distance(model, data, gid1, gid2):
    """
    MuJoCo 直接计算两个 geom 的最短距离。
    返回：
      distance
      fromto[6]
    """
    if not hasattr(mujoco, "mj_geomDistance"):
        raise RuntimeError(
            "当前 mujoco Python 包没有 mj_geomDistance()。\n"
            "你当前项目之前使用的是 MuJoCo 3.x，正常应当支持。\n"
            "请把 `python -c \"import mujoco; print(mujoco.__version__)\"` "
            "的结果发给我。"
        )

    fromto = np.zeros(6, dtype=np.float64)

    dist = mujoco.mj_geomDistance(
        model,
        data,
        int(gid1),
        int(gid2),
        float(GEOM_DISTANCE_MAX_M),
        fromto,
    )

    return float(dist), fromto


def min_group_distance(model, data, group_a, group_b):
    best_dist = float(GEOM_DISTANCE_MAX_M)
    best_pair = None
    best_fromto = None

    for gid_a, name_a in zip(
        group_a["ids"],
        group_a["names"],
    ):
        for gid_b, name_b in zip(
            group_b["ids"],
            group_b["names"],
        ):
            dist, fromto = geom_distance(
                model,
                data,
                gid_a,
                gid_b,
            )

            if dist < best_dist:
                best_dist = dist
                best_pair = (name_a, name_b)
                best_fromto = fromto.copy()

    return best_dist, best_pair, best_fromto


# ============================================================================
# 主扫描
# ============================================================================

def scan_envelope(model, data, ids, active_signs, collision_groups):
    alpha_deg = np.arange(
        ALPHA_MIN_DEG,
        ALPHA_MAX_DEG + 0.5 * ALPHA_STEP_DEG,
        ALPHA_STEP_DEG,
        dtype=np.float64,
    )

    K = len(alpha_deg)

    center_tool = np.zeros((K, 3, 3), dtype=np.float64)
    surface_center_tool = np.zeros((K, 3, 3), dtype=np.float64)
    normal_tool = np.zeros((K, 3, 3), dtype=np.float64)
    surface_corners_tool = np.zeros((K, 3, 4, 3), dtype=np.float64)
    surface_grid_tool = np.zeros(
        (K, 3, SURFACE_GRID_N, SURFACE_GRID_N, 3),
        dtype=np.float64,
    )

    pairwise_distance = np.zeros((K, 3), dtype=np.float64)
    mean_radius = np.zeros(K, dtype=np.float64)
    equivalent_diameter = np.zeros(K, dtype=np.float64)

    # normal_alignment_deg:
    # 法向和“传感器中心 -> 三指抓取中心”方向夹角
    normal_alignment_deg = np.zeros((K, 3), dtype=np.float64)

    G_all = len(ALL_COLLISION_GROUP_PAIRS)
    G_valid = len(VALIDATING_COLLISION_GROUP_PAIRS)

    collision_group_distance = np.zeros((K, G_all), dtype=np.float64)
    collision_group_hit = np.zeros((K, G_all), dtype=bool)
    collision_group_best_pair = [
        [None for _ in range(G_all)]
        for _ in range(K)
    ]

    # 这里只由 finger-finger 三组距离决定。
    self_collision = np.zeros(K, dtype=bool)
    min_self_distance = np.zeros(K, dtype=np.float64)

    # finger-base 只作为诊断量保留。
    min_base_distance = np.zeros(K, dtype=np.float64)

    print("\n开始扫描：")
    print(
        f"alpha = {ALPHA_MIN_DEG:.1f}° ~ {ALPHA_MAX_DEG:.1f}°，"
        f"step = {ALPHA_STEP_DEG:.1f}°，"
        f"共 {K} 个状态。"
    )

    for k, a_deg in enumerate(alpha_deg):
        a_rad = math.radians(float(a_deg))

        set_finger_alpha(
            model,
            data,
            ids["finger_joint_ids"],
            a_rad,
        )

        tool_pos = data.site_xpos[ids["tool_site_id"]].copy()
        tool_R = data.site_xmat[ids["tool_site_id"]].reshape(3, 3).copy()

        # ----- 3 个 GelSight -----
        for i, gid in enumerate(ids["gel_geom_ids"]):
            g = gel_surface_geometry_tool(
                model,
                data,
                gid,
                active_signs[i],
                tool_pos,
                tool_R,
            )

            center_tool[k, i] = g["geom_center_tool"]
            surface_center_tool[k, i] = g["surface_center_tool"]
            normal_tool[k, i] = g["normal_tool"]
            surface_corners_tool[k, i] = g["corners_tool"]
            surface_grid_tool[k, i] = g["grid_tool"]

        # ----- 三指几何中心/直径 -----
        P = surface_center_tool[k]
        C = P.mean(axis=0)

        d12 = np.linalg.norm(P[0] - P[1])
        d13 = np.linalg.norm(P[0] - P[2])
        d23 = np.linalg.norm(P[1] - P[2])

        pairwise_distance[k] = [d12, d13, d23]

        radii = np.linalg.norm(
            P - C[None, :],
            axis=1,
        )

        mean_radius[k] = float(np.mean(radii))
        equivalent_diameter[k] = 2.0 * mean_radius[k]

        # ----- 法向对抓取中心的对准程度 -----
        for i in range(3):
            toward_center = C - P[i]

            if np.linalg.norm(toward_center) < 1e-9:
                normal_alignment_deg[k, i] = np.nan
            else:
                tc = normalize(toward_center)
                dot = float(
                    np.clip(
                        np.dot(normal_tool[k, i], tc),
                        -1.0,
                        1.0,
                    )
                )

                normal_alignment_deg[k, i] = math.degrees(
                    math.acos(dot)
                )

        # ----- 自碰撞/净空 -----
        # IMPORTANT:
        #   只有 finger-finger 三组参与真正的 self-collision 判定。
        #   finger-base 因 hinge 安装区域存在固定几何重叠，只记录诊断。
        validating_distances = []
        diagnostic_base_distances = []

        for gi, (ga, gb) in enumerate(ALL_COLLISION_GROUP_PAIRS):
            dist, best_pair, _ = min_group_distance(
                model,
                data,
                collision_groups[ga],
                collision_groups[gb],
            )

            collision_group_distance[k, gi] = dist
            collision_group_hit[k, gi] = (
                dist <= COLLISION_TOLERANCE_M
            )
            collision_group_best_pair[k][gi] = best_pair

            if gi < G_valid:
                validating_distances.append(dist)
            else:
                diagnostic_base_distances.append(dist)

        min_self_distance[k] = float(np.min(validating_distances))
        self_collision[k] = bool(
            np.any(
                collision_group_hit[k, :G_valid]
            )
        )

        min_base_distance[k] = float(
            np.min(diagnostic_base_distances)
        )

        if (
            k == 0
            or k == K - 1
            or k % max(1, int(round(5.0 / ALPHA_STEP_DEG))) == 0
        ):
            print(
                f"  alpha={a_deg:+6.1f}° | "
                f"D_eq={equivalent_diameter[k]*1000:7.2f} mm | "
                f"min_clearance={min_self_distance[k]*1000:7.3f} mm | "
                f"collision={self_collision[k]}"
            )

    collision_free = ~self_collision

    safe_clearance = (
        min_self_distance >= SAFE_CLEARANCE_M
    )

    return {
        "alpha_deg": alpha_deg,
        "center_tool": center_tool,
        "surface_center_tool": surface_center_tool,
        "normal_tool": normal_tool,
        "surface_corners_tool": surface_corners_tool,
        "surface_grid_tool": surface_grid_tool,
        "pairwise_distance": pairwise_distance,
        "mean_radius": mean_radius,
        "equivalent_diameter": equivalent_diameter,
        "normal_alignment_deg": normal_alignment_deg,
        "collision_group_distance": collision_group_distance,
        "collision_group_hit": collision_group_hit,
        "collision_group_best_pair": collision_group_best_pair,
        "self_collision": self_collision,
        "collision_free": collision_free,
        "min_self_distance": min_self_distance,
        "min_base_distance": min_base_distance,
        "safe_clearance": safe_clearance,
    }


# ============================================================================
# 保存数值结果
# ============================================================================

def save_numeric_results(result, active_signs, source_xml, collision_groups):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = OUTPUT_DIR / "envelope.npz"

    np.savez_compressed(
        npz_path,
        alpha_deg=result["alpha_deg"],
        active_face_signs=active_signs,

        center_tool=result["center_tool"],
        surface_center_tool=result["surface_center_tool"],
        normal_tool=result["normal_tool"],
        surface_corners_tool=result["surface_corners_tool"],
        surface_grid_tool=result["surface_grid_tool"],

        pairwise_distance=result["pairwise_distance"],
        mean_radius=result["mean_radius"],
        equivalent_diameter=result["equivalent_diameter"],
        normal_alignment_deg=result["normal_alignment_deg"],

        collision_group_distance=result["collision_group_distance"],
        collision_group_hit=result["collision_group_hit"],
        self_collision=result["self_collision"],
        collision_free=result["collision_free"],
        min_self_distance=result["min_self_distance"],
        min_base_distance=result["min_base_distance"],
        safe_clearance=result["safe_clearance"],
    )

    alpha = result["alpha_deg"]
    cf = result["collision_free"]
    safe = result["safe_clearance"]
    D = result["equivalent_diameter"]
    min_dist = result["min_self_distance"]

    summary = {
        "source_xml": str(source_xml),
        "alpha_min_deg": ALPHA_MIN_DEG,
        "alpha_max_deg": ALPHA_MAX_DEG,
        "alpha_step_deg": ALPHA_STEP_DEG,
        "num_states": int(len(alpha)),
        "surface_grid_n": SURFACE_GRID_N,

        "active_face_signs": active_signs.tolist(),
        "active_face_sign_meaning": {
            "+1": "GelSight active normal uses geom local +Z",
            "-1": "GelSight active normal uses geom local -Z",
        },

        "collision_tolerance_m": COLLISION_TOLERANCE_M,
        "safe_clearance_m": SAFE_CLEARANCE_M,

        "validating_collision_group_names": [
            f"{a}__{b}"
            for a, b in VALIDATING_COLLISION_GROUP_PAIRS
        ],

        "diagnostic_collision_group_names": [
            f"{a}__{b}"
            for a, b in DIAGNOSTIC_COLLISION_GROUP_PAIRS
        ],

        "collision_group_names": [
            f"{a}__{b}"
            for a, b in ALL_COLLISION_GROUP_PAIRS
        ],

        "collision_group_geom_names": {
            name: value["names"]
            for name, value in collision_groups.items()
        },

        "collision_free_intervals_deg": contiguous_intervals(
            alpha,
            cf,
        ),

        "safe_clearance_intervals_deg": contiguous_intervals(
            alpha,
            safe,
        ),

        "num_collision_free_states": int(np.sum(cf)),
        "num_self_collision_states": int(np.sum(~cf)),

        "minimum_finger_finger_clearance_m": float(np.min(min_dist)),
        "minimum_finger_finger_clearance_angle_deg": float(
            alpha[int(np.argmin(min_dist))]
        ),

        "minimum_finger_base_diagnostic_distance_m": float(
            np.min(result["min_base_distance"])
        ),
        "finger_base_note": (
            "finger-base distances are diagnostic only and do NOT "
            "participate in collision-free classification because "
            "the hinge mounting geometry contains intentional overlap."
        ),

        "theoretical_equivalent_diameter_mm": {
            "min": float(np.min(D) * 1000.0),
            "max": float(np.max(D) * 1000.0),
        },
    }

    if np.any(cf):
        summary["collision_free_equivalent_diameter_mm"] = {
            "min": float(np.min(D[cf]) * 1000.0),
            "max": float(np.max(D[cf]) * 1000.0),
        }
    else:
        summary["collision_free_equivalent_diameter_mm"] = None

    if np.any(safe):
        summary["safe_clearance_equivalent_diameter_mm"] = {
            "min": float(np.min(D[safe]) * 1000.0),
            "max": float(np.max(D[safe]) * 1000.0),
        }
    else:
        summary["safe_clearance_equivalent_diameter_mm"] = None

    summary_path = OUTPUT_DIR / "envelope_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # CSV：方便直接用 Excel 查看
    csv_path = OUTPUT_DIR / "envelope_table.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        header = [
            "alpha_deg",
            "collision_free",
            "safe_clearance_0p5mm",
            "min_finger_finger_clearance_mm",
            "min_finger_base_diagnostic_distance_mm",
            "equivalent_diameter_mm",
            "d12_mm",
            "d13_mm",
            "d23_mm",
            "normal_align_f1_deg",
            "normal_align_f2_deg",
            "normal_align_f3_deg",
        ]

        header += [
            f"clearance_{a}__{b}_mm"
            for a, b in ALL_COLLISION_GROUP_PAIRS
        ]

        writer.writerow(header)

        for k, a in enumerate(result["alpha_deg"]):
            row = [
                float(a),
                int(result["collision_free"][k]),
                int(result["safe_clearance"][k]),
                float(result["min_self_distance"][k] * 1000.0),
                float(result["min_base_distance"][k] * 1000.0),
                float(result["equivalent_diameter"][k] * 1000.0),

                float(result["pairwise_distance"][k, 0] * 1000.0),
                float(result["pairwise_distance"][k, 1] * 1000.0),
                float(result["pairwise_distance"][k, 2] * 1000.0),

                float(result["normal_alignment_deg"][k, 0]),
                float(result["normal_alignment_deg"][k, 1]),
                float(result["normal_alignment_deg"][k, 2]),
            ]

            row += [
                float(x * 1000.0)
                for x in result["collision_group_distance"][k]
            ]

            writer.writerow(row)

    return npz_path, summary_path, csv_path, summary


# ============================================================================
# 绘图
# ============================================================================

def set_equal_3d(ax, points):
    points = np.asarray(points)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    center = 0.5 * (mins + maxs)
    half = max(
        0.5 * float(np.max(maxs - mins)),
        1e-4,
    )

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def plot_center_trajectories(result):
    P = result["surface_center_tool"] * 1000.0

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    labels = ["finger1", "finger2", "finger3"]

    for i in range(3):
        ax.plot(
            P[:, i, 0],
            P[:, i, 1],
            P[:, i, 2],
            label=labels[i],
        )

        ax.scatter(
            [P[0, i, 0], P[-1, i, 0]],
            [P[0, i, 1], P[-1, i, 1]],
            [P[0, i, 2], P[-1, i, 2]],
            s=25,
        )

    ax.set_xlabel("tool X (mm)")
    ax.set_ylabel("tool Y (mm)")
    ax.set_zlabel("tool Z (mm)")
    ax.set_title("Three GelSight surface-center trajectories")
    ax.legend()

    set_equal_3d(ax, P.reshape(-1, 3))

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "center_trajectories.png",
        dpi=220,
    )
    plt.close(fig)


def plot_swept_surfaces(result):
    G = result["surface_grid_tool"] * 1000.0

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 为了图不太密，每隔几个 alpha 画一层 surface grid。
    stride = max(
        1,
        int(round(2.5 / ALPHA_STEP_DEG)),
    )

    all_points = []

    for k in range(0, G.shape[0], stride):
        for i in range(3):
            pts = G[k, i].reshape(-1, 3)
            all_points.append(pts)

            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=2,
                alpha=0.18,
            )

    all_points = np.concatenate(all_points, axis=0)

    ax.set_xlabel("tool X (mm)")
    ax.set_ylabel("tool Y (mm)")
    ax.set_zlabel("tool Z (mm)")
    ax.set_title("Swept GelSight active surfaces")

    set_equal_3d(ax, all_points)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "swept_gelsight_surfaces.png",
        dpi=220,
    )
    plt.close(fig)


def plot_normals_3d(result):
    P = result["surface_center_tool"] * 1000.0
    N = result["normal_tool"]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    stride = max(1, NORMAL_PLOT_STRIDE)

    selected_points = []

    for i in range(3):
        p = P[::stride, i, :]
        n = N[::stride, i, :]

        selected_points.append(p)

        ax.quiver(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            n[:, 0],
            n[:, 1],
            n[:, 2],
            length=12.0,
            normalize=True,
        )

        ax.plot(
            P[:, i, 0],
            P[:, i, 1],
            P[:, i, 2],
            alpha=0.5,
        )

    selected_points = np.concatenate(
        selected_points,
        axis=0,
    )

    ax.set_xlabel("tool X (mm)")
    ax.set_ylabel("tool Y (mm)")
    ax.set_zlabel("tool Z (mm)")
    ax.set_title("GelSight active-surface normals")

    set_equal_3d(
        ax,
        np.concatenate(
            [P.reshape(-1, 3), selected_points],
            axis=0,
        ),
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "normals_3d.png",
        dpi=220,
    )
    plt.close(fig)


def plot_normal_alignment(result):
    a = result["alpha_deg"]
    y = result["normal_alignment_deg"]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for i, label in enumerate(
        ["finger1", "finger2", "finger3"]
    ):
        ax.plot(
            a,
            y[:, i],
            label=label,
        )

    ax.set_xlabel("finger joint angle α (deg)")
    ax.set_ylabel("normal-to-grasp-center angle (deg)")
    ax.set_title("GelSight normal alignment vs joint angle")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "normal_alignment_vs_angle.png",
        dpi=220,
    )
    plt.close(fig)


def plot_collision_state(result):
    a = result["alpha_deg"]
    y = result["self_collision"].astype(float)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.step(
        a,
        y,
        where="mid",
    )

    ax.set_xlabel("finger joint angle α (deg)")
    ax.set_ylabel("self collision")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["free", "collision"])
    ax.set_title("Self-collision state vs joint angle")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "self_collision_vs_angle.png",
        dpi=220,
    )
    plt.close(fig)


def plot_min_clearance(result):
    a = result["alpha_deg"]
    y = result["min_self_distance"] * 1000.0

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(a, y)
    ax.axhline(0.0, linestyle="--")
    ax.axhline(
        SAFE_CLEARANCE_M * 1000.0,
        linestyle=":",
    )

    ax.set_xlabel("finger joint angle α (deg)")
    ax.set_ylabel("minimum finger-finger clearance (mm)")
    ax.set_title("Minimum finger-finger clearance vs joint angle")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "minimum_clearance_vs_angle.png",
        dpi=220,
    )
    plt.close(fig)


def plot_grasp_diameter(result):
    a = result["alpha_deg"]
    D = result["equivalent_diameter"] * 1000.0
    cf = result["collision_free"]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(
        a,
        D,
        label="theoretical D_eq",
    )

    if np.any(cf):
        ax.scatter(
            a[cf],
            D[cf],
            s=15,
            label="collision-free",
        )

    ax.set_xlabel("finger joint angle α (deg)")
    ax.set_ylabel("equivalent grasp diameter (mm)")
    ax.set_title("Equivalent grasp diameter vs joint angle")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "grasp_diameter_vs_angle.png",
        dpi=220,
    )
    plt.close(fig)


def plot_collision_free_envelope(result):
    mask = result["collision_free"]

    if not np.any(mask):
        print(
            "[WARNING] 没有 collision-free alpha，"
            "跳过 collision_free_envelope.png"
        )
        return

    G = result["surface_grid_tool"][mask] * 1000.0

    # 点很多，绘图时适当抽稀。
    points = G.reshape(-1, 3)

    if len(points) > 35000:
        idx = np.linspace(
            0,
            len(points) - 1,
            35000,
            dtype=int,
        )
        points = points[idx]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=2,
        alpha=0.22,
    )

    ax.set_xlabel("tool X (mm)")
    ax.set_ylabel("tool Y (mm)")
    ax.set_zlabel("tool Z (mm)")
    ax.set_title("Collision-free tactile reachable envelope")

    set_equal_3d(ax, points)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "collision_free_envelope.png",
        dpi=220,
    )
    plt.close(fig)


def make_all_plots(result):
    print("\n生成图像 ...")

    plot_center_trajectories(result)
    plot_swept_surfaces(result)
    plot_normals_3d(result)
    plot_normal_alignment(result)
    plot_collision_state(result)
    plot_min_clearance(result)
    plot_grasp_diameter(result)
    plot_collision_free_envelope(result)


# ============================================================================
# main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_xml = find_source_xml()

    print("=" * 84)
    print("Three-Finger GelSight Reachable Envelope Analysis")
    print("=" * 84)
    print("XML:")
    print("  ", source_xml)
    print("OUTPUT:")
    print("  ", OUTPUT_DIR)

    model = mujoco.MjModel.from_xml_path(
        str(source_xml)
    )
    data = mujoco.MjData(model)

    reset_to_home(model, data)

    ids = {
        "tool_site_id": require_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            TOOL_SITE,
        ),

        "finger_joint_ids": [
            require_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in FINGER_JOINTS
        ],

        "gel_site_ids": [
            require_id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                name,
            )
            for name in GEL_SITES
        ],

        "gel_geom_ids": [
            require_id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            for name in GEL_GEOMS
        ],
    }

    print("\nGelSight box half sizes:")
    for name, gid in zip(
        GEL_GEOMS,
        ids["gel_geom_ids"],
    ):
        size = model.geom_size[gid][:3]
        print(
            f"  {name}: "
            f"{size[0]*1000:.3f} × "
            f"{size[1]*1000:.3f} × "
            f"{size[2]*1000:.3f} mm half-size"
        )

    active_signs = infer_active_face_signs(
        model,
        data,
        ids,
    )

    print("\n自动判断的 GelSight 有效面：")
    for i, sign in enumerate(
        active_signs,
        start=1,
    ):
        side = "+Z" if sign > 0 else "-Z"
        print(
            f"  finger{i}: local {side}"
        )

    collision_groups = resolve_collision_groups(
        model
    )

    print("\n自碰撞分析 geom groups:")
    for name, value in collision_groups.items():
        print(
            f"  {name}: {value['names']}"
        )

    print("\n[判定规则]")
    print("  collision-free 只由以下三组 finger-finger 几何对决定：")
    for a, b in VALIDATING_COLLISION_GROUP_PAIRS:
        print(f"    {a} <-> {b}")
    print("  finger <-> base 只记录诊断，不参与 collision-free 判定。")

    result = scan_envelope(
        model,
        data,
        ids,
        active_signs,
        collision_groups,
    )

    npz_path, summary_path, csv_path, summary = save_numeric_results(
        result,
        active_signs,
        source_xml,
        collision_groups,
    )

    make_all_plots(result)

    print("\n" + "=" * 84)
    print("分析完成")
    print("=" * 84)

    print("\ncollision-free alpha intervals:")
    print(
        summary["collision_free_intervals_deg"]
    )

    print("\nsafe-clearance alpha intervals "
          f"(>= {SAFE_CLEARANCE_M*1000:.1f} mm):")
    print(
        summary["safe_clearance_intervals_deg"]
    )

    print("\n理论 D_eq 范围:")
    print(
        summary["theoretical_equivalent_diameter_mm"]
    )

    print("\ncollision-free D_eq 范围:")
    print(
        summary["collision_free_equivalent_diameter_mm"]
    )

    print("\nsafe-clearance D_eq 范围:")
    print(
        summary["safe_clearance_equivalent_diameter_mm"]
    )

    print("\n最小 self-clearance:")
    print(
        f"  {summary['minimum_finger_finger_clearance_m']*1000:.4f} mm "
        f"@ alpha={summary['minimum_finger_finger_clearance_angle_deg']:.1f}°"
    )

    print("\n输出文件:")
    print("  ", npz_path)
    print("  ", summary_path)
    print("  ", csv_path)
    print("  ", OUTPUT_DIR / "center_trajectories.png")
    print("  ", OUTPUT_DIR / "swept_gelsight_surfaces.png")
    print("  ", OUTPUT_DIR / "normals_3d.png")
    print("  ", OUTPUT_DIR / "normal_alignment_vs_angle.png")
    print("  ", OUTPUT_DIR / "self_collision_vs_angle.png")
    print("  ", OUTPUT_DIR / "minimum_clearance_vs_angle.png")
    print("  ", OUTPUT_DIR / "grasp_diameter_vs_angle.png")
    print("  ", OUTPUT_DIR / "collision_free_envelope.png")

    print(
        "\n下一步请把控制台最后这几项结果，以及 "
        "self_collision_vs_angle.png / "
        "grasp_diameter_vs_angle.png / "
        "collision_free_envelope.png 发给我。"
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
            "\n把完整控制台输出发给我，我继续根据你的 MuJoCo 模型修。"
        )
        sys.exit(1)
