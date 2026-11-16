# -*- coding: utf-8 -*-
r"""
audit_candidate_kinematic_sweep.py

目的：
    在“不做动力学、不推物体”的前提下，把某个 G_c 的执行路径拆成两段审计：

    A. approach sweep
       fingers 固定在 alpha_pre，
       tool0 从 pregrasp -> G_c 直线接近。

    B. closing sweep
       tool0 固定在 G_c，
       fingers 从 alpha_pre -> alpha_c -> alpha_c - preload。

    每个状态只做：
        qpos / IK 设置
        mj_forward()
    不做：
        mj_step()
        接触力
        lift
        Taxim

为什么要做：
    static replay 已证明 Candidate #4 的目标终态 G_c 是正确的；
    现在要定位“动态执行路径中，哪个时刻先发生了错误接触”。

当前 XML 的一个重要事实：
    finger*_vis 和 finger*_gelsight_housing 的 contype/conaffinity 都是 0，
    它们不会在当前 MuJoCo 动力学里推物体。
    真正会和 tomato 碰撞的是 finger*_gel。

因此重点判断：
    若 gel whole-box 已经接近/穿入 tomato，
    但 GelSight active-face center 仍离 CAD 表面很远，
    则说明发生的是 gel 的侧边/棱角/背面先接触，而不是正面触觉接触。

默认：
    Candidate #4
    pregrasp distance = 35 mm
    alpha_pre = alpha_c + 10 deg
    closing 到 alpha_c - 1 deg

输出：
    grasp_candidates/tomato_soup_can/kinematic_audit/
        candidate_004_approach.csv
        candidate_004_closing.csv
        candidate_004_approach.png
        candidate_004_closing.png

运行：
    python audit_candidate_kinematic_sweep.py --candidate 4

也可看静态 Viewer 最危险状态：
    python audit_candidate_kinematic_sweep.py --candidate 4 --viewer-worst
"""

from pathlib import Path
import argparse
import csv
import math
import sys
import time

import numpy as np
import mujoco
from mujoco import viewer as mj_viewer

try:
    from scipy.spatial import cKDTree
except ImportError:
    raise ImportError("缺少 scipy，请 pip install scipy")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

XML_PATH = (
    PROJECT_ROOT
    / "xml"
    / "ur3_with_threefinger_gelsight_table_tomato.xml"
)

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "grasp_candidates"
    / "tomato_soup_can"
    / "candidates_top50.npz"
)

ENVELOPE_PATH = (
    PROJECT_ROOT
    / "gripper_envelope"
    / "envelope.npz"
)

CAD_CACHE_PATH = (
    PROJECT_ROOT
    / "grasp_candidates"
    / "tomato_soup_can"
    / "cad_surface_cache.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "grasp_candidates"
    / "tomato_soup_can"
    / "kinematic_audit"
)


# =============================================================================
# Model names
# =============================================================================

TOOL_SITE = "tool0"
OBJECT_BODY = "tomato_soup_can"

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

ARM_ACTUATORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

FINGER_JOINTS = [
    "finger1_hinge",
    "finger2_hinge",
    "finger3_hinge",
]

FINGER_ACTUATORS = [
    "finger1_ctrl",
    "finger2_ctrl",
    "finger3_ctrl",
]

GEL_GEOMS = [
    "finger1_gel",
    "finger2_gel",
    "finger3_gel",
]

HOUSING_GEOMS = [
    "finger1_gelsight_housing",
    "finger2_gelsight_housing",
    "finger3_gelsight_housing",
]

FINGER_VIS_GEOMS = [
    "finger1_vis",
    "finger2_vis",
    "finger3_vis",
]

OBJECT_COLLISION_PREFIX = "tomato_collision_"


# =============================================================================
# Audit settings
# =============================================================================

PREGRASP_DISTANCE_M = 0.035
PREOPEN_EXTRA_DEG = 10.0

APPROACH_SAMPLES = 31

CLOSE_STEP_DEG = 0.25
AUDIT_PRELOAD_DEG = 1.0

# whole gel box 几何距离 <= 0：已经发生几何接触/穿透
BOX_CONTACT_THRESHOLD_M = 0.0

# 若 whole gel box 已接触，但 active face center 仍距 CAD > 2 mm，
# 高度怀疑是侧边/棱角/背面先碰。
FACE_STILL_FAR_M = 0.002

# 正面法向允许误差；这里只用于诊断，不直接决定 G*。
FACE_NORMAL_WARN_DEG = 40.0

# mj_geomDistance 查询范围
GEOM_DISTANCE_MAX_M = 0.20


# =============================================================================
# IK settings
# =============================================================================

IK_MAX_ITERS = 450
IK_DAMPING = 0.03
IK_GAIN = 0.70
IK_MAX_STEP = math.radians(7.0)
IK_POSITION_TOL_M = 0.0012
IK_ROTATION_TOL_RAD = math.radians(1.2)


# =============================================================================
# Math / MuJoCo utilities
# =============================================================================

def mj_id(model, typ, name):
    idx = mujoco.mj_name2id(model, typ, name)
    if idx < 0:
        raise RuntimeError(f"MuJoCo 中找不到：{name}")
    return idx


def optional_geom_id(model, name):
    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )
    return idx if idx >= 0 else None


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise RuntimeError("zero vector")
    return v / n


def pose_matrix(p, R):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def get_body_pose(data, bid):
    return pose_matrix(
        data.xpos[bid].copy(),
        data.xmat[bid].reshape(3, 3).copy(),
    )


def get_site_pose(data, sid):
    return pose_matrix(
        data.site_xpos[sid].copy(),
        data.site_xmat[sid].reshape(3, 3).copy(),
    )


def rotation_log(R):
    c = float(
        np.clip(
            (np.trace(R) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
    )
    theta = math.acos(c)

    if theta < 1e-9:
        return np.array(
            [
                0.5 * (R[2, 1] - R[1, 2]),
                0.5 * (R[0, 2] - R[2, 0]),
                0.5 * (R[1, 0] - R[0, 1]),
            ],
            dtype=np.float64,
        )

    s = 2.0 * math.sin(theta)
    if abs(s) < 1e-8:
        return np.zeros(3)

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    ) / s

    return axis * theta


def get_joint_q(model, data, jid):
    return float(
        data.qpos[
            int(model.jnt_qposadr[jid])
        ]
    )


def set_joint_q(model, data, jid, q):
    if bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        q = float(np.clip(q, lo, hi))

    data.qpos[
        int(model.jnt_qposadr[jid])
    ] = q


def set_ctrl(model, data, aid, q):
    if bool(model.actuator_ctrllimited[aid]):
        lo, hi = model.actuator_ctrlrange[aid]
        q = float(np.clip(q, lo, hi))
    data.ctrl[aid] = q


def arm_q(model, data, ids):
    return np.array(
        [
            get_joint_q(model, data, jid)
            for jid in ids["arm_jids"]
        ],
        dtype=np.float64,
    )


def set_arm_q(model, data, ids, q):
    for jid, val in zip(ids["arm_jids"], q):
        set_joint_q(model, data, jid, float(val))
        dadr = int(model.jnt_dofadr[jid])
        data.qvel[dadr] = 0.0

    mujoco.mj_forward(model, data)

    for jid, aid in zip(
        ids["arm_jids"],
        ids["arm_aids"],
    ):
        set_ctrl(
            model,
            data,
            aid,
            get_joint_q(model, data, jid),
        )


def set_fingers_direct(model, data, ids, alpha_deg):
    q = math.radians(float(alpha_deg))

    for jid, aid in zip(
        ids["finger_jids"],
        ids["finger_aids"],
    ):
        set_joint_q(model, data, jid, q)
        set_ctrl(model, data, aid, q)
        data.qvel[
            int(model.jnt_dofadr[jid])
        ] = 0.0

    mujoco.mj_forward(model, data)


def solve_ik_once(
    model,
    data,
    ids,
    target_p,
    target_R,
):
    best_q = arm_q(model, data, ids)
    best_ep = float("inf")
    best_er = float("inf")
    best_score = float("inf")

    for _ in range(IK_MAX_ITERS):
        mujoco.mj_forward(model, data)

        cur_p = data.site_xpos[
            ids["tool_sid"]
        ].copy()

        cur_R = data.site_xmat[
            ids["tool_sid"]
        ].reshape(3, 3).copy()

        ep_vec = target_p - cur_p
        er_vec = rotation_log(
            target_R @ cur_R.T
        )

        ep = float(np.linalg.norm(ep_vec))
        er = float(np.linalg.norm(er_vec))
        score = ep + 0.08 * er

        if score < best_score:
            best_score = score
            best_ep = ep
            best_er = er
            best_q = arm_q(model, data, ids)

        if (
            ep <= IK_POSITION_TOL_M
            and er <= IK_ROTATION_TOL_RAD
        ):
            return True, ep, er, arm_q(model, data, ids)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            ids["tool_sid"],
        )

        cols = ids["arm_dof_adrs"]

        J = np.vstack(
            [
                jacp[:, cols],
                jacr[:, cols],
            ]
        )

        e = np.concatenate(
            [ep_vec, er_vec]
        )

        A = (
            J @ J.T
            + IK_DAMPING**2
            * np.eye(6)
        )

        try:
            dq = J.T @ np.linalg.solve(A, e)
        except np.linalg.LinAlgError:
            dq = np.linalg.pinv(J) @ e

        dq *= IK_GAIN
        dq = np.clip(
            dq,
            -IK_MAX_STEP,
            +IK_MAX_STEP,
        )

        set_arm_q(
            model,
            data,
            ids,
            arm_q(model, data, ids) + dq,
        )

    set_arm_q(model, data, ids, best_q)
    return False, best_ep, best_er, best_q


def solve_ik_multistart(
    model,
    data,
    ids,
    target_p,
    target_R,
    home_q,
):
    seeds = [
        home_q.copy(),
        np.radians([0, -90, 90, -90, -90, 0]),
        np.radians([0, -70, 110, -130, -90, 0]),
        np.radians([30, -90, 90, -90, -90, 0]),
        np.radians([-30, -90, 90, -90, -90, 0]),
        np.radians([60, -80, 100, -110, -90, 0]),
        np.radians([-60, -80, 100, -110, -90, 0]),
    ]

    best = None

    for q0 in seeds:
        set_arm_q(model, data, ids, q0)

        ok, ep, er, q = solve_ik_once(
            model,
            data,
            ids,
            target_p,
            target_R,
        )

        score = ep + 0.08 * er

        if best is None or score < best["score"]:
            best = {
                "ok": ok,
                "ep": ep,
                "er": er,
                "q": q.copy(),
                "score": score,
            }

        if ok:
            break

    set_arm_q(model, data, ids, best["q"])
    return best


def min_geom_distance_to_object(
    model,
    data,
    geom_id,
    object_geom_ids,
):
    best = float("inf")
    best_obj_gid = None

    fromto = np.zeros(6, dtype=np.float64)

    for ogid in object_geom_ids:
        d = mujoco.mj_geomDistance(
            model,
            data,
            int(geom_id),
            int(ogid),
            GEOM_DISTANCE_MAX_M,
            fromto,
        )

        d = float(d)

        if d < best:
            best = d
            best_obj_gid = int(ogid)

    return best, best_obj_gid


# =============================================================================
# Candidate/envelope/CAD
# =============================================================================

def get_active_face_signs(envelope):
    if "active_face_signs" in envelope:
        signs = envelope[
            "active_face_signs"
        ].astype(np.int32)

        if signs.shape == (3,):
            return signs

    return np.array([-1, -1, -1], dtype=np.int32)


def envelope_approach_axis_tool(
    envelope,
    alpha_candidate_deg,
):
    alpha = envelope[
        "alpha_deg"
    ].astype(np.float64)

    idx = int(
        np.argmin(
            np.abs(
                alpha - float(alpha_candidate_deg)
            )
        )
    )

    centers = envelope[
        "surface_center_tool"
    ][idx].astype(np.float64)

    return normalize(np.mean(centers, axis=0))


def gel_active_face_state_object(
    model,
    data,
    ids,
    T_world_object,
    active_signs,
):
    T_object_world = np.linalg.inv(T_world_object)

    centers_obj = np.zeros((3, 3))
    normals_obj = np.zeros((3, 3))

    for i, gid in enumerate(ids["gel_geom_ids"]):
        p_w = data.geom_xpos[gid].copy()
        R_w = data.geom_xmat[gid].reshape(3, 3).copy()

        hz = float(model.geom_size[gid][2])
        sign = float(active_signs[i])

        p_face_w = (
            p_w
            + R_w
            @ np.array(
                [0.0, 0.0, sign * hz],
                dtype=np.float64,
            )
        )

        n_face_w = normalize(
            R_w
            @ np.array(
                [0.0, 0.0, sign],
                dtype=np.float64,
            )
        )

        centers_obj[i] = (
            T_object_world[:3, :3]
            @ p_face_w
            + T_object_world[:3, 3]
        )

        normals_obj[i] = normalize(
            T_object_world[:3, :3]
            @ n_face_w
        )

    return centers_obj, normals_obj


def query_active_face_to_cad(
    centers_obj,
    normals_obj,
    tree,
    cad_points,
    cad_normals,
):
    d, idx = tree.query(centers_obj, k=1)

    nearest_p = cad_points[idx]
    nearest_n = cad_normals[idx]

    # 理想 GelSight active normal 指向物体，CAD normal 向外，
    # 因此 n_gel ≈ -n_cad
    dots = np.sum(
        normals_obj * (-nearest_n),
        axis=1,
    )

    dots = np.clip(dots, -1.0, 1.0)

    angle = np.degrees(
        np.arccos(dots)
    )

    return (
        np.asarray(d),
        np.asarray(angle),
        nearest_p,
        nearest_n,
    )


# =============================================================================
# Measurement
# =============================================================================

def measure_state(
    phase,
    parameter,
    alpha_deg,
    model,
    data,
    ids,
    T_world_object,
    active_signs,
    tree,
    cad_points,
    cad_normals,
):
    centers_obj, normals_obj = (
        gel_active_face_state_object(
            model,
            data,
            ids,
            T_world_object,
            active_signs,
        )
    )

    face_d, face_ang, _, _ = (
        query_active_face_to_cad(
            centers_obj,
            normals_obj,
            tree,
            cad_points,
            cad_normals,
        )
    )

    row = {
        "phase": phase,
        "parameter": float(parameter),
        "alpha_deg": float(alpha_deg),
    }

    for i in range(3):
        gel_d, _ = (
            min_geom_distance_to_object(
                model,
                data,
                ids["gel_geom_ids"][i],
                ids["object_geom_ids"],
            )
        )

        housing_d = np.nan
        if ids["housing_geom_ids"][i] is not None:
            housing_d, _ = (
                min_geom_distance_to_object(
                    model,
                    data,
                    ids["housing_geom_ids"][i],
                    ids["object_geom_ids"],
                )
            )

        finger_d = np.nan
        if ids["finger_vis_geom_ids"][i] is not None:
            finger_d, _ = (
                min_geom_distance_to_object(
                    model,
                    data,
                    ids["finger_vis_geom_ids"][i],
                    ids["object_geom_ids"],
                )
            )

        suspicious = bool(
            gel_d <= BOX_CONTACT_THRESHOLD_M
            and (
                face_d[i] > FACE_STILL_FAR_M
                or face_ang[i] > FACE_NORMAL_WARN_DEG
            )
        )

        row[f"F{i+1}_gel_box_dist_m"] = float(gel_d)
        row[f"F{i+1}_active_face_center_dist_m"] = float(face_d[i])
        row[f"F{i+1}_active_face_normal_error_deg"] = float(face_ang[i])
        row[f"F{i+1}_housing_dist_m"] = float(housing_d)
        row[f"F{i+1}_finger_vis_dist_m"] = float(finger_d)
        row[f"F{i+1}_suspicious_side_contact"] = suspicious

    return row


# =============================================================================
# Save / plot
# =============================================================================

def save_csv(rows, path):
    if not rows:
        return

    keys = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_phase(rows, phase_name, x_key, x_label, output_path):
    x = np.asarray(
        [r[x_key] for r in rows],
        dtype=np.float64,
    )

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    for i in range(3):
        gel = np.asarray(
            [
                r[f"F{i+1}_gel_box_dist_m"]
                for r in rows
            ]
        ) * 1000.0

        face = np.asarray(
            [
                r[
                    f"F{i+1}_active_face_center_dist_m"
                ]
                for r in rows
            ]
        ) * 1000.0

        ax.plot(
            x,
            gel,
            label=f"F{i+1} whole gel-box distance",
        )

        ax.plot(
            x,
            face,
            linestyle="--",
            label=f"F{i+1} active-face center distance",
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Distance to tomato (mm)")
    ax.set_title(
        f"{phase_name}: whole gel box vs active-face center"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# =============================================================================
# Diagnostics
# =============================================================================

def first_suspicious(rows):
    for row in rows:
        bad = [
            i + 1
            for i in range(3)
            if row[
                f"F{i+1}_suspicious_side_contact"
            ]
        ]

        if bad:
            return row, bad

    return None, []


def print_phase_summary(rows, phase_name):
    print("\n" + "-" * 84)
    print(phase_name)
    print("-" * 84)

    bad_row, bad_fingers = first_suspicious(rows)

    if bad_row is None:
        print(
            "没有发现“whole gel box 已接触、但 active face 仍明显未到位”的状态。"
        )
    else:
        print(
            "发现疑似 gel 侧边/棱角/背面提前接触："
        )
        print(
            "  parameter =",
            bad_row["parameter"],
        )
        print(
            "  alpha_deg =",
            bad_row["alpha_deg"],
        )
        print(
            "  fingers   =",
            bad_fingers,
        )

        for f in bad_fingers:
            print(
                f"  F{f}: "
                f"gel_box={bad_row[f'F{f}_gel_box_dist_m']*1000:.3f} mm, "
                f"active_face={bad_row[f'F{f}_active_face_center_dist_m']*1000:.3f} mm, "
                f"normal_err={bad_row[f'F{f}_active_face_normal_error_deg']:.2f} deg"
            )

    print("\n阶段终点：")
    end = rows[-1]

    for i in range(3):
        print(
            f"  F{i+1}: "
            f"gel_box={end[f'F{i+1}_gel_box_dist_m']*1000:.3f} mm, "
            f"active_face={end[f'F{i+1}_active_face_center_dist_m']*1000:.3f} mm, "
            f"normal_err={end[f'F{i+1}_active_face_normal_error_deg']:.2f} deg"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--viewer-worst",
        action="store_true",
        help="审计完成后打开 Viewer 显示第一个疑似提前接触状态",
    )

    args = parser.parse_args()

    rank = int(args.candidate)

    for path in [
        XML_PATH,
        CANDIDATE_PATH,
        ENVELOPE_PATH,
        CAD_CACHE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate = np.load(
        CANDIDATE_PATH,
        allow_pickle=True,
    )

    envelope = np.load(
        ENVELOPE_PATH,
        allow_pickle=True,
    )

    cad_cache = np.load(
        CAD_CACHE_PATH,
        allow_pickle=True,
    )

    cad_points = cad_cache[
        "surface_points"
    ].astype(np.float64)

    cad_normals = cad_cache[
        "surface_normals"
    ].astype(np.float64)

    tree = cKDTree(cad_points)

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(model)

    ids = {}

    ids["tool_sid"] = mj_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        TOOL_SITE,
    )

    ids["object_bid"] = mj_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        OBJECT_BODY,
    )

    ids["arm_jids"] = [
        mj_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        for name in ARM_JOINTS
    ]

    ids["arm_aids"] = [
        mj_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in ARM_ACTUATORS
    ]

    ids["arm_dof_adrs"] = [
        int(model.jnt_dofadr[jid])
        for jid in ids["arm_jids"]
    ]

    ids["finger_jids"] = [
        mj_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        for name in FINGER_JOINTS
    ]

    ids["finger_aids"] = [
        mj_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in FINGER_ACTUATORS
    ]

    ids["gel_geom_ids"] = [
        mj_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        for name in GEL_GEOMS
    ]

    ids["housing_geom_ids"] = [
        optional_geom_id(model, name)
        for name in HOUSING_GEOMS
    ]

    ids["finger_vis_geom_ids"] = [
        optional_geom_id(model, name)
        for name in FINGER_VIS_GEOMS
    ]

    ids["object_geom_ids"] = []

    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            gid,
        )

        if (
            name is not None
            and name.startswith(
                OBJECT_COLLISION_PREFIX
            )
        ):
            ids["object_geom_ids"].append(gid)

    if not ids["object_geom_ids"]:
        raise RuntimeError(
            "找不到 tomato_collision_*"
        )

    # ---------------------------------------------------------
    # Confirm collision configuration
    # ---------------------------------------------------------

    print("=" * 88)
    print("Candidate kinematic sweep audit")
    print("=" * 88)

    print(f"Candidate #{rank:03d}")

    print("\n当前 XML 碰撞配置：")

    for i in range(3):
        gid = ids["gel_geom_ids"][i]
        print(
            f"  F{i+1} gel: "
            f"contype={model.geom_contype[gid]}, "
            f"conaffinity={model.geom_conaffinity[gid]}"
        )

        hgid = ids["housing_geom_ids"][i]
        if hgid is not None:
            print(
                f"       housing: "
                f"contype={model.geom_contype[hgid]}, "
                f"conaffinity={model.geom_conaffinity[hgid]}"
            )

        fgid = ids["finger_vis_geom_ids"][i]
        if fgid is not None:
            print(
                f"       finger_vis: "
                f"contype={model.geom_contype[fgid]}, "
                f"conaffinity={model.geom_conaffinity[fgid]}"
            )

    # reset home keyframe
    home_key = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(
            model,
            data,
            home_key,
        )
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)

    home_q = arm_q(model, data, ids)

    idx = rank - 1

    if idx < 0 or idx >= len(
        candidate["T_object_tool"]
    ):
        raise ValueError("candidate rank 超范围")

    T_object_tool = candidate[
        "T_object_tool"
    ][idx].astype(np.float64)

    alpha_c = float(
        candidate["alpha_deg"][idx]
    )

    alpha_pre = min(
        30.0,
        alpha_c + PREOPEN_EXTRA_DEG,
    )

    active_signs = get_active_face_signs(
        envelope
    )

    approach_axis_tool = (
        envelope_approach_axis_tool(
            envelope,
            alpha_c,
        )
    )

    T_world_object = get_body_pose(
        data,
        ids["object_bid"],
    )

    T_world_tool_target = (
        T_world_object
        @ T_object_tool
    )

    target_p = (
        T_world_tool_target[:3, 3].copy()
    )

    target_R = (
        T_world_tool_target[:3, :3].copy()
    )

    approach_axis_world = normalize(
        target_R @ approach_axis_tool
    )

    pre_p = (
        target_p
        - PREGRASP_DISTANCE_M
        * approach_axis_world
    )

    print(
        f"\nalpha_c   = {alpha_c:+.3f} deg"
    )
    print(
        f"alpha_pre = {alpha_pre:+.3f} deg"
    )
    print(
        f"pregrasp  = {PREGRASP_DISTANCE_M*1000:.1f} mm"
    )

    # ---------------------------------------------------------
    # Phase A: approach
    # ---------------------------------------------------------

    print("\n[Phase A] approach sweep ...")

    set_fingers_direct(
        model,
        data,
        ids,
        alpha_pre,
    )

    ik_pre = solve_ik_multistart(
        model,
        data,
        ids,
        pre_p,
        target_R,
        home_q,
    )

    if not ik_pre["ok"]:
        raise RuntimeError(
            "pregrasp IK failed，无法做 approach sweep"
        )

    approach_rows = []

    for k, lam in enumerate(
        np.linspace(
            0.0,
            1.0,
            APPROACH_SAMPLES,
        )
    ):
        p = (
            (1.0 - lam) * pre_p
            + lam * target_p
        )

        ok, ep, er, q = solve_ik_once(
            model,
            data,
            ids,
            p,
            target_R,
        )

        if not ok:
            raise RuntimeError(
                f"approach λ={lam:.3f} IK failed: "
                f"ep={ep*1000:.2f}mm "
                f"er={math.degrees(er):.2f}deg"
            )

        set_fingers_direct(
            model,
            data,
            ids,
            alpha_pre,
        )

        row = measure_state(
            phase="approach",
            parameter=lam,
            alpha_deg=alpha_pre,
            model=model,
            data=data,
            ids=ids,
            T_world_object=T_world_object,
            active_signs=active_signs,
            tree=tree,
            cad_points=cad_points,
            cad_normals=cad_normals,
        )

        approach_rows.append(row)

    # ---------------------------------------------------------
    # Phase B: close
    # ---------------------------------------------------------

    print("[Phase B] closing sweep ...")

    ik_target = solve_ik_multistart(
        model,
        data,
        ids,
        target_p,
        target_R,
        home_q,
    )

    if not ik_target["ok"]:
        raise RuntimeError(
            "target IK failed，无法做 closing sweep"
        )

    end_alpha = alpha_c - AUDIT_PRELOAD_DEG

    if alpha_pre >= end_alpha:
        alphas = np.arange(
            alpha_pre,
            end_alpha - 1e-9,
            -CLOSE_STEP_DEG,
        )
    else:
        alphas = np.arange(
            alpha_pre,
            end_alpha + 1e-9,
            +CLOSE_STEP_DEG,
        )

    # 确保 alpha_c 和 end_alpha 都出现
    alphas = np.unique(
        np.concatenate(
            [
                alphas,
                np.array(
                    [alpha_c, end_alpha],
                    dtype=np.float64,
                ),
            ]
        )
    )[::-1]

    closing_rows = []

    for a in alphas:
        set_arm_q(
            model,
            data,
            ids,
            ik_target["q"],
        )

        set_fingers_direct(
            model,
            data,
            ids,
            float(a),
        )

        row = measure_state(
            phase="closing",
            parameter=float(a),
            alpha_deg=float(a),
            model=model,
            data=data,
            ids=ids,
            T_world_object=T_world_object,
            active_signs=active_signs,
            tree=tree,
            cad_points=cad_points,
            cad_normals=cad_normals,
        )

        closing_rows.append(row)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    approach_csv = (
        OUTPUT_DIR
        / f"candidate_{rank:03d}_approach.csv"
    )

    closing_csv = (
        OUTPUT_DIR
        / f"candidate_{rank:03d}_closing.csv"
    )

    save_csv(
        approach_rows,
        approach_csv,
    )

    save_csv(
        closing_rows,
        closing_csv,
    )

    approach_png = (
        OUTPUT_DIR
        / f"candidate_{rank:03d}_approach.png"
    )

    closing_png = (
        OUTPUT_DIR
        / f"candidate_{rank:03d}_closing.png"
    )

    plot_phase(
        approach_rows,
        "Approach sweep",
        "parameter",
        "Approach fraction λ (0=pregrasp, 1=Gc)",
        approach_png,
    )

    plot_phase(
        closing_rows,
        "Closing sweep",
        "alpha_deg",
        "Finger alpha (deg)",
        closing_png,
    )

    # ---------------------------------------------------------
    # Print diagnostics
    # ---------------------------------------------------------

    print_phase_summary(
        approach_rows,
        "Phase A: approach",
    )

    print_phase_summary(
        closing_rows,
        "Phase B: closing",
    )

    print("\n保存：")
    print(" ", approach_csv)
    print(" ", closing_csv)
    print(" ", approach_png)
    print(" ", closing_png)

    # ---------------------------------------------------------
    # Determine first suspicious state overall
    # ---------------------------------------------------------

    bad_a, bad_f_a = first_suspicious(
        approach_rows
    )

    bad_c, bad_f_c = first_suspicious(
        closing_rows
    )

    if bad_a is not None:
        print(
            "\n结论：问题首先出现在 APPROACH 路径。"
        )
        print(
            "说明 alpha_pre 固定时，夹爪在到达 Gc 之前已有 gel 侧边/棱角提前接触风险。"
        )

    elif bad_c is not None:
        print(
            "\n结论：Approach 基本干净，问题首先出现在 CLOSING 路径。"
        )
        print(
            "说明固定 Gc 后，从 alpha_pre 闭合到 alpha_c 的过程中，gel box 先以非正面方式碰物体。"
        )

    else:
        print(
            "\n没有发现明显的 whole-gel-box 提前接触。"
        )
        print(
            "若 V3 仍把物体推走，下一步检查 actuator 动态响应、软接触 margin/solref/solimp 和关闭到 alpha_c 后的过量闭合。"
        )

    # ---------------------------------------------------------
    # Viewer worst static state
    # ---------------------------------------------------------

    if args.viewer_worst:
        chosen = None

        if bad_a is not None:
            chosen = ("approach", bad_a)
        elif bad_c is not None:
            chosen = ("closing", bad_c)

        if chosen is None:
            print(
                "\n没有 suspicious state 可显示。"
            )
            return

        phase, row = chosen

        if phase == "approach":
            lam = row["parameter"]

            p = (
                (1.0 - lam) * pre_p
                + lam * target_p
            )

            solve_ik_multistart(
                model,
                data,
                ids,
                p,
                target_R,
                home_q,
            )

            set_fingers_direct(
                model,
                data,
                ids,
                alpha_pre,
            )

        else:
            solve_ik_multistart(
                model,
                data,
                ids,
                target_p,
                target_R,
                home_q,
            )

            set_fingers_direct(
                model,
                data,
                ids,
                row["alpha_deg"],
            )

        mujoco.mj_forward(model, data)

        print(
            "\nViewer 显示第一个 suspicious static state。"
        )
        print(
            "不调用 mj_step，因此 tomato 不会被推走。"
        )

        with mj_viewer.launch_passive(
            model,
            data,
        ) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n用户终止。")

    except Exception as exc:
        print("\n[ERROR]")
        print(repr(exc))
        print(
            "\n把完整控制台输出发给我。"
        )
        sys.exit(1)
