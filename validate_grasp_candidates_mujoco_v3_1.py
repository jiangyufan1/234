# -*- coding: utf-8 -*-
r"""
validate_grasp_candidates_mujoco.py

目的：
    把已经生成的几何候选 G_c 放回 MuJoCo，做“第一轮物理可执行性验证”。

本脚本只验证：
    1) G_c -> 世界坐标是否能正确转换
    2) UR3 是否能到达 pregrasp / contact pose
    3) 接近过程中 gripper 是否撞桌面
    4) 三指从预张开状态闭合后，是否真的产生 F1/F2/F3 “GelSight正面有效接触”
    5) 第一次有效接触后施加轻微 preload，检查最终三点接触能否维持
    6) MuJoCo 实际有效接触点与几何候选预测接触点之间的误差

暂时不做：
    - GWS / epsilon
    - lift test
    - Taxim
    - 强化学习
    - 最终 G* 判定

默认文件：
    XML:
      E:\project\p1\deepgelsight\xml\
      ur3_with_threefinger_gelsight_table_tomato.xml

    candidates:
      E:\project\p1\deepgelsight\grasp_candidates\
      tomato_soup_can\candidates_top50.npz

    envelope:
      E:\project\p1\deepgelsight\gripper_envelope\envelope.npz

运行示例：

    # 先可视化测试第1个候选
    python validate_grasp_candidates_mujoco.py --candidate 1 --viewer

    # 批量测试 Top-10
    python validate_grasp_candidates_mujoco.py --top 10

    # 批量测试 Top-20
    python validate_grasp_candidates_mujoco.py --top 20

输出：
    grasp_candidates/tomato_soup_can/mujoco_validation/
        validation_results.csv
        validation_results.json
        candidate_001_result.npz
        ...

说明：
    candidate 的 T_object_tool 来自几何模板匹配。
    MuJoCo 中物体真实世界位姿 ^W T_O 由仿真直接读取。

    目标工具姿态：
        ^W T_G = ^W T_O @ ^O T_G

    pregrasp：
        在实际 gripper approach axis 上向后退 PREGRASP_DISTANCE_M。

    finger alpha：
        alpha_candidate 是 envelope 匹配得到的理论接触角。
        执行时先打开：
            alpha_pre = alpha_candidate + PREOPEN_EXTRA_DEG
        到达 G_c 后再逐步闭合，最多闭到：
            alpha_candidate - CLOSE_EXTRA_DEG
        或 collision-free envelope 的最小安全角度。

重要：
    这是“候选验证”，不是最终抓取实验。
"""

from pathlib import Path
import argparse
import csv
import json
import math
import sys
import time

import numpy as np
import mujoco
from mujoco import viewer as mj_viewer


# ============================================================================
# 项目路径
# ============================================================================

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "grasp_candidates"
    / "tomato_soup_can"
    / "mujoco_validation"
)


# ============================================================================
# 模型名称
# ============================================================================

TOOL_SITE = "tool0"

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

OBJECT_BODY = "tomato_soup_can"
OBJECT_FREE_JOINT = "tomato_soup_can_free"

OBJECT_COLLISION_PREFIX = "tomato_collision_"

TABLE_GEOMS = [
    "table_top",
]

# 用来检查 gripper-table 几何穿透。
GRIPPER_TABLE_CHECK_GEOMS = [
    "gripper_base_vis",

    "finger1_vis",
    "finger1_gelsight_housing",
    "finger1_gel",

    "finger2_vis",
    "finger2_gelsight_housing",
    "finger2_gel",

    "finger3_vis",
    "finger3_gelsight_housing",
    "finger3_gel",
]


# ============================================================================
# 测试参数
# ============================================================================

# 物体先在桌面上稳定。
OBJECT_SETTLE_STEPS = 1500

# 候选目标姿态前，沿 gripper approach axis 后退 35 mm。
PREGRASP_DISTANCE_M = 0.035

# 几何候选 alpha 基础上，多打开 10°。
PREOPEN_EXTRA_DEG = 10.0

# 到 Gc 后允许继续向内闭 8°，寻找真实接触。
# V3.1：候选 alpha_c 已经是几何接触构型，只允许额外闭合 1° 形成轻微预载。
CLOSE_EXTRA_DEG = 1.0

# 逐步闭合分辨率。
FINGER_CLOSE_STEP_DEG = 0.25

# 每次更新 finger actuator 后模拟多少步。
FINGER_SETTLE_STEPS_PER_INCREMENT = 35

# ---------------- 有效 GelSight 接触判定 ----------------
# 当前 envelope 分析已经确认 3 个 GelSight 的有效接触面都是 local -Z。
# 程序运行时仍优先从 envelope.npz 的 active_face_signs 读取。
DEFAULT_ACTIVE_FACE_SIGNS = [-1, -1, -1]

# MuJoCo contact point 转到 GelSight geom local frame 后，
# 要落在 active face 附近。GelSight box half-thickness = 1.5 mm，
# 这里允许约 1.2 mm 的接触数值/软接触误差。
ACTIVE_FACE_Z_TOL_M = 0.0012

# 有效面四周留一点边缘余量，避免把侧边/棱角接触当正面触觉接触。
ACTIVE_FACE_EDGE_MARGIN_M = 0.0003

# 接触法向应基本垂直于 GelSight 有效面。
# 这里使用 |dot|，避免依赖 MuJoCo geom1/geom2 法向符号约定；
# 记录时再统一把 normal 翻到 active-face outward 方向。
MAX_ACTIVE_FACE_NORMAL_ANGLE_DEG = 45.0

# ---------------- 轻微预载荷 ----------------
# 第一次“有效正面接触”后不立刻冻结，而是再闭合 0.75°，
# 形成轻微 preload，减少另外两指继续闭合时该指脱离。
PRELOAD_AFTER_CONTACT_DEG = 0.75
PRELOAD_SETTLE_STEPS = 120

# 第一次三指都接触后再稳定一下。
FINAL_CONTACT_SETTLE_STEPS = 180

# pregrasp -> Gc 直线接近的 waypoint 数。
APPROACH_WAYPOINTS = 24

# 每个 approach waypoint 后推进一点物理仿真。
APPROACH_SETTLE_STEPS = 8

# gripper-table distance <= 此值认为发生接触/穿透。
TABLE_COLLISION_TOL_M = 0.0

# IK
IK_MAX_ITERS = 350
IK_DAMPING = 0.035
IK_GAIN = 0.70
IK_MAX_STEP_RAD = math.radians(7.0)

IK_POSITION_TOL_M = 0.0015
IK_ROTATION_TOL_RAD = math.radians(1.5)

# 如果 IK 多解困难，使用多个种子。
USE_IK_MULTISTART = True

# viewer 模式下稍微放慢，便于看。
VIEWER_SLEEP_SEC = 0.002


# ============================================================================
# 基础函数
# ============================================================================

def mj_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
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
        raise RuntimeError("遇到零长度向量。")

    return v / n


def pose_matrix(pos, R):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def get_body_pose(data, body_id):
    p = data.xpos[body_id].copy()
    R = data.xmat[body_id].reshape(3, 3).copy()
    return pose_matrix(p, R)


def get_site_pose(data, site_id):
    p = data.site_xpos[site_id].copy()
    R = data.site_xmat[site_id].reshape(3, 3).copy()
    return pose_matrix(p, R)


def transform_points(T, P):
    P = np.asarray(P, dtype=np.float64)

    return (
        T[:3, :3] @ P.T
    ).T + T[:3, 3][None, :]


def world_to_object_points(T_world_object, P_world):
    T_object_world = np.linalg.inv(T_world_object)
    return transform_points(T_object_world, P_world)


def rotation_log(R):
    """
    R 是 target_R @ current_R.T。
    返回 world-frame rotation error vector。
    """
    R = np.asarray(R, dtype=np.float64)

    c = float(
        np.clip(
            (np.trace(R) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
    )

    theta = math.acos(c)

    if theta < 1e-8:
        return np.array(
            [
                0.5 * (R[2, 1] - R[1, 2]),
                0.5 * (R[0, 2] - R[2, 0]),
                0.5 * (R[1, 0] - R[0, 1]),
            ],
            dtype=np.float64,
        )

    if abs(math.pi - theta) < 1e-4:
        A = (R + np.eye(3)) * 0.5

        axis = np.sqrt(
            np.maximum(np.diag(A), 0.0)
        )

        if np.linalg.norm(axis) < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])

        axis = normalize(axis)

        return axis * theta

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(theta))

    return axis * theta


def joint_q(model, data, jid):
    qadr = int(model.jnt_qposadr[jid])
    return float(data.qpos[qadr])


def set_joint_q(model, data, jid, value):
    qadr = int(model.jnt_qposadr[jid])

    if bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        value = float(np.clip(value, lo, hi))

    data.qpos[qadr] = value


def set_ctrl(model, data, aid, value):
    if bool(model.actuator_ctrllimited[aid]):
        lo, hi = model.actuator_ctrlrange[aid]
        value = float(np.clip(value, lo, hi))

    data.ctrl[aid] = value


def sync_arm_ctrl(model, data, ids):
    for jid, aid in zip(
        ids["arm_jids"],
        ids["arm_aids"],
    ):
        set_ctrl(
            model,
            data,
            aid,
            joint_q(model, data, jid),
        )


def sync_finger_ctrl(model, data, ids):
    for jid, aid in zip(
        ids["finger_jids"],
        ids["finger_aids"],
    ):
        set_ctrl(
            model,
            data,
            aid,
            joint_q(model, data, jid),
        )


def set_arm_q(model, data, ids, q):
    for jid, value in zip(
        ids["arm_jids"],
        q,
    ):
        set_joint_q(
            model,
            data,
            jid,
            float(value),
        )

    # 清掉 arm velocity，避免 IK 设置 qpos 后产生残留。
    for jid in ids["arm_jids"]:
        dadr = int(model.jnt_dofadr[jid])
        data.qvel[dadr] = 0.0

    mujoco.mj_forward(model, data)
    sync_arm_ctrl(model, data, ids)


def current_arm_q(model, data, ids):
    return np.array(
        [
            joint_q(model, data, jid)
            for jid in ids["arm_jids"]
        ],
        dtype=np.float64,
    )


def set_all_fingers_kinematic(
    model,
    data,
    ids,
    alpha_rad,
):
    for jid, aid in zip(
        ids["finger_jids"],
        ids["finger_aids"],
    ):
        set_joint_q(
            model,
            data,
            jid,
            alpha_rad,
        )
        set_ctrl(
            model,
            data,
            aid,
            alpha_rad,
        )

        dadr = int(model.jnt_dofadr[jid])
        data.qvel[dadr] = 0.0

    mujoco.mj_forward(model, data)


def step_sim(
    model,
    data,
    n,
    viewer=None,
):
    for _ in range(int(n)):
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

            if VIEWER_SLEEP_SEC > 0:
                time.sleep(VIEWER_SLEEP_SEC)


# ============================================================================
# IK
# ============================================================================

def solve_ik_once(
    model,
    data,
    ids,
    target_pos,
    target_R,
):
    best_q = current_arm_q(
        model,
        data,
        ids,
    )

    best_score = float("inf")
    best_ep = float("inf")
    best_er = float("inf")

    for _ in range(IK_MAX_ITERS):
        mujoco.mj_forward(model, data)

        cur_pos = data.site_xpos[
            ids["tool_sid"]
        ].copy()

        cur_R = data.site_xmat[
            ids["tool_sid"]
        ].reshape(3, 3).copy()

        ep_vec = target_pos - cur_pos
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
            best_q = current_arm_q(
                model,
                data,
                ids,
            )

        if (
            ep <= IK_POSITION_TOL_M
            and er <= IK_ROTATION_TOL_RAD
        ):
            sync_arm_ctrl(
                model,
                data,
                ids,
            )

            return True, ep, er, current_arm_q(
                model,
                data,
                ids,
            )

        jacp = np.zeros(
            (3, model.nv),
            dtype=np.float64,
        )

        jacr = np.zeros(
            (3, model.nv),
            dtype=np.float64,
        )

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
            + (IK_DAMPING ** 2) * np.eye(6)
        )

        try:
            dq = (
                J.T
                @ np.linalg.solve(A, e)
            )
        except np.linalg.LinAlgError:
            dq = np.linalg.pinv(J) @ e

        dq *= IK_GAIN
        dq = np.clip(
            dq,
            -IK_MAX_STEP_RAD,
            +IK_MAX_STEP_RAD,
        )

        q = current_arm_q(
            model,
            data,
            ids,
        )

        set_arm_q(
            model,
            data,
            ids,
            q + dq,
        )

    # 恢复本次迭代最优解。
    set_arm_q(
        model,
        data,
        ids,
        best_q,
    )

    return (
        False,
        best_ep,
        best_er,
        best_q,
    )


def ik_seed_list(home_q, current_q):
    seeds = [
        current_q.copy(),
        home_q.copy(),
    ]

    # UR3 常见的几种 elbow / wrist seed。
    variants_deg = [
        [0, -90, +90, -90, -90, 0],
        [0, -70, +110, -130, -90, 0],
        [30, -90, +90, -90, -90, 0],
        [-30, -90, +90, -90, -90, 0],
        [60, -80, +100, -110, -90, 0],
        [-60, -80, +100, -110, -90, 0],
        [0, -110, +100, -80, +90, 0],
    ]

    for v in variants_deg:
        seeds.append(
            np.radians(
                np.asarray(v, dtype=np.float64)
            )
        )

    return seeds


def solve_ik_multistart(
    model,
    data,
    ids,
    target_pos,
    target_R,
    home_q,
):
    q_before = current_arm_q(
        model,
        data,
        ids,
    )

    seeds = (
        ik_seed_list(
            home_q,
            q_before,
        )
        if USE_IK_MULTISTART
        else [q_before]
    )

    best = None

    for seed in seeds:
        set_arm_q(
            model,
            data,
            ids,
            seed,
        )

        ok, ep, er, q = solve_ik_once(
            model,
            data,
            ids,
            target_pos,
            target_R,
        )

        score = ep + 0.08 * er

        if (
            best is None
            or score < best["score"]
        ):
            best = {
                "ok": ok,
                "ep": ep,
                "er": er,
                "q": q.copy(),
                "score": score,
            }

        if ok:
            break

    set_arm_q(
        model,
        data,
        ids,
        best["q"],
    )

    return (
        bool(best["ok"]),
        float(best["ep"]),
        float(best["er"]),
        best["q"].copy(),
    )


# ============================================================================
# Contact / collision
# ============================================================================

def _gel_active_face_geometry(
    model,
    data,
    ids,
    finger_index,
):
    """
    返回第 i 个 GelSight 当前 world pose 和 active-face 几何。

    active_sign:
        +1 -> local +Z
        -1 -> local -Z
    """
    gid = ids["gel_geom_ids"][finger_index]

    pos_w = data.geom_xpos[gid].copy()
    R_w = data.geom_xmat[gid].reshape(3, 3).copy()

    hx, hy, hz = model.geom_size[gid][:3]

    sign = float(
        ids["active_face_signs"][finger_index]
    )

    active_z = sign * float(hz)

    n_active_w = normalize(
        R_w @ np.array(
            [0.0, 0.0, sign],
            dtype=np.float64,
        )
    )

    return {
        "gid": gid,
        "pos_world": pos_w,
        "R_world": R_w,
        "hx": float(hx),
        "hy": float(hy),
        "hz": float(hz),
        "active_sign": sign,
        "active_z": active_z,
        "active_normal_world": n_active_w,
    }


def _raw_gel_object_contacts(
    model,
    data,
    ids,
    finger_index,
):
    """
    返回某个 GelSight geom 与 tomato collision geoms 的全部 raw contacts。
    这里还没有判断是否落在 GelSight 正面。
    """
    gel_gid = ids["gel_geom_ids"][finger_index]
    object_ids = set(ids["object_geom_ids"])

    out = []

    for ci, c in enumerate(data.contact):
        direct = (
            c.geom1 == gel_gid
            and c.geom2 in object_ids
        )

        reverse = (
            c.geom2 == gel_gid
            and c.geom1 in object_ids
        )

        if not (direct or reverse):
            continue

        # MuJoCo contact.frame[:3] 是接触法向轴，但其符号依赖 geom1/geom2。
        # 这里先按 gel->object 尝试统一；后续还会基于 active face 再统一一次。
        n_raw = np.asarray(
            c.frame[:3],
            dtype=np.float64,
        ).copy()

        if reverse:
            n_raw = -n_raw

        out.append(
            {
                "contact_index": int(ci),
                "dist": float(c.dist),
                "pos_world": np.asarray(
                    c.pos,
                    dtype=np.float64,
                ).copy(),
                "normal_raw_world": normalize(n_raw),
            }
        )

    return out


def classify_tactile_contact(
    model,
    data,
    ids,
    finger_index,
    raw_contact,
):
    """
    把 raw GelSight-object contact 分类为：
        VALID active-face tactile contact
    或
        INVALID side/back/edge contact

    判据：
      1) contact point 转到 gel local frame；
      2) z 必须靠近 active ±Z face；
      3) x/y 必须在有效矩形内，并避开极小边缘；
      4) contact normal 必须近似垂直于 active face。

    返回 diagnostics，便于以后分析。
    """
    g = _gel_active_face_geometry(
        model,
        data,
        ids,
        finger_index,
    )

    p_w = raw_contact["pos_world"]

    p_local = (
        g["R_world"].T
        @ (p_w - g["pos_world"])
    )

    z_error = abs(
        float(p_local[2]) - g["active_z"]
    )

    inside_x = (
        abs(float(p_local[0]))
        <= max(
            0.0,
            g["hx"] - ACTIVE_FACE_EDGE_MARGIN_M,
        )
    )

    inside_y = (
        abs(float(p_local[1]))
        <= max(
            0.0,
            g["hy"] - ACTIVE_FACE_EDGE_MARGIN_M,
        )
    )

    on_active_plane = (
        z_error <= ACTIVE_FACE_Z_TOL_M
    )

    # 法向符号不作为 reject 依据，只看和 face normal 是否平行。
    n_raw = normalize(
        raw_contact["normal_raw_world"]
    )

    signed_dot = float(
        np.clip(
            np.dot(
                n_raw,
                g["active_normal_world"],
            ),
            -1.0,
            1.0,
        )
    )

    normal_angle = math.degrees(
        math.acos(
            np.clip(
                abs(signed_dot),
                0.0,
                1.0,
            )
        )
    )

    normal_ok = (
        normal_angle
        <= MAX_ACTIVE_FACE_NORMAL_ANGLE_DEG
    )

    # 最终统一 normal 朝 active-face outward 方向。
    n_oriented = n_raw.copy()

    if np.dot(
        n_oriented,
        g["active_normal_world"],
    ) < 0:
        n_oriented = -n_oriented

    valid = bool(
        on_active_plane
        and inside_x
        and inside_y
        and normal_ok
    )

    item = dict(raw_contact)

    item.update(
        {
            "valid_tactile_contact": valid,
            "pos_gel_local": p_local.copy(),
            "active_face_z_local": float(
                g["active_z"]
            ),
            "active_face_z_error_m": float(
                z_error
            ),
            "inside_active_x": bool(inside_x),
            "inside_active_y": bool(inside_y),
            "on_active_plane": bool(
                on_active_plane
            ),
            "normal_alignment_ok": bool(
                normal_ok
            ),
            "normal_to_active_face_angle_deg": float(
                normal_angle
            ),
            "normal_signed_dot": float(
                signed_dot
            ),
            "normal_gel_to_object_world": (
                n_oriented.copy()
            ),
            "active_face_normal_world": (
                g[
                    "active_normal_world"
                ].copy()
            ),
        }
    )

    return item


def valid_tactile_contacts_for_finger(
    model,
    data,
    ids,
    finger_index,
):
    raw = _raw_gel_object_contacts(
        model,
        data,
        ids,
        finger_index,
    )

    classified = [
        classify_tactile_contact(
            model,
            data,
            ids,
            finger_index,
            c,
        )
        for c in raw
    ]

    valid = [
        c
        for c in classified
        if c["valid_tactile_contact"]
    ]

    return valid, classified


def representative_valid_tactile_contact_for_finger(
    model,
    data,
    ids,
    finger_index,
):
    """
    只从 GelSight 正面有效接触里选 representative contact。

    优先：
      1) active-face z residual 更小；
      2) contact dist 更小。
    """
    valid, classified = (
        valid_tactile_contacts_for_finger(
            model,
            data,
            ids,
            finger_index,
        )
    )

    if not valid:
        return None, classified

    valid = sorted(
        valid,
        key=lambda c: (
            c["active_face_z_error_m"],
            c["dist"],
        ),
    )

    return valid[0], classified


def object_contact_flags(
    model,
    data,
    ids,
):
    """
    注意：这里返回的是“有效 GelSight 正面触觉接触”，
    不再是 raw box-object contact。
    """
    flags = []
    counts = []

    for i in range(3):
        valid, _ = (
            valid_tactile_contacts_for_finger(
                model,
                data,
                ids,
                i,
            )
        )

        flags.append(
            bool(len(valid) > 0)
        )
        counts.append(
            int(len(valid))
        )

    return flags, counts


def raw_object_contact_counts(
    model,
    data,
    ids,
):
    """
    调试量：记录每根 GelSight box 与 object 的 raw contact 数量，
    即使这些接触发生在 side/back 也会计数。
    """
    return [
        len(
            _raw_gel_object_contacts(
                model,
                data,
                ids,
                i,
            )
        )
        for i in range(3)
    ]


def geom_distance(
    model,
    data,
    gid1,
    gid2,
):
    if not hasattr(
        mujoco,
        "mj_geomDistance",
    ):
        raise RuntimeError(
            "当前 MuJoCo Python 包没有 mj_geomDistance()。"
        )

    fromto = np.zeros(
        6,
        dtype=np.float64,
    )

    d = mujoco.mj_geomDistance(
        model,
        data,
        int(gid1),
        int(gid2),
        0.10,
        fromto,
    )

    return float(d)


def gripper_table_min_distance(
    model,
    data,
    ids,
):
    best = float("inf")
    best_pair = None

    for gid, gname in zip(
        ids["gripper_table_geom_ids"],
        ids["gripper_table_geom_names"],
    ):
        for tid, tname in zip(
            ids["table_geom_ids"],
            ids["table_geom_names"],
        ):
            d = geom_distance(
                model,
                data,
                gid,
                tid,
            )

            if d < best:
                best = d
                best_pair = (
                    gname,
                    tname,
                )

    return float(best), best_pair


# ============================================================================
# Candidate / envelope
# ============================================================================

def load_candidate_data(path):
    data = np.load(
        path,
        allow_pickle=True,
    )

    required = [
        "T_object_tool",
        "score",
        "alpha_deg",
        "pred_contact_object",
        "matched_contact_object",
    ]

    for key in required:
        if key not in data:
            raise RuntimeError(
                f"候选 npz 缺少字段：{key}"
            )

    return data


def load_envelope(path):
    env = np.load(path)

    required = [
        "alpha_deg",
        "surface_center_tool",
    ]

    for key in required:
        if key not in env:
            raise RuntimeError(
                f"envelope.npz 缺少字段：{key}"
            )

    return env


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
                alpha
                - float(alpha_candidate_deg)
            )
        )
    )

    centers = envelope[
        "surface_center_tool"
    ][idx]

    center = np.mean(
        centers,
        axis=0,
    )

    return normalize(center), float(alpha[idx])


def envelope_safe_min_alpha_deg(
    envelope,
):
    alpha = envelope[
        "alpha_deg"
    ].astype(np.float64)

    if (
        "safe_clearance" in envelope
        and np.any(
            envelope["safe_clearance"]
        )
    ):
        mask = envelope[
            "safe_clearance"
        ].astype(bool)

        return float(
            np.min(alpha[mask])
        )

    if (
        "collision_free" in envelope
        and np.any(
            envelope["collision_free"]
        )
    ):
        mask = envelope[
            "collision_free"
        ].astype(bool)

        return float(
            np.min(alpha[mask])
        )

    return float(np.min(alpha))


# ============================================================================
# Test candidate
# ============================================================================

def reset_candidate_state(
    model,
    data,
    ids,
    home_q,
    settled_object_qpos,
):
    key_id = ids["home_key"]

    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(
            model,
            data,
            key_id,
        )
    else:
        mujoco.mj_resetData(
            model,
            data,
        )

    # arm 回 home
    set_arm_q(
        model,
        data,
        ids,
        home_q,
    )

    # 恢复已经稳定在桌面上的物体位姿。
    qadr = ids["object_qadr"]
    dadr = ids["object_dadr"]

    data.qpos[
        qadr:qadr + 7
    ] = settled_object_qpos

    data.qvel[
        dadr:dadr + 6
    ] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    sync_arm_ctrl(
        model,
        data,
        ids,
    )

    sync_finger_ctrl(
        model,
        data,
        ids,
    )


def approach_to_target(
    model,
    data,
    ids,
    home_q,
    target_T_world_tool,
    approach_axis_tool,
    viewer=None,
):
    target_p = target_T_world_tool[
        :3, 3
    ].copy()

    target_R = target_T_world_tool[
        :3, :3
    ].copy()

    approach_axis_world = normalize(
        target_R @ approach_axis_tool
    )

    pre_p = (
        target_p
        - PREGRASP_DISTANCE_M
        * approach_axis_world
    )

    # 先到 pregrasp
    ok_pre, ep_pre, er_pre, _ = (
        solve_ik_multistart(
            model,
            data,
            ids,
            pre_p,
            target_R,
            home_q,
        )
    )

    if not ok_pre:
        return {
            "ok": False,
            "stage": "pregrasp_ik",
            "pregrasp_ik_ok": False,
            "target_ik_ok": False,
            "ep_pre": ep_pre,
            "er_pre": er_pre,
            "ep_target": np.nan,
            "er_target": np.nan,
            "table_collision": False,
            "min_table_distance": np.nan,
            "min_table_pair": None,
        }

    # 从 pregrasp 沿直线接近 Gc。
    table_collision = False
    min_table_distance = float("inf")
    min_table_pair = None

    q_seed = current_arm_q(
        model,
        data,
        ids,
    )

    ep_target = np.nan
    er_target = np.nan

    for wi in range(
        1,
        APPROACH_WAYPOINTS + 1,
    ):
        u = wi / APPROACH_WAYPOINTS

        p = (
            (1.0 - u) * pre_p
            + u * target_p
        )

        # waypoint 之间用当前解作为初值，
        # 不做全 multistart，保证轨迹连续。
        ok, ep, er, q = solve_ik_once(
            model,
            data,
            ids,
            p,
            target_R,
        )

        if not ok:
            # 对当前 waypoint 再尝试 multistart。
            ok, ep, er, q = (
                solve_ik_multistart(
                    model,
                    data,
                    ids,
                    p,
                    target_R,
                    home_q,
                )
            )

        if not ok:
            return {
                "ok": False,
                "stage": "approach_ik",
                "pregrasp_ik_ok": True,
                "target_ik_ok": False,
                "ep_pre": ep_pre,
                "er_pre": er_pre,
                "ep_target": ep,
                "er_target": er,
                "table_collision": table_collision,
                "min_table_distance": min_table_distance,
                "min_table_pair": min_table_pair,
            }

        q_seed = q.copy()

        # 几何 table check
        d_table, pair = (
            gripper_table_min_distance(
                model,
                data,
                ids,
            )
        )

        if d_table < min_table_distance:
            min_table_distance = d_table
            min_table_pair = pair

        if (
            d_table
            <= TABLE_COLLISION_TOL_M
        ):
            table_collision = True

            return {
                "ok": False,
                "stage": "table_collision",
                "pregrasp_ik_ok": True,
                "target_ik_ok": False,
                "ep_pre": ep_pre,
                "er_pre": er_pre,
                "ep_target": ep,
                "er_target": er,
                "table_collision": True,
                "min_table_distance": min_table_distance,
                "min_table_pair": min_table_pair,
            }

        step_sim(
            model,
            data,
            APPROACH_SETTLE_STEPS,
            viewer=viewer,
        )

        ep_target = ep
        er_target = er

    return {
        "ok": True,
        "stage": "target_reached",
        "pregrasp_ik_ok": True,
        "target_ik_ok": True,
        "ep_pre": ep_pre,
        "er_pre": er_pre,
        "ep_target": ep_target,
        "er_target": er_target,
        "table_collision": table_collision,
        "min_table_distance": min_table_distance,
        "min_table_pair": min_table_pair,
    }


def close_fingers_until_contacts(
    model,
    data,
    ids,
    alpha_start_deg,
    alpha_candidate_deg,
    safe_min_alpha_deg,
    viewer=None,
):
    """
    三指从 alpha_start 向闭合方向移动。

    V3 改动：
      1) 只有 GelSight active face 的有效正面接触才算“触觉接触”；
      2) 第一次有效接触后不立即冻结；
      3) 给该指额外 PRELOAD_AFTER_CONTACT_DEG 的轻微闭合量；
      4) 其余未接触手指继续搜索；
      5) 最终 THREE-CONTACT 也只按有效正面接触判定。
    """
    alpha_limit_deg = max(
        safe_min_alpha_deg,
        alpha_candidate_deg
        - CLOSE_EXTRA_DEG,
    )

    targets_deg = np.full(
        3,
        float(alpha_start_deg),
        dtype=np.float64,
    )

    # SEARCH -> PRELOAD/HOLD
    states = [
        "SEARCH",
        "SEARCH",
        "SEARCH",
    ]

    first_contacts = [
        None,
        None,
        None,
    ]

    ever_contact = [
        False,
        False,
        False,
    ]

    raw_contact_ever = [
        False,
        False,
        False,
    ]

    invalid_contact_diagnostics = [
        [],
        [],
        [],
    ]

    preload_target_deg = np.full(
        3,
        np.nan,
        dtype=np.float64,
    )

    # 允许足够循环完成 search + preload。
    max_iter = int(
        math.ceil(
            max(
                0.0,
                alpha_start_deg
                - alpha_limit_deg,
            )
            / FINGER_CLOSE_STEP_DEG
        )
    ) + 20

    for _ in range(max_iter):
        # ----------------------------------------------------------
        # A. 读取当前接触
        # ----------------------------------------------------------
        for i in range(3):
            raw_contacts = _raw_gel_object_contacts(
                model,
                data,
                ids,
                i,
            )

            if raw_contacts:
                raw_contact_ever[i] = True

            rep, classified = (
                representative_valid_tactile_contact_for_finger(
                    model,
                    data,
                    ids,
                    i,
                )
            )

            # 记录少量 invalid 诊断，避免文件过大。
            if (
                rep is None
                and classified
                and len(
                    invalid_contact_diagnostics[i]
                ) < 5
            ):
                # 只存当前最“接近有效面”的那个 raw contact
                best_bad = min(
                    classified,
                    key=lambda c: (
                        c[
                            "active_face_z_error_m"
                        ],
                        c[
                            "normal_to_active_face_angle_deg"
                        ],
                    ),
                )

                invalid_contact_diagnostics[i].append(
                    {
                        "pos_gel_local": (
                            best_bad[
                                "pos_gel_local"
                            ].copy()
                        ),
                        "z_error_m": float(
                            best_bad[
                                "active_face_z_error_m"
                            ]
                        ),
                        "normal_angle_deg": float(
                            best_bad[
                                "normal_to_active_face_angle_deg"
                            ]
                        ),
                        "inside_x": bool(
                            best_bad[
                                "inside_active_x"
                            ]
                        ),
                        "inside_y": bool(
                            best_bad[
                                "inside_active_y"
                            ]
                        ),
                        "on_active_plane": bool(
                            best_bad[
                                "on_active_plane"
                            ]
                        ),
                    }
                )

            # 只在 SEARCH 状态首次触发 valid tactile contact。
            if (
                states[i] == "SEARCH"
                and rep is not None
            ):
                ever_contact[i] = True
                states[i] = "PRELOAD"

                current_q_deg = float(
                    math.degrees(
                        joint_q(
                            model,
                            data,
                            ids["finger_jids"][i],
                        )
                    )
                )

                preload_target_deg[i] = max(
                    alpha_limit_deg,
                    current_q_deg
                    - PRELOAD_AFTER_CONTACT_DEG,
                )

                targets_deg[i] = (
                    preload_target_deg[i]
                )

                set_ctrl(
                    model,
                    data,
                    ids["finger_aids"][i],
                    math.radians(
                        preload_target_deg[i]
                    ),
                )

                T_world_object_now = get_body_pose(
                    data,
                    ids["object_bid"],
                )

                p_obj = world_to_object_points(
                    T_world_object_now,
                    rep["pos_world"][None, :],
                )[0]

                first_contacts[i] = {
                    "alpha_deg": current_q_deg,
                    "preload_target_deg": float(
                        preload_target_deg[i]
                    ),
                    "pos_world": (
                        rep["pos_world"].copy()
                    ),
                    "pos_object": p_obj.copy(),
                    "pos_gel_local": (
                        rep["pos_gel_local"].copy()
                    ),
                    "active_face_z_error_m": float(
                        rep[
                            "active_face_z_error_m"
                        ]
                    ),
                    "normal_to_active_face_angle_deg": float(
                        rep[
                            "normal_to_active_face_angle_deg"
                        ]
                    ),
                    "normal_world": (
                        rep[
                            "normal_gel_to_object_world"
                        ].copy()
                    ),
                    "active_face_normal_world": (
                        rep[
                            "active_face_normal_world"
                        ].copy()
                    ),
                    "dist": float(rep["dist"]),
                }

        # ----------------------------------------------------------
        # B. 三指都已找到有效接触 -> 让 preload 生效后结束搜索
        # ----------------------------------------------------------
        if all(ever_contact):
            step_sim(
                model,
                data,
                PRELOAD_SETTLE_STEPS,
                viewer=viewer,
            )
            break

        # ----------------------------------------------------------
        # C. 只有 SEARCH fingers 继续闭合。
        #    PRELOAD fingers 始终保持 preload target。
        # ----------------------------------------------------------
        any_motion = False

        for i in range(3):
            if states[i] != "SEARCH":
                continue

            if (
                targets_deg[i]
                <= alpha_limit_deg
                + 1e-9
            ):
                continue

            targets_deg[i] = max(
                alpha_limit_deg,
                targets_deg[i]
                - FINGER_CLOSE_STEP_DEG,
            )

            set_ctrl(
                model,
                data,
                ids["finger_aids"][i],
                math.radians(
                    targets_deg[i]
                ),
            )

            any_motion = True

        if not any_motion:
            # 没有任何 SEARCH 指还能继续闭合。
            # 仍然让已经发生的 preload 稳定一下。
            step_sim(
                model,
                data,
                PRELOAD_SETTLE_STEPS,
                viewer=viewer,
            )
            break

        step_sim(
            model,
            data,
            FINGER_SETTLE_STEPS_PER_INCREMENT,
            viewer=viewer,
        )

    # --------------------------------------------------------------
    # D. 最终稳定
    # --------------------------------------------------------------
    step_sim(
        model,
        data,
        FINAL_CONTACT_SETTLE_STEPS,
        viewer=viewer,
    )

    final_flags, final_counts = (
        object_contact_flags(
            model,
            data,
            ids,
        )
    )

    raw_final_counts = (
        raw_object_contact_counts(
            model,
            data,
            ids,
        )
    )

    final_contacts = [
        None,
        None,
        None,
    ]

    for i in range(3):
        rep, _ = (
            representative_valid_tactile_contact_for_finger(
                model,
                data,
                ids,
                i,
            )
        )

        if rep is None:
            continue

        T_world_object_now = get_body_pose(
            data,
            ids["object_bid"],
        )

        p_obj = world_to_object_points(
            T_world_object_now,
            rep["pos_world"][None, :],
        )[0]

        final_contacts[i] = {
            "pos_world": rep["pos_world"].copy(),
            "pos_object": p_obj.copy(),
            "pos_gel_local": rep[
                "pos_gel_local"
            ].copy(),
            "normal_world": rep[
                "normal_gel_to_object_world"
            ].copy(),
            "normal_to_active_face_angle_deg": float(
                rep[
                    "normal_to_active_face_angle_deg"
                ]
            ),
            "active_face_z_error_m": float(
                rep[
                    "active_face_z_error_m"
                ]
            ),
            "dist": float(rep["dist"]),
        }

    return {
        "ever_contact": ever_contact,
        "ever_three_contact": bool(
            all(ever_contact)
        ),
        "raw_contact_ever": raw_contact_ever,

        "final_contact": final_flags,
        "final_three_contact": bool(
            all(final_flags)
        ),
        "final_counts": final_counts,
        "raw_final_counts": raw_final_counts,

        "first_contacts": first_contacts,
        "final_contacts": final_contacts,
        "invalid_contact_diagnostics": (
            invalid_contact_diagnostics
        ),

        "alpha_limit_deg": float(
            alpha_limit_deg
        ),
        "preload_target_deg": (
            preload_target_deg.copy()
        ),

        "final_finger_q_deg": [
            float(
                math.degrees(
                    joint_q(
                        model,
                        data,
                        jid,
                    )
                )
            )
            for jid in ids["finger_jids"]
        ],
    }


def compare_contacts_to_candidate(
    first_contacts,
    predicted_contacts_object,
):
    actual = np.full(
        (3, 3),
        np.nan,
        dtype=np.float64,
    )

    errors = np.full(
        3,
        np.nan,
        dtype=np.float64,
    )

    for i in range(3):
        if first_contacts[i] is None:
            continue

        actual[i] = (
            first_contacts[i][
                "pos_object"
            ]
        )

        errors[i] = np.linalg.norm(
            actual[i]
            - predicted_contacts_object[i]
        )

    valid = np.isfinite(errors)

    if np.any(valid):
        mean_error = float(
            np.mean(errors[valid])
        )
        max_error = float(
            np.max(errors[valid])
        )
    else:
        mean_error = np.nan
        max_error = np.nan

    return {
        "actual_contact_object": actual,
        "contact_error_m": errors,
        "mean_contact_error_m": mean_error,
        "max_contact_error_m": max_error,
    }


def test_one_candidate(
    rank,
    candidate_data,
    envelope,
    model,
    data,
    ids,
    home_q,
    settled_object_qpos,
    viewer=None,
):
    idx = rank - 1

    T_object_tool = candidate_data[
        "T_object_tool"
    ][idx].astype(np.float64)

    alpha_candidate_deg = float(
        candidate_data[
            "alpha_deg"
        ][idx]
    )

    score = float(
        candidate_data[
            "score"
        ][idx]
    )

    predicted_contacts_object = (
        candidate_data[
            "matched_contact_object"
        ][idx].astype(np.float64)
    )

    approach_axis_tool, envelope_alpha = (
        envelope_approach_axis_tool(
            envelope,
            alpha_candidate_deg,
        )
    )

    safe_min_alpha_deg = (
        envelope_safe_min_alpha_deg(
            envelope
        )
    )

    alpha_pre_deg = min(
        30.0,
        alpha_candidate_deg
        + PREOPEN_EXTRA_DEG,
    )

    # 重置环境
    reset_candidate_state(
        model,
        data,
        ids,
        home_q,
        settled_object_qpos,
    )

    # 手指先打开
    set_all_fingers_kinematic(
        model,
        data,
        ids,
        math.radians(alpha_pre_deg),
    )

    # 当前 object world pose
    T_world_object_start = get_body_pose(
        data,
        ids["object_bid"],
    )

    # candidate world pose
    T_world_tool_target = (
        T_world_object_start
        @ T_object_tool
    )

    # approach
    approach_result = approach_to_target(
        model,
        data,
        ids,
        home_q,
        T_world_tool_target,
        approach_axis_tool,
        viewer=viewer,
    )

    base_result = {
        "rank": int(rank),
        "score": score,
        "alpha_candidate_deg": alpha_candidate_deg,
        "envelope_alpha_deg": envelope_alpha,
        "alpha_pre_deg": alpha_pre_deg,
        "safe_min_alpha_deg": safe_min_alpha_deg,
        "T_object_tool": T_object_tool.copy(),
        "T_world_tool_target": (
            T_world_tool_target.copy()
        ),
        "approach_axis_tool": (
            approach_axis_tool.copy()
        ),
        "approach": approach_result,
    }

    if not approach_result["ok"]:
        base_result.update(
            {
                "ever_three_contact": False,
                "final_three_contact": False,
                "ever_contact": [
                    False,
                    False,
                    False,
                ],
                "final_contact": [
                    False,
                    False,
                    False,
                ],
                "mean_contact_error_m": np.nan,
                "max_contact_error_m": np.nan,
                "contact_error_m": np.full(
                    3,
                    np.nan,
                ),
                "actual_contact_object": np.full(
                    (3, 3),
                    np.nan,
                ),
                "raw_contact_ever": [
                    False,
                    False,
                    False,
                ],
                "raw_final_counts": [
                    0,
                    0,
                    0,
                ],
                "first_contacts": [
                    None,
                    None,
                    None,
                ],
                "final_contacts": [
                    None,
                    None,
                    None,
                ],
                "invalid_contact_diagnostics": [
                    [],
                    [],
                    [],
                ],
                "preload_target_deg": np.full(
                    3,
                    np.nan,
                ),
            }
        )

        return base_result

    # 到 Gc 后慢慢闭合
    close_result = (
        close_fingers_until_contacts(
            model,
            data,
            ids,
            alpha_pre_deg,
            alpha_candidate_deg,
            safe_min_alpha_deg,
            viewer=viewer,
        )
    )

    comp = compare_contacts_to_candidate(
        close_result[
            "first_contacts"
        ],
        predicted_contacts_object,
    )

    # object 在过程中的位姿变化
    T_world_object_end = get_body_pose(
        data,
        ids["object_bid"],
    )

    object_translation_shift = float(
        np.linalg.norm(
            T_world_object_end[:3, 3]
            - T_world_object_start[:3, 3]
        )
    )

    base_result.update(
        {
            "ever_three_contact": (
                close_result[
                    "ever_three_contact"
                ]
            ),
            "final_three_contact": (
                close_result[
                    "final_three_contact"
                ]
            ),
            "ever_contact": (
                close_result[
                    "ever_contact"
                ]
            ),
            "final_contact": (
                close_result[
                    "final_contact"
                ]
            ),
            "final_counts": (
                close_result[
                    "final_counts"
                ]
            ),
            "raw_contact_ever": (
                close_result[
                    "raw_contact_ever"
                ]
            ),
            "raw_final_counts": (
                close_result[
                    "raw_final_counts"
                ]
            ),
            "first_contacts": (
                close_result[
                    "first_contacts"
                ]
            ),
            "final_contacts": (
                close_result[
                    "final_contacts"
                ]
            ),
            "invalid_contact_diagnostics": (
                close_result[
                    "invalid_contact_diagnostics"
                ]
            ),
            "preload_target_deg": (
                close_result[
                    "preload_target_deg"
                ]
            ),
            "alpha_limit_deg": (
                close_result[
                    "alpha_limit_deg"
                ]
            ),
            "final_finger_q_deg": (
                close_result[
                    "final_finger_q_deg"
                ]
            ),
            "actual_contact_object": (
                comp[
                    "actual_contact_object"
                ]
            ),
            "contact_error_m": (
                comp[
                    "contact_error_m"
                ]
            ),
            "mean_contact_error_m": (
                comp[
                    "mean_contact_error_m"
                ]
            ),
            "max_contact_error_m": (
                comp[
                    "max_contact_error_m"
                ]
            ),
            "object_translation_shift_m": (
                object_translation_shift
            ),
            "T_world_object_start": (
                T_world_object_start
            ),
            "T_world_object_end": (
                T_world_object_end
            ),
        }
    )

    return base_result


# ============================================================================
# Save
# ============================================================================

def result_to_jsonable(r):
    return {
        "rank": r["rank"],
        "score": r["score"],
        "alpha_candidate_deg": (
            r["alpha_candidate_deg"]
        ),
        "alpha_pre_deg": r["alpha_pre_deg"],

        "approach_stage": (
            r["approach"]["stage"]
        ),
        "pregrasp_ik_ok": bool(
            r["approach"][
                "pregrasp_ik_ok"
            ]
        ),
        "target_ik_ok": bool(
            r["approach"][
                "target_ik_ok"
            ]
        ),

        "pregrasp_position_error_mm": (
            float(
                r["approach"]["ep_pre"]
                * 1000.0
            )
            if np.isfinite(
                r["approach"]["ep_pre"]
            )
            else None
        ),

        "pregrasp_rotation_error_deg": (
            float(
                math.degrees(
                    r["approach"]["er_pre"]
                )
            )
            if np.isfinite(
                r["approach"]["er_pre"]
            )
            else None
        ),

        "target_position_error_mm": (
            float(
                r["approach"]["ep_target"]
                * 1000.0
            )
            if np.isfinite(
                r["approach"]["ep_target"]
            )
            else None
        ),

        "target_rotation_error_deg": (
            float(
                math.degrees(
                    r["approach"]["er_target"]
                )
            )
            if np.isfinite(
                r["approach"]["er_target"]
            )
            else None
        ),

        "table_collision": bool(
            r["approach"][
                "table_collision"
            ]
        ),

        "min_table_distance_mm": (
            float(
                r["approach"][
                    "min_table_distance"
                ]
                * 1000.0
            )
            if np.isfinite(
                r["approach"][
                    "min_table_distance"
                ]
            )
            else None
        ),

        "min_table_pair": (
            list(
                r["approach"][
                    "min_table_pair"
                ]
            )
            if r["approach"][
                "min_table_pair"
            ] is not None
            else None
        ),

        "ever_contact": [
            bool(x)
            for x in r["ever_contact"]
        ],

        "ever_three_contact": bool(
            r["ever_three_contact"]
        ),

        "final_contact": [
            bool(x)
            for x in r["final_contact"]
        ],

        "final_three_contact": bool(
            r["final_three_contact"]
        ),

        "mean_contact_error_mm": (
            float(
                r["mean_contact_error_m"]
                * 1000.0
            )
            if np.isfinite(
                r["mean_contact_error_m"]
            )
            else None
        ),

        "max_contact_error_mm": (
            float(
                r["max_contact_error_m"]
                * 1000.0
            )
            if np.isfinite(
                r["max_contact_error_m"]
            )
            else None
        ),

        "contact_error_mm": [
            (
                float(x * 1000.0)
                if np.isfinite(x)
                else None
            )
            for x in r["contact_error_m"]
        ],

        "object_translation_shift_mm": (
            float(
                r.get(
                    "object_translation_shift_m",
                    np.nan,
                )
                * 1000.0
            )
            if np.isfinite(
                r.get(
                    "object_translation_shift_m",
                    np.nan,
                )
            )
            else None
        ),

        "final_finger_q_deg": (
            [
                float(x)
                for x in r.get(
                    "final_finger_q_deg",
                    [],
                )
            ]
        ),

        "raw_contact_ever": [
            bool(x)
            for x in r.get(
                "raw_contact_ever",
                [False, False, False],
            )
        ],

        "raw_final_counts": [
            int(x)
            for x in r.get(
                "raw_final_counts",
                [0, 0, 0],
            )
        ],

        "preload_target_deg": [
            (
                float(x)
                if np.isfinite(x)
                else None
            )
            for x in np.asarray(
                r.get(
                    "preload_target_deg",
                    np.full(3, np.nan),
                )
            )
        ],

        "first_contact_gel_local_mm": [
            (
                (
                    np.asarray(c["pos_gel_local"])
                    * 1000.0
                ).tolist()
                if c is not None
                else None
            )
            for c in r.get(
                "first_contacts",
                [None, None, None],
            )
        ],

        "first_contact_face_normal_angle_deg": [
            (
                float(
                    c[
                        "normal_to_active_face_angle_deg"
                    ]
                )
                if c is not None
                else None
            )
            for c in r.get(
                "first_contacts",
                [None, None, None],
            )
        ],
    }


def save_one_result(r):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = OUTPUT_DIR / (
        f"candidate_{r['rank']:03d}_result.npz"
    )

    np.savez_compressed(
        path,
        rank=np.array(r["rank"]),
        score=np.array(r["score"]),
        alpha_candidate_deg=np.array(
            r["alpha_candidate_deg"]
        ),
        T_object_tool=r["T_object_tool"],
        T_world_tool_target=(
            r["T_world_tool_target"]
        ),
        approach_axis_tool=(
            r["approach_axis_tool"]
        ),
        ever_contact=np.asarray(
            r["ever_contact"],
            dtype=bool,
        ),
        final_contact=np.asarray(
            r["final_contact"],
            dtype=bool,
        ),
        ever_three_contact=np.array(
            r["ever_three_contact"]
        ),
        final_three_contact=np.array(
            r["final_three_contact"]
        ),
        actual_contact_object=(
            r["actual_contact_object"]
        ),
        contact_error_m=(
            r["contact_error_m"]
        ),
        mean_contact_error_m=np.array(
            r["mean_contact_error_m"]
        ),
        max_contact_error_m=np.array(
            r["max_contact_error_m"]
        ),
    )


def save_summary(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    jsonable = [
        result_to_jsonable(r)
        for r in results
    ]

    json_path = (
        OUTPUT_DIR
        / "validation_results.json"
    )

    json_path.write_text(
        json.dumps(
            jsonable,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = (
        OUTPUT_DIR
        / "validation_results.csv"
    )

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
                "alpha_candidate_deg",
                "approach_stage",
                "pregrasp_ik_ok",
                "target_ik_ok",
                "table_collision",
                "min_table_distance_mm",

                "F1_raw_ever",
                "F2_raw_ever",
                "F3_raw_ever",

                "F1_valid_ever",
                "F2_valid_ever",
                "F3_valid_ever",
                "ever_three_valid_contact",

                "F1_valid_final",
                "F2_valid_final",
                "F3_valid_final",
                "final_three_valid_contact",

                "mean_contact_error_mm",
                "max_contact_error_mm",

                "F1_contact_error_mm",
                "F2_contact_error_mm",
                "F3_contact_error_mm",

                "object_translation_shift_mm",
            ]
        )

        for j in jsonable:
            ce = j["contact_error_mm"]

            writer.writerow(
                [
                    j["rank"],
                    j["score"],
                    j["alpha_candidate_deg"],
                    j["approach_stage"],
                    int(
                        j["pregrasp_ik_ok"]
                    ),
                    int(
                        j["target_ik_ok"]
                    ),
                    int(
                        j["table_collision"]
                    ),
                    j["min_table_distance_mm"],

                    int(j["raw_contact_ever"][0]),
                    int(j["raw_contact_ever"][1]),
                    int(j["raw_contact_ever"][2]),

                    int(j["ever_contact"][0]),
                    int(j["ever_contact"][1]),
                    int(j["ever_contact"][2]),
                    int(
                        j[
                            "ever_three_contact"
                        ]
                    ),

                    int(j["final_contact"][0]),
                    int(j["final_contact"][1]),
                    int(j["final_contact"][2]),
                    int(
                        j[
                            "final_three_contact"
                        ]
                    ),

                    j["mean_contact_error_mm"],
                    j["max_contact_error_mm"],

                    ce[0],
                    ce[1],
                    ce[2],

                    j[
                        "object_translation_shift_mm"
                    ],
                ]
            )

    return csv_path, json_path


# ============================================================================
# Setup
# ============================================================================

def build_ids(model):
    ids = {}

    ids["tool_sid"] = mj_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        TOOL_SITE,
    )

    ids["arm_jids"] = [
        mj_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            x,
        )
        for x in ARM_JOINTS
    ]

    ids["arm_aids"] = [
        mj_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            x,
        )
        for x in ARM_ACTUATORS
    ]

    ids["arm_dof_adrs"] = [
        int(
            model.jnt_dofadr[jid]
        )
        for jid in ids["arm_jids"]
    ]

    ids["finger_jids"] = [
        mj_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            x,
        )
        for x in FINGER_JOINTS
    ]

    ids["finger_aids"] = [
        mj_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            x,
        )
        for x in FINGER_ACTUATORS
    ]

    ids["gel_geom_ids"] = [
        mj_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            x,
        )
        for x in GEL_GEOMS
    ]

    ids["object_bid"] = mj_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        OBJECT_BODY,
    )

    object_jid = mj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        OBJECT_FREE_JOINT,
    )

    ids["object_jid"] = object_jid
    ids["object_qadr"] = int(
        model.jnt_qposadr[
            object_jid
        ]
    )
    ids["object_dadr"] = int(
        model.jnt_dofadr[
            object_jid
        ]
    )

    object_geom_ids = []

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
            object_geom_ids.append(gid)

    if not object_geom_ids:
        raise RuntimeError(
            "没有找到 tomato_collision_* geoms。"
        )

    ids["object_geom_ids"] = (
        object_geom_ids
    )

    gripper_ids = []
    gripper_names = []

    for name in GRIPPER_TABLE_CHECK_GEOMS:
        gid = optional_geom_id(
            model,
            name,
        )

        if gid is not None:
            gripper_ids.append(gid)
            gripper_names.append(name)

    if not gripper_ids:
        raise RuntimeError(
            "没有找到用于 table collision check 的 gripper geoms。"
        )

    ids[
        "gripper_table_geom_ids"
    ] = gripper_ids

    ids[
        "gripper_table_geom_names"
    ] = gripper_names

    table_ids = []
    table_names = []

    for name in TABLE_GEOMS:
        gid = optional_geom_id(
            model,
            name,
        )

        if gid is not None:
            table_ids.append(gid)
            table_names.append(name)

    if not table_ids:
        raise RuntimeError(
            "没有找到 table_top geom。"
        )

    ids["table_geom_ids"] = table_ids
    ids["table_geom_names"] = (
        table_names
    )

    ids["home_key"] = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home",
        )
    )

    return ids


def prepare_initial_state(
    model,
    data,
    ids,
):
    if ids["home_key"] >= 0:
        mujoco.mj_resetDataKeyframe(
            model,
            data,
            ids["home_key"],
        )
    else:
        mujoco.mj_resetData(
            model,
            data,
        )

    mujoco.mj_forward(
        model,
        data,
    )

    sync_arm_ctrl(
        model,
        data,
        ids,
    )

    sync_finger_ctrl(
        model,
        data,
        ids,
    )

    home_q = current_arm_q(
        model,
        data,
        ids,
    )

    print(
        f"[Init] 让 tomato 在桌面稳定 "
        f"{OBJECT_SETTLE_STEPS} steps ..."
    )

    step_sim(
        model,
        data,
        OBJECT_SETTLE_STEPS,
    )

    qadr = ids["object_qadr"]

    settled_object_qpos = (
        data.qpos[
            qadr:qadr + 7
        ].copy()
    )

    T_world_object = get_body_pose(
        data,
        ids["object_bid"],
    )

    print(
        "[Init] settled tomato pos =",
        np.round(
            T_world_object[:3, 3],
            6,
        ),
    )

    return (
        home_q,
        settled_object_qpos,
    )


# ============================================================================
# Console
# ============================================================================

def print_result(r):
    a = r["approach"]

    print(
        f"\nCandidate #{r['rank']:03d} "
        f"| score={r['score']:.3f} "
        f"| alpha={r['alpha_candidate_deg']:+.1f}°"
    )

    print(
        "  approach stage :",
        a["stage"],
    )

    print(
        "  pregrasp IK    :",
        "PASS"
        if a["pregrasp_ik_ok"]
        else "FAIL",
    )

    print(
        "  target IK      :",
        "PASS"
        if a["target_ik_ok"]
        else "FAIL",
    )

    print(
        "  table collision:",
        "YES"
        if a["table_collision"]
        else "NO",
    )

    if np.isfinite(
        a["min_table_distance"]
    ):
        print(
            "  min table dist :",
            f"{a['min_table_distance']*1000:.3f} mm",
            a["min_table_pair"],
        )

    print(
        "  ever contact   :",
        r["ever_contact"],
    )

    print(
        "  THREE ever     :",
        r["ever_three_contact"],
    )

    print(
        "  raw ever       :",
        r.get(
            "raw_contact_ever",
            [False, False, False],
        ),
    )

    print(
        "  valid ever     :",
        r["ever_contact"],
    )

    print(
        "  THREE valid ever:",
        r["ever_three_contact"],
    )

    print(
        "  valid final    :",
        r["final_contact"],
    )

    print(
        "  THREE valid final:",
        r["final_three_contact"],
    )

    local_contacts = []

    for c in r.get(
        "first_contacts",
        [None, None, None],
    ):
        if c is None:
            local_contacts.append(None)
        else:
            local_contacts.append(
                np.round(
                    c["pos_gel_local"]
                    * 1000.0,
                    3,
                ).tolist()
            )

    print(
        "  first gel local:",
        local_contacts,
        "mm",
    )

    if np.isfinite(
        r["mean_contact_error_m"]
    ):
        print(
            "  contact error  :",
            np.round(
                r["contact_error_m"]
                * 1000.0,
                3,
            ),
            "mm",
        )

        print(
            "  mean / max err :",
            f"{r['mean_contact_error_m']*1000:.3f} / "
            f"{r['max_contact_error_m']*1000:.3f} mm",
        )


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="批量测试 Top-N，默认10",
    )

    parser.add_argument(
        "--candidate",
        type=int,
        default=None,
        help="只测试指定 rank，例如 --candidate 1",
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
        help="打开 MuJoCo Viewer；建议和 --candidate 一起使用",
    )

    args = parser.parse_args()

    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"找不到 XML：\n{XML_PATH}"
        )

    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            f"找不到 candidates：\n{CANDIDATE_PATH}"
        )

    if not ENVELOPE_PATH.exists():
        raise FileNotFoundError(
            f"找不到 envelope：\n{ENVELOPE_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_data = (
        load_candidate_data(
            CANDIDATE_PATH
        )
    )

    envelope = load_envelope(
        ENVELOPE_PATH
    )

    n_candidates = len(
        candidate_data[
            "T_object_tool"
        ]
    )

    if args.candidate is not None:
        ranks = [
            int(args.candidate)
        ]
    else:
        ranks = list(
            range(
                1,
                min(
                    int(args.top),
                    n_candidates,
                )
                + 1,
            )
        )

    for rank in ranks:
        if (
            rank < 1
            or rank > n_candidates
        ):
            raise ValueError(
                f"candidate rank 超范围：{rank}"
            )

    print("=" * 86)
    print("MuJoCo Geometric Grasp Candidate Validation")
    print("=" * 86)

    print("XML:")
    print("  ", XML_PATH)

    print("Candidates:")
    print("  ", CANDIDATE_PATH)

    print("Ranks:")
    print("  ", ranks)

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(
        model
    )

    ids = build_ids(
        model
    )

    if (
        "active_face_signs" in envelope
        and len(
            envelope[
                "active_face_signs"
            ]
        ) == 3
    ):
        ids["active_face_signs"] = (
            envelope[
                "active_face_signs"
            ].astype(np.int32)
        )
    else:
        ids["active_face_signs"] = np.asarray(
            DEFAULT_ACTIVE_FACE_SIGNS,
            dtype=np.int32,
        )

    print(
        "[Contact] GelSight active-face signs =",
        ids["active_face_signs"].tolist(),
    )
    print(
        "[Contact] active-face Z tolerance = "
        f"{ACTIVE_FACE_Z_TOL_M*1000:.2f} mm"
    )
    print(
        "[Contact] edge margin = "
        f"{ACTIVE_FACE_EDGE_MARGIN_M*1000:.2f} mm"
    )
    print(
        "[Contact] normal angle <= "
        f"{MAX_ACTIVE_FACE_NORMAL_ANGLE_DEG:.1f} deg"
    )
    print(
        "[Contact] preload after first valid contact = "
        f"{PRELOAD_AFTER_CONTACT_DEG:.2f} deg"
    )

    home_q, settled_object_qpos = (
        prepare_initial_state(
            model,
            data,
            ids,
        )
    )

    results = []

    viewer = None

    if args.viewer:
        viewer = mj_viewer.launch_passive(
            model,
            data,
        )

    try:
        for rank in ranks:
            r = test_one_candidate(
                rank,
                candidate_data,
                envelope,
                model,
                data,
                ids,
                home_q,
                settled_object_qpos,
                viewer=viewer,
            )

            results.append(r)

            print_result(r)

            save_one_result(r)

    finally:
        if viewer is not None:
            viewer.close()

    csv_path, json_path = (
        save_summary(
            results
        )
    )

    print("\n" + "=" * 86)
    print("Validation summary")
    print("=" * 86)

    n = len(results)

    pre_ok = sum(
        int(
            r["approach"][
                "pregrasp_ik_ok"
            ]
        )
        for r in results
    )

    target_ok = sum(
        int(
            r["approach"][
                "target_ik_ok"
            ]
        )
        for r in results
    )

    no_table = sum(
        int(
            not r["approach"][
                "table_collision"
            ]
        )
        for r in results
    )

    ever_three = sum(
        int(
            r[
                "ever_three_contact"
            ]
        )
        for r in results
    )

    final_three = sum(
        int(
            r[
                "final_three_contact"
            ]
        )
        for r in results
    )

    print(
        f"tested              : {n}"
    )
    print(
        f"pregrasp IK PASS    : {pre_ok}/{n}"
    )
    print(
        f"target IK PASS      : {target_ok}/{n}"
    )
    print(
        f"no table collision  : {no_table}/{n}"
    )
    print(
        f"3-contact EVER      : {ever_three}/{n}"
    )
    print(
        f"3-contact FINAL     : {final_three}/{n}"
    )

    valid_errors = [
        r[
            "mean_contact_error_m"
        ]
        for r in results
        if np.isfinite(
            r[
                "mean_contact_error_m"
            ]
        )
    ]

    if valid_errors:
        print(
            "mean contact error  : "
            f"{np.mean(valid_errors)*1000:.3f} mm"
        )

    print("\nSaved:")
    print("  ", csv_path)
    print("  ", json_path)

    print(
        "\n第一轮先看：IK、table collision、"
        "3-contact、contact error。"
    )

    print(
        "暂时不要因为某个候选 lift 不成功而改候选生成器，"
        "因为本脚本还没有做 lift。"
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
