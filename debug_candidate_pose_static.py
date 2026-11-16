# -*- coding: utf-8 -*-
r"""
debug_candidate_pose_static.py

用途：
    先把“候选姿态本身到底对不对”单独验证清楚。

    本程序完全不执行：
        pregrasp
        approach
        finger closing
        mj_step 动力学
        contact
        Taxim

    它只做：
        1) 读取 candidates_top50.npz 中的 ^O T_G
        2) 读取 tomato 当前 ^W T_O
        3) 计算 ^W T_G = ^W T_O @ ^O T_G
        4) UR3 IK 直接到目标 tool0 pose
        5) 三根手指直接设置到 candidate alpha（qpos，不走闭合过程）
        6) mj_forward
        7) 从 MuJoCo 读取三个 GelSight active-face center
        8) 转回 object frame，与候选文件里的 pred_contact_object 对比

判定：

    A. 如果
        actual GelSight face centers
        ≈ pred_contact_object
       每指误差接近 0~1 mm：

       说明：
           T_object_tool 的坐标变换、IK、tool0、夹爪几何都是对的。
       那么 V3 的问题不是“姿态应用错了”，而是：
           candidate 本身不是物理合适的抓取模式
       或
           preopen -> approach -> closing 路径发生提前碰撞。

    B. 如果这里误差仍然几十毫米：

       说明真正存在：
           坐标系 / tool0 convention / XML几何不一致
       必须先修这个，不能继续做接触验证。

运行：

    python debug_candidate_pose_static.py --candidate 4 --viewer

建议先：
    candidate 4
    candidate 9

Viewer 模式：
    程序只 mj_forward，不 mj_step，因此 tomato 不会被手指推走。
"""

from pathlib import Path
import argparse
import math
import sys
import time

import numpy as np
import mujoco
from mujoco import viewer as mj_viewer


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


TOOL_SITE = "tool0"
OBJECT_BODY = "tomato_soup_can"
OBJECT_FREE_JOINT = "tomato_soup_can_free"

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


IK_MAX_ITERS = 500
IK_DAMPING = 0.03
IK_GAIN = 0.7
IK_MAX_STEP = math.radians(7.0)
IK_POS_TOL = 0.001
IK_ROT_TOL = math.radians(1.0)


def mj_id(model, typ, name):
    idx = mujoco.mj_name2id(model, typ, name)
    if idx < 0:
        raise RuntimeError(f"MuJoCo 中找不到：{name}")
    return idx


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
            ]
        )

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    )

    s = 2.0 * math.sin(theta)

    if abs(s) < 1e-8:
        return np.zeros(3)

    return axis / s * theta


def rotation_error_deg(R1, R2):
    R = R1.T @ R2

    c = float(
        np.clip(
            (np.trace(R) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
    )

    return math.degrees(math.acos(c))


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
    for jid, val in zip(
        ids["arm_jids"],
        q,
    ):
        set_joint_q(
            model,
            data,
            jid,
            float(val),
        )

        dadr = int(
            model.jnt_dofadr[jid]
        )

        data.qvel[dadr] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    for jid, aid in zip(
        ids["arm_jids"],
        ids["arm_aids"],
    ):
        set_ctrl(
            model,
            data,
            aid,
            get_joint_q(
                model,
                data,
                jid,
            ),
        )


def set_fingers_direct(
    model,
    data,
    ids,
    alpha_deg,
):
    q = math.radians(
        float(alpha_deg)
    )

    for jid, aid in zip(
        ids["finger_jids"],
        ids["finger_aids"],
    ):
        set_joint_q(
            model,
            data,
            jid,
            q,
        )

        set_ctrl(
            model,
            data,
            aid,
            q,
        )

        data.qvel[
            int(model.jnt_dofadr[jid])
        ] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


def solve_ik_once(
    model,
    data,
    ids,
    target_p,
    target_R,
):
    best_q = arm_q(
        model,
        data,
        ids,
    )

    best_score = float("inf")
    best_ep = float("inf")
    best_er = float("inf")

    for _ in range(IK_MAX_ITERS):
        mujoco.mj_forward(
            model,
            data,
        )

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

        ep = float(
            np.linalg.norm(ep_vec)
        )
        er = float(
            np.linalg.norm(er_vec)
        )

        score = ep + 0.08 * er

        if score < best_score:
            best_score = score
            best_ep = ep
            best_er = er
            best_q = arm_q(
                model,
                data,
                ids,
            )

        if (
            ep <= IK_POS_TOL
            and er <= IK_ROT_TOL
        ):
            return (
                True,
                ep,
                er,
                arm_q(
                    model,
                    data,
                    ids,
                ),
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

        cols = ids[
            "arm_dof_adrs"
        ]

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
            dq = (
                J.T
                @ np.linalg.solve(
                    A,
                    e,
                )
            )
        except np.linalg.LinAlgError:
            dq = (
                np.linalg.pinv(J)
                @ e
            )

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
            arm_q(
                model,
                data,
                ids,
            )
            + dq,
        )

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


def solve_ik_multistart(
    model,
    data,
    ids,
    target_p,
    target_R,
    home_q,
):
    seeds_deg = [
        None,
        [0, -90, 90, -90, -90, 0],
        [0, -70, 110, -130, -90, 0],
        [30, -90, 90, -90, -90, 0],
        [-30, -90, 90, -90, -90, 0],
        [60, -80, 100, -110, -90, 0],
        [-60, -80, 100, -110, -90, 0],
    ]

    best = None

    for seed in seeds_deg:
        if seed is None:
            q0 = home_q.copy()
        else:
            q0 = np.radians(
                np.asarray(
                    seed,
                    dtype=np.float64,
                )
            )

        set_arm_q(
            model,
            data,
            ids,
            q0,
        )

        ok, ep, er, q = (
            solve_ik_once(
                model,
                data,
                ids,
                target_p,
                target_R,
            )
        )

        score = ep + 0.08 * er

        if (
            best is None
            or score
            < best["score"]
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

    return best


def get_active_face_signs(
    envelope,
):
    if (
        "active_face_signs"
        in envelope
    ):
        signs = envelope[
            "active_face_signs"
        ].astype(np.int32)

        if signs.shape == (3,):
            return signs

    return np.array(
        [-1, -1, -1],
        dtype=np.int32,
    )


def actual_gelsight_face_centers_object(
    model,
    data,
    ids,
    T_world_object,
    active_signs,
):
    T_object_world = np.linalg.inv(
        T_world_object
    )

    centers_obj = np.zeros(
        (3, 3),
        dtype=np.float64,
    )

    normals_obj = np.zeros(
        (3, 3),
        dtype=np.float64,
    )

    for i, gid in enumerate(
        ids["gel_geom_ids"]
    ):
        p_w = data.geom_xpos[
            gid
        ].copy()

        R_w = data.geom_xmat[
            gid
        ].reshape(3, 3).copy()

        hx, hy, hz = model.geom_size[
            gid
        ][:3]

        sign = float(
            active_signs[i]
        )

        face_local = np.array(
            [
                0.0,
                0.0,
                sign * float(hz),
            ],
            dtype=np.float64,
        )

        n_local = np.array(
            [
                0.0,
                0.0,
                sign,
            ],
            dtype=np.float64,
        )

        face_w = (
            p_w
            + R_w @ face_local
        )

        n_w = normalize(
            R_w @ n_local
        )

        centers_obj[i] = (
            T_object_world[:3, :3]
            @ face_w
            + T_object_world[:3, 3]
        )

        normals_obj[i] = normalize(
            T_object_world[:3, :3]
            @ n_w
        )

    return (
        centers_obj,
        normals_obj,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
    )

    args = parser.parse_args()

    rank = int(
        args.candidate
    )

    if not XML_PATH.exists():
        raise FileNotFoundError(
            XML_PATH
        )

    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            CANDIDATE_PATH
        )

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(
        model
    )

    ids = {
        "tool_sid": mj_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            TOOL_SITE,
        ),
        "object_bid": mj_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            OBJECT_BODY,
        ),
        "arm_jids": [
            mj_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                x,
            )
            for x in ARM_JOINTS
        ],
        "arm_aids": [
            mj_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                x,
            )
            for x in ARM_ACTUATORS
        ],
        "finger_jids": [
            mj_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                x,
            )
            for x in FINGER_JOINTS
        ],
        "finger_aids": [
            mj_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                x,
            )
            for x in FINGER_ACTUATORS
        ],
        "gel_geom_ids": [
            mj_id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                x,
            )
            for x in GEL_GEOMS
        ],
    }

    ids["arm_dof_adrs"] = [
        int(
            model.jnt_dofadr[jid]
        )
        for jid in ids[
            "arm_jids"
        ]
    ]

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
        mujoco.mj_resetData(
            model,
            data,
        )

    mujoco.mj_forward(
        model,
        data,
    )

    home_q = arm_q(
        model,
        data,
        ids,
    )

    candidate = np.load(
        CANDIDATE_PATH,
        allow_pickle=True,
    )

    envelope = np.load(
        ENVELOPE_PATH,
        allow_pickle=True,
    )

    n = len(
        candidate[
            "T_object_tool"
        ]
    )

    if rank < 1 or rank > n:
        raise ValueError(
            f"rank={rank}, available 1~{n}"
        )

    idx = rank - 1

    T_object_tool_target = (
        candidate[
            "T_object_tool"
        ][idx].astype(
            np.float64
        )
    )

    alpha_deg = float(
        candidate[
            "alpha_deg"
        ][idx]
    )

    pred_contact_object = (
        candidate[
            "pred_contact_object"
        ][idx].astype(
            np.float64
        )
    )

    matched_contact_object = (
        candidate[
            "matched_contact_object"
        ][idx].astype(
            np.float64
        )
    )

    saved_gel_normal_object = (
        candidate[
            "gel_normal_object"
        ][idx].astype(
            np.float64
        )
        if "gel_normal_object"
        in candidate
        else None
    )

    active_signs = (
        get_active_face_signs(
            envelope
        )
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    # 不让 tomato settle / 不 mj_step。
    # 直接使用 keyframe 中的 object pose。
    # 这样排除物体动力学移动。
    # --------------------------------------------------------------

    T_world_object = get_body_pose(
        data,
        ids["object_bid"],
    )

    T_world_tool_target = (
        T_world_object
        @ T_object_tool_target
    )

    print("=" * 88)
    print("STATIC candidate pose replay")
    print("=" * 88)

    print(
        f"candidate #{rank:03d}"
    )

    print(
        f"alpha = {alpha_deg:+.3f} deg"
    )

    print(
        "active face signs =",
        active_signs.tolist(),
    )

    print("\nT_object_tool target =")
    print(
        np.array2string(
            T_object_tool_target,
            precision=6,
            suppress_small=True,
        )
    )

    print(
        "\nobject world pos =",
        np.round(
            T_world_object[:3, 3],
            6,
        ),
    )

    # 手指先直接设到 candidate alpha
    set_fingers_direct(
        model,
        data,
        ids,
        alpha_deg,
    )

    # IK 直接去 contact candidate，不走 pregrasp
    best = solve_ik_multistart(
        model,
        data,
        ids,
        T_world_tool_target[:3, 3],
        T_world_tool_target[:3, :3],
        home_q,
    )

    print("\nIK:")
    print(
        "  ok      =",
        best["ok"],
    )

    print(
        "  pos err =",
        f"{best['ep']*1000:.3f} mm",
    )

    print(
        "  rot err =",
        f"{math.degrees(best['er']):.3f} deg",
    )

    mujoco.mj_forward(
        model,
        data,
    )

    T_world_tool_actual = get_site_pose(
        data,
        ids["tool_sid"],
    )

    T_object_tool_actual = (
        np.linalg.inv(
            T_world_object
        )
        @ T_world_tool_actual
    )

    tool_translation_error = (
        np.linalg.norm(
            T_object_tool_actual[:3, 3]
            - T_object_tool_target[:3, 3]
        )
    )

    tool_rotation_error = (
        rotation_error_deg(
            T_object_tool_actual[:3, :3],
            T_object_tool_target[:3, :3],
        )
    )

    print("\nactual T_object_tool =")
    print(
        np.array2string(
            T_object_tool_actual,
            precision=6,
            suppress_small=True,
        )
    )

    print(
        "\ntool pose replay error:"
    )

    print(
        "  translation =",
        f"{tool_translation_error*1000:.3f} mm",
    )

    print(
        "  rotation    =",
        f"{tool_rotation_error:.3f} deg",
    )

    actual_centers_obj, actual_normals_obj = (
        actual_gelsight_face_centers_object(
            model,
            data,
            ids,
            T_world_object,
            active_signs,
        )
    )

    e_pred = np.linalg.norm(
        actual_centers_obj
        - pred_contact_object,
        axis=1,
    )

    e_matched = np.linalg.norm(
        actual_centers_obj
        - matched_contact_object,
        axis=1,
    )

    print(
        "\nGelSight active-face centers in OBJECT frame"
    )

    for i in range(3):
        print(
            f"\nF{i+1}:"
        )

        print(
            "  actual   =",
            np.round(
                actual_centers_obj[i]
                * 1000.0,
                3,
            ),
            "mm",
        )

        print(
            "  predicted=",
            np.round(
                pred_contact_object[i]
                * 1000.0,
                3,
            ),
            "mm",
        )

        print(
            "  CAD match=",
            np.round(
                matched_contact_object[i]
                * 1000.0,
                3,
            ),
            "mm",
        )

        print(
            "  error actual-pred =",
            f"{e_pred[i]*1000:.3f} mm",
        )

        print(
            "  error actual-CAD  =",
            f"{e_matched[i]*1000:.3f} mm",
        )

    print(
        "\nmean actual-pred error =",
        f"{np.mean(e_pred)*1000:.3f} mm",
    )

    print(
        "max actual-pred error  =",
        f"{np.max(e_pred)*1000:.3f} mm",
    )

    print(
        "mean actual-CAD error  =",
        f"{np.mean(e_matched)*1000:.3f} mm",
    )

    if (
        saved_gel_normal_object
        is not None
    ):
        normal_errors = []

        print(
            "\nGelSight normal replay:"
        )

        for i in range(3):
            dot = float(
                np.clip(
                    np.dot(
                        actual_normals_obj[i],
                        saved_gel_normal_object[i],
                    ),
                    -1.0,
                    1.0,
                )
            )

            ang = math.degrees(
                math.acos(dot)
            )

            normal_errors.append(
                ang
            )

            print(
                f"  F{i+1}: "
                f"{ang:.3f} deg"
            )

        print(
            "  max normal replay error =",
            f"{max(normal_errors):.3f} deg",
        )

    print("\n判定：")

    if (
        np.max(e_pred) <= 0.002
        and tool_translation_error <= 0.002
        and tool_rotation_error <= 2.0
    ):
        print(
            "  STATIC REPLAY = PASS"
        )
        print(
            "  => T_object_tool / IK / tool0 / envelope 几何基本一致。"
        )
        print(
            "  => V3 动态抓取出错，应检查候选抓取模式或 preopen/闭合路径提前碰撞。"
        )
    else:
        print(
            "  STATIC REPLAY = FAIL"
        )
        print(
            "  => 先不要继续动态抓取。"
        )
        print(
            "  => 存在坐标系、tool0 convention、XML几何或候选变换不一致。"
        )

    if args.viewer:
        print(
            "\nViewer 已打开。"
        )
        print(
            "这里不会 mj_step，物体不会被推走。"
        )
        print(
            "直接观察 candidate 的静态目标姿态。关闭窗口结束。"
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
            "\n请把完整输出发给我。"
        )
        sys.exit(1)
