"""
collect_underactuated_tactile_dataset.py

MuJoCo 3.6.x
UR3 + underactuated three-finger gripper + 3 GelSight pads + YCB object

采集流程
--------
1) home 复位
2) 自动判定 gripper_ctrl 的张开/闭合方向（也可手工指定）
3) 三指张开
4) 随机化物体 x/y/yaw，并让物体自由沉降
5) UR3 保持固定末端姿态，移动到物体上方安全位
6) 沿 tool +X 接近轴做近似笛卡尔直线下降，直到三指名义抓取中心与物体中心重合
7) 欠驱动 tendon 缓慢闭合
8) 第一个 GelSight 接触出现后：
   - 保存接触前 4 帧
   - 再保存动态接触 20 帧
   - 再保持并保存 8 帧
   => 每个有效 episode 固定 32 个同步采样帧
9) 保存 signals.csv + metadata.json
10) GelSight 图像接口单独留在 capture_gelsight_images()。
    当前 XML 没有三路 GelSight 成像相机/触觉渲染器，所以不能用普通 RGB 相机冒充触觉图。

重要：
- 先用 NUM_EPISODES=5~20 做 pilot。
- 确认运动、闭合方向和接触判定正确后再批量采。
"""

import contextlib
import csv
import json
import math
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

try:
    import mujoco.viewer
except Exception:
    mujoco.viewer = None


# ============================================================
# 0. 路径与采集规模
# ============================================================

XML_PATH = Path(
    r"E:\project\p1\deepgelsight\xml"
    r"\ur3_with_threefinger_gelsight_table_tomato_underactuated.xml"
)

OUTPUT_ROOT = Path(
    r"E:\project\p1\deepgelsight\dataset_underactuated"
)

OBJECT_BODY = "tomato_soup_can"
OBJECT_FREEJOINT = "tomato_soup_can_free"

NUM_EPISODES = 10
BASE_SEED = 20260924

SHOW_VIEWER = True


# ============================================================
# 1. 模型名称
# ============================================================

TOOL_SITE = "tool0"

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

FINGER_JOINTS = [
    "finger1_hinge",
    "finger2_hinge",
    "finger3_hinge",
]

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

GRIPPER_ACTUATOR = "gripper_ctrl"


# ============================================================
# 2. 已验证的夹爪名义抓取中心
#    grasp_center = tool0 + R_tool @ [0.14965, 0, 0]
# ============================================================

GRASP_OFFSET_TOOL = np.array(
    [0.14965, 0.0, 0.0],
    dtype=float,
)


# ============================================================
# 3. 随机物体位姿
# ============================================================

# 只做小范围 x/y 随机 + yaw 随机。
# 对当前 tomato_soup_can，yaw 几何意义较小；
# 以后换成 box / bottle / mug 时此设置可直接沿用。
XY_RANDOM_RANGE_M = 0.004       # ±4 mm
YAW_MIN_DEG = 0.0
YAW_MAX_DEG = 360.0

# 复位后物体的 z 先沿用 XML / home keyframe，
# 然后自由沉降到桌面。
OBJECT_SETTLE_TIME = 0.50


# ============================================================
# 4. UR3 运动参数
# ============================================================

# tool +X 是夹爪接近轴。
SAFE_DISTANCE = 0.120           # 安全位：离物体中心 120 mm
PREGRASP_DISTANCE = 0.055       # 预抓取位：离物体中心 55 mm

SAFE_MOVE_TIME = 2.8

# 从安全位 -> 抓取中心采用离散笛卡尔直线路径。
# 越大越直，但运行越慢。
CARTESIAN_WAYPOINTS = 18
WAYPOINT_MOVE_TIME = 0.10

# 到达最终抓取位后先稳定一下再闭合。
AT_GRASP_POSE_HOLD = 0.15

# IK
IK_MAX_ITER = 600
IK_POS_TOL = 5e-5              # 0.05 mm
IK_ROT_TOL = 3e-4
IK_DAMPING = 1e-3
IK_STEP_SCALE = 0.55
IK_MAX_DQ = 0.07


# ============================================================
# 5. 欠驱动三指参数
# ============================================================

# 如果已经人工确认闭合方向，可设 +1 或 -1。
# None = 程序启动时自动比较 ±CALIBRATION_CTRL_MAG。
MANUAL_CLOSE_SIGN = None

CALIBRATION_CTRL_MAG = 0.60
CALIBRATION_TIME = 0.80

# 张开目标与最大闭合目标，单位是 tendon target:
# q_f1 + q_f2 + q_f3 [rad]
OPEN_CTRL_MAG = 0.70
CLOSE_CTRL_MAG = 1.20

OPEN_HOLD_TIME = 0.80

# tendon target 的变化速度。
# 0.35 rad/s 对三个手指平均约等于 0.117 rad/s / finger。
GRIPPER_CLOSE_RATE = 0.35

# 若一直没有任何 GelSight 接触，最多闭合这么久。
MAX_WAIT_FIRST_CONTACT = 6.0


# ============================================================
# 6. 同步采样协议
# ============================================================

CAPTURE_HZ = 50.0

PRECONTACT_FRAMES = 4
DYNAMIC_FRAMES = 20
HOLD_FRAMES = 8

EXPECTED_TOTAL_FRAMES = (
    PRECONTACT_FRAMES + DYNAMIC_FRAMES + HOLD_FRAMES
)  # 32

# 三点接触必须至少连续存在这些采样帧，才算有效。
ALL_CONTACT_STABLE_TIME = 0.04

# 是否要求有效 episode 必须出现稳定三点接触。
REQUIRE_ALL_THREE_CONTACT = True


# ============================================================
# 7. GelSight 图像保存
# ============================================================

# 当前 XML 中没有三路 GelSight 成像相机/触觉渲染器，
# 因此默认 False。
#
# 当你把 DeepGelSight 渲染接口接入 capture_gelsight_images()
# 后，再改成 True。
SAVE_GELSIGHT_IMAGES = False


# ============================================================
# 8. 基础函数
# ============================================================

def name2id(model, objtype, name):
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx == -1:
        raise ValueError(f"模型中找不到: {name}")
    return idx


def reset_home(model, data):
    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)


def get_joint_addresses(model, names):
    qpos_adr = []
    dof_adr = []

    for name in names:
        jid = name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])

    return (
        np.asarray(qpos_adr, dtype=int),
        np.asarray(dof_adr, dtype=int),
    )


def get_actuator_ids(model, names):
    return np.asarray(
        [
            name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                n,
            )
            for n in names
        ],
        dtype=int,
    )


def get_site_pose(model, data, site_name):
    sid = name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    p = data.site_xpos[sid].copy()
    R = data.site_xmat[sid].reshape(3, 3).copy()

    return p, R


def get_grasp_center(model, data):
    tool_pos, tool_R = get_site_pose(
        model,
        data,
        TOOL_SITE,
    )

    return (
        tool_pos
        + tool_R @ GRASP_OFFSET_TOOL
    )


def get_actual_gel_center(model, data):
    pts = []

    for name in GEL_SITES:
        sid = name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )
        pts.append(
            data.site_xpos[sid].copy()
        )

    return np.mean(
        np.asarray(pts),
        axis=0,
    )


def get_body_pose(model, data, body_name):
    bid = name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    pos = data.xpos[bid].copy()
    R = data.xmat[bid].reshape(3, 3).copy()

    return pos, R


def hold_arm(
    data,
    arm_act_ids,
    q_cmd,
):
    for i, aid in enumerate(arm_act_ids):
        data.ctrl[aid] = q_cmd[i]


def set_gripper_ctrl(
    data,
    gripper_act_id,
    value,
):
    data.ctrl[gripper_act_id] = value


def get_arm_q(
    data,
    arm_qpos_adr,
):
    return np.asarray(
        [
            data.qpos[a]
            for a in arm_qpos_adr
        ],
        dtype=float,
    )


def get_finger_q(
    data,
    finger_qpos_adr,
):
    return np.asarray(
        [
            data.qpos[a]
            for a in finger_qpos_adr
        ],
        dtype=float,
    )


def sync_viewer(viewer):
    if viewer is not None:
        if viewer.is_running():
            viewer.sync()


# ============================================================
# 9. GelSight 图像接口
# ============================================================

def capture_gelsight_images(model, data):
    """
    在这里接你的 DeepGelSight / Taxim / 自定义 tactile renderer。

    必须返回：
    {
        "finger1": np.ndarray(H, W, 3),  # RGB uint8
        "finger2": np.ndarray(H, W, 3),
        "finger3": np.ndarray(H, W, 3),
    }

    三张图必须来自同一个 MuJoCo simulation state，
    这样才是真正同步。

    当前 XML 只有 finger*_gelsight_site 和 finger*_gel，
    并没有真正的 GelSight 成像相机，因此这里不能用普通
    MuJoCo RGB render 冒充触觉图。

    示例（仅说明接口，不是你的实际 API）：
        img1 = tactile_renderer.render("finger1_gel", data)
        img2 = tactile_renderer.render("finger2_gel", data)
        img3 = tactile_renderer.render("finger3_gel", data)
        return {
            "finger1": img1,
            "finger2": img2,
            "finger3": img3,
        }
    """
    if SAVE_GELSIGHT_IMAGES:
        raise NotImplementedError(
            "SAVE_GELSIGHT_IMAGES=True，但 capture_gelsight_images() "
            "还没有接入你的 DeepGelSight 渲染接口。"
        )

    return None


def save_rgb_image(path, image):
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    try:
        from PIL import Image

        Image.fromarray(image).save(path)
        return
    except ImportError:
        pass

    try:
        import cv2

        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        cv2.imwrite(
            str(path),
            image,
        )
        return
    except ImportError:
        pass

    raise RuntimeError(
        "保存 GelSight 图像需要 Pillow 或 OpenCV。"
    )


# ============================================================
# 10. 接触检测
# ============================================================

def build_contact_ids(model):
    gel_geom_ids = [
        name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            n,
        )
        for n in GEL_GEOMS
    ]

    object_body_id = name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        OBJECT_BODY,
    )

    object_geom_ids = {
        gid
        for gid in range(model.ngeom)
        if model.geom_bodyid[gid] == object_body_id
        and model.geom_contype[gid] != 0
    }

    return gel_geom_ids, object_geom_ids


def get_finger_contact_state(
    model,
    data,
    gel_geom_ids,
    object_geom_ids,
):
    """
    返回每个手指：
      contact: bool
      count: 与目标物体的接触点数
      normal_force: 所有关联 contact 的法向力绝对值之和
    """

    states = [
        {
            "contact": False,
            "count": 0,
            "normal_force": 0.0,
        }
        for _ in range(3)
    ]

    gel_to_finger = {
        gid: i
        for i, gid in enumerate(gel_geom_ids)
    }

    for ci in range(data.ncon):
        c = data.contact[ci]

        g1 = int(c.geom1)
        g2 = int(c.geom2)

        finger_idx = None

        if (
            g1 in gel_to_finger
            and g2 in object_geom_ids
        ):
            finger_idx = gel_to_finger[g1]

        elif (
            g2 in gel_to_finger
            and g1 in object_geom_ids
        ):
            finger_idx = gel_to_finger[g2]

        if finger_idx is None:
            continue

        states[finger_idx]["contact"] = True
        states[finger_idx]["count"] += 1

        force6 = np.zeros(
            6,
            dtype=float,
        )

        mujoco.mj_contactForce(
            model,
            data,
            ci,
            force6,
        )

        states[finger_idx]["normal_force"] += abs(
            float(force6[0])
        )

    return states


def object_has_unwanted_robot_contact(
    model,
    data,
    gel_geom_ids,
    object_geom_ids,
):
    """
    机械臂移动阶段：目标物体若碰到任何非 GelSight 的机器人 geom，
    视作意外碰撞。

    物体碰桌面是允许的。
    """

    table_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "table",
    )

    gel_set = set(gel_geom_ids)

    for ci in range(data.ncon):
        c = data.contact[ci]

        g1 = int(c.geom1)
        g2 = int(c.geom2)

        if g1 in object_geom_ids:
            other = g2
        elif g2 in object_geom_ids:
            other = g1
        else:
            continue

        if other in gel_set:
            continue

        other_body = int(
            model.geom_bodyid[other]
        )

        if other_body == table_body_id:
            continue

        # world floor body id == 0
        if other_body == 0:
            continue

        return True

    return False


# ============================================================
# 11. 旋转误差 + IK
# ============================================================

def rotation_error_world(
    R_current,
    R_target,
):
    R_err = R_target @ R_current.T

    return 0.5 * np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )


def solve_ik(
    model,
    reference_data,
    target_tool_pos,
    target_tool_R,
    q_start,
    arm_qpos_adr,
    arm_dof_adr,
):
    work = mujoco.MjData(model)

    work.qpos[:] = reference_data.qpos
    work.qvel[:] = 0.0
    work.ctrl[:] = reference_data.ctrl

    tool_id = name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        TOOL_SITE,
    )

    q = q_start.copy()

    for iteration in range(
        IK_MAX_ITER
    ):
        for i, adr in enumerate(
            arm_qpos_adr
        ):
            work.qpos[adr] = q[i]

        work.qvel[:] = 0.0

        mujoco.mj_forward(
            model,
            work,
        )

        p_cur = work.site_xpos[
            tool_id
        ].copy()

        R_cur = work.site_xmat[
            tool_id
        ].reshape(3, 3).copy()

        e_pos = (
            target_tool_pos - p_cur
        )

        e_rot = rotation_error_world(
            R_cur,
            target_tool_R,
        )

        if (
            np.linalg.norm(e_pos)
            < IK_POS_TOL
            and np.linalg.norm(e_rot)
            < IK_ROT_TOL
        ):
            return q, True

        jacp = np.zeros(
            (3, model.nv)
        )
        jacr = np.zeros(
            (3, model.nv)
        )

        mujoco.mj_jacSite(
            model,
            work,
            jacp,
            jacr,
            tool_id,
        )

        J = np.vstack(
            [
                jacp[:, arm_dof_adr],
                jacr[:, arm_dof_adr],
            ]
        )

        e = np.concatenate(
            [e_pos, e_rot]
        )

        A = (
            J @ J.T
            + (IK_DAMPING ** 2)
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
            return q, False

        dq *= IK_STEP_SCALE

        norm_dq = np.linalg.norm(dq)

        if norm_dq > IK_MAX_DQ:
            dq *= (
                IK_MAX_DQ
                / norm_dq
            )

        q += dq

        for i, joint_name in enumerate(
            ARM_JOINTS
        ):
            jid = name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if model.jnt_limited[jid]:
                q[i] = np.clip(
                    q[i],
                    model.jnt_range[jid, 0],
                    model.jnt_range[jid, 1],
                )

    return q, False


def grasp_center_to_tool_target(
    grasp_center_target,
    target_tool_R,
):
    return (
        grasp_center_target
        - target_tool_R
        @ GRASP_OFFSET_TOOL
    )


# ============================================================
# 12. 机械臂运动
# ============================================================

def smoothstep(x):
    return (
        3.0 * x * x
        - 2.0 * x * x * x
    )


def move_arm_joint_space(
    model,
    data,
    viewer,
    arm_act_ids,
    gripper_act_id,
    gripper_ctrl,
    q_start,
    q_target,
    duration,
    gel_geom_ids=None,
    object_geom_ids=None,
):
    steps = max(
        1,
        int(
            duration
            / model.opt.timestep
        ),
    )

    for k in range(steps):
        alpha = (
            (k + 1) / steps
        )

        s = smoothstep(alpha)

        q_cmd = (
            q_start
            + s
            * (q_target - q_start)
        )

        hold_arm(
            data,
            arm_act_ids,
            q_cmd,
        )

        set_gripper_ctrl(
            data,
            gripper_act_id,
            gripper_ctrl,
        )

        mujoco.mj_step(
            model,
            data,
        )

        if (
            gel_geom_ids is not None
            and object_geom_ids is not None
            and object_has_unwanted_robot_contact(
                model,
                data,
                gel_geom_ids,
                object_geom_ids,
            )
        ):
            return False

        sync_viewer(viewer)

    return True


def move_grasp_center_cartesian(
    model,
    data,
    viewer,
    arm_act_ids,
    gripper_act_id,
    gripper_ctrl,
    start_center,
    end_center,
    target_tool_R,
    arm_qpos_adr,
    arm_dof_adr,
    gel_geom_ids,
    object_geom_ids,
):
    q_seed = get_arm_q(
        data,
        arm_qpos_adr,
    )

    prev_center = (
        np.asarray(
            start_center,
            dtype=float,
        )
    )

    for i in range(
        1,
        CARTESIAN_WAYPOINTS + 1,
    ):
        alpha = (
            i
            / CARTESIAN_WAYPOINTS
        )

        center_target = (
            start_center
            + alpha
            * (end_center - start_center)
        )

        tool_target = (
            grasp_center_to_tool_target(
                center_target,
                target_tool_R,
            )
        )

        q_target, ok = solve_ik(
            model=model,
            reference_data=data,
            target_tool_pos=tool_target,
            target_tool_R=target_tool_R,
            q_start=q_seed,
            arm_qpos_adr=arm_qpos_adr,
            arm_dof_adr=arm_dof_adr,
        )

        if not ok:
            print(
                f"[ABORT] Cartesian waypoint "
                f"{i}/{CARTESIAN_WAYPOINTS} IK failed."
            )
            return False

        q_now = get_arm_q(
            data,
            arm_qpos_adr,
        )

        moved = move_arm_joint_space(
            model=model,
            data=data,
            viewer=viewer,
            arm_act_ids=arm_act_ids,
            gripper_act_id=gripper_act_id,
            gripper_ctrl=gripper_ctrl,
            q_start=q_now,
            q_target=q_target,
            duration=WAYPOINT_MOVE_TIME,
            gel_geom_ids=gel_geom_ids,
            object_geom_ids=object_geom_ids,
        )

        if not moved:
            print(
                "[ABORT] 机械臂接近阶段发生非 GelSight "
                "物体碰撞。"
            )
            return False

        q_seed = q_target
        prev_center = center_target

    return True


# ============================================================
# 13. 自动判定 gripper_ctrl 正负方向
# ============================================================

def mean_gel_spread(model, data):
    pts = []

    for name in GEL_SITES:
        sid = name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )
        pts.append(
            data.site_xpos[sid].copy()
        )

    pts = np.asarray(pts)

    d01 = np.linalg.norm(
        pts[0] - pts[1]
    )
    d02 = np.linalg.norm(
        pts[0] - pts[2]
    )
    d12 = np.linalg.norm(
        pts[1] - pts[2]
    )

    return (
        d01 + d02 + d12
    ) / 3.0


def test_gripper_ctrl_spread(
    xml_path,
    ctrl_value,
):
    test_model = (
        mujoco.MjModel.from_xml_path(
            str(xml_path)
        )
    )

    test_data = mujoco.MjData(
        test_model
    )

    # 自动方向判定时不要让物体接触影响手指。
    test_model.opt.disableflags |= int(
        mujoco.mjtDisableBit.mjDSBL_CONTACT
    )

    reset_home(
        test_model,
        test_data,
    )

    arm_ids = get_actuator_ids(
        test_model,
        ARM_ACTUATORS,
    )

    gripper_id = name2id(
        test_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        GRIPPER_ACTUATOR,
    )

    arm_qpos_adr, _ = (
        get_joint_addresses(
            test_model,
            ARM_JOINTS,
        )
    )

    q_home = get_arm_q(
        test_data,
        arm_qpos_adr,
    )

    steps = int(
        CALIBRATION_TIME
        / test_model.opt.timestep
    )

    for _ in range(steps):
        hold_arm(
            test_data,
            arm_ids,
            q_home,
        )

        set_gripper_ctrl(
            test_data,
            gripper_id,
            ctrl_value,
        )

        mujoco.mj_step(
            test_model,
            test_data,
        )

    mujoco.mj_forward(
        test_model,
        test_data,
    )

    return mean_gel_spread(
        test_model,
        test_data,
    )


def determine_close_sign(
    xml_path,
):
    if MANUAL_CLOSE_SIGN in (
        -1,
        1,
    ):
        return int(
            MANUAL_CLOSE_SIGN
        )

    spread_positive = (
        test_gripper_ctrl_spread(
            xml_path,
            +CALIBRATION_CTRL_MAG,
        )
    )

    spread_negative = (
        test_gripper_ctrl_spread(
            xml_path,
            -CALIBRATION_CTRL_MAG,
        )
    )

    print(
        "[GRIPPER CALIBRATION]"
    )
    print(
        f"  ctrl=+{CALIBRATION_CTRL_MAG:.2f}: "
        f"mean GelSight spacing = "
        f"{spread_positive*1000:.3f} mm"
    )
    print(
        f"  ctrl=-{CALIBRATION_CTRL_MAG:.2f}: "
        f"mean GelSight spacing = "
        f"{spread_negative*1000:.3f} mm"
    )

    # 更小的三点间距定义为闭合方向。
    if (
        spread_positive
        < spread_negative
    ):
        close_sign = +1
    else:
        close_sign = -1

    print(
        f"  => 自动判定 CLOSE_SIGN = "
        f"{close_sign:+d}"
    )

    return close_sign


# ============================================================
# 14. 随机化物体
# ============================================================

def randomize_object_pose(
    model,
    data,
    rng,
):
    jid = name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        OBJECT_FREEJOINT,
    )

    qadr = int(
        model.jnt_qposadr[jid]
    )

    # 当前 home 中的物体位置作为名义中心。
    base_xyz = data.qpos[
        qadr:qadr + 3
    ].copy()

    dx = rng.uniform(
        -XY_RANDOM_RANGE_M,
        +XY_RANDOM_RANGE_M,
    )

    dy = rng.uniform(
        -XY_RANDOM_RANGE_M,
        +XY_RANDOM_RANGE_M,
    )

    yaw_deg = rng.uniform(
        YAW_MIN_DEG,
        YAW_MAX_DEG,
    )

    yaw = math.radians(
        yaw_deg
    )

    quat = np.array(
        [
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ],
        dtype=float,
    )

    data.qpos[
        qadr:qadr + 3
    ] = (
        base_xyz
        + np.array(
            [dx, dy, 0.0]
        )
    )

    data.qpos[
        qadr + 3:qadr + 7
    ] = quat

    dof_adr = int(
        model.jnt_dofadr[jid]
    )

    # freejoint 有 6 个速度自由度。
    data.qvel[
        dof_adr:dof_adr + 6
    ] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    return {
        "dx_m": float(dx),
        "dy_m": float(dy),
        "yaw_deg": float(yaw_deg),
    }


# ============================================================
# 15. 采样记录
# ============================================================

def make_sample(
    model,
    data,
    finger_qpos_adr,
    gripper_act_id,
    contact_states,
    phase,
):
    tool_pos, tool_R = get_site_pose(
        model,
        data,
        TOOL_SITE,
    )

    grasp_center = get_grasp_center(
        model,
        data,
    )

    actual_gel_center = (
        get_actual_gel_center(
            model,
            data,
        )
    )

    object_pos, object_R = (
        get_body_pose(
            model,
            data,
            OBJECT_BODY,
        )
    )

    finger_q = get_finger_q(
        data,
        finger_qpos_adr,
    )

    sample = {
        "time": float(data.time),
        "phase": phase,
        "gripper_ctrl": float(
            data.ctrl[gripper_act_id]
        ),
        "finger_q": finger_q.tolist(),
        "contacts": [
            bool(s["contact"])
            for s in contact_states
        ],
        "contact_counts": [
            int(s["count"])
            for s in contact_states
        ],
        "normal_forces": [
            float(s["normal_force"])
            for s in contact_states
        ],
        "tool_pos": tool_pos.tolist(),
        "tool_R": tool_R.reshape(-1).tolist(),
        "grasp_center": grasp_center.tolist(),
        "gel_center": actual_gel_center.tolist(),
        "object_pos": object_pos.tolist(),
        "object_R": object_R.reshape(-1).tolist(),
        "images": capture_gelsight_images(
            model,
            data,
        ),
    }

    return sample


def save_episode(
    episode_dir,
    samples,
    metadata,
):
    episode_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_GELSIGHT_IMAGES:
        for finger in (
            "finger1",
            "finger2",
            "finger3",
        ):
            (
                episode_dir / finger
            ).mkdir(
                exist_ok=True
            )

    csv_path = (
        episode_dir / "signals.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "time",
                "phase",
                "gripper_ctrl",

                "q_f1",
                "q_f2",
                "q_f3",

                "contact_f1",
                "contact_f2",
                "contact_f3",

                "contact_count_f1",
                "contact_count_f2",
                "contact_count_f3",

                "normal_force_f1",
                "normal_force_f2",
                "normal_force_f3",

                "grasp_x",
                "grasp_y",
                "grasp_z",

                "obj_x",
                "obj_y",
                "obj_z",
            ]
        )

        for frame_idx, s in enumerate(
            samples
        ):
            writer.writerow(
                [
                    frame_idx,
                    f"{s['time']:.6f}",
                    s["phase"],
                    f"{s['gripper_ctrl']:.9f}",

                    *[
                        f"{x:.9f}"
                        for x in s["finger_q"]
                    ],

                    *[
                        int(x)
                        for x in s["contacts"]
                    ],

                    *s["contact_counts"],

                    *[
                        f"{x:.9f}"
                        for x in s["normal_forces"]
                    ],

                    *[
                        f"{x:.9f}"
                        for x in s["grasp_center"]
                    ],

                    *[
                        f"{x:.9f}"
                        for x in s["object_pos"]
                    ],
                ]
            )

            if SAVE_GELSIGHT_IMAGES:
                images = s["images"]

                if images is None:
                    raise RuntimeError(
                        "SAVE_GELSIGHT_IMAGES=True，"
                        "但 capture_gelsight_images() 返回 None。"
                    )

                for finger in (
                    "finger1",
                    "finger2",
                    "finger3",
                ):
                    path = (
                        episode_dir
                        / finger
                        / f"frame_{frame_idx:04d}.png"
                    )

                    save_rgb_image(
                        path,
                        images[finger],
                    )

    # 图片数组不写进 JSON。
    metadata = dict(metadata)
    metadata["num_frames"] = len(samples)

    with (
        episode_dir / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 16. 三指缓慢闭合并采集固定 32 帧
# ============================================================

def collect_contact_sequence(
    model,
    data,
    viewer,
    arm_act_ids,
    q_arm_hold,
    gripper_act_id,
    finger_qpos_adr,
    gel_geom_ids,
    object_geom_ids,
    open_ctrl,
    close_ctrl_limit,
    close_sign,
):
    dt = model.opt.timestep

    sample_every_steps = max(
        1,
        int(
            round(
                1.0
                / (
                    CAPTURE_HZ
                    * dt
                )
            )
        ),
    )

    # 第一个接触出现前，只保留最后 PRECONTACT_FRAMES。
    prebuffer = deque(
        maxlen=PRECONTACT_FRAMES
    )

    samples = []

    first_contact_seen = False
    dynamic_capture_count = 0
    hold_capture_count = 0

    stable_all_contact_steps = 0
    ever_stable_all_contact = False
    all_contact_stable_steps_required = max(1, int(round(ALL_CONTACT_STABLE_TIME / dt)))

    ctrl = float(open_ctrl)

    max_steps = int(
        MAX_WAIT_FIRST_CONTACT
        / dt
    )

    # 闭合阶段。
    step_counter = 0

    while True:
        hold_arm(
            data,
            arm_act_ids,
            q_arm_hold,
        )

        if (
            first_contact_seen
            and dynamic_capture_count
            >= DYNAMIC_FRAMES
        ):
            # 动态段结束，固定 tendon target 进入保持。
            pass
        else:
            next_ctrl = (
                ctrl
                + close_sign
                * GRIPPER_CLOSE_RATE
                * dt
            )

            if close_sign > 0:
                ctrl = min(
                    next_ctrl,
                    close_ctrl_limit,
                )
            else:
                ctrl = max(
                    next_ctrl,
                    close_ctrl_limit,
                )

        set_gripper_ctrl(
            data,
            gripper_act_id,
            ctrl,
        )

        mujoco.mj_step(
            model,
            data,
        )

        sync_viewer(viewer)

        contact_states = (
            get_finger_contact_state(
                model,
                data,
                gel_geom_ids,
                object_geom_ids,
            )
        )

        contacts = [
            s["contact"]
            for s in contact_states
        ]

        any_contact = any(
            contacts
        )

        all_contact = all(
            contacts
        )

        if all_contact:
            stable_all_contact_steps += 1
        else:
            stable_all_contact_steps = 0

        if stable_all_contact_steps >= all_contact_stable_steps_required:
            ever_stable_all_contact = True

        # 50 Hz 统一采样。
        if (
            step_counter
            % sample_every_steps
            == 0
        ):
            if not first_contact_seen:
                if any_contact:
                    first_contact_seen = True

                    # 只把真正接触前的最后 PRECONTACT_FRAMES 帧写入正式序列。
                    samples.extend(list(prebuffer))

                    # 当前采样帧是第一个接触帧，作为 dynamic_contact 的第 1 帧。
                    sample = make_sample(
                        model,
                        data,
                        finger_qpos_adr,
                        gripper_act_id,
                        contact_states,
                        phase="dynamic_contact",
                    )
                    samples.append(sample)
                    dynamic_capture_count = 1

                    print(
                        f"    first contact at "
                        f"t={data.time:.3f}s, "
                        f"ctrl={ctrl:.4f}"
                    )
                else:
                    sample = make_sample(
                        model,
                        data,
                        finger_qpos_adr,
                        gripper_act_id,
                        contact_states,
                        phase="precontact",
                    )
                    prebuffer.append(sample)

            else:
                if (
                    dynamic_capture_count
                    < DYNAMIC_FRAMES
                ):
                    phase = "dynamic_contact"

                    sample = make_sample(
                        model,
                        data,
                        finger_qpos_adr,
                        gripper_act_id,
                        contact_states,
                        phase=phase,
                    )

                    samples.append(
                        sample
                    )

                    dynamic_capture_count += 1

                else:
                    phase = "hold"

                    sample = make_sample(
                        model,
                        data,
                        finger_qpos_adr,
                        gripper_act_id,
                        contact_states,
                        phase=phase,
                    )

                    samples.append(
                        sample
                    )

                    hold_capture_count += 1

                    if (
                        hold_capture_count
                        >= HOLD_FRAMES
                    ):
                        break

        # 第一个接触之前的失败条件。
        if not first_contact_seen:
            if step_counter >= max_steps:
                return {
                    "valid": False,
                    "reason": "no_first_contact_timeout",
                    "samples": [],
                    "final_ctrl": ctrl,
                    "all_contact": False,
                }

            reached_limit = (
                (
                    close_sign > 0
                    and ctrl
                    >= close_ctrl_limit
                )
                or
                (
                    close_sign < 0
                    and ctrl
                    <= close_ctrl_limit
                )
            )

            if reached_limit:
                return {
                    "valid": False,
                    "reason": "close_limit_without_contact",
                    "samples": [],
                    "final_ctrl": ctrl,
                    "all_contact": False,
                }

        step_counter += 1

    # 由于第一接触采样帧可能也在 prebuffer 里，
    # 数量正常应接近 32。
    # 强制保留最后 EXPECTED_TOTAL_FRAMES。
    if len(samples) > EXPECTED_TOTAL_FRAMES:
        samples = samples[
            -EXPECTED_TOTAL_FRAMES:
        ]

    valid = (
        ever_stable_all_contact
        if REQUIRE_ALL_THREE_CONTACT
        else True
    )

    reason = (
        "ok"
        if valid
        else "no_stable_three_point_contact"
    )

    return {
        "valid": valid,
        "reason": reason,
        "samples": samples,
        "final_ctrl": ctrl,
        "all_contact": ever_stable_all_contact,
    }


# ============================================================
# 17. 单 episode
# ============================================================

def run_episode(
    model,
    data,
    viewer,
    episode_idx,
    close_sign,
    arm_act_ids,
    gripper_act_id,
    arm_qpos_adr,
    arm_dof_adr,
    finger_qpos_adr,
    gel_geom_ids,
    object_geom_ids,
):
    episode_seed = (
        BASE_SEED
        + episode_idx
    )

    rng = np.random.default_rng(
        episode_seed
    )

    reset_home(
        model,
        data,
    )

    # 确保 contact 开启。
    model.opt.disableflags &= ~int(
        mujoco.mjtDisableBit.mjDSBL_CONTACT
    )

    q_home = get_arm_q(
        data,
        arm_qpos_adr,
    )

    open_ctrl = (
        -close_sign
        * OPEN_CTRL_MAG
    )

    close_ctrl_limit = (
        close_sign
        * CLOSE_CTRL_MAG
    )

    # --------------------------------------------------------
    # A. 先张开爪子
    # --------------------------------------------------------
    open_steps = int(
        OPEN_HOLD_TIME
        / model.opt.timestep
    )

    for _ in range(open_steps):
        hold_arm(
            data,
            arm_act_ids,
            q_home,
        )

        set_gripper_ctrl(
            data,
            gripper_act_id,
            open_ctrl,
        )

        mujoco.mj_step(
            model,
            data,
        )

        sync_viewer(viewer)

    # --------------------------------------------------------
    # B. 随机物体位姿
    # --------------------------------------------------------
    random_pose = (
        randomize_object_pose(
            model,
            data,
            rng,
        )
    )

    settle_steps = int(
        OBJECT_SETTLE_TIME
        / model.opt.timestep
    )

    for _ in range(settle_steps):
        hold_arm(
            data,
            arm_act_ids,
            q_home,
        )

        set_gripper_ctrl(
            data,
            gripper_act_id,
            open_ctrl,
        )

        mujoco.mj_step(
            model,
            data,
        )

        sync_viewer(viewer)

    object_pos_settled, object_R_settled = (
        get_body_pose(
            model,
            data,
            OBJECT_BODY,
        )
    )

    # --------------------------------------------------------
    # C. 固定抓取姿态
    # --------------------------------------------------------
    _, target_tool_R = get_site_pose(
        model,
        data,
        TOOL_SITE,
    )

    approach_axis = (
        target_tool_R[:, 0].copy()
    )

    approach_axis /= np.linalg.norm(
        approach_axis
    )

    target_grasp_center = (
        object_pos_settled.copy()
    )

    safe_center = (
        target_grasp_center
        - SAFE_DISTANCE
        * approach_axis
    )

    pregrasp_center = (
        target_grasp_center
        - PREGRASP_DISTANCE
        * approach_axis
    )

    # --------------------------------------------------------
    # D. 先去安全位
    # --------------------------------------------------------
    target_tool_pos_safe = (
        grasp_center_to_tool_target(
            safe_center,
            target_tool_R,
        )
    )

    q_now = get_arm_q(
        data,
        arm_qpos_adr,
    )

    q_safe, ok = solve_ik(
        model=model,
        reference_data=data,
        target_tool_pos=target_tool_pos_safe,
        target_tool_R=target_tool_R,
        q_start=q_now,
        arm_qpos_adr=arm_qpos_adr,
        arm_dof_adr=arm_dof_adr,
    )

    if not ok:
        return {
            "valid": False,
            "reason": "safe_pose_ik_failed",
            "metadata": {
                "seed": episode_seed,
                "random_pose": random_pose,
            },
            "samples": [],
        }

    moved = move_arm_joint_space(
        model=model,
        data=data,
        viewer=viewer,
        arm_act_ids=arm_act_ids,
        gripper_act_id=gripper_act_id,
        gripper_ctrl=open_ctrl,
        q_start=q_now,
        q_target=q_safe,
        duration=SAFE_MOVE_TIME,
        gel_geom_ids=gel_geom_ids,
        object_geom_ids=object_geom_ids,
    )

    if not moved:
        return {
            "valid": False,
            "reason": "collision_on_safe_move",
            "metadata": {
                "seed": episode_seed,
                "random_pose": random_pose,
            },
            "samples": [],
        }

    # --------------------------------------------------------
    # E. 安全位 -> 预抓取位（直线）
    # --------------------------------------------------------
    ok = move_grasp_center_cartesian(
        model=model,
        data=data,
        viewer=viewer,
        arm_act_ids=arm_act_ids,
        gripper_act_id=gripper_act_id,
        gripper_ctrl=open_ctrl,
        start_center=safe_center,
        end_center=pregrasp_center,
        target_tool_R=target_tool_R,
        arm_qpos_adr=arm_qpos_adr,
        arm_dof_adr=arm_dof_adr,
        gel_geom_ids=gel_geom_ids,
        object_geom_ids=object_geom_ids,
    )

    if not ok:
        return {
            "valid": False,
            "reason": "failed_safe_to_pregrasp",
            "metadata": {
                "seed": episode_seed,
                "random_pose": random_pose,
            },
            "samples": [],
        }

    # --------------------------------------------------------
    # F. 预抓取位 -> 物体中心（直线）
    # --------------------------------------------------------
    ok = move_grasp_center_cartesian(
        model=model,
        data=data,
        viewer=viewer,
        arm_act_ids=arm_act_ids,
        gripper_act_id=gripper_act_id,
        gripper_ctrl=open_ctrl,
        start_center=pregrasp_center,
        end_center=target_grasp_center,
        target_tool_R=target_tool_R,
        arm_qpos_adr=arm_qpos_adr,
        arm_dof_adr=arm_dof_adr,
        gel_geom_ids=gel_geom_ids,
        object_geom_ids=object_geom_ids,
    )

    if not ok:
        return {
            "valid": False,
            "reason": "failed_pregrasp_to_grasp",
            "metadata": {
                "seed": episode_seed,
                "random_pose": random_pose,
            },
            "samples": [],
        }

    # 当前机械臂关节角，闭合期间保持。
    q_arm_hold = get_arm_q(
        data,
        arm_qpos_adr,
    )

    hold_steps = int(
        AT_GRASP_POSE_HOLD
        / model.opt.timestep
    )

    for _ in range(hold_steps):
        hold_arm(
            data,
            arm_act_ids,
            q_arm_hold,
        )

        set_gripper_ctrl(
            data,
            gripper_act_id,
            open_ctrl,
        )

        mujoco.mj_step(
            model,
            data,
        )

        sync_viewer(viewer)

    # --------------------------------------------------------
    # G. 缓慢闭合 + 同步采集
    # --------------------------------------------------------
    result = collect_contact_sequence(
        model=model,
        data=data,
        viewer=viewer,
        arm_act_ids=arm_act_ids,
        q_arm_hold=q_arm_hold,
        gripper_act_id=gripper_act_id,
        finger_qpos_adr=finger_qpos_adr,
        gel_geom_ids=gel_geom_ids,
        object_geom_ids=object_geom_ids,
        open_ctrl=open_ctrl,
        close_ctrl_limit=close_ctrl_limit,
        close_sign=close_sign,
    )

    object_pos_final, object_R_final = (
        get_body_pose(
            model,
            data,
            OBJECT_BODY,
        )
    )

    metadata = {
        "episode_index": episode_idx,
        "seed": episode_seed,
        "object": OBJECT_BODY,

        "random_pose_command": random_pose,

        "settled_object_position_m":
            object_pos_settled.tolist(),

        "settled_object_rotation_matrix":
            object_R_settled.reshape(-1).tolist(),

        "final_object_position_m":
            object_pos_final.tolist(),

        "final_object_rotation_matrix":
            object_R_final.reshape(-1).tolist(),

        "approach_axis_world":
            approach_axis.tolist(),

        "safe_center_m":
            safe_center.tolist(),

        "pregrasp_center_m":
            pregrasp_center.tolist(),

        "target_grasp_center_m":
            target_grasp_center.tolist(),

        "close_sign": int(close_sign),
        "open_ctrl": float(open_ctrl),
        "close_ctrl_limit":
            float(close_ctrl_limit),

        "capture_hz": CAPTURE_HZ,
        "precontact_frames":
            PRECONTACT_FRAMES,
        "dynamic_frames":
            DYNAMIC_FRAMES,
        "hold_frames":
            HOLD_FRAMES,

        "stable_three_point_contact":
            bool(result["all_contact"]),

        "all_contact_stable_time_s":
            ALL_CONTACT_STABLE_TIME,

        "valid": bool(result["valid"]),
        "reason": result["reason"],
    }

    return {
        "valid": result["valid"],
        "reason": result["reason"],
        "metadata": metadata,
        "samples": result["samples"],
    }


# ============================================================
# 18. 主程序
# ============================================================

def main():
    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"\n找不到 XML:\n"
            f"{XML_PATH}\n"
            "请先修改 XML_PATH。"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("UNDERACTUATED THREE-FINGER TACTILE DATA COLLECTION")
    print("=" * 76)
    print(f"XML: {XML_PATH}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Capture: {CAPTURE_HZ:.1f} Hz")
    print(
        f"Frames/valid episode: "
        f"{EXPECTED_TOTAL_FRAMES}"
    )
    print(
        f"Save GelSight images: "
        f"{SAVE_GELSIGHT_IMAGES}"
    )
    print("=" * 76)

    close_sign = determine_close_sign(
        XML_PATH
    )

    model = (
        mujoco.MjModel.from_xml_path(
            str(XML_PATH)
        )
    )

    data = mujoco.MjData(
        model
    )

    # 正常采集必须开启 contact。
    model.opt.disableflags &= ~int(
        mujoco.mjtDisableBit.mjDSBL_CONTACT
    )

    arm_qpos_adr, arm_dof_adr = (
        get_joint_addresses(
            model,
            ARM_JOINTS,
        )
    )

    finger_qpos_adr, _ = (
        get_joint_addresses(
            model,
            FINGER_JOINTS,
        )
    )

    arm_act_ids = get_actuator_ids(
        model,
        ARM_ACTUATORS,
    )

    gripper_act_id = name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        GRIPPER_ACTUATOR,
    )

    (
        gel_geom_ids,
        object_geom_ids,
    ) = build_contact_ids(
        model
    )

    viewer_context = (
        mujoco.viewer.launch_passive(
            model,
            data,
        )
        if (
            SHOW_VIEWER
            and mujoco.viewer is not None
        )
        else contextlib.nullcontext(
            None
        )
    )

    valid_count = 0
    invalid_count = 0

    summary_rows = []

    with viewer_context as viewer:
        for episode_idx in range(
            NUM_EPISODES
        ):
            print(
                f"\n[EPISODE "
                f"{episode_idx:04d}]"
            )

            result = run_episode(
                model=model,
                data=data,
                viewer=viewer,
                episode_idx=episode_idx,
                close_sign=close_sign,
                arm_act_ids=arm_act_ids,
                gripper_act_id=gripper_act_id,
                arm_qpos_adr=arm_qpos_adr,
                arm_dof_adr=arm_dof_adr,
                finger_qpos_adr=finger_qpos_adr,
                gel_geom_ids=gel_geom_ids,
                object_geom_ids=object_geom_ids,
            )

            episode_dir = (
                OUTPUT_ROOT
                / OBJECT_BODY
                / f"episode_{episode_idx:04d}"
            )

            metadata = result[
                "metadata"
            ]

            metadata["xml_path"] = str(
                XML_PATH
            )

            metadata[
                "save_gelsight_images"
            ] = SAVE_GELSIGHT_IMAGES

            if result["samples"]:
                save_episode(
                    episode_dir,
                    result["samples"],
                    metadata,
                )
            else:
                episode_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with (
                    episode_dir
                    / "metadata.json"
                ).open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        metadata,
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

            if result["valid"]:
                valid_count += 1
                status = "VALID"
            else:
                invalid_count += 1
                status = "INVALID"

            print(
                f"  => {status}: "
                f"{result['reason']}"
            )

            if result["samples"]:
                print(
                    f"  frames saved: "
                    f"{len(result['samples'])}"
                )

            summary_rows.append(
                [
                    episode_idx,
                    status,
                    result["reason"],
                    len(result["samples"]),
                ]
            )

            # Viewer 被人工关闭时终止 batch。
            if (
                viewer is not None
                and not viewer.is_running()
            ):
                print(
                    "Viewer closed. "
                    "Stopping batch."
                )
                break

    summary_path = (
        OUTPUT_ROOT
        / "collection_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "episode",
                "status",
                "reason",
                "num_frames",
            ]
        )

        writer.writerows(
            summary_rows
        )

    print("\n" + "=" * 76)
    print("COLLECTION FINISHED")
    print("=" * 76)
    print(f"valid   : {valid_count}")
    print(f"invalid : {invalid_count}")
    print(
        f"summary : {summary_path}"
    )
    print("=" * 76)

    if not SAVE_GELSIGHT_IMAGES:
        print(
            "\n注意：这次脚本会保存完整运动/接触/标签数据，"
            "但不会保存 GelSight 图像。"
        )
        print(
            "要用于你的“光流 + 三点触觉识别”，下一步必须把"
            " capture_gelsight_images() 接到现有 DeepGelSight 渲染器。"
        )


if __name__ == "__main__":
    main()
