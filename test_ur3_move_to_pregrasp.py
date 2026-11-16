import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# 配置
# ============================================================
XML_PATH = Path(
    r"E:\project\p1\deepgelsight\xml\ur3_with_threefinger_gelsight_table_tomato.xml"
)

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

FINGER_ACTUATORS = [
    "finger1_ctrl",
    "finger2_ctrl",
    "finger3_ctrl",
]

GEL_SITES = [
    "finger1_gelsight_site",
    "finger2_gelsight_site",
    "finger3_gelsight_site",
]

# 已由上一个测试验证：
# 三个 GelSight 名义几何中心 = tool0 坐标系 +X 方向约 149.65 mm
GRASP_OFFSET_TOOL = np.array([0.14965, 0.0, 0.0], dtype=float)

# 预抓取中心位于物体前方 50 mm
PREGRASP_DISTANCE = 0.050

# 物体先自由沉降
SETTLE_TIME = 0.50

# UR3 从 home 平滑运动到预抓取位姿所用时间
MOVE_TIME = 3.0

# IK 参数
IK_MAX_ITER = 500
IK_POS_TOL = 2e-5          # 0.02 mm
IK_ROT_TOL = 2e-4          # rad
IK_DAMPING = 1e-3
IK_STEP_SCALE = 0.55
MAX_DQ_PER_ITER = 0.08     # rad


# ============================================================
# 基础工具函数
# ============================================================
def name2id(model, objtype, name):
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx == -1:
        raise ValueError(f"模型中找不到: {name}")
    return idx


def get_site_pose(model, data, site_name):
    sid = name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    pos = data.site_xpos[sid].copy()
    R = data.site_xmat[sid].reshape(3, 3).copy()
    return pos, R


def get_body_position(model, data, body_name):
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[bid].copy()


def get_grasp_center(model, data):
    tool_pos, tool_R = get_site_pose(model, data, TOOL_SITE)
    grasp_center = tool_pos + tool_R @ GRASP_OFFSET_TOOL
    return grasp_center


def get_actual_gel_center(model, data):
    pts = []
    for name in GEL_SITES:
        sid = name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pts.append(data.site_xpos[sid].copy())
    return np.mean(np.asarray(pts), axis=0)


def reset_home(model, data):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("[OK] 已加载 home keyframe")
    else:
        mujoco.mj_resetData(model, data)
        print("[WARN] 没找到 home keyframe，使用默认状态")

    mujoco.mj_forward(model, data)


def joint_addresses(model):
    qpos_adr = []
    dof_adr = []
    jids = []

    for name in ARM_JOINTS:
        jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        jids.append(jid)
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])

    return (
        np.asarray(jids, dtype=int),
        np.asarray(qpos_adr, dtype=int),
        np.asarray(dof_adr, dtype=int),
    )


def actuator_ids(model, names):
    return np.asarray(
        [name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names],
        dtype=int,
    )


# ============================================================
# 旋转误差
# ============================================================
def rotation_error_world(R_current, R_target):
    """
    返回一个世界坐标系下的 3D 小角度旋转误差向量。
    目标姿态与初始姿态相同，因此迭代过程中这个表达足够稳定。
    """
    R_err = R_target @ R_current.T

    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


# ============================================================
# 6D Damped Least Squares IK
# ============================================================
def solve_ik(
    model,
    data,
    target_tool_pos,
    target_tool_R,
    q_start,
    qpos_adr,
    dof_adr,
):
    """
    保持 tool0 姿态不变，只通过 6 个 UR3 关节把 tool0 移到目标位置。
    使用 tool0 的位置 + 姿态 6D Jacobian。
    """

    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    work.qvel[:] = 0.0
    work.ctrl[:] = data.ctrl

    tool_id = name2id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE)

    q = q_start.copy()

    for it in range(IK_MAX_ITER):
        # 写入当前迭代关节角
        for k, adr in enumerate(qpos_adr):
            work.qpos[adr] = q[k]

        work.qvel[:] = 0.0
        mujoco.mj_forward(model, work)

        p_cur = work.site_xpos[tool_id].copy()
        R_cur = work.site_xmat[tool_id].reshape(3, 3).copy()

        e_pos = target_tool_pos - p_cur
        e_rot = rotation_error_world(R_cur, target_tool_R)

        pos_err = np.linalg.norm(e_pos)
        rot_err = np.linalg.norm(e_rot)

        if pos_err < IK_POS_TOL and rot_err < IK_ROT_TOL:
            return q, True, it + 1, pos_err, rot_err

        # 全模型 Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, work, jacp, jacr, tool_id)

        # 只取 UR3 六个自由度
        J = np.vstack([
            jacp[:, dof_adr],
            jacr[:, dof_adr],
        ])

        e = np.concatenate([e_pos, e_rot])

        # Damped Least Squares:
        # dq = J^T (J J^T + lambda^2 I)^-1 e
        A = J @ J.T + (IK_DAMPING ** 2) * np.eye(6)

        try:
            dq = J.T @ np.linalg.solve(A, e)
        except np.linalg.LinAlgError:
            return q, False, it + 1, pos_err, rot_err

        dq *= IK_STEP_SCALE

        n = np.linalg.norm(dq)
        if n > MAX_DQ_PER_ITER:
            dq *= MAX_DQ_PER_ITER / n

        q += dq

        # 裁剪至关节范围
        for k, joint_name in enumerate(ARM_JOINTS):
            jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if model.jnt_limited[jid]:
                q[k] = np.clip(
                    q[k],
                    model.jnt_range[jid, 0],
                    model.jnt_range[jid, 1],
                )

    # 最后再评估一次
    for k, adr in enumerate(qpos_adr):
        work.qpos[adr] = q[k]
    mujoco.mj_forward(model, work)

    p_cur = work.site_xpos[tool_id].copy()
    R_cur = work.site_xmat[tool_id].reshape(3, 3).copy()

    pos_err = np.linalg.norm(target_tool_pos - p_cur)
    rot_err = np.linalg.norm(rotation_error_world(R_cur, target_tool_R))

    return q, False, IK_MAX_ITER, pos_err, rot_err


# ============================================================
# 仿真辅助
# ============================================================
def hold_home_controls(model, data, q_home, arm_act_ids, finger_act_ids):
    for i, aid in enumerate(arm_act_ids):
        data.ctrl[aid] = q_home[i]

    # 三个手指保持 home / 张开参考位
    for aid in finger_act_ids:
        data.ctrl[aid] = 0.0


def settle_object(
    model,
    data,
    q_home,
    arm_act_ids,
    finger_act_ids,
    viewer=None,
):
    steps = int(SETTLE_TIME / model.opt.timestep)

    print(f"[INFO] 物体自由沉降 {SETTLE_TIME:.2f} s ...")

    for _ in range(steps):
        hold_home_controls(
            model, data, q_home, arm_act_ids, finger_act_ids
        )
        mujoco.mj_step(model, data)

        if viewer is not None and viewer.is_running():
            viewer.sync()


def smooth_move_arm(
    model,
    data,
    q_start,
    q_target,
    arm_act_ids,
    finger_act_ids,
    viewer,
):
    steps = int(MOVE_TIME / model.opt.timestep)

    print(f"[INFO] UR3 平滑运动到预抓取位姿，持续 {MOVE_TIME:.1f} s ...")

    for step in range(steps):
        t = (step + 1) / steps

        # smoothstep: 3t^2 - 2t^3
        s = 3.0 * t * t - 2.0 * t * t * t

        q_cmd = q_start + s * (q_target - q_start)

        for i, aid in enumerate(arm_act_ids):
            data.ctrl[aid] = q_cmd[i]

        # 本测试不闭合手指
        for aid in finger_act_ids:
            data.ctrl[aid] = 0.0

        mujoco.mj_step(model, data)

        if viewer.is_running():
            viewer.sync()

        # 不强制 real-time，只稍微放慢 viewer 刷新
        if step % 5 == 0:
            time.sleep(0.001)


# ============================================================
# 主程序
# ============================================================
def main():
    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"\n找不到 XML:\n{XML_PATH}\n"
            "请把 XML_PATH 改成你本机 XML 的实际路径。"
        )

    print(f"Loading:\n{XML_PATH}\n")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    reset_home(model, data)

    _, qpos_adr, dof_adr = joint_addresses(model)
    arm_act_ids = actuator_ids(model, ARM_ACTUATORS)
    finger_act_ids = actuator_ids(model, FINGER_ACTUATORS)

    q_home = np.array([data.qpos[a] for a in qpos_adr], dtype=float)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # --------------------------------------------------------
        # 1) 先让 tomato 在桌面稳定
        # --------------------------------------------------------
        settle_object(
            model,
            data,
            q_home,
            arm_act_ids,
            finger_act_ids,
            viewer,
        )

        mujoco.mj_forward(model, data)

        # --------------------------------------------------------
        # 2) 读取真实物体位置 + 当前 tool0 姿态
        # --------------------------------------------------------
        object_pos = get_body_position(model, data, OBJECT_BODY)

        tool_pos_now, tool_R_now = get_site_pose(
            model, data, TOOL_SITE
        )

        grasp_now = get_grasp_center(model, data)

        # 当前 tool0 的 +X 世界方向
        approach_axis_world = tool_R_now[:, 0].copy()
        approach_axis_world /= np.linalg.norm(approach_axis_world)

        # --------------------------------------------------------
        # 3) 定义预抓取中心
        #
        # 物体前方 50 mm：
        # 从 pregrasp 沿 tool +X 方向前进，就到物体中心
        # --------------------------------------------------------
        pregrasp_center = (
            object_pos
            - PREGRASP_DISTANCE * approach_axis_world
        )

        # grasp_center = tool_pos + R * offset
        # 因而：
        # tool_target = grasp_target - R * offset
        target_tool_pos = (
            pregrasp_center
            - tool_R_now @ GRASP_OFFSET_TOOL
        )

        target_tool_R = tool_R_now.copy()

        print("=" * 72)
        print("PRE-GRASP TARGET")
        print("=" * 72)

        print("物体中心 [m]          :", np.round(object_pos, 6))
        print("当前抓取中心 [m]      :", np.round(grasp_now, 6))
        print("当前 tool +X 方向     :", np.round(approach_axis_world, 6))
        print("目标预抓取中心 [m]    :", np.round(pregrasp_center, 6))
        print("目标 tool0 位置 [m]   :", np.round(target_tool_pos, 6))

        print(
            "预抓取中心到物体距离  : "
            f"{np.linalg.norm(pregrasp_center - object_pos)*1000:.3f} mm"
        )

        # --------------------------------------------------------
        # 4) 求解 IK
        # --------------------------------------------------------
        print("\n[INFO] 开始求解 UR3 逆运动学 ...")

        q_target, ok, n_iter, pos_err, rot_err = solve_ik(
            model=model,
            data=data,
            target_tool_pos=target_tool_pos,
            target_tool_R=target_tool_R,
            q_start=q_home,
            qpos_adr=qpos_adr,
            dof_adr=dof_adr,
        )

        print("\nIK 结果:")
        print("success     :", ok)
        print("iterations  :", n_iter)
        print("position err:", f"{pos_err*1000:.4f} mm")
        print("rotation err:", f"{rot_err:.6f} rad")

        print("\nq_home [rad]:")
        print(np.round(q_home, 6))

        print("\nq_target [rad]:")
        print(np.round(q_target, 6))

        if not ok:
            print(
                "\n[STOP] IK 未达到设定误差阈值。"
                "\n程序不会驱动机器人。"
            )

            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)

            return

        # --------------------------------------------------------
        # 5) 平滑运动到预抓取位置
        # --------------------------------------------------------
        q_start_move = np.array(
            [data.qpos[a] for a in qpos_adr],
            dtype=float,
        )

        smooth_move_arm(
            model,
            data,
            q_start=q_start_move,
            q_target=q_target,
            arm_act_ids=arm_act_ids,
            finger_act_ids=finger_act_ids,
            viewer=viewer,
        )

        # 稳定一小段
        for _ in range(int(0.3 / model.opt.timestep)):
            for i, aid in enumerate(arm_act_ids):
                data.ctrl[aid] = q_target[i]

            for aid in finger_act_ids:
                data.ctrl[aid] = 0.0

            mujoco.mj_step(model, data)

            if viewer.is_running():
                viewer.sync()

        mujoco.mj_forward(model, data)

        # --------------------------------------------------------
        # 6) 验证最终实际位置
        # --------------------------------------------------------
        object_pos_final = get_body_position(
            model, data, OBJECT_BODY
        )

        grasp_final = get_grasp_center(model, data)
        gel_center_final = get_actual_gel_center(model, data)

        target_err = np.linalg.norm(
            grasp_final - pregrasp_center
        )

        object_distance = np.linalg.norm(
            grasp_final - object_pos_final
        )

        gel_formula_err = np.linalg.norm(
            gel_center_final - grasp_final
        )

        print("\n" + "=" * 72)
        print("FINAL CHECK")
        print("=" * 72)

        print(
            "最终物体中心 [m]      :",
            np.round(object_pos_final, 6)
        )

        print(
            "最终抓取中心 [m]      :",
            np.round(grasp_final, 6)
        )

        print(
            "目标预抓取中心 [m]    :",
            np.round(pregrasp_center, 6)
        )

        print(
            "抓取中心 -> 预抓取目标误差:",
            f"{target_err*1000:.3f} mm"
        )

        print(
            "抓取中心 -> 物体中心距离  :",
            f"{object_distance*1000:.3f} mm"
        )

        print(
            "三 GelSight 实际中心与公式中心误差:",
            f"{gel_formula_err*1000:.4f} mm"
        )

        if target_err < 0.002:
            print("\n[PASS] UR3 已到达预抓取位置。")
        else:
            print(
                "\n[CHECK] UR3 已运动，但预抓取位置误差大于 2 mm。"
            )

        print(
            "\n本测试到此停止："
            "\n- 不继续向物体靠近"
            "\n- 不闭合三个手指"
            "\n请观察 Viewer 确认夹爪方向与位置。"
        )

        # 保持最终姿态直到关闭 viewer
        while viewer.is_running():
            for i, aid in enumerate(arm_act_ids):
                data.ctrl[aid] = q_target[i]

            for aid in finger_act_ids:
                data.ctrl[aid] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()
