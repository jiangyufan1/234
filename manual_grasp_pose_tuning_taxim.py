# -*- coding: utf-8 -*-
r"""
manual_grasp_pose_tuning.py

用途：
    在 UR3 + 三指 GelSight + 桌子 + YCB cracker_box 场景中，
    手动寻找并保存 cracker_box 的标准三指抓取姿态 G*。

默认 XML：
    E:\project\p1\deepgelsight\xml\
    ur3_with_threefinger_gelsight_table_cracker.xml

保存目录：
    E:\project\p1\deepgelsight\grasp_targets\cracker_box\

核心保存内容：
    - 6 个 UR3 关节角
    - 3 个手指关节角
    - tool0 世界位姿
    - cracker_box 世界位姿
    - T_world_tool
    - T_world_object
    - T_object_tool   <-- 推荐作为后续“标准抓取姿态 G*”
    - 三个 GelSight 的接触点、接触法向、接触力
    - 三指是否都有效接触
    - 三路 Taxim/GelSight RGB
    - 三路 Taxim depth height map

操作方式：
    OpenCV 控制窗口必须获得键盘焦点。

模式 1：笛卡尔模式（默认）
    W / S : tool0 世界 X + / -
    D / A : tool0 世界 Y + / -
    Q / E : tool0 世界 Z + / -
    I / K : 绕世界 X 旋转 + / -
    L / J : 绕世界 Y 旋转 + / -
    U / O : 绕世界 Z 旋转 + / -

模式 2：关节模式
    M     : 笛卡尔 / 关节模式切换
    1~6   : 选择 UR3 joint
    7~9   : 选择 finger1~3
    A / D : 选中关节 - / +

通用：
    Z / X : 三个手指控制量同时 - / +
    B     : 重置 cracker_box 到桌面初始位姿
    R     : 整个场景重置到 home keyframe
    P     : 保存当前候选 G* + 三路 GelSight RGB/Depth
    V     : 显示/隐藏 cracker_box 碰撞体
    ESC   : 退出

说明：
    1. P 即使在未形成三指接触时也会保存，但会标记 valid_three_contact=False。
       正式 G* 建议只采用 valid_three_contact=True 的候选。
    2. T_object_tool = inv(T_world_object) @ T_world_tool。
       它不依赖物体在桌面上的世界位置，更适合后续作为标准抓取姿态。
"""

from pathlib import Path
import json
import math
import sys
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from TaximSensor import TaximSensor


# ============================================================
# 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

XML_PATH = (
    PROJECT_ROOT
    / "xml"
    / "ur3_with_threefinger_gelsight_table_cracker.xml"
)

SAVE_DIR = (
    PROJECT_ROOT
    / "grasp_targets"
    / "cracker_box"
)


# ============================================================
# Taxim / GelSight
# ============================================================

TAXIM_SENSOR_TYPE = "gelsight_r1.5"

# 每个 TaximSensor 只负责一根手指；三个实例共享同一个 MuJoCo model/data。
TAXIM_FINGERS = [
    {
        "name": "finger1",
        "site": "finger1_gelsight_site",
        "pad": "finger1_gel",
    },
    {
        "name": "finger2",
        "site": "finger2_gelsight_site",
        "pad": "finger2_gel",
    },
    {
        "name": "finger3",
        "site": "finger3_gelsight_site",
        "pad": "finger3_gel",
    },
]

# Taxim 中注册的是 cracker_box 的完整 visual mesh。
# MuJoCo 的真实碰撞仍然由 textured_coacd_0/1.stl 两个 collision hull 完成。
#
# mujoco-taxim 的 render_taxim 会通过这个 geom 所属 body 检查
# cracker_box body 与各 finger pad 的接触/穿透，因此一个 visual geom
# 就可以代表同一个物体，无需把两个 CoACD hull 当成两个“Taxim物体”。
TAXIM_OBJECT_GEOM = "cracker_visual_geom"
TAXIM_OBJECT_MESH = "cracker_visual_mesh"

TAXIM_SHADOW = True
TAXIM_IMG_NOISE_SIGMA = 2
TAXIM_SAVE_POINT_CLOUD = True


# ============================================================
# 模型名称
# ============================================================

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

OBJECT_COLLISION_GEOMS = [
    "cracker_collision_0",
    "cracker_collision_1",
]

TOOL_SITE = "tool0"
OBJECT_BODY = "cracker_box"
OBJECT_FREE_JOINT = "cracker_box_free"
HOME_KEY = "home"


# ============================================================
# 调节步长
# ============================================================

CART_TRANSLATION_STEP = 0.003       # 3 mm / key
CART_ROTATION_STEP = math.radians(2.0)  # 2 deg / key

ARM_JOINT_STEP = math.radians(2.0)
FINGER_JOINT_STEP = 0.015

IK_DAMPING = 0.04
IK_MAX_DQ = math.radians(4.0)

SIM_STEPS_PER_FRAME = 5


def obj_id(model, obj_type, name):
    i = mujoco.mj_name2id(model, obj_type, name)
    if i < 0:
        raise RuntimeError(f"MuJoCo 中找不到：{name}")
    return i


def rotation_matrix_to_quat_wxyz(R):
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(R, dtype=np.float64).reshape(9))
    return quat


def pose_matrix(pos, R):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def joint_qpos_scalar(model, data, joint_id):
    adr = int(model.jnt_qposadr[joint_id])
    return float(data.qpos[adr])


def set_hinge_joint_qpos(model, data, joint_id, value):
    adr = int(model.jnt_qposadr[joint_id])

    if bool(model.jnt_limited[joint_id]):
        lo, hi = model.jnt_range[joint_id]
        value = float(np.clip(value, lo, hi))

    data.qpos[adr] = value
    return value


def set_actuator_ctrl(model, data, actuator_id, value):
    if bool(model.actuator_ctrllimited[actuator_id]):
        lo, hi = model.actuator_ctrlrange[actuator_id]
        value = float(np.clip(value, lo, hi))
    data.ctrl[actuator_id] = value
    return value


def reset_scene(model, data, ids):
    key_id = ids["home_key"]
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)


def reset_object(model, data, ids, initial_object_qpos):
    qadr = ids["object_qpos_adr"]
    dadr = ids["object_dof_adr"]

    data.qpos[qadr:qadr + 7] = initial_object_qpos
    data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def sync_arm_ctrl_to_qpos(model, data, ids):
    for jid, aid in zip(ids["arm_joint_ids"], ids["arm_actuator_ids"]):
        q = joint_qpos_scalar(model, data, jid)
        set_actuator_ctrl(model, data, aid, q)


def sync_finger_ctrl_to_qpos(model, data, ids):
    for jid, aid in zip(ids["finger_joint_ids"], ids["finger_actuator_ids"]):
        q = joint_qpos_scalar(model, data, jid)
        set_actuator_ctrl(model, data, aid, q)


def cartesian_jog(model, data, ids, delta_translation=None, delta_rotation=None):
    """
    用 tool0 的 6D Jacobian 做小步差分 IK。
    仅使用 UR3 的 6 个关节列，不动 finger 和 object freejoint。
    """
    nv = model.nv
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)

    mujoco.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        ids["tool_site_id"],
    )

    dof_cols = ids["arm_dof_adrs"]
    J = np.vstack(
        [
            jacp[:, dof_cols],
            jacr[:, dof_cols],
        ]
    )

    twist = np.zeros(6, dtype=np.float64)

    if delta_translation is not None:
        twist[:3] = np.asarray(delta_translation, dtype=np.float64)

    if delta_rotation is not None:
        twist[3:] = np.asarray(delta_rotation, dtype=np.float64)

    A = J @ J.T + (IK_DAMPING ** 2) * np.eye(6)
    dq = J.T @ np.linalg.solve(A, twist)
    dq = np.clip(dq, -IK_MAX_DQ, IK_MAX_DQ)

    for local_i, jid in enumerate(ids["arm_joint_ids"]):
        q_old = joint_qpos_scalar(model, data, jid)
        q_new = set_hinge_joint_qpos(
            model,
            data,
            jid,
            q_old + dq[local_i],
        )
        set_actuator_ctrl(
            model,
            data,
            ids["arm_actuator_ids"][local_i],
            q_new,
        )

    mujoco.mj_forward(model, data)


def joint_jog(model, data, ids, selected_index, direction):
    """
    selected_index:
        0..5 -> arm
        6..8 -> fingers
    """
    if selected_index < 6:
        jid = ids["arm_joint_ids"][selected_index]
        aid = ids["arm_actuator_ids"][selected_index]
        step = ARM_JOINT_STEP
    else:
        finger_idx = selected_index - 6
        jid = ids["finger_joint_ids"][finger_idx]
        aid = ids["finger_actuator_ids"][finger_idx]
        step = FINGER_JOINT_STEP

    q_old = joint_qpos_scalar(model, data, jid)
    q_new = set_hinge_joint_qpos(
        model,
        data,
        jid,
        q_old + direction * step,
    )
    set_actuator_ctrl(model, data, aid, q_new)
    mujoco.mj_forward(model, data)


def all_fingers_jog(model, data, ids, direction):
    for jid, aid in zip(ids["finger_joint_ids"], ids["finger_actuator_ids"]):
        q_old = joint_qpos_scalar(model, data, jid)
        q_new = set_hinge_joint_qpos(
            model,
            data,
            jid,
            q_old + direction * FINGER_JOINT_STEP,
        )
        set_actuator_ctrl(model, data, aid, q_new)

    mujoco.mj_forward(model, data)


def collect_contacts(model, data, ids):
    """
    返回每个 finger 对 cracker_box 的所有接触。
    法向统一为 GelSight pad -> cracker_box 方向。
    """
    all_contacts = []

    object_geom_ids = set(ids["object_collision_geom_ids"])

    for finger_idx, gel_id in enumerate(ids["gel_geom_ids"]):
        finger_contacts = []

        for ci in range(data.ncon):
            c = data.contact[ci]

            direct = (
                c.geom1 == gel_id
                and c.geom2 in object_geom_ids
            )
            reverse = (
                c.geom2 == gel_id
                and c.geom1 in object_geom_ids
            )

            if not (direct or reverse):
                continue

            normal = np.asarray(c.frame[:3], dtype=np.float64).copy()

            # MuJoCo contact normal is geom1 -> geom2.
            # We standardize to gel -> object.
            if reverse:
                normal = -normal

            contact_force6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                model,
                data,
                ci,
                contact_force6,
            )

            finger_contacts.append(
                {
                    "contact_index": int(ci),
                    "pos_world": np.asarray(c.pos, dtype=np.float64).copy(),
                    "normal_gel_to_object_world": normal,
                    "distance": float(c.dist),
                    "force_contact_frame": contact_force6.copy(),
                    "geom1": int(c.geom1),
                    "geom2": int(c.geom2),
                }
            )

        all_contacts.append(finger_contacts)

    return all_contacts


def representative_contact(finger_contacts):
    if not finger_contacts:
        return None

    # 选 penetration 最大（dist 最负）的接触点
    return min(
        finger_contacts,
        key=lambda x: x["distance"],
    )


def get_pose_data(model, data, ids):
    tool_pos = data.site_xpos[ids["tool_site_id"]].copy()
    tool_R = data.site_xmat[ids["tool_site_id"]].reshape(3, 3).copy()
    tool_quat = rotation_matrix_to_quat_wxyz(tool_R)

    object_pos = data.xpos[ids["object_body_id"]].copy()
    object_R = data.xmat[ids["object_body_id"]].reshape(3, 3).copy()
    object_quat = rotation_matrix_to_quat_wxyz(object_R)

    T_world_tool = pose_matrix(tool_pos, tool_R)
    T_world_object = pose_matrix(object_pos, object_R)

    T_object_tool = (
        np.linalg.inv(T_world_object)
        @ T_world_tool
    )

    T_tool_object = (
        np.linalg.inv(T_world_tool)
        @ T_world_object
    )

    return {
        "tool_pos_world": tool_pos,
        "tool_R_world": tool_R,
        "tool_quat_wxyz_world": tool_quat,
        "object_pos_world": object_pos,
        "object_R_world": object_R,
        "object_quat_wxyz_world": object_quat,
        "T_world_tool": T_world_tool,
        "T_world_object": T_world_object,
        "T_object_tool": T_object_tool,
        "T_tool_object": T_tool_object,
    }



def init_taxim_sensors(model, data):
    """
    初始化三路 Taxim。

    注意：
    mujoco-taxim 的 add_geom_mujoco 文档要求至少执行过一次 mj_step，
    所以 main() 会在调用本函数前先 step 一次，然后再 reset 回 home。
    """
    sensors = []

    print("\n[Taxim] 初始化三路 GelSight ...")

    for cfg in TAXIM_FINGERS:
        sim = TaximSensor(
            sensor_type=TAXIM_SENSOR_TYPE,
            bg_file=None,
            bg_index=0,
            resize=None,
            preprocess_bg=True,
        )

        sim.add_geom_mujoco(
            TAXIM_OBJECT_GEOM,
            model=model,
            data=data,
            mesh_name=TAXIM_OBJECT_MESH,
        )

        sim.add_camera_mujoco(
            cfg["site"],
            model,
            data,
        )

        sim.set_sensor_pad_geom(
            cfg["pad"]
        )

        sensors.append(
            {
                **cfg,
                "sim": sim,
            }
        )

        print(
            f"  [OK] {cfg['name']}: "
            f"{cfg['site']} + {cfg['pad']}"
        )

    print(
        f"[Taxim] object = {TAXIM_OBJECT_GEOM} "
        f"(mesh={TAXIM_OBJECT_MESH})"
    )

    return sensors


def render_three_taxim(model, data, taxim_sensors):
    """
    在当前 MuJoCo 状态下同步渲染三路 GelSight。
    如果某根手指没有接触物体，Taxim 会返回该传感器背景图。
    """
    outputs = []

    for item in taxim_sensors:
        rgb, depth, pcn = item["sim"].render_taxim(
            model,
            data,
            shadow=TAXIM_SHADOW,
            get_depth=True,
            img_noise_sigma=TAXIM_IMG_NOISE_SIGMA,
            pcn_add_noise=False,
            visualize=False,
            cycle_bg=False,
        )

        outputs.append(
            {
                "name": item["name"],
                "rgb": np.asarray(rgb).copy(),
                "depth": np.asarray(depth).copy(),
                "pcn": np.asarray(pcn).copy(),
            }
        )

    return outputs


def next_grasp_dir():
    """
    自动生成：
      grasp_001
      grasp_002
      ...
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    max_id = 0
    for p in SAVE_DIR.glob("grasp_*"):
        if not p.is_dir():
            continue
        m = re.match(r"grasp_(\d+)$", p.name)
        if m:
            max_id = max(max_id, int(m.group(1)))

    d = SAVE_DIR / f"grasp_{max_id + 1:03d}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def depth_preview(depth):
    """
    仅用于人眼检查，不改变保存的原始 depth.npy。
    """
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d)

    if not np.any(finite):
        gray = np.zeros(d.shape, dtype=np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    vals = d[finite]
    lo = float(np.min(vals))
    hi = float(np.max(vals))

    if abs(hi - lo) < 1e-12:
        gray = np.zeros(d.shape, dtype=np.uint8)
    else:
        x = np.zeros(d.shape, dtype=np.float32)
        x[finite] = (d[finite] - lo) / (hi - lo)
        gray = np.clip(x * 255.0, 0, 255).astype(np.uint8)

    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def save_taxim_outputs(grasp_dir, taxim_outputs):
    """
    保存：
      gelsight_1.png
      gelsight_2.png
      gelsight_3.png
      gelsight_i_depth.npy
      gelsight_i_pcn.npy（若有）
      gelsight_preview.png
    """
    rgb_panels = []

    for i, item in enumerate(taxim_outputs, start=1):
        rgb = item["rgb"]
        depth = item["depth"]
        pcn = item["pcn"]

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        rgb_path = grasp_dir / f"gelsight_{i}.png"
        depth_path = grasp_dir / f"gelsight_{i}_depth.npy"

        cv2.imwrite(str(rgb_path), rgb_bgr)
        np.save(depth_path, depth)

        if TAXIM_SAVE_POINT_CLOUD and pcn.size > 0:
            np.save(
                grasp_dir / f"gelsight_{i}_pcn.npy",
                pcn,
            )

        panel = rgb_bgr.copy()
        cv2.putText(
            panel,
            f"finger{i}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rgb_panels.append(panel)

        # 另外保存一张 depth 可视化图，方便快速检查；训练仍读取 .npy
        cv2.imwrite(
            str(grasp_dir / f"gelsight_{i}_depth_preview.png"),
            depth_preview(depth),
        )

    # 三张 RGB 高度统一后横向拼接
    h = max(x.shape[0] for x in rgb_panels)
    resized = []
    for x in rgb_panels:
        if x.shape[0] != h:
            scale = h / x.shape[0]
            x = cv2.resize(
                x,
                (int(round(x.shape[1] * scale)), h),
            )
        resized.append(x)

    preview = np.concatenate(resized, axis=1)
    cv2.imwrite(
        str(grasp_dir / "gelsight_preview.png"),
        preview,
    )

    # 保存后弹出三路预览；不阻塞主循环
    cv2.imshow(
        "Saved Three GelSight RGB",
        preview,
    )
    cv2.waitKey(1)


def save_target(model, data, ids, contacts, taxim_sensors):
    """
    按 P 时：
      1) 读取当前三指接触状态；
      2) 同步渲染三路 Taxim；
      3) 保存 G*、机器人/物体状态、接触点/法向/力；
      4) 保存三路 GelSight RGB + depth。

    每次 P 创建一个独立目录 grasp_XXX。
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    pose = get_pose_data(model, data, ids)

    arm_qpos = np.array(
        [
            joint_qpos_scalar(model, data, jid)
            for jid in ids["arm_joint_ids"]
        ],
        dtype=np.float64,
    )

    finger_qpos = np.array(
        [
            joint_qpos_scalar(model, data, jid)
            for jid in ids["finger_joint_ids"]
        ],
        dtype=np.float64,
    )

    reps = [
        representative_contact(x)
        for x in contacts
    ]

    valid = np.array(
        [x is not None for x in reps],
        dtype=bool,
    )

    contact_pos = np.full((3, 3), np.nan, dtype=np.float64)
    contact_normal = np.full((3, 3), np.nan, dtype=np.float64)
    contact_force = np.full((3, 6), np.nan, dtype=np.float64)
    contact_distance = np.full(3, np.nan, dtype=np.float64)

    for i, rep in enumerate(reps):
        if rep is None:
            continue

        contact_pos[i] = rep["pos_world"]
        contact_normal[i] = rep["normal_gel_to_object_world"]
        contact_force[i] = rep["force_contact_frame"]
        contact_distance[i] = rep["distance"]

    valid_three_contact = bool(np.all(valid))

    # 先渲染，再写文件；如果 Taxim 报错，不生成半套 G* 数据。
    print("\n[Taxim] 正在同步渲染 3 路 GelSight ...")
    taxim_outputs = render_three_taxim(
        model,
        data,
        taxim_sensors,
    )

    grasp_dir = next_grasp_dir()

    object_qpos = data.qpos[
        ids["object_qpos_adr"]:
        ids["object_qpos_adr"] + 7
    ].copy()

    # ---------- tactile images ----------
    save_taxim_outputs(
        grasp_dir,
        taxim_outputs,
    )

    # ---------- numeric target ----------
    npz_path = grasp_dir / "target.npz"
    json_path = grasp_dir / "target.json"

    sensor_pos_world = []
    sensor_R_world = []

    for cfg in TAXIM_FINGERS:
        sid = obj_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            cfg["site"],
        )
        sensor_pos_world.append(
            data.site_xpos[sid].copy()
        )
        sensor_R_world.append(
            data.site_xmat[sid].reshape(3, 3).copy()
        )

    sensor_pos_world = np.stack(sensor_pos_world, axis=0)
    sensor_R_world = np.stack(sensor_R_world, axis=0)

    taxim_depth_shapes = np.array(
        [
            item["depth"].shape
            for item in taxim_outputs
        ],
        dtype=np.int32,
    )

    np.savez(
        npz_path,

        # validity
        valid_three_contact=np.array(valid_three_contact),
        contact_valid=valid,

        # robot / hand
        arm_qpos=arm_qpos,
        finger_qpos=finger_qpos,
        ctrl=data.ctrl.copy(),

        # object
        object_freejoint_qpos=object_qpos,

        # world poses
        tool_pos_world=pose["tool_pos_world"],
        tool_R_world=pose["tool_R_world"],
        tool_quat_wxyz_world=pose["tool_quat_wxyz_world"],
        object_pos_world=pose["object_pos_world"],
        object_R_world=pose["object_R_world"],
        object_quat_wxyz_world=pose["object_quat_wxyz_world"],

        # transforms
        T_world_tool=pose["T_world_tool"],
        T_world_object=pose["T_world_object"],

        # 这个是后续标准抓取姿态 G* 的核心
        T_object_tool=pose["T_object_tool"],
        T_tool_object=pose["T_tool_object"],

        # contact
        contact_pos_world=contact_pos,
        contact_normal_gel_to_object_world=contact_normal,
        contact_force_contact_frame=contact_force,
        contact_distance=contact_distance,

        # sensor poses
        sensor_pos_world=sensor_pos_world,
        sensor_R_world=sensor_R_world,

        # Taxim bookkeeping
        taxim_depth_shapes=taxim_depth_shapes,
    )

    meta = {
        "object": "cracker_box",
        "grasp_dir": grasp_dir.name,

        "valid_three_contact": valid_three_contact,
        "contact_valid": valid.astype(int).tolist(),

        "arm_joint_names": ARM_JOINTS,
        "finger_joint_names": FINGER_JOINTS,

        "arm_qpos_rad": arm_qpos.tolist(),
        "arm_qpos_deg": np.degrees(arm_qpos).tolist(),
        "finger_qpos_rad": finger_qpos.tolist(),

        "tool_pos_world_m": pose["tool_pos_world"].tolist(),
        "tool_quat_wxyz_world": pose["tool_quat_wxyz_world"].tolist(),

        "object_pos_world_m": pose["object_pos_world"].tolist(),
        "object_quat_wxyz_world": pose["object_quat_wxyz_world"].tolist(),

        # canonical G*
        "T_object_tool": pose["T_object_tool"].tolist(),

        "contact_pos_world_m": contact_pos.tolist(),
        "contact_normal_gel_to_object_world": contact_normal.tolist(),

        "taxim": {
            "sensor_type": TAXIM_SENSOR_TYPE,
            "object_geom": TAXIM_OBJECT_GEOM,
            "object_mesh": TAXIM_OBJECT_MESH,
            "rgb_files": [
                "gelsight_1.png",
                "gelsight_2.png",
                "gelsight_3.png",
            ],
            "depth_files": [
                "gelsight_1_depth.npy",
                "gelsight_2_depth.npy",
                "gelsight_3_depth.npy",
            ],
            "preview_file": "gelsight_preview.png",
            "depth_note": (
                "Raw Taxim returned ground-truth height map. "
                "mujoco-taxim internally converts world pose to millimeters "
                "during tactile rendering."
            ),
        },

        "note": (
            "Use T_object_tool as the canonical target grasp G*. "
            "A formal target grasp should normally have "
            "valid_three_contact=true and be physically stable."
        ),
    }

    json_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("[SAVE DIR]", grasp_dir)
    print("[SAVE]", npz_path)
    print("[SAVE]", json_path)
    print("[SAVE] gelsight_1.png / 2.png / 3.png")
    print("[SAVE] gelsight_1_depth.npy / 2 / 3")
    print("valid_three_contact =", valid_three_contact)
    print("contact_valid       =", valid.tolist())
    print("\nT_object_tool = G*")
    print(
        np.array2string(
            pose["T_object_tool"],
            precision=6,
            suppress_small=True,
        )
    )

    if not valid_three_contact:
        print(
            "\n[WARNING] 当前没有形成三指全部有效接触。"
            "\n这组图像和状态已经保存，但暂时不要作为正式 G*。"
        )
    else:
        print(
            "\n[OK] THREE-CONTACT = YES。"
            "\n这是一条完整的候选 G*："
            "姿态 + 三点接触 + 三路 GelSight RGB/Depth 已同步保存。"
        )

    print("=" * 76)



def make_control_panel(model, data, ids, mode, selected, contacts, collision_visible):
    H, W = 720, 980
    panel = np.zeros((H, W, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(text, y, scale=0.55, thickness=1, x=20):
        cv2.putText(
            panel,
            text,
            (x, y),
            font,
            scale,
            (235, 235, 235),
            thickness,
            cv2.LINE_AA,
        )

    cv2.putText(
        panel,
        "UR3 + Three-Finger GelSight | cracker_box grasp target tuning",
        (20, 32),
        font,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    mode_text = "CARTESIAN" if mode == "cartesian" else "JOINT"
    cv2.putText(
        panel,
        f"MODE: {mode_text}   [M toggles mode]",
        (20, 64),
        font,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    pose = get_pose_data(model, data, ids)
    p = pose["tool_pos_world"]
    put(
        f"tool0 pos W: x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f} m",
        100,
    )

    arm_vals = [
        joint_qpos_scalar(model, data, jid)
        for jid in ids["arm_joint_ids"]
    ]
    finger_vals = [
        joint_qpos_scalar(model, data, jid)
        for jid in ids["finger_joint_ids"]
    ]

    put(
        "UR3 joints deg: "
        + "  ".join(f"{math.degrees(v):+.1f}" for v in arm_vals),
        128,
    )
    put(
        "Finger joints rad: "
        + "  ".join(f"{v:+.3f}" for v in finger_vals),
        156,
    )

    y = 196
    for i, c in enumerate(contacts):
        rep = representative_contact(c)
        if rep is None:
            text = f"finger{i+1}: NO CONTACT"
            color = (70, 70, 255)
        else:
            fn = abs(float(rep["force_contact_frame"][0]))
            pen_mm = max(0.0, -rep["distance"] * 1000.0)
            text = (
                f"finger{i+1}: CONTACT  n={len(c)}  "
                f"penetration={pen_mm:.3f} mm  Fn~{fn:.3f} N"
            )
            color = (80, 255, 80)

        cv2.putText(
            panel,
            text,
            (20, y),
            font,
            0.58,
            color,
            2 if rep is not None else 1,
            cv2.LINE_AA,
        )
        y += 28

    three_ok = all(len(c) > 0 for c in contacts)
    cv2.putText(
        panel,
        "THREE-CONTACT: YES" if three_ok else "THREE-CONTACT: NO",
        (20, 294),
        font,
        0.7,
        (80, 255, 80) if three_ok else (70, 70, 255),
        2,
        cv2.LINE_AA,
    )

    if mode == "cartesian":
        put("CARTESIAN:", 340, 0.62, 2)
        put("W/S : world X +/-     D/A : world Y +/-     Q/E : world Z +/-", 370)
        put("I/K : rot X +/-       L/J : rot Y +/-       U/O : rot Z +/-", 398)
    else:
        selected_names = ARM_JOINTS + FINGER_JOINTS
        put("JOINT:", 340, 0.62, 2)
        put(
            f"Selected {selected+1}: {selected_names[selected]}",
            370,
            0.58,
            2,
        )
        put("1~6 UR3 | 7~9 fingers | A/D decrease/increase", 398)

    put("Z/X : all three finger controls - / +", 448)
    put("B   : reset cracker_box upright on table", 476)
    put("R   : reset whole scene to HOME", 504)
    put("P   : save G* + 3 GelSight RGB/depth", 532)
    put(
        f"V   : collision hulls {'VISIBLE' if collision_visible else 'HIDDEN'}",
        560,
    )
    put("ESC : quit", 588)

    put(
        "Recommended final G*: save when THREE-CONTACT = YES and grasp pose is stable.",
        642,
        0.53,
        2,
    )
    put(
        "Canonical pose saved as T_object_tool = inv(T_world_object) @ T_world_tool.",
        670,
        0.50,
    )

    return panel


def set_collision_visibility(model, ids, visible):
    alpha = 0.22 if visible else 0.0
    for gid in ids["object_collision_geom_ids"]:
        model.geom_rgba[gid, 3] = alpha


def main():
    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"找不到 XML：\n{XML_PATH}"
        )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("XML =", XML_PATH)
    print("SAVE_DIR =", SAVE_DIR)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    ids = {}

    ids["tool_site_id"] = obj_id(
        model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE
    )
    ids["object_body_id"] = obj_id(
        model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY
    )

    object_joint_id = obj_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_FREE_JOINT
    )
    ids["object_joint_id"] = object_joint_id
    ids["object_qpos_adr"] = int(model.jnt_qposadr[object_joint_id])
    ids["object_dof_adr"] = int(model.jnt_dofadr[object_joint_id])

    ids["arm_joint_ids"] = [
        obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, x)
        for x in ARM_JOINTS
    ]
    ids["finger_joint_ids"] = [
        obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, x)
        for x in FINGER_JOINTS
    ]

    ids["arm_actuator_ids"] = [
        obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, x)
        for x in ARM_ACTUATORS
    ]
    ids["finger_actuator_ids"] = [
        obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, x)
        for x in FINGER_ACTUATORS
    ]

    ids["arm_dof_adrs"] = [
        int(model.jnt_dofadr[jid])
        for jid in ids["arm_joint_ids"]
    ]

    ids["gel_geom_ids"] = [
        obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, x)
        for x in GEL_GEOMS
    ]

    ids["object_collision_geom_ids"] = [
        obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, x)
        for x in OBJECT_COLLISION_GEOMS
    ]

    home_key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        HOME_KEY,
    )
    ids["home_key"] = home_key_id

    reset_scene(model, data, ids)

    initial_object_qpos = data.qpos[
        ids["object_qpos_adr"]:
        ids["object_qpos_adr"] + 7
    ].copy()

    sync_arm_ctrl_to_qpos(model, data, ids)
    sync_finger_ctrl_to_qpos(model, data, ids)

    # mujoco-taxim 的 add_geom_mujoco 要求仿真至少 step 过一次。
    mujoco.mj_step(model, data)

    taxim_sensors = init_taxim_sensors(
        model,
        data,
    )

    # Taxim 初始化后恢复到完全相同的 home 初态，
    # 避免上面的单步仿真造成物体/机械臂微小位移。
    reset_scene(model, data, ids)
    sync_arm_ctrl_to_qpos(model, data, ids)
    sync_finger_ctrl_to_qpos(model, data, ids)

    mode = "cartesian"
    selected = 0
    collision_visible = True
    set_collision_visibility(model, ids, collision_visible)

    print("\n控制窗口已经启动。")
    print("请点击 OpenCV 的控制面板后使用键盘。")
    print("P = 保存当前候选 G* + 三路 GelSight RGB/Depth。")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for _ in range(SIM_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)

            contacts = collect_contacts(model, data, ids)

            panel = make_control_panel(
                model,
                data,
                ids,
                mode,
                selected,
                contacts,
                collision_visible,
            )

            cv2.imshow("Manual Grasp Pose Tuning - Controls", panel)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            # ---------------- mode ----------------
            elif key in (ord("m"), ord("M")):
                mode = (
                    "joint"
                    if mode == "cartesian"
                    else "cartesian"
                )
                print("[MODE]", mode)

            # ---------------- universal ----------------
            elif key in (ord("z"), ord("Z")):
                all_fingers_jog(model, data, ids, -1.0)

            elif key in (ord("x"), ord("X")):
                all_fingers_jog(model, data, ids, +1.0)

            elif key in (ord("b"), ord("B")):
                reset_object(
                    model,
                    data,
                    ids,
                    initial_object_qpos,
                )
                print("[RESET] cracker_box")

            elif key in (ord("r"), ord("R")):
                reset_scene(model, data, ids)
                sync_arm_ctrl_to_qpos(model, data, ids)
                sync_finger_ctrl_to_qpos(model, data, ids)
                print("[RESET] HOME")

            elif key in (ord("p"), ord("P")):
                contacts = collect_contacts(
                    model,
                    data,
                    ids,
                )
                save_target(
                    model,
                    data,
                    ids,
                    contacts,
                    taxim_sensors,
                )

            elif key in (ord("v"), ord("V")):
                collision_visible = not collision_visible
                set_collision_visibility(
                    model,
                    ids,
                    collision_visible,
                )

            # ---------------- cartesian ----------------
            elif mode == "cartesian":
                if key in (ord("w"), ord("W")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_translation=[
                            +CART_TRANSLATION_STEP,
                            0,
                            0,
                        ],
                    )

                elif key in (ord("s"), ord("S")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_translation=[
                            -CART_TRANSLATION_STEP,
                            0,
                            0,
                        ],
                    )

                elif key in (ord("d"), ord("D")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_translation=[
                            0,
                            +CART_TRANSLATION_STEP,
                            0,
                        ],
                    )

                elif key in (ord("a"), ord("A")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_translation=[
                            0,
                            -CART_TRANSLATION_STEP,
                            0,
                        ],
                    )

                elif key in (ord("q"), ord("Q")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_translation=[
                            0,
                            0,
                            +CART_TRANSLATION_STEP,
                        ],
                    )

                elif key in (ord("e"), ord("E")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_translation=[
                            0,
                            0,
                            -CART_TRANSLATION_STEP,
                        ],
                    )

                elif key in (ord("i"), ord("I")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_rotation=[
                            +CART_ROTATION_STEP,
                            0,
                            0,
                        ],
                    )

                elif key in (ord("k"), ord("K")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_rotation=[
                            -CART_ROTATION_STEP,
                            0,
                            0,
                        ],
                    )

                elif key in (ord("l"), ord("L")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_rotation=[
                            0,
                            +CART_ROTATION_STEP,
                            0,
                        ],
                    )

                elif key in (ord("j"), ord("J")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_rotation=[
                            0,
                            -CART_ROTATION_STEP,
                            0,
                        ],
                    )

                elif key in (ord("u"), ord("U")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_rotation=[
                            0,
                            0,
                            +CART_ROTATION_STEP,
                        ],
                    )

                elif key in (ord("o"), ord("O")):
                    cartesian_jog(
                        model,
                        data,
                        ids,
                        delta_rotation=[
                            0,
                            0,
                            -CART_ROTATION_STEP,
                        ],
                    )

            # ---------------- joint ----------------
            else:
                if ord("1") <= key <= ord("9"):
                    selected = key - ord("1")
                    selected_names = ARM_JOINTS + FINGER_JOINTS
                    print(
                        "[SELECT]",
                        selected + 1,
                        selected_names[selected],
                    )

                elif key in (ord("a"), ord("A")):
                    joint_jog(
                        model,
                        data,
                        ids,
                        selected,
                        -1.0,
                    )

                elif key in (ord("d"), ord("D")):
                    joint_jog(
                        model,
                        data,
                        ids,
                        selected,
                        +1.0,
                    )

            viewer.sync()

    cv2.destroyAllWindows()
    print("\n程序结束。")
    print("候选 G* 保存目录：", SAVE_DIR)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n用户终止。")
    except Exception as exc:
        cv2.destroyAllWindows()
        print("\n[ERROR]", repr(exc))
        print("请把完整报错发给我。")
        sys.exit(1)
