"""
采集计划 v3：按物体稳定摆放与物体系接触配置分组采集。

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
10) 使用项目内 TaximSensor 对三路 GelSight 同步渲染 RGB 与深度图。
    每个 episode 保存时间序列压缩包，便于后续光流计算。

重要：
- 先用 NUM_EPISODES=5~20 做 pilot。
- 确认运动、闭合方向和接触判定正确后再批量采。
"""

import contextlib
import csv
import json
import math
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from collection_storage import (
    clear_episode_files,
    episode_paths,
    load_png_sequences,
    save_png_sequences,
    storage_metadata,
)

_SCRIPT_ROOT = Path(__file__).resolve().parent
_taxim_source_candidates = (
    _SCRIPT_ROOT / "third_party" / "mujoco-taxim-main" / "src",
    Path(r"E:\project\p1\deepgelsight\third_party\mujoco-taxim-main\src"),
)
_taxim_deps_candidates = (
    _SCRIPT_ROOT / "third_party" / "_deps",
    _SCRIPT_ROOT / "_deps",
    _SCRIPT_ROOT.parent / "rl" / "新建文件夹" / "output" / "tactile_collection" / "_deps",
    Path(r"E:\project\p1\rl\新建文件夹\output\tactile_collection\_deps"),
)
TAXIM_SOURCE = next((p for p in _taxim_source_candidates if p.is_dir()), None)
TAXIM_DEPS = next((p for p in _taxim_deps_candidates if p.is_dir()), None)
# 插入顺序保证工作目录内源码优先，依赖目录仍可被 Taxim 导入。
for _taxim_path in (TAXIM_DEPS, TAXIM_SOURCE):
    if _taxim_path is not None:
        sys.path.insert(0, str(_taxim_path))
# Taxim 的 Numba 装饰器在导入时启用磁盘缓存；源码可能来自只读 E 盘。
# 将默认缓存放到系统临时目录，避免导入阶段探测不可写的源码目录。
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "mcb_taxim_numba_cache"))
from TaximSensor import TaximSensor

try:
    import mujoco.viewer
except Exception:
    mujoco.viewer = None


# ============================================================
# 0. 路径与采集规模
# ============================================================

XML_PATH = Path(os.environ.get(
    "TACTILE_XML_PATH",
    str(Path(r"E:\project\p1\deepgelsight\xml")
        / "ur3_with_threefinger_gelsight_table_tomato_underactuated.xml"),
))

OUTPUT_ROOT = Path(
    r"E:\project\p1\deepgelsight\dataset_underactuated"
)
RUN_OUTPUT_ROOT = Path(os.environ.get(
    "TACTILE_RUN_OUTPUT_ROOT",
    str(OUTPUT_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")),
))
# legacy 保留旧数据格式兼容；位置/姿态入口显式选择 by_type 或 by_episode。
OUTPUT_LAYOUT = "legacy"

OBJECT_BODY = "tomato_soup_can"
OBJECT_FREEJOINT = "tomato_soup_can_free"
DATASET_OBJECT_LABEL = os.environ.get("TACTILE_OBJECT_LABEL", OBJECT_BODY)
EPISODE_SHARD_INDEX = int(os.environ.get("TACTILE_SHARD_INDEX", "0"))
EPISODE_SHARD_COUNT = max(1, int(os.environ.get("TACTILE_SHARD_COUNT", "1")))
RESUME_COMPLETED_EPISODES = os.environ.get("TACTILE_RESUME", "0") == "1"

# 固定网格直接采集；每个接触中心和 roll 组合做 3 次独立尝试。
REPEATS_PER_CONFIG = 3
# v3 首轮仅使用 identity 摆放。确认模型局部坐标中罐体轴向后，
# 再将经过验证的稳定四元数加入此表；不要直接均匀随机 SO(3)。
OBJECT_POSES = {
    "upright": [1.0, 0.0, 0.0, 0.0],
}
OBJECT_POSE_DROP_Z = {}
# 物体系抓取中心基准，单位 m；各采集配置再叠加三维偏移。
GRASP_CENTER_OBJECT = np.array([0.0, 0.0, 0.0], dtype=float)
# 对 XML 中约 68 mm 直径、103 mm 高的罐体，在可抓取侧面区域按
# 5 mm 网格移动“夹爪名义抓取中心”。XY 圆盘偏移覆盖不同周向区域，
# Z 偏移覆盖罐体不同高度；接近倾角先固定为基准值。
CONTACT_GRID_SPACING_M = 0.005
CONTACT_CENTER_RADIAL_LIMIT_M = 0.015
CONTACT_CENTER_AXIAL_LIMIT_M = 0.030
GRIPPER_ROLLS_DEG = [0.0, 30.0, 60.0, 90.0]
_xy_grid = np.arange(
    -CONTACT_CENTER_RADIAL_LIMIT_M,
    CONTACT_CENTER_RADIAL_LIMIT_M + CONTACT_GRID_SPACING_M / 2,
    CONTACT_GRID_SPACING_M,
)
_z_grid = np.arange(
    -CONTACT_CENTER_AXIAL_LIMIT_M,
    CONTACT_CENTER_AXIAL_LIMIT_M + CONTACT_GRID_SPACING_M / 2,
    CONTACT_GRID_SPACING_M,
)
_dense_grid_configs = [
    {
        "grasp_offset_object_m": [round(float(x), 4), round(float(y), 4), round(float(z), 4)],
        "approach_azimuth_deg": 0.0,
        "approach_tilt_deg": 0.0,
        "gripper_roll_deg": float(roll),
    }
    for z in _z_grid
    for x in _xy_grid
    for y in _xy_grid
    if x * x + y * y <= CONTACT_CENTER_RADIAL_LIMIT_M ** 2 + 1e-12
    for roll in GRIPPER_ROLLS_DEG
]
_dense_grid_configs.sort(
    key=lambda cfg: (
        sum(value * value for value in cfg["grasp_offset_object_m"]),
        abs(cfg["gripper_roll_deg"]),
    )
)
POSE_STAGES = [{"name": "dense_contact_grid", "configs": _dense_grid_configs}]
XY_RANDOM_RANGE_M = 0.005
NUM_EPISODES = sum(len(stage["configs"]) for stage in POSE_STAGES) * REPEATS_PER_CONFIG

# None = 每次重新运行脚本都会产生新的随机序列。
# 如果需要论文复现实验，可改成固定整数，例如 20260924。
BASE_SEED = None

SHOW_VIEWER = False


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

# 物体先按离散、可复现的稳定摆放设置，再自由沉降。
OBJECT_DROP_CENTER_Z = 0.840    # m
OBJECT_SETTLE_TIME = 1.20       # s


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

# 使用项目内 TaximSensor 为三个 GelSight 单元生成同步 RGB 与深度图。
SAVE_GELSIGHT_IMAGES = True
# 相邻同步 RGB 帧估计的稠密光流；由上层采集入口按需启用。
SAVE_OPTICAL_FLOW = False
# Taxim 原始帧为 480x640。统一缩到 240x320（宽 x 高）以控制全量
# 高密度数据集体积，同时保持 50 Hz 时序和足够的接触形变细节。
TACTILE_SAVE_RESOLUTION = (240, 320)
TACTILE_IMAGE_NOISE_SIGMA = 2
TAXIM_SENSOR_TYPE = "gelsight_r1.5"
TACTILE_RENDERERS = {}


# ============================================================
# 8. 基础函数
# ============================================================

def load_model(xml_path):
    """兼容 Windows 的中文 XML 路径，同时保留相对资源路径语义。"""
    xml_path = Path(xml_path).expanduser().resolve()
    try:
        return mujoco.MjModel.from_xml_path(str(xml_path))
    except ValueError:
        # 部分 Windows MuJoCo 构建无法直接打开含 Unicode 的路径。
        if (os.name != "nt" or not xml_path.is_file()
                or str(xml_path).isascii()):
            raise

    root = ET.fromstring(xml_path.read_bytes())
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    assetdir = compiler.get("assetdir", ".")
    for attr in ("meshdir", "texturedir"):
        resource_dir = Path(compiler.get(attr, assetdir))
        if not resource_dir.is_absolute():
            resource_dir = xml_path.parent / resource_dir
        compiler.set(attr, str(resource_dir.resolve()))
    compiler.attrib.pop("assetdir", None)
    for include in root.iter("include"):
        if "file" in include.attrib:
            include_path = Path(include.get("file"))
            if not include_path.is_absolute():
                include_path = xml_path.parent / include_path
            include.set("file", str(include_path.resolve()))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


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

def init_tactile_renderers(model, data):
    """注册三路 Taxim 传感器，跟踪完整番茄罐视觉网格。"""
    global TACTILE_RENDERERS
    if not SAVE_GELSIGHT_IMAGES:
        return

    TACTILE_RENDERERS.clear()
    for finger in ("finger1", "finger2", "finger3"):
        sensor = TaximSensor(
            sensor_type=TAXIM_SENSOR_TYPE,
            bg_index=0,
            resize=None,
            preprocess_bg=True,
        )
        sensor.add_geom_mujoco(
            "tomato_visual_geom",
            model=model,
            data=data,
            mesh_name="tomato_visual_mesh",
        )
        sensor.add_camera_mujoco(
            f"{finger}_gelsight_site",
            model,
            data,
        )
        sensor.set_sensor_pad_geom(f"{finger}_gel")
        TACTILE_RENDERERS[finger] = sensor


def capture_gelsight_images(model, data):
    """在同一仿真状态下连续渲染三路 RGB 和接触深度图。"""
    if not SAVE_GELSIGHT_IMAGES:
        return {"rgb": None, "depth": None}
    if len(TACTILE_RENDERERS) != 3:
        raise RuntimeError("Taxim 三路传感器尚未初始化。")

    rgb_images = {}
    depth_images = {}
    for finger in ("finger1", "finger2", "finger3"):
        rgb, depth, _ = TACTILE_RENDERERS[finger].render_taxim(
            model,
            data,
            shadow=True,
            get_depth=True,
            img_noise_sigma=TACTILE_IMAGE_NOISE_SIGMA,
            pcn_add_noise=False,
            visualize=False,
            cycle_bg=False,
        )
        rgb = np.asarray(rgb, dtype=np.uint8)
        depth = np.asarray(depth, dtype=np.float32)
        # Taxim renders 480x640; 2x2 area averaging gives 240x320 without
        # adding an OpenCV import-time dependency to the batch collector.
        h, w = rgb.shape[:2]
        if h % 2 or w % 2 or depth.shape != (h, w):
            raise ValueError(
                f"Unexpected Taxim frame sizes: rgb={rgb.shape}, depth={depth.shape}"
            )
        rgb_small = rgb.reshape(h // 2, 2, w // 2, 2, 3).mean(
            axis=(1, 3)
        ).astype(np.uint8)
        depth_small = depth.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        rgb_images[finger] = rgb_small
        depth_images[finger] = depth_small

    return {"rgb": rgb_images, "depth": depth_images}


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
        load_model(xml_path)
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
    pose_cfg,
):
    """
    随机化目标物体初始位姿。

    位置：名义 x/y 与 object_drop_center_z 组成基准中心，再叠加
        object_position_offset_world_m（世界系，m）和 x/y 随机扰动。
        object_position_world_m 可覆盖基准中心，偏移仍会叠加。

    姿态：使用预先定义的稳定摆放组，不进行无约束 SO(3) 随机。

    随机化后还会在 run_episode() 中自由沉降 OBJECT_SETTLE_TIME，
    所以真正用于标签和抓取的是沉降后的稳定姿态。
    """

    jid = name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        OBJECT_FREEJOINT,
    )

    qadr = int(
        model.jnt_qposadr[jid]
    )

    base_xyz = data.qpos[
        qadr:qadr + 3
    ].copy()

    nominal_xyz = np.array(
        [base_xyz[0], base_xyz[1],
         float(pose_cfg.get("object_drop_center_z", OBJECT_DROP_CENTER_Z))],
        dtype=float,
    )
    position_center = np.asarray(
        pose_cfg.get("object_position_world_m", nominal_xyz), dtype=float
    )
    position_offset = np.asarray(
        pose_cfg.get("object_position_offset_world_m", [0.0, 0.0, 0.0]),
        dtype=float,
    )
    for name, values in (
        ("nominal_position_m", nominal_xyz),
        ("object_position_world_m", position_center),
        ("object_position_offset_world_m", position_offset),
    ):
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} 必须是包含 3 个有限数值的 xyz（m）。")
    quat = np.array(pose_cfg["object_quat_wxyz"], dtype=float, copy=True)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("object_quat_wxyz 必须是包含 4 个有限数值的四元数。")
    quat_norm = np.linalg.norm(quat)
    if not np.isfinite(quat_norm) or quat_norm < 1e-12:
        raise ValueError("object_quat_wxyz 的范数必须大于 0。")
    quat /= quat_norm
    if not np.isfinite(XY_RANDOM_RANGE_M) or XY_RANDOM_RANGE_M < 0:
        raise ValueError("XY_RANDOM_RANGE_M 必须是非负有限数值。")

    dx = rng.uniform(
        -XY_RANDOM_RANGE_M,
        +XY_RANDOM_RANGE_M,
    )

    dy = rng.uniform(
        -XY_RANDOM_RANGE_M,
        +XY_RANDOM_RANGE_M,
    )

    initial_xyz = position_center + position_offset + np.array([dx, dy, 0.0])
    if not np.all(np.isfinite(initial_xyz)):
        raise ValueError("物体命令位置溢出或包含非有限数值。")

    data.qpos[
        qadr:qadr + 3
    ] = initial_xyz

    data.qpos[
        qadr + 3:qadr + 7
    ] = quat

    dof_adr = int(
        model.jnt_dofadr[jid]
    )

    # freejoint 6 个速度自由度全部清零。
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
        "nominal_position_m": nominal_xyz.tolist(),
        "position_center_world_m": position_center.tolist(),
        "object_position_offset_world_m": position_offset.tolist(),
        "absolute_position_requested": "object_position_world_m" in pose_cfg,
        "initial_position_m": initial_xyz.tolist(),
        "initial_quaternion_wxyz": quat.tolist(),
        "pose_group": pose_cfg["pose_group"],
        "full_3d_random_orientation": False,
    }


def build_pose_stages():
    """复制各阶段配置到每种物体摆放，保持单因素、按阶段扩展。"""
    stages = []
    for stage in POSE_STAGES:
        configs = []
        for pose_group, quat in OBJECT_POSES.items():
            for cfg in stage["configs"]:
                configs.append({
                    "stage": stage["name"],
                    "pose_group": pose_group,
                    "object_quat_wxyz": list(quat),
                    "object_drop_center_z": float(
                        OBJECT_POSE_DROP_Z.get(pose_group, OBJECT_DROP_CENTER_Z)
                    ),
                    **cfg,
                })
        stages.append({"name": stage["name"], "configs": configs})
    return stages


def perturb_tool_rotation(base_R, azimuth_deg, tilt_deg, roll_deg):
    """围绕已验证的 home 姿态施加小幅相对旋转。"""
    az = math.radians(azimuth_deg)
    tilt = math.radians(tilt_deg)
    roll = math.radians(roll_deg)
    # 围绕基准 tool +X 施加 roll，再对接近轴作方位明确的小倾斜。
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll), -math.sin(roll)],
        [0.0, math.sin(roll), math.cos(roll)],
    ])
    # 构造局部 x 轴指向倾斜后的接近方向。
    x_local = np.array([
        math.cos(tilt),
        math.sin(tilt) * math.cos(az),
        math.sin(tilt) * math.sin(az),
    ])
    ref = np.array([0.0, 0.0, 1.0])
    y_local = np.cross(ref, x_local)
    if np.linalg.norm(y_local) < 1e-8:
        y_local = np.array([0.0, 1.0, 0.0])
    y_local /= np.linalg.norm(y_local)
    z_local = np.cross(x_local, y_local)
    return base_R @ np.column_stack([x_local, y_local, z_local]) @ Rx


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
    gel_centers_object = []
    gel_rotations_object = []
    for site_name in GEL_SITES:
        site_id = name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        gel_centers_object.append(
            (object_R.T @ (data.site_xpos[site_id] - object_pos)).tolist()
        )
        gel_rotations_object.append(
            (object_R.T @ data.site_xmat[site_id].reshape(3, 3)).reshape(-1).tolist()
        )

    finger_q = get_finger_q(
        data,
        finger_qpos_adr,
    )

    tactile = capture_gelsight_images(model, data)
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
        "gel_centers_object": gel_centers_object,
        "gel_rotations_object": gel_rotations_object,
        "object_pos": object_pos.tolist(),
        "object_R": object_R.reshape(-1).tolist(),
        "images": tactile["rgb"],
        "depth_images": tactile["depth"],
    }

    return sample


def clear_episode_outputs(episode_dir):
    """仅清理当前 episode 的已知输出，先验证所有实际路径。"""
    clear_episode_files(episode_dir, RUN_OUTPUT_ROOT, OUTPUT_LAYOUT)


def save_episode(
    episode_dir,
    samples,
    metadata,
):
    paths = episode_paths(episode_dir, RUN_OUTPUT_ROOT, OUTPUT_LAYOUT)
    for name in ("signals", "metadata", "depth", "flow"):
        paths[name].parent.mkdir(parents=True, exist_ok=True)

    # 先标记为未完成，覆盖同目录旧标记；任何图像/光流写入异常都不能
    # 被后续 --resume 误判为一个完整样本。
    metadata = dict(metadata)
    metadata["storage"] = storage_metadata(episode_dir, RUN_OUTPUT_ROOT, OUTPUT_LAYOUT)
    metadata["collection_complete"] = False
    metadata["num_frames"] = len(samples)
    metadata["tactile_renderer"] = TAXIM_SENSOR_TYPE if SAVE_GELSIGHT_IMAGES else None
    metadata["tactile_images_saved"] = False
    metadata["optical_flow_requested"] = bool(SAVE_OPTICAL_FLOW)
    metadata["optical_flow_saved"] = False
    metadata.pop("optical_flow", None)
    metadata["tactile_saved_width_height"] = list(TACTILE_SAVE_RESOLUTION)
    metadata["tactile_image_noise_sigma"] = float(TACTILE_IMAGE_NOISE_SIGMA)
    metadata["tactile_depth_units"] = "dimensionless"
    metadata["tactile_depth_representation"] = "Taxim heightMap normalized by its per-render maximum"
    metadata["tactile_depth_metric"] = False
    metadata["signals_pose_units"] = "positions_m; rotation_matrices_row_major"
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if SAVE_OPTICAL_FLOW and not SAVE_GELSIGHT_IMAGES:
        raise ValueError("SAVE_OPTICAL_FLOW=True 要求 SAVE_GELSIGHT_IMAGES=True。")
    clear_episode_outputs(episode_dir)

    csv_path = paths["signals"]

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

                "gel1_obj_x",
                "gel1_obj_y",
                "gel1_obj_z",
                "gel2_obj_x",
                "gel2_obj_y",
                "gel2_obj_z",
                "gel3_obj_x",
                "gel3_obj_y",
                "gel3_obj_z",

                *[f"tool_{axis}_m" for axis in "xyz"],
                *[f"tool_R_world_{r}{c}" for r in range(3) for c in range(3)],
                *[f"object_R_world_{r}{c}" for r in range(3) for c in range(3)],
                *[
                    f"gel{finger}_R_object_{r}{c}"
                    for finger in range(1, 4) for r in range(3) for c in range(3)
                ],
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

                    *[
                        f"{x:.9f}"
                        for center in s["gel_centers_object"]
                        for x in center
                    ],
                    *[f"{x:.9f}" for x in s["tool_pos"]],
                    *[f"{x:.9f}" for x in s["tool_R"]],
                    *[f"{x:.9f}" for x in s["object_R"]],
                    *[
                        f"{x:.9f}"
                        for rotation in s["gel_rotations_object"] for x in rotation
                    ],
                ]
            )

            if SAVE_GELSIGHT_IMAGES:
                if s["images"] is None or s["depth_images"] is None:
                    raise RuntimeError(
                        "Taxim 图像或深度图没有生成。"
                    )

    if SAVE_GELSIGHT_IMAGES and samples:
        fingers = ("finger1", "finger2", "finger3")
        rgb = {
            finger: np.stack([s["images"][finger] for s in samples])
            for finger in fingers
        }
        depth = {
            finger: np.stack([s["depth_images"][finger] for s in samples])
            for finger in fingers
        }
        if OUTPUT_LAYOUT == "legacy":
            np.savez_compressed(paths["rgb_npz"], **rgb)
        else:
            save_png_sequences(paths["rgb"], rgb)
        np.savez_compressed(paths["depth"], **depth)
        metadata["tactile_images_saved"] = True
        metadata["tactile_saved_width_height"] = [
            int(rgb["finger1"].shape[2]), int(rgb["finger1"].shape[1])
        ]
        if SAVE_OPTICAL_FLOW:
            from tactile_optical_flow import save_episode_flow
            metadata["optical_flow"] = save_episode_flow(
                episode_dir,
                rgb_path=paths["rgb"],
                signals_path=paths["signals"],
                output_path=paths["flow"],
            )
            metadata["optical_flow_saved"] = True

    # 图片数组不写进 JSON。
    metadata["collection_complete"] = True

    with (
        paths["metadata"]
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


def episode_output_complete(episode_dir, metadata, pose_cfg):
    """断点续采只复用配置一致、图像及所请求光流均完整的 episode。"""
    if not metadata.get("collection_complete"):
        return False
    if metadata.get("pose_config") != pose_cfg:
        return False
    try:
        paths = episode_paths(episode_dir, RUN_OUTPUT_ROOT, OUTPUT_LAYOUT)
        expected_storage = storage_metadata(episode_dir, RUN_OUTPUT_ROOT, OUTPUT_LAYOUT)
        # 新格式必须对应相同布局和相对路径；旧文件没有 storage 字段仍可续采。
        if metadata.get("storage", expected_storage if OUTPUT_LAYOUT == "legacy" else None) != expected_storage:
            return False
        num_frames = int(metadata.get("num_frames", 0))
        if num_frames == 0:
            return not metadata.get("valid", False) and not metadata.get("optical_flow_saved", False)
        if num_frames < 0 or not paths["signals"].is_file():
            return False
        with paths["signals"].open(encoding="utf-8-sig", newline="") as f:
            if sum(1 for _ in csv.DictReader(f)) != num_frames:
                return False
        if SAVE_GELSIGHT_IMAGES:
            if not metadata.get("tactile_images_saved"):
                return False
            if metadata.get("tactile_image_noise_sigma") != float(TACTILE_IMAGE_NOISE_SIGMA):
                return False
            rgb_context = (
                np.load(paths["rgb_npz"], allow_pickle=False)
                if OUTPUT_LAYOUT == "legacy"
                else contextlib.nullcontext(load_png_sequences(paths["rgb"], num_frames))
            )
            with rgb_context as rgb, np.load(paths["depth"], allow_pickle=False) as depth:
                image_shape = None
                for finger in ("finger1", "finger2", "finger3"):
                    rgb_array, depth_array = rgb[finger], depth[finger]
                    if (rgb_array.ndim != 4 or rgb_array.shape[0] != num_frames
                            or rgb_array.shape[-1] != 3 or rgb_array.dtype != np.uint8
                            or depth_array.shape != rgb_array.shape[:3]):
                        return False
                    if image_shape is not None and rgb_array.shape != image_shape:
                        return False
                    image_shape = rgb_array.shape
        if SAVE_OPTICAL_FLOW:
            if not metadata.get("optical_flow_saved") or not paths["flow"].is_file():
                return False
            # 此模块会只读校验已存在光流的三路尺寸、时间及源文件指纹。
            from tactile_optical_flow import save_episode_flow
            save_episode_flow(
                episode_dir,
                rgb_path=paths["rgb"],
                signals_path=paths["signals"],
                output_path=paths["flow"],
            )
    except Exception:
        return False
    return True


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
    pose_cfg,
    close_sign,
    arm_act_ids,
    gripper_act_id,
    arm_qpos_adr,
    arm_dof_adr,
    finger_qpos_adr,
    gel_geom_ids,
    object_geom_ids,
):
    # BASE_SEED=None 时，每次启动脚本都会从系统熵产生新 seed，
    # 因此即使 NUM_EPISODES=1，多次运行也不会重复同一姿态。
    if BASE_SEED is None:
        episode_seed = int(
            np.random.SeedSequence()
            .generate_state(1, dtype=np.uint32)[0]
        )
    else:
        episode_seed = (
            int(BASE_SEED)
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
    reset_home(model, data)
    mujoco.mj_forward(model, data)
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
    # 保存 v2 已使用的 home 夹爪朝向和物体参考朝向。
    object_pos_reference, object_R_reference = get_body_pose(
        model, data, OBJECT_BODY
    )
    _, tool_R_reference = get_site_pose(model, data, TOOL_SITE)

    random_pose = (
        randomize_object_pose(
            model,
            data,
            rng,
            pose_cfg,
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

    print(
        "    random initial xyz [m]: "
        f"{np.round(random_pose['initial_position_m'], 5)}"
    )
    print(
        "    random initial quat [wxyz]: "
        f"{np.round(random_pose['initial_quaternion_wxyz'], 5)}"
    )
    print(
        "    settled object xyz [m]: "
        f"{np.round(object_pos_settled, 5)}"
    )

    # --------------------------------------------------------
    # C. 依据沉降后的物体姿态和当前实验配置设置夹爪姿态
    # --------------------------------------------------------
    # 兼容旧采集的“夹爪跟随物体”方式。False 时使用世界系 home 朝向，
    # 物体旋转才会改变物体与传感器的相对朝向。
    follow_object_orientation = pose_cfg.get("follow_object_orientation", True)
    if not isinstance(follow_object_orientation, (bool, np.bool_)):
        raise ValueError("follow_object_orientation 必须是布尔值。")
    base_tool_R = (
        object_R_settled @ object_R_reference.T @ tool_R_reference
        if follow_object_orientation else tool_R_reference
    )
    target_tool_R = perturb_tool_rotation(
        base_tool_R,
        pose_cfg["approach_azimuth_deg"],
        pose_cfg["approach_tilt_deg"],
        pose_cfg["gripper_roll_deg"],
    )

    approach_axis = (
        target_tool_R[:, 0].copy()
    )

    approach_axis /= np.linalg.norm(
        approach_axis
    )

    target_grasp_center = (
        object_pos_settled
        + object_R_settled @ (
            GRASP_CENTER_OBJECT
            + np.asarray(pose_cfg["grasp_offset_object_m"], dtype=float)
        )
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

    pose_metadata = {
        "episode_index": episode_idx,
        "seed": episode_seed,
        "object": DATASET_OBJECT_LABEL,
        "object_body_name": OBJECT_BODY,
        "pose_config": pose_cfg,
        "random_pose_command": random_pose,
        "settled_object_position_m": object_pos_settled.tolist(),
        "settled_object_rotation_matrix": object_R_settled.reshape(-1).tolist(),
        "follow_object_orientation": bool(follow_object_orientation),
        "target_tool_rotation_matrix": target_tool_R.reshape(-1).tolist(),
        "target_tool_rotation_in_object": (
            object_R_settled.T @ target_tool_R
        ).reshape(-1).tolist(),
        "target_grasp_center_m": target_grasp_center.tolist(),
        "grasp_offset_object_m": list(pose_cfg["grasp_offset_object_m"]),
        "rotation_matrix_layout": "row_major",
    }

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
            "metadata": dict(pose_metadata),
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
            "metadata": dict(pose_metadata),
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
            "metadata": dict(pose_metadata),
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
            "metadata": dict(pose_metadata),
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
        **pose_metadata,
        "episode_index": episode_idx,
        "seed": episode_seed,
        "object": DATASET_OBJECT_LABEL,
        "object_body_name": OBJECT_BODY,
        "pose_config": pose_cfg,
        "grasp_center_object_m": GRASP_CENTER_OBJECT.tolist(),
        "xy_perturbation_range_m": XY_RANDOM_RANGE_M,

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
    if SAVE_OPTICAL_FLOW and not SAVE_GELSIGHT_IMAGES:
        raise ValueError("SAVE_OPTICAL_FLOW=True 要求 SAVE_GELSIGHT_IMAGES=True。")
    if not np.isfinite(TACTILE_IMAGE_NOISE_SIGMA) or TACTILE_IMAGE_NOISE_SIGMA < 0:
        raise ValueError("TACTILE_IMAGE_NOISE_SIGMA 必须是非负有限数值。")
    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"\n找不到 XML:\n"
            f"{XML_PATH}\n"
            "请先修改 XML_PATH。"
        )

    RUN_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("UNDERACTUATED THREE-FINGER TACTILE DATA COLLECTION")
    print("=" * 76)
    print(f"XML: {XML_PATH}")
    print(f"Output: {RUN_OUTPUT_ROOT}")
    assigned_episode_indices = list(
        range(EPISODE_SHARD_INDEX, NUM_EPISODES, EPISODE_SHARD_COUNT)
    )
    print(
        f"Episodes: {len(assigned_episode_indices)} assigned / {NUM_EPISODES} total "
        f"(worker {EPISODE_SHARD_INDEX + 1}/{EPISODE_SHARD_COUNT})"
    )
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
        load_model(XML_PATH)
    )

    data = mujoco.MjData(
        model
    )

    # The renderers hold references to this model/data pair. Build them once
    # per worker/model, not once per episode (which repeated calibration work).
    init_tactile_renderers(model, data)

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
    pose_stages = build_pose_stages()
    pose_configs = []
    stage_end_episodes = []
    for stage in pose_stages:
        pose_configs.extend(stage["configs"])
        stage_end_episodes.append(
            len(pose_configs) * REPEATS_PER_CONFIG
        )
    stage_valid = {stage["name"]: 0 for stage in pose_stages}
    stage_attempted = {stage["name"]: 0 for stage in pose_stages}
    print(f"Pose stages: {[stage['name'] for stage in pose_stages]}")
    print(f"Repeats/configuration: {REPEATS_PER_CONFIG}")

    with viewer_context as viewer:
        for episode_idx in assigned_episode_indices:
            print(
                f"\n[EPISODE "
                f"{episode_idx:04d}]"
            )
            pose_cfg = pose_configs[
                episode_idx // REPEATS_PER_CONFIG
            ]
            print(f"  stage/config: {pose_cfg['stage']} / {pose_cfg}")

            episode_dir = (
                RUN_OUTPUT_ROOT
                / DATASET_OBJECT_LABEL
                / f"episode_{episode_idx:04d}"
            )
            paths = episode_paths(episode_dir, RUN_OUTPUT_ROOT, OUTPUT_LAYOUT)
            completed_metadata = paths["metadata"]
            if RESUME_COMPLETED_EPISODES and completed_metadata.is_file():
                try:
                    prior = json.loads(completed_metadata.read_text(encoding="utf-8"))
                except Exception:
                    prior = {}
                if episode_output_complete(episode_dir, prior, pose_cfg):
                    was_valid = bool(prior.get("valid", False))
                    prior_reason = str(prior.get("reason", "resumed_completed"))
                    prior_frames = int(prior.get("num_frames", 0))
                    stage_attempted[pose_cfg["stage"]] += 1
                    if was_valid:
                        valid_count += 1
                        stage_valid[pose_cfg["stage"]] += 1
                        prior_status = "VALID"
                    else:
                        invalid_count += 1
                        prior_status = "INVALID"
                    summary_rows.append([
                        episode_idx,
                        pose_cfg["stage"],
                        json.dumps(pose_cfg, ensure_ascii=False),
                        prior_status,
                        prior_reason,
                        prior_frames,
                    ])
                    print(f"  => RESUME SKIP: {prior_status} ({prior_reason})")
                    continue

            # 在仿真开始前清除旧完成标记，仿真/渲染中途异常也能正确续采。
            completed_metadata.parent.mkdir(parents=True, exist_ok=True)
            completed_metadata.write_text(json.dumps({
                "episode_index": episode_idx,
                "pose_config": pose_cfg,
                "collection_complete": False,
                "optical_flow_requested": bool(SAVE_OPTICAL_FLOW),
                "optical_flow_saved": False,
                "storage": storage_metadata(episode_dir, RUN_OUTPUT_ROOT, OUTPUT_LAYOUT),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            clear_episode_outputs(episode_dir)

            result = run_episode(
                model=model,
                data=data,
                viewer=viewer,
                episode_idx=episode_idx,
                pose_cfg=pose_cfg,
                close_sign=close_sign,
                arm_act_ids=arm_act_ids,
                gripper_act_id=gripper_act_id,
                arm_qpos_adr=arm_qpos_adr,
                arm_dof_adr=arm_dof_adr,
                finger_qpos_adr=finger_qpos_adr,
                gel_geom_ids=gel_geom_ids,
                object_geom_ids=object_geom_ids,
            )

            metadata = result[
                "metadata"
            ]

            metadata["collection_complete"] = False
            metadata["valid"] = bool(result["valid"])
            metadata["reason"] = result["reason"]
            metadata["optical_flow_requested"] = bool(SAVE_OPTICAL_FLOW)
            metadata["optical_flow_saved"] = False
            metadata["storage"] = storage_metadata(episode_dir, RUN_OUTPUT_ROOT, OUTPUT_LAYOUT)

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
                completed_metadata.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                metadata["collection_complete"] = True
                metadata["num_frames"] = 0
                metadata["tactile_images_saved"] = False

                with (
                    completed_metadata
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
                stage_valid[pose_cfg["stage"]] += 1
                status = "VALID"
            else:
                invalid_count += 1
                status = "INVALID"
            stage_attempted[pose_cfg["stage"]] += 1

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
                    pose_cfg["stage"],
                    json.dumps(pose_cfg, ensure_ascii=False),
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

            if (episode_idx + 1) in stage_end_episodes:
                stage_name = pose_cfg["stage"]
                stage_rate = (
                    stage_valid[stage_name]
                    / max(1, stage_attempted[stage_name])
                )
                print(
                    f"Stage {stage_name}: "
                    f"{stage_valid[stage_name]}/"
                    f"{stage_attempted[stage_name]} valid "
                    f"({stage_rate:.0%})"
                )

    summary_root = RUN_OUTPUT_ROOT / "signals_csv" if OUTPUT_LAYOUT == "by_type" else RUN_OUTPUT_ROOT
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / (
        f"collection_summary_{DATASET_OBJECT_LABEL}_"
        f"worker_{EPISODE_SHARD_INDEX:02d}.csv"
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
                "stage",
                "pose_config_json",
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
