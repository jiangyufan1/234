import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# 1. 修改为你本机 XML 的实际路径
# ============================================================
XML_PATH = Path(
    r"E:\project\p1\deepgelsight\xml\ur3_with_threefinger_gelsight_table_tomato.xml"
)

# 根据当前 XML 推导出的：
# 三个 GelSight 名义几何中心相对于 tool0 的偏移
# 单位：m
GRASP_OFFSET_IN_TOOL = np.array([0.14965, 0.0, 0.0], dtype=float)

TOOL_SITE = "tool0"

GEL_SITES = [
    "finger1_gelsight_site",
    "finger2_gelsight_site",
    "finger3_gelsight_site",
]

OBJECT_BODY = "tomato_soup_can"


def get_site_id(model, name):
    """按名称取得 site id，并在名称错误时给出明确提示。"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id == -1:
        raise ValueError(f"找不到 site: {name}")
    return site_id


def get_body_id(model, name):
    """按名称取得 body id。"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        raise ValueError(f"找不到 body: {name}")
    return body_id


def reset_home(model, data):
    """优先使用 XML 中的 home keyframe。"""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("[OK] 已加载 keyframe: home")
    else:
        mujoco.mj_resetData(model, data)
        print("[WARN] XML 中没有 home keyframe，使用默认 qpos=0")

    # 更新所有正运动学量
    mujoco.mj_forward(model, data)


def calculate_grasp_center(model, data):
    """
    同时计算：
    1) 三个 GelSight site 的实际几何中心
    2) tool0 + 固定偏移得到的预测抓取中心
    """

    # ---------- tool0 ----------
    tool_id = get_site_id(model, TOOL_SITE)

    tool_pos_world = data.site_xpos[tool_id].copy()

    # site_xmat 是 9 个数，需要变成 3x3 旋转矩阵
    R_world_tool = data.site_xmat[tool_id].reshape(3, 3).copy()

    # tool 坐标系中的偏移 -> 世界坐标系
    predicted_center_world = (
        tool_pos_world + R_world_tool @ GRASP_OFFSET_IN_TOOL
    )

    # ---------- 三个 GelSight ----------
    gel_positions = []

    for name in GEL_SITES:
        sid = get_site_id(model, name)
        p = data.site_xpos[sid].copy()
        gel_positions.append(p)

    gel_positions = np.asarray(gel_positions)

    actual_center_world = gel_positions.mean(axis=0)

    return (
        tool_pos_world,
        R_world_tool,
        gel_positions,
        actual_center_world,
        predicted_center_world,
    )


def print_result(model, data):
    (
        tool_pos,
        R_tool,
        gel_pos,
        actual_center,
        predicted_center,
    ) = calculate_grasp_center(model, data)

    error = np.linalg.norm(actual_center - predicted_center)

    body_id = get_body_id(model, OBJECT_BODY)
    object_pos = data.xpos[body_id].copy()

    grasp_to_object = np.linalg.norm(actual_center - object_pos)

    print("\n" + "=" * 72)
    print("夹爪抓取中心坐标验证")
    print("=" * 72)

    print("\ntool0 世界坐标 [m]:")
    print(np.round(tool_pos, 6))

    print("\ntool0 旋转矩阵 R_world_tool:")
    print(np.round(R_tool, 6))

    for i, p in enumerate(gel_pos, start=1):
        print(f"\nFinger {i} GelSight 中心世界坐标 [m]:")
        print(np.round(p, 6))

    print("\n------------------------------------------------------------")
    print("三个 GelSight 实际几何中心 [m]:")
    print(np.round(actual_center, 6))

    print("\ntool0 + [0.14965, 0, 0] 预测抓取中心 [m]:")
    print(np.round(predicted_center, 6))

    print("\n两种计算方法误差:")
    print(f"{error:.9f} m")
    print(f"{error * 1000:.6f} mm")

    print("\n------------------------------------------------------------")
    print(f"{OBJECT_BODY} body 原点世界坐标 [m]:")
    print(np.round(object_pos, 6))

    print("\n当前抓取中心到物体 body 原点距离:")
    print(f"{grasp_to_object:.6f} m")
    print(f"{grasp_to_object * 1000:.3f} mm")

    print("\n------------------------------------------------------------")

    if error < 0.0001:       # 0.1 mm
        print("[PASS] 坐标关系正确：误差 < 0.1 mm")
    elif error < 0.001:      # 1 mm
        print("[CHECK] 基本一致，但存在 0.1~1 mm 偏差，建议进一步检查。")
    else:
        print("[FAIL] 误差较大，149.65 mm 不能直接作为最终抓取中心偏移。")

    print("=" * 72 + "\n")


def main():
    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"\n找不到 XML：\n{XML_PATH}\n"
            "请先把 XML_PATH 改成你本机文件的实际路径。"
        )

    print(f"Loading:\n{XML_PATH}")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 回到 XML 中定义的 home 状态
    reset_home(model, data)

    # 第一次输出结果
    print_result(model, data)

    print(
        "\nViewer 已打开。\n"
        "目前只用于观察模型，不执行自动抓取。\n"
        "关闭 Viewer 即可结束程序。\n"
    )

    # passive viewer：这里不主动 mj_step，避免物体下落改变测试状态
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print = time.time()

        while viewer.is_running():
            # 更新正运动学
            mujoco.mj_forward(model, data)

            # 每 2 秒重新打印一次，方便观察
            if time.time() - last_print > 2.0:
                (
                    _,
                    _,
                    _,
                    actual_center,
                    predicted_center,
                ) = calculate_grasp_center(model, data)

                err_mm = (
                    np.linalg.norm(actual_center - predicted_center) * 1000
                )

                print(
                    "实际三点中心:",
                    np.round(actual_center, 5),
                    "  预测中心:",
                    np.round(predicted_center, 5),
                    f"  error={err_mm:.4f} mm",
                )

                last_print = time.time()

            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
