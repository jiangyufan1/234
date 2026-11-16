# -*- coding: utf-8 -*-
r"""
test_finger1_taxim.py

用途：
1. 加载 E:\project\p1\deepgelsight\xml 下的 UR3 + 三指 GelSight MJCF
2. 使用 finger1_gelsight_site 作为 Taxim 相机平面
3. 使用 finger1_gel 作为 MuJoCo 接触 pad
4. 自动临时加入一个球体，让它沿 finger1 GelSight 局部 Z 轴压入
5. 实时显示 MuJoCo + Taxim GelSight 图像
6. 按 P 保存当前 GelSight RGB/Depth

运行：
    conda activate mujoco
    cd /d E:\project\p1\deepgelsight
    python test_finger1_taxim.py

键盘（请先点击 Taxim 图像窗口，使其获得焦点）：
    W : 增加压入量 +0.1 mm
    S : 减少压入量 -0.1 mm
    A/D : 沿 GelSight 局部 X 方向移动小球
    Q/E : 沿 GelSight 局部 Y 方向移动小球
    V : 翻转到 GelSight 局部 Z 轴另一侧
    P : 保存当前 RGB + depth
    R : 恢复初始测试位置
    ESC : 退出
"""

from pathlib import Path
import time
import sys

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from TaximSensor import TaximSensor


PROJECT_ROOT = Path(__file__).resolve().parent
XML_DIR = PROJECT_ROOT / "xml"
OUTPUT_DIR = PROJECT_ROOT / "taxim_output" / "finger1"

# 如需强制指定 XML 文件名，可以改这里
XML_FILE = None

TEMP_XML_NAME = "_taxim_finger1_test.xml"

SENSOR_SITE = "finger1_gelsight_site"
PAD_GEOM = "finger1_gel"

TEST_BODY = "taxim_test_sphere_body"
TEST_GEOM = "taxim_test_sphere_geom"

SPHERE_RADIUS = 0.0080
PAD_HALF_THICKNESS = 0.0015
INITIAL_PENETRATION = 0.0005
PENETRATION_STEP = 0.0001
MAX_PENETRATION = 0.0025

XY_STEP = 0.0005
MAX_XY = 0.010

SENSOR_TYPE = "gelsight_r1.5"
IMG_NOISE_SIGMA = 2
USE_SHADOW = True


def find_source_xml():
    if not XML_DIR.exists():
        raise FileNotFoundError(
            f"找不到 XML 目录：{XML_DIR}\n"
            r"请确认脚本位于 E:\project\p1\deepgelsight"
        )

    if XML_FILE is not None:
        p = XML_DIR / XML_FILE
        if not p.exists():
            raise FileNotFoundError(f"指定的 XML 不存在：{p}")
        return p

    preferred = [
        "ur3_with_threefinger_gelsight.xml",
        "ur3_with_threefinger_gelsight (2).xml",
        "ur3_with_threefinger_gripper.xml",
        "ur3_with_threefinger_gripper_aligned.xml",
    ]

    for name in preferred:
        p = XML_DIR / name
        if p.exists():
            return p

    candidates = sorted(
        p for p in XML_DIR.glob("ur3_with_threefinger*.xml")
        if p.name != TEMP_XML_NAME
    )

    if not candidates:
        raise FileNotFoundError(
            f"在 {XML_DIR} 中没有找到 ur3_with_threefinger*.xml"
        )

    return candidates[0]


def make_temp_test_xml(src_xml: Path):
    text = src_xml.read_text(encoding="utf-8")

    if SENSOR_SITE not in text:
        raise RuntimeError(f"XML 中找不到 site：{SENSOR_SITE}")
    if PAD_GEOM not in text:
        raise RuntimeError(f"XML 中找不到 geom：{PAD_GEOM}")

    test_body_xml = f"""
    <!-- ===== Taxim finger1 temporary test object ===== -->
    <body name="{TEST_BODY}" mocap="true" pos="0 0 1">
      <geom name="{TEST_GEOM}"
            type="sphere"
            size="{SPHERE_RADIUS}"
            rgba="0.95 0.65 0.10 1"
            contype="4"
            conaffinity="2"
            condim="3"
            friction="0.8 0.02 0.001"
            margin="0"/>
    </body>
"""

    marker = "</worldbody>"
    idx = text.find(marker)
    if idx < 0:
        raise RuntimeError("XML 中找不到 </worldbody>，无法加入测试球。")

    text = text[:idx] + test_body_xml + "\n  " + text[idx:]
    temp_xml = src_xml.parent / TEMP_XML_NAME
    temp_xml.write_text(text, encoding="utf-8")
    return temp_xml


def require_id(model, obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"MuJoCo 模型中找不到：{name}")
    return obj_id


def reset_home(model, data):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("[OK] 已加载 keyframe: home")
    else:
        mujoco.mj_resetData(model, data)
        print("[INFO] 未找到 home keyframe，使用默认 qpos。")
    mujoco.mj_forward(model, data)


def get_contact_info(model, data, geom_a_id, geom_b_id):
    count = 0
    max_pen = 0.0
    best_pos = None
    best_normal = None

    for i in range(data.ncon):
        c = data.contact[i]

        direct = (c.geom1 == geom_a_id and c.geom2 == geom_b_id)
        reverse = (c.geom1 == geom_b_id and c.geom2 == geom_a_id)

        if not (direct or reverse):
            continue

        count += 1
        pen = max(0.0, -float(c.dist))

        if best_pos is None or pen >= max_pen:
            max_pen = pen
            best_pos = np.array(c.pos, dtype=float).copy()

            n = np.array(c.frame[:3], dtype=float).copy()
            if reverse:
                n = -n
            best_normal = n

    return count, max_pen, best_pos, best_normal


def depth_to_bgr(depth):
    if depth is None or depth.size == 0:
        return None

    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d)

    if not np.any(finite):
        vis = np.zeros(d.shape, dtype=np.uint8)
    else:
        vals = d[finite]
        lo = float(np.min(vals))
        hi = float(np.max(vals))

        if abs(hi - lo) < 1e-12:
            vis = np.zeros(d.shape, dtype=np.uint8)
        else:
            vis = np.zeros(d.shape, dtype=np.float32)
            vis[finite] = (d[finite] - lo) / (hi - lo) * 255.0
            vis = np.clip(vis, 0, 255).astype(np.uint8)

    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src_xml = find_source_xml()
    print("=" * 72)
    print("源 XML：", src_xml)
    print("Sensor site :", SENSOR_SITE)
    print("Pad geom    :", PAD_GEOM)
    print("Taxim type  :", SENSOR_TYPE)
    print("=" * 72)

    temp_xml = make_temp_test_xml(src_xml)
    print("[OK] 临时测试 XML：", temp_xml)

    model = mujoco.MjModel.from_xml_path(str(temp_xml))
    data = mujoco.MjData(model)

    reset_home(model, data)

    site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, SENSOR_SITE)
    pad_geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, PAD_GEOM)
    sphere_geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, TEST_GEOM)
    sphere_body_id = require_id(model, mujoco.mjtObj.mjOBJ_BODY, TEST_BODY)

    mocap_id = int(model.body_mocapid[sphere_body_id])
    if mocap_id < 0:
        raise RuntimeError("测试球 body 没有正确创建为 mocap body。")

    print("[INFO] 正在初始化 Taxim GelSight R1.5 ...")
    sim = TaximSensor(
        sensor_type=SENSOR_TYPE,
        bg_file=None,
        bg_index=0,
        resize=None,
        preprocess_bg=True,
    )

    sim.add_geom_mujoco(
        TEST_GEOM,
        model=model,
        data=data,
        mesh_name="",
    )

    sim.add_camera_mujoco(
        SENSOR_SITE,
        model,
        data,
    )

    sim.set_sensor_pad_geom(PAD_GEOM)

    print("[OK] Taxim 初始化完成。")
    print()
    print("操作：")
    print("  W/S : 增加/减少压入量")
    print("  A/D : GelSight local X 左右移动")
    print("  Q/E : GelSight local Y 上下移动")
    print("  V   : 翻转 local Z 接触侧")
    print("  P   : 保存当前 GelSight RGB/Depth")
    print("  R   : 重置测试球")
    print("  ESC : 退出")
    print()
    print("请点击 'Finger1 GelSight - Taxim' 图像窗口后再按键。")

    penetration = INITIAL_PENETRATION
    local_x = 0.0
    local_y = 0.0
    side_sign = -1.0

    last_rgb = None
    last_depth = None
    save_counter = 0
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            site_pos = data.site_xpos[site_id].copy()
            site_R = data.site_xmat[site_id].reshape(3, 3).copy()

            local_offset = np.array(
                [
                    local_x,
                    local_y,
                    side_sign * (
                        SPHERE_RADIUS
                        + PAD_HALF_THICKNESS
                        - penetration
                    ),
                ],
                dtype=float,
            )

            sphere_world_pos = site_pos + site_R @ local_offset

            data.mocap_pos[mocap_id] = sphere_world_pos
            data.mocap_quat[mocap_id] = np.array(
                [1.0, 0.0, 0.0, 0.0]
            )

            mujoco.mj_step(model, data)

            count, max_pen, contact_pos, contact_normal = get_contact_info(
                model,
                data,
                pad_geom_id,
                sphere_geom_id,
            )

            try:
                rgb, depth, _ = sim.render_taxim(
                    model,
                    data,
                    shadow=USE_SHADOW,
                    get_depth=True,
                    img_noise_sigma=IMG_NOISE_SIGMA,
                    pcn_add_noise=False,
                    visualize=False,
                    cycle_bg=False,
                )
                last_rgb = rgb
                last_depth = depth
            except Exception as exc:
                print("\n[Taxim ERROR]", repr(exc))
                print("请把完整报错发给我。")
                break

            rgb_bgr = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR)

            depth_bgr = depth_to_bgr(last_depth)
            if depth_bgr is not None:
                if depth_bgr.shape[:2] != rgb_bgr.shape[:2]:
                    depth_bgr = cv2.resize(
                        depth_bgr,
                        (rgb_bgr.shape[1], rgb_bgr.shape[0]),
                    )
                display = np.concatenate(
                    [rgb_bgr, depth_bgr],
                    axis=1,
                )
            else:
                display = rgb_bgr.copy()

            status1 = (
                f"contact={count}  "
                f"MuJoCo_pen={max_pen * 1000:.3f} mm  "
                f"command_pen={penetration * 1000:.2f} mm"
            )
            status2 = (
                f"local xy=({local_x * 1000:.1f}, "
                f"{local_y * 1000:.1f}) mm   "
                f"side={'-Z' if side_sign < 0 else '+Z'}"
            )

            cv2.putText(
                display,
                status1,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0) if count > 0 else (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                status2,
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Finger1 GelSight - Taxim", display)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key in (ord("w"), ord("W")):
                penetration = min(
                    MAX_PENETRATION,
                    penetration + PENETRATION_STEP,
                )
            elif key in (ord("s"), ord("S")):
                penetration = max(
                    -0.002,
                    penetration - PENETRATION_STEP,
                )
            elif key in (ord("a"), ord("A")):
                local_x = max(-MAX_XY, local_x - XY_STEP)
            elif key in (ord("d"), ord("D")):
                local_x = min(MAX_XY, local_x + XY_STEP)
            elif key in (ord("q"), ord("Q")):
                local_y = min(MAX_XY, local_y + XY_STEP)
            elif key in (ord("e"), ord("E")):
                local_y = max(-MAX_XY, local_y - XY_STEP)
            elif key in (ord("v"), ord("V")):
                side_sign *= -1.0
                print(
                    "[INFO] 已翻转接触侧：",
                    "-Z" if side_sign < 0 else "+Z",
                )
            elif key in (ord("r"), ord("R")):
                penetration = INITIAL_PENETRATION
                local_x = 0.0
                local_y = 0.0
                side_sign = -1.0
                print("[INFO] 已重置测试球位置。")
            elif key in (ord("p"), ord("P")):
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_counter += 1

                rgb_path = OUTPUT_DIR / (
                    f"finger1_{ts}_{save_counter:03d}_rgb.png"
                )
                depth_path = OUTPUT_DIR / (
                    f"finger1_{ts}_{save_counter:03d}_depth.npy"
                )

                cv2.imwrite(str(rgb_path), rgb_bgr)
                np.save(str(depth_path), last_depth)

                print()
                print("[SAVE]", rgb_path)
                print("[SAVE]", depth_path)

                if contact_pos is not None:
                    print(
                        "contact_pos(world,m) =",
                        np.array2string(
                            contact_pos,
                            precision=6,
                            suppress_small=True,
                        ),
                    )
                if contact_normal is not None:
                    print(
                        "contact_normal(world) =",
                        np.array2string(
                            contact_normal,
                            precision=6,
                            suppress_small=True,
                        ),
                    )

            now = time.time()
            if now - last_print > 0.5:
                print(
                    "\r"
                    f"contact={count:2d} | "
                    f"MuJoCo penetration={max_pen*1000:6.3f} mm | "
                    f"command={penetration*1000:5.2f} mm | "
                    f"side={'-Z' if side_sign < 0 else '+Z'}",
                    end="",
                    flush=True,
                )
                last_print = now

            viewer.sync()

    cv2.destroyAllWindows()

    print("\n\n程序结束。")
    print("保存目录：", OUTPUT_DIR)
    print("临时 XML：", temp_xml)
    print("你的原始 UR3 XML 没有被修改。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n用户终止。")
    except Exception as e:
        cv2.destroyAllWindows()
        print("\n[ERROR]", repr(e))
        print("\n请把完整报错复制给我。")
        sys.exit(1)
