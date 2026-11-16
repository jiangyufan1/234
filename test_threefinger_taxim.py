# -*- coding: utf-8 -*-
r"""
test_threefinger_taxim.py

用途：
1. 加载 E:\project\p1\deepgelsight\xml 下的 UR3 + 三指 GelSight MJCF
2. 同时注册 3 个 Taxim/GelSight 传感器：
      finger1_gelsight_site + finger1_gel
      finger2_gelsight_site + finger2_gel
      finger3_gelsight_site + finger3_gel
3. 自动临时加入 3 个测试球，分别压向 3 个 GelSight
4. 实时显示 3 路 GelSight RGB + Depth
5. 按 P 同时保存 3 张 RGB、3 份 depth，以及三点接触元数据 meta.npz

运行：
    conda activate mujoco
    cd /d E:\project\p1\deepgelsight
    python test_threefinger_taxim.py

键盘（请先点击 Taxim 图像窗口，使其获得焦点）：
    1 / 2 / 3 : 选择当前控制的手指
    W / S     : 当前手指增加 / 减少压入量（0.1 mm）
    A / D     : 当前手指沿 GelSight 局部 X 方向移动
    Q / E     : 当前手指沿 GelSight 局部 Y 方向移动
    V         : 翻转当前手指接触侧（local -Z / +Z）
    R         : 重置当前手指
    T         : 重置全部 3 个手指
    P         : 保存当前 3 路 RGB / depth / meta
    ESC       : 退出
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
OUTPUT_DIR = PROJECT_ROOT / "taxim_output" / "threefinger"

XML_FILE = None
TEMP_XML_NAME = "_taxim_threefinger_test.xml"

# 3 个传感器定义
FINGERS = [
    {
        "name": "finger1",
        "site": "finger1_gelsight_site",
        "pad": "finger1_gel",
        "test_body": "taxim_test_sphere_body_1",
        "test_geom": "taxim_test_sphere_geom_1",
    },
    {
        "name": "finger2",
        "site": "finger2_gelsight_site",
        "pad": "finger2_gel",
        "test_body": "taxim_test_sphere_body_2",
        "test_geom": "taxim_test_sphere_geom_2",
    },
    {
        "name": "finger3",
        "site": "finger3_gelsight_site",
        "pad": "finger3_gel",
        "test_body": "taxim_test_sphere_body_3",
        "test_geom": "taxim_test_sphere_geom_3",
    },
]

# 球和 pad 几何参数（单位：m）
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

    for finger in FINGERS:
        if finger["site"] not in text:
            raise RuntimeError(f"XML 中找不到 site：{finger['site']}")
        if finger["pad"] not in text:
            raise RuntimeError(f"XML 中找不到 geom：{finger['pad']}")

    insertion_blocks = []
    x_offsets = [-0.04, 0.0, 0.04]

    for i, finger in enumerate(FINGERS):
        insertion_blocks.append(
            f"""
    <!-- ===== Taxim {finger['name']} temporary test object ===== -->
    <body name="{finger['test_body']}" mocap="true" pos="{x_offsets[i]} 0 1">
      <geom name="{finger['test_geom']}"
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
        )

    marker = "</worldbody>"
    idx = text.find(marker)
    if idx < 0:
        raise RuntimeError("XML 中找不到 </worldbody>，无法加入测试球。")

    text = text[:idx] + "".join(insertion_blocks) + "\n  " + text[idx:]
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


def panel_for_one(rgb, depth, title, status_lines):
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_bgr = depth_to_bgr(depth)

    if depth_bgr is not None:
        if depth_bgr.shape[:2] != rgb_bgr.shape[:2]:
            depth_bgr = cv2.resize(
                depth_bgr,
                (rgb_bgr.shape[1], rgb_bgr.shape[0]),
            )
        img = np.concatenate([rgb_bgr, depth_bgr], axis=1)
    else:
        img = rgb_bgr.copy()

    cv2.putText(
        img,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y = 46
    for text, color in status_lines:
        cv2.putText(
            img,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 22

    return img


def make_grid(images):
    # 3 个面板：上面 2 个，下面 1 个 + 空白
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)

    padded = []
    for img in images:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        padded.append(canvas)

    blank = np.zeros((h, w, 3), dtype=np.uint8)
    row1 = np.concatenate([padded[0], padded[1]], axis=1)
    row2 = np.concatenate([padded[2], blank], axis=1)
    grid = np.concatenate([row1, row2], axis=0)

    tip = "Keys: 1/2/3 select | W/S pen | A/D X | Q/E Y | V flip | R reset current | T reset all | P save | ESC quit"
    cv2.putText(
        grid,
        tip,
        (10, grid.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return grid


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src_xml = find_source_xml()
    print("=" * 80)
    print("源 XML：", src_xml)
    print("Taxim type：", SENSOR_TYPE)
    print("=" * 80)

    temp_xml = make_temp_test_xml(src_xml)
    print("[OK] 临时测试 XML：", temp_xml)

    model = mujoco.MjModel.from_xml_path(str(temp_xml))
    data = mujoco.MjData(model)

    reset_home(model, data)

    # 准备 3 个 finger 配置
    runtime = []
    for finger in FINGERS:
        site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, finger["site"])
        pad_geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, finger["pad"])
        sphere_geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, finger["test_geom"])
        sphere_body_id = require_id(model, mujoco.mjtObj.mjOBJ_BODY, finger["test_body"])

        mocap_id = int(model.body_mocapid[sphere_body_id])
        if mocap_id < 0:
            raise RuntimeError(f"{finger['name']} 的测试球 body 不是 mocap body。")

        sim = TaximSensor(
            sensor_type=SENSOR_TYPE,
            bg_file=None,
            bg_index=0,
            resize=None,
            preprocess_bg=True,
        )

        sim.add_geom_mujoco(
            finger["test_geom"],
            model=model,
            data=data,
            mesh_name="",
        )
        sim.add_camera_mujoco(
            finger["site"],
            model,
            data,
        )
        sim.set_sensor_pad_geom(finger["pad"])

        runtime.append(
            {
                **finger,
                "site_id": site_id,
                "pad_geom_id": pad_geom_id,
                "sphere_geom_id": sphere_geom_id,
                "sphere_body_id": sphere_body_id,
                "mocap_id": mocap_id,
                "sim": sim,
                "penetration": INITIAL_PENETRATION,
                "local_x": 0.0,
                "local_y": 0.0,
                "side_sign": -1.0,
                "last_rgb": None,
                "last_depth": None,
                "last_contact_pos": None,
                "last_contact_normal": None,
                "last_contact_count": 0,
                "last_contact_pen": 0.0,
                "sensor_pos": None,
                "sensor_rot": None,
            }
        )

    print("[OK] 3 个 Taxim 传感器初始化完成。")
    print()
    print("操作：")
    print("  1 / 2 / 3 : 选择当前控制手指")
    print("  W / S     : 当前手指 增加/减少压入量")
    print("  A / D     : 当前手指 沿 local X 移动")
    print("  Q / E     : 当前手指 沿 local Y 移动")
    print("  V         : 翻转当前手指接触侧")
    print("  R         : 重置当前手指")
    print("  T         : 重置全部手指")
    print("  P         : 同时保存三路 RGB / depth / meta")
    print("  ESC       : 退出")
    print()

    active = 0
    save_counter = 0
    last_print = 0.0

    def reset_one(idx):
        runtime[idx]["penetration"] = INITIAL_PENETRATION
        runtime[idx]["local_x"] = 0.0
        runtime[idx]["local_y"] = 0.0
        runtime[idx]["side_sign"] = -1.0

    def reset_all():
        for i in range(3):
            reset_one(i)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 先根据 3 个 site 位姿放置 3 个测试球
            for item in runtime:
                site_pos = data.site_xpos[item["site_id"]].copy()
                site_R = data.site_xmat[item["site_id"]].reshape(3, 3).copy()

                item["sensor_pos"] = site_pos
                item["sensor_rot"] = site_R

                local_offset = np.array(
                    [
                        item["local_x"],
                        item["local_y"],
                        item["side_sign"] * (
                            SPHERE_RADIUS
                            + PAD_HALF_THICKNESS
                            - item["penetration"]
                        ),
                    ],
                    dtype=float,
                )

                sphere_world_pos = site_pos + site_R @ local_offset

                data.mocap_pos[item["mocap_id"]] = sphere_world_pos
                data.mocap_quat[item["mocap_id"]] = np.array([1.0, 0.0, 0.0, 0.0])

            mujoco.mj_step(model, data)

            panels = []
            all_ok = True

            for idx, item in enumerate(runtime):
                count, max_pen, contact_pos, contact_normal = get_contact_info(
                    model,
                    data,
                    item["pad_geom_id"],
                    item["sphere_geom_id"],
                )

                item["last_contact_count"] = count
                item["last_contact_pen"] = max_pen
                item["last_contact_pos"] = contact_pos
                item["last_contact_normal"] = contact_normal

                try:
                    rgb, depth, _ = item["sim"].render_taxim(
                        model,
                        data,
                        shadow=USE_SHADOW,
                        get_depth=True,
                        img_noise_sigma=IMG_NOISE_SIGMA,
                        pcn_add_noise=False,
                        visualize=False,
                        cycle_bg=False,
                    )
                    item["last_rgb"] = rgb
                    item["last_depth"] = depth
                except Exception as exc:
                    print(f"\n[Taxim ERROR] {item['name']}: {repr(exc)}")
                    all_ok = False
                    break

                title = f"{item['name'].upper()} {'[ACTIVE]' if idx == active else ''}"
                status_lines = [
                    (f"contact={count}  MuJoCo_pen={max_pen*1000:.3f} mm", (0, 255, 0) if count > 0 else (0, 0, 255)),
                    (f"cmd_pen={item['penetration']*1000:.2f} mm  side={'-Z' if item['side_sign'] < 0 else '+Z'}", (255, 255, 255)),
                    (f"local xy=({item['local_x']*1000:.1f}, {item['local_y']*1000:.1f}) mm", (255, 255, 255)),
                ]
                panels.append(panel_for_one(rgb, depth, title, status_lines))

            if not all_ok:
                break

            grid = make_grid(panels)
            cv2.imshow("ThreeFinger GelSight - Taxim", grid)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord("1"):
                active = 0
                print("[INFO] 当前控制：finger1")
            elif key == ord("2"):
                active = 1
                print("[INFO] 当前控制：finger2")
            elif key == ord("3"):
                active = 2
                print("[INFO] 当前控制：finger3")
            elif key in (ord("w"), ord("W")):
                runtime[active]["penetration"] = min(
                    MAX_PENETRATION,
                    runtime[active]["penetration"] + PENETRATION_STEP,
                )
            elif key in (ord("s"), ord("S")):
                runtime[active]["penetration"] = max(
                    -0.002,
                    runtime[active]["penetration"] - PENETRATION_STEP,
                )
            elif key in (ord("a"), ord("A")):
                runtime[active]["local_x"] = max(
                    -MAX_XY, runtime[active]["local_x"] - XY_STEP
                )
            elif key in (ord("d"), ord("D")):
                runtime[active]["local_x"] = min(
                    MAX_XY, runtime[active]["local_x"] + XY_STEP
                )
            elif key in (ord("q"), ord("Q")):
                runtime[active]["local_y"] = min(
                    MAX_XY, runtime[active]["local_y"] + XY_STEP
                )
            elif key in (ord("e"), ord("E")):
                runtime[active]["local_y"] = max(
                    -MAX_XY, runtime[active]["local_y"] - XY_STEP
                )
            elif key in (ord("v"), ord("V")):
                runtime[active]["side_sign"] *= -1.0
                print(
                    f"[INFO] {runtime[active]['name']} 接触侧切换为：",
                    "-Z" if runtime[active]["side_sign"] < 0 else "+Z",
                )
            elif key in (ord("r"), ord("R")):
                reset_one(active)
                print(f"[INFO] 已重置 {runtime[active]['name']}")
            elif key in (ord("t"), ord("T")):
                reset_all()
                print("[INFO] 已重置全部 3 个手指")
            elif key in (ord("p"), ord("P")):
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_counter += 1

                sample_dir = OUTPUT_DIR / f"sample_{ts}_{save_counter:03d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                meta = {
                    "qpos": data.qpos.copy(),
                    "qvel": data.qvel.copy(),
                    "finger_names": np.array([it["name"] for it in runtime], dtype=object),
                    "sensor_sites": np.array([it["site"] for it in runtime], dtype=object),
                    "pad_geoms": np.array([it["pad"] for it in runtime], dtype=object),
                    "sensor_pos": np.stack([it["sensor_pos"] for it in runtime], axis=0),
                    "sensor_rot": np.stack([it["sensor_rot"] for it in runtime], axis=0),
                    "contact_pos": np.stack([
                        it["last_contact_pos"] if it["last_contact_pos"] is not None else np.full(3, np.nan)
                        for it in runtime
                    ], axis=0),
                    "contact_normal": np.stack([
                        it["last_contact_normal"] if it["last_contact_normal"] is not None else np.full(3, np.nan)
                        for it in runtime
                    ], axis=0),
                    "contact_count": np.array([it["last_contact_count"] for it in runtime], dtype=np.int32),
                    "contact_penetration": np.array([it["last_contact_pen"] for it in runtime], dtype=np.float64),
                    "command_penetration": np.array([it["penetration"] for it in runtime], dtype=np.float64),
                    "local_xy": np.array([[it["local_x"], it["local_y"]] for it in runtime], dtype=np.float64),
                    "side_sign": np.array([it["side_sign"] for it in runtime], dtype=np.float64),
                }

                for i, item in enumerate(runtime, start=1):
                    rgb_bgr = cv2.cvtColor(item["last_rgb"], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(sample_dir / f"gelsight_{i}.png"), rgb_bgr)
                    np.save(str(sample_dir / f"gelsight_{i}_depth.npy"), item["last_depth"])

                np.savez(sample_dir / "meta.npz", **meta)

                print("\n[SAVE]", sample_dir)

            now = time.time()
            if now - last_print > 0.6:
                msg = " | ".join(
                    f"{item['name']}: c={item['last_contact_count']} "
                    f"pen={item['last_contact_pen']*1000:.2f}mm "
                    f"cmd={item['penetration']*1000:.2f}mm"
                    for item in runtime
                )
                print("\r" + msg, end="", flush=True)
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
