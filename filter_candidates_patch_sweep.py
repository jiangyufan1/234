# -*- coding: utf-8 -*-
r"""
filter_candidates_patch_sweep.py

用途：
    对已经生成的 Top-N G_c 做“有限尺寸 GelSight 有效面 + 闭合扫掠”几何过滤。

背景：
    center/normal 静态回放已经证明 T_object_tool 是正确的；
    但 Candidate #4 在 alpha=-4° 时，F2 whole gel box 已先接触，
    而候选目标 alpha_c=-7.5°。

    说明仅匹配 GelSight 中心点 + 法向还不够。
    24.4 x 26.3 mm 的有限尺寸触觉面，在法向有一定偏差时，
    可能由边缘先碰到曲面。

本程序完全不做：
    UR3
    MuJoCo dynamics
    mj_step
    contact force
    Taxim

只做 CAD 几何：

    对每个候选 ^O T_G：

        alpha_pre = alpha_c + 10°

        从 alpha_pre -> alpha_c
        读取 envelope.npz 中每个 alpha 的 surface_grid_tool

        将整个 GelSight active-face grid 变换到 object frame：

            p^O = R_OG p^G + t_OG

        对每个 grid point 找 CAD 最近点 q 和 outward normal n，
        计算近似 signed clearance：

            s = (p - q) · n

        对 CAD 外部：
            s > 0

        近似接触：
            s ~ 0

        穿入物体：
            s < 0

    如果在 alpha_c 之前很多度，某块 GelSight 的 active face
    已出现明显负 signed clearance，则判定为 premature edge contact。

输出：
    grasp_candidates/tomato_soup_can/patch_sweep_filter/
        patch_sweep_results.csv
        feasible_candidates.npz
        candidate_XXX_patch_sweep.png (前若干个)

说明：
    signed clearance 由 point-cloud nearest normal 近似，
    是几何筛选器，不代替后续 MuJoCo 验证。
"""

from pathlib import Path
import argparse
import csv
import math
import sys

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    raise ImportError("缺少 scipy，请 pip install scipy")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent

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
    / "patch_sweep_filter"
)


# ---------------------------------------------------------------------------
# Filter parameters
# ---------------------------------------------------------------------------

PREOPEN_EXTRA_DEG = 10.0

# 比 0 稍宽松，降低 KD-tree / point-cloud 法向噪声影响。
PENETRATION_TOL_MM = 0.50

# 如果首次明显穿入发生在 alpha_c 之前超过 1°，认为提前接触。
MAX_EARLY_CONTACT_DEG = 1.0

# 在候选目标 alpha_c，允许 active face 局部最多约 1 mm 几何穿入。
# 这是几何过滤容差，不是实际弹性压入量。
MAX_TARGET_PATCH_PENETRATION_MM = 1.0

# target 时 active-face center 仍应靠近 CAD。
MAX_TARGET_CENTER_DISTANCE_MM = 2.0

# 默认画前多少个候选的 sweep 曲线。
PLOT_TOP_N = 12


def nearest_envelope_index(alpha_all, alpha):
    return int(
        np.argmin(
            np.abs(
                alpha_all - float(alpha)
            )
        )
    )


def transform_points(T, P):
    P = np.asarray(P, dtype=np.float64)
    return (
        T[:3, :3] @ P.T
    ).T + T[:3, 3]


def signed_clearance_to_cad(
    P,
    tree,
    cad_points,
    cad_normals,
):
    """
    对每个点：
        nearest CAD q, n
        signed s=(p-q).n

    CAD normals 假设 outward。
    """
    P = np.asarray(P, dtype=np.float64)

    d, idx = tree.query(P, k=1)

    q = cad_points[idx]
    n = cad_normals[idx]

    signed = np.sum(
        (P - q) * n,
        axis=1,
    )

    return (
        signed,
        np.asarray(d),
        idx,
    )


def candidate_sweep(
    rank,
    T_object_tool,
    alpha_c,
    env,
    tree,
    cad_points,
    cad_normals,
):
    alpha_all = env[
        "alpha_deg"
    ].astype(np.float64)

    surface_grid_all = env[
        "surface_grid_tool"
    ].astype(np.float64)

    center_all = env[
        "surface_center_tool"
    ].astype(np.float64)

    alpha_pre = min(
        float(np.max(alpha_all)),
        float(alpha_c) + PREOPEN_EXTRA_DEG,
    )

    # 使用 envelope 原始 alpha 网格；按“从开到闭”排序。
    mask = (
        (alpha_all >= alpha_c - 1e-9)
        & (alpha_all <= alpha_pre + 1e-9)
    )

    idxs = np.flatnonzero(mask)

    idxs = idxs[
        np.argsort(alpha_all[idxs])[::-1]
    ]

    if len(idxs) == 0:
        raise RuntimeError(
            f"Candidate {rank}: alpha sweep empty"
        )

    records = []

    first_pen_alpha = [
        None,
        None,
        None,
    ]

    pen_tol_m = (
        PENETRATION_TOL_MM / 1000.0
    )

    for ei in idxs:
        a = float(alpha_all[ei])

        # surface_grid_tool 可能是：
        #   (3, N, 3)
        # 或 (3, Nu, Nv, 3)
        grids = np.asarray(
            surface_grid_all[ei],
            dtype=np.float64,
        )

        grids = grids.reshape(
            3,
            -1,
            3,
        )

        centers = np.asarray(
            center_all[ei],
            dtype=np.float64,
        )

        row = {
            "alpha_deg": a,
        }

        for fi in range(3):
            P_obj = transform_points(
                T_object_tool,
                grids[fi],
            )

            signed, euclid, _ = (
                signed_clearance_to_cad(
                    P_obj,
                    tree,
                    cad_points,
                    cad_normals,
                )
            )

            center_obj = transform_points(
                T_object_tool,
                centers[fi][None, :],
            )[0]

            center_d, _ = tree.query(
                center_obj,
                k=1,
            )

            min_signed = float(
                np.min(signed)
            )

            p05_signed = float(
                np.percentile(
                    signed,
                    5.0,
                )
            )

            median_signed = float(
                np.median(signed)
            )

            min_euclid = float(
                np.min(euclid)
            )

            if (
                first_pen_alpha[fi]
                is None
                and min_signed
                < -pen_tol_m
            ):
                first_pen_alpha[fi] = a

            row[
                f"F{fi+1}_min_signed_m"
            ] = min_signed

            row[
                f"F{fi+1}_p05_signed_m"
            ] = p05_signed

            row[
                f"F{fi+1}_median_signed_m"
            ] = median_signed

            row[
                f"F{fi+1}_min_euclid_m"
            ] = min_euclid

            row[
                f"F{fi+1}_center_dist_m"
            ] = float(center_d)

        records.append(row)

    # target row（最接近 alpha_c）
    target_row = min(
        records,
        key=lambda r: abs(
            r["alpha_deg"] - alpha_c
        ),
    )

    early_by_finger = []

    target_pen_by_finger = []

    center_bad_by_finger = []

    for fi in range(3):
        first_a = first_pen_alpha[fi]

        if first_a is None:
            early = False
        else:
            early = bool(
                first_a
                > alpha_c
                + MAX_EARLY_CONTACT_DEG
            )

        early_by_finger.append(
            early
        )

        target_min_signed = target_row[
            f"F{fi+1}_min_signed_m"
        ]

        target_pen = bool(
            target_min_signed
            < -(
                MAX_TARGET_PATCH_PENETRATION_MM
                / 1000.0
            )
        )

        target_pen_by_finger.append(
            target_pen
        )

        center_bad = bool(
            target_row[
                f"F{fi+1}_center_dist_m"
            ]
            > (
                MAX_TARGET_CENTER_DISTANCE_MM
                / 1000.0
            )
        )

        center_bad_by_finger.append(
            center_bad
        )

    feasible = bool(
        not any(early_by_finger)
        and not any(target_pen_by_finger)
        and not any(center_bad_by_finger)
    )

    summary = {
        "rank": int(rank),
        "alpha_c_deg": float(alpha_c),
        "alpha_pre_deg": float(alpha_pre),
        "feasible": feasible,

        "F1_first_pen_alpha_deg": first_pen_alpha[0],
        "F2_first_pen_alpha_deg": first_pen_alpha[1],
        "F3_first_pen_alpha_deg": first_pen_alpha[2],

        "F1_early_contact": early_by_finger[0],
        "F2_early_contact": early_by_finger[1],
        "F3_early_contact": early_by_finger[2],

        "F1_target_min_signed_mm": (
            target_row[
                "F1_min_signed_m"
            ] * 1000.0
        ),
        "F2_target_min_signed_mm": (
            target_row[
                "F2_min_signed_m"
            ] * 1000.0
        ),
        "F3_target_min_signed_mm": (
            target_row[
                "F3_min_signed_m"
            ] * 1000.0
        ),

        "F1_target_center_dist_mm": (
            target_row[
                "F1_center_dist_m"
            ] * 1000.0
        ),
        "F2_target_center_dist_mm": (
            target_row[
                "F2_center_dist_m"
            ] * 1000.0
        ),
        "F3_target_center_dist_mm": (
            target_row[
                "F3_center_dist_m"
            ] * 1000.0
        ),
    }

    return summary, records


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


def plot_candidate(
    rank,
    summary,
    records,
    output_path,
):
    a = np.asarray(
        [r["alpha_deg"] for r in records]
    )

    fig = plt.figure(
        figsize=(9, 6)
    )

    ax = fig.add_subplot(111)

    for fi in range(3):
        y = np.asarray(
            [
                r[
                    f"F{fi+1}_min_signed_m"
                ]
                for r in records
            ]
        ) * 1000.0

        ax.plot(
            a,
            y,
            marker=".",
            label=f"F{fi+1} min signed clearance",
        )

    ax.axhline(
        -PENETRATION_TOL_MM,
        linestyle="--",
        linewidth=1.0,
        label="premature penetration threshold",
    )

    ax.axvline(
        summary["alpha_c_deg"],
        linestyle=":",
        linewidth=1.0,
        label="alpha_c",
    )

    ax.set_xlabel("Finger alpha (deg)")
    ax.set_ylabel(
        "Minimum signed clearance of active face grid (mm)"
    )

    ax.set_title(
        f"Candidate #{rank:03d} patch sweep | "
        f"feasible={summary['feasible']}"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="检查前多少个候选",
    )

    parser.add_argument(
        "--plot",
        type=int,
        default=PLOT_TOP_N,
        help="前多少个候选保存曲线图",
    )

    args = parser.parse_args()

    for path in [
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

    cand = np.load(
        CANDIDATE_PATH,
        allow_pickle=True,
    )

    env = np.load(
        ENVELOPE_PATH,
        allow_pickle=True,
    )

    if "surface_grid_tool" not in env:
        raise RuntimeError(
            "envelope.npz 没有 surface_grid_tool，"
            "请确认使用 analyze_gripper_tactile_envelope_v2.py 生成的数据。"
        )

    cad = np.load(
        CAD_CACHE_PATH,
        allow_pickle=True,
    )

    cad_points = cad[
        "surface_points"
    ].astype(np.float64)

    cad_normals = cad[
        "surface_normals"
    ].astype(np.float64)

    tree = cKDTree(cad_points)

    n_total = len(
        cand["T_object_tool"]
    )

    n = min(
        int(args.top),
        n_total,
    )

    print("=" * 90)
    print("Finite GelSight Patch + Closing Sweep Filter")
    print("=" * 90)

    print(
        f"Candidates: {n}/{n_total}"
    )

    print(
        f"premature penetration threshold = "
        f"-{PENETRATION_TOL_MM:.2f} mm"
    )

    print(
        f"allowed early-contact window = "
        f"{MAX_EARLY_CONTACT_DEG:.2f} deg before alpha_c"
    )

    summaries = []

    all_records = {}

    for idx in range(n):
        rank = idx + 1

        T = cand[
            "T_object_tool"
        ][idx].astype(np.float64)

        alpha_c = float(
            cand["alpha_deg"][idx]
        )

        summary, records = (
            candidate_sweep(
                rank,
                T,
                alpha_c,
                env,
                tree,
                cad_points,
                cad_normals,
            )
        )

        summary[
            "original_score"
        ] = float(
            cand["score"][idx]
        )

        summaries.append(
            summary
        )

        all_records[rank] = (
            records
        )

        first_pen = [
            summary[
                "F1_first_pen_alpha_deg"
            ],
            summary[
                "F2_first_pen_alpha_deg"
            ],
            summary[
                "F3_first_pen_alpha_deg"
            ],
        ]

        print(
            f"#{rank:03d} "
            f"score={summary['original_score']:.3f} "
            f"alpha_c={alpha_c:+.1f} "
            f"feasible={summary['feasible']} "
            f"first_pen={first_pen} "
            f"target_min_signed(mm)="
            f"[{summary['F1_target_min_signed_mm']:.2f}, "
            f"{summary['F2_target_min_signed_mm']:.2f}, "
            f"{summary['F3_target_min_signed_mm']:.2f}]"
        )

    # CSV
    result_csv = (
        OUTPUT_DIR
        / "patch_sweep_results.csv"
    )

    save_csv(
        summaries,
        result_csv,
    )

    # plot
    n_plot = min(
        int(args.plot),
        n,
    )

    for rank in range(
        1,
        n_plot + 1,
    ):
        plot_candidate(
            rank,
            summaries[rank - 1],
            all_records[rank],
            OUTPUT_DIR
            / f"candidate_{rank:03d}_patch_sweep.png",
        )

    feasible_idx = [
        i
        for i, s in enumerate(summaries)
        if s["feasible"]
    ]

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"feasible = {len(feasible_idx)}/{n}"
    )

    if feasible_idx:
        print(
            "feasible ranks =",
            [
                i + 1
                for i in feasible_idx
            ],
        )

        # 保存原 candidate 数据中的可行子集
        sel = np.asarray(
            feasible_idx,
            dtype=np.int64,
        )

        feasible_npz = (
            OUTPUT_DIR
            / "feasible_candidates.npz"
        )

        payload = {}

        for key in cand.files:
            arr = cand[key]

            if (
                isinstance(arr, np.ndarray)
                and arr.ndim >= 1
                and len(arr) == n_total
            ):
                payload[key] = arr[sel]

        payload[
            "original_rank"
        ] = sel + 1

        np.savez_compressed(
            feasible_npz,
            **payload,
        )

        print(
            "saved feasible subset:",
            feasible_npz,
        )

    else:
        print(
            "Top candidates 中没有通过 finite-patch sweep 的候选。"
        )
        print(
            "这时不要继续 V3；应该把 patch/sweep 约束前移到候选生成器。"
        )

    print(
        "\nresults:",
        result_csv,
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
