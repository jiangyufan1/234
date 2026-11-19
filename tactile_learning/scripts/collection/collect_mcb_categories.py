#!/usr/bin/env python
"""Collect a representative raw OBJ from every MCB train/test category.

Selection preserves the previous 68-category check: categories are
sorted by name, the first train OBJ is preferred, and test is the fallback.
No pre-generated XML list or object-count limit participates in selection.
The public launcher is collect_mcb_position_pose.py --all-categories.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import collect_mcb_position_pose as core
from collect_mcb_any_object import select_representatives


COUNTERS = ("planned", "completed", "valid", "invalid", "rgb_samples",
            "flow_samples", "incomplete")


def select_categories(dataset_root, start=1, limit=None):
    """Return total category count and a slice retaining its original 1-based IDs."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"MCB 数据集目录不存在：{dataset_root}")
    category_names = sorted({
        folder.name
        for split in ("train", "test")
        for folder in (dataset_root / split).iterdir()
        if folder.is_dir()
    })
    count = len(category_names)
    if not count:
        raise ValueError("train/test 中没有物体类别")
    if start < 1 or start > count:
        raise ValueError(f"--start-category 必须在 1 到 {count} 之间")
    if limit is not None and limit < 1:
        raise ValueError("--limit-categories 必须大于 0")
    selected = list(enumerate(category_names, 1))[start - 1:]
    if limit is not None:
        selected = selected[:limit]

    rows = []
    for category_index, category in selected:
        candidates = []
        for split in ("train", "test"):
            folder = dataset_root / split / category
            if folder.is_dir():
                candidates.extend(sorted(folder.glob("*.obj")))
        unique = list(dict.fromkeys(path.resolve() for path in candidates))
        for instance_index, obj_path in enumerate(unique[:3], 1):
            rows.append({
                "category_index": category_index,
                "instance_index": instance_index,
                "category": category,
                "label": core.models.slug(
                    f"cat{category_index:02d}_instance{instance_index:02d}_{obj_path.stem}"
                ),
                "source_obj": str(obj_path),
            })
    return count, rows


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _shared_fingerprints():
    """Fingerprint the template and robot meshes, excluding its obsolete MCB OBJ."""
    here = Path(__file__).resolve().parent
    template = core.models.TEMPLATE_XML
    tree = ET.parse(template)
    root = tree.getroot()

    compiler = root.find("compiler")
    mesh_dir = Path(
        compiler.get("meshdir", ".") if compiler is not None else "."
    )
    if not mesh_dir.is_absolute():
        mesh_dir = (template.parent / mesh_dir).resolve()

    visual = root.find(".//geom[@name='mcb_part_visual']")
    obsolete_object_mesh = visual.get("mesh") if visual is not None else None

    model_sources = {
        str(template.resolve()): _sha256(template)
    }

    for mesh in root.findall(".//mesh"):
        if mesh.get("name") == obsolete_object_mesh:
            continue

        filename = mesh.get("file")
        if not filename:
            continue

        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = mesh_dir / mesh_path

        mesh_path = mesh_path.resolve()

        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"Robot mesh missing: {mesh_path}"
            )

        model_sources[str(mesh_path)] = _sha256(mesh_path)

    sources = [
        Path(core.__file__),
        Path(core.models.__file__),
        Path(core.adapter.__file__),
        core.adapter.DEFAULT_COLLECTOR,
        Path(__file__),
        here / "collection_storage.py",
        here / "tactile_optical_flow.py",
        here / "sensor_collision_guard.py",
    ]

    return {
        "model_sources_sha256": model_sources,
        "code_sources_sha256": {
            str(path.resolve()): _sha256(path)
            for path in sorted(set(sources))
        },
    }


def make_batch_plan(args, rows, total_categories, configs, dimensions):
    objects = []
    for row in rows:
        item = dict(row)
        try:
            item["source_obj_sha256"] = _sha256(row["source_obj"])
        except OSError as exc:
            # A single unreadable source must not prevent the remaining
            # categories from receiving a check/collection result.
            item["source_obj_sha256"] = None
            item["source_error"] = f"{type(exc).__name__}: {exc}"
        objects.append(item)
    return {"schema_version": 1, "mode": "all_categories",
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "total_available_categories": total_categories,
            "selected_category_count": len({item["category_index"] for item in objects}),
            "selected_object_instance_count": len(objects), "objects": objects,
            "configs": configs, "dimensions": dimensions,
            "episodes_per_category": dimensions["episodes"],
            "total_planned_episodes": len(objects) * dimensions["episodes"],
            "seed": args.seed, "category_seed_rule": "seed + category_index * 100000",
            "free_floating": args.free_floating,
            "sensor_safety": {"minimum_clearance_m": args.sensor_clearance_mm / 1000,
                              "scope": "different_fingers_gel_and_housing",
                              "rejection": "discard_entire_episode"},
            "object_center_world_m": args.center_m,
            "output_layout": args.layout,
            "save_rgb": True, "save_depth": True, "save_flow": True,
            "formats": {"rgb": "png", "depth": "npz", "flow": "npz",
                        "signals": "csv", "metadata": "json"},
            "flow_backend": "opencv_farneback_adjacent_rgb",
            "image_noise_sigma": 0.0, **_shared_fingerprints()}


def _type_roots(output, layout):
    output = Path(output)
    return ((output / "metadata_json", output / "signals_csv")
            if layout == "by_type" else (output, output))


def _atomic_text(path, value, encoding="utf-8"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding=encoding)
    temporary.replace(path)


def _write_json(path, value):
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def save_batch_plan(output, plan, resume=False):
    """Allow only a new empty batch or an exactly matching explicit resume."""
    output = Path(output).resolve()
    meta_root, _ = _type_roots(output, plan["output_layout"])
    path = meta_root / "category_batch_plan.json"
    if output.exists() and not output.is_dir():
        raise ValueError(f"输出位置不是目录：{output}")
    if output.exists() and any(output.iterdir()):
        if not resume or not path.is_file():
            raise ValueError(f"输出目录已有文件；请使用新的目录，或 --resume 指定原批次：{output}")
        if ((output / "collection_plan.json").exists()
                or (output / "metadata_json" / "collection_plan.json").exists()):
            raise ValueError("批次目录包含单物体采集计划，拒绝混用已有数据")
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != plan:
            raise ValueError("续采类别、OBJ、采样配置或采集代码/模型发生变化，请使用新输出目录")
        return
    if resume:
        raise ValueError("--resume 需要已有 category_batch_plan.json 的原批次目录")
    _write_json(path, plan)


def _object_data_exists(output, layout, label):
    output = Path(output)
    paths = ([output / kind / label for kind in
              ("rgb_png", "depth_npz", "flow_npz", "signals_csv", "metadata_json")]
             if layout == "by_type" else [output / label])
    return any(path.exists() for path in paths)


def save_object_plan(args, prepared, output):
    """Store per-category plans without invoking the single-object root guard."""
    meta_root, signal_root = _type_roots(output, args.layout)
    label, plan = prepared["label"], prepared["plan"]
    path = meta_root / "object_plans" / f"{label}.json"
    csv_path = signal_root / "object_plans" / f"{label}.csv"
    if path.exists():
        if not args.resume or json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError(f"物体计划与原批次不一致：{label}")
    else:
        if _object_data_exists(output, args.layout, label):
            raise ValueError(f"物体数据缺少原采集计划，拒绝混入：{label}")
        _write_json(path, plan)
    # CSV is a readable derivative of the strictly checked JSON plan.
    if not csv_path.exists():
        core.write_plan_csv(csv_path, plan)


def _empty_counts(planned):
    return {key: planned if key in ("planned", "incomplete") else 0 for key in COUNTERS}


def _result_status(counts):
    if (counts["incomplete"] or counts["completed"] != counts["planned"]
            or counts["rgb_samples"] < counts["valid"]
            or counts["flow_samples"] < counts["valid"]):
        return "partial"
    if not counts["valid"]:
        return "no_valid_samples"
    return "completed"


def _refresh_saved_counts(args, record, output):
    """Include episodes saved before an execution error or interruption."""
    try:
        counts = core.summarize_object(output, record["label"], args.layout, record["planned"])
        record.update({key: int(counts[key]) for key in COUNTERS})
    except Exception:
        # Preserve the original preparation/execution failure if diagnostics
        # also fail (for example because the output volume disconnected).
        pass


def _batch_summary(records, total_available, *, check_only=False, interrupted=False):
    statuses = {name: sum(row["status"] == name for row in records)
                for name in ("pending", "preparing", "collecting", "completed",
                             "no_valid_samples", "partial", "error", "interrupted",
                             "checked")}
    counts = {key: sum(int(row.get(key, 0)) for row in records) for key in COUNTERS}
    if interrupted:
        status = "interrupted"
    elif check_only:
        status = "checked" if statuses["checked"] == len(records) else "error"
    elif statuses["completed"] == len(records):
        status = "completed"
    elif statuses["no_valid_samples"] == len(records):
        status = "no_valid_samples"
    else:
        status = "partial" if counts["completed"] else "error"
    return {"status": status, "check_only": check_only,
            "total_available_categories": total_available,
            "selected_category_count": len(records), "category_status_counts": statuses,
            "episode_counts": counts, "categories": records}


def _write_progress(output, layout, records, total_available, *, interrupted=False, final=False):
    meta_root, signal_root = _type_roots(output, layout)
    summary = _batch_summary(records, total_available, interrupted=interrupted)
    _write_json(meta_root / "category_batch_progress.json", summary)
    if final:
        _write_json(meta_root / "category_batch_summary.json", summary)
    stream = io.StringIO(newline="")
    fields = ["category_index", "instance_index", "category", "label",
              "source_obj", "status", *COUNTERS, "message"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    _atomic_text(signal_root / "category_queue.csv", stream.getvalue(), "utf-8-sig")
    return summary


def run_all_categories(args):
    """Check or collect the raw dataset's whole category inventory, sequentially."""
    total, rows = select_categories(args.dataset_root, args.start_category, args.limit_categories)
    # Formal protocol: free object, fixed object-local grasp centre and fixed gripper orientation.
    args.free_floating = True
    args.contact_mm = [[0.0, 0.0, 0.0]]
    args.follow_object_orientation = False
    configs, dimensions = core.make_configs(args)
    output = (args.output or core.DEFAULT_OUTPUT /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f_68categories")).resolve()
    planned = dimensions["episodes"]
    print(f"[BATCH] 原始模型库 {total} 类；本次 {len(rows)} 类 × {planned} = "
          f"{len(rows) * planned} episodes；每类选一个代表 OBJ。", flush=True)
    print(f"[OUTPUT] {output}", flush=True)
    records = [{**row, "status": "pending", "message": "", **_empty_counts(planned)}
               for row in rows]
    planned_objects = {}
    if args.run:
        plan = make_batch_plan(args, rows, total, configs, dimensions)
        planned_objects = {item["label"]: item for item in plan["objects"]}
        save_batch_plan(output, plan, args.resume)
        _write_progress(output, args.layout, records, total)
    for record in records:
        print(f"\n[CATEGORY {record['category_index']}/{total}] "
              f"{record['category']} ({record['label']})", flush=True)
        try:
            record["status"] = "preparing"
            prepared = core.prepare_object(
                args, Path(record["source_obj"]), label=record["label"],
                configs=configs, dimensions=dimensions,
                category=record["category"], category_index=record["category_index"])
            core.print_object_plan(args, prepared, output)
            if not args.run:
                record["status"] = "checked"
                continue
            source_sha = planned_objects[record["label"]]["source_obj_sha256"]
            if source_sha != prepared["plan"]["source_obj_sha256"]:
                raise ValueError("OBJ 在批次计划建立后发生变化，请使用新的输出目录")
            save_object_plan(args, prepared, output)
            record["status"] = "collecting"
            _write_progress(output, args.layout, records, total)
            result = core.run_prepared_object(args, prepared, output)
            record.update({key: int(result[key]) for key in COUNTERS})
            record["status"] = _result_status(record)
        except KeyboardInterrupt:
            if args.run:
                _refresh_saved_counts(args, record, output)
            record["status"] = "interrupted"
            record["message"] = "KeyboardInterrupt"
            if args.run:
                _write_progress(output, args.layout, records, total, interrupted=True, final=True)
            print("[INTERRUPTED] 批次已停止；同样参数加 --resume 和原 --output 可续采。", flush=True)
            return 130
        except Exception as exc:
            if args.run:
                _refresh_saved_counts(args, record, output)
            record["status"] = "error"
            record["message"] = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {record['category']}: {record['message']}", flush=True)
        finally:
            if args.run and record["status"] != "interrupted":
                _write_progress(output, args.layout, records, total)
        print(f"[RESULT] {record['category']}: {record['status']}; "
              f"complete={record['completed']}/{planned}; valid={record['valid']}; "
              f"RGB={record['rgb_samples']}; flow={record['flow_samples']}", flush=True)
    if args.run:
        summary = _write_progress(output, args.layout, records, total, final=True)
    else:
        summary = _batch_summary(records, total, check_only=True)
    print(f"\n[BATCH RESULT] {json.dumps({key: summary[key] for key in ('status', 'selected_category_count', 'category_status_counts', 'episode_counts')}, ensure_ascii=False)}",
          flush=True)
    if not args.run:
        print("[CHECK ONLY] 已逐类编译检查；未运行物理采集，未写入数据目录。加 --run 开始采集。", flush=True)
    return 0 if summary["status"] in ("completed", "checked") else 1


if __name__ == "__main__":
    raise SystemExit(core.main(["--all-categories", *__import__("sys").argv[1:]]))
