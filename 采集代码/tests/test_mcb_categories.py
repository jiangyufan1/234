"""Batch orchestration tests use synthetic OBJ files and no robot simulation."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collect_mcb_categories as batch
import collect_mcb_position_pose as core
import collect_mcb_any_object as models


class CategoryBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.output = self.root / "new_batch"
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def obj(self, category, split="train", filename="representative.obj", content=None):
        path = self.dataset / split / category / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content or f"# {category}/{split}/{filename}\nv 0 0 0\n", encoding="utf-8")
        return path

    def populate(self, count):
        return [self.obj(f"category_{index:02d}", filename=f"part_{index:02d}.obj")
                for index in range(1, count + 1)]

    def args(self, *extra):
        return core.parse_args(["--all-categories", "--dataset-root", str(self.dataset),
                                "--max-configs", "2", "--output", str(self.output), *extra])

    @staticmethod
    def fake_prepare(args, obj_path, label=None, configs=None, dimensions=None,
                     category=None, category_index=None):
        plan = {"object": label, "source_obj": str(obj_path),
                "source_obj_sha256": hashlib.sha256(obj_path.read_bytes()).hexdigest(),
                "category": category, "category_index": category_index,
                "configs": copy.deepcopy(configs), "dimensions": copy.deepcopy(dimensions),
                "seed": args.seed + category_index * 100000, "output_layout": args.layout}
        return {"label": label, "obj_path": obj_path, "xml_path": obj_path.with_suffix(".xml"),
                "plan": plan, "configs": plan["configs"], "dimensions": plan["dimensions"]}

    @staticmethod
    def complete(args, prepared, output):
        planned = prepared["dimensions"]["episodes"]
        return {"planned": planned, "completed": planned, "valid": planned, "invalid": 0,
                "rgb_samples": planned, "flow_samples": planned, "incomplete": 0}

    def patches(self, prepare=None, run=None):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(core, "prepare_object", side_effect=prepare or self.fake_prepare))
        stack.enter_context(patch.object(core, "print_object_plan"))
        stack.enter_context(patch.object(batch, "_shared_fingerprints", return_value={
            "model_sources_sha256": {"robot.xml": "robot_hash"},
            "code_sources_sha256": {"base.py": "base_hash", "adapter.py": "adapter_hash"}}))
        runner = stack.enter_context(patch.object(core, "run_prepared_object", side_effect=run or self.complete))
        return stack, runner

    def summary(self, layout="by_type"):
        folder = self.output / "metadata_json" if layout == "by_type" else self.output
        return json.loads((folder / "category_batch_summary.json").read_text(encoding="utf-8"))

    def test_selection_prefers_train_then_test_and_keeps_stable_indices(self):
        train = self.obj("a", filename="a.obj")
        self.obj("a", filename="z.obj")
        self.obj("a", "test", "0.obj")
        fallback = self.obj("b", "test", "b.obj")
        self.obj("z")
        total, rows = batch.select_categories(self.dataset, 1, None)
        self.assertEqual(total, 3)
        self.assertEqual([r["category"] for r in rows], ["a", "b", "z"])
        self.assertEqual(rows[0]["source_obj"], str(train.resolve()))
        self.assertEqual(rows[1]["source_obj"], str(fallback.resolve()))
        _, tail = batch.select_categories(self.dataset, 2, 1)
        self.assertEqual(tail, rows[1:2])
        self.assertEqual(tail[0]["label"], "cat02_b")
        with patch.object(models, "DATASET_ROOT", self.dataset):
            self.assertEqual(models.select_representatives(),
                             models.select_representatives(self.dataset))

    def test_selection_bounds_and_empty_dataset(self):
        self.populate(2)
        for start, limit in ((0, None), (3, None), (1, 0), (1, -1)):
            with self.subTest(start=start, limit=limit), self.assertRaises(ValueError):
                batch.select_categories(self.dataset, start, limit)
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            batch.select_categories(empty)
        with self.assertRaises(FileNotFoundError):
            batch.select_categories(self.root / "missing")

    def test_no_run_checks_all_68_without_output_or_collection(self):
        self.populate(68)
        stack, runner = self.patches()
        with stack, patch.object(core.models, "choose_obj", side_effect=AssertionError("no picker")):
            self.assertEqual(batch.run_all_categories(self.args()), 0)
            self.assertEqual(core.prepare_object.call_count, 68)
            self.assertEqual(core.prepare_object.call_args_list[-1].kwargs["category_index"], 68)
            self.assertEqual(core.prepare_object.call_args_list[-1].kwargs["label"], "cat68_part_68")
            runner.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_run_has_no_20_object_limit_and_keeps_formats_in_shared_root(self):
        self.populate(68)
        stack, runner = self.patches()
        with stack:
            self.assertEqual(batch.run_all_categories(self.args("--run", "--repeats", "3")), 0)
        self.assertEqual(runner.call_count, 68)
        self.assertTrue(all(call.args[2] == self.output.resolve() for call in runner.call_args_list))
        summary = self.summary()
        self.assertEqual(summary["selected_category_count"], 68)
        self.assertEqual(summary["episode_counts"]["planned"], 68 * 2 * 3)
        self.assertEqual(summary["episode_counts"]["rgb_samples"], 68 * 2 * 3)
        self.assertEqual(summary["category_status_counts"]["completed"], 68)
        self.assertTrue((self.output / "signals_csv" / "category_queue.csv").is_file())
        self.assertTrue((self.output / "metadata_json" / "object_plans" / "cat68_part_68.json").is_file())
        self.assertTrue((self.output / "signals_csv" / "object_plans" / "cat68_part_68.csv").is_file())
        self.assertFalse((self.output / "cat68_part_68" / "metadata_json").exists())

    def test_prepare_failure_continues_in_check_and_run(self):
        self.populate(3)

        def prepare(args, obj_path, **kwargs):
            if kwargs["category_index"] == 2:
                raise RuntimeError("broken source geometry")
            return self.fake_prepare(args, obj_path, **kwargs)

        stack, runner = self.patches(prepare=prepare)
        with stack:
            self.assertEqual(batch.run_all_categories(self.args()), 1)
            self.assertEqual(core.prepare_object.call_count, 3)
            runner.assert_not_called()
        self.assertFalse(self.output.exists())
        stack, runner = self.patches(prepare=prepare)
        with stack:
            self.assertEqual(batch.run_all_categories(self.args("--run")), 1)
            self.assertEqual(core.prepare_object.call_count, 3)
            self.assertEqual(runner.call_count, 2)
        result = self.summary()
        self.assertEqual([row["status"] for row in result["categories"]],
                         ["completed", "error", "completed"])
        self.assertIn("broken source geometry", result["categories"][1]["message"])

    def test_exit_success_with_no_valid_samples_is_reported_as_failure(self):
        self.populate(5)

        def run(args, prepared, output):
            index = prepared["plan"]["category_index"]
            counts = self.complete(args, prepared, output)
            if index == 2:
                counts.update(valid=0, invalid=2, rgb_samples=0, flow_samples=0)
            elif index == 3:
                counts.update(completed=1, valid=1, rgb_samples=1, flow_samples=1, incomplete=1)
            elif index == 4:
                raise OSError("render failed")
            return counts

        stack, runner = self.patches(run=run)
        with stack:
            self.assertEqual(batch.run_all_categories(self.args("--run")), 1)
        self.assertEqual(runner.call_count, 5)
        result = self.summary()
        self.assertEqual([row["status"] for row in result["categories"]],
                         ["completed", "no_valid_samples", "partial", "error", "completed"])
        self.assertEqual(result["episode_counts"]["valid"], 5)
        self.assertEqual(result["category_status_counts"]["error"], 1)

    def test_keyboard_interrupt_saves_progress_and_stops(self):
        self.populate(3)

        def run(args, prepared, output):
            if prepared["plan"]["category_index"] == 2:
                raise KeyboardInterrupt
            return self.complete(args, prepared, output)

        stack, runner = self.patches(run=run)
        with stack:
            self.assertEqual(batch.run_all_categories(self.args("--run")), 130)
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(self.summary()["status"], "interrupted")
        self.assertEqual([row["status"] for row in self.summary()["categories"]],
                         ["completed", "interrupted", "pending"])
        progress = json.loads((self.output / "metadata_json" / "category_batch_progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["status"], "interrupted")

    def test_resume_matches_all_parameters_and_never_overwrites_conflicts(self):
        paths = self.populate(2)
        stack, _ = self.patches()
        with stack:
            self.assertEqual(batch.run_all_categories(self.args("--run")), 0)
            self.assertEqual(batch.run_all_categories(self.args("--run", "--resume")), 0)
            original = {str(p): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
            for extra in (("--seed", "123"), ("--max-configs", "1"),
                          ("--layout", "by_episode"), ("--limit-categories", "1")):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    batch.run_all_categories(self.args("--run", "--resume", *extra))
                self.assertEqual(original, {str(p): p.read_bytes() for p in self.output.rglob("*") if p.is_file()})
            paths[0].write_text("# changed OBJ", encoding="utf-8")
            with self.assertRaises(ValueError):
                batch.run_all_categories(self.args("--run", "--resume"))
            self.assertEqual(original, {str(p): p.read_bytes() for p in self.output.rglob("*") if p.is_file()})

    def test_unknown_or_single_object_output_is_never_reused(self):
        self.populate(1)
        self.output.mkdir()
        existing = self.output / "old_data.txt"
        existing.write_bytes(b"preserve")
        stack, runner = self.patches()
        with stack:
            for flags in (("--run",), ("--run", "--resume")):
                with self.assertRaises(ValueError):
                    batch.run_all_categories(self.args(*flags))
            runner.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [existing])
        self.assertEqual(existing.read_bytes(), b"preserve")

    def test_own_batch_with_single_object_plan_is_also_rejected(self):
        self.populate(1)
        stack, _ = self.patches()
        with stack:
            self.assertEqual(batch.run_all_categories(self.args("--run")), 0)
            single = self.output / "metadata_json" / "collection_plan.json"
            single.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "单物体"):
                batch.run_all_categories(self.args("--run", "--resume"))
            self.assertEqual(single.read_text(encoding="utf-8"), "{}")

    def test_by_episode_places_batch_metadata_and_queue_at_root(self):
        self.populate(1)
        stack, _ = self.patches()
        with stack:
            self.assertEqual(batch.run_all_categories(self.args("--run", "--layout", "by_episode")), 0)
        self.assertEqual(self.summary("by_episode")["status"], "completed")
        self.assertTrue((self.output / "category_queue.csv").is_file())
        self.assertTrue((self.output / "category_batch_plan.json").is_file())
        self.assertTrue((self.output / "object_plans" / "cat01_part_01.json").is_file())
        self.assertTrue((self.output / "object_plans" / "cat01_part_01.csv").is_file())

    def test_shared_model_and_code_changes_conflict_on_resume(self):
        self.populate(1)
        stack, _ = self.patches()
        with stack:
            self.assertEqual(batch.run_all_categories(self.args("--run")), 0)
            with patch.object(batch, "_shared_fingerprints", return_value={
                    "model_sources_sha256": {"robot.xml": "changed"},
                    "code_sources_sha256": {"base.py": "changed"}}):
                with self.assertRaises(ValueError):
                    batch.run_all_categories(self.args("--run", "--resume"))


if __name__ == "__main__":
    unittest.main()
