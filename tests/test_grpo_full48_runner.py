"""Frozen scheduling, fail-closed evidence and real process ownership boundaries."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from writing_agent.catalog import fingerprint, save_json
from writing_agent.grpo import file_hashes, seal_directory
from writing_agent.grpo_full48 import load_full48_release
from writing_agent.grpo_full48_runner import (
    SETTINGS,
    admit_coverage,
    coverage,
    frozen_plan,
    preflight,
    prepare,
    schedule,
)
from writing_agent.grpo_full48_supervisor import supervise, verify_lease, writer_lease
from writing_agent.grpo_rollout import ProtocolError, RolloutGroups


def fixture_task():
    return {"id": "fixture", "visible": {"budgets": {"max_steps": 48}}}


class SchedulerTests(unittest.TestCase):
    def test_order_seeds_and_original_probe_caps(self):
        tasks = [{**fixture_task(), "id": str(i)} for i in range(48)]
        rows = schedule(tasks, SETTINGS)
        self.assertEqual([g["task"] for g in rows], [str(i) for i in range(48)] * 2)
        seeds = [s["seed"] + d for g in rows for s in g["slots"] for d in range(s["decisions"])]
        self.assertEqual(len(seeds), 384 * 48)
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertEqual((min(seeds), max(seeds)), (42, 18473))
        with self.assertRaises(ValueError):
            replace(SETTINGS, runtime_profile="probe").validate()
        with self.assertRaises(ValueError):
            schedule(
                [{**fixture_task(), "visible": {"budgets": {"max_steps": 49}}}],
                replace(SETTINGS, max_steps=2),
            )

    def test_reject_wrong_task_exhausted_schedule_and_repeat_before_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = [fixture_task(), {**fixture_task(), "id": "other"}]
            settings = replace(SETTINGS, max_steps=4)
            callback = RolloutGroups(tasks, settings, tmp, None, None, "", invocation_id="fixture")
            trainer = SimpleNamespace(state=SimpleNamespace(global_step=0))
            with self.assertRaisesRegex(ProtocolError, "schedule"):
                callback(["other"] * 4, trainer)
            trainer.state.global_step = 4
            with self.assertRaisesRegex(ProtocolError, "schedule"):
                callback(["fixture"] * 4, trainer)
            trainer.state.global_step = 0
            (Path(tmp) / "groups" / "step-000000-interrupted").mkdir(parents=True)
            with self.assertRaisesRegex(ProtocolError, "resampling"):
                callback(["fixture"] * 4, trainer)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tasks = [fixture_task()]
        self.settings = replace(SETTINGS, max_steps=2)
        manifest = {
            "plan": {
                "settings": asdict(self.settings),
                "tasks": self.tasks,
            }
        }
        self.identity = fingerprint(manifest)
        save_json(self.root / "experiment.json", {"identity": self.identity, "manifest": manifest})
        save_json(
            self.root / "invocations/one/started.json",
            {
                "identity": self.identity,
                "resumed_step": 0,
                "resumed_from": None,
            },
        )

    def group(self, step):
        expected = schedule(self.tasks, self.settings)[step]
        path = self.root / "groups" / f"step-{step:06d}-fixture"
        save_json(
            path / "started.json",
            {k: v for k, v in expected.items() if k != "slots"}
            | {
                "invocation": "one",
                "invocation_hash": fingerprint(
                    json.loads((self.root / "invocations/one/started.json").read_text())
                ),
            },
        )
        for slot in expected["slots"]:
            a = path / f"attempt-{slot['slot']:03d}"
            save_json(
                a / "started.json",
                {
                    "seed": slot["seed"],
                    "slot": slot["slot"],
                    "group": step,
                    "task": "fixture",
                    "pass_index": step,
                    "decision_seed_stride": 48,
                },
            )
            save_json(a / "result.json", {"seed": slot["seed"], "failure_class": None})
            save_json(a / "reward.json", {"status": "ok", "value": 1})
            save_json(
                a / "tokens.json",
                {
                    "prompt_ids": [1],
                    "completion_ids": [2],
                    "env_mask": [1],
                    "boundaries": [{"input_ids": [1], "completion_offset": 0, "output_ids": [2]}],
                },
            )
        save_json(
            path / "group.json", {"status": "ok", "tie_policy": "continue", "zero_variance": True}
        )
        save_json(path / "complete.json", {"status": "scored", "attempts": 4})
        return path

    def checkpoint(self, step):
        path = self.root / f"checkpoint-{step}"
        path.mkdir()
        for name in (
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            (path / name).write_text("fixture bytes, never loaded")
        save_json(path / "trainer_state.json", {"global_step": step})
        seal_directory(
            path, self.identity, "trainer", group_files=file_hashes(self.root / "groups")
        )
        return path

    def report(self):
        return coverage(self.root, self.tasks, self.settings)

    def test_scoring_is_not_an_optimizer_boundary_and_no_retry(self):
        self.group(0)
        report = self.report()
        self.assertEqual(report["groups_scored"], 1)
        self.assertEqual(report["completed_optimizer_boundaries"], 0)
        self.assertEqual(report["attempts_started"], 4)
        with self.assertRaisesRegex(ValueError, "resampling"):
            admit_coverage(report, 1, resume=True)
        self.checkpoint(1)
        report = self.report()
        self.assertFalse(report["errors"])
        self.assertEqual(admit_coverage(report, 1, resume=True), 2)
        self.group(1)
        self.checkpoint(2)
        report = self.report()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["tied_groups"], 2)
        self.assertEqual(report["missing_slots"], [])
        with self.assertRaisesRegex(ValueError, "unexhausted"):
            admit_coverage(report, 1, resume=True)

    def test_corruption_missing_slots_duplicates_and_lineage_fail_closed(self):
        group = self.group(0)
        checkpoint = self.checkpoint(1)
        (checkpoint / "optimizer.pt").write_text("corrupt")
        self.assertTrue(self.report()["errors"])
        seal_directory(
            checkpoint, self.identity, "trainer", group_files=file_hashes(self.root / "groups")
        )
        start = group / "attempt-000/started.json"
        saved = json.loads(start.read_text())
        save_json(start, {**saved, "seed": 999})
        report = self.report()
        self.assertTrue(report["errors"])
        self.assertEqual(report["attempts_started"], 4)  # Still count later slots on corruption.
        save_json(start, saved)
        started = json.loads((group / "started.json").read_text())
        save_json(group / "started.json", {**started, "invocation": "unknown"})
        self.assertTrue(self.report()["errors"])
        save_json(group / "started.json", started)
        import shutil

        shutil.copytree(group, group.with_name("step-000000-duplicate"))
        self.assertEqual(len(self.report()["duplicate_slots"]), 4)

    def test_wrong_json_shapes_report_corruption_and_count_later_slots(self):
        group = self.group(0)
        self.checkpoint(1)
        for name in ("reward.json", "result.json", "tokens.json"):
            with self.subTest(artifact=name):
                path = group / "attempt-000" / name
                original = path.read_text()
                path.write_text("[]")
                report = self.report()
                self.assertEqual(report["attempts_started"], 4)
                self.assertEqual(report["slots_with_identity"], 4)
                self.assertTrue(report["errors"])
                self.assertEqual(report["status"], "partial")
                with self.assertRaises(ValueError):
                    admit_coverage(report, 1, resume=True)
                path.write_text(original)

    def test_intermediate_boundary_resume_does_not_skip_required_pass_one_stop(self):
        report = {
            "errors": [],
            "uncommitted_groups": [],
            "completed_optimizer_boundaries": 17,
            "latest_complete_checkpoint": "checkpoint-17",
            "scheduled_groups": 96,
        }
        self.assertEqual(admit_coverage(report, 48, resume=True), 48)


class SupervisorTests(unittest.TestCase):
    def test_writer_lease_and_quiet_real_child_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with writer_lease(root) as fd:
                verify_lease(root, fd)
                with self.assertRaisesRegex(RuntimeError, "owns"):
                    with writer_lease(root):
                        pass
            command = [sys.executable, "-c", "import time; time.sleep(0.12)"]
            result = supervise(root, command, admit=lambda: {"cpu_fixture": True}, interval=0.02)
            self.assertEqual(result["status"], "completed")
            self.assertIsNone(result["elapsed_cutoff"])
            progress = next(root.glob("supervision/*/progress.jsonl")).read_text().splitlines()
            self.assertGreater(len(progress), 1)
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                supervise(root, [sys.executable, "-c", "raise SystemExit(7)"], admit=lambda: {})

    def test_child_keeps_lock_after_supervisor_closes_its_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with writer_lease(root) as fd:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import time; print('ready', flush=True); time.sleep(.2)",
                    ],
                    pass_fds=(fd,),
                    stdout=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(process.stdout.readline().strip(), "ready")
            try:
                with self.assertRaisesRegex(RuntimeError, "owns"):
                    with writer_lease(root):
                        pass
            finally:
                process.wait()
                process.stdout.close()
            with writer_lease(root):
                pass

    def test_import_and_inspect_have_no_optional_model_imports(self):
        command = [
            sys.executable,
            "-c",
            "import sys; import writing_agent.grpo_full48_runner; "
            "assert not {'torch', 'transformers', 'trl'} & set(sys.modules)",
        ]
        subprocess.run(command, check=True)


@unittest.skipUnless(os.environ.get("FULL48_RELEASE"), "Set FULL48_RELEASE to original release")
class OriginalReleaseTests(unittest.TestCase):
    def test_original_admission_prepare_and_preflight(self):
        release = Path(os.environ["FULL48_RELEASE"])
        before = copy.deepcopy(load_full48_release(release)["tasks"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            plan = frozen_plan(release, root)
            self.assertFalse(root.exists())
            self.assertEqual(plan["total_attempts"], 384)
            prepare(release, root)
            self.assertEqual(preflight(release, root)["status"], "preflight-passed")
            with self.assertRaises(FileExistsError):
                prepare(release, root)
            saved = json.loads((root / "prepared.json").read_text())
            saved["plan"]["first_stop"] = 1
            save_json(root / "prepared.json", saved)
            with self.assertRaisesRegex(ValueError, "recipe changed"):
                preflight(release, root)
        self.assertEqual(before, load_full48_release(release)["tasks"])
