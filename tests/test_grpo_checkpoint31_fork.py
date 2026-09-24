"""CPU-only checkpoint-31 fork admission and imported-slot regression checks."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from writing_agent.agent import SYSTEM_PROMPT
from writing_agent.catalog import fingerprint, save_json
from writing_agent.grpo import canonical_json_value
from writing_agent.grpo_checkpoint31_fork import (
    GROUP_NAME,
    Checkpoint31RolloutGroups,
    _check_source_checkpoint,
    _check_source_group,
    apply_native_wandb_binding,
    fork_coverage,
    fork_preflight,
    fork_trainer_options,
    native_wandb_config,
    prepare_fork,
    verify_fork_manifest,
)
from writing_agent.grpo_rollout import GroupPending, NativeRolloutBackend
from writing_agent.reward import Reward

SOURCE = Path(
    "/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/"
    "full48-production-v2/trainer"
)
GROUP = SOURCE / "groups/step-000031-87b060b85fc043198c5ef1f8807f47c5"


class ForkAdmissionTests(unittest.TestCase):
    def test_model_config_integer_keys_canonicalize(self):
        source = {"id2label": {"0": "LABEL_0", "1": "LABEL_1"}}
        current = {"id2label": {0: "LABEL_0", 1: "LABEL_1"}}
        self.assertEqual(canonical_json_value(source), canonical_json_value(current))

    @unittest.skipUnless(SOURCE.exists(), "production evidence is not mounted")
    def test_manifest_pins_source_and_preserves_bytes(self):
        before_checkpoint = _check_source_checkpoint(SOURCE / "checkpoint-31")
        before_group = _check_source_group(GROUP)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fork"
            prepare_fork(SOURCE / "checkpoint-31", GROUP, output, wandb_run_id="h6dmlw8f")
            manifest = verify_fork_manifest(output)
            self.assertEqual(manifest["fork_id"], "checkpoint31-dapo-native-fork-v1")
            self.assertEqual(_check_source_checkpoint(SOURCE / "checkpoint-31"), before_checkpoint)
            self.assertEqual(_check_source_group(GROUP), before_group)
            imported = output / "trainer/groups" / GROUP_NAME
            self.assertEqual(
                (imported / "attempt-000/tokens.json").read_bytes(),
                (GROUP / "attempt-000/tokens.json").read_bytes(),
            )
            self.assertEqual(
                (imported / "attempt-002/source/tokens.json").read_bytes(),
                (GROUP / "attempt-002/tokens.json").read_bytes(),
            )
            report = fork_coverage(output)
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["original_prefix"]["through_checkpoint"], 31)
            self.assertEqual(fork_preflight(output)["status"], "preflight-passed")
            options = fork_trainer_options(output)
            self.assertEqual(options["stop_after_steps"], 48)
            self.assertEqual(options["resume_checkpoint_identity"], before_checkpoint["identity"])
            self.assertEqual(options["fork_manifest_identity"], manifest["identity"])
            self.assertEqual(options["report_to"], "wandb")
            self.assertEqual(options["wandb_run_name"], "h6dmlw8f")
            with self.assertRaises(ValueError):
                fork_preflight(output, source_checkpoint=Path(tmp) / "wrong-checkpoint")
            (output / "trainer" / "groups" / "partial-step").mkdir()
            with self.assertRaises(ValueError):
                fork_preflight(output)

    def test_launcher_ownership_records_are_unique_without_gpu(self):
        path = Path(__file__).parents[1] / "scripts" / "run_grpo_checkpoint31_fork.py"
        spec = importlib.util.spec_from_file_location("fork_launcher", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        first = module._ownership_path(Path("/tmp/fork"))
        second = module._ownership_path(Path("/tmp/fork"))
        self.assertNotEqual(first, second)

    def test_wandb_binding_is_scalar_only_and_explicit(self):
        config = native_wandb_config(run_id="run-31")
        self.assertEqual(config["report_to"], "wandb")
        self.assertEqual(config["env"]["WANDB_RUN_ID"], "run-31")
        self.assertEqual(config["env"]["WANDB_LOG_MODEL"], "false")
        self.assertIn("task text", config["privacy"]["deny"])

    def test_wandb_existing_wrong_project_is_rejected(self):
        config = native_wandb_config(run_id="run-31")
        fake = SimpleNamespace(
            run=SimpleNamespace(id="run-31", entity="immpanda", project="wrong-project")
        )
        with patch.dict("sys.modules", {"wandb": fake}):
            with self.assertRaises(ValueError):
                apply_native_wandb_binding(config)

    @unittest.skipUnless(SOURCE.exists(), "production evidence is not mounted")
    def test_imported_slots_do_not_call_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fork"
            prepare_fork(SOURCE / "checkpoint-31", GROUP, root, wandb_run_id="h6dmlw8f")
            trainer_output = root / "trainer"
            task = {
                "id": "wave1-train-032",
                "visible": {
                    "brief": "fixture",
                    "tools": [],
                    "followups": [],
                    "initial_files": {},
                    "budgets": {"max_total_bytes": 1_000_000},
                },
            }
            from writing_agent.grpo_full48_runner import SETTINGS

            invocation = trainer_output / "invocations" / "fixture" / "started.json"
            save_json(invocation, {"identity": fingerprint({"fixture": True})})

            calls = []

            def backend_factory(*args):
                calls.append(args)
                raise AssertionError("imported slots must not construct a backend")

            def reward(_task, result):
                values = {5994: 0.1, 6042: 0.2, 6138: 0.3}
                return Reward("ok", value=values.get(result.get("seed"), 0.8))

            groups = Checkpoint31RolloutGroups(
                [task],
                SETTINGS,
                trainer_output,
                reward,
                backend_factory,
                SYSTEM_PROMPT,
                invocation_id="fixture",
                fork_output=trainer_output,
                task=task,
            )
            original = groups._continue_slot002
            groups._continue_slot002 = lambda _trainer: {
                "tokens": copy.deepcopy(
                    json.loads(
                        (GROUP / "attempt-002/tokens.json").read_text()
                    )
                ),
                "reward": Reward("ok", value=0.8),
            }
            try:
                result = groups(
                    [task["id"]] * 4,
                    SimpleNamespace(state=SimpleNamespace(global_step=31)),
                )
            finally:
                groups._continue_slot002 = original
            self.assertEqual(calls, [])
            self.assertEqual(result["rollout_rewards"], [0.1, 0.2, 0.8, 0.3])
            self.assertEqual(
                json.loads(
                    (trainer_output / "groups" / GROUP_NAME / "group.json").read_text()
                )["status"],
                "ok",
            )
            self.assertTrue((trainer_output / "groups" / GROUP_NAME / "complete.json").exists())

    @unittest.skipUnless(SOURCE.exists(), "production evidence is not mounted")
    def test_infrastructure_continuation_stays_pending(self):
        from writing_agent.grpo_full48_runner import SETTINGS

        source_attempt = GROUP / "attempt-002"
        experiment = json.loads((SOURCE / "experiment.json").read_text())
        task = next(
            item
            for item in experiment["manifest"]["plan"]["tasks"]
            if item["id"] == "wave1-train-032"
        )
        before = (source_attempt / "result.json").read_bytes()

        class FailingBackend(NativeRolloutBackend):
            def __init__(self, *_args):
                self.failure = None
                self.prompt_ids = []
                self.completion_ids = []
                self.env_mask = []
                self.boundaries = []
                self.calls = 0

            def restore_answer_followup(self, _history, evidence):
                self.prompt_ids = list(evidence["prompt_ids"])
                self.completion_ids = list(evidence["completion_ids"])
                self.env_mask = list(evidence["env_mask"])
                self.boundaries = copy.deepcopy(evidence["boundaries"])
                self.calls = len(self.boundaries)

            def complete(self, _messages, _tools, *, emit=lambda _event: None):
                self.failure = "infrastructure"
                raise OSError("host unavailable")

            def evidence(self):
                return json.loads((source_attempt / "tokens.json").read_text())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fork"
            prepare_fork(SOURCE / "checkpoint-31", GROUP, root, wandb_run_id="h6dmlw8f")
            trainer_output = root / "trainer"
            groups = Checkpoint31RolloutGroups(
                [task],
                SETTINGS,
                trainer_output,
                lambda _task, _result: Reward("ok", value=0.8),
                lambda *_args: FailingBackend(),
                SYSTEM_PROMPT,
                invocation_id="fixture",
                fork_output=trainer_output,
                task=task,
            )
            with self.assertRaises(GroupPending):
                groups._continue_slot002(SimpleNamespace(model=None, processing_class=None))
            self.assertEqual(before, (source_attempt / "result.json").read_bytes())
            self.assertFalse(
                (trainer_output / "groups" / GROUP_NAME / "complete.json").exists()
            )

    @unittest.skipUnless(SOURCE.exists(), "production evidence is not mounted")
    def test_eligible_continuation_scores_complete_trace(self):
        from writing_agent.grpo_full48_runner import SETTINGS

        base = SOURCE / "experiment.json"
        task = next(
            item
            for item in json.loads(base.read_text())["manifest"]["plan"]["tasks"]
            if item["id"] == "wave1-train-032"
        )
        task = copy.deepcopy(task)
        task["visible"]["followups"] = task["visible"]["followups"][:1]
        source_attempt = GROUP / "attempt-002"
        seen = []

        class EligibleBackend(NativeRolloutBackend):
            def __init__(self, *_args):
                self.failure = None
                self.prompt_ids = []
                self.completion_ids = []
                self.env_mask = []
                self.boundaries = []
                self.calls = 0

            def restore_answer_followup(self, _history, evidence):
                self.prompt_ids = list(evidence["prompt_ids"])
                self.completion_ids = list(evidence["completion_ids"])
                self.env_mask = list(evidence["env_mask"])
                self.boundaries = copy.deepcopy(evidence["boundaries"])
                self.calls = len(self.boundaries)

            def complete(self, _messages, _tools, *, emit=lambda _event: None):
                emit({"type": "model_output", "output_ids": [1], "input_ids": [1]})
                return SimpleNamespace(
                    message={"role": "assistant", "content": "eligible continuation"},
                    usage={},
                )

            def evidence(self):
                return json.loads((source_attempt / "tokens.json").read_text())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fork"
            prepare_fork(SOURCE / "checkpoint-31", GROUP, root, wandb_run_id="h6dmlw8f")
            trainer_output = root / "trainer"

            def reward(_task, result):
                seen.append(result["trace"])
                return Reward("ok", value=0.7857142857142857)

            groups = Checkpoint31RolloutGroups(
                [task],
                SETTINGS,
                trainer_output,
                reward,
                lambda *_args: EligibleBackend(),
                SYSTEM_PROMPT,
                invocation_id="fixture",
                fork_output=trainer_output,
                task=task,
            )
            result = groups._continue_slot002(SimpleNamespace(model=None, processing_class=None))
            self.assertEqual(result["reward"].value, 0.7857142857142857)
            self.assertEqual(len(seen), 1)
            source_trace = (source_attempt / "trace.jsonl").read_text().splitlines()
            self.assertGreater(len(seen[0]), len(source_trace))


if __name__ == "__main__":
    unittest.main()
