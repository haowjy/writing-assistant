"""CPU-only checkpoint-31 fork admission and imported-slot regression checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from writing_agent.grpo_checkpoint31_fork import (
    GROUP_NAME,
    Checkpoint31RolloutGroups,
    _check_source_checkpoint,
    _check_source_group,
    fork_coverage,
    fork_preflight,
    fork_trainer_options,
    native_wandb_config,
    prepare_fork,
    verify_fork_manifest,
)
from writing_agent.reward import Reward

SOURCE = Path(
    "/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/"
    "full48-production-v2/trainer"
)
GROUP = SOURCE / "groups/step-000031-87b060b85fc043198c5ef1f8807f47c5"


class ForkAdmissionTests(unittest.TestCase):
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

    def test_wandb_binding_is_scalar_only_and_explicit(self):
        config = native_wandb_config(run_id="run-31")
        self.assertEqual(config["report_to"], "wandb")
        self.assertEqual(config["env"]["WANDB_RUN_ID"], "run-31")
        self.assertEqual(config["env"]["WANDB_LOG_MODEL"], "false")
        self.assertIn("task text", config["privacy"]["deny"])

    @unittest.skipUnless(SOURCE.exists(), "production evidence is not mounted")
    def test_imported_slots_do_not_call_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fork"
            prepare_fork(SOURCE / "checkpoint-31", GROUP, root)
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
                "system",
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


if __name__ == "__main__":
    unittest.main()
