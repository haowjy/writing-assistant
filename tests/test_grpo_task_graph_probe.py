"""Probe phase contracts that can be checked without a model or GPU."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from writing_agent.grpo_task_graph_probe import (
    ProbeError,
    _latest_checkpoint,
    _require_latest_checkpoint,
    _select_verdict,
    inspect,
    prepare,
)
from writing_agent.grpo_task_graph_probe_evidence import _criterion, inspect_run


class TaskGraphProbeTests(unittest.TestCase):
    def test_inspect_is_read_only_and_prepare_persists_its_pre_run_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            with (
                patch.dict(
                    os.environ,
                    {
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "CUDA_VISIBLE_DEVICES": "",
                    },
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe._prepare_record",
                    return_value={
                        "schema": 1,
                        "run_id": "run",
                        "recipe": {"execution_mode": "cpu-dry-run"},
                        "source": {"commit": "commit", "tree": "tree"},
                    },
                ),
                patch.dict(
                    sys.modules,
                    {
                        "torch": object(),
                        "writing_agent.grpo_probe": object(),
                    },
                ),
            ):
                before = inspect(root, mode="cpu-dry-run")
                self.assertFalse(root.exists())
                prepare(root, mode="cpu-dry-run")

            self.assertFalse(before["writes"])
            self.assertEqual(json.loads((root / "inspect.json").read_text()), before)

    def test_import_is_legacy_probe_and_torch_free(self):
        code = (
            "import sys; import writing_agent.grpo_task_graph_probe; "
            "assert 'torch' not in sys.modules; "
            "assert 'writing_agent.grpo_probe' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_non_latest_resume_checkpoint_is_refused_before_model_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "training"
            for step in (1, 2):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "complete.json").write_text("{}")
            self.assertEqual(_latest_checkpoint(root), (root / "checkpoint-2").resolve())
            with self.assertRaisesRegex(ProbeError, "latest complete checkpoint"):
                _require_latest_checkpoint(root, root / "checkpoint-1")

    def test_all_tie_plumbing_is_inconclusive_not_pass(self):
        criteria = {
            "criterion_1": _criterion(
                True,
                False,
                {"structural_and_admission": True, "ready_group_count": 0},
            ),
            "criterion_2": _criterion(
                True,
                False,
                {"optimizer_steps": 3, "gradients_finite": True, "adapter_changed": False},
            ),
            **{f"criterion_{number}": _criterion(True, True, {}) for number in (3, 4, 5, 6)},
        }
        measurements = {"group_count": 3, "tie_count": 3}
        self.assertEqual(_select_verdict(criteria, measurements), "inconclusive_no_signal")
        self.assertNotEqual(_select_verdict(criteria, measurements), "pass")

    def test_all_tie_with_missing_criterion_input_is_fail(self):
        criteria = {
            "criterion_1": _criterion(True, False, {"structural_and_admission": True}),
            "criterion_2": _criterion(True, True, {"optimizer_steps": 3, "gradients_finite": True}),
            "criterion_3": _criterion(False, False, {}, ["checkpoint-3/complete.json"]),
            "criterion_4": _criterion(True, True, {}),
            "criterion_5": _criterion(True, True, {}),
            "criterion_6": _criterion(True, True, {}),
        }
        self.assertEqual(_select_verdict(criteria, {"group_count": 3, "tie_count": 3}), "fail")

    def test_inspect_run_missing_evidence_writes_fail_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            (root / "prepare.json").write_text(
                json.dumps({"recipe": {"execution_mode": "cpu-dry-run"}})
            )
            (root / "training").mkdir()
            with patch.dict(
                os.environ,
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                },
            ):
                result = inspect_run(root, mode="cpu-dry-run")
            self.assertEqual(result["verdict"], "fail")
            self.assertEqual(set(result["criteria"]), {f"criterion_{i}" for i in range(1, 7)})
            self.assertTrue(all(item["passed"] is False for item in result["criteria"].values()))
            self.assertEqual(json.loads((root / "result.json").read_text())["verdict"], "fail")


if __name__ == "__main__":
    unittest.main()
