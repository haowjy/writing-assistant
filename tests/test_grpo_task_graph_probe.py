"""Probe phase contracts that can be checked without a model or GPU."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.task_graph_rollout_fixtures import build_rollout_fixture, run_slice
from writing_agent.grpo_gpu import DISPLAY_POLICY
from writing_agent.grpo_task_graph_probe import (
    N3_POLICY,
    ProbeError,
    _latest_checkpoint,
    _require_latest_checkpoint,
    _select_verdict,
    inspect,
    prepare,
)
from writing_agent.grpo_task_graph_probe_evidence import (
    _criterion,
    _criterion_1,
    inspect_run,
)
from writing_agent.grpo_task_graph_probe_evidence import (
    _select_verdict as _select_evidence_verdict,
)
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_tool_outcomes import read_member_tool_outcomes


class TaskGraphProbeTests(unittest.TestCase):
    def test_desktop_probe_policy_extends_the_shared_gpu_policy(self):
        self.assertEqual(N3_POLICY, {**DISPLAY_POLICY, "mode": "desktop"})
        self.assertIs(N3_POLICY["names"], DISPLAY_POLICY["names"])

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
                {
                    "structural_and_admission": True,
                    "ready_group_count": 0,
                    "tool_outcomes_complete": True,
                    "protocol_shaped_tool_rejection_count": 0,
                },
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
        self.assertEqual(
            _select_evidence_verdict(criteria, {"group_count": 3, "tie_count": 3}), "fail"
        )

    def test_all_tie_with_protocol_rejection_is_fail_not_inconclusive(self):
        criteria = {
            "criterion_1": _criterion(
                True,
                False,
                {
                    "structural_and_admission": True,
                    "tool_outcomes_complete": True,
                    "protocol_shaped_tool_rejection_count": 1,
                },
            ),
            "criterion_2": _criterion(
                True, False, {"optimizer_steps": 3, "gradients_finite": True}
            ),
            **{f"criterion_{number}": _criterion(True, True, {}) for number in (3, 4, 5, 6)},
        }
        self.assertEqual(
            _select_evidence_verdict(criteria, {"group_count": 3, "tie_count": 3}), "fail"
        )
        self.assertEqual(_select_verdict(criteria, {"group_count": 3, "tie_count": 3}), "fail")

    def test_criterion_1_missing_member_outcome_is_not_computed(self):
        outcome = {"calls": [], "counts_by_code": {}, "files_changed": False}
        groups = []
        for group_index, status in enumerate(("ready", "tie", "tie")):
            members = []
            for member_index in range(4):
                member = {
                    "member_id": f"member-{group_index}-{member_index}",
                    "eligibility": "structurally_eligible",
                    "tool_outcomes": outcome,
                }
                if group_index == 0 and member_index == 0:
                    member.pop("tool_outcomes")
                members.append(member)
            groups.append(
                {
                    "group_id": f"group-{group_index}",
                    "decision": SimpleNamespace(status=status),
                    "admission": SimpleNamespace(
                        members=tuple({"status": "admitted"} for _ in range(4))
                    ),
                    "members": members,
                }
            )

        computed, passed, evidence = _criterion_1(groups, group_count=3, group_errors=[])
        self.assertFalse(computed)
        self.assertFalse(passed)
        self.assertFalse(evidence["tool_outcomes_complete"])

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

    def test_protocol_rejection_in_store_fails_probe_criterion_1(self):
        samples = (
            SampleResult(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "same-id",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": {"path": "draft.txt"}},
                        },
                        {
                            "id": "same-id",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": {"path": "draft.txt"}},
                        },
                    ],
                }
            ),
            SampleResult(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "write-id",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": {"path": "draft.txt", "content": "revised"},
                            },
                        }
                    ],
                }
            ),
            SampleResult(
                {"role": "assistant", "content": "The revision is ready.", "tool_calls": []}
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_rollout_fixture(
                Path(temporary) / "source-store", mode="slice", sample_results=samples
            )
            final = run_slice(fixture)
            outcomes = read_member_tool_outcomes(
                fixture.gate.view(fixture.store, fixture.checkpoint_ids[0]),
                fixture.gate.view(fixture.store, final.checkpoint_id),
            )
            self.assertEqual(outcomes["protocol_shaped_rejection_count"], 1)

            groups = []
            for group_index, status in enumerate(("ready", "tie", "tie")):
                members = []
                for member_index in range(4):
                    member_outcomes = (
                        outcomes
                        if group_index == 0 and member_index == 0
                        else {
                            "calls": [],
                            "counts_by_code": {},
                            "protocol_shaped_rejection_count": 0,
                            "files_changed": False,
                            "changed_paths": [],
                        }
                    )
                    members.append(
                        {
                            "member_id": f"member-{group_index}-{member_index}",
                            "eligibility": "structurally_eligible",
                            "turns": [],
                            "tool_outcomes": member_outcomes,
                        }
                    )
                groups.append(
                    {
                        "group_id": f"group-{group_index}",
                        "decision": SimpleNamespace(status=status),
                        "admission": SimpleNamespace(
                            members=tuple({"status": "admitted"} for _ in range(4))
                        ),
                        "members": members,
                        "terminations": {},
                        "rewards": [],
                    }
                )

            run_dir = Path(temporary) / "probe-run"
            training_root = run_dir / "training"
            for group_index in range(3):
                (training_root / "groups" / f"group-{group_index}").mkdir(parents=True)
            invocation_root = training_root / "invocations"
            invocation_root.mkdir(parents=True)
            for step, before, after in ((2, "before", "middle"), (3, "middle", "after")):
                invocation = invocation_root / f"invocation-{step}"
                invocation.mkdir()
                (invocation / "complete.json").write_text(
                    json.dumps(
                        {
                            "global_step": step,
                            "resumed_step": 2 if step == 3 else None,
                            "trainable_before": {"sha256": before},
                            "trainable_after": {"sha256": after},
                        }
                    )
                )
            for stage, step_count in (("train", 2), ("resume", 1)):
                stage_dir = run_dir / "stages" / stage
                stage_dir.mkdir(parents=True)
                (stage_dir / "gradient-observer.json").write_text(
                    json.dumps(
                        {"finite": True, "gradient_tensor_count": 1, "step_count": step_count}
                    )
                )
            prepared = {
                "recipe": {"execution_mode": "cpu-dry-run"},
                "source": {"commit": "test", "tree": "test"},
                "task_graph_hashes": [
                    {"task_id": task}
                    for task in ("t1-lighthouse", "t2-winter-garden", "t3-coastal-post")
                ],
                "ceilings": {
                    "peak_rss_bytes": 1000,
                    "run_directory_growth_bytes": 1000,
                    "peak_reserved_gpu_bytes": 1000,
                    "aggregate_gpu_seconds": 1000,
                },
            }
            run_dir.mkdir(exist_ok=True)
            (run_dir / "prepare.json").write_text(json.dumps(prepared))
            inspected = [
                {
                    "status": "admitted",
                    "member_count": 4,
                    "admitted_count": 4,
                    "mismatch_count": 0,
                }
            ] * 6
            resources = {
                stage: {
                    "status": "completed",
                    "ceilings_exceeded": [],
                    "peak_rss_bytes": 100,
                    "run_directory_growth_bytes": 100,
                }
                for stage in ("train", "resume")
            }
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
                    "writing_agent.grpo_task_graph_probe_evidence._collect_groups",
                    return_value=(groups, []),
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe_evidence._inspections",
                    return_value=(inspected, True, []),
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe_evidence._tamper_controls",
                    return_value={"all_refused": True},
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe_evidence._checkpoint_evidence",
                    return_value={
                        "verified_steps": [1, 2, 3],
                        "resume_from_step": 2,
                        "global_step": 3,
                        "export_matches_checkpoint": True,
                        "peft_reload_matches_checkpoint": True,
                        "resident_trainable_hash_matches_export": True,
                    },
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe_evidence._adapter_reload",
                    return_value={},
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe_evidence._resource_stages",
                    return_value=resources,
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe_evidence._ledger_metrics",
                    return_value={},
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe_evidence._generation_metrics",
                    return_value={"decision_count": 1, "mean_generate_time_seconds": 0.0},
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe_evidence._observer_metrics",
                    return_value={"mean": 0.0},
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe_evidence._checkpoint_disk_metrics",
                    return_value=[],
                ),
                patch(
                    "writing_agent.grpo_task_graph_probe_evidence._privacy_scan",
                    return_value={"hits": [], "checked_files": 0},
                ),
                patch("writing_agent.grpo_task_graph_probe_evidence._disk_bytes", return_value=0),
            ):
                result = inspect_run(run_dir, mode="cpu-dry-run")

            self.assertEqual(result["verdict"], "fail")
            self.assertTrue(result["criteria"]["criterion_1"]["computed"])
            self.assertFalse(result["criteria"]["criterion_1"]["passed"])
            self.assertEqual(
                result["criteria"]["criterion_1"]["evidence"][
                    "protocol_shaped_tool_rejection_count"
                ],
                1,
            )
            self.assertGreaterEqual(result["measurements"]["tool_call_count"], 1)
            member_outcome = result["measurements"]["tool_call_outcomes_by_member"][0]
            self.assertTrue(member_outcome["files_changed"])
            self.assertEqual(member_outcome["counts_by_code"]["duplicate_id"], 1)
            self.assertEqual(json.loads((run_dir / "result.json").read_text())["verdict"], "fail")


if __name__ == "__main__":
    unittest.main()
