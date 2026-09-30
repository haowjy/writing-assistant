"""CLI contract and tiny-model integration tests for the S11 trace-check runner."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.task_graph_trace_check import _run_trace, claim_output_directory, main, parse_args
from scripts.task_graph_trace_check_support import (
    TOKENIZER_ROOT,
    classify_protocol_shape,
    decision_summaries,
    load_trace_task_entry,
    same_incomplete_reason,
    tool_result_protocol_errors,
    trace_completion_outcome,
    trace_events_for_artifact,
)
from writing_agent.task_graph import canonical_bytes

_TINY_TRACE_AVAILABLE = all(
    importlib.util.find_spec(module) is not None
    for module in ("torch", "transformers", "peft", "trl")
) and all(
    (TOKENIZER_ROOT / name).is_file()
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
)


class TaskGraphTraceCheckCliTests(unittest.TestCase):
    def test_real_t1_startup_validates_budgets_and_builds_admitted_entry(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "configs/phase8/probe-tasks/t1-lighthouse.json"
        )
        config, entry = load_trace_task_entry(config_path)

        self.assertEqual(config["id"], "t1-lighthouse")
        self.assertNotIn("max_author_calls", config["probe_settings"])
        self.assertIsNotNone(entry.state)
        self.assertEqual(entry.graph.node(entry.node_id).contract.interaction_contract.mode, "none")

    def test_real_t1_trace_startup_reaches_model_load_without_loading_it(self):
        class ModelLoadReached(RuntimeError):
            pass

        state = {"stage": "startup"}
        args = SimpleNamespace(model_path=None, tokenizer_root=TOKENIZER_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "scripts.task_graph_trace_check.load_model_and_tokenizer",
                side_effect=ModelLoadReached,
            ):
                with self.assertRaises(ModelLoadReached):
                    _run_trace(args, Path(temporary), state)

        self.assertEqual(state["stage"], "load_model")

    def test_startup_rejects_budget_drift_and_any_author_budget(self):
        config_path = (
            Path(__file__).resolve().parents[1] / "configs/phase8/probe-tasks/t1-lighthouse.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "t1.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["probe_settings"]["max_tool_calls"] += 1
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trace_task_entry(path)

            config["probe_settings"]["max_tool_calls"] -= 1
            config["probe_settings"]["max_author_calls"] = 0
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trace_task_entry(path)

    @unittest.skipUnless(
        _TINY_TRACE_AVAILABLE,
        "requires the Phase 8 torch/Transformers/PEFT/TRL overlay and cached Gemma tokenizer",
    )
    def test_tiny_local_gemma_trace_reaches_report_with_none_simulator(self):
        import torch
        from transformers import AutoTokenizer

        from writing_agent.grpo_task_graph_probe_experiment import tiny_gemma

        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_ROOT), local_files_only=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "tiny-model"
            output = root / "trace"
            tiny_gemma(tokenizer.vocab_size).save_pretrained(model_path, safe_serialization=True)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main(
                    [
                        str(output),
                        "--model-path",
                        str(model_path),
                        "--tokenizer-root",
                        str(TOKENIZER_ROOT),
                    ]
                )

            report_path = output / "report.json"
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn(report["status"], {"pass", "fail", "halt"})
            self.assertIn(report["group_status"], {"ready", "tie"})
            self.assertEqual(
                report["group_policy"]["simulator"],
                {"implementation": "scripted-author-v1", "script_ref": None},
            )
            self.assertEqual(
                [member["ordinal"] for member in report["members"]],
                [0, 1],
            )
            self.assertEqual(
                sorted({event["member_ordinal"] for event in report["decision_generate_times"]}),
                [0, 1],
            )

    def test_dry_run_reports_cpu_plan_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            stream = io.StringIO()
            with redirect_stdout(stream):
                self.assertEqual(main([str(output), "--dry-run"]), 0)
            self.assertFalse(output.exists())
            self.assertIn('"dry_run": true', stream.getvalue())
            self.assertIn('"device": "cpu"', stream.getvalue())

    def test_output_claim_refuses_existing_directory_and_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing_dir = root / "existing-dir"
            existing_dir.mkdir()
            existing_file = root / "existing-file"
            existing_file.write_text("do not overwrite", encoding="utf-8")
            for path in (existing_dir, existing_file):
                with self.subTest(path=path), self.assertRaises(FileExistsError):
                    claim_output_directory(path)
            claimed = claim_output_directory(root / "new" / "run")
            self.assertTrue(claimed.is_dir())
            self.assertEqual(claimed.stat().st_mode & 0o777, 0o700)

    def test_cli_refuses_existing_output_before_touching_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            marker = output / "evidence.txt"
            marker.write_text("preserve", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                self.assertEqual(main([str(output)]), 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            self.assertIn("already exists", stderr.getvalue())

    def test_local_model_override_is_parsed_without_model_dependencies(self):
        args = parse_args(["/tmp/run", "--model-path", "/tmp/tiny-model"])
        self.assertEqual(args.model_path, Path("/tmp/tiny-model"))

    def test_same_incomplete_reason_is_a_fixture_revision_signal(self):
        members = [
            {"task_status": "incomplete", "stop_reason": "context_tokens_budget"},
            {"task_status": "incomplete", "stop_reason": "context_tokens_budget"},
        ]
        self.assertEqual(same_incomplete_reason(members), "context_tokens_budget")
        members[1]["stop_reason"] = "decision_token_limit"
        self.assertIsNone(same_incomplete_reason(members))

    def test_audit_refusal_classifies_failed_checks_as_protocol_shape(self):
        self.assertEqual(
            classify_protocol_shape(
                {
                    "type": "TrainingAuditError",
                    "message": "native training audit refused one or more members",
                    "failed_checks": ["external_suffix_and_context_limit"],
                }
            ),
            "delta",
        )
        self.assertEqual(
            classify_protocol_shape(
                {
                    "type": "TrainingAuditError",
                    "message": "native training audit refused one or more members",
                    "failed_checks": ["raw_output_and_message"],
                }
            ),
            "parse",
        )
        self.assertEqual(
            classify_protocol_shape(
                {
                    "type": "ToolOutcomeError",
                    "protocol_shape": "tool_result",
                    "message": "committed calls and results do not pair",
                }
            ),
            "tool_result",
        )

    def test_failure_trace_timing_is_canonical_integer_data(self):
        events = trace_events_for_artifact(
            [
                {
                    "member_ordinal": 0,
                    "generate_seconds": 0.25,
                    "sample_seconds": 0.002,
                    "sample_error": {"type": "ProtocolError", "message": "refused"},
                }
            ]
        )

        self.assertEqual(events[0]["generate_nanoseconds"], 250_000_000)
        self.assertEqual(events[0]["sample_nanoseconds"], 2_000_000)
        self.assertNotIn("generate_seconds", events[0])
        canonical_bytes({"kind": "trace-check-failure-v1", "sample_trace": events})

    def test_decision_report_preserves_each_tool_call_and_result(self):
        calls = [{"id": "lineage:call:2:0", "name": "write_file", "result": "ok"}]
        self.assertEqual(
            decision_summaries(
                [
                    {
                        "member_ordinal": 1,
                        "decision_ordinal": 2,
                        "generated_tokens": 34,
                        "termination": {"kind": "native_stop"},
                        "generate_seconds": 1.25,
                        "tool_calls": calls,
                    }
                ]
            ),
            [
                {
                    "member_ordinal": 1,
                    "decision_ordinal": 2,
                    "generated_tokens": 34,
                    "termination": {"kind": "native_stop"},
                    "generate_seconds": 1.25,
                    "tool_calls": calls,
                }
            ],
        )

    def test_tool_outcome_codes_are_reported_and_only_protocol_codes_halt(self):
        events = [
            {
                "member_ordinal": 0,
                "decision_ordinal": 1,
                "tool_calls": [
                    {
                        "name": "write_file",
                        "result": {"code": "path_missing", "text": "file path is missing"},
                    },
                    {
                        "name": "write_file",
                        "result": {"code": "duplicate_id", "text": "Duplicate tool call id"},
                    },
                ],
            }
        ]
        self.assertEqual(
            tool_result_protocol_errors(events),
            [
                {
                    "member_ordinal": 0,
                    "decision_ordinal": 1,
                    "name": "write_file",
                    "result": {"code": "duplicate_id", "text": "Duplicate tool call id"},
                    "code": "duplicate_id",
                }
            ],
        )
        self.assertEqual(
            trace_completion_outcome(
                tool_result_protocol_errors(events),
                inspector_identical=True,
                incomplete_reason=None,
            )[0],
            "halt",
        )
        self.assertEqual(
            trace_completion_outcome([], inspector_identical=True, incomplete_reason=None),
            ("pass", None),
        )
        missing = [{"member_ordinal": 0, "tool_calls": [{"name": "read_file"}]}]
        self.assertEqual(len(tool_result_protocol_errors(missing)), 1)


if __name__ == "__main__":
    unittest.main()
