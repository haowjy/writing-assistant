"""Torch-free CLI contract tests for the S11 native trace-check runner."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts.task_graph_trace_check import claim_output_directory, main, parse_args
from scripts.task_graph_trace_check_support import (
    _tool_results_by_id,
    classify_protocol_shape,
    decision_summaries,
    same_incomplete_reason,
    tool_result_protocol_errors,
    trace_completion_outcome,
    trace_events_for_artifact,
)
from writing_agent.task_graph import canonical_bytes


class TaskGraphTraceCheckCliTests(unittest.TestCase):
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
                {"type": "ProtocolError", "message": "Task-graph tool result pairing mismatch"}
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

    def test_tool_results_are_reported_and_only_protocol_shaped_errors_halt(self):
        results = _tool_results_by_id(
            [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "call_id": "ok-id",
                            "content": {"ok": True, "result": "written"},
                        },
                        {
                            "type": "tool_result",
                            "call_id": "argument-error-id",
                            "content": {"ok": False, "error": "bad ask_author arguments"},
                        },
                        {
                            "type": "tool_result",
                            "call_id": "duplicate-id",
                            "content": {"ok": False, "error": "Duplicate tool call id"},
                        },
                    ],
                }
            ]
        )
        self.assertEqual(
            results,
            {
                "ok-id": "ok",
                "argument-error-id": "bad ask_author arguments",
                "duplicate-id": "Duplicate tool call id",
            },
        )
        events = [
            {
                "member_ordinal": 0,
                "decision_ordinal": 1,
                "tool_calls": [
                    {"name": "ask_author", "result": "bad ask_author arguments"},
                    {"name": "write_file", "result": "Duplicate tool call id"},
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
                    "result": "Duplicate tool call id",
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


if __name__ == "__main__":
    unittest.main()
