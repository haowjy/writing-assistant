"""Torch-free CLI contract tests for the S11 native trace-check runner."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts.task_graph_trace_check import claim_output_directory, main, parse_args
from scripts.task_graph_trace_check_support import same_incomplete_reason


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


if __name__ == "__main__":
    unittest.main()
