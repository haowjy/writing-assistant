"""Protect external-run resume identities and failure preservation."""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from writing_agent.external import generate_tasks


class ExternalGenerationTests(unittest.TestCase):
    def test_resume_and_changed_selection(self):
        calls = []

        @contextmanager
        def checkpoint(config):
            calls.append(config)
            yield None, None, config

        class Backend:
            def __init__(self, *args):
                pass

            def complete(self, messages, tools, *, emit):
                emit({"type": "model_output", "text": "failed raw output"})
                raise ValueError("Bad model tool syntax")

        with tempfile.TemporaryDirectory() as root:
            destination = Path(root)
            tasks = [{"id": "1", "prompt": "Write"}, {"id": "2", "prompt": "Continue"}]
            with (
                patch("writing_agent.external.load_checkpoint", checkpoint),
                patch("writing_agent.external.TransformersBackend", Backend),
            ):
                first = generate_tasks(tasks, {}, destination)
                second = generate_tasks(tasks, {}, destination)
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 1)
                self.assertEqual(first["count"], 2)
                record = json.loads((destination / "items/1/generation.json").read_text())
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["trace"][0]["text"], "failed raw output")
                with self.assertRaises(ValueError):
                    generate_tasks(tasks, {"seed": 1}, destination)

    def test_invalid_task_paths(self):
        with tempfile.TemporaryDirectory() as root:
            for key in ("..", "a/b", ""):
                with self.assertRaises(ValueError):
                    generate_tasks([{"id": key, "prompt": "x"}], {}, Path(root))
            self.assertFalse((Path(root) / "generation-manifest.json").exists())

    def test_private_data_separation_and_frozen_grading_selection(self):
        from scripts import external_checks as checks

        instructions = [
            {"key": i, "prompt": f"Task {i}", "kwargs": ["private"]} for i in range(541)
        ]
        coding = [
            {
                "task_id": f"HumanEval/{i}",
                "prompt": "def f(): ...",
                "canonical_solution": "PRIVATE SOLUTION",
            }
            for i in range(32)
        ]
        with (
            tempfile.TemporaryDirectory() as root,
            patch.object(checks, "OUTPUT", Path(root)),
            patch.object(
                checks,
                "read_jsonl",
                side_effect=lambda p: instructions if "ifeval" in str(p) else coding,
            ),
        ):
            tasks = checks.prepare()
            self.assertNotIn("PRIVATE SOLUTION", json.dumps(tasks))
            self.assertNotIn("kwargs", json.dumps(tasks))
            instructions[0]["prompt"] = "Changed input"
            with self.assertRaises(ValueError):
                checks.prepare()
