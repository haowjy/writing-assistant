"""Ensure literary grading replaces coding instructions and sends only its packet."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from writing_agent.grading import CodexGrader


class GraderIsolationTests(unittest.TestCase):
    def test_custom_base_instructions_and_fresh_launch(self):
        packet = {"rubrics": {}, "checks": [], "output": "A fictional scene"}
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            self.assertEqual(json.loads(kwargs["input"]), packet)
            self.assertIn("--ephemeral", command)
            self.assertIn("--ignore-user-config", command)
            self.assertIn("project_doc_max_bytes=0", command)
            self.assertIn("skills.include_instructions=false", command)
            self.assertIn('developer_instructions=""', command)
            entry = next(x for x in command if x.startswith("model_instructions_file="))
            instructions = Path(json.loads(entry.split("=", 1)[1])).read_text()
            self.assertIn("independent evaluator", instructions)
            self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text(json.dumps({"assessments": [], "checks": []}))
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            tempfile.TemporaryDirectory() as root,
            patch("writing_agent.grading.shutil.which", return_value="/codex"),
            patch("writing_agent.grading.subprocess.run", side_effect=execute),
        ):
            grader = CodexGrader(Path(root), max_calls=1)
            first = grader.grade(packet)
            self.assertEqual(first["status"], "ok")
            self.assertEqual(grader.grade(packet), first)
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                grader.grade({**packet, "output": "Different"})["status"], "budget_exhausted"
            )
