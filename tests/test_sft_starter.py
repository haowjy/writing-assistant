import json
import posixpath
import re
import tempfile
import unittest
from pathlib import Path

from scripts.train_sft import ROOT, evaluation_source_groups
from writing_agent.data import read_records, validate_records
from writing_agent.development import author_development
from writing_agent.workspace import Workspace, dispatch


class StarterDatasetTests(unittest.TestCase):
    def test_saved_trajectories_replay_and_preserve_files(self):
        records = read_records(ROOT / "data/training/sft-v1.jsonl")
        validate_records(records)
        for record in records:
            with self.subTest(record=record["id"]), tempfile.TemporaryDirectory() as tmp:
                workspace = Workspace(Path(tmp))
                for path, content in record["initial_files"].items():
                    workspace.write_file(path, content)
                observations = {}
                for message in record["messages"]:
                    for call in message.get("tool_calls", []):
                        function = call["function"]
                        result = dispatch(workspace, function["name"], function["arguments"])
                        self.assertTrue(result["ok"])
                        observations[call["id"]] = result
                    if message["role"] == "tool":
                        self.assertEqual(
                            json.loads(message["content"]),
                            observations.pop(message["tool_call_id"]),
                        )
                self.assertFalse(observations)
                self.assertEqual(workspace.snapshot(), record["expected_files"])
                for path, text in record["expected_files"].items():
                    if path.startswith("kb/"):
                        for target in re.findall(r"\]\(([^)]+)\)", text):
                            resolved = posixpath.normpath(
                                posixpath.join(posixpath.dirname(path), target)
                            )
                            self.assertIn(resolved, record["expected_files"])

    def test_evaluation_and_validation_lineage_are_separate(self):
        excluded = evaluation_source_groups()
        sources, scenarios = author_development(ROOT / "data/scenarios/worlds.json")
        self.assertTrue({s["id"] for s in sources} <= excluded)
        self.assertTrue({g for s in scenarios for g in s["source_ids"]} <= excluded)
        records = read_records(ROOT / "data/training/sft-v1.jsonl")
        split_groups = {}
        for record in records:
            self.assertFalse(set(record["source_groups"]) & excluded)
            self.assertNotEqual(record["source_dataset"], "hanna")
            if record["source_dataset"] == "tell_me_a_story":
                self.assertEqual(record["upstream_split"], "train")
            for group in record["source_groups"]:
                self.assertEqual(split_groups.setdefault(group, record["split"]), record["split"])
