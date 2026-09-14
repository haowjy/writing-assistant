"""Reference assignments must stay bound to reviewed texts and scenario versions."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from writing_agent.references import load_matched_references

ROOT = Path(__file__).resolve().parents[1]


class ReferenceTests(unittest.TestCase):
    def test_frozen_manifest_rejects_changed_text_scenario_or_training_role(self):
        original = json.loads((ROOT / "data/references/pilot-matched-v1.json").read_text())
        scenarios = {
            key: {
                "visible_hash": value["visible_hash"],
                "visible": {"prose": bool(value["reference_ids"])},
            }
            for key, value in original["assignments"].items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "refs.json"
            path.write_text(json.dumps(original))
            self.assertEqual(len(load_matched_references(path, scenarios)["references"]), 9)
            for field, value in [("text", "Changed"), ("role", "train")]:
                changed = copy.deepcopy(original)
                changed["references"][0][field] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_matched_references(path, scenarios)
            path.write_text(json.dumps(original))
            changed_scenarios = copy.deepcopy(scenarios)
            changed_scenarios["F1-01"]["visible_hash"] = "new scenario"
            with self.assertRaises(ValueError):
                load_matched_references(path, changed_scenarios)
