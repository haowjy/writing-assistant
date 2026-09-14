"""Pin the paid rubric parser and upstream task expansion."""

import unittest

from scripts.creative_writing_v3 import UPSTREAM, parse_scores, tasks


class CreativeWritingTests(unittest.TestCase):
    @unittest.skipUnless(
        (UPSTREAM / "data/creative_writing_prompts_v3.json").exists(),
        "Upstream benchmark data not downloaded",
    )
    def test_variant_expansion(self):
        selected = tasks()
        self.assertEqual(len(selected), 96)
        self.assertEqual(len({t["id"] for t in selected}), 96)
        self.assertTrue(all("<SEED>" not in t["prompt"] for t in selected))

    def test_complete_scores(self):
        self.assertEqual(
            parse_scores("[Scores]\nClarity: 17\nCoherence: [15]", ["Clarity", "Coherence"]),
            {"Clarity": 17, "Coherence": 15},
        )
        self.assertEqual(parse_scores("Clarity: 17", ["Clarity", "Coherence"]), {"Clarity": 17})
        for text in ("Clarity: 30\nCoherence: 15", "No scores here"):
            with self.assertRaises(ValueError):
                parse_scores(text, ["Clarity", "Coherence"])

    def test_approved_subset_budget_and_report_denominator(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from scripts import creative_writing_v3 as creative
        from writing_agent.catalog import save_json

        variants = [{"id": f"{i}-{v}", "iteration": v} for i in range(32) for v in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            save_json(root / "revision.json", {"revision": "fixture"})
            with (
                patch.object(creative, "UPSTREAM", root),
                patch.object(creative, "OUTPUT", root / "run"),
                patch.object(creative, "tasks", return_value=variants),
            ):
                manifest = creative.prepare()
                self.assertEqual(manifest["paid_budget_usd"], 2)
                self.assertEqual(len(manifest["tasks"]), 32)
                self.assertTrue(all(t["iteration"] == 1 for t in manifest["tasks"]))
                for task in manifest["tasks"]:
                    folder = creative.OUTPUT / "items" / task["id"]
                    save_json(folder / "generation.json", {"status": "completed"})
                    save_json(folder / "judgment.json", {"status": "completed", "score_0_100": 50})
                report = creative.report()
                self.assertEqual(report["expected"], 32)
                self.assertEqual(report["selected_score_0_100"], 50)
                self.assertIsNone(report["score_0_100"])
                self.assertEqual(
                    json.loads((creative.OUTPUT / "manifest.json").read_text()), manifest
                )
