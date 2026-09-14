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
        for text in ("Clarity: 30\nCoherence: 15", "Clarity: 17"):
            with self.assertRaises(ValueError):
                parse_scores(text, ["Clarity", "Coherence"])
