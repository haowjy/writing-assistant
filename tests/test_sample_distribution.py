"""Pooled distribution measurements and the sample floors that gate them.

The per-attempt profile cannot compute MMD, self-BLEU or dispersion, because they are
defined across outputs rather than within one. These tests use a stand-in extractor so the
pooling and the floors are covered without loading a tokenizer or an embedding model.
"""

import hashlib
import unittest
from dataclasses import asdict

from writing_agent.catalog import fingerprint
from writing_agent.prose import (
    MINIMUM_SAMPLES,
    RELIABLE_SAMPLES,
    FeatureConfig,
    bandwidth,
    lexical_features,
    sample_distribution,
    sample_power,
    sampling_plan,
)


class FakeExtractor:
    """Satisfies the feature contract without any model work.

    It returns a deterministic pseudo-embedding so the embedding-consuming measures are
    covered too; the real extractor differs only in where the vectors come from.
    """

    def __init__(self):
        self.config = FeatureConfig()

    def extract(self, text, *, tokens=False, embeddings=False, allow_download=False):
        record = {
            "prose_hash": fingerprint(text),
            "config": asdict(self.config),
            "lexical": lexical_features(text),
            "tokens": text.split(),
        }
        if embeddings:
            digest = hashlib.sha256(text.encode()).digest()
            record["embedding"] = [byte / 255.0 for byte in digest[:16]]
        return record

    def reference_features(self, texts):
        return [self.extract(text, tokens=True, embeddings=True) for text in texts]


def card(scenario, text):
    return {"scenario_id": scenario, "artifacts": [{"id": "prose", "status": "ok", "text": text}]}


def cards(scenario, texts):
    return [card(scenario, text) for text in texts]


WORDS = "archive keeper tide salt ledger shelf stair cold grey water".split()


def prose(index: int, length: int = 40) -> str:
    """Deterministic pseudo-prose with enough variety to be a distinct sample."""
    return " ".join(WORDS[(index * 7 + step * 3) % len(WORDS)] for step in range(length))


class PoolingTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FakeExtractor()

    def test_attempts_of_one_scenario_pool_into_one_distribution(self):
        result = sample_distribution(cards("LF-01", [prose(i) for i in range(10)]), self.extractor)
        self.assertEqual(result["scenario_id"], "LF-01")
        self.assertEqual(result["attempts"], 10)
        self.assertEqual(result["samples"], 10)

    def test_pooling_two_scenarios_is_rejected(self):
        mixed = cards("LF-01", [prose(0)]) + cards("LF-02", [prose(1)])
        with self.assertRaisesRegex(ValueError, "one scenario"):
            sample_distribution(mixed, self.extractor)

    def test_pooling_nothing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No attempts"):
            sample_distribution([], self.extractor)

    def test_repeated_attempts_make_the_across_output_measures_exist(self):
        result = sample_distribution(cards("LF-01", [prose(i) for i in range(10)]), self.extractor)
        for metric in ("D4", "D6"):
            self.assertEqual(result["metrics"][metric]["status"], "ok", metric)
        self.assertIsInstance(result["metrics"]["D4"]["value"], float)
        self.assertIsInstance(result["metrics"]["D6"]["value"], float)

    def test_mmd_needs_a_reference_set_and_a_frozen_bandwidth(self):
        texts = [prose(i) for i in range(24)]
        reference = [prose(100 + i) for i in range(24)]
        features = self.extractor.reference_features(reference)
        sigma = bandwidth([f["embedding"] for f in features])
        result = sample_distribution(
            cards("LF-01", texts), self.extractor, references=features, sigma=sigma
        )
        self.assertEqual(result["metrics"]["D2"]["status"], "ok")
        self.assertEqual(result["reference_samples"], 24)

    def test_mmd_without_a_bandwidth_is_unavailable_rather_than_wrong(self):
        features = self.extractor.reference_features([prose(i) for i in range(24)])
        result = sample_distribution(
            cards("LF-01", [prose(i) for i in range(24)]), self.extractor, references=features
        )
        self.assertEqual(result["metrics"]["D2"]["status"], "unavailable")

    def test_identical_outputs_are_kept_because_they_are_the_signal(self):
        texts = [prose(0) for _ in range(6)] + [prose(i) for i in range(1, 5)]
        result = sample_distribution(cards("LF-01", texts), self.extractor)
        self.assertEqual(result["samples"], 10)
        self.assertAlmostEqual(result["metrics"]["D11"]["value"]["duplicate_output_rate"], 0.5)

    def test_a_single_attempt_still_pools_rather_than_failing(self):
        result = sample_distribution(cards("LF-01", [prose(0)]), self.extractor)
        self.assertEqual(result["samples"], 1)
        # Structure is per-sample and still measurable; the across-output measures are not.
        self.assertEqual(result["metrics"]["D9"]["status"], "ok")
        for metric in ("D4", "D6"):
            self.assertEqual(result["metrics"][metric]["status"], "insufficient_samples", metric)


class SamplingPlanTests(unittest.TestCase):
    def test_a_plan_names_every_distributional_measure(self):
        self.assertEqual(set(sampling_plan(50)), set(MINIMUM_SAMPLES))

    def test_a_cheap_run_is_told_what_it_cannot_measure(self):
        plan = sampling_plan(4)
        self.assertEqual(plan["D2"], "insufficient")
        self.assertEqual(plan["D4"], "insufficient")

    def test_a_run_just_under_a_reliable_count_is_low_not_ok(self):
        self.assertEqual(sampling_plan(RELIABLE_SAMPLES["D4"] - 1)["D4"], "low")
        self.assertEqual(sampling_plan(RELIABLE_SAMPLES["D4"])["D4"], "ok")

    def test_the_plan_and_the_enforced_floor_agree(self):
        for metric, floor in MINIMUM_SAMPLES.items():
            with self.subTest(metric=metric):
                self.assertEqual(sampling_plan(floor - 1)[metric], "insufficient")
                self.assertIn(sampling_plan(floor)[metric], {"low", "ok"})

    def test_a_nonsense_sample_count_is_rejected(self):
        for samples in (0, -1, 1.5, "4", None):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                sampling_plan(samples)


class SampleFloorTests(unittest.TestCase):
    def test_each_distributional_measure_declares_both_floors(self):
        self.assertEqual(set(MINIMUM_SAMPLES), set(RELIABLE_SAMPLES))
        for metric in MINIMUM_SAMPLES:
            self.assertLess(MINIMUM_SAMPLES[metric], RELIABLE_SAMPLES[metric])

    def test_below_the_floor_a_measure_is_withheld_not_reported(self):
        entry = sample_power("D2", {"status": "ok", "value": 0.01}, MINIMUM_SAMPLES["D2"] - 1)
        self.assertEqual(entry["status"], "insufficient_samples")
        self.assertIsNone(entry.get("value"))
        self.assertIn(str(MINIMUM_SAMPLES["D2"]), entry["reason"])

    def test_at_the_floor_it_reports_with_low_power(self):
        entry = sample_power("D2", {"status": "ok", "value": 0.01}, MINIMUM_SAMPLES["D2"])
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["power"], "low")
        self.assertEqual(entry["samples"], MINIMUM_SAMPLES["D2"])

    def test_at_the_reliable_count_power_is_ok(self):
        entry = sample_power("D4", {"status": "ok", "value": 0.1}, RELIABLE_SAMPLES["D4"])
        self.assertEqual(entry["power"], "ok")

    def test_an_already_unavailable_measure_is_left_alone(self):
        original = {"status": "unavailable", "reason": "no embeddings"}
        self.assertEqual(sample_power("D2", original, 100), original)

    def test_a_pooled_run_withholds_an_underpowered_measure(self):
        few = MINIMUM_SAMPLES["D4"] - 1
        result = sample_distribution(
            cards("LF-01", [prose(i) for i in range(few)]), FakeExtractor()
        )
        self.assertEqual(result["metrics"]["D4"]["status"], "insufficient_samples")
        self.assertEqual(result["metrics"]["D4"]["samples"], few)

    def test_a_pooled_run_reports_a_powered_measure_with_its_count(self):
        many = RELIABLE_SAMPLES["D4"]
        result = sample_distribution(
            cards("LF-01", [prose(i) for i in range(many)]), FakeExtractor()
        )
        self.assertEqual(result["metrics"]["D4"]["status"], "ok")
        self.assertEqual(result["metrics"]["D4"]["power"], "ok")
        self.assertEqual(result["metrics"]["D4"]["samples"], many)

    def test_non_distributional_measures_pass_through(self):
        result = sample_distribution(cards("LF-01", [prose(i) for i in range(10)]), FakeExtractor())
        self.assertNotIn("samples", result["metrics"]["D9"])
        self.assertEqual(result["metrics"]["D9"]["status"], "ok")

    def test_features_without_embeddings_cost_only_the_embedding_measures(self):
        result = sample_distribution(
            cards("LF-01", [prose(i) for i in range(10)]),
            FakeExtractor(),
            embeddings=False,
        )
        # The fake honours the flag, so D2/D6 disappear while the n-gram measures survive.
        self.assertEqual(result["metrics"]["D4"]["status"], "ok")
        self.assertIn("D6", result["metrics"])


if __name__ == "__main__":
    unittest.main()
