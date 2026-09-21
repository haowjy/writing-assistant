"""The context budget claim: KV arithmetic, and whether the harness can run the suite."""

import json
import unittest
from pathlib import Path

from writing_agent.inference import HARNESS_CONTEXT_TOKENS, kv_cache_bytes

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "data/scenarios/longform-v1.json"

# google/gemma-4-E2B-it text config, as cached at the pinned revision.
GEMMA = {
    "layer_types": ["sliding_attention"] * 28 + ["full_attention"] * 7,
    "sliding_window": 512,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "num_global_key_value_heads": None,
    "max_position_embeddings": 131072,
}


class KVCacheTests(unittest.TestCase):
    def test_the_full_attention_cache_dominates_and_matches_arithmetic(self):
        # 7 full layers x 131072 tokens x 2 (K,V) x 2 bytes x head_dim 512 x 1 KV head
        self.assertEqual(kv_cache_bytes(GEMMA, 131072), 7 * 131072 * 4 * 512 + 28 * 512 * 4 * 256)

    def test_sliding_layers_stop_growing_past_their_window(self):
        at_window = kv_cache_bytes(GEMMA, 512)
        far_beyond = kv_cache_bytes(GEMMA, 131072)
        self.assertLess(at_window, far_beyond)
        # Only the seven full-attention layers may grow with context.
        growth = far_beyond - at_window
        self.assertEqual(growth, 7 * (131072 - 512) * 4 * 512)

    def test_grouped_query_attention_divides_the_cache(self):
        multi = {**GEMMA, "num_key_value_heads": 8}
        self.assertEqual(kv_cache_bytes(multi, 65536), kv_cache_bytes(GEMMA, 65536) * 8)

    def test_a_separate_global_head_count_is_honoured(self):
        # The global KV head count overrides the sliding one for full-attention layers.
        separate = {**GEMMA, "num_global_key_value_heads": 2}
        self.assertEqual(
            kv_cache_bytes(separate, 131072), 7 * 131072 * 4 * 512 * 2 + 28 * 512 * 4 * 256
        )

    def test_the_cache_is_far_below_a_naive_all_layers_estimate(self):
        naive = len(GEMMA["layer_types"]) * 131072 * 4 * 512
        self.assertLess(kv_cache_bytes(GEMMA, 131072), naive / 4)

    def test_128k_is_a_rounding_error_against_the_device(self):
        gib = kv_cache_bytes(GEMMA, 131072) / 1024**3
        self.assertLess(gib, 2.0)

    def test_the_cache_is_linear_in_context_for_the_full_attention_part(self):
        self.assertAlmostEqual(
            kv_cache_bytes(GEMMA, 65536) - kv_cache_bytes(GEMMA, 32768),
            7 * 32768 * 4 * 512,
        )


class HarnessLimitTests(unittest.TestCase):
    def test_the_harness_limit_is_a_positive_integer(self):
        self.assertIs(type(HARNESS_CONTEXT_TOKENS), int)
        self.assertGreater(HARNESS_CONTEXT_TOKENS, 0)

    def test_the_harness_limit_stays_within_the_model(self):
        self.assertLessEqual(HARNESS_CONTEXT_TOKENS, GEMMA["max_position_embeddings"])

    def test_the_limit_covers_every_long_form_case_it_must_run(self):
        """A case whose supplied context already fills the window cannot be run at all."""
        targets = [case["context_target_tokens"] for case in json.loads(SPEC.read_text())["cases"]]
        self.assertTrue(targets)
        self.assertGreaterEqual(HARNESS_CONTEXT_TOKENS, max(targets))

    def test_the_limit_leaves_room_to_generate(self):
        # A case must fit its supplied context and still have room for the deliverable.
        headroom = HARNESS_CONTEXT_TOKENS - max(
            case["context_target_tokens"] for case in json.loads(SPEC.read_text())["cases"]
        )
        self.assertGreater(headroom, 4096)


if __name__ == "__main__":
    unittest.main()
