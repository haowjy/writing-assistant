"""Exercise the real native sampler through the trainer's DAPO member loop."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from writing_agent.grpo_task_graph_probe_experiment import TOKENIZER_PATH

_NATIVE_TRAINER_AVAILABLE = all(
    importlib.util.find_spec(module) is not None
    for module in ("torch", "transformers", "peft", "trl")
) and all(
    (TOKENIZER_PATH / name).is_file()
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
)


@unittest.skipUnless(
    _NATIVE_TRAINER_AVAILABLE,
    "requires the Phase 8 torch/Transformers/PEFT/TRL overlay and cached Gemma tokenizer",
)
class NativeBackendTrainerTests(unittest.TestCase):
    def test_real_native_sampler_completes_one_audited_dapo_step(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        from transformers import AutoTokenizer

        from writing_agent.grpo_task_graph_probe_experiment import (
            TOKENIZER_PATH,
            make_task_graph_run,
            settings,
            tiny_gemma,
        )
        from writing_agent.native_audit import inspect_group_offline
        from writing_agent.native_gemma import NativeGemmaSampleBackend
        from writing_agent.task_graph_group import GroupCoordinatorV1

        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER_PATH), local_files_only=True, trust_remote_code=False
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = make_task_graph_run(
                output,
                model_factory=lambda: tiny_gemma(tokenizer.vocab_size),
                sample_backend_factory=NativeGemmaSampleBackend,
                stop_after_steps=1,
                tokenizer=tokenizer,
                tokenizer_root=TOKENIZER_PATH,
                recipe=settings(),
            )

            self.assertEqual(result["global_step"], 1)
            observer = result["task_graph_observer"]
            self.assertEqual(observer["steps"], [0])
            self.assertEqual(observer["loss_calls"], 4)
            self.assertTrue(observer["masks_match_export"])
            self.assertTrue(observer["advantages_match_export"])
            self.assertTrue(observer["dapodenominator_matches_active_tokens"])
            self.assertTrue(observer["tie_loss_zero"])
            self.assertLess(observer["max_sampled_recomputed_logprob_drift"], 1e-4)

            groups = GroupCoordinatorV1.groups_by_sequence(output / "groups")
            self.assertEqual(set(groups), {0})
            spec = groups[0]
            self.assertIsNotNone(spec)
            inspection = json.loads(inspect_group_offline(output, spec.group_id))
            self.assertEqual(inspection["status"], "admitted")
            self.assertEqual(inspection["member_count"], settings().group_size)
            self.assertEqual(inspection["admitted_count"], settings().group_size)
            self.assertEqual(inspection["refused_count"], 0)


if __name__ == "__main__":
    unittest.main()
