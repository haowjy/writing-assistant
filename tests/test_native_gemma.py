"""Native Gemma rendering, per-turn sampling, and V2 rollout integration."""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

from writing_agent.native_gemma import (
    NATIVE_STOP_TOKEN_IDS,
    NativeGemmaSampleBackend,
    make_native_manifest_descriptors,
)
from writing_agent.task_graph import MessageV1, canonical_json
from writing_agent.task_graph_errors import AdapterContractError
from writing_agent.task_graph_ports import PreparedSamplingInput, RuntimeDependenciesV1
from writing_agent.task_graph_records import RuntimeManifestV2, WriterTurnV2
from writing_agent.task_graph_sampling import NativeSamplingBudget

TOKENIZER_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
TOKENIZER_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
    / TOKENIZER_REVISION
)
H = "a" * 64
P = "b" * 64
Q = "c" * 64


def _tokenizer_file_hashes() -> dict[str, str]:
    names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    return {
        name: hashlib.sha256((TOKENIZER_PATH / name).read_bytes()).hexdigest()
        for name in names
        if (TOKENIZER_PATH / name).is_file()
    }


def _tiny_peft_gemma(tokenizer):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

    torch.manual_seed(123)
    vocab_size = tokenizer.vocab_size
    text_config = Gemma4TextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        num_global_key_value_heads=1,
        max_position_embeddings=4096,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=16,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=vocab_size,
        num_kv_shared_layers=2,
        attention_k_eq_v=True,
        enable_moe_block=False,
        final_logit_softcapping=30.0,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(
            text_config=text_config,
            tie_word_embeddings=False,
            attn_implementation="eager",
        )
    ).to(dtype=torch.float32, device="cpu")
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
    generation_config = model.generation_config
    generation_config.eos_token_id = list(NATIVE_STOP_TOKEN_IDS)
    generation_config.pad_token_id = tokenizer.pad_token_id
    for name, value in {
        "min_length": 0,
        "min_new_tokens": None,
        "min_p": None,
        "typical_p": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "bad_words_ids": None,
        "sequence_bias": None,
        "exponential_decay_length_penalty": None,
    }.items():
        setattr(generation_config, name, value)

    turn_id = tokenizer.convert_tokens_to_ids("<turn|>")

    def terminate_each_decision(_module, _inputs, logits):
        logits[..., turn_id] = logits.max(dim=-1).values + 100.0
        return logits

    hook = model.get_base_model().lm_head.register_forward_hook(terminate_each_decision)
    model.eval()
    return model, hook


def _descriptors(tokenizer, *, template_ref=H, tool_schema_ref=P, max_tokens=1):
    return make_native_manifest_descriptors(
        tokenizer,
        model_id="google/gemma-4-E2B-it",
        revision=TOKENIZER_REVISION,
        tokenizer_files_sha256=_tokenizer_file_hashes(),
        template_ref=template_ref,
        tool_schema_ref=tool_schema_ref,
        max_tokens_per_decision=max_tokens,
    )


def _prepared(descriptors, *, messages=None, ordinal=0, history=None, max_context=256):
    renderer, tokenizer, decoding = descriptors
    message = MessageV1(role="user", content=("Write a short draft.",), origin="entry")
    values = [message.to_dict()] if messages is None else messages
    rendering = {
        "template_ref": renderer.template_ref,
        "tokenizer_ref": tokenizer.identity(),
        "tool_schema_ref": renderer.tool_schema_ref,
    }
    return PreparedSamplingInput(
        context_content_hash="d" * 64,
        context_revision_ref="e" * 64,
        writer_seed=37,
        model_ref=None,
        behavior_policy_ref="f" * 64,
        decoding_ref=decoding.identity(),
        tokenizer_ref=tokenizer.identity(),
        template_ref=renderer.template_ref,
        messages_json=canonical_json(values),
        tools_json="[]",
        rendering_json=canonical_json(rendering),
        native_sampling_budget=NativeSamplingBudget(24, max_context),
        adapter_ref="1" * 64,
        decision_ordinal=ordinal,
        native_history=history,
    )


class NativeGemmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TOKENIZER_PATH.is_dir():
            raise unittest.SkipTest("the pinned Gemma tokenizer is not cached")
        import torch
        from transformers import AutoTokenizer

        cls.torch = torch
        cls.tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
        cls.model, cls.logit_hook = _tiny_peft_gemma(cls.tokenizer)
        cls.descriptors = _descriptors(cls.tokenizer)
        cls.backend = NativeGemmaSampleBackend(
            cls.model,
            cls.tokenizer,
            manifest_descriptors=cls.descriptors,
        )

    @classmethod
    def tearDownClass(cls):
        cls.logit_hook.remove()

    def test_logprob_matches_fp32_model_logits_for_the_single_sampled_step(self):
        result = self.backend.sample(_prepared(self.descriptors))
        self.assertEqual(len(result.generated_token_ids), 1)
        with self.torch.inference_mode():
            logits = (
                self.model(
                    input_ids=self.torch.tensor([result.input_token_ids], dtype=self.torch.long)
                )
                .logits[0, -1]
                .float()
            )
            expected = self.torch.log_softmax(logits, dim=-1)[result.generated_token_ids[0]]
        recorded = struct.unpack("<f", result.logprobs.data)[0]
        self.assertLessEqual(abs(recorded - float(expected)), 1e-5)
        self.assertEqual(result.termination["kind"], "native_stop")
        self.assertEqual(result.usage["prefill_tokens"], len(result.input_token_ids))
        self.assertEqual(result.usage["cached_input_tokens"], 0)

    def test_generation_keeps_no_past_key_values_between_sample_calls(self):
        original = self.model.generate
        caches = []
        calls = []

        def capture(*args, **kwargs):
            self.assertNotIn("past_key_values", kwargs)
            self.assertNotIn("cache", kwargs)
            self.assertTrue(kwargs["use_cache"])
            self.assertEqual(kwargs["cache_implementation"], "dynamic")
            calls.append(kwargs["max_new_tokens"])
            result = original(*args, **kwargs)
            cache = getattr(result, "past_key_values", None)
            if cache is not None:
                caches.append(weakref.ref(cache))
            return result

        with patch.object(self.model, "generate", side_effect=capture):
            first = self.backend.sample(_prepared(self.descriptors, ordinal=0))
            second = self.backend.sample(_prepared(self.descriptors, ordinal=1))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls, [1, 1])
        self.assertEqual(first.input_token_ids, second.input_token_ids)
        self.assertTrue(all(cache() is None for cache in caches))
        self.assertNotIn("past_key_values", vars(self.backend))
        self.assertNotIn("cache", vars(self.backend))

    def test_disabled_adapter_is_rejected(self):
        with self.model.disable_adapter():
            with self.assertRaises(AdapterContractError):
                self.backend.sample(_prepared(self.descriptors))

    def test_context_exhaustion_records_a_zero_generation_turn(self):
        result = self.backend.sample(_prepared(self.descriptors, max_context=0))
        self.assertEqual(result.generated_token_ids, ())
        self.assertEqual(result.logprobs.data, b"")
        self.assertEqual(
            result.termination,
            {"kind": "context_limit", "stop_token_id": None, "limit": "context"},
        )

    def test_torch_and_transformers_import_only_when_sampling(self):
        source_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(source_root / "src"))
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, writing_agent.native_gemma; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules",
            ],
            check=True,
            cwd=source_root,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_tiny_gemma_samples_three_turns_through_the_v2_rollout_driver(self):
        from tests.task_graph_rollout_fixtures import (
            _entry_fixture,
            build_rollout_fixture,
            make_gatherers,
        )
        from tests.test_task_graph_v2_writer import _native_policy
        from writing_agent.task_graph_composition import RuntimeSession
        from writing_agent.task_graph_gate import StoreArtifactReader
        from writing_agent.task_graph_group import GroupCoordinatorV1
        from writing_agent.task_graph_local import (
            DeterministicEvaluator,
            LocalTextToolProvider,
            LocalWorkspaceEnvironment,
        )
        from writing_agent.task_graph_records import MemberStartV1
        from writing_agent.task_graph_rollout import RolloutDriver

        tokenizer_record = self.descriptors[1]
        entry = _entry_fixture(
            "triple_feedback",
            "deterministic-file-v1",
            rendering_overrides={"tokenizer_ref": tokenizer_record.identity()},
            public_records=(tokenizer_record,),
        )
        descriptors = _descriptors(
            self.tokenizer,
            template_ref=entry.params.rendering["template_ref"],
            tool_schema_ref=entry.params.rendering["tool_schema_ref"],
            max_tokens=1,
        )
        backend = NativeGemmaSampleBackend(
            self.model,
            self.tokenizer,
            manifest_descriptors=descriptors,
        )
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_rollout_fixture(
                Path(tmp) / "native-rollout",
                mode="triple_feedback",
                entry_fixture=entry,
            )
            tools = LocalTextToolProvider()
            dependencies = RuntimeDependenciesV1(
                backend, LocalWorkspaceEnvironment(tools), tools, DeterministicEvaluator()
            )
            self.assertIsInstance(dependencies.manifest(), RuntimeManifestV2)
            unbound = RuntimeSession.create(fixture.store, dependencies)
            fixture.env.session = unbound.bind(fixture.store, unbound.manifest_ref)
            policy = _native_policy(
                fixture.store,
                fixture.runtime.context.rendering,
                dependencies.manifest(),
            )
            coordinator = GroupCoordinatorV1(fixture.env, session=fixture.env.session)
            spec = coordinator.seal(
                fixture.runtime.checkpoint_id,
                policy=policy,
                group_seed=17,
                group_sequence=93,
                member_count=2,
                runner_mode="fixture",
                training_mode="native",
            )
            runtime = fixture.env.start_member(
                fixture.runtime.checkpoint_id,
                MemberStartV1(spec.identity(), 0),
            )
            result = RolloutDriver(
                fixture.env,
                make_gatherers(fixture, sampler=backend),
            ).run(runtime, max_steps=80)
            self.assertEqual(result.directive.kind, "done")
            self.assertEqual(len(fixture.sampler_inputs), 3)
            view = fixture.env.verify(result.runtime)
            self.assertEqual(len(view.samples), 3)
            reader = StoreArtifactReader(fixture.store)
            turns = [
                WriterTurnV2.from_dict(reader.artifact(sample.turn_ref)) for sample in view.samples
            ]
            self.assertTrue(all(turn.termination["kind"] == "native_stop" for turn in turns))
            for index in range(1, len(turns)):
                prior = fixture.sampler_inputs[index].native_history
                self.assertIsNotNone(prior)
                prefix = prior.input_token_ids + prior.generated_token_ids
                data = reader.bytes_artifact(turns[index].input_token_ids_ref)
                input_ids = tuple(
                    struct.unpack_from("<I", data, offset)[0] for offset in range(0, len(data), 4)
                )
                self.assertEqual(input_ids[: len(prefix)], prefix)


if __name__ == "__main__":
    unittest.main()
