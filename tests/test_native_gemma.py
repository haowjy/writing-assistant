"""Native Gemma rendering, per-turn sampling, and V2 rollout integration."""

from __future__ import annotations

import hashlib
import importlib.util
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
    NativeGemmaRenderer,
    NativeGemmaSampleBackend,
    make_native_manifest_descriptors,
)
from writing_agent.task_graph import MessageV1, canonical_json
from writing_agent.task_graph_calls import intake_message
from writing_agent.task_graph_errors import AdapterContractError
from writing_agent.task_graph_native_contracts import NativeSamplingBudget, NativeSamplingHistory
from writing_agent.task_graph_ports import PreparedSamplingInput, RuntimeDependenciesV1
from writing_agent.task_graph_records import RuntimeManifestV2, WriterTurnV2

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


class NativeGemmaImportTests(unittest.TestCase):
    def test_torch_and_transformers_import_only_when_sampling(self):
        source_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(source_root / "src"))
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, writing_agent.native_gemma, writing_agent.native_audit; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules",
            ],
            check=True,
            cwd=source_root,
            env=env,
            capture_output=True,
            text=True,
        )


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft")),
    "requires optional torch, transformers, and peft model dependencies",
)
class NativeGemmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    def test_external_suffix_matches_tool_results_by_order_not_event_call_id(self):
        from writing_agent.inference import parse_response

        raw = (
            '<|tool_call>call:write_file{content:<|"|>x<|"|>,'
            'path:<|"|>scene.txt<|"|>}<tool_call|><|tool_response>'
        )
        generated = tuple(self.tokenizer.encode(raw, add_special_tokens=False))
        action_id = "member:action:0"
        parsed = parse_response(self.tokenizer, raw, prefix="")
        turn = WriterTurnV2(
            action_id=action_id,
            context_revision_ref="a" * 64,
            raw_output_ref=None,
            usage={
                "prompt_tokens": 1,
                "completion_tokens": len(generated),
                "total_tokens": 1 + len(generated),
                "prefill_tokens": 1,
                "cached_input_tokens": 0,
            },
            adapter_trace=None,
            message=intake_message(parsed),
            input_token_ids_ref="b" * 64,
            input_token_count=1,
            generated_token_ids_ref="c" * 64,
            generated_token_count=len(generated),
            logprobs={"ref": "d" * 64, "codec": "f32-le", "shape": [len(generated)]},
            termination={
                "kind": "native_stop",
                "stop_token_id": generated[-1],
                "limit": None,
            },
            sampling_pins={
                "manifest_ref": "e" * 64,
                "behavior_policy_ref": "f" * 64,
                "decoding_ref": self.descriptors[2].identity(),
                "renderer_ref": self.descriptors[0].identity(),
                "seed": 7,
            },
        )
        history = NativeSamplingHistory(turn, (1,), generated)
        messages = (
            MessageV1(
                role="assistant",
                content=(
                    {
                        "type": "tool_call",
                        "id": "call_0",
                        "name": "write_file",
                        "arguments": {"path": "scene.txt", "content": "x"},
                    },
                ),
                origin=action_id,
                loss_eligible=True,
            ),
            MessageV1(
                role="tool",
                content=(
                    {
                        "type": "tool_result",
                        "call_id": "member:call:0:0",
                        "content": {"ok": True, "result": "written"},
                    },
                ),
                origin=action_id,
            ),
        )

        suffix = NativeGemmaRenderer(self.tokenizer, self.descriptors[0]).external_suffix(
            history, messages
        )

        self.assertTrue(suffix)

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

    def test_native_batch_admission_offline_inspection_and_tamper_refusal(self):
        import base64
        import shutil
        from dataclasses import replace

        from tests.task_graph_fixtures import EntryFixture
        from tests.task_graph_rollout_fixtures import (
            AUTHOR_PACKET_CANARY,
            EVALUATOR_PACKET_CANARY,
            LEDGER_CANARY,
            _entry_fixture,
            build_rollout_fixture,
            make_gatherers,
        )
        from tests.test_task_graph_v2_writer import _native_policy
        from writing_agent.native_audit import (
            TrainingAuditError,
            audit_training_batch,
            inspect_group_offline,
            require_training_admission,
        )
        from writing_agent.task_graph import canonical_bytes, load_canonical_json, thaw
        from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
        from writing_agent.task_graph_composition import RuntimeSession
        from writing_agent.task_graph_derive_entry import derive_entry
        from writing_agent.task_graph_errors import StoreError
        from writing_agent.task_graph_gate import StoreArtifactReader
        from writing_agent.task_graph_group import GroupCoordinatorV1
        from writing_agent.task_graph_group_records import GroupMemberResultV1
        from writing_agent.task_graph_local import (
            DeterministicEvaluator,
            LocalTextToolProvider,
            LocalWorkspaceEnvironment,
        )
        from writing_agent.task_graph_ports import RuntimeDependenciesV1
        from writing_agent.task_graph_records import (
            RuntimeManifestV1,
            RuntimeManifestV2,
            RuntimePortDescriptorV1,
            WriterTurnV2,
        )
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
        for artifact in entry.artifacts:
            if artifact.kind == "context_revision":
                entry.reader.context_revisions[artifact.ref] = artifact.value.to_wire()
            elif artifact.kind == "context_node":
                entry.reader.context_nodes[artifact.ref] = artifact.value.to_wire()
        entry_context = entry.reader.context(entry.state.context_ref)
        _prompt, initial_ids = backend.renderer.render_initial(
            entry_context.messages, thaw(entry_context.tools)
        )
        context_cap = len(initial_ids) + 1
        entry_node = entry.graph.node(entry.node_id)
        capped_contract = replace(
            entry_node.contract,
            budgets=replace(
                entry_node.contract.budget_contract,
                max_context_tokens=context_cap,
            ),
        )
        entry.reader.public[capped_contract.identity()] = capped_contract.to_dict()
        capped_spec = replace(entry_node.spec, entry_contract=capped_contract.identity())
        capped_instance = replace(entry.graph.instance, nodes=(capped_spec,))
        capped_graph = admit_graph(
            capped_instance,
            MappingArtifactResolver(entry.reader.public, entry.reader.private),
            policy=entry.graph.policy,
        )
        capped_entry = derive_entry(capped_graph, entry.node_id, entry.params, entry.reader)
        entry = EntryFixture(
            graph=capped_graph,
            node_id=entry.node_id,
            params=entry.params,
            reader=entry.reader,
            state=capped_entry.state,
            artifacts=capped_entry.artifacts,
        )
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_rollout_fixture(
                Path(tmp) / "native-audit",
                mode="triple_feedback",
                entry_fixture=entry,
            )
            tools = LocalTextToolProvider()
            dependencies = RuntimeDependenciesV1(
                backend, LocalWorkspaceEnvironment(tools), tools, DeterministicEvaluator()
            )
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
                group_seed=117,
                group_sequence=940,
                member_count=2,
                runner_mode="real",
                training_mode="native",
            )
            for ordinal, member in enumerate(spec.members):
                runtime = coordinator.start(spec, ordinal, policy=policy)
                result = RolloutDriver(
                    fixture.env,
                    make_gatherers(fixture, sampler=backend),
                ).run(runtime, max_steps=80)
                self.assertEqual(result.directive.kind, "done")
                view = fixture.env.verify(result.runtime)
                outcome_ref = result.runtime.state.outcome_ref
                if view.outcome.reward_ref is not None:
                    outcome_ref = fixture.store.get_artifact(view.outcome.reward_ref)[
                        "terminal_outcome_ref"
                    ]
                coordinator.collect(
                    spec,
                    GroupMemberResultV1(
                        group_id=spec.group_id,
                        member_id=member.member_id,
                        start_checkpoint_id=coordinator._start_receipt(spec, ordinal)[
                            "start_checkpoint_id"
                        ],
                        final_checkpoint_id=result.runtime.checkpoint_id,
                        terminal_outcome_ref=outcome_ref,
                        availability_ref=view.outcome.reward_ref,
                        execution_status="valid",
                    ),
                )
            decision = coordinator.finalize(spec)
            self.assertIn(decision.status, {"ready", "tie"})
            admission = audit_training_batch(
                fixture.store,
                spec,
                decision,
                self.tokenizer,
                adapter_hash_before=policy["behavior_policy_ref"],
                adapter_hash_after=policy["behavior_policy_ref"],
                tokenizer_root=TOKENIZER_PATH,
            )
            self.assertEqual(
                [member["status"] for member in admission.members],
                ["admitted", "admitted"],
                admission.to_wire(),
            )
            require_training_admission(admission)
            batch = fixture.store.get_artifact(admission.batch_ref)
            self.assertEqual(batch["record_type"], "TrainingBatchV1")
            self.assertEqual(batch["max_context_tokens"], context_cap)
            self.assertTrue(
                all(member.get("trailing_context_limit_turn_ref") for member in batch["members"])
            )

            for sampler_input in fixture.sampler_inputs:
                encoded = repr(
                    (sampler_input.messages, sampler_input.tools, sampler_input.rendering)
                )
                for canary in (AUTHOR_PACKET_CANARY, EVALUATOR_PACKET_CANARY, LEDGER_CANARY):
                    self.assertNotIn(canary, encoded)
            private_safe = repr((batch, admission.to_wire()))
            for canary in (AUTHOR_PACKET_CANARY, EVALUATOR_PACKET_CANARY, LEDGER_CANARY):
                self.assertNotIn(canary, private_safe)

            report_1 = inspect_group_offline(fixture.store.root, spec.group_id)
            report_2 = inspect_group_offline(fixture.store.root, spec.group_id)
            self.assertEqual(report_1, report_2)
            self.assertEqual(
                (fixture.store.root / "groups" / spec.group_id / "inspection.json").read_bytes(),
                report_2,
            )
            self.assertNotIn(AUTHOR_PACKET_CANARY.encode(), report_1)
            self.assertNotIn(EVALUATOR_PACKET_CANARY.encode(), report_1)
            self.assertNotIn(LEDGER_CANARY.encode(), report_1)

            reader = StoreArtifactReader(fixture.store)
            first_result = GroupMemberResultV1.from_dict(
                reader.artifact(decision.member_result_refs[0])
            )
            first_view = fixture.env.gate.view(fixture.store, first_result.final_checkpoint_id)
            first_turn = WriterTurnV2.from_dict(reader.artifact(first_view.samples[0].turn_ref))
            second_turn = WriterTurnV2.from_dict(reader.artifact(first_view.samples[1].turn_ref))
            manifest = RuntimeManifestV2.from_dict(reader.artifact(spec.policy["adapter_ref"]))
            tamper_cases = (
                ("generated", first_turn.generated_token_ids_ref, "bytes"),
                ("external", second_turn.input_token_ids_ref, "bytes"),
                ("logprob", first_turn.logprobs["ref"], "bytes"),
                ("policy", spec.policy["behavior_policy_ref"], "json"),
                ("termination", first_view.samples[0].turn_ref, "termination"),
                ("v1_manifest", spec.policy["adapter_ref"], "manifest"),
            )
            # Store-layer controls: changing an envelope body breaks its content hash, so
            # LineageGate refuses before the tokenizer-backed admission audit can run.
            for label, ref, kind in tamper_cases:
                with self.subTest(tamper=label), tempfile.TemporaryDirectory() as tamper_tmp:
                    copied = Path(tamper_tmp) / "store"
                    shutil.copytree(fixture.store.root, copied)
                    target = copied / "artifacts" / ref
                    envelope = load_canonical_json(target.read_bytes())
                    if kind == "bytes":
                        raw = bytearray(base64.b64decode(envelope["body"]))
                        self.assertTrue(raw)
                        raw[0] ^= 1
                        envelope["body"] = base64.b64encode(raw).decode("ascii")
                    elif kind == "json":
                        envelope["body"] = {"policy": "tampered"}
                    elif kind == "termination":
                        envelope["body"]["termination"] = {
                            "kind": "token_limit",
                            "stop_token_id": None,
                            "limit": "decision",
                        }
                    else:
                        v1_ports = []
                        for port in manifest.ports:
                            configuration = dict(port.configuration)
                            if port.role == "sampling":
                                configuration["capabilities"] = ["usage_reporting"]
                            v1_ports.append(
                                RuntimePortDescriptorV1(
                                    schema=1,
                                    role=port.role,
                                    implementation=port.implementation,
                                    version=port.version,
                                    configuration=configuration,
                                )
                            )
                        envelope["body"] = RuntimeManifestV1(
                            schema=1,
                            ports=tuple(v1_ports),
                        ).to_wire()
                    target.write_bytes(canonical_bytes(envelope))
                    with self.assertRaises(StoreError):
                        inspect_group_offline(copied, spec.group_id)

            class PaddedContextLimitBackend:
                descriptor = backend.descriptor
                manifest_descriptors = backend.manifest_descriptors

                def sample(self, prepared):
                    from dataclasses import replace as replace_record

                    result = backend.sample(prepared)
                    if result.termination["kind"] != "context_limit":
                        return result
                    input_ids = (*result.input_token_ids, 0)
                    usage = dict(result.usage)
                    for key in ("prompt_tokens", "total_tokens", "prefill_tokens"):
                        usage[key] = len(input_ids)
                    return replace_record(result, input_token_ids=input_ids, usage=usage)

            padded_backend = PaddedContextLimitBackend()
            padded_session = RuntimeSession.create(
                fixture.store,
                RuntimeDependenciesV1(
                    padded_backend,
                    LocalWorkspaceEnvironment(tools),
                    tools,
                    DeterministicEvaluator(),
                ),
            )
            fixture.env.session = padded_session.bind(fixture.store, padded_session.manifest_ref)
            padded_coordinator = GroupCoordinatorV1(fixture.env, session=fixture.env.session)
            padded_spec = padded_coordinator.seal(
                fixture.runtime.checkpoint_id,
                policy=policy,
                group_seed=118,
                group_sequence=941,
                member_count=2,
                runner_mode="real",
                training_mode="native",
            )
            for ordinal, member in enumerate(padded_spec.members):
                runtime = padded_coordinator.start(padded_spec, ordinal, policy=policy)
                result = RolloutDriver(
                    fixture.env,
                    make_gatherers(fixture, sampler=padded_backend),
                ).run(runtime, max_steps=80)
                view = fixture.env.verify(result.runtime)
                outcome_ref = result.runtime.state.outcome_ref
                if view.outcome.reward_ref is not None:
                    outcome_ref = fixture.store.get_artifact(view.outcome.reward_ref)[
                        "terminal_outcome_ref"
                    ]
                padded_coordinator.collect(
                    padded_spec,
                    GroupMemberResultV1(
                        group_id=padded_spec.group_id,
                        member_id=member.member_id,
                        start_checkpoint_id=padded_coordinator._start_receipt(padded_spec, ordinal)[
                            "start_checkpoint_id"
                        ],
                        final_checkpoint_id=result.runtime.checkpoint_id,
                        terminal_outcome_ref=outcome_ref,
                        availability_ref=view.outcome.reward_ref,
                        execution_status="valid",
                    ),
                )
            padded_decision = padded_coordinator.finalize(padded_spec)
            padded_admission = audit_training_batch(
                fixture.store,
                padded_spec,
                padded_decision,
                self.tokenizer,
                adapter_hash_before=policy["behavior_policy_ref"],
                adapter_hash_after=policy["behavior_policy_ref"],
                tokenizer_root=TOKENIZER_PATH,
            )
            self.assertEqual(
                [member["failed_check"] for member in padded_admission.members],
                ["external_suffix_and_context_limit"] * 2,
            )
            self.assertEqual(
                fixture.store.get_artifact(padded_admission.identity()), padded_admission.to_wire()
            )
            with self.assertRaises(TrainingAuditError):
                require_training_admission(padded_admission)


if __name__ == "__main__":
    unittest.main()
