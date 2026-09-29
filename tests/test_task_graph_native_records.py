from __future__ import annotations

import struct
import unittest
from dataclasses import replace

from tests.test_task_graph_records import record_examples
from writing_agent import task_graph_ports, task_graph_records
from writing_agent.task_graph import canonical_json
from writing_agent.task_graph_composition import RuntimeSession
from writing_agent.task_graph_errors import AdapterContractError
from writing_agent.task_graph_ports import (
    BinaryLogprobEvidence,
    PortDescriptorV1,
    PreparedSamplingInput,
    SampleResultV2,
)
from writing_agent.task_graph_record_contracts import GroupSpecV1, group_identity
from writing_agent.task_graph_records import (
    DecodingDescriptorV1,
    RendererDescriptorV1,
    RuntimeManifestV1,
    RuntimeManifestV2,
    RuntimePortDescriptorV1,
    TokenizerDescriptorV1,
    TrainingAdmissionV1,
    WriterTurnV2,
)

H = "a" * 64
P = "b" * 64
Q = "c" * 64
REQUIRED_CAPABILITIES = ("native_token_ledger", "sampled_logprobs", "usage_reporting")


def native_descriptors(*, port_capabilities=REQUIRED_CAPABILITIES, processors=()):
    tokenizer = TokenizerDescriptorV1(
        model_id="tests/toy-tokenizer",
        revision="toy-r1",
        files_sha256={"tokenizer": P},
    )
    decoding = DecodingDescriptorV1(
        temperature=1,
        top_p=1,
        top_k=0,
        processors=processors,
        max_tokens_per_decision=64,
        seed_rule="writer_seed ⊕ action ordinal (sha256-domain-v1)",
        logprob_convention="log_softmax(model logits after model softcap), fp32",
        trainer_ratio="recomputed, num_iterations=1",
    )
    renderer = RendererDescriptorV1(
        implementation="gemma4-native-append-v1",
        template_ref=H,
        tokenizer_ref=tokenizer.identity(),
        tool_schema_ref=Q,
        stop_token_ids=(1, 106, 50),
        enable_thinking=False,
        suffix_rules_version="native-suffix-v1",
    )
    capabilities = tuple(sorted(REQUIRED_CAPABILITIES))
    port_capabilities = tuple(sorted(port_capabilities))
    ports = tuple(
        RuntimePortDescriptorV1(
            schema=1,
            role=role,
            implementation=f"tests.{role.title()}",
            version="1",
            configuration={"capabilities": port_capabilities} if role == "sampling" else {},
        )
        for role in ("sampling", "environment", "tools", "evaluator")
    )
    manifest = RuntimeManifestV2(
        schema=2,
        ports=ports,
        capabilities=capabilities,
        renderer=renderer,
        tokenizer=tokenizer,
        decoding=decoding,
    )
    policy = {
        "template_ref": renderer.template_ref,
        "tokenizer_ref": tokenizer.identity(),
        "decoding_ref": decoding.identity(),
        "adapter_ref": manifest.identity(),
    }
    rendering = {
        "template_ref": renderer.template_ref,
        "tokenizer_ref": renderer.tokenizer_ref,
        "tool_schema_ref": renderer.tool_schema_ref,
    }
    return tokenizer, decoding, renderer, manifest, policy, rendering


class FakeNativeSampleBackend:
    """Test-only sampler that chains toy-tokenizer inputs without model dependencies."""

    def __init__(self):
        self.descriptor = PortDescriptorV1(
            "sampling", "tests.FakeNativeSampleBackend", "1", capabilities=REQUIRED_CAPABILITIES
        )
        self.last_input: tuple[int, ...] = ()
        self.last_output: tuple[int, ...] = ()
        self.calls = 0

    @staticmethod
    def _toy_encode(text: str) -> tuple[int, ...]:
        return tuple(16 + byte for byte in text.encode("utf-8"))

    def sample(self, prepared: PreparedSamplingInput) -> SampleResultV2:
        suffix = self._toy_encode(prepared.messages_json)
        input_ids = self.last_input + self.last_output + suffix
        output_ids = (300 + self.calls, 1)
        self.calls += 1
        self.last_input, self.last_output = input_ids, output_ids
        return SampleResultV2(
            message={"content": "A synthetic native answer."},
            input_token_ids=input_ids,
            generated_token_ids=output_ids,
            usage={
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(output_ids),
                "prefill_tokens": len(input_ids),
                "cached_input_tokens": 0,
            },
            logprobs=BinaryLogprobEvidence(
                struct.pack("<ff", -0.25, -0.5), "f32-le", (len(output_ids),)
            ),
            termination={"kind": "native_stop", "stop_token_id": 1, "limit": None},
            sampling_pins={
                "manifest_ref": H,
                "behavior_policy_ref": P,
                "decoding_ref": Q,
                "renderer_ref": P,
                "seed": prepared.writer_seed,
            },
            trace={"model": "fake-native"},
        )


class NativeRecordTests(unittest.TestCase):
    def test_s3_record_and_port_types_exist(self):
        for name in (
            "WriterTurnV2",
            "RuntimeManifestV2",
            "RendererDescriptorV1",
            "TokenizerDescriptorV1",
            "DecodingDescriptorV1",
            "TrainingAdmissionV1",
        ):
            with self.subTest(record=name):
                self.assertTrue(hasattr(task_graph_records, name), name)
        self.assertTrue(hasattr(task_graph_ports, "SampleResultV2"))

    def test_training_eligibility_supports_structural_status(self):
        codec = task_graph_records.RECORD_TYPES["TrainingEligibilityV1"]
        self.assertIn("structurally_eligible", codec.fields.required["status"].values)
        self.assertIn(
            "structurally_eligible",
            task_graph_records.OutcomeV1.FIELD_SPEC.required["training_eligibility"].values,
        )
        outcome = next(
            record
            for record in record_examples()
            if isinstance(record, task_graph_records.OutcomeV1)
        )
        self.assertEqual(
            replace(outcome, training_eligibility="structurally_eligible").training_eligibility,
            "structurally_eligible",
        )
        body = {
            "record_type": "TrainingEligibilityV1",
            "schema": 1,
            "terminal_outcome_ref": H,
            "status": "structurally_eligible",
            "reason": "native_evidence_structural",
        }
        self.assertEqual(codec.from_dict(body), body)

    def test_native_records_round_trip_and_validate_local_shape(self):
        tokenizer, decoding, renderer, manifest, _policy, _rendering = native_descriptors()
        self.assertEqual(TokenizerDescriptorV1.from_dict(tokenizer.to_wire()), tokenizer)
        self.assertEqual(DecodingDescriptorV1.from_dict(decoding.to_wire()), decoding)
        self.assertEqual(RendererDescriptorV1.from_dict(renderer.to_wire()), renderer)
        self.assertEqual(RuntimeManifestV2.from_dict(manifest.to_wire()), manifest)

        admission = TrainingAdmissionV1(
            group_id=H,
            decision_ref=P,
            batch_ref=Q,
            audit_version="native-audit-v1",
            renderer_ref=renderer.identity(),
            tokenizer_descriptor_ref=tokenizer.identity(),
            adapter_hash_before=P,
            adapter_hash_after=Q,
            members=({"member_id": "grp-example-00", "status": "admitted", "failed_check": None},),
        )
        self.assertEqual(TrainingAdmissionV1.from_dict(admission.to_wire()), admission)

        turn = WriterTurnV2(
            action_id="writer:action:0",
            context_revision_ref=H,
            raw_output_ref=None,
            usage={"prompt_tokens": 3, "completion_tokens": 2},
            adapter_trace={"model": "fake"},
            message={"content": "synthetic", "tool_calls_was_list": False, "calls": []},
            input_token_ids_ref=P,
            input_token_count=3,
            generated_token_ids_ref=Q,
            generated_token_count=2,
            logprobs={"ref": H, "codec": "f32-le", "shape": [2]},
            termination={"kind": "native_stop", "stop_token_id": 1, "limit": None},
            sampling_pins={
                "manifest_ref": manifest.identity(),
                "behavior_policy_ref": P,
                "decoding_ref": decoding.identity(),
                "renderer_ref": renderer.identity(),
                "seed": 31,
            },
        )
        self.assertEqual(WriterTurnV2.from_dict(turn.to_wire()), turn)
        self.assertEqual(canonical_json(turn.to_wire()), turn.to_json())

        bad_count = turn.to_wire()
        bad_count["generated_token_count"] = -1
        with self.assertRaises(ValueError):
            WriterTurnV2.from_dict(bad_count)
        bad_shape = turn.to_wire()
        bad_shape["logprobs"]["shape"] = [2, 1]
        with self.assertRaises(ValueError):
            WriterTurnV2.from_dict(bad_shape)
        bad_codec = turn.to_wire()
        bad_codec["logprobs"]["codec"] = "f64-le"
        with self.assertRaises(ValueError):
            WriterTurnV2.from_dict(bad_codec)

    def test_training_mode_is_omitted_when_absent_and_decoded_as_none(self):
        spec = next(record for record in record_examples() if isinstance(record, GroupSpecV1))
        wire = spec.to_wire()
        self.assertNotIn("training_mode", wire)
        self.assertIsNone(GroupSpecV1.from_dict(wire).training_mode)
        explicit_none = {**wire, "training_mode": None}
        with self.assertRaises(ValueError):
            GroupSpecV1.from_dict(explicit_none)

        native_id = group_identity(
            spec.group_sequence,
            spec.environment,
            spec.policy,
            spec.group_seed,
            spec.runner_mode,
            len(spec.members),
            "native",
        )
        native_members = tuple(
            replace(member, member_id=f"grp-{native_id[:24]}-{member.ordinal:02d}")
            for member in spec.members
        )
        native = replace(
            spec,
            group_id=native_id,
            members=native_members,
            training_mode="native",
        )
        self.assertNotEqual(native.identity(), spec.identity())
        self.assertEqual(GroupSpecV1.from_dict(native.to_wire()), native)

    def test_fake_native_backend_produces_a_chained_toy_token_ledger(self):
        backend = FakeNativeSampleBackend()
        prepared = PreparedSamplingInput(
            context_content_hash=H,
            context_revision_ref=P,
            writer_seed=41,
            model_ref=Q,
            behavior_policy_ref=P,
            decoding_ref=Q,
            tokenizer_ref=P,
            template_ref=Q,
            messages_json='[{"role":"user","content":"first"}]',
            tools_json="[]",
            rendering_json="{}",
        )
        first = backend.sample(prepared)
        second = backend.sample(prepared)
        self.assertEqual(
            second.input_token_ids[: len(first.input_token_ids) + len(first.generated_token_ids)],
            first.input_token_ids + first.generated_token_ids,
        )
        self.assertEqual(first.logprobs.codec, "f32-le")
        self.assertEqual(first.logprobs.shape, (first.usage["completion_tokens"],))
        self.assertEqual(first.sampling_pins["seed"], 41)

    def test_native_manifest_refusals_cover_each_missing_capability_and_pin(self):
        for missing in REQUIRED_CAPABILITIES:
            caps = tuple(
                capability for capability in REQUIRED_CAPABILITIES if capability != missing
            )
            *_, manifest, policy, rendering = native_descriptors(port_capabilities=caps)
            with self.subTest(manifest_missing=missing), self.assertRaises(ValueError):
                body = manifest.to_wire()
                body["capabilities"].remove(missing)
                RuntimeManifestV2.from_dict(body)
            with (
                self.subTest(sampling_port_missing=missing),
                self.assertRaises(AdapterContractError),
            ):
                task_graph_ports.require_native_manifest_binding(
                    manifest, manifest.identity(), policy=policy, rendering=rendering
                )
            self.assertIn(missing, manifest.capabilities)

        _tokenizer, _decoding, _renderer, manifest, policy, rendering = native_descriptors()
        for field in ("template_ref", "tokenizer_ref", "decoding_ref", "adapter_ref"):
            mismatched = {**policy, field: "d" * 64}
            with self.subTest(policy_field=field), self.assertRaises(AdapterContractError):
                task_graph_ports.require_native_manifest_binding(
                    manifest, manifest.identity(), policy=mismatched, rendering=rendering
                )
        for field in ("template_ref", "tokenizer_ref", "tool_schema_ref"):
            mismatched = {**rendering, field: "e" * 64}
            with self.subTest(rendering_field=field), self.assertRaises(AdapterContractError):
                task_graph_ports.require_native_manifest_binding(
                    manifest, manifest.identity(), policy=policy, rendering=mismatched
                )

    def test_native_binding_refuses_v1_and_non_neutral_processors(self):
        ports = tuple(
            RuntimePortDescriptorV1(
                schema=1,
                role=role,
                implementation=f"tests.{role.title()}",
                version="1",
                configuration={},
            )
            for role in ("sampling", "environment", "tools", "evaluator")
        )
        v1 = RuntimeManifestV1(schema=1, ports=ports)
        _tokenizer, _decoding, _renderer, _manifest, policy, rendering = native_descriptors()
        with self.assertRaises(AdapterContractError):
            task_graph_ports.require_native_manifest_binding(
                v1, H, policy=policy, rendering=rendering
            )
        v1_native_sampler = RuntimePortDescriptorV1(
            schema=1,
            role="sampling",
            implementation="tests.Sampler",
            version="1",
            configuration={"capabilities": ["native_token_ledger"]},
        )
        with self.assertRaises(ValueError):
            RuntimeManifestV1(
                schema=1,
                ports=(v1_native_sampler, *ports[1:]),
            )

        _tokenizer, _decoding, _renderer, processed, policy, rendering = native_descriptors(
            processors=("temperature-warper",)
        )
        with self.assertRaises(AdapterContractError):
            task_graph_ports.require_native_manifest_binding(
                processed, processed.identity(), policy=policy, rendering=rendering
            )

    def test_runtime_session_seal_checks_v2_context_and_group_pins(self):
        _tokenizer, _decoding, _renderer, manifest, policy, rendering = native_descriptors()

        class ManifestDependencies:
            def manifest(self):
                return manifest

        session = RuntimeSession(ManifestDependencies(), manifest.identity(), manifest.identity())
        session.require_seal(
            manifest.identity(),
            training_mode="native",
            policy=policy,
            rendering=rendering,
        )
        for field in ("template_ref", "tokenizer_ref", "decoding_ref"):
            with self.subTest(group_pin=field), self.assertRaises(AdapterContractError):
                session.require_seal(
                    manifest.identity(),
                    training_mode="native",
                    policy={**policy, field: "e" * 64},
                    rendering=rendering,
                )
        with self.assertRaises(AdapterContractError):
            session.require_seal(
                manifest.identity(),
                training_mode="native",
                policy=policy,
                rendering={**rendering, "tool_schema_ref": "f" * 64},
            )


if __name__ == "__main__":
    unittest.main()
