"""Independent adapters cross the rollout environment and gatherer boundaries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.task_graph_rollout_fixtures import build_rollout_fixture, make_gatherers, run_slice
from writing_agent.task_graph import canonical_json, tree_hash
from writing_agent.task_graph_composition import RuntimeSession
from writing_agent.task_graph_errors import AdapterContractError, DriverBudgetError
from writing_agent.task_graph_evaluation import EvaluationEvidenceV1
from writing_agent.task_graph_group import GroupCoordinatorV1
from writing_agent.task_graph_local import (
    DeterministicEvaluator,
    LocalTextToolProvider,
    LocalWorkspaceEnvironment,
    ScriptedSampleBackend,
)
from writing_agent.task_graph_ports import (
    BinaryLogprobEvidence,
    EnvironmentResult,
    EnvironmentSnapshot,
    ExecutionInfrastructureError,
    PortDescriptorV1,
    RuntimeDependenciesV1,
    SampleResult,
    ToolManifest,
)
from writing_agent.task_graph_record_contracts import POLICY_FIELDS, ContextPolicyV1
from writing_agent.task_graph_records import WriterTurnV1


def _call(name, arguments, call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class IndependentBackend:
    descriptor = PortDescriptorV1("sampling", "independent-script-v1", "1")

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def sample(self, prepared):
        self.calls += 1
        self.prepared = prepared
        assert prepared.messages_json and prepared.context_revision_ref
        return SampleResult({"role": "assistant", "content": self.text}, raw_output=self.text)


class IndependentProvider:
    descriptor = PortDescriptorV1("tools", "independent-text-provider-v1", "1")

    def __init__(self, schemas):
        self.schemas = schemas
        self.calls = 0

    def tool_manifest(self, allowlist, interaction_policy=None):
        return ToolManifest(canonical_json(self.schemas))

    def execute(self, spec, snapshot, action):
        self.calls += 1
        files = snapshot.files()
        files["draft.txt"] = "independent environment state\n"
        return EnvironmentResult(
            {"ok": True, "valid": True, "result": "independent provider observation"},
            EnvironmentSnapshot.from_files(files),
        )


class IndependentEvaluator:
    descriptor = PortDescriptorV1(
        "evaluator", "fixture-file-count-v1", "1", '{"family":"fixture-file-count-v1"}'
    )
    family = "fixture-file-count-v1"

    def __init__(self):
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        files = request.files()
        return EvaluationEvidenceV1(
            self.family,
            "pass" if len(files) % 2 else "fail",
            {"tree_hash": tree_hash(files), "file_count": len(files), "rule": "odd-file-count-v1"},
        )


class TranscriptEvaluator:
    descriptor = PortDescriptorV1(
        "evaluator",
        "offline-transcript-v1",
        "1",
        '{"family":"transcript-review-v1"}',
    )
    family = "transcript-review-v1"

    def evaluate(self, request):
        return EvaluationEvidenceV1(
            self.family,
            "pass",
            {
                "transcript": [
                    {
                        "role": "request",
                        "payload": {
                            "target_checkpoint": request.target_checkpoint,
                            "check_contract_hash": request.check.identity(),
                            "evaluator_packet_ref": request.evaluator_packet_ref,
                            "candidate_tree_hash": tree_hash(request.files()),
                        },
                    },
                    {"role": "response", "status": "pass", "text": "Recorded opinion."},
                ],
                "declared_status": "pass",
            },
        )


class RuntimePortsIntegrationTest(unittest.TestCase):
    def fixture(self, **options):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return build_rollout_fixture(Path(temporary.name) / "rollout", **options)

    @staticmethod
    def input_record(fixture, runtime):
        event = fixture.store.load_event(runtime.state.history["head"])
        return WriterTurnV1.from_dict(fixture.store.get_artifact(event.payload_ref))

    @staticmethod
    def advance(fixture, runtime, *, max_steps=1):
        try:
            return fixture.driver().run(runtime, max_steps=max_steps).runtime
        except DriverBudgetError as exhausted:
            return exhausted.runtime

    @staticmethod
    def session(fixture, backend, *, provider=None, evaluator=None):
        provider = provider or LocalTextToolProvider()
        dependencies = RuntimeDependenciesV1(
            backend,
            LocalWorkspaceEnvironment(provider),
            provider,
            evaluator or DeterministicEvaluator(),
        )
        unbound = RuntimeSession.create(fixture.store, dependencies)
        return unbound.bind(fixture.store, unbound.manifest_ref)

    @staticmethod
    def group_policy(fixture, session, *, decoding_ref=None):
        store = fixture.store
        policy = {
            field: store.put_artifact({"pin": field})
            for field in POLICY_FIELDS
            if field != "rng_derivation_version"
        }
        policy.update(
            model_ref=store.put_artifact({"model_id": "group-model-v1"}),
            adapter_ref=session.manifest_ref,
            tokenizer_ref=fixture.runtime.context.rendering["tokenizer_ref"],
            template_ref=fixture.runtime.context.rendering["template_ref"],
            context_policy_ref=store.put_artifact(ContextPolicyV1("carry").to_wire()),
            rng_derivation_version="sha256-domain-v1",
        )
        if decoding_ref is not None:
            policy["decoding_ref"] = decoding_ref
        return policy

    def test_sampling_options_are_bound_only_through_pinned_decoding_ref(self):
        class Capture(ScriptedSampleBackend):
            def sample(self, prepared):
                self.prepared = prepared
                return super().sample(prepared)

        fixture = self.fixture(mode="none")
        backend = Capture([SampleResult({"role": "assistant", "content": "answer"})])
        session = self.session(fixture, backend)
        fixture.env.session = session
        decoding = {"seed": 13, "temperature": "0.37"}
        decoding_ref = fixture.store.put_artifact(decoding)
        policy = self.group_policy(fixture, session, decoding_ref=decoding_ref)
        coordinator = GroupCoordinatorV1(fixture.env, fixture.root / "workers", session=session)
        spec = coordinator.seal(
            fixture.runtime.checkpoint_id,
            policy=policy,
            group_seed=13,
            group_sequence=0,
            member_count=2,
        )
        member = coordinator.start(spec, 0, policy=policy)
        fixture.gatherers = make_gatherers(fixture, sampler=backend)

        run_slice(fixture, runtime=member)

        prepared = backend.prepared
        self.assertEqual(prepared.decoding_ref, decoding_ref)
        self.assertEqual(fixture.store.get_artifact(prepared.decoding_ref), decoding)
        self.assertIsInstance(json.loads(prepared.messages_json), list)

    def test_binary_logprobs_persist_and_gate_replay_without_sampling(self):
        class NativeShaped(IndependentBackend):
            def sample(self, prepared):
                self.prepared = prepared
                self.calls += 1
                return SampleResult(
                    {"role": "assistant", "content": "answer"},
                    usage={"completion_tokens": 2},
                    trace={"generated_token_ids": [7, 8]},
                    logprobs=BinaryLogprobEvidence(b"\x00\x00\x80?" * 2, "f32-le", (2,)),
                )

        fixture = self.fixture(mode="none")
        backend = NativeShaped("unused")
        fixture.gatherers = make_gatherers(fixture, sampler=backend)
        runtime = self.advance(fixture, fixture.runtime)
        turn = self.input_record(fixture, runtime)
        trace = turn.adapter_trace
        self.assertEqual(
            fixture.store.get_artifact(
                trace["per_token_logprobs_ref"], expected_domain="payload:bytes"
            ),
            b"\x00\x00\x80?" * 2,
        )
        self.assertEqual(trace["per_token_logprobs_codec"], "f32-le")
        self.assertEqual(trace["per_token_logprobs_shape"], (2,))
        self.assertEqual(backend.calls, 1)

        def disabled(_prepared):
            raise AssertionError("sampling called during gate replay")

        backend.sample = disabled
        reopened = fixture.env.open(runtime.checkpoint_id)
        self.assertEqual(reopened.state, runtime.state)
        self.assertEqual(backend.calls, 1)

    def test_misaligned_binary_logprobs_reject_before_publication(self):
        class Misaligned(IndependentBackend):
            def sample(self, prepared):
                self.calls += 1
                return SampleResult(
                    {"role": "assistant", "content": "answer"},
                    trace={"generated_token_ids": [7, 8]},
                    logprobs=BinaryLogprobEvidence(b"\x00\x00\x80?", "f32-le", (1,)),
                )

        fixture = self.fixture(mode="none")
        backend = Misaligned("unused")
        fixture.gatherers = make_gatherers(fixture, sampler=backend)
        head = fixture.store.read_head(fixture.lineage_id)
        with self.assertRaises(AdapterContractError):
            fixture.driver().run(fixture.runtime, max_steps=1)
        self.assertEqual(fixture.store.read_head(fixture.lineage_id), head)
        self.assertIsNone(head)

    def test_transcript_evidence_replays_after_evaluator_is_disabled(self):
        fixture = self.fixture(
            mode="none",
            evaluator_family="transcript-review-v1",
            sample_results=(SampleResult({"role": "assistant", "content": "Done."}),),
        )
        evaluator = TranscriptEvaluator()
        fixture.gatherers = make_gatherers(fixture, evaluator=evaluator)
        runtime = run_slice(fixture)

        def disabled(_request):
            raise AssertionError("live evaluator called during replay")

        evaluator.evaluate = disabled
        reopened = fixture.env.open(runtime.checkpoint_id)
        view = fixture.env.verify(reopened)
        self.assertEqual(view.state.position["phase"], "terminal")

    def test_scripted_backend_descriptor_binds_its_samples(self):
        left = ScriptedSampleBackend([SampleResult({"role": "assistant", "content": "A"})])
        right = ScriptedSampleBackend([SampleResult({"role": "assistant", "content": "B"})])
        self.assertNotEqual(left.descriptor.identity(), right.descriptor.identity())

    def test_independent_sampling_backends_use_the_same_gatherer(self):
        for backend, expected in (
            (
                ScriptedSampleBackend([SampleResult({"role": "assistant", "content": "local"})]),
                "local",
            ),
            (IndependentBackend("alternate"), "alternate"),
        ):
            with self.subTest(backend=type(backend).__name__):
                fixture = self.fixture(mode="none")
                fixture.gatherers = make_gatherers(fixture, sampler=backend)
                runtime = run_slice(fixture)
                self.assertEqual(runtime.context.messages[-1].content[0]["text"], expected)

    def test_independent_tool_provider_and_evaluator_replay(self):
        fixture = self.fixture(
            mode="slice",
            evaluator_family="fixture-file-count-v1",
            sample_results=(
                SampleResult(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            _call(
                                "write_file",
                                {"path": "draft.txt", "content": "independent environment state\n"},
                            )
                        ],
                    }
                ),
                SampleResult({"role": "assistant", "content": "Done."}),
            ),
        )
        provider = IndependentProvider(fixture.runtime.context.tools)
        evaluator = IndependentEvaluator()
        fixture.gatherers = make_gatherers(fixture, tools=provider, evaluator=evaluator)
        runtime = run_slice(fixture)
        view = fixture.env.verify(runtime)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(evaluator.calls, 1)
        self.assertEqual(view.state.files["draft.txt"], "independent environment state\n")

        def disabled(*_args):
            raise AssertionError("producer port called during gate replay")

        provider.execute = disabled
        evaluator.evaluate = disabled
        self.assertEqual(fixture.env.verify(fixture.env.open(runtime.checkpoint_id)), view)

    def test_tool_infrastructure_failure_leaves_the_published_head_unmoved(self):
        class FailingProvider(IndependentProvider):
            def execute(self, spec, snapshot, action):
                self.calls += 1
                raise ExecutionInfrastructureError("transient_failure")

        fixture = self.fixture(
            mode="slice",
            sample_results=(
                SampleResult(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            _call(
                                "write_file",
                                {"path": "draft.txt", "content": "not written"},
                            )
                        ],
                    }
                ),
            ),
        )
        provider = FailingProvider(fixture.runtime.context.tools)
        fixture.gatherers = make_gatherers(fixture, tools=provider)
        runtime = self.advance(fixture, fixture.runtime)
        published = fixture.store.read_head(fixture.lineage_id)
        self.assertIsNotNone(published)
        with self.assertRaises(ExecutionInfrastructureError) as raised:
            fixture.driver().run(runtime, max_steps=1)
        self.assertEqual(raised.exception.classification, "transient_failure")
        self.assertEqual(fixture.store.read_head(fixture.lineage_id), published)
        self.assertEqual(runtime.state.files, fixture.runtime.state.files)

    def test_group_sampling_seal_is_enforced_by_the_new_core(self):
        fixture = self.fixture(
            mode="none",
            sample_results=(SampleResult({"role": "assistant", "content": "answer"}),),
        )
        backend_a = ScriptedSampleBackend([SampleResult({"role": "assistant", "content": "A"})])
        backend_b = ScriptedSampleBackend([SampleResult({"role": "assistant", "content": "B"})])
        session_a = self.session(fixture, backend_a)
        session_b = self.session(fixture, backend_b)
        fixture.env.session = session_a
        policy = self.group_policy(fixture, session_a)
        coordinator = GroupCoordinatorV1(fixture.env, fixture.root / "group-a", session=session_a)
        spec = coordinator.seal(
            fixture.runtime.checkpoint_id,
            policy=policy,
            group_seed=7,
            group_sequence=0,
            member_count=2,
        )

        wrong = GroupCoordinatorV1(fixture.env, fixture.root / "group-b", session=session_b)
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            wrong.start(spec, 0, policy=policy)
        self.assertIsNone(fixture.store.read_head(spec.members[0].member_id))
        self.assertEqual(backend_a.calls, 0)
        self.assertEqual(backend_b.calls, 0)

        member = coordinator.start(spec, 0, policy=policy)
        fixture.gatherers = make_gatherers(fixture, sampler=backend_a)
        runtime = run_slice(fixture, runtime=member)
        self.assertEqual(backend_a.calls, 1)
        backend_a.descriptor = PortDescriptorV1("sampling", "mutated-backend", "1")
        with self.assertRaises(AdapterContractError):
            fixture.env.verify(runtime)


if __name__ == "__main__":
    unittest.main()
