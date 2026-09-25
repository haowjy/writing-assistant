"""Independent runtime adapters cross the same sealed production boundaries."""

import unittest
from dataclasses import replace

from tests import test_task_graph_scripted as scripted_tests
from tests import test_task_graph_writer as writer_tests
from writing_agent.task_graph import EnvironmentStateV1, canonical_json, tree_hash
from writing_agent.task_graph_admission import StoreArtifactResolver, admit_graph
from writing_agent.task_graph_checks import DeterministicChecksV1
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_composition import RuntimeRunner, RuntimeSession
from writing_agent.task_graph_evaluation import EvaluationEvidenceV1
from writing_agent.task_graph_group import GroupCoordinatorV1
from writing_agent.task_graph_group_contract import POLICY_FIELDS
from writing_agent.task_graph_local import (
    DeterministicEvaluator,
    LocalTextToolProvider,
    LocalWorkspaceEnvironment,
    ScriptedSampleBackend,
)
from writing_agent.task_graph_ports import (
    EnvironmentResult,
    EnvironmentSnapshot,
    ExecutionInfrastructureError,
    PortDescriptorV1,
    RuntimeDependenciesV1,
    SampleResult,
    ToolManifest,
)
from writing_agent.task_graph_projection import project_writer_context
from writing_agent.task_graph_terminal import ScriptedTerminalV1
from writing_agent.task_graph_writer import TransactionalWriterV1


class IndependentBackend:
    descriptor = PortDescriptorV1("sampling", "independent-script-v1", "1")

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def sample(self, prepared):
        self.calls += 1
        assert prepared.messages_json and prepared.prepared_request_ref
        return SampleResult({"role": "assistant", "content": self.text}, raw_output=self.text)


class IndependentEnvironment:
    descriptor = PortDescriptorV1("environment", "in-memory-workspace-v1", "1")

    def __init__(self, schemas):
        self.schemas = schemas
        self.calls = 0
        self.infrastructure = "ok"

    def tool_manifest(self, allowlist, interaction_policy=None):
        return ToolManifest(canonical_json(self.schemas))

    def execute(self, spec, handle, snapshot, action):
        self.calls += 1
        if self.infrastructure != "ok":
            return EnvironmentResult({}, snapshot, self.infrastructure)
        files = snapshot.files()
        files["draft.txt"] = "independent environment state\n"
        return EnvironmentResult(
            {"ok": True, "valid": True, "result": "independent observation"},
            EnvironmentSnapshot.from_files(files),
        )


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


class RuntimePortsIntegrationTest(unittest.TestCase):
    def test_scripted_backend_descriptor_binds_its_samples(self):
        left = ScriptedSampleBackend([SampleResult({"role": "assistant", "content": "A"})])
        right = ScriptedSampleBackend([SampleResult({"role": "assistant", "content": "B"})])
        self.assertNotEqual(left.descriptor.identity(), right.descriptor.identity())

    def fixture(self, cls):
        fixture = cls()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def session(self, fixture, backend, *, environment=None, provider=None, evaluator=None):
        provider = provider or LocalTextToolProvider()
        dependencies = RuntimeDependenciesV1(
            backend,
            environment or LocalWorkspaceEnvironment(provider),
            provider,
            evaluator or DeterministicEvaluator(),
        )
        session = RuntimeSession.create(fixture.store, "rollout-1", fixture.start, dependencies)
        return session.bind(fixture.store, session.manifest_ref)

    def admit_fixture_family(self, fixture):
        original = fixture.writer.graph.node("legacy-writer")
        check = original.checks["nonempty"]
        fixture_check = replace(
            check,
            evaluator_version="fixture-file-count-v1",
            spec={
                "id": check.id,
                "metric": "Q1",
                "kind": "fixture_file_count",
                "method": "fixture",
                "required": True,
            },
        )
        fixture.store.put_artifact(fixture_check.to_dict(), private=True)
        contract = replace(original.contract, mandatory_checks=(fixture_check.identity(),))
        fixture.store.put_artifact(contract.to_dict())
        instance = fixture.writer.graph.instance
        node = replace(instance.nodes[0], entry_contract=contract.identity())
        instance = replace(instance, nodes=(node,))
        fixture.store.persist(instance)
        graph = admit_graph(instance, StoreArtifactResolver(fixture.store))
        state = fixture.runtime.state.to_dict()
        state["instance_ref"] = instance.identity()
        state["position"]["entry_contract"] = contract.identity()
        fixture.start = fixture.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        fixture.runtime = fixture.store.restore(
            fixture.start, fixture.root / "fixture-family-entry"
        )
        return graph

    def test_two_independent_sampling_backends_through_same_runner(self):
        backends = (
            ScriptedSampleBackend([SampleResult({"role": "assistant", "content": "local"})]),
            IndependentBackend("alternate"),
        )
        for backend in backends:
            with self.subTest(backend=type(backend).__name__):
                fixture = self.fixture(writer_tests.WriterFixture)
                session = self.session(fixture, backend)
                writer = TransactionalWriterV1(
                    fixture.store,
                    fixture.bundle.admission(),
                    "rollout-1",
                    fixture.start,
                    session=session,
                )
                result = RuntimeRunner(writer, session).sample(fixture.runtime)
                message = result.runtime.context.messages[-1]
                expected = "local" if isinstance(backend, ScriptedSampleBackend) else "alternate"
                self.assertEqual(message.content[0]["text"], expected)
                self.assertFalse(
                    fixture.store.get_artifact(
                        fixture.store.get_artifact(result.record_ref)["trace_ref"]
                    )["native_on_policy_eligible"]
                )

    def test_independent_environment_provider_and_evaluator_replay(self):
        fixture = self.fixture(scripted_tests.ScriptedFixture)
        graph = self.admit_fixture_family(fixture)
        schemas = fixture.runtime.context.tools
        environment = IndependentEnvironment(schemas)
        provider = IndependentProvider(schemas)
        evaluator = IndependentEvaluator()
        session = self.session(
            fixture,
            IndependentBackend("unused"),
            environment=environment,
            provider=provider,
            evaluator=evaluator,
        )
        writer = TransactionalWriterV1(
            fixture.store, graph, "rollout-1", fixture.start, session=session
        )
        action = writer.submit_action(
            fixture.runtime, fixture.action(fixture.call("read_file", {"path": "draft.txt"}))
        )
        observation = writer.step_tool(action.runtime)
        self.assertEqual(environment.calls, 1)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(
            observation.runtime.state.files["draft.txt"], "independent environment state\n"
        )
        self.assertEqual(
            fixture.store.get_artifact(observation.record_ref)["observation"]["result"],
            "independent observation",
        )
        final = writer.submit_action(observation.runtime, fixture.action(content="Done."))
        checks = DeterministicChecksV1(writer)
        requested = checks.request_checks(final.runtime)
        checked = checks.check_next(requested.runtime)
        self.assertEqual(evaluator.calls, 1)
        project_writer_context(fixture.store, fixture.start, checked.runtime.checkpoint_id)
        terminal = ScriptedTerminalV1(writer)
        outcome = terminal.terminal_outcome(checked.runtime)
        reward = terminal.reward(outcome.runtime)
        self.assertEqual(
            fixture.store.get_artifact(outcome.runtime.state.outcome_ref)["task_status"],
            "incomplete",
        )
        project_writer_context(fixture.store, fixture.start, reward.runtime.checkpoint_id)
        # A provider can replace the local tool implementation independently.
        fixture2 = self.fixture(writer_tests.WriterFixture)
        provider2 = IndependentProvider(fixture2.runtime.context.tools)
        session2 = self.session(fixture2, IndependentBackend("unused"), provider=provider2)
        writer2 = TransactionalWriterV1(
            fixture2.store,
            fixture2.bundle.admission(),
            "rollout-1",
            fixture2.start,
            session=session2,
        )
        action2 = writer2.submit_action(
            fixture2.runtime, fixture2.action(fixture2.call("read_file", {"path": "draft.txt"}))
        )
        observation2 = writer2.step_tool(action2.runtime)
        self.assertEqual(provider2.calls, 1)
        self.assertEqual(
            fixture2.store.get_artifact(observation2.record_ref)["observation"]["result"],
            "independent provider observation",
        )

    def test_infrastructure_failure_interrupts_without_tool_effect(self):
        fixture = self.fixture(writer_tests.WriterFixture)
        environment = IndependentEnvironment(fixture.runtime.context.tools)
        environment.infrastructure = "transient_failure"
        session = self.session(fixture, IndependentBackend("unused"), environment=environment)
        writer = TransactionalWriterV1(
            fixture.store,
            fixture.bundle.admission(),
            "rollout-1",
            fixture.start,
            session=session,
        )
        action = writer.submit_action(
            fixture.runtime, fixture.action(fixture.call("read_file", {"path": "draft.txt"}))
        )
        head = fixture.store.read_head("rollout-1")
        with self.assertRaises(ExecutionInfrastructureError) as raised:
            writer.step_tool(action.runtime)
        self.assertEqual(raised.exception.classification, "transient_failure")
        self.assertEqual(fixture.store.read_head("rollout-1"), head)
        self.assertEqual(action.runtime.state.files, fixture.runtime.state.files)

    def test_sealed_manifest_mismatch_rejects_before_effect(self):
        fixture = self.fixture(scripted_tests.ScriptedFixture)
        backend_a = IndependentBackend("A")
        backend_b = ScriptedSampleBackend(())
        session_a = self.session(fixture, backend_a)
        session_b = self.session(fixture, backend_b)
        self.assertNotEqual(session_a.manifest_ref, session_b.manifest_ref)
        policy = {
            field: fixture.store.put_artifact({"pin": field})
            for field in POLICY_FIELDS
            if field != "rng_derivation_version"
        }
        policy.update(
            adapter_ref=session_a.manifest_ref,
            tokenizer_ref=fixture.runtime.context.rendering["tokenizer_ref"],
            template_ref=fixture.runtime.context.rendering["template_ref"],
            context_policy_ref=fixture.store.put_artifact(ContextPolicyV1("carry").to_dict()),
            rng_derivation_version="sha256-domain-v1",
        )
        spec = GroupCoordinatorV1(fixture.store, fixture.root / "group-a", session=session_a).seal(
            fixture.start, policy=policy, group_seed=7, group_sequence=0, member_count=2
        )
        heads_before = fixture.store.read_head("rollout-1")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            GroupCoordinatorV1(fixture.store, fixture.root / "group-b", session=session_b).start(
                spec, 0, policy=policy
            )
        self.assertEqual(fixture.store.read_head("rollout-1"), heads_before)
        self.assertIsNone(fixture.store.read_head(spec.members[0].member_id))
        self.assertEqual(backend_a.calls, 0)
        runtime = GroupCoordinatorV1(
            fixture.store, fixture.root / "group-a", session=session_a
        ).start(spec, 0, policy=policy)
        member_head = fixture.store.read_head(spec.members[0].member_id)
        unbound_writer = TransactionalWriterV1(
            fixture.store,
            fixture.writer.graph,
            spec.members[0].member_id,
            runtime.checkpoint_id,
        )
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            unbound_writer.submit_action(runtime, fixture.action(content="bypass"))
        self.assertEqual(fixture.store.read_head(spec.members[0].member_id), member_head)
        mismatched_member_session = RuntimeSession.create(
            fixture.store,
            spec.members[0].member_id,
            runtime.checkpoint_id,
            session_b.dependencies,
        ).bind(fixture.store, session_b.manifest_ref)
        mismatched_writer = TransactionalWriterV1(
            fixture.store,
            fixture.writer.graph,
            spec.members[0].member_id,
            runtime.checkpoint_id,
            session=mismatched_member_session,
        )
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            RuntimeRunner(mismatched_writer, mismatched_member_session).sample(runtime)
        self.assertEqual(fixture.store.read_head(spec.members[0].member_id), member_head)
        self.assertEqual(backend_b.calls, 0)
        member_session = RuntimeSession.create(
            fixture.store,
            spec.members[0].member_id,
            runtime.checkpoint_id,
            session_a.dependencies,
        ).bind(fixture.store, spec.policy["adapter_ref"])
        writer = TransactionalWriterV1(
            fixture.store,
            fixture.writer.graph,
            spec.members[0].member_id,
            runtime.checkpoint_id,
            session=member_session,
        )
        RuntimeRunner(writer, member_session).sample(runtime)
        self.assertEqual(backend_a.calls, 1)
        backend_a.descriptor = PortDescriptorV1("sampling", "mutated-backend", "1")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            writer.validate_runtime(runtime)


if __name__ == "__main__":
    unittest.main()
