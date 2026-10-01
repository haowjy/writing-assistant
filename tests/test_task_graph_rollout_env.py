"""Producer, persistence, privacy, and recovery contracts for the new environment."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

from tests.task_graph_fixtures import make_entry_fixture
from tests.task_graph_rollout_env_support import (
    CANARY,
    EVENT_KINDS,
    _assert_fault_matrix,
    _author_check_fixture,
    _foreign_context_revision,
    _group_spec,
    _persist_fixture,
)
from tests.test_task_graph_derive_author import _answered_reply
from tests.test_task_graph_derive_writer import call, make_turn
from writing_agent.task_graph import (
    EventV1,
    MaterializedContextV1,
    domain_hash,
    load_canonical_json,
)
from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_composition import RuntimeSession
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_environment import (
    AuthorInput,
    CheckInput,
    RolloutEnvironment,
    RuntimeHandle,
    SamplerInput,
    ToolInput,
)
from writing_agent.task_graph_errors import (
    AdapterContractError,
    ConcurrentUpdateError,
    CorruptRecordError,
    DriverBudgetError,
    MissingReferenceError,
    ProjectionError,
    WriterRuntimeError,
)
from writing_agent.task_graph_gate import DERIVE, LineageGate, StoreArtifactReader
from writing_agent.task_graph_gatherers import (
    CheckRunner,
    Gatherers,
    SamplingRunner,
    ScriptedAuthorSource,
    ToolRunner,
)
from writing_agent.task_graph_local import (
    DeterministicEvaluator,
    LocalTextToolProvider,
    LocalWorkspaceEnvironment,
    ScriptedSampleBackend,
)
from writing_agent.task_graph_ports import PortDescriptorV1, RuntimeDependenciesV1, SampleResult
from writing_agent.task_graph_records import (
    AdmissionPolicyV1 as AdmissionPolicyRecord,
)
from writing_agent.task_graph_records import (
    ContextOperationInputV1,
    EnvironmentStepV1,
    MemberStartV1,
    ToolObservationV1,
)
from writing_agent.task_graph_rollout import RolloutDriver
from writing_agent.task_graph_store import TaskGraphStore


class RolloutEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = make_entry_fixture()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gate = LineageGate()
        self.store = TaskGraphStore(self.root / "store", verifier=self.gate)
        self.entry = _persist_fixture(self.store, self.fixture)
        self.environment = RolloutEnvironment(
            self.store, self.fixture.graph, None, self.gate, self.fixture.graph.policy
        )
        self.runtime = self.environment.open(self.entry)

    def test_with_session_returns_a_separate_environment_binding(self):
        session = object()

        bound = self.environment.with_session(session)

        self.assertIsNot(bound, self.environment)
        self.assertIs(bound.session, session)
        self.assertIsNone(self.environment.session)
        self.assertIs(bound.store, self.environment.store)
        self.assertIs(bound.graph, self.environment.graph)
        self.assertIs(bound.gate, self.environment.gate)

    def test_real_runtime_session_seals_are_checked_against_state(self):
        tools = LocalTextToolProvider()
        backend = ScriptedSampleBackend(())
        dependencies = RuntimeDependenciesV1(
            backend,
            LocalWorkspaceEnvironment(tools),
            tools,
            DeterministicEvaluator(),
        )
        unbound = RuntimeSession.create(self.store, dependencies)
        session = unbound.bind(self.store, unbound.manifest_ref)
        environment = RolloutEnvironment(
            self.store,
            self.fixture.graph,
            session,
            self.gate,
            self.fixture.graph.policy,
        )
        runtime = environment.open(self.entry)
        self.assertIsInstance(environment.step_input(runtime)[2], SamplerInput)

        backend.descriptor = PortDescriptorV1("sampling", "altered-backend", "1")
        with self.assertRaises(AdapterContractError):
            environment.verify(runtime)

    def test_group_member_session_seal_uses_the_verified_view_manifest(self):
        tools = LocalTextToolProvider()
        backend = ScriptedSampleBackend(())
        backend.descriptor = PortDescriptorV1("sampling", "group-runtime-a", "1")
        dependencies = RuntimeDependenciesV1(
            backend,
            LocalWorkspaceEnvironment(tools),
            tools,
            DeterministicEvaluator(),
        )
        unbound = RuntimeSession.create(self.store, dependencies)
        session = unbound.bind(self.store, unbound.manifest_ref)
        environment = RolloutEnvironment(
            self.store,
            self.fixture.graph,
            session,
            self.gate,
            self.fixture.graph.policy,
        )
        model_ref = self.store.put_artifact({"model_id": "group-runtime-model"})
        spec = _group_spec(
            self.fixture,
            self.entry,
            self.store,
            model_ref=model_ref,
            adapter_ref=session.manifest_ref,
        )
        runtime = environment.start_member(
            self.entry,
            MemberStartV1(group_spec_ref=spec.identity(), ordinal=0),
        )
        self.assertEqual(environment.verify(runtime).group, spec)

        other_tools = LocalTextToolProvider()
        other_backend = ScriptedSampleBackend(())
        other_backend.descriptor = PortDescriptorV1("sampling", "group-runtime-b", "1")
        other_dependencies = RuntimeDependenciesV1(
            other_backend,
            LocalWorkspaceEnvironment(other_tools),
            other_tools,
            DeterministicEvaluator(),
        )
        other_unbound = RuntimeSession.create(self.store, other_dependencies)
        other_session = other_unbound.bind(self.store, other_unbound.manifest_ref)
        other_environment = RolloutEnvironment(
            self.store,
            self.fixture.graph,
            other_session,
            self.gate,
            self.fixture.graph.policy,
        )
        with self.assertRaises(AdapterContractError):
            other_environment.open_head(spec.members[0].member_id)

    def _alternate_policy_entry(self, lineage: str):
        policy = replace(
            self.fixture.graph.policy,
            allowed_tools=self.fixture.graph.policy.allowed_tools - {"ask_author"},
        )
        graph = admit_graph(
            self.fixture.graph.instance,
            MappingArtifactResolver(self.fixture.reader.public, self.fixture.reader.private),
            policy=policy,
        )
        policy_ref = self.store.put_artifact(
            AdmissionPolicyRecord.from_admission_policy(policy).to_wire()
        )
        versions = self.store.get_artifact(self.fixture.state.versions_ref)
        versions["admission_policy_ref"] = policy_ref
        versions_ref = self.store.put_artifact(versions)
        params = replace(
            self.fixture.params,
            lineage_id=lineage,
            visit_id=lineage,
            versions_ref=versions_ref,
        )
        return graph, policy, params

    def _enter_alternate_policy(self, lineage: str):
        graph, policy, params = self._alternate_policy_entry(lineage)
        environment = RolloutEnvironment(self.store, graph, None, self.gate, policy)
        runtime = environment.enter(self.fixture.node_id, params)
        return environment, runtime

    def test_enter_rejects_entry_params_with_a_different_admission_policy(self):
        _, _, params = self._alternate_policy_entry("policy-enter")
        with self.assertRaises(ProjectionError):
            self.environment.enter(
                self.fixture.node_id,
                params,
            )
        self.assertIsNone(self.store.read_head("policy-enter"))

    def test_open_rejects_a_lineage_pinned_to_a_different_admission_policy(self):
        _, runtime = self._enter_alternate_policy("policy-open")
        with self.assertRaises(WriterRuntimeError):
            self.environment.open(runtime.checkpoint_id)

    def test_verify_rejects_a_lineage_pinned_to_a_different_admission_policy(self):
        _, runtime = self._enter_alternate_policy("policy-verify")
        with self.assertRaises(WriterRuntimeError):
            self.environment.verify(runtime)

    def test_writer_and_tool_commits_close_without_artifact_refs_and_fold_once(self):
        view = self.environment.verify(self.runtime)
        self.assertEqual(next_step(view).kind, "sample_writer")
        counts = {"writer": 0, "tool": 0}
        writer, tool = DERIVE["WriterTurnV1"], DERIVE["ToolObservationV1"]

        def count_writer(*args):
            counts["writer"] += 1
            return writer(*args)

        def count_tool(*args):
            counts["tool"] += 1
            return tool(*args)

        with patch.dict(DERIVE, {"WriterTurnV1": count_writer, "ToolObservationV1": count_tool}):
            self.environment.verify(self.runtime)
            self.assertEqual(counts["writer"], 0)
            turn = make_turn(
                view,
                content="",
                calls=(call("read_file", {"path": "draft.txt"}, "read-first"),),
            )
            _assert_fault_matrix(
                self,
                self.environment,
                self.runtime,
                turn,
                self.root,
                event_kind="writer_action",
            )
            counts["writer"] = 0
            first = self.environment.commit(self.runtime, turn)
            self.assertEqual(counts["writer"], 2)
            first_checkpoint = self.store.load_checkpoint(first.runtime.checkpoint_id)
            self.assertEqual(first_checkpoint.artifact_refs, ())

            before_verify = counts.copy()
            self.environment.verify(first.runtime)
            self.assertEqual(counts, before_verify)
            port = self.environment.step_input(first.runtime)[2]
            self.assertIsInstance(port, ToolInput)
            queued = port.queue_entry
            observation = ToolObservationV1(
                call_id=queued["call_id"],
                dispatch={
                    "spec": {
                        "max_file_bytes": port.tool_spec.max_file_bytes,
                        "max_workspace_bytes": port.tool_spec.max_workspace_bytes,
                    },
                    "observation": {"ok": True, "valid": True, "result": "alpha"},
                    "effect": {},
                },
            )
            _assert_fault_matrix(
                self,
                self.environment,
                first.runtime,
                observation,
                self.root,
                event_kind="tool_result",
            )
            counts["tool"] = 0
            second = self.environment.commit(first.runtime, observation)
            self.assertEqual(counts["tool"], 2)
            second_checkpoint = self.store.load_checkpoint(second.runtime.checkpoint_id)
            self.assertEqual(second_checkpoint.artifact_refs, ())
            self.assertEqual(
                self.environment.verify(second.runtime).checkpoint_id,
                second.runtime.checkpoint_id,
            )

    def test_context_operation_commits_through_gate(self):
        policy = ContextPolicyV1("drop")
        policy_ref = self.store.put_artifact(policy.to_wire())
        operation = ContextOperationInputV1(policy_ref=policy_ref)
        _assert_fault_matrix(
            self,
            self.environment,
            self.runtime,
            operation,
            self.root,
            event_kind="context_changed",
        )
        result = self.environment.commit(self.runtime, operation)
        self.assertEqual(result.runtime.state.history["seq"], 1)
        self.assertEqual(self.store.load_checkpoint(result.runtime.checkpoint_id).artifact_refs, ())
        self.assertEqual(
            self.environment.verify(result.runtime).checkpoint_id,
            result.runtime.checkpoint_id,
        )

    def test_author_check_transition_and_reward_inputs_commit(self):
        from writing_agent.task_graph_evaluation import (
            FAMILIES,
            EvaluationRequestV1,
            produce_evaluation_evidence,
        )
        from writing_agent.task_graph_records import EvaluatorResultV1

        fixture = _author_check_fixture()
        root = self.root / "author-check"
        root.mkdir()
        root.chmod(0o700)
        gate = LineageGate()
        store = TaskGraphStore(root / "store", verifier=gate)
        entry = _persist_fixture(store, fixture)
        env = RolloutEnvironment(store, fixture.graph, None, gate, fixture.graph.policy)
        runtime = env.open(entry)

        ask = call(
            "ask_author",
            {
                "question": "Which door?",
                "decision_ids": ["pick"],
                "proposals": [{"id": "p1", "text": "The blue door"}],
                "option_refs": ["p1"],
            },
            "ask-door",
        )
        ask_turn = make_turn(env.verify(runtime), content="", calls=(ask,))
        _assert_fault_matrix(self, env, runtime, ask_turn, root, event_kind="writer_action")
        writer_step = env.commit(runtime, ask_turn)
        self.assertEqual(store.load_event(writer_step.event_id).kind, "writer_action")
        request_author = EnvironmentStepV1(
            directive={"kind": "request_author", "source": "writer_request"}
        )
        _assert_fault_matrix(
            self,
            env,
            writer_step.runtime,
            request_author,
            root,
            event_kind="external_requested",
        )
        requested = env.commit(writer_step.runtime, request_author)
        author_view = env.verify(requested.runtime)
        author_port = env.step_input(requested.runtime)[2]
        self.assertIsInstance(author_port, AuthorInput)
        request_ref = requested.runtime.state.continuation["author_request"]
        fixture.reader.private[request_ref] = env.reader.artifact(request_ref, private=True)
        author_reply = _answered_reply(fixture, author_view)
        _assert_fault_matrix(
            self, env, requested.runtime, author_reply, root, event_kind="author_turn"
        )
        replied = env.commit(requested.runtime, author_reply)
        self.assertEqual(store.load_event(replied.event_id).kind, "author_turn")

        final_turn = make_turn(env.verify(replied.runtime), content="Done.")
        _assert_fault_matrix(
            self, env, replied.runtime, final_turn, root, event_kind="writer_action"
        )
        final = env.commit(replied.runtime, final_turn)
        self.assertEqual(final.directive.kind, "request_checks")
        request_checks = EnvironmentStepV1(directive={"kind": "request_checks"})
        _assert_fault_matrix(
            self,
            env,
            final.runtime,
            request_checks,
            root,
            event_kind="external_requested",
        )
        requested_checks = env.commit(final.runtime, request_checks)
        check_port = env.step_input(requested_checks.runtime)[2]
        self.assertIsInstance(check_port, CheckInput)
        check = fixture.graph.node(fixture.node_id).checks[check_port.request["check_id"]]
        family = next(
            item for item in FAMILIES.values() if item.check_version == check.evaluator_version
        )
        request_ref = requested_checks.runtime.state.continuation["check_requests"][0]
        packet_ref = check_port.request["evaluator_packet_ref"]
        evaluation = EvaluationRequestV1.create(
            family.name,
            check_port.request["target_checkpoint"],
            check,
            packet_ref,
            check_port.files,
            evaluator_packet=check_port.evaluator_packet,
        )
        evidence = produce_evaluation_evidence(evaluation).to_wire(evaluation)
        evidence_ref = store.put_artifact(evidence)
        evaluation_result = EvaluatorResultV1(
            request_ref=request_ref,
            status="pass",
            evidence_ref=evidence_ref,
        )
        _assert_fault_matrix(
            self,
            env,
            requested_checks.runtime,
            evaluation_result,
            root,
            event_kind="check_recorded",
        )
        checked = env.commit(requested_checks.runtime, evaluation_result)
        self.assertEqual(store.load_event(checked.event_id).kind, "check_recorded")

        view = env.verify(checked.runtime)
        edge = next_step(view).edge_id
        transition_input = EnvironmentStepV1(
            directive={"kind": "commit_transition", "edge_id": edge}
        )
        _assert_fault_matrix(
            self,
            env,
            checked.runtime,
            transition_input,
            root,
            event_kind="transition_committed",
        )
        transitioned = env.commit(checked.runtime, transition_input)
        self.assertEqual(store.load_event(transitioned.event_id).kind, "transition_committed")
        seal_input = EnvironmentStepV1(
            directive={
                "kind": "seal_outcome",
                "task_status": "complete",
                "stop_reason": None,
            }
        )
        _assert_fault_matrix(
            self,
            env,
            transitioned.runtime,
            seal_input,
            root,
            event_kind="termination_recorded",
        )
        sealed = env.commit(transitioned.runtime, seal_input)
        self.assertEqual(store.load_event(sealed.event_id).kind, "termination_recorded")
        reward_input = EnvironmentStepV1(directive={"kind": "publish_reward"})
        _assert_fault_matrix(
            self,
            env,
            sealed.runtime,
            reward_input,
            root,
            event_kind="reward_recorded",
        )
        rewarded = env.commit(sealed.runtime, reward_input)
        self.assertEqual(store.load_event(rewarded.event_id).kind, "reward_recorded")
        self.assertEqual(env.verify(rewarded.runtime).checkpoint_id, rewarded.runtime.checkpoint_id)

    def test_port_allowlists_and_private_canaries(self):
        fixture = _author_check_fixture()
        root = self.root / "privacy"
        root.mkdir()
        root.chmod(0o700)
        gate = LineageGate()
        store = TaskGraphStore(root / "store", verifier=gate)
        entry = _persist_fixture(store, fixture)
        env = RolloutEnvironment(store, fixture.graph, None, gate, fixture.graph.policy)
        runtime = env.open(entry)

        node = fixture.graph.node(fixture.node_id)
        self.assertIn(CANARY, repr(node.author_packet))
        self.assertIn(CANARY, repr(node.evaluator_packet))
        private_ledger = store.get_artifact(runtime.state.requirements_ref, private=True)
        self.assertIn(CANARY, repr(private_ledger))
        view = env.verify(runtime)
        sampler = env.step_input(runtime)[2]
        self.assertIsInstance(sampler, SamplerInput)
        self.assertNotIn(CANARY, repr(sampler))

        turn = make_turn(
            view,
            content="",
            calls=(call("read_file", {"path": "draft.txt"}, "private-safe-read"),),
        )
        writer_step = env.commit(runtime, turn)
        tool_input = env.step_input(writer_step.runtime)[2]
        self.assertIsInstance(tool_input, ToolInput)
        self.assertNotIn(CANARY, repr(tool_input))
        self.assertEqual(
            {field.name for field in fields(SamplerInput)},
            {
                "messages",
                "tools",
                "rendering",
                "context_content_hash",
                "context_revision_ref",
                "action_id",
                "writer_seed",
                "model_ref",
                "behavior_policy_ref",
                "tokenizer_ref",
                "template_ref",
                "decoding_ref",
                "native_sampling_budget",
                "adapter_ref",
                "decision_ordinal",
                "native_history",
            },
        )
        self.assertEqual(
            {field.name for field in fields(ToolInput)},
            {"files", "queue_entry", "tool_spec", "dispatch_permitted"},
        )
        self.assertEqual(
            {field.name for field in fields(AuthorInput)},
            {"request_ref", "request", "script", "decisions", "disclosures"},
        )
        self.assertEqual(
            {field.name for field in fields(CheckInput)},
            {"request_ref", "request", "files", "evaluator_packet"},
        )

    def test_start_member_publishes_the_sealed_member_start(self):
        spec = _group_spec(self.fixture, self.entry, self.store)
        member_start = MemberStartV1(group_spec_ref=spec.identity(), ordinal=0)
        _assert_fault_matrix(
            self,
            self.environment,
            self.runtime,
            member_start,
            self.root,
            event_kind="rollout_started",
            entry_checkpoint=self.entry,
            lineage=spec.members[0].member_id,
        )
        started = self.environment.start_member(
            self.entry,
            member_start,
        )
        self.assertEqual(started.state.position["lineage_id"], spec.members[0].member_id)
        self.assertEqual(self.store.read_head(spec.members[0].member_id) is not None, True)
        sampler = self.environment.step_input(started)[2]
        self.assertIsInstance(sampler, SamplerInput)
        self.assertEqual(sampler.writer_seed, spec.members[0].writer_seed)
        self.assertEqual(sampler.model_ref, spec.policy["model_ref"])

    def test_group_member_sampled_through_driver_reports_its_sealed_seed(self):
        model_ref = self.store.put_artifact({"model_id": "pinned-model"})
        spec = _group_spec(self.fixture, self.entry, self.store, model_ref=model_ref)
        member_start = MemberStartV1(group_spec_ref=spec.identity(), ordinal=0)
        runtime = self.environment.start_member(self.entry, member_start)
        member = spec.members[0]
        initial_context_ref = self.environment.verify(runtime).context.content_ref

        class CapturingScriptedBackend(ScriptedSampleBackend):
            def sample(self, prepared):
                self.prepared = prepared
                return super().sample(prepared)

        backend = CapturingScriptedBackend(
            [SampleResult({"role": "assistant", "content": "member draft", "tool_calls": []})]
        )
        checks = {
            check.identity(): check
            for node in self.fixture.graph.nodes.values()
            for check in node.checks.values()
        }
        gatherers = Gatherers(
            SamplingRunner(self.store, backend),
            ToolRunner(LocalTextToolProvider()),
            ScriptedAuthorSource(),
            CheckRunner(self.store, DeterministicEvaluator(), checks),
        )
        try:
            result = RolloutDriver(self.environment, gatherers).run(runtime, max_steps=1)
            runtime = result.runtime
        except DriverBudgetError as exc:
            runtime = exc.runtime

        event = self.store.load_event(runtime.state.history["head"])
        turn = self.store.get_artifact(event.payload_ref)
        trace = turn["adapter_trace"]
        self.assertEqual(backend.prepared.writer_seed, member.writer_seed)
        self.assertEqual(trace["seed"], member.writer_seed)
        self.assertEqual(backend.prepared.model_ref, spec.policy["model_ref"])
        self.assertEqual(backend.prepared.behavior_policy_ref, spec.policy["behavior_policy_ref"])
        self.assertEqual(backend.prepared.decoding_ref, spec.policy["decoding_ref"])
        self.assertEqual(backend.prepared.tokenizer_ref, spec.policy["tokenizer_ref"])
        self.assertEqual(backend.prepared.template_ref, spec.policy["template_ref"])
        self.assertEqual(
            backend.prepared.context_content_hash,
            initial_context_ref,
        )

    def test_enter_derives_an_unpublished_root_without_materializing_a_workspace(self):
        fixture = make_entry_fixture()
        root = self.root / "enter"
        root.mkdir()
        root.chmod(0o700)
        gate = LineageGate()
        store = TaskGraphStore(root / "store", verifier=gate)
        _persist_fixture(store, fixture, checkpoint=False)
        env = RolloutEnvironment(store, fixture.graph, None, gate, fixture.graph.policy)
        workspace = root / "workspace"
        runtime = env.enter(fixture.node_id, fixture.params)
        self.assertIsNone(store.read_head(fixture.params.lineage_id))
        self.assertEqual(runtime.state, fixture.state)
        self.assertFalse(workspace.exists())
        self.assertFalse(hasattr(runtime, "workspace"))
        self.assertEqual(env.verify(runtime).checkpoint_id, runtime.checkpoint_id)

    def test_open_head_requires_a_published_head_and_entry_can_open_directly(self):
        with self.assertRaises(ConcurrentUpdateError):
            self.environment.open_head(self.fixture.params.lineage_id)
        self.assertEqual(self.environment.open(self.entry).checkpoint_id, self.entry)

    def test_step_input_rejects_an_unpublished_candidate(self):
        view = self.environment.verify(self.runtime)
        transition = DERIVE["WriterTurnV1"](
            view, make_turn(view, content="draft"), self.environment.reader
        )
        context = transition.view.context
        candidate = RuntimeHandle(
            transition.view.checkpoint_id,
            transition.state,
            MaterializedContextV1(context.messages, context.tools, context.rendering),
        )
        with self.assertRaises(ConcurrentUpdateError):
            self.environment.step_input(candidate)

    def test_head_and_caller_error_classes_are_consistent(self):
        view = self.environment.verify(self.runtime)
        foreign_env, foreign_runtime = self._enter_alternate_policy("policy-port")
        wrong_state = replace(
            self.runtime,
            state=replace(
                self.runtime.state,
                position={**self.runtime.state.position, "visit_id": "forged-visit"},
            ),
        )
        wrong_context = replace(
            self.runtime,
            context=replace(self.runtime.context, rendering={"forged": True}),
        )
        cases = (
            (
                "foreign admission policy",
                lambda: self.environment.step_input(foreign_runtime),
                WriterRuntimeError,
            ),
            (
                "handle state mismatch",
                lambda: self.environment.verify(wrong_state),
                WriterRuntimeError,
            ),
            (
                "handle context mismatch",
                lambda: self.environment.verify(wrong_context),
                WriterRuntimeError,
            ),
        )
        for label, action, error in cases:
            with self.subTest(case=label), self.assertRaises(error):
                action()

        committed = self.environment.commit(self.runtime, make_turn(view, content="winner"))
        with self.assertRaises(ConcurrentUpdateError):
            self.environment.step_input(self.runtime)
        self.assertEqual(
            self.store.read_head(view.state.position["lineage_id"]), committed.commit_id
        )

    def test_identical_commit_retry_returns_the_published_head(self):
        view = self.environment.verify(self.runtime)
        turn = make_turn(view, content="same candidate")
        other_environment = RolloutEnvironment(
            self.store,
            self.fixture.graph,
            None,
            self.gate,
            self.fixture.graph.policy,
        )
        winner = self.environment.commit(self.runtime, turn)
        retry = other_environment.commit(self.runtime, turn)
        self.assertEqual(retry.commit_id, winner.commit_id)
        self.assertEqual(retry.runtime.checkpoint_id, winner.runtime.checkpoint_id)

    def test_adapter_contract_rejection_writes_nothing_and_gate_keeps_projection_error(self):
        view = self.environment.verify(self.runtime)
        bad = make_turn(
            view,
            usage={},
            adapter_trace={"generated_token_ids": [7]},
        )
        old_head = self.store.read_head(view.state.position["lineage_id"])
        with self.assertRaises(AdapterContractError):
            self.environment.commit(self.runtime, bad)
        self.assertEqual(self.store.read_head(view.state.position["lineage_id"]), old_head)

        bad_ref = self.store.put_artifact(bad.to_wire())
        event = EventV1(
            previous=view.head_event_id,
            seq=view.state.history["seq"] + 1,
            lineage_id=view.state.position["lineage_id"],
            rollout_id=view.state.position["lineage_id"],
            node_visit_id=view.state.position["visit_id"],
            kind="writer_action",
            actor="writer",
            audience=("controller", "trainer"),
            payload_ref=bad_ref,
            versions_ref=view.state.versions_ref,
            provenance_ref=view.state.provenance_ref,
        )
        with self.assertRaises(ProjectionError):
            self.gate.verify_commit(self.store, view.checkpoint_id, (event,), self.runtime.state)
        self.assertEqual(self.store.read_head(view.state.position["lineage_id"]), old_head)

    def test_non_group_context_claims_are_bound_on_fresh_and_persisted_paths(self):
        view = self.environment.verify(self.runtime)
        wrong = domain_hash("payload", {"not": "this context"})
        wrong_revision = _foreign_context_revision(self.store, view)
        turn = make_turn(
            view,
            adapter_trace={
                "context_revision_ref": wrong_revision,
                "context_content_hash": wrong,
            },
        )
        lineage = view.state.position["lineage_id"]
        old_head = self.store.read_head(lineage)
        with self.assertRaises(AdapterContractError):
            self.environment.commit(self.runtime, turn)
        self.assertEqual(self.store.read_head(lineage), old_head)

        payload_ref = self.store.put_artifact(turn.to_wire())
        event = EventV1(
            previous=view.head_event_id,
            seq=view.state.history["seq"] + 1,
            lineage_id=lineage,
            rollout_id=lineage,
            node_visit_id=view.state.position["visit_id"],
            kind="writer_action",
            actor="writer",
            audience=("controller", "trainer", "writer"),
            payload_ref=payload_ref,
            versions_ref=view.state.versions_ref,
            provenance_ref=view.state.provenance_ref,
        )
        with self.assertRaises(ProjectionError):
            self.store.publish(
                lineage,
                old_head,
                (event,),
                view.state,
                parent_checkpoint=view.checkpoint_id,
            )
        self.assertEqual(self.store.read_head(lineage), old_head)

    def test_group_sampling_pin_drift_is_adapter_error_before_commit_and_projection_on_replay(self):
        model_ref = self.store.put_artifact({"model_id": "pinned-model"})
        spec = _group_spec(self.fixture, self.entry, self.store, model_ref=model_ref)
        group_spec_ref = self.store.put_artifact(spec.to_wire())
        runtime = self.environment.start_member(
            self.entry, MemberStartV1(group_spec_ref=group_spec_ref, ordinal=0)
        )
        view = self.environment.verify(runtime)
        port = self.environment.step_input(runtime)[2]
        expected = {
            "seed": port.writer_seed,
            "model": "pinned-model",
            "policy_ref": port.behavior_policy_ref,
            "context_revision_ref": port.context_revision_ref,
            "rendering": dict(port.rendering),
            "context_content_hash": view.context.content_ref,
        }
        variants = (
            ("seed", {**expected, "seed": port.writer_seed + 1}),
            ("model", {**expected, "model": "other-model"}),
            (
                "policy_ref",
                {**expected, "policy_ref": self.store.put_artifact({"wrong": "policy"})},
            ),
            (
                "context_revision_ref",
                {
                    **expected,
                    "context_revision_ref": _foreign_context_revision(self.store, view),
                },
            ),
            (
                "rendering",
                {**expected, "rendering": {**expected["rendering"], "prefix_id": "wrong"}},
            ),
            (
                "context_content_hash",
                {**expected, "context_content_hash": domain_hash("payload", {"wrong": "content"})},
            ),
        )
        lineage = runtime.state.position["lineage_id"]
        for field, claims in variants:
            with self.subTest(field=field):
                turn = make_turn(view, content="sample", adapter_trace=claims)
                old_head = self.store.read_head(lineage)
                with self.assertRaises(AdapterContractError):
                    self.environment.commit(runtime, turn)
                self.assertEqual(self.store.read_head(lineage), old_head)

                payload_ref = self.store.put_artifact(turn.to_wire())
                event = EventV1(
                    previous=view.head_event_id,
                    seq=view.state.history["seq"] + 1,
                    lineage_id=lineage,
                    rollout_id=lineage,
                    node_visit_id=view.state.position["visit_id"],
                    kind="writer_action",
                    actor="writer",
                    audience=("controller", "trainer", "writer"),
                    payload_ref=payload_ref,
                    versions_ref=view.state.versions_ref,
                    provenance_ref=view.state.provenance_ref,
                )
                with self.assertRaises(ProjectionError):
                    self.store.publish(lineage, old_head, (event,), runtime.state)
                self.assertEqual(self.store.read_head(lineage), old_head)

    def test_producer_corruption_stays_a_store_error_and_missing_input_ref_is_projection(self):
        policy = ContextPolicyV1("drop")
        policy_ref = self.store.put_artifact(policy.to_wire())
        path = self.store._artifact_path(policy_ref, False)
        path.write_bytes(b"corrupted bytes")
        old_head = self.store.read_head(self.fixture.params.lineage_id)

        with self.assertRaises(CorruptRecordError):
            self.environment.commit(
                self.runtime,
                ContextOperationInputV1(policy_ref=policy_ref),
            )
        self.assertEqual(self.store.read_head(self.fixture.params.lineage_id), old_head)

        missing_ref = "f" * 64
        with self.assertRaises(ProjectionError) as rejected:
            self.environment.commit(
                self.runtime,
                ContextOperationInputV1(policy_ref=missing_ref),
            )
        self.assertIsInstance(rejected.exception.__cause__, MissingReferenceError)
        self.assertEqual(self.store.read_head(self.fixture.params.lineage_id), old_head)

    def test_gate_decode_path_preserves_disk_corruption(self):
        view = self.environment.verify(self.runtime)
        policy = ContextPolicyV1("drop")
        policy_ref = self.store.put_artifact(policy.to_wire())
        input_record = ContextOperationInputV1(policy_ref=policy_ref)
        payload_ref = self.store.put_artifact(input_record.to_wire())
        event = EventV1(
            previous=view.head_event_id,
            seq=view.state.history["seq"] + 1,
            lineage_id=view.state.position["lineage_id"],
            rollout_id=view.state.position["lineage_id"],
            node_visit_id=view.state.position["visit_id"],
            kind="context_changed",
            actor="environment",
            audience=("controller", "writer"),
            payload_ref=payload_ref,
            versions_ref=view.state.versions_ref,
            provenance_ref=view.state.provenance_ref,
        )
        self.store._artifact_path(policy_ref, False).write_bytes(b"tampered")

        with self.assertRaises(CorruptRecordError):
            self.gate._derive(self.store, view, event, StoreArtifactReader(self.store))

    def test_missing_derived_artifact_fails_publish_without_advancing_head(self):
        view = self.environment.verify(self.runtime)
        turn = make_turn(view, content="persisted turn")
        original = self.store.persist_artifact
        head = self.store.read_head(view.state.position["lineage_id"])

        def omit_budget(artifact):
            if artifact.ref != view.state.budgets_ref and artifact.kind == "artifact":
                # Only omit the changed budget, not the turn payload.
                if artifact.value_kind == "canonical_json":
                    body = load_canonical_json(artifact.value)
                    if body.get("schema") == 1 and "consumed" in body and "limits" in body:
                        return None
            return original(artifact)

        with patch.object(self.store, "persist_artifact", side_effect=omit_budget):
            with self.assertRaises(MissingReferenceError):
                self.environment.commit(self.runtime, turn)
        self.assertEqual(self.store.read_head(view.state.position["lineage_id"]), head)

    def test_environment_methods_share_one_outer_store_scope(self):
        original = self.store.operation
        depth = 0
        scopes = 0

        @contextmanager
        def counted():
            nonlocal depth, scopes
            if depth == 0:
                scopes += 1
            depth += 1
            try:
                with original():
                    yield
            finally:
                depth -= 1

        spec = _group_spec(self.fixture, self.entry, self.store)
        self.store.operation = counted
        entered = self.environment.enter(self.fixture.node_id, self.fixture.params)
        self.assertEqual(entered.checkpoint_id, self.entry)
        self.assertEqual(scopes, 1)
        reopened = self.environment.open(self.entry)
        self.assertEqual(scopes, 2)
        self.environment.start_member(
            self.entry,
            MemberStartV1(group_spec_ref=spec.identity(), ordinal=0),
        )
        self.assertEqual(scopes, 3)
        view = self.environment.verify(reopened)
        self.assertEqual(scopes, 4)
        self.environment.step_input(reopened)
        self.assertEqual(scopes, 5)
        self.environment.commit(reopened, make_turn(view, content="one scope"))
        self.assertEqual(scopes, 6)

    def test_design_event_kind_inventory(self):
        self.assertEqual(len(EVENT_KINDS), 11)
        self.assertEqual(len(set(EVENT_KINDS)), 11)


if __name__ == "__main__":
    unittest.main()
