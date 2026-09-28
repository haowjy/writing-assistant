"""Store-backed acceptance checks for lineage verification and its view cache."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.task_graph_fixtures import EntryFixture, make_entry_fixture
from tests.test_task_graph_derive_writer import call, make_turn
from writing_agent import task_graph_gate
from writing_agent.task_graph import (
    CheckpointV1,
    CommitV1,
    MessageV1,
    domain_hash,
    load_canonical_json,
)
from writing_agent.task_graph_admission import AdmissionPolicyV1 as RuntimeAdmissionPolicy
from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
from writing_agent.task_graph_contracts import NodeContractV1
from writing_agent.task_graph_derive_entry import derive_entry
from writing_agent.task_graph_errors import (
    CorruptRecordError,
    MissingReferenceError,
    ProjectionError,
)
from writing_agent.task_graph_gate import DERIVE, LineageGate, StoreArtifactReader, ViewCache
from writing_agent.task_graph_records import (
    AdmissionPolicyV1,
    ContextContentV1,
    ContextRevisionV1,
    ToolObservationV1,
)
from writing_agent.task_graph_store import TaskGraphStore


class GateStoreFixture:
    def __init__(self, test: unittest.TestCase, fixture=None, *, cache=None) -> None:
        self.fixture = make_entry_fixture() if fixture is None else fixture
        self.directory = tempfile.TemporaryDirectory()
        test.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "store"
        self.gate = LineageGate(cache=cache)
        self.store = TaskGraphStore(self.root, verifier=self.gate)
        for body in self.fixture.reader.public.values():
            self.store.put_artifact(body)
        for body in self.fixture.reader.private.values():
            self.store.put_artifact(body, private=True)
        self.store.persist(self.fixture.graph.instance)
        for artifact in self.fixture.artifacts:
            if artifact.kind in {"context_node", "context_revision"}:
                self.store.persist(artifact.value)
        self.root_id = self.store.save_checkpoint(self.fixture.state)
        self.reader = StoreArtifactReader(self.store)

    def first_transition(self):
        view = self.verified_view(self.root_id)
        turn = make_turn(
            view,
            content="",
            calls=(call("read_file", {"path": "draft.txt"}),),
        )
        return DERIVE["WriterTurnV1"](view, turn, self.reader)

    def verified_view(self, checkpoint_id):
        with self.store.operation():
            return self.gate.view(self.store, checkpoint_id)

    def next_tool_transition(self, view):
        queued = view.state.continuation["tool_queue"][0]
        observation = ToolObservationV1(
            call_id=queued["call_id"],
            dispatch={
                "spec": {
                    "max_file_bytes": view.tool_spec.max_file_bytes,
                    "max_workspace_bytes": view.tool_spec.max_workspace_bytes,
                },
                "observation": {"ok": True, "valid": True, "result": "alpha"},
                "effect": {},
            },
        )
        return DERIVE["ToolObservationV1"](view, observation, self.reader)

    def persist_artifacts(self, transition) -> None:
        self.store.persist(transition.event)
        for artifact in transition.artifacts:
            if artifact.kind in {"context_node", "context_revision"}:
                self.store.persist(artifact.value)
                continue
            if artifact.value_kind == "canonical_json":
                value = load_canonical_json(artifact.value)
            elif artifact.value_kind == "record":
                value = artifact.value.to_wire()
            else:
                self.store.put_bytes_artifact(artifact.value, private=artifact.kind == "private")
                continue
            self.store.put_artifact(value, private=artifact.kind == "private")

    def publish(self, transition, *, expected_head=None, fault=None):
        self.persist_artifacts(transition)
        commit_id = self.store.publish(
            transition.state.position["lineage_id"],
            expected_head,
            (transition.event,),
            transition.state,
            parent_checkpoint=(self.root_id if expected_head is None else None),
            fault=fault,
        )
        published_view = self.gate.record_published(self.store, commit_id)
        return commit_id, published_view.checkpoint_id


def narrow_tool_entry_fixture() -> EntryFixture:
    """Build a valid graph admitted under only read_file, with its policy pinned."""
    fixture = make_entry_fixture()
    public = dict(fixture.reader.public)
    node = fixture.graph.node(fixture.node_id)
    contract = NodeContractV1.from_dict(public[node.spec.entry_contract])
    narrow_contract = replace(
        contract,
        entry=replace(contract.entry_contract, tool_allowlist=("read_file",)),
    )
    contract_body = narrow_contract.to_dict()
    contract_ref = domain_hash("payload", contract_body)
    public[contract_ref] = contract_body
    fixture.reader.public[contract_ref] = contract_body
    instance = replace(
        fixture.graph.instance,
        nodes=tuple(
            replace(spec, entry_contract=contract_ref) if spec.id == fixture.node_id else spec
            for spec in fixture.graph.instance.nodes
        ),
    )
    policy = RuntimeAdmissionPolicy(allowed_tools=frozenset({"read_file"}))
    wire_policy = AdmissionPolicyV1.from_admission_policy(policy)
    policy_ref = fixture.reader.add(wire_policy.to_wire())
    versions = fixture.reader.artifact(fixture.state.versions_ref)
    versions["admission_policy_ref"] = policy_ref
    versions_ref = fixture.reader.add(versions)
    graph = admit_graph(
        instance,
        MappingArtifactResolver(public, fixture.reader.private),
        policy=policy,
    )
    params = replace(fixture.params, versions_ref=versions_ref)
    entry = derive_entry(graph, fixture.node_id, params, fixture.reader)
    for artifact in entry.artifacts:
        value = (
            load_canonical_json(artifact.value)
            if artifact.value_kind == "canonical_json"
            else artifact.value.to_wire()
            if artifact.value_kind == "record"
            else artifact.value
        )
        if artifact.kind == "private":
            fixture.reader.private[artifact.ref] = value
        elif artifact.kind == "context_node":
            fixture.reader.context_nodes[artifact.ref] = value
        elif artifact.kind == "context_revision":
            fixture.reader.context_revisions[artifact.ref] = value
        else:
            fixture.reader.public[artifact.ref] = value
    return EntryFixture(
        graph,
        fixture.node_id,
        params,
        fixture.reader,
        entry.state,
        entry.artifacts,
    )


class LineageGateTests(unittest.TestCase):
    def assert_projection_path(self, path: str, action, *, store=None) -> None:
        if store is None:
            with self.assertRaises(ProjectionError) as rejected:
                action()
        else:
            with store.operation(), self.assertRaises(ProjectionError) as rejected:
                action()
        self.assertIn(path, str(rejected.exception))

    def test_registry_has_normalized_keys_and_rejects_import_duplicates(self) -> None:
        self.assertEqual(
            set(DERIVE),
            {
                "WriterTurnV1",
                "ToolObservationV1",
                "AuthorReplyV1",
                "EvaluatorResultV1",
                "ContextOperationInputV1",
                "MemberStartV1",
                ("EnvironmentStepV1", "request_author"),
                ("EnvironmentStepV1", "request_checks"),
                ("EnvironmentStepV1", "commit_transition"),
                ("EnvironmentStepV1", "seal_outcome"),
                ("EnvironmentStepV1", "stop_exhausted"),
                ("EnvironmentStepV1", "publish_reward"),
            },
        )
        route = next(iter(DERIVE))
        with patch.object(task_graph_gate, "AUTHOR_DERIVES", {route: object()}):
            with self.assertRaises(RuntimeError):
                task_graph_gate._assemble_derives()

    def test_view_cache_is_a_64_entry_lru(self) -> None:
        cache = ViewCache(capacity=2)
        fixture = GateStoreFixture(self, cache=cache)
        first = fixture.first_transition()
        first_commit, first_id = fixture.publish(first)
        second = fixture.next_tool_transition(first.view)
        second_commit, second_id = fixture.publish(second, expected_head=first_commit)
        third_view = second.view
        third = DERIVE["WriterTurnV1"](
            third_view, make_turn(third_view, content="done"), fixture.reader
        )
        _, third_id = fixture.publish(third, expected_head=second_commit)

        self.assertIsNone(cache.get(fixture.root_id, first_id))
        self.assertIsNotNone(cache.get(fixture.root_id, second_id))
        self.assertEqual(cache.get(fixture.root_id, third_id).checkpoint_id, third_id)
        self.assertEqual(len(cache._views), 2)
        self.assertEqual(ViewCache().capacity, 64)

    def test_entry_is_fixed_point_and_parentless_midrun_root_is_rejected(self) -> None:
        fixture = GateStoreFixture(self)
        transition = fixture.first_transition()
        fixture.persist_artifacts(transition)
        checkpoint = CheckpointV1(
            state=transition.state,
            event_head=transition.event.id,
        )
        midrun_root = fixture.store.persist(checkpoint)

        self.assert_projection_path(
            "state.budgets_ref",
            lambda: fixture.gate.view(fixture.store, midrun_root),
            store=fixture.store,
        )

        root_view = fixture.verified_view(fixture.root_id)
        self.assertEqual(root_view.state.identity(), fixture.fixture.state.identity())

    def test_runtime_checkpoints_reject_supplemental_artifact_refs(self) -> None:
        fixture = GateStoreFixture(self)
        checkpoint_id = fixture.store.save_checkpoint(
            fixture.fixture.state,
            artifact_refs=(fixture.fixture.state.provenance_ref,),
        )
        self.assert_projection_path(
            "checkpoint.artifact_refs",
            lambda: fixture.gate.view(fixture.store, checkpoint_id),
            store=fixture.store,
        )

        transition = fixture.first_transition()
        fixture.persist_artifacts(transition)
        with self.assertRaises(ProjectionError):
            fixture.store.publish(
                "rollout-fixture",
                None,
                (transition.event,),
                transition.state,
                parent_checkpoint=fixture.root_id,
                artifact_refs=(transition.state.provenance_ref,),
            )
        self.assertIsNone(fixture.store.read_head("rollout-fixture"))

    def test_first_operation_entry_canary_is_rejected_before_a_request_is_built(self) -> None:
        fixture = GateStoreFixture(self)
        context = fixture.reader.context(fixture.fixture.state.context_ref)
        content = ContextContentV1(
            parent_ref=None,
            messages=(
                *context.messages,
                MessageV1(role="user", content=("PRIVATE-CANARY",), origin="forged:1"),
            ),
            tools=context.tools,
            rendering=context.rendering,
        )
        revision = ContextRevisionV1(
            content_ref=content.identity(),
            event_head=None,
            provenance_refs=(),
        )
        fixture.store.persist(content)
        fixture.store.persist(revision)
        forged_state = replace(fixture.fixture.state, context_ref=revision.identity())
        checkpoint_id = fixture.store.save_checkpoint(forged_state)
        built_requests = []

        def build_request_after_verify() -> None:
            verified = fixture.verified_view(checkpoint_id)
            built_requests.append({"messages": verified.context.messages})

        self.assert_projection_path("state.context_ref", build_request_after_verify)
        self.assertEqual(built_requests, [])

    def test_publish_verifier_rejects_forged_event_state_payload_and_visibility(self) -> None:
        for forged in ("event", "state", "payload", "visibility"):
            with self.subTest(forgery=forged):
                fixture = GateStoreFixture(self)
                transition = fixture.first_transition()
                fixture.persist_artifacts(transition)
                event = transition.event
                state = transition.state

                if forged == "event":
                    event = replace(event, actor="environment", id=None)
                    state = replace(
                        state,
                        history={**state.history, "head": event.id},
                    )
                elif forged == "state":
                    state = replace(
                        state,
                        position={**state.position, "phase": "terminal"},
                    )
                elif forged == "payload":
                    input_body = transition.input.to_wire()
                    input_body["message"] = {
                        **input_body["message"],
                        "content": "forged content",
                    }
                    payload_ref = fixture.store.put_artifact(input_body)
                    event = replace(event, payload_ref=payload_ref, id=None)
                    state = replace(
                        state,
                        history={**state.history, "head": event.id},
                    )
                else:
                    requirements = fixture.store.get_artifact(state.requirements_ref, private=True)
                    requirements["active"] = {"forged": "requirement"}
                    public_ref = fixture.store.put_artifact(requirements)
                    state = replace(state, requirements_ref=public_ref)

                with self.assertRaises(ProjectionError) as rejected:
                    fixture.store.publish(
                        state.position["lineage_id"],
                        None,
                        (event,),
                        state,
                        parent_checkpoint=fixture.root_id,
                    )
                expected_path = {
                    "event": "event.actor",
                    "state": "state.position.phase",
                    "payload": "state.context_ref",
                }.get(forged)
                if expected_path is not None:
                    self.assertIn(expected_path, str(rejected.exception))
                candidate_id = CheckpointV1(
                    parents=(fixture.root_id,), state=state, event_head=event.id
                ).identity()
                self.assertIsNone(fixture.gate.cache.get(fixture.root_id, candidate_id))
                self.assertIsNone(fixture.store.read_head(state.position["lineage_id"]))

    def test_candidate_closure_forgeries_are_projection_errors(self) -> None:
        fixture = GateStoreFixture(self)
        honest = fixture.first_transition()
        fixture.persist_artifacts(honest)

        other_params = replace(fixture.fixture.params, lineage_id="rollout-other", visit_id="other")
        other_entry = derive_entry(
            fixture.fixture.graph,
            fixture.fixture.node_id,
            other_params,
            fixture.fixture.reader,
        )
        for artifact in other_entry.artifacts:
            if artifact.kind in {"context_node", "context_revision"}:
                fixture.store.persist(artifact.value)
                continue
            value = (
                load_canonical_json(artifact.value)
                if artifact.value_kind == "canonical_json"
                else artifact.value.to_wire()
            )
            fixture.store.put_artifact(value, private=artifact.kind == "private")
        other_root = fixture.store.save_checkpoint(other_entry.state)
        other_view = fixture.gate.view(fixture.store, other_root)
        other_transition = DERIVE["WriterTurnV1"](
            other_view,
            make_turn(
                other_view,
                content="",
                calls=(call("read_file", {"path": "draft.txt"}),),
            ),
            fixture.reader,
        )
        fixture.persist_artifacts(other_transition)

        versions = fixture.store.get_artifact(honest.state.versions_ref)
        versions["tool_spec"] = {"max_file_bytes": 64_000, "max_workspace_bytes": 2048}
        changed_versions_ref = fixture.store.put_artifact(versions)
        changed_event = replace(honest.event, versions_ref=changed_versions_ref, id=None)
        cases = (
            (
                "C2",
                lambda: fixture.store.publish(
                    "rollout-fixture",
                    None,
                    (other_transition.event,),
                    honest.state,
                    parent_checkpoint=fixture.root_id,
                ),
            ),
            (
                "C3",
                lambda: fixture.store.publish(
                    "rollout-other",
                    None,
                    (honest.event,),
                    other_transition.state,
                    parent_checkpoint=other_root,
                ),
            ),
            (
                "E3",
                lambda: fixture.store.publish(
                    "rollout-fixture",
                    None,
                    (changed_event,),
                    honest.state,
                    parent_checkpoint=fixture.root_id,
                ),
            ),
        )
        for case, action in cases:
            with self.subTest(case=case), self.assertRaises(ProjectionError):
                action()
            self.assertIsNone(fixture.store.read_head("rollout-fixture"))
            self.assertIsNone(fixture.store.read_head("rollout-other"))

    def test_malformed_record_payloads_fail_at_write_and_derive_errors_keep_causes(self) -> None:
        fixture = GateStoreFixture(self)
        with self.assertRaises(ValueError):
            fixture.store.put_artifact(
                {"record_type": "ToolObservationV1", "call_id": "call-1", "dispatch": {}}
            )

        transition = fixture.first_transition()
        fixture.persist_artifacts(transition)
        for failure in (IndexError("missing list item"), AttributeError("missing field")):

            def broken_derive(*_args, error=failure):
                raise error

            with (
                patch.dict(DERIVE, {"WriterTurnV1": broken_derive}),
                fixture.store.operation(),
                self.assertRaises(ProjectionError) as rejected,
            ):
                fixture.gate._derive(
                    fixture.store,
                    fixture.gate.view(fixture.store, fixture.root_id),
                    transition.event,
                    StoreArtifactReader(fixture.store),
                )
            self.assertIsInstance(rejected.exception.__cause__, type(failure))

    def test_v1_semantics_refuse_verifierless_store_including_terminal_root(self) -> None:
        fixture = GateStoreFixture(self)
        legacy_store = TaskGraphStore(fixture.root)
        transition = fixture.first_transition()
        fixture.persist_artifacts(transition)
        forged = replace(
            transition.state,
            position={**transition.state.position, "phase": "terminal"},
        )

        with self.assertRaises(ProjectionError):
            legacy_store.publish(
                "rollout-fixture",
                None,
                (transition.event,),
                forged,
                parent_checkpoint=fixture.root_id,
            )
        self.assertIsNone(legacy_store.read_head("rollout-fixture"))
        with self.assertRaises(ProjectionError):
            legacy_store.save_checkpoint(forged)
        with self.assertRaises(ProjectionError):
            legacy_store.restore(fixture.root_id, fixture.root.parent / "no-verifier-workspace")

        terminal_root = replace(
            fixture.fixture.state,
            position={**fixture.fixture.state.position, "phase": "terminal"},
        )
        with self.assertRaises(ProjectionError):
            legacy_store.save_checkpoint(terminal_root)
        self.assertIsNone(legacy_store.read_head("rollout-fixture"))

    def test_initial_commit_cannot_start_from_midlineage_and_branch_is_refused_under_gate(
        self,
    ) -> None:
        fixture = GateStoreFixture(self)
        midlineage = fixture.store.save_checkpoint(fixture.fixture.state, parent=fixture.root_id)
        transition = fixture.first_transition()
        fixture.persist_artifacts(transition)
        with self.assertRaises(ProjectionError):
            fixture.store.publish(
                "rollout-fixture",
                None,
                (transition.event,),
                transition.state,
                parent_checkpoint=midlineage,
            )
        with self.assertRaises(ProjectionError):
            fixture.store.branch(
                fixture.root_id,
                "rollout-branch",
                (transition.event,),
                transition.state,
            )
        self.assertIsNone(fixture.store.read_head("rollout-fixture"))
        self.assertIsNone(fixture.store.read_head("rollout-branch"))

    def test_cached_view_is_not_a_validation_memo_and_warm_view_uses_checkpoint_id(self) -> None:
        fixture = GateStoreFixture(self)
        honest = fixture.first_transition()
        _, honest_id = fixture.publish(honest)

        derive = DERIVE["WriterTurnV1"]
        calls = 0

        def counted(*args):
            nonlocal calls
            calls += 1
            return derive(*args)

        with patch.dict(DERIVE, {"WriterTurnV1": counted}):
            with fixture.store.operation():
                cached = fixture.gate.view(fixture.store, honest_id)
        self.assertEqual(calls, 0)
        self.assertEqual(cached.checkpoint_id, honest_id)

        budget = fixture.store.get_artifact(honest.state.budgets_ref)
        budget["consumed"]["writer_turns"] = 0
        lower_budget_ref = fixture.store.put_artifact(budget)
        sibling_state = replace(honest.state, budgets_ref=lower_budget_ref)
        sibling = CheckpointV1(
            parents=(fixture.root_id,),
            state=sibling_state,
            event_head=honest.event.id,
        )
        sibling_id = fixture.store.persist(sibling)

        self.assert_projection_path(
            "state.budgets_ref",
            lambda: fixture.gate.view(fixture.store, sibling_id),
            store=fixture.store,
        )
        self.assert_projection_path(
            "state.budgets_ref",
            lambda: fixture.gate.view(fixture.store, sibling_id),
            store=fixture.store,
        )
        self.assert_projection_path(
            "state.budgets_ref",
            lambda: fixture.gate.verify_commit(
                fixture.store, sibling_id, (honest.event,), honest.state
            ),
            store=fixture.store,
        )

        second = fixture.next_tool_transition(honest.view)

        def fail_after_gate(stage: str) -> None:
            if stage == "after_immutable_writes":
                raise RuntimeError("injected post-verifier failure")

        with self.assertRaisesRegex(RuntimeError, "post-verifier"):
            fixture.publish(
                second,
                expected_head=fixture.store.read_head("rollout-fixture"),
                fault=fail_after_gate,
            )
        self.assert_projection_path(
            "state.budgets_ref",
            lambda: fixture.gate.view(fixture.store, sibling_id),
            store=fixture.store,
        )
        self.assert_projection_path(
            "state.budgets_ref",
            lambda: fixture.gate.view(fixture.store, sibling_id),
            store=fixture.store,
        )
        self.assert_projection_path(
            "state.budgets_ref",
            lambda: fixture.gate.verify_commit(
                fixture.store, sibling_id, (second.event,), second.state
            ),
            store=fixture.store,
        )

    def test_verify_commit_on_cached_base_derives_once_and_does_not_cache_candidate(self) -> None:
        fixture = GateStoreFixture(self)
        first = fixture.first_transition()
        first_commit, _ = fixture.publish(first)
        second = fixture.next_tool_transition(first.view)
        fixture.persist_artifacts(second)
        derive = DERIVE["ToolObservationV1"]
        calls = 0

        def counted(*args):
            nonlocal calls
            calls += 1
            return derive(*args)

        with patch.dict(DERIVE, {"ToolObservationV1": counted}):
            commit_id = fixture.store.publish(
                "rollout-fixture",
                first_commit,
                (second.event,),
                second.state,
            )

        self.assertEqual(calls, 1)
        self.assertIsNone(fixture.gate.cache.get(fixture.root_id, second.view.checkpoint_id))
        # The store's returned candidate is the only object the gate will cache.
        gate_view = fixture.gate.record_published(fixture.store, commit_id)
        self.assertIs(fixture.gate.cache.get(fixture.root_id, gate_view.checkpoint_id), gate_view)

    def test_warm_gate_rejects_forged_outcome_sibling_too(self) -> None:
        fixture = GateStoreFixture(self)
        honest = fixture.first_transition()
        _, honest_id = fixture.publish(honest)
        forged_outcome = fixture.store.get_artifact(honest.state.outcome_ref)
        forged_outcome["stop_reason"] = "forged"
        forged_state = replace(
            honest.state,
            outcome_ref=fixture.store.put_artifact(forged_outcome),
        )
        sibling_id = fixture.store.persist(
            CheckpointV1(
                parents=(fixture.root_id,),
                state=forged_state,
                event_head=honest.event.id,
            )
        )

        self.assert_projection_path(
            "state.outcome_ref",
            lambda: fixture.gate.view(fixture.store, sibling_id),
            store=fixture.store,
        )
        self.assert_projection_path(
            "state.outcome_ref",
            lambda: fixture.gate.view(fixture.store, sibling_id),
            store=fixture.store,
        )
        self.assert_projection_path(
            "state.outcome_ref",
            lambda: fixture.gate.verify_commit(
                fixture.store, sibling_id, (honest.event,), honest.state
            ),
            store=fixture.store,
        )

        # The cached honest checkpoint is not selected by the sibling's identity.
        self.assertIsNotNone(fixture.gate.cache.get(fixture.root_id, honest_id))

        second = fixture.next_tool_transition(honest.view)

        def fail_after_gate(stage: str) -> None:
            if stage == "after_immutable_writes":
                raise RuntimeError("injected post-verifier failure")

        with self.assertRaisesRegex(RuntimeError, "post-verifier"):
            fixture.publish(
                second,
                expected_head=fixture.store.read_head("rollout-fixture"),
                fault=fail_after_gate,
            )
        self.assert_projection_path(
            "state.outcome_ref",
            lambda: fixture.gate.view(fixture.store, sibling_id),
            store=fixture.store,
        )
        self.assert_projection_path(
            "state.outcome_ref",
            lambda: fixture.gate.view(fixture.store, sibling_id),
            store=fixture.store,
        )
        self.assert_projection_path(
            "state.outcome_ref",
            lambda: fixture.gate.verify_commit(
                fixture.store, sibling_id, (second.event,), second.state
            ),
            store=fixture.store,
        )

    def test_unknown_transition_semantics_is_refused(self) -> None:
        fixture = GateStoreFixture(self)
        versions = fixture.store.get_artifact(fixture.fixture.state.versions_ref)
        versions["transition_semantics"] = "task-graph-derive-v2"
        versions_ref = fixture.store.put_artifact(versions)

        with fixture.store.operation(), self.assertRaises(ProjectionError):
            fixture.gate._admitted_graph(
                fixture.store,
                fixture.fixture.state.instance_ref,
                versions_ref,
            )

    def test_admission_cache_does_not_reuse_a_graph_across_store_roots(self) -> None:
        fixture = GateStoreFixture(self)
        state = fixture.fixture.state
        with fixture.store.operation():
            fixture.gate._admitted_graph(fixture.store, state.instance_ref, state.versions_ref)

        other_store = TaskGraphStore(fixture.root.parent / "other-store", verifier=fixture.gate)
        versions = fixture.store.get_artifact(state.versions_ref)
        other_store.put_artifact(fixture.store.get_artifact(versions["admission_policy_ref"]))
        other_store.put_artifact(versions)
        with other_store.operation(), self.assertRaises(MissingReferenceError):
            fixture.gate._admitted_graph(other_store, state.instance_ref, state.versions_ref)

    def test_narrow_pinned_admission_rejects_a_forged_tool_execution_on_cold_restore(
        self,
    ) -> None:
        fixture = GateStoreFixture(self, narrow_tool_entry_fixture())
        view = fixture.verified_view(fixture.root_id)
        turn = make_turn(
            view,
            content="",
            calls=(call("write_file", {"path": "draft.txt", "content": "forged"}),),
        )
        transition = DERIVE["WriterTurnV1"](view, turn, fixture.reader)
        queue = list(transition.state.continuation["tool_queue"])
        self.assertEqual(queue[0]["name"], "invalid_call")
        queue[0] = {
            **queue[0],
            "name": "write_file",
            "arguments": {"path": "draft.txt", "content": "forged"},
            "rejection": None,
        }
        continuation = {**transition.state.continuation, "tool_queue": tuple(queue)}
        forged_state = replace(transition.state, continuation=continuation)
        fixture.persist_artifacts(transition)
        checkpoint_id = fixture.store.persist(
            CheckpointV1(
                parents=(fixture.root_id,),
                state=forged_state,
                event_head=transition.event.id,
            )
        )

        cold_store = TaskGraphStore(fixture.root, verifier=LineageGate())
        with (
            tempfile.TemporaryDirectory() as workspace,
            self.assertRaises(ProjectionError) as rejected,
        ):
            cold_store.restore(checkpoint_id, Path(workspace) / "runtime")
        self.assertIn("state.continuation.tool_queue[0].arguments", str(rejected.exception))

    def test_cached_prefix_does_not_authorize_a_forged_suffix(self) -> None:
        fixture = GateStoreFixture(self)
        first = fixture.first_transition()
        first_commit, first_id = fixture.publish(first)
        second = fixture.next_tool_transition(first.view)
        fixture.persist_artifacts(second)
        forged_payload = replace(
            second.input,
            dispatch={
                **second.input.dispatch,
                "observation": {"ok": True, "valid": True, "result": "forged"},
            },
        )
        forged_payload_ref = fixture.store.put_artifact(forged_payload.to_wire())
        forged_event = replace(second.event, payload_ref=forged_payload_ref, id=None)
        forged_state = replace(
            second.state,
            history={**second.state.history, "head": forged_event.id},
        )

        with self.assertRaises(ProjectionError) as rejected:
            fixture.store.publish(
                "rollout-fixture",
                first_commit,
                (forged_event,),
                forged_state,
            )
        self.assertIn("state.context_ref", str(rejected.exception))
        self.assertEqual(fixture.store.read_head("rollout-fixture"), first_commit)
        self.assertEqual(fixture.verified_view(first_id).checkpoint_id, first_id)

    def test_two_event_commit_is_rejected(self) -> None:
        fixture = GateStoreFixture(self)
        transition = fixture.first_transition()
        fixture.persist_artifacts(transition)
        second = replace(
            transition.event,
            previous=transition.event.id,
            seq=transition.event.seq + 1,
            id=None,
        )
        state = replace(
            transition.state,
            history={**transition.state.history, "head": second.id, "seq": second.seq},
        )

        with self.assertRaises(ProjectionError):
            fixture.store.publish(
                "rollout-fixture",
                None,
                (transition.event, second),
                state,
                parent_checkpoint=fixture.root_id,
            )
        self.assertIsNone(fixture.store.read_head("rollout-fixture"))

    def test_on_disk_tamper_fails_cached_operation_and_cold_restore(self) -> None:
        fixture = GateStoreFixture(self)
        transition = fixture.first_transition()
        _, checkpoint_id = fixture.publish(transition)
        path = fixture.store._artifact_path(transition.state.budgets_ref, False)
        path.write_bytes(b"tampered")

        with fixture.store.operation(), self.assertRaises(CorruptRecordError):
            fixture.gate.view(fixture.store, checkpoint_id)

        cold_gate = LineageGate()
        cold_store = TaskGraphStore(fixture.root, verifier=cold_gate)
        with tempfile.TemporaryDirectory() as workspace, self.assertRaises(CorruptRecordError):
            cold_store.restore(checkpoint_id, Path(workspace) / "runtime")

    def test_record_published_falls_back_to_a_verified_fold_for_an_honest_retry(self) -> None:
        fixture = GateStoreFixture(self)
        self.assertFalse(hasattr(fixture.gate.cache, "insert"))
        transition = fixture.first_transition()
        commit_id, checkpoint_id = fixture.publish(transition)
        fresh_gate = LineageGate()
        published_view = fresh_gate.record_published(fixture.store, commit_id)
        self.assertEqual(published_view.checkpoint_id, checkpoint_id)
        self.assertEqual(fixture.store.read_head("rollout-fixture"), commit_id)
        self.assertEqual(
            fresh_gate.cache.get(fixture.root_id, checkpoint_id), published_view
        )

    def test_record_published_requires_the_commit_to_be_the_published_head(self) -> None:
        fixture = GateStoreFixture(self)
        transition = fixture.first_transition()
        fixture.persist_artifacts(transition)
        checkpoint = CheckpointV1(
            parents=(fixture.root_id,), state=transition.state, event_head=transition.event.id
        )
        commit = CommitV1(events=(transition.event.id,), checkpoint=checkpoint.identity())
        fixture.store.persist(checkpoint)
        fixture.store.persist(commit)

        with self.assertRaises(ProjectionError):
            fixture.gate.record_published(fixture.store, commit.identity())
        self.assertIsNone(fixture.store.read_head("rollout-fixture"))
        self.assertIsNone(fixture.gate.cache.get(fixture.root_id, checkpoint.identity()))

    def test_record_published_does_not_cache_a_candidate_without_a_published_head(self) -> None:
        fixture = GateStoreFixture(self)
        transition = fixture.first_transition()
        fixture.persist_artifacts(transition)
        with fixture.store.operation():
            fixture.gate.verify_commit(
                fixture.store, fixture.root_id, (transition.event,), transition.state
            )
        checkpoint = CheckpointV1(
            parents=(fixture.root_id,), state=transition.state, event_head=transition.event.id
        )
        commit = CommitV1(events=(transition.event.id,), checkpoint=checkpoint.identity())
        fixture.store.persist(checkpoint)
        fixture.store.persist(commit)

        with self.assertRaises(ProjectionError):
            fixture.gate.record_published(fixture.store, commit.identity())
        self.assertIsNone(fixture.store.read_head("rollout-fixture"))
        self.assertIsNone(fixture.gate.cache.get(fixture.root_id, checkpoint.identity()))


if __name__ == "__main__":
    unittest.main()
