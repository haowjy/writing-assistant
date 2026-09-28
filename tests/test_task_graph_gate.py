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
from writing_agent.task_graph import CheckpointV1, MessageV1, domain_hash, load_canonical_json
from writing_agent.task_graph_admission import AdmissionPolicyV1 as RuntimeAdmissionPolicy
from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
from writing_agent.task_graph_contracts import NodeContractV1
from writing_agent.task_graph_derive_entry import derive_entry
from writing_agent.task_graph_errors import CorruptRecordError, ProjectionError
from writing_agent.task_graph_gate import DERIVE, LineageGate, StoreArtifactReader, ViewCache
from writing_agent.task_graph_records import (
    AdmissionPolicyV1,
    ContextContentV1,
    ContextRevisionV1,
    ToolObservationV1,
)
from writing_agent.task_graph_store import TaskGraphStore


class GateStoreFixture:
    def __init__(self, test: unittest.TestCase, fixture=None) -> None:
        self.fixture = make_entry_fixture() if fixture is None else fixture
        self.directory = tempfile.TemporaryDirectory()
        test.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "store"
        self.gate = LineageGate()
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
        return commit_id, transition.view.checkpoint_id


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
        fixture = GateStoreFixture(self)
        root_view = fixture.verified_view(fixture.root_id)
        cache = ViewCache(capacity=2)
        first = replace(root_view, checkpoint_id="1" * 64)
        second = replace(root_view, checkpoint_id="2" * 64)
        third = replace(root_view, checkpoint_id="3" * 64)
        cache.insert(first)
        cache.insert(second)
        self.assertIs(cache.get(first.root_checkpoint_id, first.checkpoint_id), first)
        cache.insert(third)
        self.assertIsNone(cache.get(second.root_checkpoint_id, second.checkpoint_id))
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
            lambda: fixture.gate.verify_checkpoint(fixture.store, midrun_root),
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
            lambda: fixture.gate.verify_checkpoint(fixture.store, checkpoint_id),
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

    def test_cached_view_is_not_a_validation_memo_and_warm_view_uses_checkpoint_id(self) -> None:
        fixture = GateStoreFixture(self)
        honest = fixture.first_transition()
        _, honest_id = fixture.publish(honest)
        fixture.gate.cache.insert(honest.view)

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
            lambda: fixture.gate.verify_checkpoint(fixture.store, sibling_id),
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
            lambda: fixture.gate.verify_checkpoint(fixture.store, sibling_id),
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
        fixture.gate.cache.insert(first.view)
        second = fixture.next_tool_transition(first.view)
        fixture.persist_artifacts(second)
        derive = DERIVE["ToolObservationV1"]
        calls = 0

        def counted(*args):
            nonlocal calls
            calls += 1
            return derive(*args)

        with patch.dict(DERIVE, {"ToolObservationV1": counted}):
            fixture.store.publish(
                "rollout-fixture",
                first_commit,
                (second.event,),
                second.state,
            )

        self.assertEqual(calls, 1)
        self.assertIsNone(fixture.gate.cache.get(fixture.root_id, second.view.checkpoint_id))
        fixture.gate.cache.insert(second.view)
        self.assertIs(
            fixture.gate.cache.get(fixture.root_id, second.view.checkpoint_id), second.view
        )

    def test_warm_gate_rejects_forged_outcome_sibling_too(self) -> None:
        fixture = GateStoreFixture(self)
        honest = fixture.first_transition()
        _, honest_id = fixture.publish(honest)
        fixture.gate.cache.insert(honest.view)
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
            lambda: fixture.gate.verify_checkpoint(fixture.store, sibling_id),
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
            lambda: fixture.gate.verify_checkpoint(fixture.store, sibling_id),
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
        fixture.gate.cache.insert(first.view)
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
        fixture.gate.cache.insert(transition.view)
        path = fixture.store._artifact_path(transition.state.budgets_ref, False)
        path.write_bytes(b"tampered")

        with fixture.store.operation(), self.assertRaises(CorruptRecordError):
            fixture.gate.view(fixture.store, checkpoint_id)

        cold_gate = LineageGate()
        cold_store = TaskGraphStore(fixture.root, verifier=cold_gate)
        with tempfile.TemporaryDirectory() as workspace, self.assertRaises(CorruptRecordError):
            cold_store.restore(checkpoint_id, Path(workspace) / "runtime")


if __name__ == "__main__":
    unittest.main()
