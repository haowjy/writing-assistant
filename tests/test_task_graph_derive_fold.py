"""Cross-derive transition fold: every successor is the next derive's only view."""

from __future__ import annotations

import unittest
from dataclasses import replace

from tests.task_graph_fixtures import make_entry_fixture, make_outcome_fixture
from tests.test_task_graph_derive_author import _answered_reply
from tests.test_task_graph_derive_outcome import evaluation_result, make_view
from tests.test_task_graph_derive_writer import call, make_turn
from writing_agent.task_graph import CheckpointV1, canonical_bytes, load_canonical_json
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_contracts import (
    AuthorPacketV1,
    DecisionBindingsV1,
    InteractionContractV1,
    InteractionPolicyV1,
    ScriptedAuthorV1,
)
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_derive_author import derive_author_reply, derive_author_request
from writing_agent.task_graph_derive_context import derive_context_operation, derive_member_start
from writing_agent.task_graph_derive_outcome import (
    derive_check_request,
    derive_check_result,
    derive_exhausted_stop,
    derive_reward,
    derive_seal,
    derive_transition,
)
from writing_agent.task_graph_derive_writer import derive_tool_result, derive_writer_turn
from writing_agent.task_graph_records import (
    ContextOperationInputV1,
    EnvironmentStepV1,
    MemberStartV1,
    ToolObservationV1,
)
from writing_agent.task_graph_transition import CheckpointChain, LineageMode


def _signature(transition) -> bytes:
    def artifact_value(artifact):
        if artifact.value_kind == "canonical_json":
            return artifact.value.hex()
        if artifact.value_kind == "record":
            return artifact.value.to_wire()
        return artifact.value.hex()

    from writing_agent.task_graph import canonical_bytes

    return canonical_bytes(
        {
            "event": transition.event.to_dict(),
            "state": transition.state.to_dict(),
            "artifacts": [
                {
                    "ref": artifact.ref,
                    "kind": artifact.kind,
                    "value_kind": artifact.value_kind,
                    "value": artifact_value(artifact),
                }
                for artifact in transition.artifacts
            ],
        }
    )


def _persist(reader, previous, transition) -> None:
    for artifact in transition.artifacts:
        value = (
            load_canonical_json(artifact.value)
            if artifact.value_kind == "canonical_json"
            else artifact.value.to_wire()
            if artifact.value_kind == "record"
            else artifact.value
        )
        if artifact.kind == "context_node":
            reader.context_nodes[artifact.ref] = value
        elif artifact.kind == "context_revision":
            reader.context_revisions[artifact.ref] = value
        elif artifact.kind == "private":
            reader.private[artifact.ref] = value
        else:
            reader.public[artifact.ref] = value
    checkpoint = CheckpointV1(
        parents=(previous,), state=transition.state, event_head=transition.event.id
    )
    if not checkpoint.artifact_refs:
        if transition.view.checkpoint_id != checkpoint.identity():
            raise AssertionError("derived checkpoint differs from the published runtime form")
    reader.checkpoints[checkpoint.identity()] = checkpoint


def _apply(test, derive, view, record, reader):
    transition = derive(view, record, reader)
    decoded = type(record).from_json(record.to_json())
    test.assertEqual(_signature(transition), _signature(derive(view, decoded, reader)))
    previous = view.checkpoint_id
    _persist(reader, previous, transition)
    return transition.view


def _scripted_check_view():
    fixture = make_outcome_fixture()
    old_node = fixture.graph.node(fixture.node_id)
    reader = fixture.reader
    packet = AuthorPacketV1(preferences={"choice": "blue"}, requirements={"r1": "A blue door."})
    packet_ref = reader.add(packet.to_dict(), private=True)
    script = ScriptedAuthorV1(
        answers={
            "door": {
                "mode": "declared_option_id",
                "utterance": "The blue door.",
                "value": "blue",
                "selector": "p1",
                "prerequisite_check_ids": [],
            }
        }
    )
    script_ref = reader.add(script.to_dict(), private=True)
    policy = InteractionPolicyV1(public_decisions=({"id": "door", "label": "door choice"},))
    policy_ref = reader.add(policy.to_dict())
    bindings = DecisionBindingsV1(bindings={"door": "choice"})
    bindings_ref = reader.add(bindings.to_dict(), private=True)
    interaction = InteractionContractV1(
        mode="scripted_author",
        script_ref=script_ref,
        author_packet_ref=packet_ref,
        interaction_policy_ref=policy_ref,
        decision_bindings_ref=bindings_ref,
    )
    entry = replace(
        old_node.contract.entry_contract,
        tool_allowlist=(*old_node.contract.entry_contract.tool_allowlist, "ask_author"),
    )
    budgets = replace(old_node.contract.budget_contract, max_author_calls=3)
    contract = replace(old_node.contract, entry=entry, interaction=interaction, budgets=budgets)
    node = replace(
        old_node,
        contract=contract,
        script=script,
        author_packet=packet,
        interaction_policy=policy,
        decision_bindings=bindings,
    )

    state = fixture.state
    decisions = reader.artifact(state.decisions_ref)
    disclosures = reader.artifact(state.disclosures_ref)
    decisions_ref = reader.add(decisions)
    disclosures_ref = reader.add(disclosures)
    requirements_ref = reader.add(
        {
            "record_type": "RequirementLedgerV1",
            "schema": 1,
            "active": {"r1": "A blue door."},
            "superseded": {},
        },
        private=True,
    )
    budget = reader.artifact(state.budgets_ref)
    budget["limits"]["author_calls"] = 3
    budget["consumed"]["author_calls"] = 0
    budget_ref = reader.add(budget)
    state = replace(
        state,
        author_packet_ref=packet_ref,
        requirements_ref=requirements_ref,
        decisions_ref=decisions_ref,
        disclosures_ref=disclosures_ref,
        budgets_ref=budget_ref,
    )
    view = make_view(replace(fixture, state=state), "ready_writer")
    checkpoint = CheckpointV1(state=state, event_head=state.history["head"])
    view = replace(
        view,
        root_checkpoint_id=checkpoint.identity(),
        checkpoint_id=checkpoint.identity(),
        state=state,
        budget=budget,
        node=node,
        mode=LineageMode.for_node(node),
        ancestry=CheckpointChain(checkpoint.identity(), view.context),
    )
    return fixture, view


class DeriveFoldTests(unittest.TestCase):
    def test_full_runtime_fold_and_independent_steps(self):
        fixture, view = _scripted_check_view()
        reader = fixture.reader
        author_arguments = {
            "question": "Which door?",
            "decision_ids": ["door"],
            "proposals": [{"id": "p1", "text": "The blue door"}],
            "option_refs": ["p1"],
        }

        view = _apply(
            self,
            derive_writer_turn,
            view,
            make_turn(view, content="", calls=(call("ask_author", author_arguments),)),
            reader,
        )
        request = EnvironmentStepV1(
            directive={"kind": "request_author", "source": "writer_request"}
        )
        view = _apply(self, derive_author_request, view, request, reader)
        reply = _answered_reply(fixture, view)
        view = _apply(self, derive_author_reply, view, reply, reader)

        write = call("write_file", {"path": "draft.txt", "content": "revised\n"}, "raw-write")
        action = make_turn(view, content="", calls=(write,))
        view = _apply(self, derive_writer_turn, view, action, reader)
        queued = view.state.continuation["tool_queue"][0]
        observation = ToolObservationV1(
            call_id=queued["call_id"],
            dispatch={
                "spec": {
                    "max_file_bytes": view.tool_spec.max_file_bytes,
                    "max_workspace_bytes": view.tool_spec.max_workspace_bytes,
                },
                "observation": {"ok": True, "valid": True, "result": "written"},
                "effect": {"draft.txt": {"before": "alpha\n", "after": "revised\n"}},
            },
        )
        view = _apply(self, derive_tool_result, view, observation, reader)
        view = _apply(self, derive_writer_turn, view, make_turn(view, content="finished"), reader)

        check_step = EnvironmentStepV1(directive={"kind": "request_checks"})
        requested = derive_check_request(view, check_step, reader)
        decoded_check_step = EnvironmentStepV1.from_json(check_step.to_json())
        self.assertEqual(
            _signature(requested),
            _signature(derive_check_request(view, decoded_check_step, reader)),
        )
        _persist(reader, view.checkpoint_id, requested)
        view = requested.view
        result = evaluation_result(fixture, requested)
        view = _apply(self, derive_check_result, view, result, reader)
        edge_id = next_step(view).edge_id
        transition = EnvironmentStepV1(directive={"kind": "commit_transition", "edge_id": edge_id})
        view = _apply(self, derive_transition, view, transition, reader)
        seal = EnvironmentStepV1(
            directive={"kind": "seal_outcome", "task_status": "complete", "stop_reason": None}
        )
        view = _apply(self, derive_seal, view, seal, reader)
        view = _apply(
            self,
            derive_reward,
            view,
            EnvironmentStepV1(directive={"kind": "publish_reward"}),
            reader,
        )
        self.assertEqual(view.outcome.reward_status, "available")

    def test_exhausted_context_and_member_start_folds(self):
        fixture = make_outcome_fixture()
        exhausted = make_view(fixture, "ready_writer")
        budget = load_canonical_json(canonical_bytes(exhausted.budget))
        budget["limits"]["writer_turns"] = 1
        budget["consumed"]["writer_turns"] = 1
        exhausted = replace(exhausted, budget=budget)
        reason = "writer_budget"
        stopped = _apply(
            self,
            derive_exhausted_stop,
            exhausted,
            EnvironmentStepV1(directive={"kind": "stop_exhausted", "stop_reason": reason}),
            fixture.reader,
        )
        self.assertEqual(
            stopped.checkpoint_id,
            CheckpointV1(
                parents=(exhausted.checkpoint_id,),
                state=stopped.state,
                event_head=stopped.head_event_id,
            ).identity(),
        )

        context_fixture = make_entry_fixture()
        from tests.test_task_graph_derive_context import _view

        context_view = _view(context_fixture)
        policy = ContextPolicyV1("drop")
        operation = ContextOperationInputV1(policy_ref=context_fixture.reader.add(policy.to_wire()))
        _apply(self, derive_context_operation, context_view, operation, context_fixture.reader)

        from tests.test_task_graph_derive_context import _legacy_group_entry_view, _StoreReader
        from tests.test_task_graph_group import GroupCoordinatorTests

        group_test = GroupCoordinatorTests("test_full_contract_drift_and_start_isolation")
        group_test.setUp()
        try:
            spec = group_test.group()
            reader = _StoreReader(group_test.store)
            reader.public[spec.identity()] = spec.to_wire()
            group_view = _legacy_group_entry_view(group_test)
            start = MemberStartV1(group_spec_ref=spec.identity(), ordinal=0)
            transition = derive_member_start(group_view, start, reader)
            roundtrip = MemberStartV1.from_json(start.to_json())
            self.assertEqual(
                _signature(transition),
                _signature(derive_member_start(group_view, roundtrip, reader)),
            )
            self.assertEqual(
                transition.view.checkpoint_id,
                CheckpointV1(
                    parents=(group_view.checkpoint_id,),
                    state=transition.state,
                    event_head=transition.event.id,
                ).identity(),
            )
        finally:
            group_test.doCleanups()


if __name__ == "__main__":
    unittest.main()
