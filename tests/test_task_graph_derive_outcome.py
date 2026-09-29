"""Outcome derive lifecycle, authority and admission decoupling tests."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from tests.task_graph_fixtures import make_entry_fixture, make_outcome_fixture
from writing_agent.task_graph import CheckpointV1, canonical_bytes, load_canonical_json, tree_hash
from writing_agent.task_graph_controller import Directive, next_step
from writing_agent.task_graph_derive_outcome import (
    derive_check_request,
    derive_check_result,
    derive_exhausted_stop,
    derive_reward,
    derive_seal,
    derive_transition,
)
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_evaluation import (
    FAMILIES,
    EvaluationRequestV1,
    produce_evaluation_evidence,
)
from writing_agent.task_graph_records import (
    EnvironmentStepV1,
    EvaluatorResultV1,
    OutcomeV1,
)
from writing_agent.task_graph_sampling import CURRENT_ELIGIBILITY
from writing_agent.task_graph_transition import (
    CheckpointChain,
    ContextView,
    LineageMode,
    LineageView,
    ToolSpec,
    Transition,
)


def make_view(fixture, phase: str, *, files=None, consumed=None) -> LineageView:
    state = fixture.state
    position = {**state.position, "phase": phase}
    changes = {"position": position}
    if files is not None:
        changes.update(files=files, tree_hash=tree_hash(files))
    state = replace(state, **changes)
    checkpoint = CheckpointV1(state=state, event_head=state.history["head"])
    fixture.reader.checkpoints[checkpoint.identity()] = checkpoint
    context = fixture.reader.context(state.context_ref)
    revision = fixture.reader.artifact(state.context_ref, domain="context_revision")
    context_view = ContextView(
        messages=context.messages,
        sources=(None,) * len(context.messages),
        tools=context.tools,
        rendering=context.rendering,
        content_ref=revision["content_ref"],
        revision_ref=state.context_ref,
    )
    budget = copy.deepcopy(fixture.reader.artifact(state.budgets_ref))
    if consumed:
        budget["consumed"].update(consumed)
    versions = fixture.reader.artifact(state.versions_ref)
    node = fixture.graph.node(fixture.node_id)
    return LineageView(
        root_checkpoint_id=checkpoint.identity(),
        checkpoint_id=checkpoint.identity(),
        head_event_id=state.history["head"],
        state=state,
        budget=budget,
        outcome=OutcomeV1.from_dict(fixture.reader.artifact(state.outcome_ref)),
        check_statuses={},
        context=context_view,
        raw_call_ids=frozenset(),
        call_sources={},
        samples=(),
        ancestry=CheckpointChain(checkpoint.identity(), context_view),
        node=node,
        mode=LineageMode.for_node(node),
        tool_spec=ToolSpec(**versions["tool_spec"]),
    )


def persist_transition(fixture, previous: LineageView, transition: Transition) -> None:
    for artifact in transition.artifacts:
        value = (
            load_canonical_json(artifact.value)
            if artifact.value_kind == "canonical_json"
            else artifact.value.to_wire()
            if hasattr(artifact.value, "to_wire")
            else artifact.value
        )
        target = fixture.reader.private if artifact.kind == "private" else fixture.reader.public
        target[artifact.ref] = value
    checkpoint = CheckpointV1(
        parents=(previous.checkpoint_id,),
        state=transition.state,
        event_head=transition.event.id,
    )
    if checkpoint.identity() != transition.view.checkpoint_id:
        raise AssertionError("derive view checkpoint does not match a runtime checkpoint")
    fixture.reader.checkpoints[checkpoint.identity()] = checkpoint


def evaluation_result(fixture, request_transition, *, result_status=None, family_mismatch=False):
    request_ref = request_transition.state.continuation["check_requests"][0]
    request = fixture.reader.artifact(request_ref, private=True)
    check = fixture.graph.node(fixture.node_id).checks[request["check_id"]]
    family = next(
        item for item in FAMILIES.values() if item.check_version == check.evaluator_version
    )
    target = fixture.reader.checkpoint(request["target_checkpoint"])
    packet = fixture.reader.artifact(request["evaluator_packet_ref"], private=True)
    evaluation_request = EvaluationRequestV1.create(
        family.name,
        request["target_checkpoint"],
        check,
        request["evaluator_packet_ref"],
        target.state.files,
        evaluator_packet=packet,
    )
    evidence = produce_evaluation_evidence(evaluation_request).to_wire(evaluation_request)
    if family_mismatch:
        evidence["record_type"] = "FixtureFileCountEvidenceV1"
    evidence_ref = fixture.reader.add(evidence)
    status = evidence["status"] if result_status is None else result_status
    return EvaluatorResultV1(request_ref=request_ref, status=status, evidence_ref=evidence_ref)


def transition_bytes(transition: Transition) -> bytes:
    return canonical_bytes(
        {
            "event": transition.event.to_dict(),
            "state": transition.state.to_dict(),
            "artifacts": [
                {
                    "ref": item.ref,
                    "kind": item.kind,
                    "value_kind": item.value_kind,
                    "value": (
                        load_canonical_json(item.value)
                        if item.value_kind == "canonical_json"
                        else item.value.to_wire()
                        if hasattr(item.value, "to_wire")
                        else item.value.hex()
                    ),
                }
                for item in transition.artifacts
            ],
        }
    )


def round_trip(record):
    return type(record).from_dict(load_canonical_json(canonical_bytes(record.to_wire())))


class OutcomeAdmissionTests(unittest.TestCase):
    def test_none_mode_resolves_declared_evaluator_and_reward_contracts(self):
        fixture = make_outcome_fixture()
        node = fixture.graph.node(fixture.node_id)
        self.assertEqual(node.contract.interaction_contract.mode, "none")
        self.assertEqual(node.checks["nonempty"].evaluator_version, "deterministic-v1")
        self.assertEqual(node.evaluator_packet.check_ids, ("nonempty",))
        self.assertEqual(node.reward_contract.components, {"nonempty": 10_000})
        view = make_view(fixture, "checking")
        self.assertTrue(view.mode.evaluation)
        self.assertEqual(view.mode.reward, node.reward_contract)
        legacy = make_entry_fixture()
        legacy_node = legacy.graph.node(legacy.node_id)
        self.assertIsNone(legacy_node.evaluator_packet)
        self.assertIsNone(legacy_node.reward_contract)


class OutcomeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = make_outcome_fixture(incomplete_score=2500)

    def request(self, *, files=None):
        view = make_view(self.fixture, "checking", files=files)
        step = EnvironmentStepV1(directive={"kind": "request_checks"})
        self.assertEqual(next_step(view), Directive("request_checks"))
        transition = derive_check_request(view, step, self.fixture.reader)
        persist_transition(self.fixture, view, transition)
        return view, step, transition

    def result(self, request_transition, **changes):
        result = evaluation_result(self.fixture, request_transition, **changes)
        return result

    def finish_check(self, request_transition, *, result=None):
        result = result or self.result(request_transition)
        derived = derive_check_result(request_transition.view, result, self.fixture.reader)
        persist_transition(self.fixture, request_transition.view, derived)
        return result, derived

    def test_check_request_resets_statuses_and_enters_the_check_phase(self):
        view = replace(make_view(self.fixture, "checking"), check_statuses={"nonempty": "pass"})
        step = EnvironmentStepV1(directive={"kind": "request_checks"})

        transition = derive_check_request(view, step, self.fixture.reader)

        self.assertEqual(transition.view.check_statuses["nonempty"], None)
        self.assertEqual(transition.state.position["phase"], "awaiting_checks")

    def test_none_mode_unit_fold_from_checks_through_seal_and_reward(self):
        _, _, requested = self.request()
        request_ref = requested.state.continuation["check_requests"][0]
        self.assertEqual(requested.event.kind, "external_requested")
        self.assertEqual(
            requested.view.outcome.candidate_checkpoint,
            requested.view.ancestry.parent.checkpoint_id,
        )
        self.assertEqual(requested.view.outcome.checks[0]["request_ref"], request_ref)
        self.assertIsNone(requested.view.outcome.checks[0]["result_ref"])

        result, checked = self.finish_check(requested)
        self.assertEqual(checked.event.kind, "check_recorded")
        self.assertEqual(checked.view.check_statuses["nonempty"], "pass")
        self.assertEqual(
            checked.view.state.continuation["applied_responses"],
            ("rollout-fixture:check:0:nonempty",),
        )
        self.assertEqual(next_step(checked.view).kind, "commit_transition")

        edge_id = next_step(checked.view).edge_id
        transitioned = derive_transition(
            checked.view,
            EnvironmentStepV1(directive={"kind": "commit_transition", "edge_id": edge_id}),
            self.fixture.reader,
        )
        persist_transition(self.fixture, checked.view, transitioned)
        self.assertEqual(transitioned.view.outcome.task_status, "complete")
        self.assertEqual(transitioned.view.outcome.transition_edge_id, edge_id)
        self.assertEqual(next_step(transitioned.view).kind, "seal_outcome")

        sealed = derive_seal(
            transitioned.view,
            EnvironmentStepV1(
                directive={
                    "kind": "seal_outcome",
                    "task_status": "complete",
                    "stop_reason": None,
                }
            ),
            self.fixture.reader,
        )
        persist_transition(self.fixture, transitioned.view, sealed)
        before_reward = sealed.view.outcome
        self.assertIsInstance(before_reward, OutcomeV1)
        self.assertEqual(before_reward.execution_status, "valid")
        self.assertEqual(before_reward.reward_status, "pending")
        self.assertEqual(before_reward.training_eligibility, "pending")
        self.assertEqual(next_step(sealed.view).kind, "publish_reward")

        reward = derive_reward(
            sealed.view,
            EnvironmentStepV1(directive={"kind": "publish_reward"}),
            self.fixture.reader,
        )
        persist_transition(self.fixture, sealed.view, reward)
        reward_value = self.fixture.reader.artifact(reward.view.outcome.reward_ref)
        eligibility = self.fixture.reader.artifact(reward.view.outcome.eligibility_ref)
        self.assertEqual(reward.event.kind, "reward_recorded")
        self.assertEqual(reward_value["numerator"], 10_000)
        self.assertEqual(
            reward_value["check_result_refs"], [checked.view.outcome.checks[0]["result_ref"]]
        )
        self.assertEqual(eligibility["status"], "ineligible")
        self.assertEqual(eligibility["reason"], CURRENT_ELIGIBILITY.training_reason)
        self.assertFalse(CURRENT_ELIGIBILITY.native_on_policy_eligible)
        self.assertEqual(CURRENT_ELIGIBILITY.training_status, "ineligible")
        for field in (
            "task_status",
            "execution_status",
            "stop_reason",
            "candidate_checkpoint",
            "requirement_version",
            "checks",
            "transition_edge_id",
            "failed_request_ref",
        ):
            self.assertEqual(
                getattr(reward.view.outcome, field), getattr(before_reward, field), field
            )
        self.assertEqual(reward.view.outcome.reward_status, "available")
        self.assertEqual(reward.view.outcome.training_eligibility, "ineligible")

    def test_failing_candidate_cannot_transition_and_gets_incomplete_reward(self):
        _, _, requested = self.request(files={"draft.txt": ""})
        failed = self.result(requested)
        self.assertEqual(failed.status, "fail")
        _, checked = self.finish_check(requested, result=failed)
        self.assertEqual(
            next_step(checked.view),
            Directive(
                "seal_outcome", task_status="incomplete", stop_reason="required_check_failed"
            ),
        )
        forged_transition = EnvironmentStepV1(
            directive={"kind": "commit_transition", "edge_id": "legacy-terminate"}
        )
        with self.assertRaises(ProjectionError):
            derive_transition(checked.view, forged_transition, self.fixture.reader)

        sealed = derive_seal(
            checked.view,
            EnvironmentStepV1(
                directive={
                    "kind": "seal_outcome",
                    "task_status": "incomplete",
                    "stop_reason": "required_check_failed",
                }
            ),
            self.fixture.reader,
        )
        persist_transition(self.fixture, checked.view, sealed)
        reward = derive_reward(
            sealed.view,
            EnvironmentStepV1(directive={"kind": "publish_reward"}),
            self.fixture.reader,
        )
        persist_transition(self.fixture, sealed.view, reward)
        result = self.fixture.reader.artifact(reward.view.outcome.reward_ref)
        self.assertEqual(result["numerator"], 2500)
        self.assertEqual(result["components"]["nonempty"]["status"], "fail")

    def test_evaluator_family_forged_status_stale_and_duplicate_results_reject(self):
        _, _, requested = self.request()
        with self.assertRaises(ProjectionError):
            derive_check_result(
                requested.view, self.result(requested, family_mismatch=True), self.fixture.reader
            )

        mismatched_status = self.result(requested, result_status="fail")
        with self.assertRaises(ProjectionError):
            derive_check_result(requested.view, mismatched_status, self.fixture.reader)

        _, _, failing_candidate = self.request(files={"draft.txt": ""})
        forged_pass = self.result(failing_candidate, result_status="pass")
        with self.assertRaises(ProjectionError):
            derive_check_result(failing_candidate.view, forged_pass, self.fixture.reader)

        stale = EvaluatorResultV1(
            request_ref="f" * 64,
            status="pass",
            evidence_ref=mismatched_status.evidence_ref,
        )
        with self.assertRaises(ProjectionError):
            derive_check_result(requested.view, stale, self.fixture.reader)

        result, checked = self.finish_check(requested)
        with self.assertRaises(ProjectionError):
            derive_check_result(checked.view, result, self.fixture.reader)

        with self.assertRaises(ValueError):
            EvaluatorResultV1(
                request_ref="f" * 64,
                status="unavailable",
                evidence_ref="e" * 64,
            )

    def test_transition_requires_selected_edge_and_terminate_effect(self):
        _, _, requested = self.request()
        _, checked = self.finish_check(requested)
        with self.assertRaises(ProjectionError):
            derive_transition(
                checked.view,
                EnvironmentStepV1(
                    directive={"kind": "commit_transition", "edge_id": "forged-edge"}
                ),
                self.fixture.reader,
            )
        node = checked.view.node
        nonterminal_edge = replace(node.edges[0], effect="advance", target_node=node.spec.id)
        node = replace(node, edges=(nonterminal_edge,))
        view = replace(checked.view, node=node)
        with self.assertRaises(ProjectionError):
            derive_transition(
                view,
                EnvironmentStepV1(
                    directive={"kind": "commit_transition", "edge_id": nonterminal_edge.edge_id}
                ),
                self.fixture.reader,
            )

    def test_exhausted_stop_is_directed_and_seals_current_candidate(self):
        view = make_view(self.fixture, "ready_writer", consumed={"writer_turns": 5})
        self.assertEqual(next_step(view), Directive("stop_exhausted", stop_reason="writer_budget"))
        stopped = derive_exhausted_stop(
            view,
            EnvironmentStepV1(directive={"kind": "stop_exhausted", "stop_reason": "writer_budget"}),
            self.fixture.reader,
        )
        self.assertEqual(stopped.event.kind, "termination_recorded")
        self.assertEqual(stopped.event.actor, "writer_runtime")
        self.assertEqual(stopped.view.outcome.task_status, "incomplete")
        self.assertEqual(stopped.view.outcome.candidate_checkpoint, view.checkpoint_id)
        self.assertEqual(stopped.view.outcome.stop_reason, "writer_budget")
        with self.assertRaises(ProjectionError):
            derive_exhausted_stop(
                view,
                EnvironmentStepV1(
                    directive={"kind": "stop_exhausted", "stop_reason": "read_budget"}
                ),
                self.fixture.reader,
            )

    def test_decode_fixed_point_covers_every_derived_step(self):
        _, step, requested = self.request()
        repeated_request = derive_check_request(
            make_view(self.fixture, "checking"), round_trip(step), self.fixture.reader
        )
        self.assertEqual(transition_bytes(requested), transition_bytes(repeated_request))

        result = self.result(requested)
        checked = derive_check_result(requested.view, result, self.fixture.reader)
        repeated_check = derive_check_result(
            requested.view, round_trip(result), self.fixture.reader
        )
        self.assertEqual(transition_bytes(checked), transition_bytes(repeated_check))
        persist_transition(self.fixture, requested.view, checked)

        transition_step = EnvironmentStepV1(
            directive={"kind": "commit_transition", "edge_id": next_step(checked.view).edge_id}
        )
        transitioned = derive_transition(checked.view, transition_step, self.fixture.reader)
        self.assertEqual(
            transition_bytes(transitioned),
            transition_bytes(
                derive_transition(checked.view, round_trip(transition_step), self.fixture.reader)
            ),
        )
        persist_transition(self.fixture, checked.view, transitioned)
        seal_step = EnvironmentStepV1(
            directive={"kind": "seal_outcome", "task_status": "complete", "stop_reason": None}
        )
        sealed = derive_seal(transitioned.view, seal_step, self.fixture.reader)
        self.assertEqual(
            transition_bytes(sealed),
            transition_bytes(
                derive_seal(transitioned.view, round_trip(seal_step), self.fixture.reader)
            ),
        )
        persist_transition(self.fixture, transitioned.view, sealed)
        reward_step = EnvironmentStepV1(directive={"kind": "publish_reward"})
        reward = derive_reward(sealed.view, reward_step, self.fixture.reader)
        self.assertEqual(
            transition_bytes(reward),
            transition_bytes(
                derive_reward(sealed.view, round_trip(reward_step), self.fixture.reader)
            ),
        )

    def test_reward_numerator_is_not_a_caller_supplied_field(self):
        with self.assertRaises((TypeError, ValueError)):
            EnvironmentStepV1(directive={"kind": "publish_reward", "numerator": 9_999})


if __name__ == "__main__":
    unittest.main()
