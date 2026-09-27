"""Pure author-request and author-reply transitions for task-graph v1."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from writing_agent.task_graph import MessageV1, canonical_bytes, canonical_json, domain_hash
from writing_agent.task_graph_accounting import charge_tool_attempt
from writing_agent.task_graph_contracts import RequirementUpdateV1
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_derive_common import (
    DeriveKey,
    advance,
    append_context,
    build_transition,
    new_event,
    next_state,
    payload_artifact,
    wire_copy,
)
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_records import (
    AuthorReplyV1,
    EnvironmentStepV1,
    OutcomeV1,
)
from writing_agent.task_graph_scripted import (
    ScriptCoverageError,
    author_message,
    derive_feedback_request_fields,
    derive_writer_request_fields,
    frozen_prerequisite_results,
    require_unsupported_coverage,
    resolve_script_reply,
    validate_derived_author_request,
    validate_feedback_reply,
    wire_author_reply,
)
from writing_agent.task_graph_transition import (
    ArtifactReader,
    DerivedArtifact,
    LineageView,
    Transition,
)


def derive_author_request(
    view: LineageView, step: EnvironmentStepV1, reader: ArtifactReader
) -> Transition:
    """Derive the private request and charge its author-call budget."""
    directive = step.directive
    if (
        set(directive) != {"kind", "source"}
        or directive["kind"] != "request_author"
        or next_step(view).kind != "request_author"
        or next_step(view).source != directive["source"]
    ):
        raise ProjectionError("author request differs from the directive")
    if view.mode.interaction != "scripted_author" or not view.mode.ask_semantics:
        raise ProjectionError("lineage does not authorize author requests")

    source = directive["source"]
    prereqs = frozen_prerequisite_results(view=view, reader=reader)
    if source == "writer_request":
        request_fields = derive_writer_request_fields(view, reader)
    elif source == "mandatory_feedback":
        request_fields = derive_feedback_request_fields(view, prereqs)
    else:  # The input codec closes this; keep the derive fail-closed too.
        raise ProjectionError("unsupported author request source")

    consumed = view.budget["consumed"]
    limits = view.budget["limits"]
    author_calls = consumed.get("author_calls", 0)
    if author_calls >= limits.get("author_calls", 0):
        raise ProjectionError("author-call budget is exhausted")
    if author_calls >= view.node.contract.budget_contract.max_author_calls:
        raise ProjectionError("author-call contract budget is exhausted")
    request = {
        "record_type": "AuthorRequestV1",
        "schema": 1,
        "request_id": f"{view.state.position['lineage_id']}:author:{author_calls}",
        "source": source,
        **request_fields,
        "prerequisite_results": prereqs,
        "requirement_version": view.state.requirements_ref,
        "script_ref": view.node.contract.interaction_contract.script_ref,
        "author_packet_ref": view.state.author_packet_ref,
    }
    if (
        request["script_ref"] is None
        or request["author_packet_ref"] != view.node.contract.interaction_contract.author_packet_ref
    ):
        raise ProjectionError("author request differs from the admitted role contract")
    if request["requirement_version"] != view.state.requirements_ref:
        raise ProjectionError("author request must use the private active requirements")
    request_artifact = payload_artifact(request, "private")
    request_ref = request_artifact.ref

    budget = json.loads(canonical_json(view.budget))
    budget["consumed"]["author_calls"] = author_calls + 1
    budget_ref = domain_hash("payload", budget)
    continuation = wire_copy(view.state.continuation)
    if continuation["author_request"] is not None:
        raise ProjectionError("an author request is already outstanding")
    continuation["author_request"] = request_ref
    return build_transition(
        view,
        step,
        kind="external_requested",
        actor="environment",
        audience=("controller", "trainer"),
        state_changes={
            "position": {"phase": "awaiting_author"},
            "continuation": continuation,
            "budgets_ref": budget_ref,
        },
        artifacts=(request_artifact, payload_artifact(budget)),
        include_input=False,
        budget=budget,
    )


def derive_author_reply(
    view: LineageView, reply: AuthorReplyV1, reader: ArtifactReader
) -> Transition:
    """Derive one atomic author turn or a verified script-coverage termination."""
    if next_step(view).kind != "await_author_reply":
        raise ProjectionError("there is no outstanding author reply")
    request_ref = view.state.continuation["author_request"]
    if request_ref is None or reply.request_ref != request_ref:
        raise ProjectionError("author reply does not bind the outstanding request")
    request = reader.artifact(request_ref, private=True)
    validate_derived_author_request(view, request)
    script = view.node.script
    if view.mode.interaction != "scripted_author" or script is None:
        raise ProjectionError("lineage has no admitted deterministic author")

    if reply.status == "unsupported_coverage":
        return _derive_coverage_failure(view, reply, request, request_ref, script, reader)

    decisions = reader.artifact(view.state.decisions_ref)
    disclosures = reader.artifact(view.state.disclosures_ref)
    if request["source"] == "mandatory_feedback":
        validate_feedback_reply(view, reply, request)
        expected_decisions = decisions
        expected_disclosures = disclosures
    else:
        try:
            expected_decisions, expected_disclosures, expected_reply = resolve_script_reply(
                script,
                {**request, "request_ref": request_ref},
                decisions,
                disclosures,
            )
        except ScriptCoverageError as exc:
            raise ProjectionError("script cannot answer the recorded author request") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectionError("author request is not covered by the admitted script") from exc
        expected = wire_author_reply(expected_reply)
        if canonical_bytes(reply.to_wire()) != canonical_bytes(expected.to_wire()):
            raise ProjectionError("author reply differs from the admitted script")

    return _derive_answered_reply(
        view,
        reply,
        request,
        request_ref,
        expected_decisions,
        expected_disclosures,
        reader,
    )


def _derive_answered_reply(
    view: LineageView,
    reply: AuthorReplyV1,
    request: Mapping[str, Any],
    request_ref: str,
    decisions: Mapping[str, Any],
    disclosures: Mapping[str, Any],
    reader,
) -> Transition:
    source = request["source"]
    continuation = wire_copy(view.state.continuation)
    continuation["author_request"] = None
    position = wire_copy(view.state.position)
    position["phase"] = "ready_writer"
    changes: dict[str, Any] = {"continuation": continuation, "position": position}
    artifacts: list[DerivedArtifact] = []
    budget = dict(view.budget)

    if source == "writer_request":
        if request["call_id"] is None or request["action_id"] is None:
            raise ProjectionError("writer request lacks its call binding")
        budget, call_charge = charge_tool_attempt(budget)
        if call_charge != 1:
            raise ProjectionError("author acknowledgement exceeded the tool-call budget")
        continuation["next_call"] += 1
        ack_id = (
            f"{view.state.position['lineage_id']}:tool_result:"
            f"{len(view.state.history['tool_result_ids'])}"
        )
        tool_results = [*view.state.history["tool_result_ids"], ack_id]
        acknowledgement = MessageV1(
            role="tool",
            call_id=request["call_id"],
            origin=request["action_id"],
            content=(
                {
                    "type": "tool_result",
                    "call_id": request["call_id"],
                    "content": {
                        "ok": True,
                        "valid": True,
                        "result": {"status": "author_reply_follows"},
                    },
                },
            ),
        )
        history_changes = {"tool_result_ids": tool_results}
        messages = (acknowledgement, author_message(request, reply.utterance))
        artifacts.extend(
            (
                payload_artifact(dict(decisions)),
                payload_artifact(dict(disclosures)),
            )
        )
        changes["decisions_ref"] = artifacts[-2].ref
        changes["disclosures_ref"] = artifacts[-1].ref
    elif source == "mandatory_feedback":
        cursor = continuation["feedback_cursor"]
        if cursor >= len(view.mode.feedback_rules):
            raise ProjectionError("feedback cursor exceeds the admitted script")
        rule = view.mode.feedback_rules[cursor]
        if request["feedback_id"] != rule["id"]:
            raise ProjectionError("feedback reply differs from its cursor")
        continuation["feedback_cursor"] += 1
        history_changes = {}
        messages = (author_message(request, reply.utterance),)
        update_ref = rule["requirement_update_ref"]
        if update_ref is not None:
            old_ledger = reader.artifact(view.state.requirements_ref, private=True)
            update = RequirementUpdateV1.from_dict(reader.artifact(update_ref, private=True))
            if (
                not isinstance(old_ledger, Mapping)
                or set(old_ledger) != {"record_type", "schema", "active", "superseded"}
                or old_ledger["record_type"] != "RequirementLedgerV1"
                or old_ledger["schema"] != 1
                or update.supersedes not in old_ledger["active"]
                or update.id in old_ledger["active"]
                or update.id in old_ledger["superseded"]
            ):
                raise ProjectionError("authorized requirement update has no unique predecessor")
            ledger = json.loads(canonical_json(old_ledger))
            previous = ledger["active"].pop(update.supersedes)
            ledger["superseded"][update.supersedes] = previous
            ledger["active"][update.id] = update.replacement
            requirements_ref = domain_hash("payload", ledger)
            changes["requirements_ref"] = requirements_ref
            artifacts.append(payload_artifact(ledger, "private"))
    else:
        raise ProjectionError("unsupported author reply source")

    event = new_event(
        view,
        payload_artifact(reply).ref,
        kind="author_turn",
        actor="author",
        audience=("controller", "trainer", "writer"),
    )
    context, content_node, revision, charged_budget = append_context(
        replace(view, budget=budget), messages, event.id
    )
    if charged_budget is not None:
        budget = charged_budget
    changes["context_ref"] = revision.ref
    if budget != view.budget:
        budget_artifact = payload_artifact(budget)
        changes["budgets_ref"] = budget_artifact.ref
        artifacts.append(budget_artifact)
    artifacts.extend((content_node, revision))
    state = next_state(view, event, **changes, history=history_changes)
    return advance(
        view, reply, event, state, artifacts=tuple(artifacts), context=context, budget=budget
    )


def _derive_coverage_failure(
    view: LineageView,
    reply: AuthorReplyV1,
    request: Mapping[str, Any],
    request_ref: str,
    script,
    reader,
) -> Transition:
    require_unsupported_coverage(
        request,
        reply,
        request_ref,
        script,
        reader.artifact(view.state.decisions_ref),
        reader.artifact(view.state.disclosures_ref),
    )

    outcome = OutcomeV1(
        schema=1,
        task_status="unknown",
        execution_status="simulator_error",
        stop_reason="unsupported_script_coverage",
        reward_status="unavailable",
        training_eligibility="ineligible",
        candidate_checkpoint=view.checkpoint_id,
        requirement_version=None,
        checks=view.outcome.checks,
        transition_edge_id=None,
        failed_request_ref=request_ref,
        reward_ref=None,
        eligibility_ref=None,
    )
    outcome_artifact = payload_artifact(outcome)
    continuation = wire_copy(view.state.continuation)
    continuation["author_request"] = None
    position = wire_copy(view.state.position)
    position["phase"] = "terminal"
    return build_transition(
        view,
        reply,
        kind="termination_recorded",
        actor="environment",
        audience=("controller", "trainer"),
        state_changes={
            "continuation": continuation,
            "position": position,
            "outcome_ref": outcome_artifact.ref,
        },
        artifacts=(outcome_artifact,),
        include_input=False,
        outcome=outcome,
    )


DERIVES: dict[DeriveKey, Callable] = {
    ("EnvironmentStepV1", "request_author"): derive_author_request,
    "AuthorReplyV1": derive_author_reply,
}


__all__ = ["DERIVES", "derive_author_reply", "derive_author_request"]
