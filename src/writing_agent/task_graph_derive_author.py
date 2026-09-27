"""Pure author-request and author-reply transitions for task-graph v1."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from writing_agent.task_graph import MessageV1, canonical_bytes, canonical_json, domain_hash, thaw
from writing_agent.task_graph_accounting import charge_tool_attempt
from writing_agent.task_graph_checks import applicable_checks
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
)
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_records import (
    AuthorReplyV1,
    EnvironmentStepV1,
    OutcomeV1,
)
from writing_agent.task_graph_scripted import (
    ScriptCoverageError,
    resolve_script_reply,
    validate_ask_semantics,
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
    prereqs = {}
    for check in view.outcome.checks:
        result_ref = check["result_ref"]
        if result_ref is None:
            continue
        request = reader.artifact(check["request_ref"], private=True)
        result = reader.artifact(result_ref)
        prereqs[request["check_id"]] = {"result_ref": result_ref, "status": result["status"]}

    if source == "writer_request":
        continuation = view.state.continuation
        cursor = continuation["next_call"]
        queue = continuation["tool_queue"]
        if cursor >= len(queue):
            raise ProjectionError("writer author request has no pending call")
        call = queue[cursor]
        call_source = view.call_sources.get(call["call_id"])
        if (
            call["name"] != "ask_author"
            or call_source is None
            or call_source.queue_index != cursor
            or not view.state.history["action_ids"]
            or call_source.action_id != view.state.history["action_ids"][-1]
        ):
            raise ProjectionError("writer author request does not bind the pending call")
        arguments = thaw(call["arguments"])
        try:
            validate_ask_semantics(arguments, view.node, reader.artifact(view.state.decisions_ref))
        except (TypeError, ValueError, KeyError) as exc:
            raise ProjectionError("writer author request has invalid ask semantics") from exc
        public_ids = [item["id"] for item in view.node.interaction_policy.public_decisions]
        selected = set(arguments["decision_ids"])
        decision_ids = [identity for identity in public_ids if identity in selected]
        if not decision_ids:
            raise ProjectionError("writer author request has no public decision")
        request_fields = {
            "action_id": call_source.action_id,
            "call_id": call["call_id"],
            "feedback_id": None,
            "arguments": arguments,
            "decision_ids": decision_ids,
        }
    elif source == "mandatory_feedback":
        cursor = view.state.continuation["feedback_cursor"]
        rules = view.mode.feedback_rules
        if cursor >= len(rules) or view.state.continuation["check_requests"]:
            raise ProjectionError("mandatory feedback is not ready")
        rule = rules[cursor]
        expected_phase = "awaiting_checks" if applicable_checks(view.node, cursor) else "checking"
        if view.state.position["phase"] != expected_phase:
            raise ProjectionError("mandatory feedback skipped its progress checks")
        if any(
            prereqs.get(check_id, {}).get("status") != "pass"
            for check_id in rule["prerequisite_check_ids"]
        ):
            raise ProjectionError("mandatory feedback prerequisites did not pass")
        consumed, limits = view.budget["consumed"], view.budget["limits"]
        if any(
            consumed.get(key, 0) >= limits.get(key, 0) for key in ("author_calls", "writer_turns")
        ):
            raise ProjectionError("mandatory feedback budget is exhausted")
        request_fields = {
            "action_id": None,
            "call_id": None,
            "feedback_id": rule["id"],
            "arguments": None,
            "decision_ids": [],
        }
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
    continuation = thaw(view.state.continuation)
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
    interaction = view.node.contract.interaction_contract
    expected_request_id = (
        f"{view.state.position['lineage_id']}:author:"
        f"{view.budget['consumed'].get('author_calls', 0) - 1}"
    )
    if (
        not isinstance(request, Mapping)
        or request.get("record_type") != "AuthorRequestV1"
        or request.get("schema") != 1
        or request.get("requirement_version") != view.state.requirements_ref
        or request.get("author_packet_ref") != view.state.author_packet_ref
        or request.get("author_packet_ref") != interaction.author_packet_ref
        or request.get("script_ref") != interaction.script_ref
        or request.get("request_id") != expected_request_id
    ):
        raise ProjectionError("outstanding author request is not a derived role request")
    if request["source"] not in {"writer_request", "mandatory_feedback"}:
        raise ProjectionError("unsupported author request source")
    script = view.node.script
    if view.mode.interaction != "scripted_author" or script is None:
        raise ProjectionError("lineage has no admitted deterministic author")

    if reply.status == "unsupported_coverage":
        return _derive_coverage_failure(view, reply, request, request_ref, script, reader)

    decisions = reader.artifact(view.state.decisions_ref)
    disclosures = reader.artifact(view.state.disclosures_ref)
    if request["source"] == "mandatory_feedback":
        cursor = view.state.continuation["feedback_cursor"]
        if cursor >= len(view.mode.feedback_rules):
            raise ProjectionError("feedback cursor exceeds the admitted script")
        rule = view.mode.feedback_rules[cursor]
        if (
            request["feedback_id"] != rule["id"]
            or request["action_id"] is not None
            or request["call_id"] is not None
            or request["arguments"] is not None
            or request["decision_ids"]
            or reply.decision_ids
            or reply.selected_proposals
            or reply.utterance != rule["utterance"]
        ):
            raise ProjectionError("author reply differs from mandatory feedback")
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
        selected = {
            decision_id: ([] if proposal_id is None else [proposal_id])
            for decision_id, proposal_id in expected_reply["selected_proposals"].items()
        }
        expected = AuthorReplyV1(
            request_ref=expected_reply["request_ref"],
            status="answered",
            utterance=expected_reply["utterance"],
            decision_ids=expected_reply["decision_ids"],
            selected_proposals=selected,
        )
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
    continuation = thaw(view.state.continuation)
    continuation["author_request"] = None
    position = thaw(view.state.position)
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
        messages = (acknowledgement, _author_message(request, reply.utterance))
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
        messages = (_author_message(request, reply.utterance),)
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
    if request["source"] != "writer_request":
        raise ProjectionError("mandatory feedback cannot claim unsupported coverage")
    if reply.decision_ids or reply.selected_proposals:
        raise ProjectionError("coverage failure cannot disclose decisions")
    try:
        resolve_script_reply(
            script,
            {**request, "request_ref": request_ref},
            reader.artifact(view.state.decisions_ref),
            reader.artifact(view.state.disclosures_ref),
        )
    except ScriptCoverageError:
        pass
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("author request is malformed") from exc
    else:
        raise ProjectionError("script covers the request; coverage failure is forged")

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
    continuation = thaw(view.state.continuation)
    continuation["author_request"] = None
    position = thaw(view.state.position)
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


def _author_message(request: Mapping[str, Any], utterance: str) -> MessageV1:
    return MessageV1(
        role="user",
        origin=request["request_id"],
        content=({"type": "text", "text": utterance},),
    )


DERIVES: dict[DeriveKey, Callable] = {
    ("EnvironmentStepV1", "request_author"): derive_author_request,
    "AuthorReplyV1": derive_author_reply,
}


__all__ = ["DERIVES", "derive_author_reply", "derive_author_request"]
