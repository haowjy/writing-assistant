"""Pure author-request and author-reply transitions for task-graph v1."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from writing_agent.task_graph import (
    CheckpointV1,
    EventV1,
    MessageV1,
    canonical_bytes,
    canonical_json,
    domain_hash,
    load_canonical_json,
)
from writing_agent.task_graph_accounting import charge_tool_attempt
from writing_agent.task_graph_checks import applicable_checks
from writing_agent.task_graph_contracts import RequirementUpdateV1
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_records import (
    AuthorReplyV1,
    ContextContentV1,
    ContextRevisionV1,
    EnvironmentStepV1,
    OutcomeV1,
)
from writing_agent.task_graph_scripted import (
    ScriptCoverageError,
    frozen_prerequisite_results,
    resolve_script_reply,
    validate_ask_semantics,
)
from writing_agent.task_graph_transition import (
    ArtifactReader,
    CheckpointChain,
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
        request_fields = _writer_request_fields(view, reader, prereqs)
    elif source == "mandatory_feedback":
        request_fields = _feedback_request_fields(view, prereqs)
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
    request_ref = domain_hash("payload", request)

    budget = json.loads(canonical_json(view.budget))
    budget["consumed"]["author_calls"] = author_calls + 1
    budget_ref = domain_hash("payload", budget)
    position = _copy(view.state.position)
    position["phase"] = "awaiting_author"
    continuation = _copy(view.state.continuation)
    if continuation["author_request"] is not None:
        raise ProjectionError("an author request is already outstanding")
    continuation["author_request"] = request_ref
    state = _state_after(
        view,
        step,
        kind="external_requested",
        actor="environment",
        audience=("controller", "trainer"),
        changes={
            "position": position,
            "continuation": continuation,
            "budgets_ref": budget_ref,
        },
        artifacts=(
            _artifact(request_ref, request, "private"),
            _artifact(budget_ref, budget, "artifact"),
        ),
        reader=reader,
    )
    return state


def _writer_request_fields(
    view: LineageView, reader: ArtifactReader, prereqs: Mapping[str, Any]
) -> dict[str, Any]:
    continuation = view.state.continuation
    queue = continuation["tool_queue"]
    cursor = continuation["next_call"]
    if cursor >= len(queue):
        raise ProjectionError("writer author request has no pending call")
    call = queue[cursor]
    source = view.call_sources.get(call["call_id"])
    if (
        call["name"] != "ask_author"
        or source is None
        or source.queue_index != cursor
        or not view.state.history["action_ids"]
        or source.action_id != view.state.history["action_ids"][-1]
    ):
        raise ProjectionError("writer author request does not bind the pending call")
    arguments = _copy(call["arguments"])
    try:
        validate_ask_semantics(arguments, view.node, reader.artifact(view.state.decisions_ref))
    except (TypeError, ValueError, KeyError) as exc:
        raise ProjectionError("writer author request has invalid ask semantics") from exc
    policy = view.node.interaction_policy
    public_ids = [item["id"] for item in policy.public_decisions]
    selected = set(arguments["decision_ids"])
    decision_ids = [identity for identity in public_ids if identity in selected]
    if not decision_ids:
        raise ProjectionError("writer author request has no public decision")
    return {
        "action_id": source.action_id,
        "call_id": call["call_id"],
        "feedback_id": None,
        "arguments": arguments,
        "decision_ids": decision_ids,
    }


def _feedback_request_fields(view: LineageView, prereqs: Mapping[str, Any]) -> dict[str, Any]:
    cursor = view.state.continuation["feedback_cursor"]
    rules = view.mode.feedback_rules
    if cursor >= len(rules) or view.state.continuation["check_requests"]:
        raise ProjectionError("mandatory feedback is not ready")
    rule = rules[cursor]
    progress = applicable_checks(view.node, cursor)
    expected_phase = "awaiting_checks" if progress else "checking"
    if view.state.position["phase"] != expected_phase:
        raise ProjectionError("mandatory feedback skipped its progress checks")
    if any(
        prereqs.get(check_id, {}).get("status") != "pass"
        for check_id in rule["prerequisite_check_ids"]
    ):
        raise ProjectionError("mandatory feedback prerequisites did not pass")
    consumed, limits = view.budget["consumed"], view.budget["limits"]
    if any(consumed.get(key, 0) >= limits.get(key, 0) for key in ("author_calls", "writer_turns")):
        raise ProjectionError("mandatory feedback budget is exhausted")
    return {
        "action_id": None,
        "call_id": None,
        "feedback_id": rule["id"],
        "arguments": None,
        "decision_ids": [],
    }


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
    _validate_request(view, request)
    script = view.node.script
    if view.mode.interaction != "scripted_author" or script is None:
        raise ProjectionError("lineage has no admitted deterministic author")

    if reply.status == "unsupported_coverage":
        return _derive_coverage_failure(view, reply, request, request_ref, script, reader)

    decisions = reader.artifact(view.state.decisions_ref)
    disclosures = reader.artifact(view.state.disclosures_ref)
    if request["source"] == "mandatory_feedback":
        _validate_feedback_reply(view, reply, request)
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
        expected = _wire_reply(expected_reply)
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


def _validate_request(view: LineageView, request: Any) -> None:
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


def _validate_feedback_reply(
    view: LineageView, reply: AuthorReplyV1, request: Mapping[str, Any]
) -> None:
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
    continuation = _copy(view.state.continuation)
    continuation["author_request"] = None
    position = _copy(view.state.position)
    position["phase"] = "ready_writer"
    changes: dict[str, Any] = {"continuation": continuation, "position": position}
    artifacts: list[DerivedArtifact] = []

    if source == "writer_request":
        if request["call_id"] is None or request["action_id"] is None:
            raise ProjectionError("writer request lacks its call binding")
        budget, call_charge = charge_tool_attempt(dict(view.budget))
        if call_charge != 1:
            raise ProjectionError("author acknowledgement exceeded the tool-call budget")
        budget_ref = domain_hash("payload", budget)
        changes["budgets_ref"] = budget_ref
        artifacts.append(_artifact(budget_ref, budget, "artifact"))
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
                _artifact(domain_hash("payload", dict(decisions)), dict(decisions), "artifact"),
                _artifact(domain_hash("payload", dict(disclosures)), dict(disclosures), "artifact"),
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
            artifacts.append(_artifact(requirements_ref, ledger, "private"))
    else:
        raise ProjectionError("unsupported author reply source")

    transition = _state_after(
        view,
        reply,
        kind="author_turn",
        actor="author",
        audience=("controller", "trainer", "writer"),
        changes=changes,
        history_changes=history_changes,
        context_messages=messages,
        artifacts=tuple(artifacts),
        reader=reader,
    )
    return transition


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
    outcome_ref = domain_hash("payload", outcome.to_wire())
    continuation = _copy(view.state.continuation)
    continuation["author_request"] = None
    position = _copy(view.state.position)
    position["phase"] = "terminal"
    return _state_after(
        view,
        reply,
        kind="termination_recorded",
        actor="environment",
        audience=("controller", "trainer"),
        changes={
            "continuation": continuation,
            "position": position,
            "outcome_ref": outcome_ref,
        },
        artifacts=(_artifact(outcome_ref, outcome, "artifact"),),
        reader=reader,
    )


def _author_message(request: Mapping[str, Any], utterance: str) -> MessageV1:
    return MessageV1(
        role="user",
        origin=request["request_id"],
        content=({"type": "text", "text": utterance},),
    )


def _wire_reply(legacy_reply: Mapping[str, Any]) -> AuthorReplyV1:
    selected = {
        decision_id: ([] if proposal_id is None else [proposal_id])
        for decision_id, proposal_id in legacy_reply["selected_proposals"].items()
    }
    return AuthorReplyV1(
        request_ref=legacy_reply["request_ref"],
        status="answered",
        utterance=legacy_reply["utterance"],
        decision_ids=legacy_reply["decision_ids"],
        selected_proposals=selected,
    )


def _state_after(
    view: LineageView,
    input_record,
    *,
    kind: str,
    actor: str,
    audience: tuple[str, ...],
    changes: Mapping[str, Any],
    reader: ArtifactReader,
    artifacts: tuple[DerivedArtifact, ...] = (),
    history_changes: Mapping[str, Any] = (),
    context_messages: tuple[MessageV1, ...] = (),
) -> Transition:
    payload_ref = domain_hash("payload", input_record.to_wire())
    event = EventV1(
        previous=view.state.history["head"],
        seq=view.state.history["seq"] + 1,
        lineage_id=view.state.position["lineage_id"],
        rollout_id=view.state.position["lineage_id"],
        node_visit_id=view.state.position["visit_id"],
        kind=kind,
        actor=actor,
        audience=audience,
        payload_ref=payload_ref,
        versions_ref=view.state.versions_ref,
        provenance_ref=view.state.provenance_ref,
    )
    history = _copy(view.state.history)
    history.update(history_changes)
    history["head"] = event.id
    history["seq"] += 1
    state = replace(view.state, history=history, **changes)

    context = view.context
    new_artifacts = list(artifacts)
    if context_messages:
        prior_revision = ContextRevisionV1.from_dict(
            reader.artifact(view.context.revision_ref, domain="context_revision")
        )
        content = ContextContentV1(
            parent_ref=view.context.content_ref,
            messages=context_messages,
            tools=None,
            rendering=None,
        )
        revision = ContextRevisionV1(
            content_ref=content.identity(),
            event_head=event.id,
            provenance_refs=(*prior_revision.provenance_refs, event.id),
        )
        context = replace(
            context,
            messages=(*view.context.messages, *context_messages),
            sources=(*view.context.sources, *((event.id,) * len(context_messages))),
            content_ref=content.identity(),
            revision_ref=revision.identity(),
        )
        state = replace(state, context_ref=revision.identity())
        new_artifacts.extend(
            (
                DerivedArtifact(content.identity(), content, "context_node"),
                DerivedArtifact(revision.identity(), revision, "context_revision"),
            )
        )

    checkpoint = CheckpointV1(
        parents=(view.checkpoint_id,), state=state, event_head=event.id, artifact_refs=()
    )
    next_view = replace(
        view,
        checkpoint_id=checkpoint.identity(),
        head_event_id=event.id,
        state=state,
        budget=_next_value(
            state.budgets_ref, view.state.budgets_ref, view.budget, new_artifacts, reader
        ),
        outcome=_next_value(
            state.outcome_ref,
            view.state.outcome_ref,
            view.outcome,
            new_artifacts,
            reader,
            OutcomeV1,
        ),
        context=context,
        ancestry=CheckpointChain(checkpoint.identity(), context, view.ancestry),
    )
    return Transition(event, input_record, state, tuple(new_artifacts), next_view)


def _next_value(ref, old_ref, old_value, artifacts, reader, record_type=None):
    if ref == old_ref:
        return old_value
    for artifact in artifacts:
        if artifact.ref == ref:
            value = (
                load_canonical_json(artifact.value)
                if isinstance(artifact.value, bytes)
                else artifact.value
            )
            body = value.to_wire() if hasattr(value, "to_wire") else value
            return record_type.from_dict(body) if record_type else body
    body = reader.artifact(ref)
    return record_type.from_dict(body) if record_type else body


def _artifact(ref: str, value: Any, kind: str) -> DerivedArtifact:
    return DerivedArtifact(
        ref, canonical_bytes(value.to_wire() if hasattr(value, "to_wire") else value), kind
    )


def _copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_copy(item) for item in value]
    return value


DERIVES: dict[str, Callable] = {
    "EnvironmentStepV1": derive_author_request,
    "AuthorReplyV1": derive_author_reply,
}


__all__ = ["DERIVES", "derive_author_reply", "derive_author_request"]
