"""Pure writer-turn and tool-result transitions for the V1 task-graph seam."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from writing_agent.task_graph import (
    CheckpointV1,
    EnvironmentStateV1,
    EventV1,
    MessageV1,
    canonical_bytes,
    canonical_json,
    domain_hash,
    tree_hash,
)
from writing_agent.task_graph_accounting import (
    observation_read_tokens,
    sampled_usage_charge,
    tool_error,
    tool_result_charge,
)
from writing_agent.task_graph_calls import (
    ToolQueueEntry,
    apply_effect,
    parse_calls,
    tool_effect_contract,
)
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_errors import (
    AdapterContractError,
    ProjectionError,
    WriterRuntimeError,
)
from writing_agent.task_graph_records import (
    ContextContentV1,
    ContextRevisionV1,
    OutcomeV1,
    ToolObservationV1,
    WriterTurnV1,
    decode_canonical_value,
)
from writing_agent.task_graph_sampling import (
    WriterTurnSamplingBindingV1,
    bind_group_sampling_claims,
    decode_and_bind_sampling,
)
from writing_agent.task_graph_scripted import validate_ask_semantics
from writing_agent.task_graph_transition import (
    CallSource,
    ContextView,
    DerivedArtifact,
    LineageView,
    SampleRef,
    Transition,
)

READ_BUDGET_EXCEEDED = "Read-token budget exceeded"


def derive_writer_turn(view: LineageView, turn: WriterTurnV1, reader: Any) -> Transition:
    """Derive the one event, state and context caused by a sampled writer turn."""
    directive = _directive(view)
    if directive.kind != "sample_writer":
        raise ProjectionError("writer turn is not the next legal step")
    if not isinstance(turn, WriterTurnV1):
        raise ProjectionError("writer turn input must use its strict wire codec")

    action_ordinal = len(view.state.history["action_ids"])
    action_id = f"{view.state.position['lineage_id']}:action:{action_ordinal}"
    if turn.action_id != action_id or turn.context_revision_ref != view.context.revision_ref:
        raise ProjectionError("writer turn is not bound to the active action and context")

    try:
        decode_and_bind_sampling(WriterTurnSamplingBindingV1(turn, view.context, reader))
    except (AdapterContractError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ProjectionError):
            raise
        raise ProjectionError("writer sampling evidence is invalid") from exc

    if view.group is not None:
        _bind_group_sampling(view, turn, reader)

    content = turn.message.content
    if content is not None and not isinstance(content, str):
        raise ProjectionError("sampled assistant content must be text or null")
    content = "" if content is None else content

    usage = turn.usage
    next_budget, exceeded = sampled_usage_charge(dict(view.budget), usage)
    turn_ref = domain_hash("payload", turn.to_wire())
    input_artifact = DerivedArtifact(turn_ref, turn, "artifact")
    if exceeded is not None:
        stop_reason = f"{exceeded}_budget"
        outcome = _overrun_outcome(view, stop_reason)
        outcome_ref = outcome.identity()
        event = _event(
            view,
            turn_ref,
            "budget_charged",
            "writer_runtime",
            ("controller", "evaluator", "trainer"),
        )
        state = _state_after(
            view.state,
            event,
            budgets_ref=domain_hash("payload", next_budget),
            outcome_ref=outcome_ref,
            phase="terminal",
        )
        artifacts = (
            input_artifact,
            _payload_artifact(state.budgets_ref, next_budget),
            DerivedArtifact(outcome_ref, outcome, "artifact"),
        )
        result_view = _next_view(
            view,
            event,
            state,
            budget=next_budget,
            outcome=outcome,
            samples=(*view.samples, SampleRef(action_id, event.id, turn_ref, "budget_stop")),
        )
        return Transition(event, turn, state, artifacts, result_view)

    queue = _parse_turn_calls(view, turn, reader)

    if _requires_usage_evidence(next_budget, usage):
        raise ProjectionError("token-limited writer turn lacks usage evidence")

    assistant = _assistant_message(action_id, content, turn, queue)
    event = _event(view, turn_ref, "writer_action", "writer", ("controller", "trainer", "writer"))
    context, content_node, revision = _append_context(view.context, assistant, event.id)
    budget = _charge_context_append(next_budget, context, content_node, revision)
    budget_ref = (
        domain_hash("payload", budget)
        if budget is not None
        else domain_hash("payload", next_budget)
    )
    if budget is None:
        budget = next_budget

    continuation = view.state.to_dict()["continuation"]
    continuation["tool_queue"] = [
        {"call_id": item.call_id, "name": item.name, "arguments": item.arguments} for item in queue
    ]
    continuation["next_call"] = 0
    phase = "ready_writer" if queue else "checking"
    state = _state_after(
        view.state,
        event,
        budgets_ref=budget_ref,
        context_ref=revision.identity(),
        continuation=continuation,
        phase=phase,
        action_ids=(*view.state.history["action_ids"], action_id),
    )
    sources = dict(view.call_sources)
    for index, call in enumerate(queue):
        if call.call_id in sources:
            raise ProjectionError("writer turn reuses a logical tool-call ID")
        sources[call.call_id] = CallSource(action_id, index)
    raw_call_ids = _updated_raw_call_ids(view.raw_call_ids, turn)
    artifacts = (
        input_artifact,
        _payload_artifact(budget_ref, budget),
        DerivedArtifact(content_node.identity(), content_node, "context_node"),
        DerivedArtifact(revision.identity(), revision, "context_revision"),
    )
    result_view = _next_view(
        view,
        event,
        state,
        context=context,
        budget=budget,
        raw_call_ids=raw_call_ids,
        call_sources=sources,
        samples=(*view.samples, SampleRef(action_id, event.id, turn_ref, "action")),
    )
    return Transition(event, turn, state, artifacts, result_view)


def derive_tool_result(view: LineageView, obs: ToolObservationV1, reader: Any) -> Transition:
    """Derive a queued tool result, its exact effect and visible observation."""
    directive = _directive(view)
    continuation = view.state.continuation
    queue = continuation["tool_queue"]
    cursor = continuation["next_call"]
    if (
        directive.kind != "execute_tool"
        or directive.call_index != cursor
        or not isinstance(obs, ToolObservationV1)
        or cursor >= len(queue)
    ):
        raise ProjectionError("tool result is not the next legal step")
    queued = queue[cursor]
    if obs.call_id != queued["call_id"]:
        raise ProjectionError("tool result does not name the queued call")

    source = view.call_sources.get(obs.call_id)
    if source is None or source.queue_index != cursor:
        raise ProjectionError("tool result has no matching writer-call source")
    parsed = _source_call(view, source, reader)
    if _queue_body(parsed) != dict(queued):
        raise ProjectionError("queued call differs from its sampled source")

    predispatch_error = tool_error(dict(view.budget), parsed.rejection, parsed.name)
    if predispatch_error is not None:
        if obs.dispatch is not None:
            raise ProjectionError("pre-dispatch tool error unexpectedly executed")
        observation = predispatch_error
        files_after = dict(view.state.files)
        read_charge = 0
    else:
        if obs.dispatch is None:
            raise ProjectionError("eligible tool call has no dispatch observation")
        observation = obs.dispatch["observation"]
        expected_spec = {
            "max_file_bytes": view.tool_spec.max_file_bytes,
            "max_workspace_bytes": view.tool_spec.max_workspace_bytes,
        }
        if canonical_bytes(obs.dispatch["spec"]) != canonical_bytes(expected_spec):
            raise ProjectionError("tool dispatch spec differs from the pinned tool spec")
        try:
            files_after = apply_effect(view.state.files, obs.dispatch["effect"])
            tool_effect_contract(
                parsed.name,
                parsed.arguments,
                view.state.files,
                files_after,
                observation["ok"],
                max_file_bytes=view.tool_spec.max_file_bytes,
                max_workspace_bytes=view.tool_spec.max_workspace_bytes,
                storage_bytes_limit=view.budget["limits"]["storage_bytes"],
            )
            read_charge = observation_read_tokens(
                observation, parsed.name, view.budget["read_tokenizer"]
            )
        except (AdapterContractError, TypeError, ValueError) as exc:
            raise ProjectionError("tool dispatch violates the pinned effect contract") from exc
        remaining_reads = view.budget["limits"].get("read_tokens", 0) - view.budget["consumed"].get(
            "read_tokens", 0
        )
        if read_charge > remaining_reads:
            observation = {"ok": False, "valid": True, "error": READ_BUDGET_EXCEEDED}
            files_after = dict(view.state.files)
            read_charge = 0

    next_budget, _charge = tool_result_charge(
        dict(view.budget), view.state.files, files_after, read_charge
    )
    result_id = (
        f"{view.state.position['lineage_id']}:tool_result:"
        f"{len(view.state.history['tool_result_ids'])}"
    )
    message = MessageV1(
        role="tool",
        call_id=obs.call_id,
        origin=source.action_id,
        content=(
            {
                "type": "tool_result",
                "call_id": obs.call_id,
                "content": observation,
            },
        ),
        loss_eligible=False,
    )
    obs_ref = domain_hash("payload", obs.to_wire())
    event = _event(view, obs_ref, "tool_result", "environment", ("controller", "trainer", "writer"))
    context, content_node, revision = _append_context(view.context, message, event.id)
    budget = _charge_context_append(next_budget, context, content_node, revision)
    if budget is None:
        budget = next_budget
    budget_ref = domain_hash("payload", budget)
    next_continuation = view.state.to_dict()["continuation"]
    next_continuation["next_call"] = cursor + 1
    state = _state_after(
        view.state,
        event,
        files=files_after,
        budgets_ref=budget_ref,
        context_ref=revision.identity(),
        continuation=next_continuation,
        tool_result_ids=(*view.state.history["tool_result_ids"], result_id),
    )
    artifacts = (
        DerivedArtifact(obs_ref, obs, "artifact"),
        _payload_artifact(budget_ref, budget),
        DerivedArtifact(content_node.identity(), content_node, "context_node"),
        DerivedArtifact(revision.identity(), revision, "context_revision"),
    )
    result_view = _next_view(view, event, state, context=context, budget=budget)
    return Transition(event, obs, state, artifacts, result_view)


def _directive(view: LineageView):
    try:
        directive = next_step(view)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("lineage view has no valid next directive") from exc
    if view.head_event_id != view.state.history["head"]:
        raise ProjectionError("lineage view event head differs from its state")
    if view.context.revision_ref != view.state.context_ref:
        raise ProjectionError("lineage view context differs from its state")
    return directive


def _bind_group_sampling(view: LineageView, turn: WriterTurnV1, reader: Any) -> None:
    spec = view.group
    lineage_id = view.state.position["lineage_id"]
    member = next((item for item in spec.members if item.member_id == lineage_id), None)
    if member is None:
        raise ProjectionError("writer lineage is absent from its sealed group spec")
    try:
        model = reader.artifact(spec.policy["model_ref"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("sealed group model is unavailable") from exc
    if not isinstance(model, Mapping) or not isinstance(model.get("model_id"), str):
        raise ProjectionError("sealed group model has no model ID")
    claims = turn.adapter_trace
    bind_group_sampling_claims(
        spec.policy,
        member.writer_seed,
        claims or {},
        claims or {},
        model_id=model["model_id"],
        context_content_hash=view.context.content_ref,
        context_revision_ref=view.context.revision_ref,
        rendering=view.context.rendering,
    )


def _parse_turn_calls(view: LineageView, turn: WriterTurnV1, reader: Any) -> list[ToolQueueEntry]:
    action_id = turn.action_id
    ordinal = action_id.removeprefix(f"{view.state.position['lineage_id']}:action:")
    if not ordinal.isdecimal():
        raise ProjectionError("sampled action ID has an invalid ordinal")
    id_prefix = f"{view.state.position['lineage_id']}:call:{ordinal}"
    if not turn.message.tool_calls_was_list:
        return [ToolQueueEntry(f"{id_prefix}:0", "invalid_call", {}, "tool_calls must be an array")]
    prior = _prior_raw_call_ids(view, action_id, reader)
    ask = None
    if view.mode.ask_semantics:
        decisions = reader.artifact(view.state.decisions_ref)

        def ask(arguments):
            validate_ask_semantics(arguments, view.node, decisions)

    try:
        return parse_calls(
            turn.message,
            id_prefix=id_prefix,
            allowed=frozenset(view.node.contract.entry_contract.tool_allowlist),
            prior_raw_ids=prior,
            ask_semantics=ask,
        )
    except WriterRuntimeError as exc:
        raise ProjectionError("sampled calls violate the canonical call envelope") from exc


def _source_call(view: LineageView, source: CallSource, reader: Any) -> ToolQueueEntry:
    # State stores only sanitized queue fields; reparse the source to recover rejection.
    sample = next(
        (
            sample
            for sample in view.samples
            if sample.action_id == source.action_id and sample.outcome == "action"
        ),
        None,
    )
    if sample is None:
        raise ProjectionError("queued call source has no accepted writer turn")
    try:
        raw_turn = reader.artifact(sample.turn_ref)
        turn = raw_turn if isinstance(raw_turn, WriterTurnV1) else WriterTurnV1.from_dict(raw_turn)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("queued call source turn cannot be decoded") from exc
    if turn.action_id != source.action_id:
        raise ProjectionError("queued call source turn has a different action ID")
    calls = _parse_turn_calls(view, turn, reader)
    try:
        call = calls[source.queue_index]
    except IndexError as exc:
        raise ProjectionError("queued call index is absent from its source turn") from exc
    if call.call_id not in view.call_sources or view.call_sources[call.call_id] != source:
        raise ProjectionError("sampled call does not match the lineage call index")
    return call


def _prior_raw_call_ids(view: LineageView, action_id: str, reader: Any) -> frozenset[str]:
    prior: set[str] = set()
    for sample in view.samples:
        if sample.action_id == action_id:
            break
        if sample.outcome != "action":
            continue
        try:
            raw = reader.artifact(sample.turn_ref)
            turn = raw if isinstance(raw, WriterTurnV1) else WriterTurnV1.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectionError("writer sample history cannot be decoded") from exc
        if turn.message.tool_calls_was_list:
            prior.update(_raw_call_ids(turn))
    return frozenset(prior)


def _raw_call_ids(turn: WriterTurnV1) -> set[str]:
    found: set[str] = set()
    for item in turn.message.calls:
        if not item["bounded"]:
            continue
        raw = decode_canonical_value(item["value"])
        if not isinstance(raw, Mapping):
            continue
        raw_id = raw.get("id")
        if (
            isinstance(raw_id, str)
            and raw_id
            and not any(character.isspace() for character in raw_id)
            and raw_id.isprintable()
        ):
            found.add(raw_id)
    return found


def _updated_raw_call_ids(prior: frozenset[str], turn: WriterTurnV1) -> frozenset[str]:
    if not turn.message.tool_calls_was_list:
        return prior
    return frozenset((*prior, *_raw_call_ids(turn)))


def _assistant_message(
    action_id: str, content: str, turn: WriterTurnV1, queue: list[ToolQueueEntry]
) -> MessageV1:
    parts: list[dict[str, Any]] = []
    if content:
        parts.append({"type": "text", "text": content})
    for index, call in enumerate(queue):
        raw = (
            turn.message.calls[index]["value"]
            if turn.message.tool_calls_was_list and index < len(turn.message.calls)
            else turn.message.calls
        )
        if call.rejection is not None:
            parts.append({"type": "invalid_tool_call", "id": call.call_id, "raw": raw})
        else:
            parts.append(
                {
                    "type": "tool_call",
                    "id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )
    return MessageV1(role="assistant", content=tuple(parts), origin=action_id, loss_eligible=True)


def _requires_usage_evidence(budget: Mapping[str, Any], usage: Mapping[str, Any]) -> bool:
    limits = budget["limits"]
    return ("generated_tokens" in limits and "completion_tokens" not in usage) or (
        "total_tokens" in limits
        and "total_tokens" not in usage
        and not {"prompt_tokens", "completion_tokens"} <= usage.keys()
    )


def _overrun_outcome(view: LineageView, stop_reason: str) -> OutcomeV1:
    outcome = replace(
        view.outcome,
        task_status="incomplete",
        execution_status="valid",
        stop_reason=stop_reason,
        candidate_checkpoint=view.checkpoint_id,
        requirement_version=view.state.requirements_ref,
    )
    return outcome


def _event(
    view: LineageView,
    payload_ref: str,
    kind: str,
    actor: str,
    audience: tuple[str, ...],
) -> EventV1:
    state = view.state
    return EventV1(
        previous=state.history["head"],
        seq=state.history["seq"] + 1,
        lineage_id=state.position["lineage_id"],
        rollout_id=state.position["lineage_id"],
        node_visit_id=state.position["visit_id"],
        kind=kind,
        actor=actor,
        audience=audience,
        payload_ref=payload_ref,
        versions_ref=state.versions_ref,
        provenance_ref=state.provenance_ref,
    )


def _state_after(
    before: EnvironmentStateV1,
    event: EventV1,
    *,
    budgets_ref: str | None = None,
    outcome_ref: str | None = None,
    context_ref: str | None = None,
    continuation: Mapping[str, Any] | None = None,
    files: Mapping[str, str] | None = None,
    phase: str | None = None,
    action_ids: tuple[str, ...] | None = None,
    tool_result_ids: tuple[str, ...] | None = None,
) -> EnvironmentStateV1:
    body = before.to_dict()
    body["history"]["head"] = event.id
    body["history"]["seq"] = event.seq
    if action_ids is not None:
        body["history"]["action_ids"] = list(action_ids)
    if tool_result_ids is not None:
        body["history"]["tool_result_ids"] = list(tool_result_ids)
    if budgets_ref is not None:
        body["budgets_ref"] = budgets_ref
    if outcome_ref is not None:
        body["outcome_ref"] = outcome_ref
    if context_ref is not None:
        body["context_ref"] = context_ref
    if continuation is not None:
        body["continuation"] = dict(continuation)
    if files is not None:
        body["files"] = dict(files)
        body["tree_hash"] = tree_hash(files)
    if phase is not None:
        body["position"]["phase"] = phase
    return EnvironmentStateV1.from_dict(body)


def _append_context(
    before: ContextView, message: MessageV1, event_id: str
) -> tuple[ContextView, ContextContentV1, ContextRevisionV1]:
    content = ContextContentV1(
        parent_ref=before.content_ref,
        messages=(message,),
        tools=None,
        rendering=None,
    )
    revision = ContextRevisionV1(
        content_ref=content.identity(), event_head=event_id, provenance_refs=(event_id,)
    )
    context = ContextView(
        messages=(*before.messages, message),
        sources=(*before.sources, event_id),
        tools=before.tools,
        rendering=before.rendering,
        content_ref=content.identity(),
        revision_ref=revision.identity(),
    )
    return context, content, revision


def _charge_context_append(
    old_budget: Mapping[str, Any],
    context: ContextView,
    content: ContextContentV1,
    revision: ContextRevisionV1,
) -> dict[str, Any] | None:
    if "context_bytes" not in old_budget["limits"]:
        return None
    budget = json.loads(canonical_json(old_budget))
    budget["consumed"]["context_bytes"] = len(
        canonical_bytes(
            {
                "messages": [message.to_dict() for message in context.messages],
                "tools": list(context.tools),
                "rendering": dict(context.rendering),
            }
        )
    )
    appended_bytes = len(canonical_bytes(content.to_wire())) + len(
        canonical_bytes(revision.to_wire())
    )
    budget["consumed"]["context_storage_bytes"] = (
        budget["consumed"].get("context_storage_bytes", 0) + appended_bytes
    )
    return budget


def _payload_artifact(ref: str, value: Any) -> DerivedArtifact:
    return DerivedArtifact(ref, canonical_bytes(value), "artifact")


def _queue_body(queue: ToolQueueEntry) -> dict[str, Any]:
    return {"call_id": queue.call_id, "name": queue.name, "arguments": queue.arguments}


def _next_view(
    before: LineageView,
    event: EventV1,
    state: EnvironmentStateV1,
    *,
    context: ContextView | None = None,
    budget: Mapping[str, Any] | None = None,
    outcome: OutcomeV1 | None = None,
    raw_call_ids: frozenset[str] | None = None,
    call_sources: Mapping[str, CallSource] | None = None,
    samples: tuple[SampleRef, ...] | None = None,
) -> LineageView:
    context = before.context if context is None else context
    checkpoint_id = CheckpointV1(
        parents=(before.checkpoint_id,), state=state, event_head=event.id
    ).identity()
    return replace(
        before,
        checkpoint_id=checkpoint_id,
        head_event_id=event.id,
        state=state,
        budget=before.budget if budget is None else budget,
        outcome=before.outcome if outcome is None else outcome,
        context=context,
        raw_call_ids=before.raw_call_ids if raw_call_ids is None else raw_call_ids,
        call_sources=before.call_sources if call_sources is None else call_sources,
        samples=before.samples if samples is None else samples,
        ancestry=replace(
            before.ancestry,
            checkpoint_id=checkpoint_id,
            context=context,
            parent=before.ancestry,
        ),
    )


DERIVES: dict[str, Callable] = {
    "WriterTurnV1": derive_writer_turn,
    "ToolObservationV1": derive_tool_result,
}


__all__ = ["DERIVES", "derive_tool_result", "derive_writer_turn"]
