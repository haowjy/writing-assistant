"""Pure writer-turn and tool-result transitions for the V1 task-graph seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from writing_agent.task_graph import MessageV1, canonical_bytes, domain_hash
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
from writing_agent.task_graph_derive_common import (
    DeriveKey,
    advance,
    append_context,
    new_event,
    next_state,
    payload_artifact,
)
from writing_agent.task_graph_errors import (
    AdapterContractError,
    AdapterContractProjectionError,
    ProjectionError,
    WriterRuntimeError,
)
from writing_agent.task_graph_records import (
    ToolObservationV1,
    WriterTurnV1,
)
from writing_agent.task_graph_sampling import (
    WriterTurnSamplingBindingV1,
    bind_group_sampling_claims,
    decode_and_bind_sampling,
)
from writing_agent.task_graph_scripted import validate_ask_semantics
from writing_agent.task_graph_transition import (
    CallSource,
    LineageView,
    SampleRef,
    Transition,
)
from writing_agent.task_graph_wire import decode_canonical_value

READ_BUDGET_EXCEEDED = "Read-token budget exceeded"


def writer_action_id(view: LineageView) -> str:
    """Return the action identity bound to the view's next writer turn."""
    return f"{view.state.position['lineage_id']}:action:{len(view.state.history['action_ids'])}"


def group_member(view: LineageView):
    """Resolve this view's sealed group member, if it belongs to a group."""
    if view.group is None:
        return None
    lineage_id = view.state.position["lineage_id"]
    member = next((item for item in view.group.members if item.member_id == lineage_id), None)
    if member is None:
        raise ProjectionError("writer lineage is absent from its sealed group spec")
    return member


def sampling_usage_requirements(budget: Mapping[str, Any]) -> frozenset[str]:
    """Return usage fields required by the active token-budget limits."""
    limits = budget["limits"]
    return frozenset(
        field
        for limit, field in (
            ("generated_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        )
        if limit in limits
    )


def tool_dispatch_error(budget: Mapping[str, Any], queue_entry: Mapping[str, Any]) -> str | None:
    """Return the persisted tool call's budget or syntax rejection, if any."""
    return tool_error(dict(budget), queue_entry.get("rejection"), queue_entry["name"])


def derive_writer_turn(view: LineageView, turn: WriterTurnV1, reader: Any) -> Transition:
    """Derive the one event, state and context caused by a sampled writer turn."""
    action_ordinal, action_id, content = _validate_writer_turn(view, turn)
    _bind_writer_turn(view, turn, reader)

    usage = turn.usage
    next_budget, exceeded = sampled_usage_charge(view.budget, usage)
    input_artifact = payload_artifact(turn)
    turn_ref = input_artifact.ref
    if exceeded is not None:
        stop_reason = f"{exceeded}_budget"
        outcome = replace(
            view.outcome,
            task_status="incomplete",
            execution_status="valid",
            stop_reason=stop_reason,
            candidate_checkpoint=view.checkpoint_id,
            requirement_version=view.state.requirements_ref,
        )
        outcome_ref = outcome.identity()
        event = new_event(
            view,
            turn_ref,
            kind="budget_charged",
            actor="writer_runtime",
            audience=("controller", "evaluator", "trainer"),
        )
        state = next_state(
            view,
            event,
            budgets_ref=domain_hash("payload", next_budget),
            outcome_ref=outcome_ref,
            position={"phase": "terminal"},
        )
        artifacts = (
            input_artifact,
            payload_artifact(next_budget),
            payload_artifact(outcome),
        )
        result_view = advance(
            view,
            turn,
            event,
            state,
            artifacts=artifacts,
            budget=next_budget,
            outcome=outcome,
            samples=(*view.samples, SampleRef(action_id, event.id, turn_ref, "budget_stop")),
        )
        return result_view

    id_prefix = f"{view.state.position['lineage_id']}:call:{action_ordinal}"
    queue = _build_tool_queue(
        view,
        turn,
        reader,
        id_prefix=id_prefix,
        prior_raw_ids=view.raw_call_ids,
    )

    required_usage = sampling_usage_requirements(next_budget)
    if ("completion_tokens" in required_usage and "completion_tokens" not in usage) or (
        "total_tokens" in required_usage
        and "total_tokens" not in usage
        and not {"prompt_tokens", "completion_tokens"} <= usage.keys()
    ):
        raise AdapterContractProjectionError("input.usage.completion_tokens: required")

    assistant = _build_assistant_message(turn, action_id, content, queue)
    event = new_event(
        view,
        turn_ref,
        kind="writer_action",
        actor="writer",
        audience=("controller", "trainer", "writer"),
    )
    context, content_node, revision, budget = append_context(
        replace(view, budget=next_budget), (assistant,), event.id
    )
    budget_ref = (
        domain_hash("payload", budget)
        if budget is not None
        else domain_hash("payload", next_budget)
    )
    if budget is None:
        budget = next_budget

    continuation = view.state.to_dict()["continuation"]
    continuation["tool_queue"] = [item.to_dict() for item in queue]
    continuation["next_call"] = 0
    phase = "ready_writer" if queue else "checking"
    state = next_state(
        view,
        event,
        budgets_ref=budget_ref,
        context_ref=revision.ref,
        continuation=continuation,
        position={"phase": phase},
        history={"action_ids": (*view.state.history["action_ids"], action_id)},
    )
    sources = dict(view.call_sources)
    for index, call in enumerate(queue):
        if call.call_id in sources:
            raise ProjectionError("writer turn reuses a logical tool-call ID")
        sources[call.call_id] = CallSource(action_id, index)
    raw_call_ids = view.raw_call_ids
    if turn.message.tool_calls_was_list:
        raw_call_ids = frozenset((*raw_call_ids, *_raw_call_ids(turn)))
    artifacts = (
        input_artifact,
        payload_artifact(budget),
        content_node,
        revision,
    )
    result_view = advance(
        view,
        turn,
        event,
        state,
        artifacts=artifacts,
        context=context,
        budget=budget,
        raw_call_ids=raw_call_ids,
        call_sources=sources,
        samples=(*view.samples, SampleRef(action_id, event.id, turn_ref, "action")),
    )
    return result_view


def _validate_writer_turn(view: LineageView, turn: WriterTurnV1) -> tuple[int, str, str]:
    directive = _directive(view)
    if directive.kind != "sample_writer":
        raise ProjectionError("writer turn is not the next legal step")
    if not isinstance(turn, WriterTurnV1):
        raise ProjectionError("writer turn input must use its strict wire codec")

    ordinal = len(view.state.history["action_ids"])
    action_id = writer_action_id(view)
    if turn.action_id != action_id or turn.context_revision_ref != view.context.revision_ref:
        raise ProjectionError("writer turn is not bound to the active action and context")
    content = turn.message.content
    if content is not None and not isinstance(content, str):
        raise ProjectionError("sampled assistant content must be text or null")
    return ordinal, action_id, "" if content is None else content


def _bind_writer_turn(view: LineageView, turn: WriterTurnV1, reader: Any) -> None:
    try:
        decode_and_bind_sampling(WriterTurnSamplingBindingV1(turn, view.context, reader))
    except ProjectionError as exc:
        raise AdapterContractProjectionError(str(exc)) from exc
    except (AdapterContractError, KeyError, TypeError, ValueError) as exc:
        raise AdapterContractProjectionError("input.adapter_trace: invalid") from exc

    member = group_member(view)
    if member is None:
        return
    spec = view.group
    try:
        model = reader.artifact(spec.policy["model_ref"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("sealed group model is unavailable") from exc
    if not isinstance(model, Mapping) or not isinstance(model.get("model_id"), str):
        raise ProjectionError("sealed group model has no model ID")
    claims = turn.adapter_trace
    try:
        bind_group_sampling_claims(
            spec.policy,
            member.writer_seed,
            claims or {},
            model_id=model["model_id"],
        )
    except (ProjectionError, KeyError, TypeError, ValueError) as exc:
        raise AdapterContractProjectionError("input.adapter_trace: group pin mismatch") from exc


def _build_tool_queue(
    view: LineageView,
    turn: WriterTurnV1,
    reader: Any,
    *,
    id_prefix: str,
    prior_raw_ids: frozenset[str],
) -> list[ToolQueueEntry]:
    if not turn.message.tool_calls_was_list:
        return [ToolQueueEntry(f"{id_prefix}:0", "invalid_call", {}, "tool_calls must be an array")]

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
            prior_raw_ids=prior_raw_ids,
            ask_semantics=ask,
        )
    except WriterRuntimeError as exc:
        raise AdapterContractProjectionError(
            "sampled calls violate the canonical call envelope"
        ) from exc


def _build_assistant_message(
    turn: WriterTurnV1,
    action_id: str,
    content: str,
    queue: list[ToolQueueEntry],
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
            part = {"type": "invalid_tool_call", "id": call.call_id}

            def has_noncanonical_tag(value):
                if isinstance(value, Mapping):
                    return "$noncanonical" in value or any(
                        has_noncanonical_tag(item) for item in value.values()
                    )
                if isinstance(value, (tuple, list)):
                    return any(has_noncanonical_tag(item) for item in value)
                return False

            if turn.message.tool_calls_was_list:
                carries_sampled_content = (
                    index < len(turn.message.calls)
                    and turn.message.calls[index]["bounded"]
                    and not has_noncanonical_tag(raw)
                )
            else:
                carries_sampled_content = not has_noncanonical_tag(raw)
            part["raw"] = (
                raw if carries_sampled_content else {"$noncanonical": "no-sampled-content"}
            )
            parts.append(part)
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
    call_name = queued["name"]
    call_arguments = queued["arguments"]
    predispatch_error = tool_dispatch_error(view.budget, queued)
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
                call_name,
                call_arguments,
                view.state.files,
                files_after,
                observation["ok"],
                max_file_bytes=view.tool_spec.max_file_bytes,
                max_workspace_bytes=view.tool_spec.max_workspace_bytes,
                storage_bytes_limit=view.budget["limits"]["storage_bytes"],
            )
            read_charge = observation_read_tokens(
                observation, call_name, view.budget["read_tokenizer"]
            )
        except (AdapterContractError, TypeError, ValueError) as exc:
            raise AdapterContractProjectionError("input.dispatch.effect: invalid") from exc
        remaining_reads = view.budget["limits"].get("read_tokens", 0) - view.budget["consumed"].get(
            "read_tokens", 0
        )
        if read_charge > remaining_reads:
            observation = {"ok": False, "valid": True, "error": READ_BUDGET_EXCEEDED}
            files_after = dict(view.state.files)
            read_charge = 0

    next_budget, _charge = tool_result_charge(
        view.budget, view.state.files, files_after, read_charge
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
    observation_artifact = payload_artifact(obs)
    obs_ref = observation_artifact.ref
    event = new_event(
        view,
        obs_ref,
        kind="tool_result",
        actor="environment",
        audience=("controller", "trainer", "writer"),
    )
    context, content_node, revision, budget = append_context(
        replace(view, budget=next_budget), (message,), event.id
    )
    if budget is None:
        budget = next_budget
    budget_ref = domain_hash("payload", budget)
    next_continuation = view.state.to_dict()["continuation"]
    next_continuation["next_call"] = cursor + 1
    state = next_state(
        view,
        event,
        files=files_after,
        budgets_ref=budget_ref,
        context_ref=revision.ref,
        continuation=next_continuation,
        history={"tool_result_ids": (*view.state.history["tool_result_ids"], result_id)},
    )
    artifacts = (
        observation_artifact,
        payload_artifact(budget),
        content_node,
        revision,
    )
    return advance(view, obs, event, state, artifacts=artifacts, context=context, budget=budget)


def _directive(view: LineageView):
    try:
        return next_step(view)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("lineage view has no valid next directive") from exc


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


DERIVES: dict[DeriveKey, Callable] = {
    "WriterTurnV1": derive_writer_turn,
    "ToolObservationV1": derive_tool_result,
}


__all__ = [
    "DERIVES",
    "derive_tool_result",
    "derive_writer_turn",
    "group_member",
    "tool_dispatch_error",
    "writer_action_id",
]
