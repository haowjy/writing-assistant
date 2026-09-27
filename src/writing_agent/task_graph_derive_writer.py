"""Pure writer-turn and tool-result transitions for the V1 task-graph seam."""

from __future__ import annotations

from collections.abc import Callable
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
    apply_effect,
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
    ProjectionError,
)
from writing_agent.task_graph_records import (
    ToolObservationV1,
    WriterTurnV1,
)
from writing_agent.task_graph_sampling import (
    WriterTurnSamplingBindingV1,
    assistant_message,
    bind_group_writer_sampling,
    decode_and_bind_sampling,
    parse_writer_turn_calls,
    requires_usage_evidence,
    updated_raw_call_ids,
    usage_overrun_outcome,
)
from writing_agent.task_graph_scripted import validate_ask_semantics
from writing_agent.task_graph_transition import (
    CallSource,
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
        bind_group_writer_sampling(view, turn, reader)

    content = turn.message.content
    if content is not None and not isinstance(content, str):
        raise ProjectionError("sampled assistant content must be text or null")
    content = "" if content is None else content

    usage = turn.usage
    next_budget, exceeded = sampled_usage_charge(dict(view.budget), usage)
    input_artifact = payload_artifact(turn)
    turn_ref = input_artifact.ref
    if exceeded is not None:
        stop_reason = f"{exceeded}_budget"
        outcome = usage_overrun_outcome(
            view.outcome,
            view.checkpoint_id,
            view.state.requirements_ref,
            stop_reason,
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

    ask_semantics = validate_ask_semantics if view.mode.ask_semantics else None
    queue = parse_writer_turn_calls(view, turn, reader, ask_semantics)

    if requires_usage_evidence(next_budget, usage):
        raise ProjectionError("token-limited writer turn lacks usage evidence")

    assistant = assistant_message(action_id, content, turn, queue)
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
    raw_call_ids = updated_raw_call_ids(view.raw_call_ids, turn)
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
    predispatch_error = tool_error(dict(view.budget), queued.get("rejection"), call_name)
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


DERIVES: dict[DeriveKey, Callable] = {
    "WriterTurnV1": derive_writer_turn,
    "ToolObservationV1": derive_tool_result,
}


__all__ = ["DERIVES", "derive_tool_result", "derive_writer_turn"]
