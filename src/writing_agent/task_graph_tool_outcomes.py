"""Pure derivation of tool outcomes from verified task-graph lineage views.

This module reads only committed context and checkpoint-derived state. It does not invoke a
tool, sampling backend, tokenizer, or model stack.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from writing_agent.task_graph_calls import (
    PROTOCOL_SHAPED_REJECTION_CODES,
    TOOL_EXECUTION_FAILURE_CODE,
    rejection_code_for_message,
)
from writing_agent.task_graph_sampling import TERMINATION_STOP_REASONS


class ToolOutcomeError(ValueError):
    """A committed call/result chain is incomplete or cannot be classified safely."""

    protocol_shape = "tool_result"


def _message_value(message: Any) -> Mapping[str, Any]:
    value = message.to_dict() if callable(getattr(message, "to_dict", None)) else message
    if not isinstance(value, Mapping):
        raise ToolOutcomeError("committed context contains a malformed message")
    return value


def _call_name(part: Mapping[str, Any]) -> str | None:
    name = part.get("name")
    if isinstance(name, str):
        return name
    raw = part.get("raw")
    if isinstance(raw, Mapping):
        function = raw.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        return name if isinstance(name, str) else None
    return None


def _tool_calls(messages: tuple[Any, ...]) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        value = _message_value(message)
        if value.get("role") != "assistant":
            continue
        for part in value.get("content", ()):
            if not isinstance(part, Mapping) or part.get("type") not in {
                "tool_call",
                "invalid_tool_call",
            }:
                continue
            call_id = part.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise ToolOutcomeError("committed assistant call has no call ID")
            if call_id in calls:
                raise ToolOutcomeError("committed context repeats a tool call ID")
            calls[call_id] = {
                "call_id": call_id,
                "name": _call_name(part),
                "intake_rejected": part["type"] == "invalid_tool_call",
            }
    return calls


def _tool_results(messages: tuple[Any, ...]) -> dict[str, Mapping[str, Any]]:
    results: dict[str, Mapping[str, Any]] = {}
    for message in messages:
        value = _message_value(message)
        if value.get("role") != "tool":
            continue
        for part in value.get("content", ()):
            if not isinstance(part, Mapping) or part.get("type") != "tool_result":
                continue
            call_id = part.get("call_id")
            response = part.get("content")
            if not isinstance(call_id, str) or not isinstance(response, Mapping):
                raise ToolOutcomeError("committed tool result has an invalid shape")
            if call_id in results:
                raise ToolOutcomeError("committed context repeats a tool result")
            results[call_id] = response
    return results


def _result_for_call(call: Mapping[str, Any], response: Mapping[str, Any]) -> str | dict[str, str]:
    ok = response.get("ok")
    if ok is True:
        return "ok"
    if ok is not False or not isinstance(response.get("error"), str):
        raise ToolOutcomeError("committed tool result has no classifiable outcome")

    message = response["error"]
    code = rejection_code_for_message(message)
    if call["intake_rejected"]:
        # Every core parser rejection must stay in the reviewed table. Unknown exact text is
        # an unclassifiable criterion input, not a guessed substring-based category.
        if code is None:
            raise ToolOutcomeError("committed intake rejection has no stable rejection code")
    elif code is None:
        # After a valid call was admitted, workspace-level failures (missing paths, unmatched
        # patch text, and similar requested-effect errors) are candidate behavior. Host faults
        # propagate instead of becoming committed observations.
        code = TOOL_EXECUTION_FAILURE_CODE
    return {"code": code, "text": message}


def read_member_tool_outcomes(start: Any, final: Any) -> dict[str, Any]:
    """Derive every new call outcome and the final file delta for one member lineage.

    ``start`` and ``final`` must be verified views at the member's committed start and final
    checkpoints. Call/result pairing and call-source closure are checked exactly; incomplete
    context is an error rather than an empty outcome list.
    """
    start_lineage = start.state.position["lineage_id"]
    final_lineage = final.state.position["lineage_id"]
    if start_lineage != final_lineage:
        raise ValueError("tool outcome checkpoints name different lineages")
    try:
        ancestor = final.ancestry.context_at(start.checkpoint_id)
    except KeyError as exc:
        raise ValueError(
            "tool outcome start checkpoint is not a final-checkpoint ancestor"
        ) from exc
    if ancestor.revision_ref != start.context.revision_ref:
        raise ValueError("tool outcome start context differs from its committed checkpoint")

    all_calls = _tool_calls(final.context.messages)
    initial_call_ids = set(start.call_sources)
    final_call_ids = set(final.call_sources)
    new_call_ids = final_call_ids - initial_call_ids
    if set(all_calls) != final_call_ids:
        raise ToolOutcomeError("committed context does not expose every writer call source")
    calls = {call_id: all_calls[call_id] for call_id in new_call_ids}
    all_results = _tool_results(final.context.messages)
    results = {call_id: value for call_id, value in all_results.items() if call_id in new_call_ids}
    unpaired_results = set(results) - set(calls)
    unexecuted_call_ids = set(calls) - set(results)
    if unpaired_results:
        raise ToolOutcomeError("committed writer calls and tool results do not pair exactly")
    if unexecuted_call_ids:
        final_action_id = (
            final.samples[-1].action_id
            if final.samples and final.samples[-1].outcome == "action"
            else None
        )
        outcome = final.outcome
        if (
            outcome.task_status != "incomplete"
            or outcome.stop_reason not in TERMINATION_STOP_REASONS
            or final_action_id is None
            or any(
                final.call_sources[call_id].action_id != final_action_id
                for call_id in unexecuted_call_ids
            )
        ):
            raise ToolOutcomeError("committed writer calls and tool results do not pair exactly")

    outcomes = []
    counts: Counter[str] = Counter()
    protocol_rejection_count = 0

    def call_order(call_id: str) -> tuple[int, int]:
        source = final.call_sources[call_id]
        action_ordinal = int(source.action_id.rsplit(":action:", 1)[1])
        return action_ordinal, source.queue_index

    for call_id in sorted(calls, key=call_order):
        call = calls[call_id]
        result = (
            {"code": "not_executed_incomplete"}
            if call_id in unexecuted_call_ids
            else _result_for_call(call, results[call_id])
        )
        outcome = {"call_id": call_id, "name": call["name"], "result": result}
        outcomes.append(outcome)
        code = "ok" if result == "ok" else result["code"]
        counts[code] += 1
        if code in PROTOCOL_SHAPED_REJECTION_CODES:
            protocol_rejection_count += 1

    initial_files = start.state.files
    final_files = final.state.files
    changed_paths = sorted(
        path
        for path in initial_files.keys() | final_files.keys()
        if initial_files.get(path) != final_files.get(path)
    )
    return {
        "calls": outcomes,
        "counts_by_code": dict(sorted(counts.items())),
        "protocol_shaped_rejection_count": protocol_rejection_count,
        "files_changed": bool(changed_paths),
        "changed_paths": changed_paths,
    }


__all__ = ["ToolOutcomeError", "read_member_tool_outcomes"]
