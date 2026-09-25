"""Explicit writer-only projection of canonical task-graph events.

The event log is private. Only admitted seed context and allowlisted actor/kind/
audience pairs may become writer messages; operational payloads are never rendered.
"""

from __future__ import annotations

from writing_agent.task_graph import (
    ContextRevisionV1,
    MessageV1,
    canonical_json,
    context_content_hash,
)
from writing_agent.task_graph_store import TaskGraphStore


class ProjectionError(ValueError):
    """A causal event chain cannot be safely rendered as writer context."""


def project_writer_context(
    store: TaskGraphStore,
    base_checkpoint_id: str,
    target_checkpoint_id: str,
) -> ContextRevisionV1:
    """Rebuild the exact active writer messages from one admitted entry and suffix.

    The baseline is trusted only as an admitted seed/request projection and is
    always zero-mask. No file tree, private packet, check, reward, sibling event,
    or environment payload is copied into messages.
    """
    base = store.load_checkpoint(base_checkpoint_id)
    target = store.load_checkpoint(target_checkpoint_id)
    if base.state.instance_ref != target.state.instance_ref:
        raise ProjectionError("projection crosses graph instances")
    ancestor = target
    while ancestor.identity() != base_checkpoint_id:
        if not ancestor.parents:
            raise ProjectionError("target checkpoint is not descended from projection base")
        ancestor = store.load_checkpoint(ancestor.parents[0])
    baseline = store.load_context(base.state.context_ref)
    messages = list(baseline.messages)
    if (
        len(messages) < 2
        or messages[0].role != "system"
        or messages[0].trust != "instructions"
        or messages[1].role != "user"
    ):
        raise ProjectionError("entry context lacks system instructions and node request")
    if any(message.loss_eligible for message in messages):
        raise ProjectionError("seed context cannot contain writer targets")
    pending: list[str] = []
    seen_calls: set[str] = set()
    action_ids = list(base.state.history["action_ids"])
    result_ids = list(base.state.history["tool_result_ids"])
    for message in messages:
        for part in message.content:
            if part["type"] == "tool_call":
                if message.role != "assistant" or part["id"] in seen_calls:
                    raise ProjectionError("invalid seeded tool-call pairing")
                pending.append(part["id"])
                seen_calls.add(part["id"])
            elif part["type"] == "tool_result":
                if message.role != "tool" or not pending or pending.pop(0) != part["call_id"]:
                    raise ProjectionError("seeded tool result is unmatched")
    if pending:
        raise ProjectionError("seed context ends with an incomplete tool exchange")
    events = []
    cursor = target.event_head
    while cursor != base.event_head:
        if cursor is None:
            raise ProjectionError("target does not descend from projection base")
        event = store.load_event(cursor)
        events.append(event)
        cursor = event.previous
    events.reverse()
    last_source = baseline.event_head
    latest_context_ref = base.state.context_ref
    for event in events:
        if event.kind == "context_changed":
            if event.actor != "environment" or "writer" in event.audience:
                raise ProjectionError("context change has invalid actor or audience")
            effect = store.get_artifact(event.payload_ref, expected_domain="payload")
            if effect.get("artifact_type") != "Phase2RecordedEffectV1":
                raise ProjectionError("context event has no replayable effect")
            latest_context_ref = effect["set"].get("context_ref")
            if not isinstance(latest_context_ref, str):
                raise ProjectionError("context event lacks a revision")
            revision = store.load_context(latest_context_ref)
            if (
                revision.event_head != last_source
                or revision.provenance_refs != (last_source,)
                or revision.messages != tuple(messages)
            ):
                raise ProjectionError("context revision has false source-event provenance")
            continue
        if event.kind not in {"writer_action", "tool_result"}:
            continue
        expected_actor = "writer" if event.kind == "writer_action" else "environment"
        if event.actor != expected_actor or "writer" not in event.audience:
            raise ProjectionError("writer-visible event has invalid actor or audience")
        effect = store.get_artifact(event.payload_ref, expected_domain="payload")
        if effect.get("artifact_type") != "Phase2RecordedEffectV1":
            raise ProjectionError("writer event has no replayable effect")
        log_ref = effect["set"].get("external_inputs_ref")
        if not isinstance(log_ref, str):
            raise ProjectionError("writer event lacks its canonical metadata index")
        log = store.get_artifact(log_ref, expected_domain="payload")
        if log.get("record_type") != "WriterRuntimeLogV1" or not log.get("entries"):
            raise ProjectionError("writer event metadata index is invalid")
        entry = log["entries"][-1]
        if (
            entry["seq"] != event.seq
            or entry["kind"] != event.kind
            or log["rollout_id"] != event.rollout_id
        ):
            raise ProjectionError("writer event metadata is not causally bound")
        record = store.get_artifact(entry["record_ref"], expected_domain="payload")
        message = MessageV1.from_dict(
            store.get_artifact(entry["message_ref"], expected_domain="message")
        )
        if event.kind == "writer_action":
            if pending or record.get("record_type") != "WriterActionV1":
                raise ProjectionError("writer action starts before prior calls finish")
            if (
                message.role != "assistant"
                or not message.loss_eligible
                or message.origin != record["action_id"]
            ):
                raise ProjectionError("writer action message identity is invalid")
            trace = store.get_artifact(record["trace_ref"], expected_domain="payload")
            expected_context = context_content_hash(
                tuple(messages), tools=baseline.tools, rendering=baseline.rendering
            )
            if trace.get("context_content_hash") != expected_context:
                raise ProjectionError("writer trace names a different sampling context")
            if trace.get("context_revision_ref") != latest_context_ref:
                raise ProjectionError("writer trace names a different context revision")
            if canonical_json(trace.get("rendering")) != canonical_json(baseline.rendering):
                raise ProjectionError("writer trace rendering pins differ from context")
            calls = [part for part in message.content if part["type"] == "tool_call"]
            if [part["id"] for part in calls] != [call["call_id"] for call in record["calls"]]:
                raise ProjectionError("action syntax and call metadata differ")
            queue = effect["set"].get("continuation", {}).get("tool_queue", ())
            if [
                {"call_id": part["id"], "name": part["name"], "arguments": part["arguments"]}
                for part in calls
            ] != queue:
                raise ProjectionError("committed tool queue differs from assistant syntax")
            if record["action_id"] in action_ids:
                raise ProjectionError("duplicate writer action logical ID")
            action_ids.append(record["action_id"])
            for part in calls:
                if part["id"] in seen_calls:
                    raise ProjectionError("duplicate logical call ID")
                seen_calls.add(part["id"])
                pending.append(part["id"])
        else:
            if record.get("record_type") != "WriterToolResultV1":
                raise ProjectionError("tool event metadata has wrong type")
            if (
                message.role != "tool"
                or message.loss_eligible
                or message.call_id != record["call_id"]
            ):
                raise ProjectionError("tool observation has invalid role or loss eligibility")
            if not pending or pending.pop(0) != message.call_id:
                raise ProjectionError("tool observation is unpaired or out of order")
            if message.origin != record["action_id"]:
                raise ProjectionError("tool observation has the wrong action origin")
            if record["result_id"] in result_ids:
                raise ProjectionError("duplicate tool result logical ID")
            result_ids.append(record["result_id"])
            if canonical_json(record["file_delta"]) != canonical_json(effect["file_delta"]):
                raise ProjectionError("tool result file delta differs from recorded effect")
            expected_part = {
                "type": "tool_result",
                "call_id": message.call_id,
                "content": record["observation"],
            }
            if len(message.content) != 1 or canonical_json(message.content[0]) != canonical_json(
                expected_part
            ):
                raise ProjectionError("tool observation differs from recorded result")
        messages.append(message)
        last_source = event.id
    actual = store.load_context(target.state.context_ref)
    if target.state.context_ref != latest_context_ref:
        raise ProjectionError("context revision was changed outside a context event")
    if (
        tuple(messages) != actual.messages
        or actual.tools != baseline.tools
        or actual.rendering != baseline.rendering
    ):
        raise ProjectionError("persisted context differs from authorized event projection")
    if actual.event_head != last_source:
        raise ProjectionError("context revision lacks latest source-event provenance")
    if pending != [
        call["call_id"]
        for call in target.state.continuation["tool_queue"][
            target.state.continuation["next_call"] :
        ]
    ]:
        raise ProjectionError("pending calls differ from continuation cursor")
    if action_ids != list(target.state.history["action_ids"]) or result_ids != list(
        target.state.history["tool_result_ids"]
    ):
        raise ProjectionError("logical action/result history differs from visible events")
    return actual
