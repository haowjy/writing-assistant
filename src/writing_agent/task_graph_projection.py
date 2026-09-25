"""Explicit writer-only projection of canonical task-graph events.

The event log is private. Only admitted seed context and allowlisted actor/kind/
audience pairs may become writer messages; operational payloads are never rendered.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from writing_agent.task_graph import (
    ContextRevisionV1,
    MessageV1,
    canonical_json,
    context_content_hash,
    domain_hash,
)
from writing_agent.task_graph_store import TaskGraphStore


class ProjectionError(ValueError):
    """A causal event chain cannot be safely rendered as writer context."""


def _value_only(value):
    if isinstance(value, Mapping):
        return {
            key: _value_only(item)
            for key, item in value.items()
            if not (key.endswith("_ref") or key.endswith("_refs") or "provenance" in key)
        }
    if isinstance(value, (tuple, list)):
        return [_value_only(item) for item in value]
    if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return "<identity>"
    return value


def execution_value(store, state, context, budget) -> str:
    return domain_hash(
        "state",
        {
            "files": state.files,
            "position": {
                key: state.position[key]
                for key in ("node_id", "visit_id", "phase", "loop_counts", "lineage_id")
            },
            "messages": [message.to_dict() for message in context.messages],
            "tool_queue": state.continuation["tool_queue"],
            "next_call": state.continuation["next_call"],
            "action_ids": state.history["action_ids"],
            "tool_result_ids": state.history["tool_result_ids"],
            "budgets": budget,
            "requirements": _value_only(store.get_artifact(state.requirements_ref)),
            "decisions": _value_only(store.get_artifact(state.decisions_ref)),
            "disclosures": _value_only(store.get_artifact(state.disclosures_ref)),
            "outcome": _value_only(store.get_artifact(state.outcome_ref)),
            "rng": _value_only(store.get_artifact(state.rng_ref)),
        },
    )


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
    state = base.state
    call_sources: dict[str, tuple[str, dict]] = {}
    seen_entries: list[dict] = []
    for event in events:
        effect = store.get_artifact(event.payload_ref, expected_domain="payload")
        before = state
        state = store._apply_recorded_effect_body(before, event, effect)
        if event.kind == "context_changed":
            if event.actor != "environment" or "writer" in event.audience:
                raise ProjectionError("context change has invalid actor or audience")
            if effect.get("artifact_type") != "Phase2RecordedEffectV1":
                raise ProjectionError("context event has no replayable effect")
            if (
                set(effect["set"]) != {"context_ref"}
                or effect["history_set"]
                or effect["file_delta"]
                or state.files != before.files
                or state.budgets_ref != before.budgets_ref
                or state.continuation != before.continuation
            ):
                raise ProjectionError("context event has an unrelated execution effect")
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
        if event.kind == "budget_charged":
            log = store.get_artifact(state.external_inputs_ref, expected_domain="payload")
            if not isinstance(log, dict) or log.get("record_type") != "WriterRuntimeLogV1":
                if seen_entries:
                    raise ProjectionError("writer budget event lost its runtime log")
                continue
            entry = log["entries"][-1]
            record = store.get_artifact(entry["record_ref"], expected_domain="payload")
            trace = store.get_artifact(record["trace_ref"], expected_domain="payload")
            prepared_ref = record["prepared_request_ref"]
            if prepared_ref is not None:
                prepared = store.get_artifact(prepared_ref, expected_domain="payload")
                if (
                    prepared.get("record_type") != "PreparedWriterRequestV1"
                    or prepared.get("context_content_hash") != trace.get("context_content_hash")
                    or prepared.get("context_revision_ref") != latest_context_ref
                    or prepared.get("payload_ref") != record["request_ref"]
                    or canonical_json(prepared.get("rendering"))
                    != canonical_json(baseline.rendering)
                ):
                    raise ProjectionError("sampled stop prepared request is false")
            usage = record["usage"]
            old_budget = store.get_artifact(before.budgets_ref, expected_domain="payload")
            new_budget = store.get_artifact(state.budgets_ref, expected_domain="payload")
            expected = json.loads(canonical_json(old_budget))
            consumed = expected["consumed"]
            consumed["writer_turns"] = consumed.get("writer_turns", 0) + 1
            consumed["model_calls"] = consumed.get("model_calls", 0) + 1
            consumed["generated_tokens"] = consumed.get("generated_tokens", 0) + usage.get(
                "completion_tokens", 0
            )
            consumed["total_tokens"] = consumed.get("total_tokens", 0) + usage.get(
                "total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            )
            exceeded = [
                name
                for name in ("generated_tokens", "total_tokens")
                if name in expected["limits"] and consumed[name] > expected["limits"][name]
            ]
            outcome = store.get_artifact(state.outcome_ref, expected_domain="payload")
            if (
                event.actor != "environment"
                or "writer" in event.audience
                or entry["seq"] != event.seq
                or entry["kind"] != event.kind
                or log["rollout_id"] != event.rollout_id
                or record["record_type"] != "WriterSampledBudgetStopV1"
                or log["entries"] != [*seen_entries, entry]
                or trace.get("action_id") != f"{event.rollout_id}:action:{len(action_ids)}"
                or trace.get("context_content_hash")
                != context_content_hash(
                    tuple(messages), tools=baseline.tools, rendering=baseline.rendering
                )
                or trace.get("context_revision_ref") != latest_context_ref
                or trace.get("exact_request_ref") != record["request_ref"]
                or trace.get("raw_output_ref") != record["raw_output_ref"]
                or not exceeded
                or record["reason"] != f"{exceeded[0]}_budget"
                or outcome.get("stop_reason") != record["reason"]
                or new_budget != expected
                or state.files != before.files
                or state.context_ref != before.context_ref
                or state.continuation != before.continuation
                or state.position["phase"] != "terminal"
                or effect["file_delta"]
            ):
                raise ProjectionError("sampled budget stop has false accounting or authority")
            seen_entries.append(entry)
            continue
        if event.kind == "termination_recorded":
            log = store.get_artifact(state.external_inputs_ref, expected_domain="payload")
            if not isinstance(log, dict) or log.get("record_type") != "WriterRuntimeLogV1":
                if seen_entries:
                    raise ProjectionError("writer termination lost its runtime log")
                continue
            entry = log["entries"][-1]
            record = store.get_artifact(entry["record_ref"], expected_domain="payload")
            old_budget = store.get_artifact(before.budgets_ref, expected_domain="payload")
            outcome = store.get_artifact(state.outcome_ref, expected_domain="payload")
            exhausted = [
                name
                for name in ("writer_turns", "generated_tokens", "total_tokens")
                if name in old_budget["limits"]
                and old_budget["consumed"].get(name, 0) >= old_budget["limits"][name]
            ]
            reason = (
                "writer_budget"
                if exhausted and exhausted[0] == "writer_turns"
                else (f"{exhausted[0]}_budget" if exhausted else None)
            )
            if (
                not exhausted
                or outcome.get("stop_reason") != reason
                or event.actor != "environment"
                or "writer" in event.audience
                or log["rollout_id"] != event.rollout_id
                or log["entries"] != [*seen_entries, entry]
                or entry["seq"] != event.seq
                or entry["kind"] != event.kind
                or record != {"record_type": "WriterExhaustedStopV1", "reason": reason}
                or state.position["phase"] != "terminal"
                or state.files != before.files
                or state.context_ref != before.context_ref
                or state.continuation != before.continuation
                or state.budgets_ref != before.budgets_ref
                or effect["file_delta"]
            ):
                raise ProjectionError("termination is not an exhausted writer stop")
            seen_entries.append(entry)
            continue
        if event.kind not in {"writer_action", "tool_result"}:
            if state.context_ref != before.context_ref:
                raise ProjectionError("context revision was changed outside a context event")
            if state.files != before.files:
                raise ProjectionError("operational event altered writer files")
            if seen_entries and (
                state.budgets_ref != before.budgets_ref
                or state.continuation != before.continuation
                or state.history["action_ids"] != before.history["action_ids"]
                or state.history["tool_result_ids"] != before.history["tool_result_ids"]
            ):
                raise ProjectionError("operational event altered writer accounting")
            continue
        expected_actor = "writer" if event.kind == "writer_action" else "environment"
        if event.actor != expected_actor or "writer" not in event.audience:
            raise ProjectionError("writer-visible event has invalid actor or audience")
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
            or log["entries"] != [*seen_entries, entry]
        ):
            raise ProjectionError("writer event metadata is not causally bound")
        seen_entries.append(entry)
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
            prepared_ref = trace.get("prepared_request_ref")
            if prepared_ref is not None:
                prepared = store.get_artifact(prepared_ref, expected_domain="payload")
                if (
                    not isinstance(prepared, dict)
                    or prepared.get("record_type") != "PreparedWriterRequestV1"
                    or prepared.get("context_content_hash") != expected_context
                    or prepared.get("context_revision_ref") != latest_context_ref
                    or prepared.get("payload_ref") != trace.get("exact_request_ref")
                    or canonical_json(prepared.get("rendering"))
                    != canonical_json(baseline.rendering)
                ):
                    raise ProjectionError("prepared request differs from writer trace")
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
            if record["action_id"] != f"{event.rollout_id}:action:{len(action_ids)}":
                raise ProjectionError("writer action ordinal is false")
            old_budget = store.get_artifact(before.budgets_ref, expected_domain="payload")
            new_budget = store.get_artifact(state.budgets_ref, expected_domain="payload")
            usage = record["usage"]
            expected_budget = json.loads(canonical_json(old_budget))
            charged = expected_budget["consumed"]
            charged["writer_turns"] = charged.get("writer_turns", 0) + 1
            charged["model_calls"] = charged.get("model_calls", 0) + 1
            charged["generated_tokens"] = charged.get("generated_tokens", 0) + usage.get(
                "completion_tokens", 0
            )
            charged["total_tokens"] = charged.get("total_tokens", 0) + usage.get(
                "total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            )
            if (
                new_budget != expected_budget
                or state.files != before.files
                or effect["file_delta"]
                or state.continuation["next_call"] != 0
                or list(state.history["action_ids"]) != [*action_ids, record["action_id"]]
            ):
                raise ProjectionError("writer action state or budget charge is false")
            action_ids.append(record["action_id"])
            for index, part in enumerate(calls):
                if part["id"] in seen_calls:
                    raise ProjectionError("duplicate logical call ID")
                seen_calls.add(part["id"])
                pending.append(part["id"])
                call_sources[part["id"]] = (record["action_id"], queue[index])
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
            source = call_sources.get(message.call_id)
            if source is None or record["action_id"] != source[0] or message.origin != source[0]:
                raise ProjectionError("tool observation has the wrong action origin")
            cursor = before.continuation["next_call"]
            if (
                before.continuation["tool_queue"][cursor] != source[1]
                or state.continuation["next_call"] != cursor + 1
                or state.continuation["tool_queue"] != before.continuation["tool_queue"]
                or record["result_id"] != f"{event.rollout_id}:tool_result:{len(result_ids)}"
                or list(state.history["tool_result_ids"]) != [*result_ids, record["result_id"]]
            ):
                raise ProjectionError("tool result queue, cursor or ordinal is false")
            if record["result_id"] in result_ids:
                raise ProjectionError("duplicate tool result logical ID")
            result_ids.append(record["result_id"])
            if canonical_json(record["file_delta"]) != canonical_json(effect["file_delta"]):
                raise ProjectionError("tool result file delta differs from recorded effect")
            delta = {
                path: {"before": before.files.get(path), "after": state.files.get(path)}
                for path in sorted(set(before.files) | set(state.files))
                if before.files.get(path) != state.files.get(path)
            }
            if record["file_delta"] != delta:
                raise ProjectionError("tool result file delta is false")
            old_budget = store.get_artifact(before.budgets_ref, expected_domain="payload")
            new_budget = store.get_artifact(state.budgets_ref, expected_domain="payload")
            old = old_budget["consumed"]
            call_charge = int(old.get("tool_calls", 0) < old_budget["limits"]["tool_calls"])
            read_charge = 0
            if record["observation"].get("ok") and source[1]["name"] in {
                "read_file",
                "search",
                "list_dir",
            }:
                if old_budget["read_tokenizer"] != "whitespace-v1":
                    raise ProjectionError("unsupported read tokenizer for semantic validation")
                read_charge = len(
                    json.dumps(record["observation"]["result"], ensure_ascii=False).split()
                )
            before_bytes = sum(len(text.encode("utf-8")) for text in before.files.values())
            after_bytes = sum(len(text.encode("utf-8")) for text in state.files.values())
            expected_charge = {
                "attempted_tool_calls": 1,
                "tool_calls": call_charge,
                "read_tokens": read_charge,
                "read_tokenizer": old_budget["read_tokenizer"],
                "file_bytes_before": before_bytes,
                "file_bytes_after": after_bytes,
                "file_byte_delta": after_bytes - before_bytes,
            }
            expected_budget = json.loads(canonical_json(old_budget))
            consumed = expected_budget["consumed"]
            consumed["attempted_tool_calls"] = consumed.get("attempted_tool_calls", 0) + 1
            consumed["tool_calls"] = consumed.get("tool_calls", 0) + call_charge
            consumed["read_tokens"] = consumed.get("read_tokens", 0) + read_charge
            consumed["storage_bytes"] = after_bytes
            if record["budget_charge"] != expected_charge or new_budget != expected_budget:
                raise ProjectionError("tool result budget charge is false")
            old_context = ContextRevisionV1(
                messages=tuple(messages), tools=baseline.tools, rendering=baseline.rendering
            )
            new_context = ContextRevisionV1(
                messages=(*messages, message), tools=baseline.tools, rendering=baseline.rendering
            )
            if record["before_execution_hash"] != execution_value(
                store, before, old_context, old_budget
            ) or record["after_execution_hash"] != execution_value(
                store, state, new_context, new_budget
            ):
                raise ProjectionError("tool result execution fingerprint is false")
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
    if state != target.state:
        raise ProjectionError("semantic history does not reconstruct target state")
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
