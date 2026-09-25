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


_WRITER_EFFECT_FIELDS = {
    "writer_action": (
        {"continuation", "position", "budgets_ref", "external_inputs_ref"},
        {"action_ids"},
    ),
    "tool_result": (
        {"continuation", "position", "budgets_ref", "external_inputs_ref"},
        {"tool_result_ids"},
    ),
    "budget_charged": ({"position", "budgets_ref", "outcome_ref", "external_inputs_ref"}, set()),
    "termination_recorded": ({"position", "outcome_ref", "external_inputs_ref"}, set()),
}
NATIVE_TRACE_REASON = "native token alignment and loss masks are not implemented in Phase 4"
_ACTION_RECORD_FIELDS = {
    "record_type",
    "action_id",
    "trace_ref",
    "request_ref",
    "prepared_request_ref",
    "raw_output_ref",
    "logprob_ref",
    "calls",
    "usage",
    "model",
    "seed",
    "loss_eligibility",
}
_STOP_RECORD_FIELDS = {
    "record_type",
    "action_id",
    "reason",
    "usage",
    "model",
    "seed",
    "parsed_message_json",
    "trace_ref",
    "request_ref",
    "prepared_request_ref",
    "raw_output_ref",
    "logprob_ref",
}
_RESULT_RECORD_FIELDS = {
    "record_type",
    "result_id",
    "call_id",
    "action_id",
    "observation",
    "file_delta",
    "before_execution_hash",
    "after_execution_hash",
    "budget_charge",
    "loss_eligibility",
}
_TRACE_FIELDS = {
    "record_type",
    "action_id",
    "context_content_hash",
    "context_revision_ref",
    "rendering",
    "exact_request_ref",
    "prepared_request_ref",
    "raw_output_ref",
    "raw_output_evidence",
    "logprob_ref",
    "adapter_trace",
    "token_evidence",
    "logprob_evidence",
    "usage",
    "model",
    "seed",
    "native_on_policy_eligible",
    "reason",
}


def validate_writer_effect(before, after, event, effect, *, store=None) -> None:
    """Enforce the complete Phase 4 field-ownership and phase boundary.

    The generic Phase 2 reducer intentionally does not know event authority. This
    check is used both before publication and while projecting restored history.
    """
    allowed_set, allowed_history = _WRITER_EFFECT_FIELDS[event.kind]
    if set(effect["set"]) != allowed_set or set(effect["history_set"]) != allowed_history:
        raise ProjectionError("writer event changes fields outside its authority")
    old = before.to_dict()
    new = after.to_dict()
    for key in old:
        if key in {"history", "position", "continuation", "files", "tree_hash"}:
            continue
        if key not in allowed_set and old[key] != new[key]:
            raise ProjectionError(f"writer event changed unrelated {key}")
    for key in old["history"]:
        if key in {"head", "seq", *allowed_history}:
            continue
        if old["history"][key] != new["history"][key]:
            raise ProjectionError(f"writer event changed unrelated history {key}")
    for key in old["position"]:
        if key != "phase" and old["position"][key] != new["position"][key]:
            raise ProjectionError(f"writer event changed unrelated position {key}")
    for key in old["continuation"]:
        if (
            key not in {"tool_queue", "next_call"}
            and old["continuation"][key] != new["continuation"][key]
        ):
            raise ProjectionError(f"writer event changed unrelated continuation {key}")
    if event.kind != "tool_result" and (old["files"] != new["files"] or effect["file_delta"]):
        raise ProjectionError("non-tool writer event changed files")
    if event.kind == "tool_result" and old["files"] == new["files"] and effect["file_delta"]:
        raise ProjectionError("tool result has false file delta")
    if (
        event.kind == "tool_result"
        and old["continuation"]["tool_queue"] != new["continuation"]["tool_queue"]
    ):
        raise ProjectionError("tool result replaced its queue")
    if (
        event.kind in {"budget_charged", "termination_recorded"}
        and old["continuation"] != new["continuation"]
    ):
        raise ProjectionError("writer stop changed continuation")
    if old["position"]["phase"] != "ready_writer":
        raise ProjectionError("writer event has illegal pre-phase")
    if store is not None:
        prior_outcome = store.get_artifact(before.outcome_ref, expected_domain="payload")
        if isinstance(prior_outcome, dict) and (
            prior_outcome.get("task_status") not in {None, "unknown", "incomplete"}
            or prior_outcome.get("execution_status") not in {None, "running", "valid"}
            or prior_outcome.get("reward_status") not in {None, "pending"}
            or prior_outcome.get("training_eligibility") not in {None, "pending"}
            or prior_outcome.get("stop_reason") is not None
        ):
            raise ProjectionError("writer event has illegal pre-status")
    previous_outcome = before.outcome_ref
    if event.kind in {"writer_action", "tool_result"} and after.outcome_ref != previous_outcome:
        raise ProjectionError("writer action or result changed outcome status")
    if event.kind == "writer_action":
        expected = "ready_writer" if new["continuation"]["tool_queue"] else "checking"
        if (
            old["continuation"]["next_call"] != len(old["continuation"]["tool_queue"])
            or new["position"]["phase"] != expected
        ):
            raise ProjectionError("writer action has illegal phase or pending queue")
    elif event.kind == "tool_result":
        if new["position"]["phase"] != "ready_writer" or old["continuation"]["next_call"] >= len(
            old["continuation"]["tool_queue"]
        ):
            raise ProjectionError("tool result has illegal phase or empty queue")
    else:
        if new["position"]["phase"] != "terminal" or old["continuation"]["next_call"] != len(
            old["continuation"]["tool_queue"]
        ):
            raise ProjectionError("writer stop has illegal phase or pending queue")
        if store is not None:
            outcome = store.get_artifact(after.outcome_ref, expected_domain="payload")
            if (
                not isinstance(outcome, dict)
                or outcome
                != {
                    "schema": 1,
                    "task_status": "incomplete",
                    "execution_status": "valid",
                    "stop_reason": outcome.get("stop_reason"),
                    "reward_status": "pending",
                    "training_eligibility": "pending",
                }
                or outcome["stop_reason"]
                not in {"writer_budget", "generated_tokens_budget", "total_tokens_budget"}
            ):
                raise ProjectionError("writer stop has illegal outcome status")


def validate_action_trace(
    store, record, trace, action_id, context_hash, context_ref, rendering, message=None
) -> None:
    """Bind every duplicated sampling claim to the owning action and request."""
    if (
        not isinstance(trace, dict)
        or set(trace) != _TRACE_FIELDS
        or trace.get("record_type") != "WriterActionTraceV1"
        or trace["native_on_policy_eligible"] is not False
        or trace["reason"] != NATIVE_TRACE_REASON
    ):
        raise ProjectionError("writer action trace has wrong type")
    if record.get("record_type") == "WriterActionV1":
        if set(record) != _ACTION_RECORD_FIELDS or message is None:
            raise ProjectionError("writer action record has wrong schema")
        if message.role != "assistant" or not message.loss_eligible or message.origin != action_id:
            raise ProjectionError("writer action message has wrong role or origin")
        expected_loss = {
            "assistant_text": any(part["type"] == "text" for part in message.content),
            "tool_syntax": any(part["type"] == "tool_call" for part in message.content),
            "assistant_ending": True,
            "system": False,
            "user": False,
            "author": False,
            "tool_observation": False,
            "seed": False,
            "environment": False,
        }
        if record["loss_eligibility"] != expected_loss:
            raise ProjectionError("writer action loss eligibility contradicts message")
    elif record.get("record_type") == "WriterSampledBudgetStopV1":
        if set(record) != _STOP_RECORD_FIELDS:
            raise ProjectionError("sampled stop record has wrong schema")
    else:
        raise ProjectionError("trace has no owning action or sampled stop")
    claims = {
        "action_id": action_id,
        "context_content_hash": context_hash,
        "context_revision_ref": context_ref,
        "exact_request_ref": record["request_ref"],
        "prepared_request_ref": record["prepared_request_ref"],
        "raw_output_ref": record["raw_output_ref"],
        "logprob_ref": record["logprob_ref"],
        "usage": record["usage"],
        "model": record["model"],
        "seed": record["seed"],
    }
    if any(trace.get(key) != value for key, value in claims.items()):
        raise ProjectionError("writer trace contradicts its action")
    for key in ("request_ref", "raw_output_ref"):
        if record[key] is not None:
            store.get_artifact(record[key])
    if record["logprob_ref"] is not None:
        store.get_artifact(record["logprob_ref"], expected_domain="payload:bytes")
    if canonical_json(trace.get("rendering")) != canonical_json(rendering):
        raise ProjectionError("writer trace rendering differs from context")
    adapter = trace.get("adapter_trace")
    if adapter is not None and (
        not isinstance(adapter, dict)
        or any(key in adapter and adapter[key] != trace[key] for key in ("model", "seed", "usage"))
        or "native_on_policy_eligible" in adapter
    ):
        raise ProjectionError("adapter trace contradicts sampling claims")
    if trace.get("raw_output_evidence") != (
        "supplied" if record["raw_output_ref"] is not None else "missing"
    ):
        raise ProjectionError("raw output evidence contradicts reference")
    if trace.get("logprob_evidence") != (
        "supplied" if record["logprob_ref"] is not None else "missing"
    ):
        raise ProjectionError("logprob evidence contradicts reference")
    if trace.get("token_evidence") != (
        "supplied" if isinstance(adapter, dict) and "generated_token_ids" in adapter else "missing"
    ):
        raise ProjectionError("token evidence contradicts adapter trace")
    if (
        isinstance(adapter, dict)
        and "generated_token_ids" in adapter
        and (
            not isinstance(adapter["generated_token_ids"], list)
            or any(type(token) is not int or token < 0 for token in adapter["generated_token_ids"])
        )
    ):
        raise ProjectionError("generated token IDs are invalid")
    prepared_ref = record["prepared_request_ref"]
    if prepared_ref is not None:
        prepared = store.get_artifact(prepared_ref, expected_domain="payload")
        if not isinstance(prepared, dict) or prepared != {
            "record_type": "PreparedWriterRequestV1",
            "context_content_hash": context_hash,
            "context_revision_ref": context_ref,
            "rendering": rendering,
            "payload_ref": record["request_ref"],
        }:
            raise ProjectionError("prepared request contradicts action trace")


def validate_result_production(
    store, before, after, old_context, new_context, record, message, effect, action
) -> None:
    """Reject contradictory result metadata before its authority is published."""
    cursor = before.continuation["next_call"]
    call = before.continuation["tool_queue"][cursor]
    if (
        record.get("record_type") != "WriterToolResultV1"
        or set(record) != _RESULT_RECORD_FIELDS
        or record["loss_eligibility"] != {"tool_observation": False}
        or record["action_id"] != action["action_id"]
        or message.origin != action["action_id"]
        or record["call_id"] != call["call_id"]
        or message.call_id != call["call_id"]
        or record["result_id"] != after.history["tool_result_ids"][-1]
        or after.continuation["next_call"] != cursor + 1
    ):
        raise ProjectionError("tool result has false action, call, or cursor")
    delta = {
        path: {"before": before.files.get(path), "after": after.files.get(path)}
        for path in sorted(set(before.files) | set(after.files))
        if before.files.get(path) != after.files.get(path)
    }
    if record["file_delta"] != delta or effect["file_delta"] != delta:
        raise ProjectionError("tool result has false file delta")
    old_budget = store.get_artifact(before.budgets_ref, expected_domain="payload")
    new_budget = store.get_artifact(after.budgets_ref, expected_domain="payload")
    consumed = old_budget["consumed"]
    call_charge = int(consumed.get("tool_calls", 0) < old_budget["limits"]["tool_calls"])
    read_charge = 0
    if record["observation"].get("ok") and call["name"] in {"read_file", "search", "list_dir"}:
        if old_budget["read_tokenizer"] != "whitespace-v1":
            raise ProjectionError("unsupported read tokenizer")
        read_charge = len(json.dumps(record["observation"]["result"], ensure_ascii=False).split())
    before_bytes = sum(len(text.encode("utf-8")) for text in before.files.values())
    after_bytes = sum(len(text.encode("utf-8")) for text in after.files.values())
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
    charged = expected_budget["consumed"]
    charged["attempted_tool_calls"] = charged.get("attempted_tool_calls", 0) + 1
    charged["tool_calls"] = charged.get("tool_calls", 0) + call_charge
    charged["read_tokens"] = charged.get("read_tokens", 0) + read_charge
    charged["storage_bytes"] = after_bytes
    if record["budget_charge"] != expected_charge or new_budget != expected_budget:
        raise ProjectionError("tool result has false budget charge")
    if (
        record["before_execution_hash"] != execution_value(store, before, old_context, old_budget)
        or record["after_execution_hash"] != execution_value(store, after, new_context, new_budget)
        or message.role != "tool"
        or message.loss_eligible
        or len(message.content) != 1
        or canonical_json(message.content[0])
        != canonical_json(
            {"type": "tool_result", "call_id": call["call_id"], "content": record["observation"]}
        )
    ):
        raise ProjectionError("tool result has false message or execution fingerprint")


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
        if event.kind in _WRITER_EFFECT_FIELDS:
            validate_writer_effect(before, state, event, effect, store=store)
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
            validate_action_trace(
                store,
                record,
                trace,
                f"{event.rollout_id}:action:{len(action_ids)}",
                context_content_hash(
                    tuple(messages), tools=baseline.tools, rendering=baseline.rendering
                ),
                latest_context_ref,
                baseline.rendering,
            )
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
                or outcome
                != {
                    "schema": 1,
                    "task_status": "incomplete",
                    "execution_status": "valid",
                    "stop_reason": record["reason"],
                    "reward_status": "pending",
                    "training_eligibility": "pending",
                }
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
                or outcome
                != {
                    "schema": 1,
                    "task_status": "incomplete",
                    "execution_status": "valid",
                    "stop_reason": reason,
                    "reward_status": "pending",
                    "training_eligibility": "pending",
                }
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
            validate_action_trace(
                store,
                record,
                trace,
                record["action_id"],
                context_content_hash(
                    tuple(messages), tools=baseline.tools, rendering=baseline.rendering
                ),
                latest_context_ref,
                baseline.rendering,
                message,
            )
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
            old_context = ContextRevisionV1(
                messages=tuple(messages), tools=baseline.tools, rendering=baseline.rendering
            )
            new_context = ContextRevisionV1(
                messages=(*messages, message), tools=baseline.tools, rendering=baseline.rendering
            )
            validate_result_production(
                store,
                before,
                state,
                old_context,
                new_context,
                record,
                message,
                effect,
                {"action_id": source[0]},
            )
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
