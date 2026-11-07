"""Deterministic, writer-visible context selection and byte accounting."""

from __future__ import annotations

from writing_agent.task_graph import (
    EnvironmentStateV1,
    MaterializedContextV1,
    MessageV1,
    canonical_bytes,
    canonical_json,
)
from writing_agent.task_graph_record_contracts import CompactionError, ContextPolicyV1


def completed_exchanges(messages: tuple[MessageV1, ...]) -> tuple[tuple[int, int], ...]:
    """Return complete post-request exchanges; reject any half tool/author turn."""
    if len(messages) < 2 or messages[0].role != "system" or messages[1].role != "user":
        raise CompactionError("context lacks its immutable system/request prefix")
    groups: list[tuple[int, int]] = []
    start = 2
    while start < len(messages):
        message = messages[start]
        # A prior environment summary is a context segment, not a completed
        # writer/author exchange and must not consume retained-tail allowance.
        if message.role == "user" and ":context:" in message.origin:
            start += 1
            continue
        if message.role not in {"assistant", "user"}:
            raise CompactionError("exchange starts with an unmatched observation")
        index = start + 1
        calls = [part["id"] for part in message.content if part["type"] == "tool_call"]
        if message.role == "assistant":
            while calls:
                if index >= len(messages):
                    raise CompactionError("context ends with an unmatched tool call")
                result = messages[index]
                if result.role != "tool" or result.call_id != calls.pop(0):
                    raise CompactionError("tool exchange is incomplete or reordered")
                index += 1
            # A scripted clarification tool result is followed by one author reply.
            if any(
                part["type"] == "tool_call" and part["name"] == "ask_author"
                for part in message.content
            ):
                if index >= len(messages) or messages[index].role != "user":
                    raise CompactionError("author request has no completed reply")
                index += 1
        groups.append((start, index))
        start = index
    return tuple(groups)


def _summary_lines(messages: tuple[MessageV1, ...]):
    for message in messages:
        parts = []
        for part in message.content:
            if part["type"] == "text":
                parts.append(part["text"])
            else:
                parts.append(canonical_json(part))
        yield f"{message.role}: {' '.join(parts)}"


def fixed_summary(messages: tuple[MessageV1, ...], max_chars: int) -> str:
    """A pinned character-bounded rendering of *visible* messages only."""
    return "\n".join(_summary_lines(messages))[:max_chars]


def select_context(
    old: MaterializedContextV1,
    sources: tuple[str | None, ...],
    policy: ContextPolicyV1,
    *,
    seed: MaterializedContextV1 | None = None,
    seed_sources: tuple[str | None, ...] = (),
    summary_origin: str,
    recorded_summary: str | None = None,
) -> tuple[tuple[MessageV1, ...], tuple[str | None, ...], str | None, tuple[int, ...]]:
    """Select exact messages and return their source identities and removed indices."""
    if len(sources) != len(old.messages):
        raise CompactionError("source ledger does not match old context")
    groups = completed_exchanges(old.messages)
    prefix = old.messages[:2]
    prefix_sources = sources[:2]
    if policy.operation == "carry":
        messages, selected_sources = old.messages, sources
        summary = None
        removed = ()
    elif policy.operation == "drop":
        messages, selected_sources = prefix, prefix_sources
        summary = None
        removed = tuple(range(2, len(old.messages)))
    elif policy.operation == "seed":
        if seed is None:
            raise CompactionError("named seed is missing")
        completed_exchanges(seed.messages)
        if (
            seed.messages[:2] != prefix
            or seed.tools != old.tools
            or seed.rendering != old.rendering
        ):
            raise CompactionError("seed does not share the admitted system/request/tools")
        if len(seed_sources) != len(seed.messages):
            raise CompactionError("seed source ledger is incomplete")
        messages = (
            *prefix,
            *(MessageV1(**{**m.to_dict(), "loss_eligible": False}) for m in seed.messages[2:]),
        )
        selected_sources = (*prefix_sources, *seed_sources[2:])
        summary = None
        removed = tuple(range(2, len(old.messages)))
    else:
        keep = min(policy.retained_exchanges, len(groups))
        split = groups[-keep][0] if keep else len(old.messages)
        selected = old.messages[2:split]
        summary = fixed_summary(selected, policy.max_summary_chars)
        if recorded_summary is not None and summary != recorded_summary:
            raise CompactionError("recorded summary differs from visible source messages")
        summary_message = MessageV1(
            role="user",
            content=(summary,),
            origin=summary_origin,
            trust="untrusted_data",
            loss_eligible=False,
        )
        messages = (*prefix, summary_message, *old.messages[split:])
        selected_sources = (*prefix_sources, None, *sources[split:])
        removed = tuple(range(2, split))
    return tuple(messages), tuple(selected_sources), summary, removed


def context_bytes(messages: tuple[MessageV1, ...], old: MaterializedContextV1) -> int:
    return len(
        canonical_bytes(
            {
                "messages": [message.to_dict() for message in messages],
                "tools": list(old.tools),
                "rendering": dict(old.rendering),
            }
        )
    )


def require_quiescent(state: EnvironmentStateV1, *, pending: tuple[str, ...] = ()) -> None:
    """A context operation cannot interrupt an exchange or environment transaction."""
    continuation = state.continuation
    if (
        state.position["phase"] != "ready_writer"
        or continuation["next_call"] != len(continuation["tool_queue"])
        or continuation["author_request"] is not None
        or continuation["check_requests"]
        or continuation["external_requests"]
        or state.in_flight_effects
        or pending
    ):
        raise CompactionError("context operation requires a quiescent completed exchange")


def charge_budget(
    old_budget: dict,
    policy: ContextPolicyV1,
    old_context: MaterializedContextV1,
    new_context: MaterializedContextV1,
    summary: str | None,
) -> tuple[dict, dict]:
    """Meter the active context and immutable context storage before publication."""
    import json

    budget = json.loads(canonical_json(old_budget))
    if set(budget) != {"schema", "limits", "consumed", "read_tokenizer"} or budget["schema"] != 1:
        raise CompactionError("context operation lacks a valid budget")
    limits = budget["limits"]
    configured = {
        "context_operations": policy.max_operations,
        "context_bytes": policy.max_context_bytes,
        "context_storage_bytes": policy.max_context_storage_bytes,
    }
    for key, value in configured.items():
        if key in limits and limits[key] != value:
            raise CompactionError("context budget policy changed within the lineage")
        limits[key] = value
    consumed = budget["consumed"]
    bytes_before = context_bytes(old_context.messages, old_context)
    if "context_bytes" in consumed and consumed["context_bytes"] != bytes_before:
        raise CompactionError("old active context byte charge is false")
    bytes_now = context_bytes(new_context.messages, new_context)
    storage_charge = len(canonical_bytes(new_context.to_dict())) + (
        len(summary.encode("utf-8")) if summary is not None else 0
    )
    charges = {
        "context_operations": 1,
        "context_bytes_before": bytes_before,
        "context_bytes_after": bytes_now,
        "context_storage_bytes": storage_charge,
        "summary_bytes": len(summary.encode("utf-8")) if summary is not None else 0,
    }
    consumed["context_operations"] = consumed.get("context_operations", 0) + 1
    consumed["context_bytes"] = bytes_now
    consumed["context_storage_bytes"] = consumed.get("context_storage_bytes", 0) + storage_charge
    if any(consumed[key] > limits[key] for key in configured):
        raise CompactionError("context operation would exhaust its declared budget")
    return budget, charges
