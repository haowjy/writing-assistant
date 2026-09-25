"""Pure writer-runtime budget policy shared by producers and semantic replay.

Persisted charges remain claims: replay loads them independently and compares each
claim with the result of these functions. No function reads or writes the store.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from writing_agent.task_graph import ContextRevisionV1, canonical_bytes, canonical_json
from writing_agent.task_graph_compaction import context_bytes

READ_TOOLS = frozenset({"read_file", "search", "list_dir"})
EXHAUSTION_ORDER = (
    "writer_turns",
    "generated_tokens",
    "total_tokens",
    "context_bytes",
    "context_storage_bytes",
)


def sampled_usage_charge(budget: dict, usage: Mapping[str, Any]) -> tuple[dict, str | None]:
    """Charge one sampled call, including an overrun, with parent totals counted once."""
    result = json.loads(canonical_json(budget))
    consumed = result["consumed"]
    consumed["writer_turns"] = consumed.get("writer_turns", 0) + 1
    consumed["model_calls"] = consumed.get("model_calls", 0) + 1
    consumed["generated_tokens"] = consumed.get("generated_tokens", 0) + usage.get(
        "completion_tokens", 0
    )
    consumed["total_tokens"] = consumed.get("total_tokens", 0) + usage.get(
        "total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    )
    exceeded = next(
        (
            name
            for name in ("generated_tokens", "total_tokens")
            if name in result["limits"] and consumed[name] > result["limits"][name]
        ),
        None,
    )
    return result, exceeded


def exhausted_stop_reason(budget: Mapping[str, Any]) -> str | None:
    """Return the first drained-boundary exhaustion reason in protocol order."""
    exhausted = next(
        (
            name
            for name in EXHAUSTION_ORDER
            if name in budget["limits"]
            and budget["consumed"].get(name, 0) >= budget["limits"][name]
        ),
        None,
    )
    if exhausted is None:
        return None
    return {
        "writer_turns": "writer_budget",
        "context_bytes": "context_budget",
        "context_storage_bytes": "context_storage_budget",
    }.get(exhausted, f"{exhausted}_budget")


def tool_error(budget: Mapping[str, Any], validation_error: str | None, name: str) -> dict | None:
    """Apply tool, syntax, then author exhaustion precedence before dispatch."""
    consumed, limits = budget["consumed"], budget["limits"]
    if consumed.get("tool_calls", 0) >= limits["tool_calls"]:
        return {"ok": False, "valid": True, "error": "Tool-call budget exceeded"}
    if validation_error is not None:
        return {"ok": False, "valid": False, "error": validation_error}
    if name == "ask_author":
        if consumed.get("author_calls", 0) >= limits["author_calls"]:
            return {"ok": False, "valid": True, "error": "Author-call budget exceeded"}
    return None


def observation_read_tokens(observation: Mapping[str, Any], name: str, tokenizer: str) -> int:
    """Count returned read text, not rejected or failed observations."""
    if not observation.get("ok") or name not in READ_TOOLS:
        return 0
    if tokenizer != "whitespace-v1":
        raise ValueError("unsupported read tokenizer")
    return len(json.dumps(observation["result"], ensure_ascii=False).split())


def tool_result_charge(
    budget: dict,
    before_files: Mapping[str, str],
    after_files: Mapping[str, str],
    read_tokens: int,
) -> tuple[dict, dict]:
    """Return exact persisted tool charge and next budget without mutating inputs."""
    before_bytes = sum(len(value.encode("utf-8")) for value in before_files.values())
    after_bytes = sum(len(value.encode("utf-8")) for value in after_files.values())
    result, call_charge = charge_tool_attempt(budget)
    charge = {
        "attempted_tool_calls": 1,
        "tool_calls": call_charge,
        "read_tokens": read_tokens,
        "read_tokenizer": budget["read_tokenizer"],
        "file_bytes_before": before_bytes,
        "file_bytes_after": after_bytes,
        "file_byte_delta": after_bytes - before_bytes,
    }
    consumed = result["consumed"]
    consumed["read_tokens"] = consumed.get("read_tokens", 0) + read_tokens
    consumed["storage_bytes"] = after_bytes
    return result, charge


def charge_tool_attempt(budget: dict) -> tuple[dict, int]:
    """Charge one attempted tool call without result or file/read accounting."""
    result = json.loads(canonical_json(budget))
    consumed = result["consumed"]
    call_charge = int(consumed.get("tool_calls", 0) < result["limits"]["tool_calls"])
    consumed["attempted_tool_calls"] = consumed.get("attempted_tool_calls", 0) + 1
    consumed["tool_calls"] = consumed.get("tool_calls", 0) + call_charge
    return result, call_charge


def charge_context_append(old_budget: dict, new_context: ContextRevisionV1) -> dict | None:
    """Meter a visible append when context accounting has been activated.

    The paid action remains committed if the append itself crosses the limit; the
    next request is blocked and a drained writer boundary can record the stop.
    """
    if (
        not isinstance(old_budget, dict)
        or not isinstance(old_budget.get("limits"), dict)
        or "context_bytes" not in old_budget["limits"]
    ):
        return None
    budget = json.loads(canonical_json(old_budget))
    consumed = budget["consumed"]
    consumed["context_bytes"] = context_bytes(new_context.messages, new_context)
    consumed["context_storage_bytes"] = consumed.get("context_storage_bytes", 0) + len(
        canonical_bytes(new_context.to_dict())
    )
    return budget
