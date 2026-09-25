"""Deterministic, writer-visible context selection and its immutable evidence.

This module never reads a file, private packet, or model.  The caller supplies the
already validated writer projection; replay checks the recorded selection against
that projection without invoking a summarizer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from writing_agent.task_graph import (
    ContextRevisionV1,
    EnvironmentStateV1,
    MessageV1,
    _Record,
    canonical_bytes,
    canonical_json,
    domain_hash_bytes,
    validate_hash,
)


class CompactionError(ValueError):
    """A context operation is not safe or its recorded evidence is false."""


@dataclass(frozen=True)
class ContextOperationV1(_Record):
    """Content-addressed immutable witness for one active-context replacement."""

    record_type: str = "ContextOperationV1"
    policy_ref: str = ""
    operation: str = ""
    old_context_ref: str = ""
    old_content_hash: str = ""
    old_messages: tuple[Mapping[str, Any], ...] = ()
    source_event_ids: tuple[str, ...] = ()
    source_range: Mapping[str, int] | None = None
    summary_ref: str | None = None
    summary_text: str | None = None
    summarizer_version: str | None = None
    summarizer_config: Mapping[str, Any] | None = None
    retained_tail: tuple[Mapping[str, Any], ...] = ()
    seed_name: str | None = None
    seed_checkpoint_ref: str | None = None
    new_context_ref: str = ""
    new_content_hash: str = ""
    new_messages: tuple[Mapping[str, Any], ...] = ()
    dropped_messages: tuple[Mapping[str, Any], ...] = ()
    charges: Mapping[str, Any] = MappingProxyType({})
    DOMAIN = "payload"

    def validate(self) -> None:
        if self.record_type != "ContextOperationV1" or self.operation not in {
            "carry",
            "seed",
            "drop",
            "compact",
        }:
            raise CompactionError("invalid context operation type")
        for identity in (
            self.policy_ref,
            self.old_context_ref,
            self.old_content_hash,
            self.new_context_ref,
            self.new_content_hash,
        ):
            validate_hash(identity)
        validate_hash(self.summary_ref, optional=True)
        validate_hash(self.seed_checkpoint_ref, optional=True)
        for identity in self.source_event_ids:
            validate_hash(identity)
        if self.source_range is not None and (
            not isinstance(self.source_range, Mapping)
            or set(self.source_range) != {"first_seq", "last_seq"}
            or any(type(value) is not int or value < 1 for value in self.source_range.values())
        ):
            raise CompactionError("invalid source event range")
        if self.summary_text is not None and not isinstance(self.summary_text, str):
            raise CompactionError("summary bytes must be UTF-8 text")
        if self.summary_text is not None and self.operation != "compact":
            raise CompactionError("noncompact operation includes summary bytes")
        if self.operation == "compact" and self.summary_text is None:
            raise CompactionError("compact lacks exact summary bytes")
        for name in ("old_messages", "new_messages", "dropped_messages", "retained_tail"):
            for item in getattr(self, name):
                if not isinstance(item, Mapping) or set(item) != {
                    "index",
                    "message_ref",
                    "origin",
                    "source_event_id",
                }:
                    raise CompactionError(f"invalid {name} evidence")
                if type(item["index"]) is not int or item["index"] < 0:
                    raise CompactionError(f"invalid {name} index")
                validate_hash(item["message_ref"])
                validate_hash(item["source_event_id"], optional=True)
                if not isinstance(item["origin"], str):
                    raise CompactionError(f"invalid {name} origin")
        if self.summarizer_config is not None and (
            not isinstance(self.summarizer_config, Mapping)
            or set(self.summarizer_config) != {"max_chars"}
            or type(self.summarizer_config["max_chars"]) is not int
            or self.summarizer_config["max_chars"] < 0
        ):
            raise CompactionError("invalid summarizer configuration")
        charge_fields = {
            "context_operations",
            "context_bytes_before",
            "context_bytes_after",
            "context_storage_bytes",
            "summary_bytes",
        }
        if (
            not isinstance(self.charges, Mapping)
            or set(self.charges) != charge_fields
            or any(type(value) is not int or value < 0 for value in self.charges.values())
        ):
            raise CompactionError("invalid context operation charges")


@dataclass(frozen=True)
class ContextPolicyV1:
    operation: str
    retained_exchanges: int = 0
    seed_name: str | None = None
    seed_checkpoint_ref: str | None = None
    summarizer_version: str | None = None
    max_summary_chars: int | None = None
    max_operations: int = 32
    max_context_bytes: int = 1_000_000
    max_context_storage_bytes: int = 8_000_000
    schema: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.schema) is not int
            or self.schema != 1
            or self.operation not in {"carry", "seed", "drop", "compact"}
        ):
            raise CompactionError("unsupported context policy")
        for name in (
            "retained_exchanges",
            "max_operations",
            "max_context_bytes",
            "max_context_storage_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise CompactionError(f"{name} must be a nonnegative integer")
        if self.operation == "compact":
            if (
                self.summarizer_version != "visible-text-v1"
                or type(self.max_summary_chars) is not int
                or self.max_summary_chars < 0
                or self.seed_name is not None
                or self.seed_checkpoint_ref is not None
            ):
                raise CompactionError("compact requires the fixed visible-text-v1 algorithm")
        elif self.summarizer_version is not None or self.max_summary_chars is not None:
            raise CompactionError("only compact may configure the summarizer")
        if self.operation == "seed":
            if (
                not isinstance(self.seed_name, str)
                or not self.seed_name
                or any(c.isspace() or ord(c) < 0x20 for c in self.seed_name)
            ):
                raise CompactionError("seed requires a named immutable prefix")
            self.seed_name.encode("utf-8", "strict")
            validate_hash(self.seed_checkpoint_ref)
        elif self.seed_name is not None or self.seed_checkpoint_ref is not None:
            raise CompactionError("only seed may name a prefix")
        if self.operation != "compact" and self.retained_exchanges:
            raise CompactionError("only compact may retain a tail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "ContextPolicyV1",
            "schema": self.schema,
            "operation": self.operation,
            "retained_exchanges": self.retained_exchanges,
            "seed_name": self.seed_name,
            "seed_checkpoint_ref": self.seed_checkpoint_ref,
            "summarizer_version": self.summarizer_version,
            "max_summary_chars": self.max_summary_chars,
            "max_operations": self.max_operations,
            "max_context_bytes": self.max_context_bytes,
            "max_context_storage_bytes": self.max_context_storage_bytes,
        }

    @classmethod
    def from_dict(cls, body: Any) -> ContextPolicyV1:
        if not isinstance(body, dict) or set(body) != set(cls("carry").to_dict()):
            raise CompactionError("invalid context policy schema")
        if body["record_type"] != "ContextPolicyV1":
            raise CompactionError("invalid context policy type")
        value = cls(**{key: value for key, value in body.items() if key != "record_type"})
        if value.to_dict() != body:
            raise CompactionError("noncanonical context policy")
        return value


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


def verify_fixed_summary(messages: tuple[MessageV1, ...], max_chars: int, recorded: str) -> None:
    """Check recorded bytes without asking a summarizer to regenerate them."""
    cursor = 0
    for index, line in enumerate(_summary_lines(messages)):
        chunk = ("\n" if index else "") + line
        available = max(0, max_chars - cursor)
        expected = chunk[:available]
        if recorded[cursor : cursor + len(expected)] != expected:
            raise CompactionError("recorded summary differs from visible source messages")
        cursor += len(expected)
    if cursor != len(recorded):
        raise CompactionError("recorded summary has extra or missing text")


def select_context(
    old: ContextRevisionV1,
    sources: tuple[str | None, ...],
    policy: ContextPolicyV1,
    *,
    seed: ContextRevisionV1 | None = None,
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
        if recorded_summary is None:
            summary = fixed_summary(selected, policy.max_summary_chars)
        else:
            verify_fixed_summary(selected, policy.max_summary_chars, recorded_summary)
            summary = recorded_summary
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


def context_bytes(messages: tuple[MessageV1, ...], old: ContextRevisionV1) -> int:
    return len(
        canonical_bytes(
            {
                "messages": [message.to_dict() for message in messages],
                "tools": list(old.tools),
                "rendering": dict(old.rendering),
            }
        )
    )


def message_evidence(
    messages: tuple[MessageV1, ...], sources: tuple[str | None, ...]
) -> list[dict]:
    return [
        {
            "index": index,
            "message_ref": message.identity(),
            "origin": message.origin,
            "source_event_id": sources[index],
        }
        for index, message in enumerate(messages)
    ]


def source_range(store, source_ids: list[str]) -> dict[str, int] | None:
    if not source_ids:
        return None
    sequences = [store.load_event(identity).seq for identity in source_ids]
    return {"first_seq": min(sequences), "last_seq": max(sequences)}


def summary_hash(summary: str | None) -> str | None:
    return None if summary is None else domain_hash_bytes("payload", summary.encode("utf-8"))


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
    old_context: ContextRevisionV1,
    new_context: ContextRevisionV1,
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


def charge_context_append(old_budget: dict, new_context: ContextRevisionV1) -> dict | None:
    """Meter a paid visible observation once context accounting has been activated.

    An observation may exceed the allowance after the action has already been paid.
    Preserve that action and its exact charge; the next request is blocked and a
    drained writer boundary can seal a valid context-budget stop.
    """
    import json

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


def make_record(
    store,
    before: EnvironmentStateV1,
    old: ContextRevisionV1,
    sources: tuple[str | None, ...],
    policy: ContextPolicyV1,
    policy_ref: str,
    new: ContextRevisionV1,
    summary_ref: str | None,
    old_budget: dict,
    *,
    seed: ContextRevisionV1 | None = None,
    seed_sources: tuple[str | None, ...] = (),
    recorded_summary: str | None = None,
) -> tuple[dict, dict, tuple[str | None, ...]]:
    """Recompute all claims from the authorized active projection."""
    require_quiescent(before)
    origin = f"{before.position['lineage_id']}:context:{before.history['seq'] + 1}"
    messages, selected_sources, summary, removed = select_context(
        old,
        sources,
        policy,
        seed=seed,
        seed_sources=seed_sources,
        summary_origin=origin,
        recorded_summary=recorded_summary,
    )
    if messages != new.messages or old.tools != new.tools or old.rendering != new.rendering:
        raise CompactionError("new context differs from the deterministic selection")
    if new.event_head != before.history["head"] or new.provenance_refs != (
        (before.history["head"],) if before.history["head"] is not None else ()
    ):
        raise CompactionError("new context has false revision provenance")
    budget, charges = charge_budget(old_budget, policy, old, new, summary)
    if summary_ref != summary_hash(summary):
        raise CompactionError("recorded summary bytes do not match fixed summary")
    if summary_ref is not None and store.get_artifact(
        summary_ref, expected_domain="payload:bytes"
    ) != summary.encode("utf-8"):
        raise CompactionError("summary artifact has different exact bytes")
    dropped = message_evidence(old.messages, sources)
    dropped = [dropped[index] for index in removed]
    if policy.operation == "carry":
        source_values = sources
    elif policy.operation == "seed":
        source_values = seed_sources[2:]
    else:
        source_values = tuple(sources[index] for index in removed)
    source_ids = list(dict.fromkeys(value for value in source_values if value is not None))
    groups = completed_exchanges(old.messages)
    keep = min(policy.retained_exchanges, len(groups))
    retained_tail = (
        message_evidence(old.messages, sources)[groups[-keep][0] :]
        if policy.operation == "compact" and keep
        else []
    )
    record = {
        "record_type": "ContextOperationV1",
        "schema": 1,
        "policy_ref": policy_ref,
        "operation": policy.operation,
        "old_context_ref": before.context_ref,
        "old_content_hash": old.content_hash,
        "old_messages": message_evidence(old.messages, sources),
        "source_event_ids": source_ids,
        "source_range": source_range(store, source_ids),
        "summary_ref": summary_ref,
        "summary_text": summary,
        "summarizer_version": policy.summarizer_version,
        "summarizer_config": (
            {"max_chars": policy.max_summary_chars} if policy.operation == "compact" else None
        ),
        "retained_tail": retained_tail,
        "seed_name": policy.seed_name,
        "seed_checkpoint_ref": policy.seed_checkpoint_ref,
        "new_context_ref": new.identity(),
        "new_content_hash": new.content_hash,
        "new_messages": message_evidence(new.messages, selected_sources),
        "dropped_messages": dropped,
        "charges": charges,
    }
    return ContextOperationV1.from_dict(record).to_dict(), budget, selected_sources
