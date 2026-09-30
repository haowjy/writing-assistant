"""Pure native token layout shared by group credit and training export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from writing_agent.task_graph import CheckpointV1, EventV1, domain_hash_bytes
from writing_agent.task_graph_context_roots import context_root_changed_after
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_record_contracts import GroupError, GroupSpecV1
from writing_agent.task_graph_records import WriterTurnV2
from writing_agent.task_graph_token_ledger import decode_u32_token_ids, encode_u32_token_ids
from writing_agent.task_graph_transition import ArtifactReader


@dataclass(frozen=True)
class _Turn:
    ref: str
    record: WriterTurnV2
    input_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]


@dataclass(frozen=True)
class TokenLayout:
    """Loss sequence, generated-token mask, and exact per-turn offsets."""

    prompt_ids: tuple[int, ...]
    prompt_ids_ref: str | None
    completion_ids: tuple[int, ...]
    env_mask: tuple[int, ...]
    turn_spans: tuple[Mapping[str, Any], ...]
    trailing_context_limit_turn_ref: str | None
    ledger_hash: str


def max_context_tokens_for_group(spec: GroupSpecV1, reader: ArtifactReader) -> int | None:
    """Read and validate the one context cap used by coordinator and export layout."""
    body = _read_payload(reader, spec.environment["budget_ref"], "entry budget")
    limits = body.get("limits")
    if not isinstance(limits, Mapping):
        raise GroupError("native group entry budget has no limits")
    value = limits.get("context_tokens")
    if value is not None and (type(value) is not int or value < 0):
        raise GroupError("native group context-token cap is invalid")
    return value


def token_layout_for_checkpoint(
    checkpoint_id: str,
    member_id: str,
    reader: ArtifactReader,
    *,
    max_context_tokens: int | None,
) -> TokenLayout:
    """Derive V2 token sequence and spans from the verified member event chain."""
    try:
        checkpoint = reader.checkpoint(checkpoint_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise GroupError("training member checkpoint is unavailable") from exc
    if not isinstance(checkpoint, CheckpointV1):
        try:
            checkpoint = CheckpointV1.from_dict(checkpoint)
        except (TypeError, ValueError) as exc:
            raise GroupError("training member checkpoint is invalid") from exc
    if checkpoint.state.position.get("lineage_id") != member_id:
        raise GroupError("training checkpoint belongs to a different member")

    events: list[EventV1] = []
    turn_events: list[tuple[str, WriterTurnV2]] = []
    seen: set[str] = set()
    event_ref = checkpoint.event_head
    while event_ref is not None:
        if event_ref in seen:
            raise GroupError("training member event history contains a cycle")
        seen.add(event_ref)
        try:
            event = EventV1.from_dict(reader.artifact(event_ref, domain="event"))
        except (KeyError, TypeError, ValueError) as exc:
            raise GroupError("training member event history is unavailable") from exc
        if event.lineage_id != member_id:
            break
        events.append(event)
        if event.kind in {"writer_action", "budget_charged"}:
            try:
                body = reader.artifact(event.payload_ref)
                record_type = body.get("record_type") if isinstance(body, Mapping) else None
                if record_type != WriterTurnV2.RECORD_TYPE:
                    raise GroupError("training member contains a non-V2 writer turn")
                turn = WriterTurnV2.from_dict(body)
            except GroupError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise GroupError("training member writer turn is invalid") from exc
            turn_events.append((event.payload_ref, turn))
        event_ref = event.previous

    if any(event.kind == "budget_charged" for event in events):
        raise GroupError("overrun writer turns cannot be exported")
    try:
        context_changed = context_root_changed_after(reader, checkpoint.event_head, None)
    except ProjectionError as exc:
        raise GroupError("training member context history is invalid") from exc
    if context_changed:
        raise GroupError("training member has multiple context roots")
    turns = tuple(_read_turn(reader, ref, turn) for ref, turn in reversed(turn_events))
    return _layout_turns(turns, max_context_tokens=max_context_tokens)


def training_turn_spans(
    checkpoint_id: str,
    member_id: str,
    reader: ArtifactReader,
    *,
    max_context_tokens: int | None,
) -> Mapping[str, tuple[int, int]]:
    """Return generated-token offsets for eligible native group credits."""
    layout = token_layout_for_checkpoint(
        checkpoint_id,
        member_id,
        reader,
        max_context_tokens=max_context_tokens,
    )
    return {
        span["turn_ref"]: (span["completion_start"], span["completion_end"])
        for span in layout.turn_spans
    }


def _read_turn(reader: ArtifactReader, ref: str, turn: WriterTurnV2) -> _Turn:
    input_ids = _read_token_ids(reader, turn.input_token_ids_ref, turn.input_token_count)
    generated_ids = _read_token_ids(
        reader, turn.generated_token_ids_ref, turn.generated_token_count
    )
    return _Turn(ref, turn, input_ids, generated_ids)


def _layout_turns(turns: tuple[_Turn, ...], *, max_context_tokens: int | None) -> TokenLayout:
    generated_indices = [
        index for index, turn in enumerate(turns) if turn.record.generated_token_count
    ]
    if not generated_indices:
        return TokenLayout((), None, (), (), (), None, domain_hash_bytes("payload", b""))
    last_generated = generated_indices[-1]
    trailing = turns[last_generated + 1 :]
    trailing_ref = None
    if trailing:
        if (
            len(trailing) != 1
            or trailing[0].record.generated_token_count != 0
            or trailing[0].record.termination["kind"] != "context_limit"
        ):
            raise GroupError("only a trailing zero-generation context limit may be dropped")
        trailing_ref = trailing[0].ref
    if any(not turns[index].record.generated_token_count for index in range(last_generated + 1)):
        raise GroupError("zero-generation writer turn precedes the last generated token")

    exported_turns = turns[: last_generated + 1]
    prompt_ids = exported_turns[0].input_ids
    completion_ids: list[int] = []
    env_mask: list[int] = []
    spans: list[Mapping[str, Any]] = []
    previous: _Turn | None = None
    for current in exported_turns:
        if previous is None:
            ext_start = 0
        else:
            prefix = previous.input_ids + previous.generated_ids
            if current.input_ids[: len(prefix)] != prefix:
                raise GroupError("training member has multiple context roots or a broken prefix")
            ext_start = len(completion_ids)
            external_suffix = current.input_ids[len(prefix) :]
            completion_ids.extend(external_suffix)
            env_mask.extend(0 for _ in external_suffix)
        completion_start = len(completion_ids)
        completion_ids.extend(current.generated_ids)
        env_mask.extend(1 for _ in current.generated_ids)
        spans.append(
            {
                "action_id": current.record.action_id,
                "turn_ref": current.ref,
                "completion_start": completion_start,
                "completion_end": len(completion_ids),
                "ext_start": ext_start,
            }
        )
        previous = current

    if not env_mask or not any(env_mask):
        return TokenLayout(
            prompt_ids,
            exported_turns[0].record.input_token_ids_ref,
            tuple(completion_ids),
            tuple(env_mask),
            tuple(spans),
            trailing_ref,
            domain_hash_bytes("payload", encode_training_token_ids((*prompt_ids, *completion_ids))),
        )
    if (
        max_context_tokens is not None
        and len(prompt_ids) + len(completion_ids) > max_context_tokens
    ):
        raise GroupError("training member sequence exceeds max_context_tokens")
    if (
        previous is None
        or prompt_ids + tuple(completion_ids) != previous.input_ids + previous.generated_ids
    ):
        raise GroupError("training member sequence does not end at the last generated token")
    return TokenLayout(
        prompt_ids=prompt_ids,
        prompt_ids_ref=exported_turns[0].record.input_token_ids_ref,
        completion_ids=tuple(completion_ids),
        env_mask=tuple(env_mask),
        turn_spans=tuple(spans),
        trailing_context_limit_turn_ref=trailing_ref,
        ledger_hash=domain_hash_bytes(
            "payload", encode_training_token_ids((*prompt_ids, *completion_ids))
        ),
    )


def _read_payload(reader: ArtifactReader, ref: str, label: str) -> Mapping[str, Any]:
    try:
        value = reader.artifact(ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise GroupError(f"{label} is unavailable") from exc
    if not isinstance(value, Mapping):
        raise GroupError(f"{label} is not a record")
    return value


def _read_token_ids(reader: ArtifactReader, ref: str, count: int) -> tuple[int, ...]:
    try:
        data = reader.bytes_artifact(ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise GroupError("training token IDs are unavailable") from exc
    try:
        return decode_u32_token_ids(data, count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GroupError("training token bytes differ from their committed count") from exc


def encode_training_token_ids(token_ids: tuple[int, ...]) -> bytes:
    try:
        return encode_u32_token_ids(token_ids)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GroupError("training token ID is outside the u32 range") from exc


__all__ = [
    "TokenLayout",
    "encode_training_token_ids",
    "max_context_tokens_for_group",
    "token_layout_for_checkpoint",
    "training_turn_spans",
]
