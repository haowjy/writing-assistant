"""Pure, hash-addressed export of finalized native task-graph groups."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from writing_agent.task_graph import (
    CheckpointV1,
    EventV1,
    domain_hash_bytes,
)
from writing_agent.task_graph_context_roots import context_root_changed_after
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_group_records import (
    GroupAdvantageV1,
    GroupDecisionV1,
    GroupError,
    GroupMemberResultV1,
    GroupSegmentCreditV1,
    read_fraction,
)
from writing_agent.task_graph_record_contracts import GroupSpecV1
from writing_agent.task_graph_records import (
    RECORD_TYPES,
    OutcomeV1,
    WriterTurnV2,
)
from writing_agent.task_graph_token_ledger import (
    decode_u32_token_ids,
    encode_u32_token_ids,
)
from writing_agent.task_graph_training_records import TrainingBatchV1
from writing_agent.task_graph_transition import ArtifactReader, DerivedArtifact


@dataclass(frozen=True)
class TrainingBatchExportV1:
    """A serializable batch plus the byte artifacts its content refs name."""

    batch: TrainingBatchV1
    artifacts: tuple[DerivedArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.batch, TrainingBatchV1):
            raise TypeError("training export requires TrainingBatchV1")
        if any(not isinstance(item, DerivedArtifact) for item in self.artifacts):
            raise TypeError("training export artifacts must be derived artifacts")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


@dataclass(frozen=True)
class _Turn:
    ref: str
    record: WriterTurnV2
    input_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]


@dataclass(frozen=True)
class _TokenLayout:
    prompt_ids: tuple[int, ...]
    prompt_ids_ref: str | None
    completion_ids: tuple[int, ...]
    env_mask: tuple[int, ...]
    turn_spans: tuple[Mapping[str, Any], ...]
    trailing_context_limit_turn_ref: str | None
    ledger_hash: str


def export_training_batch(
    spec: GroupSpecV1, decision: GroupDecisionV1, reader: ArtifactReader
) -> TrainingBatchExportV1:
    """Derive a trainer batch from a sealed native group and its finalized decision.

    The reader is the same read-only, hash-addressed boundary used by task-graph derives.
    Returned byte artifacts are pure outputs for the caller to persist before the batch.
    """
    if not isinstance(spec, GroupSpecV1) or not isinstance(decision, GroupDecisionV1):
        raise GroupError("training export requires typed group spec and decision records")
    if decision.status not in {"ready", "tie"}:
        raise GroupError("pending or invalid group cannot be exported")
    if spec.training_mode != "native" or decision.group_id != spec.group_id:
        raise GroupError("training export requires the matching native group")
    if (
        len(decision.member_result_refs) != len(spec.members)
        or len(decision.advantage_refs) != len(spec.members)
        or any(ref is None for ref in decision.member_result_refs)
    ):
        raise GroupError("settled group is missing a member result or advantage")

    max_context_tokens = _max_context_tokens(spec, reader)
    member_ids = frozenset(member.member_id for member in spec.members)
    segment_credits = tuple(
        _read_record(reader, ref, GroupSegmentCreditV1, "segment credit")
        for ref in decision.segment_credit_refs
    )
    if any(
        credit.group_id != spec.group_id or credit.member_id not in member_ids
        for credit in segment_credits
    ):
        raise GroupError("segment credit belongs to another group or member")

    members: list[dict[str, Any]] = []
    artifacts: list[DerivedArtifact] = []
    for ordinal, member_spec in enumerate(spec.members):
        result_ref = decision.member_result_refs[ordinal]
        advantage_ref = decision.advantage_refs[ordinal]
        result = _read_record(reader, result_ref, GroupMemberResultV1, "member result")
        advantage = _read_record(reader, advantage_ref, GroupAdvantageV1, "group advantage")
        if (
            result.group_id != spec.group_id
            or result.member_id != member_spec.member_id
            or result.execution_status != "valid"
            or result.fixture_ref is not None
            or result.final_checkpoint_id is None
            or result.terminal_outcome_ref is None
            or advantage.group_id != spec.group_id
            or advantage.member_id != member_spec.member_id
            or advantage.result_ref != result_ref
        ):
            raise GroupError("training member is not a real, settled group result")
        _require_structural_eligibility(result, reader)

        layout = _token_layout_for_checkpoint(
            result.final_checkpoint_id,
            result.member_id,
            reader,
            max_context_tokens=max_context_tokens,
        )
        if not layout.env_mask or not any(layout.env_mask):
            raise GroupError("training member has an empty generated-token mask")
        _verify_segment_credit_spans(
            segment_credits,
            member_spec.member_id,
            advantage_ref,
            layout,
        )

        advantage_value = _advantage_f64(advantage)
        if decision.status == "tie":
            if not advantage.zero_variance or advantage_value != 0.0:
                raise GroupError("tie group must export exact zero advantages")
        elif advantage.zero_variance:
            raise GroupError("ready group cannot export a zero-variance advantage")
        advantage_bytes = struct.pack("<d", advantage_value)
        completion_bytes = _encode_training_token_ids(layout.completion_ids)
        mask_bytes = bytes(layout.env_mask)
        for value in (advantage_bytes, completion_bytes, mask_bytes):
            artifacts.append(_bytes_artifact(value))

        member: dict[str, Any] = {
            "member_id": member_spec.member_id,
            "result_ref": result_ref,
            "advantage_ref": advantage_ref,
            "advantage": advantage,
            "advantage_f64_ref": domain_hash_bytes("payload", advantage_bytes),
            "prompt_ids_ref": layout.prompt_ids_ref,
            "completion_ids_ref": domain_hash_bytes("payload", completion_bytes),
            "env_mask_ref": domain_hash_bytes("payload", mask_bytes),
            "turn_spans": list(layout.turn_spans),
            "ledger_hash": layout.ledger_hash,
        }
        if layout.trailing_context_limit_turn_ref is not None:
            member["trailing_context_limit_turn_ref"] = layout.trailing_context_limit_turn_ref
        members.append(member)

    return TrainingBatchExportV1(
        batch=TrainingBatchV1(
            schema=1,
            group_id=spec.group_id,
            decision_ref=decision.identity(),
            max_context_tokens=max_context_tokens,
            members=members,
        ),
        artifacts=tuple(artifacts),
    )


def training_turn_spans(
    checkpoint_id: str,
    member_id: str,
    reader: ArtifactReader,
    *,
    max_context_tokens: int | None,
) -> Mapping[str, tuple[int, int]]:
    """Return exported generated-token offsets for eligible native group credits."""
    layout = _token_layout_for_checkpoint(
        checkpoint_id,
        member_id,
        reader,
        max_context_tokens=max_context_tokens,
    )
    return {
        span["turn_ref"]: (span["completion_start"], span["completion_end"])
        for span in layout.turn_spans
    }


def read_advantage_f64(member: Mapping[str, Any], reader: ArtifactReader) -> float:
    """Decode the once-computed binary64 value carried by an exported member."""
    try:
        data = reader.bytes_artifact(member["advantage_f64_ref"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GroupError("training advantage bytes are unavailable") from exc
    if not isinstance(data, bytes) or len(data) != 8:
        raise GroupError("training advantage must be one f64-le value")
    value = struct.unpack("<d", data)[0]
    if not math.isfinite(value):
        raise GroupError("training advantage is not finite")
    return value


def _require_structural_eligibility(result: GroupMemberResultV1, reader: ArtifactReader) -> None:
    try:
        checkpoint = reader.checkpoint(result.final_checkpoint_id)
        if not isinstance(checkpoint, CheckpointV1):
            checkpoint = CheckpointV1.from_dict(checkpoint)
    except (KeyError, TypeError, ValueError) as exc:
        raise GroupError("training member checkpoint is unavailable") from exc
    outcome = _read_record(reader, checkpoint.state.outcome_ref, OutcomeV1, "final outcome")
    if (
        outcome.execution_status != "valid"
        or outcome.reward_status != "available"
        or outcome.training_eligibility != "structurally_eligible"
        or outcome.eligibility_ref is None
        or outcome.reward_ref != result.availability_ref
    ):
        raise GroupError("training member is not structurally eligible")
    terminal = _read_record(reader, result.terminal_outcome_ref, OutcomeV1, "terminal outcome")
    if terminal.reward_status != "pending" or terminal.execution_status != "valid":
        raise GroupError("training member terminal outcome is invalid")
    body = _read_payload(reader, outcome.eligibility_ref, "training eligibility")
    eligibility_codec = RECORD_TYPES.get("TrainingEligibilityV1")
    if eligibility_codec is None:
        raise GroupError("training eligibility codec is unavailable")
    try:
        eligibility = eligibility_codec.from_dict(body)
    except (TypeError, ValueError) as exc:
        raise GroupError("training eligibility record is invalid") from exc
    if (
        eligibility.get("record_type") != "TrainingEligibilityV1"
        or eligibility.get("terminal_outcome_ref") != result.terminal_outcome_ref
        or eligibility.get("status") != "structurally_eligible"
    ):
        raise GroupError("training member is not structurally eligible")
    reward_codec = RECORD_TYPES.get("RewardV1")
    if reward_codec is None or result.availability_ref is None:
        raise GroupError("training member has no available reward")
    try:
        reward = reward_codec.from_dict(
            dict(_read_payload(reader, result.availability_ref, "member reward"))
        )
    except (TypeError, ValueError) as exc:
        raise GroupError("training member reward is invalid") from exc
    if (
        reward.get("terminal_outcome_ref") != result.terminal_outcome_ref
        or reward.get("eligibility_ref") != outcome.eligibility_ref
        or reward.get("availability") != "available"
    ):
        raise GroupError("training eligibility is not bound to the reward")


def _verify_segment_credit_spans(
    credits: tuple[GroupSegmentCreditV1, ...],
    member_id: str,
    advantage_ref: str,
    layout: _TokenLayout,
) -> None:
    expected = {
        span["turn_ref"]: (
            span["completion_start"],
            span["completion_end"],
            span["action_id"],
        )
        for span in layout.turn_spans
    }
    observed: set[str] = set()
    for credit in credits:
        if credit.member_id != member_id:
            continue
        if credit.advantage_ref != advantage_ref:
            raise GroupError("segment credit has a different group advantage")
        span = expected.get(credit.trace_ref)
        if span is None:
            if (
                credit.trace_ref != layout.trailing_context_limit_turn_ref
                or credit.completion_start is not None
                or credit.completion_end is not None
            ):
                raise GroupError("segment credit is not bound to an exported writer turn")
            continue
        if (credit.completion_start, credit.completion_end) != span[:2] or credit.action_id != span[
            2
        ]:
            raise GroupError("segment credit token span differs from the training export")
        observed.add(credit.trace_ref)
    if observed != set(expected):
        raise GroupError("training export is missing token-span segment credits")


def _max_context_tokens(spec: GroupSpecV1, reader: ArtifactReader) -> int | None:
    body = _read_payload(reader, spec.environment["budget_ref"], "entry budget")
    limits = body.get("limits") if isinstance(body, Mapping) else None
    if not isinstance(limits, Mapping):
        raise GroupError("native group entry budget has no limits")
    value = limits.get("context_tokens")
    if value is not None and (type(value) is not int or value < 0):
        raise GroupError("native group context-token cap is invalid")
    return value


def _token_layout_for_checkpoint(
    checkpoint_id: str,
    member_id: str,
    reader: ArtifactReader,
    *,
    max_context_tokens: int | None,
) -> _TokenLayout:
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


def _read_turn(reader: ArtifactReader, ref: str, turn: WriterTurnV2) -> _Turn:
    input_ids = _read_token_ids(reader, turn.input_token_ids_ref, turn.input_token_count)
    generated_ids = _read_token_ids(
        reader, turn.generated_token_ids_ref, turn.generated_token_count
    )
    return _Turn(ref, turn, input_ids, generated_ids)


def _layout_turns(turns: tuple[_Turn, ...], *, max_context_tokens: int | None) -> _TokenLayout:
    generated_indices = [
        index for index, turn in enumerate(turns) if turn.record.generated_token_count
    ]
    if not generated_indices:
        return _TokenLayout((), None, (), (), (), None, domain_hash_bytes("payload", b""))
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
        return _TokenLayout(
            prompt_ids,
            exported_turns[0].record.input_token_ids_ref,
            tuple(completion_ids),
            tuple(env_mask),
            tuple(spans),
            trailing_ref,
            domain_hash_bytes(
                "payload", _encode_training_token_ids((*prompt_ids, *completion_ids))
            ),
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
    return _TokenLayout(
        prompt_ids=prompt_ids,
        prompt_ids_ref=exported_turns[0].record.input_token_ids_ref,
        completion_ids=tuple(completion_ids),
        env_mask=tuple(env_mask),
        turn_spans=tuple(spans),
        trailing_context_limit_turn_ref=trailing_ref,
        ledger_hash=domain_hash_bytes(
            "payload", _encode_training_token_ids((*prompt_ids, *completion_ids))
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


def _read_record(reader: ArtifactReader, ref: str, record_type: type, label: str):
    body = _read_payload(reader, ref, label)
    try:
        return record_type.from_dict(dict(body))
    except (TypeError, ValueError) as exc:
        raise GroupError(f"{label} is invalid") from exc


def _read_token_ids(reader: ArtifactReader, ref: str, count: int) -> tuple[int, ...]:
    try:
        data = reader.bytes_artifact(ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise GroupError("training token IDs are unavailable") from exc
    try:
        return decode_u32_token_ids(data, count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GroupError("training token bytes differ from their committed count") from exc


def _encode_training_token_ids(token_ids: tuple[int, ...]) -> bytes:
    try:
        return encode_u32_token_ids(token_ids)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GroupError("training token ID is outside the u32 range") from exc


def _bytes_artifact(value: bytes) -> DerivedArtifact:
    return DerivedArtifact(
        ref=domain_hash_bytes("payload", value),
        value=value,
        kind="artifact",
        value_kind="bytes",
    )


def _advantage_f64(advantage: GroupAdvantageV1) -> float:
    if advantage.zero_variance:
        if read_fraction(advantage.advantage) != Fraction():
            raise GroupError("zero-variance advantage is not exact zero")
        return 0.0
    centered = read_fraction(advantage.centered)
    variance = read_fraction(advantage.variance)
    if variance <= 0:
        raise GroupError("nonzero group advantage has no positive variance")
    try:
        value = float(centered) / math.sqrt(float(variance))
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise GroupError("group advantage cannot be represented as float64") from exc
    if not math.isfinite(value):
        raise GroupError("group advantage cannot be represented as float64")
    return value


__all__ = [
    "TrainingBatchExportV1",
    "export_training_batch",
    "read_advantage_f64",
    "training_turn_spans",
]
