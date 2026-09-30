"""Pure, hash-addressed export of finalized native task-graph groups."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

from writing_agent.task_graph import CheckpointV1, domain_hash_bytes
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
)
from writing_agent.task_graph_training_layout import (
    TokenLayout,
    encode_training_token_ids,
    max_context_tokens_for_group,
    token_layout_for_checkpoint,
    training_turn_spans,
)
from writing_agent.task_graph_training_records import TrainingBatchV1
from writing_agent.task_graph_transition import ArtifactReader, DerivedArtifact


class TrainingExportError(GroupError):
    """A fail-closed export refusal with a stable machine-readable reason."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(message)


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


def export_training_batch(
    spec: GroupSpecV1, decision: GroupDecisionV1, reader: ArtifactReader
) -> TrainingBatchExportV1:
    """Derive a trainer batch from a sealed native group and its finalized decision.

    The reader is the same read-only, hash-addressed boundary used by task-graph derives.
    Returned byte artifacts are pure outputs for the caller to persist before the batch.
    """
    if not isinstance(spec, GroupSpecV1) or not isinstance(decision, GroupDecisionV1):
        raise TrainingExportError("invalid_export_input", "training export requires typed records")
    if decision.status not in {"ready", "tie"}:
        raise TrainingExportError(
            "group_not_exportable", "pending or invalid group cannot be exported"
        )
    if spec.training_mode != "native" or decision.group_id != spec.group_id:
        raise TrainingExportError(
            "wrong_group", "training export requires the matching native group"
        )
    if (
        len(decision.member_result_refs) != len(spec.members)
        or len(decision.advantage_refs) != len(spec.members)
        or any(ref is None for ref in decision.member_result_refs)
    ):
        raise TrainingExportError(
            "incomplete_group", "settled group is missing a result or advantage"
        )

    try:
        max_context_tokens = max_context_tokens_for_group(spec, reader)
    except GroupError as exc:
        raise TrainingExportError(
            "invalid_entry_budget", "native group context cap is invalid"
        ) from exc
    member_ids = frozenset(member.member_id for member in spec.members)
    segment_credits = tuple(
        _read_record(reader, ref, GroupSegmentCreditV1, "segment credit")
        for ref in decision.segment_credit_refs
    )
    if any(
        credit.group_id != spec.group_id or credit.member_id not in member_ids
        for credit in segment_credits
    ):
        raise TrainingExportError(
            "invalid_segment_credit", "segment credit belongs to another group"
        )

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
            raise TrainingExportError(
                "member_not_exportable", "member is not a real settled result"
            )
        _require_structural_eligibility(result, reader)

        try:
            layout = token_layout_for_checkpoint(
                result.final_checkpoint_id,
                result.member_id,
                reader,
                max_context_tokens=max_context_tokens,
            )
        except GroupError as exc:
            raise TrainingExportError(
                "invalid_member_layout", "member token layout is invalid"
            ) from exc
        if not layout.env_mask or not any(layout.env_mask):
            raise TrainingExportError("empty_generated_mask", "member has no generated-token mask")
        _verify_segment_credit_spans(
            segment_credits,
            member_spec.member_id,
            advantage_ref,
            layout,
        )

        advantage_value = _advantage_f64(advantage)
        if decision.status == "tie":
            if not advantage.zero_variance or advantage_value != 0.0:
                raise TrainingExportError(
                    "advantage_status_mismatch", "tie requires exact zero advantages"
                )
        elif advantage.zero_variance:
            raise TrainingExportError(
                "advantage_status_mismatch", "ready group cannot have zero variance"
            )
        advantage_bytes = struct.pack("<d", advantage_value)
        completion_bytes = encode_training_token_ids(layout.completion_ids)
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

    try:
        batch = TrainingBatchV1(
            schema=1,
            group_id=spec.group_id,
            decision_ref=decision.identity(),
            max_context_tokens=max_context_tokens,
            members=members,
        )
    except (TypeError, ValueError) as exc:
        raise TrainingExportError(
            "invalid_training_batch", "derived training batch violates its record contract"
        ) from exc
    return TrainingBatchExportV1(batch=batch, artifacts=tuple(artifacts))


def read_advantage_f64(member: Mapping[str, Any], reader: ArtifactReader) -> float:
    """Decode the once-computed binary64 value carried by an exported member."""
    try:
        data = reader.bytes_artifact(member["advantage_f64_ref"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingExportError(
            "advantage_bytes_unavailable", "training advantage bytes are unavailable"
        ) from exc
    if not isinstance(data, bytes) or len(data) != 8:
        raise TrainingExportError(
            "invalid_advantage_bytes", "training advantage must be one f64-le value"
        )
    value = struct.unpack("<d", data)[0]
    if not math.isfinite(value):
        raise TrainingExportError("invalid_advantage_bytes", "training advantage is not finite")
    return value


def _require_structural_eligibility(result: GroupMemberResultV1, reader: ArtifactReader) -> None:
    try:
        checkpoint = reader.checkpoint(result.final_checkpoint_id)
        if not isinstance(checkpoint, CheckpointV1):
            checkpoint = CheckpointV1.from_dict(checkpoint)
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingExportError(
            "member_checkpoint_unavailable", "training member checkpoint is unavailable"
        ) from exc
    outcome = _read_record(reader, checkpoint.state.outcome_ref, OutcomeV1, "final outcome")
    if (
        outcome.execution_status != "valid"
        or outcome.reward_status != "available"
        or outcome.training_eligibility != "structurally_eligible"
        or outcome.eligibility_ref is None
        or outcome.reward_ref != result.availability_ref
    ):
        raise TrainingExportError(
            "member_not_eligible", "training member is not structurally eligible"
        )
    terminal = _read_record(reader, result.terminal_outcome_ref, OutcomeV1, "terminal outcome")
    if terminal.reward_status != "pending" or terminal.execution_status != "valid":
        raise TrainingExportError(
            "invalid_terminal_outcome", "training member terminal outcome is invalid"
        )
    body = _read_payload(reader, outcome.eligibility_ref, "training eligibility")
    eligibility_codec = RECORD_TYPES.get("TrainingEligibilityV1")
    if eligibility_codec is None:
        raise TrainingExportError(
            "eligibility_codec_unavailable", "training eligibility codec is unavailable"
        )
    try:
        eligibility = eligibility_codec.from_dict(body)
    except (TypeError, ValueError) as exc:
        raise TrainingExportError(
            "invalid_eligibility", "training eligibility record is invalid"
        ) from exc
    if (
        eligibility.get("record_type") != "TrainingEligibilityV1"
        or eligibility.get("terminal_outcome_ref") != result.terminal_outcome_ref
        or eligibility.get("status") != "structurally_eligible"
    ):
        raise TrainingExportError(
            "member_not_eligible", "training member is not structurally eligible"
        )
    reward_codec = RECORD_TYPES.get("RewardV1")
    if reward_codec is None or result.availability_ref is None:
        raise TrainingExportError("reward_unavailable", "training member has no available reward")
    try:
        reward = reward_codec.from_dict(
            dict(_read_payload(reader, result.availability_ref, "member reward"))
        )
    except (TypeError, ValueError) as exc:
        raise TrainingExportError("invalid_reward", "training member reward is invalid") from exc
    if (
        reward.get("terminal_outcome_ref") != result.terminal_outcome_ref
        or reward.get("eligibility_ref") != outcome.eligibility_ref
        or reward.get("availability") != "available"
    ):
        raise TrainingExportError(
            "eligibility_reward_mismatch", "training eligibility is not bound to the reward"
        )


def _verify_segment_credit_spans(
    credits: tuple[GroupSegmentCreditV1, ...],
    member_id: str,
    advantage_ref: str,
    layout: TokenLayout,
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
            raise TrainingExportError(
                "segment_credit_mismatch", "segment credit has a different advantage"
            )
        span = expected.get(credit.trace_ref)
        if span is None:
            if (
                credit.trace_ref != layout.trailing_context_limit_turn_ref
                or credit.completion_start is not None
                or credit.completion_end is not None
            ):
                raise TrainingExportError(
                    "segment_credit_mismatch", "segment credit is not bound to an exported turn"
                )
            continue
        if (credit.completion_start, credit.completion_end) != span[:2] or credit.action_id != span[
            2
        ]:
            raise TrainingExportError(
                "segment_credit_mismatch", "segment credit token span differs"
            )
        observed.add(credit.trace_ref)
    if observed != set(expected):
        raise TrainingExportError(
            "missing_segment_credit", "training export is missing token-span segment credits"
        )


def _read_payload(reader: ArtifactReader, ref: str, label: str) -> Mapping[str, Any]:
    try:
        value = reader.artifact(ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingExportError("artifact_unavailable", f"{label} is unavailable") from exc
    if not isinstance(value, Mapping):
        raise TrainingExportError("invalid_artifact_record", f"{label} is not a record")
    return value


def _read_record(reader: ArtifactReader, ref: str, record_type: type, label: str):
    body = _read_payload(reader, ref, label)
    try:
        return record_type.from_dict(dict(body))
    except (TypeError, ValueError) as exc:
        raise TrainingExportError("invalid_artifact_record", f"{label} is invalid") from exc


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
            raise TrainingExportError(
                "invalid_advantage", "zero-variance advantage is not exact zero"
            )
        return 0.0
    centered = read_fraction(advantage.centered)
    variance = read_fraction(advantage.variance)
    if variance <= 0:
        raise TrainingExportError("invalid_advantage", "nonzero advantage needs positive variance")
    try:
        with localcontext() as context:
            context.prec = 200
            centered_decimal = Decimal(centered.numerator) / Decimal(centered.denominator)
            variance_decimal = Decimal(variance.numerator) / Decimal(variance.denominator)
            value = float(centered_decimal / variance_decimal.sqrt())
    except (ArithmeticError, ValueError) as exc:
        raise TrainingExportError(
            "invalid_advantage", "group advantage cannot be represented as float64"
        ) from exc
    if not math.isfinite(value):
        raise TrainingExportError(
            "invalid_advantage", "group advantage cannot be represented as float64"
        )
    return value


__all__ = [
    "TrainingExportError",
    "TrainingBatchExportV1",
    "export_training_batch",
    "read_advantage_f64",
    "training_turn_spans",
]
