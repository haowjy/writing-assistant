"""Pure, ordered structural-training eligibility decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from writing_agent.task_graph_context_roots import context_root_changed_after
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_native_contracts import manifest_sampling_capabilities
from writing_agent.task_graph_records import (
    NATIVE_RUNTIME_CAPABILITIES,
    RuntimeManifestV1,
    RuntimeManifestV2,
    WriterTurnV1,
    WriterTurnV2,
    decode_runtime_manifest,
)
from writing_agent.task_graph_transition import ArtifactReader, LineageView


@dataclass(frozen=True)
class EligibilityDecisionV1:
    """One reason-code decision persisted with the terminal reward."""

    status: str
    reason: str

    def __post_init__(self) -> None:
        ineligible_reasons = {
            "native_action_trace_unavailable",
            "manifest_capability_missing",
            "group_training_mode_absent",
            "multi_segment_context",
            "reasoning_content_present",
            "no_sampled_actions",
            "execution_not_valid",
        }
        if self.status == "ineligible" and self.reason in ineligible_reasons:
            return
        if self.status == "structurally_eligible" and self.reason == "native_evidence_structural":
            return
        raise ValueError("unsupported structural eligibility decision")

    def training_wire(self, outcome_ref: str) -> dict[str, Any]:
        return {
            "record_type": "TrainingEligibilityV1",
            "schema": 1,
            "terminal_outcome_ref": outcome_ref,
            "status": self.status,
            "reason": self.reason,
        }


def _turns(view: LineageView, reader: ArtifactReader) -> tuple[WriterTurnV1 | WriterTurnV2, ...]:
    turns: list[WriterTurnV1 | WriterTurnV2] = []
    for sample in view.samples:
        value = reader.artifact(sample.turn_ref)
        if not isinstance(value, dict):
            raise ProjectionError("samples.turn_ref: writer turn is not a record")
        record_type = value.get("record_type")
        if record_type == WriterTurnV1.RECORD_TYPE:
            turns.append(WriterTurnV1.from_dict(value))
        elif record_type == WriterTurnV2.RECORD_TYPE:
            turns.append(WriterTurnV2.from_dict(value))
        else:
            raise ProjectionError("samples.turn_ref: unsupported writer turn record")
    return tuple(turns)


def _manifest(
    view: LineageView,
    turns: tuple[WriterTurnV1 | WriterTurnV2, ...],
    reader: ArtifactReader,
) -> RuntimeManifestV1 | RuntimeManifestV2 | None:
    manifest_ref = None
    if view.group is not None:
        manifest_ref = view.group.policy["adapter_ref"]
    else:
        native_turn = next((turn for turn in turns if isinstance(turn, WriterTurnV2)), None)
        if native_turn is not None:
            manifest_ref = native_turn.sampling_pins["manifest_ref"]
    if manifest_ref is None:
        return None
    return decode_runtime_manifest(reader.artifact(manifest_ref))


def _member_of_native_group(view: LineageView) -> bool:
    group = view.group
    if group is None or group.training_mode != "native":
        return False
    lineage_id = view.state.position["lineage_id"]
    return any(member.member_id == lineage_id for member in group.members)


def decide_eligibility(view: LineageView, reader: ArtifactReader) -> EligibilityDecisionV1:
    """Decide structural eligibility from a verified view and its committed evidence.

    The reader is the derive's read-only artifact boundary: every turn, manifest,
    and context-operation policy is already hash-bound by the gate.
    """
    turns = _turns(view, reader)
    manifest = _manifest(view, turns, reader)

    if (
        any(isinstance(turn, WriterTurnV1) for turn in turns)
        or isinstance(manifest, RuntimeManifestV1)
        or manifest is None
    ):
        return EligibilityDecisionV1("ineligible", "native_action_trace_unavailable")

    required = NATIVE_RUNTIME_CAPABILITIES
    if isinstance(manifest, RuntimeManifestV2) and required - manifest_sampling_capabilities(
        manifest
    ):
        return EligibilityDecisionV1("ineligible", "manifest_capability_missing")

    if not _member_of_native_group(view):
        return EligibilityDecisionV1("ineligible", "group_training_mode_absent")

    if context_root_changed_after(reader, view.head_event_id, None):
        return EligibilityDecisionV1("ineligible", "multi_segment_context")

    if any(
        turn.message.reasoning is not None
        or turn.message.thinking is not None
        or turn.message.reasoning_content is not None
        for turn in turns
        if isinstance(turn, WriterTurnV2)
    ):
        return EligibilityDecisionV1("ineligible", "reasoning_content_present")

    if not any(isinstance(turn, WriterTurnV2) and turn.generated_token_count for turn in turns):
        return EligibilityDecisionV1("ineligible", "no_sampled_actions")

    if view.outcome.execution_status != "valid":
        return EligibilityDecisionV1("ineligible", "execution_not_valid")

    return EligibilityDecisionV1("structurally_eligible", "native_evidence_structural")


__all__ = ["EligibilityDecisionV1", "decide_eligibility"]
