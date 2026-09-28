"""Pure tagged payload records for group coordination."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar

from writing_agent.task_graph_record_contracts import GroupError
from writing_agent.task_graph_wire import (
    Bool,
    Enum,
    Hash,
    Int,
    ListOf,
    Str,
    UnionOf,
    WireRecord,
    obj,
)


def fraction_wire(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def read_fraction(value: Mapping[str, int]) -> Fraction:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"numerator", "denominator"}
        or type(value["numerator"]) is not int
        or type(value["denominator"]) is not int
        or value["denominator"] <= 0
    ):
        raise GroupError("invalid exact fraction")
    fraction = Fraction(value["numerator"], value["denominator"])
    if fraction_wire(fraction) != dict(value):
        raise GroupError("fraction is not canonical and reduced")
    return fraction


_FRACTION = obj(numerator=Int(minimum=None), denominator=Int(minimum=1))
_MEMBER_ID = Str(nonempty=True, logical=True)
_OPTIONAL_ARTIFACT = Hash("artifact", optional=True)


@dataclass(frozen=True)
class GroupScriptedTerminalV1(WireRecord):
    schema: Annotated[int, Int(equals=1)]
    group_id: Annotated[str, Hash(None)]
    member_id: Annotated[str, _MEMBER_ID]
    start_checkpoint_id: Annotated[str, Hash("checkpoint")]
    execution_status: Annotated[str, Enum(frozenset({"valid", "infrastructure_invalid"}))]
    reward_status: Annotated[str, Enum(frozenset({"available", "pending", "unavailable"}))]
    reward: Annotated[
        Mapping[str, int] | None,
        UnionOf((_FRACTION, type(None))),
    ]
    native_optimizer_eligible: Annotated[bool, Bool()]
    RECORD_TYPE: ClassVar[str] = "GroupScriptedTerminalV1"

    def check(self) -> None:
        if self.native_optimizer_eligible:
            raise GroupError("scripted result cannot be native optimizer eligible")
        if self.reward_status == "available":
            if self.execution_status != "valid" or self.reward is None:
                raise GroupError("available scripted reward needs a valid exact reward")
            read_fraction(self.reward)
        elif self.reward is not None:
            raise GroupError("unavailable scripted reward includes a number")
        if (
            self.execution_status == "infrastructure_invalid"
            and self.reward_status != "unavailable"
        ):
            raise GroupError("infrastructure failure cannot carry a reward")


@dataclass(frozen=True)
class GroupExecutionFailureV1(WireRecord):
    schema: Annotated[int, Int(equals=1)]
    group_id: Annotated[str, Hash(None)]
    member_id: Annotated[str, _MEMBER_ID]
    start_checkpoint_id: Annotated[str, Hash("checkpoint")]
    reason: Annotated[str, Str(nonempty=True)]
    evidence_ref: Annotated[str | None, Hash("artifact", optional=True)]
    RECORD_TYPE: ClassVar[str] = "GroupExecutionFailureV1"


@dataclass(frozen=True)
class GroupMemberResultV1(WireRecord):
    schema: Annotated[int, Int(equals=1)] = 1
    group_id: Annotated[str, Hash(None)] = ""
    member_id: Annotated[str, _MEMBER_ID] = ""
    start_checkpoint_id: Annotated[str, Hash("checkpoint")] = ""
    final_checkpoint_id: Annotated[str | None, Hash("checkpoint", optional=True)] = None
    terminal_outcome_ref: Annotated[str | None, _OPTIONAL_ARTIFACT] = None
    availability_ref: Annotated[str | None, _OPTIONAL_ARTIFACT] = None
    fixture_ref: Annotated[str | None, _OPTIONAL_ARTIFACT] = None
    failure_ref: Annotated[str | None, _OPTIONAL_ARTIFACT] = None
    execution_status: Annotated[
        str, Enum(frozenset({"pending", "valid", "infrastructure_invalid"}))
    ] = "pending"
    RECORD_TYPE: ClassVar[str] = "GroupMemberResultV1"

    def check(self) -> None:
        evidence = (
            self.final_checkpoint_id,
            self.terminal_outcome_ref,
            self.availability_ref,
            self.fixture_ref,
            self.failure_ref,
        )
        if self.fixture_ref and any(
            (
                self.final_checkpoint_id,
                self.terminal_outcome_ref,
                self.availability_ref,
                self.failure_ref,
            )
        ):
            raise GroupError("fixture and real terminal evidence cannot be mixed")
        if self.execution_status == "pending" and any(evidence):
            raise GroupError("pending result cannot claim terminal or failure evidence")
        if self.execution_status == "infrastructure_invalid" and not (
            self.fixture_ref or self.failure_ref
        ):
            raise GroupError("infrastructure failure needs immutable cause evidence")
        if self.execution_status == "infrastructure_invalid" and any(
            (self.final_checkpoint_id, self.terminal_outcome_ref, self.availability_ref)
        ):
            raise GroupError("infrastructure failure cannot claim a terminal writer result")
        if self.execution_status == "valid" and not (
            self.fixture_ref or (self.final_checkpoint_id and self.terminal_outcome_ref)
        ):
            raise GroupError("valid member needs immutable terminal evidence")
        if self.execution_status == "valid" and self.failure_ref:
            raise GroupError("valid member cannot carry infrastructure failure evidence")


@dataclass(frozen=True)
class GroupDecisionV1(WireRecord):
    schema: Annotated[int, Int(equals=1)] = 1
    group_id: Annotated[str, Hash(None)] = ""
    status: Annotated[str, Enum(frozenset({"pending", "invalid", "ready", "tie"}))] = "pending"
    reason: Annotated[str, Str()] = ""
    member_result_refs: Annotated[
        tuple[str | None, ...] | list[str | None], ListOf(UnionOf((Hash("artifact"), type(None))))
    ] = ()
    advantage_refs: Annotated[tuple[str, ...] | list[str], ListOf(Hash("artifact"))] = ()
    segment_credit_refs: Annotated[tuple[str, ...] | list[str], ListOf(Hash("artifact"))] = ()
    native_optimizer_eligible: Annotated[bool, Bool()] = False
    RECORD_TYPE: ClassVar[str] = "GroupDecisionV1"

    def check(self) -> None:
        if self.native_optimizer_eligible:
            raise GroupError("native optimization is not implemented")
        if self.status in {"pending", "invalid"} and (
            self.advantage_refs or self.segment_credit_refs
        ):
            raise GroupError("unsettled or invalid group cannot have credit")
        if self.status in {"ready", "tie"} and len(self.advantage_refs) != len(
            self.member_result_refs
        ):
            raise GroupError("settled group needs one advantage per member")


@dataclass(frozen=True)
class GroupAdvantageV1(WireRecord):
    schema: Annotated[int, Int(equals=1)] = 1
    group_id: Annotated[str, Hash(None)] = ""
    member_id: Annotated[str, _MEMBER_ID] = ""
    result_ref: Annotated[str, Hash("artifact")] = ""
    reward: Annotated[Mapping[str, int], _FRACTION] = None  # type: ignore[assignment]
    mean: Annotated[Mapping[str, int], _FRACTION] = None  # type: ignore[assignment]
    variance: Annotated[Mapping[str, int], _FRACTION] = None  # type: ignore[assignment]
    centered: Annotated[Mapping[str, int], _FRACTION] = None  # type: ignore[assignment]
    expression: Annotated[
        str, Enum(frozenset({"centered / sqrt(population_variance)", "zero"}))
    ] = "centered / sqrt(population_variance)"
    advantage: Annotated[Mapping[str, int] | None, UnionOf((_FRACTION, type(None)))] = None
    zero_variance: Annotated[bool, Bool()] = False
    native_optimizer_eligible: Annotated[bool, Bool()] = False
    RECORD_TYPE: ClassVar[str] = "GroupAdvantageV1"

    def check(self) -> None:
        reward, mean = read_fraction(self.reward), read_fraction(self.mean)
        variance, centered = read_fraction(self.variance), read_fraction(self.centered)
        if (
            variance < 0
            or centered != reward - mean
            or self.zero_variance is not (variance == 0)
            or self.native_optimizer_eligible
            or (
                self.zero_variance
                and (
                    centered != 0 or self.expression != "zero" or read_fraction(self.advantage) != 0
                )
            )
            or (
                not self.zero_variance
                and (
                    self.expression != "centered / sqrt(population_variance)"
                    or self.advantage is not None
                )
            )
        ):
            raise GroupError("advantage expression or exact moments are inconsistent")


@dataclass(frozen=True)
class GroupSegmentCreditV1(WireRecord):
    schema: Annotated[int, Int(equals=1)] = 1
    group_id: Annotated[str, Hash(None)] = ""
    member_id: Annotated[str, _MEMBER_ID] = ""
    action_id: Annotated[str, Str(nonempty=True, logical=True)] = ""
    action_ref: Annotated[str, Hash("event")] = ""
    message_ref: Annotated[str, Hash("artifact")] = ""
    trace_ref: Annotated[str, Hash("artifact")] = ""
    original_context_ref: Annotated[str, Hash("context_revision")] = ""
    original_context_content_hash: Annotated[str, Hash(None)] = ""
    advantage_ref: Annotated[str, Hash("artifact")] = ""
    segment_kind: Annotated[
        str, Enum(frozenset({"assistant_text", "tool_syntax", "assistant_ending"}))
    ] = ""
    part_index: Annotated[int | None, Int(optional=True)] = None
    segment_content_hash: Annotated[str | None, Hash(None, optional=True)] = None
    excluded_roles: Annotated[tuple[str, ...] | list[str], ListOf(Str())] = (
        "system",
        "user",
        "author",
        "tool",
        "seed",
        "environment",
        "summary",
    )
    native_optimizer_eligible: Annotated[bool, Bool()] = False
    token_mask_ref: Annotated[None, UnionOf((type(None),))] = None
    logprob_ref: Annotated[None, UnionOf((type(None),))] = None
    RECORD_TYPE: ClassVar[str] = "GroupSegmentCreditV1"

    def check(self) -> None:
        if (
            (
                self.segment_kind == "assistant_ending"
                and (self.part_index is not None or self.segment_content_hash is not None)
            )
            or (
                self.segment_kind != "assistant_ending"
                and (
                    type(self.part_index) is not int
                    or self.part_index < 0
                    or self.segment_content_hash is None
                )
            )
            or self.excluded_roles
            != (
                "system",
                "user",
                "author",
                "tool",
                "seed",
                "environment",
                "summary",
            )
            or (
                self.native_optimizer_eligible
                or self.token_mask_ref is not None
                or self.logprob_ref is not None
            )
        ):
            raise GroupError("segment credit cannot assert native or non-writer eligibility")
