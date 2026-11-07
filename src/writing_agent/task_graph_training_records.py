"""Canonical group-level training export records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, ClassVar

from writing_agent.task_graph_group_records import GroupAdvantageV1, GroupError
from writing_agent.task_graph_wire import (
    Hash,
    Int,
    ListOf,
    RecordOf,
    Str,
    WireRecord,
    obj,
    obj_opt,
)

_MEMBER_ID = Str(nonempty=True, logical=True)
_TURN_SPAN = obj(
    action_id=Str(nonempty=True, logical=True),
    turn_ref=Hash("artifact"),
    completion_start=Int(),
    completion_end=Int(),
    ext_start=Int(),
)
_MEMBER = obj_opt(
    {
        "member_id": _MEMBER_ID,
        "result_ref": Hash("artifact"),
        "advantage_ref": Hash("artifact"),
        "advantage": RecordOf(GroupAdvantageV1),
        # Binary artifacts use fixed codecs: u32-le IDs, u8 masks, and f64-le rewards.
        "advantage_f64_ref": Hash("bytes"),
        "prompt_ids_ref": Hash("bytes"),
        "completion_ids_ref": Hash("bytes"),
        "env_mask_ref": Hash("bytes"),
        "turn_spans": ListOf(_TURN_SPAN, min_items=1),
        "ledger_hash": Hash(None),
    },
    {"trailing_context_limit_turn_ref": Hash("artifact", optional=True)},
)


@dataclass(frozen=True)
class TrainingBatchV1(WireRecord):
    """Token-level batch exported from one finalized native group.

    Token arrays are content-addressed bytes (u32-le IDs, u8 masks); the float64
    advantage is a content-addressed little-endian binary64 value. Canonical task-graph
    JSON deliberately excludes floats, so its exact binary representation is persisted
    separately and can be decoded without recomputing the group advantage.
    """

    schema: Annotated[int, Int(equals=1)]
    group_id: Annotated[str, Hash(None)]
    decision_ref: Annotated[str, Hash("artifact")]
    max_context_tokens: Annotated[int | None, Int(optional=True)] = None
    members: Annotated[
        tuple[Mapping[str, object], ...] | list[Mapping[str, object]],
        ListOf(_MEMBER, min_items=2, max_items=64),
    ] = ()
    RECORD_TYPE: ClassVar[str] = "TrainingBatchV1"
    OMIT_NONE_FIELDS: ClassVar[frozenset[str]] = frozenset({"max_context_tokens"})

    def check(self) -> None:
        member_ids = tuple(member["member_id"] for member in self.members)
        result_refs = tuple(member["result_ref"] for member in self.members)
        advantage_refs = tuple(member["advantage_ref"] for member in self.members)
        if (
            len(set(member_ids)) != len(member_ids)
            or len(set(result_refs)) != len(result_refs)
            or len(set(advantage_refs)) != len(advantage_refs)
        ):
            raise GroupError("training batch members must be unique")

        for member in self.members:
            advantage = member["advantage"]
            if (
                not isinstance(advantage, GroupAdvantageV1)
                or advantage.identity() != member["advantage_ref"]
                or advantage.group_id != self.group_id
                or advantage.member_id != member["member_id"]
                or advantage.result_ref != member["result_ref"]
            ):
                raise GroupError("training batch advantage is not bound to its member")

            spans = member["turn_spans"]
            previous_end = 0
            turn_refs: set[str] = set()
            action_ids: set[str] = set()
            for span in spans:
                if (
                    span["completion_start"] < span["ext_start"]
                    or span["completion_end"] <= span["completion_start"]
                    or span["ext_start"] < previous_end
                    or span["turn_ref"] in turn_refs
                    or span["action_id"] in action_ids
                ):
                    raise GroupError("training batch turn spans are inconsistent")
                previous_end = span["completion_end"]
                turn_refs.add(span["turn_ref"])
                action_ids.add(span["action_id"])
            if member.get("trailing_context_limit_turn_ref") in turn_refs:
                raise GroupError("audit-only turn cannot contribute to the training sequence")
