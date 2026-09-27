"""Group and context-policy wire records with their binding invariants."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

from writing_agent.task_graph import canonical_bytes, domain_hash
from writing_agent.task_graph_wire import (
    Enum,
    Hash,
    Int,
    ListOf,
    RecordOf,
    Str,
    WireRecord,
    obj,
)


class GroupError(ValueError):
    """A sealed group contract or immutable group record is invalid."""


class CompactionError(ValueError):
    """A context operation is not safe or its recorded evidence is false."""


POLICY_FIELDS = frozenset(
    {
        "model_ref",
        "behavior_policy_ref",
        "tokenizer_ref",
        "template_ref",
        "adapter_ref",
        "decoding_ref",
        "simulator_ref",
        "context_policy_ref",
        "controller_ref",
        "rng_derivation_version",
    }
)

SEMANTICS_V1 = "task-graph-derive-v1"


def _group_hash(value: Any) -> str:
    return domain_hash("payload", value)


def _group_seed(group_seed: int, role: str, ordinal: int | None = None) -> int:
    material = canonical_bytes(["GroupSeedV1", group_seed, role, ordinal])
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


_mode_spec = Enum(frozenset({"real", "fixture"}))
_operation_spec = Enum(frozenset({"carry", "seed", "drop", "compact"}))
_semantics_spec = Enum(frozenset({SEMANTICS_V1}))
_tool_spec = obj(max_file_bytes=Int(minimum=1), max_workspace_bytes=Int(minimum=1))
_environment_spec = obj(
    **{
        name: Hash(None)
        for name in "entry_state_hash entry_tree_hash instance_hash graph_hash node_contract_hash "
        "controller_contract_hash check_contracts_hash source_refs_hash request_refs_hash "
        "visible_prefix_hash context_messages_hash rendering_hash tool_schemas_hash budget_hash "
        "versions_hash continuation_hash".split()
    },
    **{
        name: Hash("artifact")
        for name in "budget_ref versions_ref decisions_ref disclosures_ref external_inputs_ref "
        "rng_ref outcome_ref provenance_ref".split()
    },
    entry_checkpoint_id=Hash("checkpoint"),
    node_id=Str(nonempty=True, logical=True),
    node_visit_id=Str(nonempty=True, logical=True),
    reward_contract_hash=Hash(None, optional=True),
    simulator_contract_hash=Hash(None, optional=True),
    context_revision_ref=Hash("context_revision"),
    author_packet_ref=Hash("private", optional=True),
    requirements_ref=Hash("private"),
    horizon=Str(nonempty=True),
)
_policy_spec = obj(
    **{name: Hash("artifact") for name in POLICY_FIELDS - {"rng_derivation_version"}},
    rng_derivation_version=Str(nonempty=True),
)


@dataclass(frozen=True)
class GroupMemberSpecV1(WireRecord):
    member_id: Annotated[str, Str(nonempty=True, logical=True)]
    ordinal: Annotated[int, Int()]
    writer_seed: Annotated[int, Int()]
    environment_seed: Annotated[int, Int()]
    seed_provenance: Annotated[str, Enum(frozenset({"sha256-domain-v1"}))] = "sha256-domain-v1"
    schema: Annotated[int, Int(equals=1)] = 1


@dataclass(frozen=True)
class GroupSpecV1(WireRecord):
    group_id: Annotated[str, Hash(None)]
    group_sequence: Annotated[int, Int(minimum=0)]
    group_seed: Annotated[int, Int(minimum=0)]
    runner_mode: Annotated[str, _mode_spec]
    environment: Annotated[Mapping[str, Any], _environment_spec]
    policy: Annotated[Mapping[str, str], _policy_spec]
    members: Annotated[
        tuple[GroupMemberSpecV1 | Mapping[str, Any], ...],
        ListOf(RecordOf(GroupMemberSpecV1), min_items=2, max_items=64),
    ]
    schema: Annotated[int, Int(equals=1)] = 1
    RECORD_TYPE: ClassVar[str] = "GroupSpecV1"

    @property
    def record_type(self) -> str:
        return self.RECORD_TYPE

    def check(self) -> None:
        expected = _group_hash(
            [
                "GroupIdV1",
                self.group_sequence,
                self.environment,
                self.policy,
                self.group_seed,
                self.runner_mode,
                len(self.members),
            ],
        )
        if self.group_id != expected:
            raise GroupError("group ID does not bind its contract and sequence")
        if tuple(member.ordinal for member in self.members) != tuple(range(len(self.members))):
            raise GroupError("member slots are not canonical")
        for member in self.members:
            if member.member_id != f"grp-{self.group_id[:24]}-{member.ordinal:02d}":
                raise GroupError("member ID does not bind its ordinal")
            if member.writer_seed != _group_seed(self.group_seed, "writer", member.ordinal):
                raise GroupError("writer seed derivation mismatch")
            if member.environment_seed != _group_seed(self.group_seed, "environment"):
                raise GroupError("environment seed derivation mismatch")
        if len({member.writer_seed for member in self.members}) != len(self.members):
            raise GroupError("writer streams are not distinct")


@dataclass(frozen=True)
class ContextPolicyV1(WireRecord):
    operation: Annotated[str, _operation_spec]
    retained_exchanges: Annotated[int, Int()] = 0
    seed_name: Annotated[
        str | None, Str(nonempty=True, optional=True, no_whitespace_or_controls=True)
    ] = None
    seed_checkpoint_ref: Annotated[str | None, Hash("checkpoint", optional=True)] = None
    summarizer_version: Annotated[str | None, Str(optional=True)] = None
    max_summary_chars: Annotated[int | None, Int(optional=True)] = None
    max_operations: Annotated[int, Int()] = 32
    max_context_bytes: Annotated[int, Int()] = 1_000_000
    max_context_storage_bytes: Annotated[int, Int()] = 8_000_000
    schema: Annotated[int, Int(equals=1)] = 1
    RECORD_TYPE: ClassVar[str] = "ContextPolicyV1"

    def check(self) -> None:
        if self.operation == "compact":
            if (
                self.summarizer_version != "visible-text-v1"
                or self.max_summary_chars is None
                or self.seed_name is not None
                or self.seed_checkpoint_ref is not None
            ):
                raise CompactionError("compact requires the fixed visible-text-v1 algorithm")
        elif self.summarizer_version is not None or self.max_summary_chars is not None:
            raise CompactionError("only compact may configure the summarizer")
        if self.operation == "seed":
            if self.seed_name is None or self.seed_checkpoint_ref is None:
                raise CompactionError("seed requires a named ancestor checkpoint")
        elif self.operation != "compact" and (
            self.seed_name is not None or self.seed_checkpoint_ref is not None
        ):
            raise CompactionError("only seed may name a prefix")
        if self.operation != "compact" and self.retained_exchanges:
            raise CompactionError("only compact may retain a tail")


@dataclass(frozen=True)
class ExecutionVersionsV1(WireRecord):
    schema: Annotated[int, Int(equals=1)]
    transition_semantics: Annotated[str, _semantics_spec]
    admission_policy_ref: Annotated[str, Hash("artifact")]
    tool_spec: Annotated[Mapping[str, Any], _tool_spec]
