"""Sealed group inputs and exact, non-native advantage/segment-credit records.

Records are immutable payload-domain values. Worker paths and operational clocks
never enter a group identity; member ordinals, not completion order, fix slot IDs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from writing_agent.task_graph import (
    _Record,
    canonical_bytes,
    domain_hash,
    validate_hash,
)
from writing_agent.task_graph_admission import StoreArtifactResolver, admit_graph
from writing_agent.task_graph_compaction import require_quiescent
from writing_agent.task_graph_projection import project_writer_context
from writing_agent.task_graph_store import TaskGraphStore


class GroupError(ValueError):
    """A group contract, immutable receipt, or member result is invalid."""


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


def _hash(value: Any) -> str:
    return domain_hash("payload", value)


def _seed(group_seed: int, role: str, ordinal: int | None = None) -> int:
    if type(group_seed) is not int or group_seed < 0:
        raise GroupError("group seed must be a nonnegative integer")
    material = canonical_bytes(["GroupSeedV1", group_seed, role, ordinal])
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _read_fraction(value: Mapping[str, int]) -> Fraction:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"numerator", "denominator"}
        or type(value["numerator"]) is not int
        or type(value["denominator"]) is not int
        or value["denominator"] <= 0
    ):
        raise GroupError("invalid exact fraction")
    fraction = Fraction(value["numerator"], value["denominator"])
    if _fraction(fraction) != dict(value):
        raise GroupError("fraction is not canonical and reduced")
    return fraction


def _required_policy(policy: dict[str, str], rendering: dict[str, str]) -> dict[str, str]:
    if set(policy) != POLICY_FIELDS or any(
        not isinstance(v, str) or not v for v in policy.values()
    ):
        raise GroupError(f"policy requires exactly {sorted(POLICY_FIELDS)}")
    if policy["rng_derivation_version"] != "sha256-domain-v1":
        raise GroupError("unsupported seed derivation")
    for field in ("tokenizer_ref", "template_ref"):
        if policy[field] != rendering[field]:
            raise GroupError(f"{field} differs from the actual writer rendering")
    for field in POLICY_FIELDS - {"rng_derivation_version"}:
        validate_hash(policy[field])
    return dict(policy)


def _environment(store: TaskGraphStore, checkpoint_id: str) -> dict[str, Any]:
    """Resolve every equality-critical entry input before admission, not only messages."""
    checkpoint = store.load_checkpoint(checkpoint_id)
    state = checkpoint.state
    if (
        state.position["phase"] != "ready_writer"
        or state.history["action_ids"]
        or state.history["tool_result_ids"]
    ):
        raise GroupError("entry must be an unsampled writer checkpoint")
    require_quiescent(state)
    instance = store.load_instance(state.instance_ref)
    graph = admit_graph(instance, StoreArtifactResolver(store))
    node = graph.node(state.position["node_id"])
    if node.spec.kind != "writer" or state.position["entry_contract"] != node.spec.entry_contract:
        raise GroupError("entry does not match admitted writer node")
    if node.reward_contract is None:
        raise GroupError("group entry needs an admitted Phase 5 reward contract")
    context = store.load_context(state.context_ref)
    if any(message.loss_eligible for message in context.messages):
        raise GroupError("entry context includes a trainable action")
    # Validate exact visible projection and the complete checkpoint reference closure.
    project_writer_context(store, checkpoint_id, checkpoint_id)
    budget = store.get_artifact(state.budgets_ref)
    if not isinstance(budget, dict) or not isinstance(budget.get("limits"), dict):
        raise GroupError("entry budget is not complete")
    contract = node.contract
    return {
        "entry_checkpoint_id": checkpoint_id,
        "entry_state_hash": state.identity(),
        "entry_tree_hash": state.tree_hash,
        "instance_hash": instance.identity(),
        "graph_hash": _hash(instance.to_dict()),
        "node_id": node.spec.id,
        "node_visit_id": state.position["visit_id"],
        "node_contract_hash": contract.identity(),
        "controller_contract_hash": _hash(contract.completion),
        "check_contracts_hash": _hash([c.to_dict() for c in node.checks.values()]),
        "reward_contract_hash": node.reward_contract.identity() if node.reward_contract else None,
        "simulator_contract_hash": node.script.identity() if node.script else None,
        # Admission resolves these content-addressed refs, including byte artifacts.
        "source_refs_hash": _hash(instance.source_refs),
        "request_refs_hash": _hash(instance.request_refs),
        "visible_prefix_hash": context.content_hash,
        "context_revision_ref": context.identity(),
        "context_messages_hash": _hash([m.to_dict() for m in context.messages]),
        "rendering_hash": _hash(context.rendering),
        "tool_schemas_hash": _hash(context.tools),
        "budget_ref": state.budgets_ref,
        "budget_hash": _hash(budget),
        "versions_ref": state.versions_ref,
        "versions_hash": _hash(store.get_artifact(state.versions_ref)),
        "author_packet_ref": state.author_packet_ref,
        "requirements_ref": state.requirements_ref,
        "decisions_ref": state.decisions_ref,
        "disclosures_ref": state.disclosures_ref,
        "external_inputs_ref": state.external_inputs_ref,
        "rng_ref": state.rng_ref,
        "outcome_ref": state.outcome_ref,
        "provenance_ref": state.provenance_ref,
        "continuation_hash": _hash(state.continuation),
        "horizon": "node_exit",
    }


@dataclass(frozen=True)
class GroupMemberSpecV1(_Record):
    member_id: str = ""
    ordinal: int = -1
    writer_seed: int = -1
    environment_seed: int = -1
    seed_provenance: str = "sha256-domain-v1"
    DOMAIN = "payload"

    def validate(self) -> None:
        if not self.member_id or type(self.ordinal) is not int or self.ordinal < 0:
            raise GroupError("invalid group member slot")
        if any(type(x) is not int or x < 0 for x in (self.writer_seed, self.environment_seed)):
            raise GroupError("invalid member seed")
        if self.seed_provenance != "sha256-domain-v1":
            raise GroupError("unknown member seed derivation")


@dataclass(frozen=True)
class GroupSpecV1(_Record):
    record_type: str = "GroupSpecV1"
    group_id: str = ""
    group_sequence: int = -1
    group_seed: int = -1
    runner_mode: str = "real"
    environment: dict[str, Any] = None  # type: ignore[assignment]
    policy: dict[str, str] = None  # type: ignore[assignment]
    members: tuple[GroupMemberSpecV1 | dict, ...] = ()
    DOMAIN = "payload"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "members",
            tuple(
                m if isinstance(m, GroupMemberSpecV1) else GroupMemberSpecV1.from_dict(m)
                for m in self.members
            ),
        )
        super().__post_init__()

    def validate(self) -> None:
        if (
            self.record_type != "GroupSpecV1"
            or type(self.group_sequence) is not int
            or self.group_sequence < 0
        ):
            raise GroupError("invalid group identity")
        if (
            type(self.group_seed) is not int
            or self.group_seed < 0
            or not 2 <= len(self.members) <= 64
        ):
            raise GroupError("group size must be 2..64 and seed nonnegative")
        if self.runner_mode not in {"real", "fixture"}:
            raise GroupError("unknown group runner mode")
        validate_hash(self.environment["entry_checkpoint_id"])
        if set(self.policy) != POLICY_FIELDS:
            raise GroupError("incomplete policy contract")
        if self.group_id != _hash(
            [
                "GroupIdV1",
                self.group_sequence,
                self.environment,
                self.policy,
                self.group_seed,
                self.runner_mode,
                len(self.members),
            ]
        ):
            raise GroupError("group ID does not bind its contract and sequence")
        if tuple(m.ordinal for m in self.members) != tuple(range(len(self.members))):
            raise GroupError("member slots are not canonical")
        for member in self.members:
            if member.member_id != f"grp-{self.group_id[:24]}-{member.ordinal:02d}":
                raise GroupError("member ID does not bind its ordinal")
            if member.writer_seed != _seed(self.group_seed, "writer", member.ordinal):
                raise GroupError("writer seed derivation mismatch")
            if member.environment_seed != _seed(self.group_seed, "environment"):
                raise GroupError("environment seed derivation mismatch")
        if len({member.writer_seed for member in self.members}) != len(self.members):
            raise GroupError("writer streams are not distinct")


@dataclass(frozen=True)
class GroupMemberResultV1(_Record):
    record_type: str = "GroupMemberResultV1"
    group_id: str = ""
    member_id: str = ""
    start_checkpoint_id: str = ""
    final_checkpoint_id: str | None = None
    terminal_outcome_ref: str | None = None
    availability_ref: str | None = None
    fixture_ref: str | None = None
    failure_ref: str | None = None
    execution_status: str = "pending"
    DOMAIN = "payload"

    def validate(self) -> None:
        if self.record_type != "GroupMemberResultV1" or self.execution_status not in {
            "pending",
            "valid",
            "infrastructure_invalid",
        }:
            raise GroupError("invalid member result status")
        validate_hash(self.start_checkpoint_id)
        for ref in (
            self.final_checkpoint_id,
            self.terminal_outcome_ref,
            self.availability_ref,
            self.fixture_ref,
            self.failure_ref,
        ):
            validate_hash(ref, optional=True)
        if self.fixture_ref and any(
            (
                self.final_checkpoint_id,
                self.terminal_outcome_ref,
                self.availability_ref,
                self.failure_ref,
            )
        ):
            raise GroupError("fixture and real terminal evidence cannot be mixed")
        if self.execution_status == "pending" and any(
            (
                self.final_checkpoint_id,
                self.terminal_outcome_ref,
                self.availability_ref,
                self.fixture_ref,
                self.failure_ref,
            )
        ):
            raise GroupError("pending result cannot claim terminal or failure evidence")
        if self.execution_status == "infrastructure_invalid" and not (
            self.fixture_ref or self.failure_ref
        ):
            raise GroupError("infrastructure failure needs immutable cause evidence")
        if self.execution_status == "infrastructure_invalid" and (
            self.final_checkpoint_id or self.terminal_outcome_ref or self.availability_ref
        ):
            raise GroupError("infrastructure failure cannot claim a terminal writer result")
        if self.execution_status == "valid" and not (
            self.fixture_ref or (self.final_checkpoint_id and self.terminal_outcome_ref)
        ):
            raise GroupError("valid member needs immutable terminal evidence")
        if self.execution_status == "valid" and self.failure_ref:
            raise GroupError("valid member cannot carry infrastructure failure evidence")


@dataclass(frozen=True)
class GroupDecisionV1(_Record):
    record_type: str = "GroupDecisionV1"
    group_id: str = ""
    status: str = "pending"
    reason: str = ""
    member_result_refs: tuple[str | None, ...] = ()
    advantage_refs: tuple[str, ...] = ()
    segment_credit_refs: tuple[str, ...] = ()
    native_optimizer_eligible: bool = False
    DOMAIN = "payload"

    def validate(self) -> None:
        if self.record_type != "GroupDecisionV1" or self.status not in {
            "pending",
            "invalid",
            "ready",
            "tie",
        }:
            raise GroupError("invalid group decision")
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
class GroupAdvantageV1(_Record):
    record_type: str = "GroupAdvantageV1"
    group_id: str = ""
    member_id: str = ""
    result_ref: str = ""
    reward: Mapping[str, int] = None  # type: ignore[assignment]
    mean: Mapping[str, int] = None  # type: ignore[assignment]
    variance: Mapping[str, int] = None  # type: ignore[assignment]
    centered: Mapping[str, int] = None  # type: ignore[assignment]
    expression: str = "centered / sqrt(population_variance)"
    advantage: Mapping[str, int] | None = None
    zero_variance: bool = False
    native_optimizer_eligible: bool = False
    DOMAIN = "payload"

    def validate(self) -> None:
        if self.record_type != "GroupAdvantageV1" or not self.member_id:
            raise GroupError("invalid advantage assignment")
        validate_hash(self.group_id)
        validate_hash(self.result_ref)
        reward, mean = _read_fraction(self.reward), _read_fraction(self.mean)
        variance, centered = _read_fraction(self.variance), _read_fraction(self.centered)
        if (
            variance < 0
            or centered != reward - mean
            or self.zero_variance is not (variance == 0)
            or self.native_optimizer_eligible
            or (
                self.zero_variance
                and (
                    centered != 0
                    or self.expression != "zero"
                    or _read_fraction(self.advantage) != 0
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
class GroupSegmentCreditV1(_Record):
    record_type: str = "GroupSegmentCreditV1"
    group_id: str = ""
    member_id: str = ""
    action_id: str = ""
    action_ref: str = ""
    message_ref: str = ""
    trace_ref: str = ""
    original_context_ref: str = ""
    original_context_content_hash: str = ""
    advantage_ref: str = ""
    segment_kind: str = ""
    part_index: int | None = None
    segment_content_hash: str | None = None
    excluded_roles: tuple[str, ...] = (
        "system",
        "user",
        "author",
        "tool",
        "seed",
        "environment",
        "summary",
    )
    native_optimizer_eligible: bool = False
    token_mask_ref: None = None
    logprob_ref: None = None
    DOMAIN = "payload"

    def validate(self) -> None:
        if self.record_type != "GroupSegmentCreditV1" or not self.member_id or not self.action_id:
            raise GroupError("invalid action credit identity")
        for ref in (
            self.group_id,
            self.action_ref,
            self.message_ref,
            self.trace_ref,
            self.original_context_ref,
            self.original_context_content_hash,
            self.advantage_ref,
        ):
            validate_hash(ref)
        validate_hash(self.segment_content_hash, optional=True)
        if (
            self.segment_kind not in {"assistant_text", "tool_syntax", "assistant_ending"}
            or (
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
            != ("system", "user", "author", "tool", "seed", "environment", "summary")
            or self.native_optimizer_eligible
            or self.token_mask_ref is not None
            or self.logprob_ref is not None
        ):
            raise GroupError("segment credit cannot assert native or non-writer eligibility")
