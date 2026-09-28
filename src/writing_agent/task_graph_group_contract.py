"""Sealed group inputs and exact, non-native advantage/segment-credit records.

Records are immutable payload-domain values. Worker paths and operational clocks
never enter a group identity; member ordinals, not completion order, fix slot IDs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Any, ClassVar

from writing_agent.task_graph import (
    Record,
    canonical_bytes,
    domain_hash,
    validate_hash,
)
from writing_agent.task_graph_compaction import require_quiescent
from writing_agent.task_graph_record_contracts import (
    POLICY_FIELDS,
    GroupError,
)
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_wire import Bool, Enum, Hash, Int, Str, UnionOf, WireRecord, obj


def payload_hash(value: Any) -> str:
    return domain_hash("payload", value)


def derive_group_seed(group_seed: int, role: str, ordinal: int | None = None) -> int:
    if type(group_seed) is not int or group_seed < 0:
        raise GroupError("group seed must be a nonnegative integer")
    material = canonical_bytes(["GroupSeedV1", group_seed, role, ordinal])
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def fraction_wire(value: Fraction) -> dict[str, int]:
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
    if fraction_wire(fraction) != dict(value):
        raise GroupError("fraction is not canonical and reduced")
    return fraction


def validate_group_policy(policy: dict[str, str], rendering: dict[str, str]) -> dict[str, str]:
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


def resolve_group_environment(
    store: TaskGraphStore,
    *,
    view: Any,
) -> dict[str, Any]:
    """Resolve the equality-critical entry contract from a verified new-core view."""
    state, node, context = view.state, view.node, view.context
    instance = store.load_instance(state.instance_ref)
    visible_prefix_hash = context.content_ref
    context_revision_ref = context.revision_ref
    if (
        state.position["phase"] != "ready_writer"
        or state.history["action_ids"]
        or state.history["tool_result_ids"]
    ):
        raise GroupError("entry must be an unsampled writer checkpoint")
    require_quiescent(state)
    if (
        node.spec.kind != "writer"
        or state.position["entry_contract"] != node.spec.entry_contract
        or node.reward_contract is None
    ):
        raise GroupError("group entry needs an admitted writer and reward contract")
    if any(message.loss_eligible for message in context.messages):
        raise GroupError("entry context includes a trainable action")
    budget = store.get_artifact(state.budgets_ref)
    if not isinstance(budget, dict) or not isinstance(budget.get("limits"), dict):
        raise GroupError("entry budget is not complete")
    contract = node.contract
    return {
        "entry_checkpoint_id": view.checkpoint_id,
        "entry_state_hash": state.identity(),
        "entry_tree_hash": state.tree_hash,
        "instance_hash": instance.identity(),
        "graph_hash": payload_hash(instance.to_dict()),
        "node_id": node.spec.id,
        "node_visit_id": state.position["visit_id"],
        "node_contract_hash": contract.identity(),
        "controller_contract_hash": payload_hash(contract.completion),
        "check_contracts_hash": payload_hash([c.to_dict() for c in node.checks.values()]),
        "reward_contract_hash": node.reward_contract.identity(),
        "simulator_contract_hash": node.script.identity() if node.script else None,
        "source_refs_hash": payload_hash(instance.source_refs),
        "request_refs_hash": payload_hash(instance.request_refs),
        "visible_prefix_hash": visible_prefix_hash,
        "context_revision_ref": context_revision_ref,
        "context_messages_hash": payload_hash([m.to_dict() for m in context.messages]),
        "rendering_hash": payload_hash(context.rendering),
        "tool_schemas_hash": payload_hash(context.tools),
        "budget_ref": state.budgets_ref,
        "budget_hash": payload_hash(budget),
        "versions_ref": state.versions_ref,
        "versions_hash": payload_hash(store.get_artifact(state.versions_ref)),
        "author_packet_ref": state.author_packet_ref,
        "requirements_ref": state.requirements_ref,
        "decisions_ref": state.decisions_ref,
        "disclosures_ref": state.disclosures_ref,
        "external_inputs_ref": state.external_inputs_ref,
        "rng_ref": state.rng_ref,
        "outcome_ref": state.outcome_ref,
        "provenance_ref": state.provenance_ref,
        "continuation_hash": payload_hash(state.continuation),
        "horizon": "node_exit",
    }


@dataclass(frozen=True)
class GroupScriptedTerminalV1(WireRecord):
    schema: Annotated[int, Int(equals=1)]
    group_id: Annotated[str, Hash(None)]
    member_id: Annotated[str, Str(nonempty=True, logical=True)]
    start_checkpoint_id: Annotated[str, Hash("checkpoint")]
    execution_status: Annotated[str, Enum(frozenset({"valid", "infrastructure_invalid"}))]
    reward_status: Annotated[str, Enum(frozenset({"available", "pending", "unavailable"}))]
    reward: Annotated[
        Mapping[str, int] | None,
        UnionOf(
            (
                obj(numerator=Int(minimum=None), denominator=Int(minimum=1)),
                type(None),
            )
        ),
    ]
    native_optimizer_eligible: Annotated[bool, Bool()]
    RECORD_TYPE: ClassVar[str] = "GroupScriptedTerminalV1"

    def check(self) -> None:
        if self.native_optimizer_eligible:
            raise GroupError("scripted result cannot be native optimizer eligible")
        if self.reward_status == "available":
            if self.execution_status != "valid" or self.reward is None:
                raise GroupError("available scripted reward needs a valid exact reward")
            _read_fraction(self.reward)
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
    member_id: Annotated[str, Str(nonempty=True, logical=True)]
    start_checkpoint_id: Annotated[str, Hash("checkpoint")]
    reason: Annotated[str, Str(nonempty=True)]
    evidence_ref: Annotated[str | None, Hash("artifact", optional=True)]
    RECORD_TYPE: ClassVar[str] = "GroupExecutionFailureV1"


@dataclass(frozen=True)
class GroupMemberResultV1(Record):
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
class GroupDecisionV1(Record):
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
class GroupAdvantageV1(Record):
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
class GroupSegmentCreditV1(Record):
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
