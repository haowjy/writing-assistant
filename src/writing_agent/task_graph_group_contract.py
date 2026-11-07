"""Sealed group inputs and exact, non-native advantage/segment-credit records.

Records are immutable payload-domain values. Worker paths and operational clocks
never enter a group identity; member ordinals, not completion order, fix slot IDs.
"""

from __future__ import annotations

import hashlib
from typing import Any

from writing_agent.task_graph import (
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


def payload_hash(value: Any) -> str:
    return domain_hash("payload", value)


def derive_group_seed(group_seed: int, role: str, ordinal: int | None = None) -> int:
    if type(group_seed) is not int or group_seed < 0:
        raise GroupError("group seed must be a nonnegative integer")
    material = canonical_bytes(["GroupSeedV1", group_seed, role, ordinal])
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


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
        or state.history["action_count"]
        or state.history["tool_result_count"]
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
