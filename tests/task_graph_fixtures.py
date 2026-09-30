"""Test-only fixture specializations over production entry construction."""

from __future__ import annotations

from dataclasses import replace

from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
from writing_agent.task_graph_contracts import CheckContractV1, EvaluatorPacketV1, RewardContractV1
from writing_agent.task_graph_derive_entry import derive_entry
from writing_agent.task_graph_probe_tasks import (
    EntryFixture,
    MemoryArtifactReader,
    make_entry_fixture,
)


def make_outcome_fixture(
    *, incomplete_score: int = 0, rendering_overrides=None, public_records=()
) -> EntryFixture:
    """Build an admitted writer entry with one deterministic required completion check."""
    base = make_entry_fixture(
        rendering_overrides=rendering_overrides, public_records=public_records
    )
    old_node = base.graph.node(base.node_id)
    check = CheckContractV1(
        id="nonempty",
        evaluator_version="deterministic-v1",
        applicability="node_exit_candidate",
        required=True,
        spec={
            "id": "nonempty",
            "metric": "Q1",
            "kind": "nonempty",
            "method": "deterministic",
            "required": True,
            "path": "draft.txt",
        },
    )
    reward = RewardContractV1(components={"nonempty": 10_000}, incomplete_score=incomplete_score)
    packet = EvaluatorPacketV1(reward_contract_ref=reward.identity(), check_ids=(check.id,))
    for record in (check, reward, packet):
        base.reader.private[record.identity()] = record.to_dict()

    completion = replace(
        old_node.contract.completion_contract,
        required_check_ids=(check.id,),
        evaluation_packet_ref=packet.identity(),
    )
    contract = replace(
        old_node.contract,
        completion=completion,
        mandatory_checks=(check.identity(),),
    )
    base.reader.public[contract.identity()] = contract.to_dict()
    spec = replace(old_node.spec, entry_contract=contract.identity())
    instance = replace(base.graph.instance, nodes=(spec,))
    graph = admit_graph(
        instance,
        MappingArtifactResolver(base.reader.public, base.reader.private),
        policy=base.graph.policy,
    )
    entry = derive_entry(graph, base.node_id, base.params, base.reader)
    return replace(base, graph=graph, state=entry.state, artifacts=entry.artifacts)


__all__ = [
    "EntryFixture",
    "MemoryArtifactReader",
    "make_entry_fixture",
    "make_outcome_fixture",
]
