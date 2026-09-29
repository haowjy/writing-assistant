"""Load public Phase 8 probe task configs into admitted rollout fixtures.

This small loader is intentionally fixture-only: it reuses the entry fixture and
production admission path rather than adding a second graph authoring DSL.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from tests.task_graph_fixtures import EntryFixture, make_entry_fixture
from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
from writing_agent.task_graph_contracts import (
    AuthorPacketV1,
    CheckContractV1,
    DecisionBindingsV1,
    EvaluatorPacketV1,
    InteractionContractV1,
    InteractionPolicyV1,
    RewardContractV1,
    ScriptedAuthorV1,
)
from writing_agent.task_graph_derive_entry import derive_entry

CONFIG_DIR = Path(__file__).resolve().parent


def load_probe_task(path: Path) -> dict[str, Any]:
    """Read one declarative public task and reject malformed top-level data."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("probe task config must be a schema 1 object")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise ValueError("probe task config needs an id")
    return value


def load_probe_tasks() -> tuple[dict[str, Any], ...]:
    """Load the three published task configs in stable task order."""
    return tuple(load_probe_task(path) for path in sorted(CONFIG_DIR.glob("t*.json")))


def build_admitted_entry(config: dict[str, Any]) -> EntryFixture:
    """Build an admitted one-writer graph from a public probe config."""
    base = make_entry_fixture()
    old_node = base.graph.node(base.node_id)
    reader = base.reader

    # Keep only the fixture's common pins, version policy, and terminal guard. The
    # generated probe graphs therefore carry no unrelated fixture/source artifacts.
    keep_public = {
        base.graph.instance.template_ref,
        base.params.versions_ref,
        base.params.provenance_ref,
        base.params.rng_ref,
        *(
            value
            for key, value in base.params.rendering.items()
            if key.endswith("_ref") and isinstance(value, str)
        ),
        *(edge.guard_ref for edge in old_node.edges),
    }
    versions = reader.public[base.params.versions_ref]
    keep_public.add(versions["admission_policy_ref"])
    reader.public = {key: value for key, value in reader.public.items() if key in keep_public}
    reader.private.clear()
    reader.context_revisions.clear()
    reader.context_nodes.clear()

    public = config["public"]
    settings = config["probe_settings"]
    checks_config = config["checks"]
    decision = public["decision"]
    feedback = public["feedback"]
    check_ids = (checks_config["required_id"], *(item["id"] for item in checks_config["optional"]))

    request_ref = reader.add({"kind": "probe-public-request", "schema": 1, "text": public["brief"]})
    files_ref = reader.add(
        {
            "kind": "legacy-visible-files",
            "schema": 1,
            "scenario_id": config["id"],
            "files": public["initial_files"],
        }
    )

    required_id = checks_config["required_id"]
    check_specs = [
        {"id": required_id, "kind": "nonempty", "path": "scene.txt"},
        *checks_config["optional"],
    ]
    checks = []
    for index, item in enumerate(check_specs, start=1):
        required = item["id"] == required_id
        spec = {
            "id": item["id"],
            "metric": f"Q{index}",
            "kind": item["kind"],
            "method": "deterministic",
            "required": required,
            **{key: value for key, value in item.items() if key not in {"id", "kind"}},
        }
        check = CheckContractV1(
            id=item["id"],
            evaluator_version="deterministic-v1",
            applicability="node_exit_candidate",
            required=required,
            spec=spec,
        )
        reader.private[check.identity()] = check.to_dict()
        checks.append(check)

    reward = RewardContractV1(components=checks_config["components"], incomplete_score=0)
    packet = EvaluatorPacketV1(reward_contract_ref=reward.identity(), check_ids=check_ids)
    reader.private[reward.identity()] = reward.to_dict()
    reader.private[packet.identity()] = packet.to_dict()

    author_packet = AuthorPacketV1(
        preferences=public["author_packet"]["preferences"], requirements={}
    )
    answer = author_packet.preferences[decision["binding"]]
    script = ScriptedAuthorV1(
        answers={
            decision["id"]: {
                "mode": "fixed_answer",
                "utterance": f"Use the {answer} choice.",
                "value": answer,
                "selector": None,
                "prerequisite_check_ids": [],
            }
        },
        feedback=(
            {
                "id": feedback["id"],
                "utterance": feedback["utterance"],
                "prerequisite_check_ids": [],
                "requirement_update_ref": None,
            },
        ),
    )
    policy = InteractionPolicyV1(
        public_decisions=({"id": decision["id"], "label": decision["label"]},),
        mandatory_feedback=(feedback["id"],),
    )
    bindings = DecisionBindingsV1(bindings={decision["id"]: decision["binding"]})
    for record in (author_packet, script, bindings):
        reader.private[record.identity()] = record.to_dict()
    reader.public[policy.identity()] = policy.to_dict()

    old_contract = old_node.contract
    entry = old_contract.entry_contract
    entry = replace(
        entry,
        request_ref=request_ref,
        files_ref=files_ref,
        tool_allowlist=(*entry.tool_allowlist, "ask_author"),
    )
    interaction = InteractionContractV1(
        mode="scripted_author",
        script_ref=script.identity(),
        author_packet_ref=author_packet.identity(),
        interaction_policy_ref=policy.identity(),
        decision_bindings_ref=bindings.identity(),
        mandatory_feedback=(feedback["id"],),
    )
    budget = old_contract.budget_contract
    budget = replace(
        budget,
        max_steps=settings["max_writer_turns"],
        max_tool_calls=settings["max_tool_calls"],
        max_author_calls=settings["max_author_calls"],
        max_generated_tokens=settings["max_generated_tokens"],
        max_total_tokens=settings["max_total_tokens"],
    )
    completion = old_contract.completion_contract
    completion = replace(
        completion,
        required_check_ids=(required_id,),
        required_script_turns=0,
        evaluation_packet_ref=packet.identity(),
    )
    contract = replace(
        old_contract,
        entry=entry,
        interaction=interaction,
        budgets=budget,
        completion=completion,
        mandatory_checks=(checks[0].identity(),),
        optional_checks=tuple(check.identity() for check in checks[1:]),
    )
    reader.public[contract.identity()] = contract.to_dict()

    spec = replace(old_node.spec, entry_contract=contract.identity())
    instance = replace(
        base.graph.instance,
        nodes=(spec,),
        source_refs=(files_ref,),
        request_refs=(request_ref,),
    )
    graph = admit_graph(
        instance,
        MappingArtifactResolver(reader.public, reader.private),
        policy=base.graph.policy,
    )
    derived = derive_entry(graph, base.node_id, base.params, reader)
    return EntryFixture(graph, base.node_id, base.params, reader, derived.state, derived.artifacts)


__all__ = ["build_admitted_entry", "load_probe_task", "load_probe_tasks"]
