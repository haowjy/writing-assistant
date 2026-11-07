"""Pure construction of admitted Phase 8 probe task graphs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from writing_agent.legacy_graph import compile_legacy_scenario
from writing_agent.task_graph import (
    CheckpointV1,
    MaterializedContextV1,
    domain_hash,
    load_canonical_json,
)
from writing_agent.task_graph_admission import (
    AdmittedGraphV1,
    MappingArtifactResolver,
    admit_graph,
)
from writing_agent.task_graph_contracts import (
    CheckContractV1,
    EvaluatorPacketV1,
    RewardContractV1,
)
from writing_agent.task_graph_derive_entry import EntryParamsV1, derive_entry
from writing_agent.task_graph_environment import RolloutEnvironment
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_records import (
    AdmissionPolicyV1,
    ContextContentV1,
    ContextRevisionV1,
    materialize_context_nodes,
)
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_transition import DerivedArtifact

AUTHOR_PACKET_CANARY = "P8R3C_PRIVATE_AUTHOR_PREF_CANARY_4172"
PRIVATE_STORE_DUMP_CANARY = "P8R3C_PRIVATE_EVALUATOR_SPEC_CANARY_8365"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "phase8" / "probe-tasks"


@dataclass(frozen=True)
class ProbeTaskEntry:
    """One persisted task graph and its environment, independent of the trainer API."""

    task_id: str
    environment: RolloutEnvironment
    entry_checkpoint_id: str


def _scenario() -> dict[str, Any]:
    """Minimal common entry used to derive admitted probe tasks."""
    return {
        "id": "transition-entry",
        "family": "F2",
        "role": "development",
        "condition": "workspace",
        "source_groups": ["synthetic"],
        "provenance": "synthetic",
        "visible": {
            "brief": "Revise the text files.",
            "initial_files": {
                "draft.txt": "alpha\n",
                "notes/source.md": "snow\nmoon\n",
            },
            "followups": [],
            "tools": ["list_dir", "read_file", "search", "write_file", "patch_file"],
            "budgets": {
                "max_steps": 5,
                "max_tool_calls": 10,
                "max_read_tokens": 100,
                "max_total_bytes": 4096,
            },
            "prose": [],
        },
        "labels": {
            "rubric_version": 1,
            "checks": [],
            "rubrics": {},
            "knowledge": [],
            "source_cutoff": "supplied files only",
        },
    }


class MemoryArtifactReader:
    """Hash-indexed in-memory artifact view shared by entry derivations."""

    def __init__(self, public: dict[str, Any], private: dict[str, Any] | None = None) -> None:
        self.public = dict(public)
        self.private = dict(private or {})
        self.byte_values: dict[str, bytes] = {}
        self.checkpoints: dict[str, CheckpointV1] = {}
        self.context_revisions: dict[str, Any] = {}
        self.context_nodes: dict[str, Any] = {}
        self.events: dict[str, Any] = {}

    def add(self, value: Any, *, private: bool = False) -> str:
        identity = domain_hash("payload", value)
        (self.private if private else self.public)[identity] = value
        return identity

    def artifact(self, ref: str, *, domain: str = "payload", private: bool = False) -> Any:
        if domain == "payload":
            value = (self.private if private else self.public)[ref]
        elif domain == "context_revision":
            value = self.context_revisions[ref]
        elif domain == "context_node":
            value = self.context_nodes[ref]
        elif domain == "event":
            value = self.events[ref]
        else:
            raise KeyError((domain, ref))
        return _copy(value)

    def context(self, ref: str) -> MaterializedContextV1:
        revision = ContextRevisionV1.from_dict(self.artifact(ref, domain="context_revision"))
        nodes = {
            identity: ContextContentV1.from_dict(_copy(value))
            for identity, value in self.context_nodes.items()
        }
        return materialize_context_nodes(revision, nodes)

    def bytes_artifact(self, ref: str) -> bytes:
        return self.byte_values[ref]

    def checkpoint(self, ref: str) -> CheckpointV1:
        return self.checkpoints[ref]


def _copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_copy(item) for item in value]
    return value


def _artifact_body(value: Any) -> Any:
    if isinstance(value, bytes):
        return load_canonical_json(value)
    if hasattr(value, "to_wire"):
        return value.to_wire()
    return _copy(value)


@dataclass(frozen=True)
class EntryFixture:
    graph: AdmittedGraphV1
    node_id: str
    params: EntryParamsV1
    reader: MemoryArtifactReader
    state: Any
    artifacts: tuple[DerivedArtifact, ...]


def make_entry_fixture(*, rendering_overrides=None, public_records=()) -> EntryFixture:
    """Build the deterministic common entry and its derived artifacts."""
    bundle = compile_legacy_scenario(_scenario())
    graph = bundle.admission()
    reader = MemoryArtifactReader(
        {identity: _copy(body) for identity, body in bundle.public_artifacts.items()},
        {identity: _copy(body) for identity, body in bundle.private_artifacts.items()},
    )
    for record in public_records:
        reader.public[record.identity()] = record.to_wire()
    rendering = {
        "projection_version": "v1",
        "prefix_id": "root",
        "template_ref": reader.add({"pin": "template"}),
        "tokenizer_ref": reader.add({"pin": "tokenizer"}),
        "tool_schema_ref": reader.add({"pin": "tools"}),
    }
    rendering.update(rendering_overrides or {})
    admission_policy = AdmissionPolicyV1.from_admission_policy(graph.policy)
    admission_policy_ref = reader.add(admission_policy.to_wire())
    versions_ref = reader.add(
        {
            "schema": 1,
            "transition_semantics": "task-graph-derive-v1",
            "admission_policy_ref": admission_policy_ref,
            "tool_spec": {"max_file_bytes": 128_000, "max_workspace_bytes": 4096},
        }
    )
    params = EntryParamsV1(
        lineage_id="rollout-fixture",
        visit_id="visit-fixture",
        rendering=rendering,
        versions_ref=versions_ref,
        provenance_ref=reader.add({"fixture": "provenance"}),
        rng_ref=reader.add({"fixture": "rng", "seed": 7}),
    )
    node_id = graph.instance.entry_node
    derived = derive_entry(graph, node_id, params, reader)
    for artifact in derived.artifacts:
        if artifact.kind in {"artifact", "private"}:
            target = reader.private if artifact.kind == "private" else reader.public
            target[artifact.ref] = _artifact_body(artifact.value)
        elif artifact.kind == "context_revision":
            reader.context_revisions[artifact.ref] = _artifact_body(artifact.value)
        elif artifact.kind == "context_node":
            reader.context_nodes[artifact.ref] = _artifact_body(artifact.value)
    return EntryFixture(graph, node_id, params, reader, derived.state, derived.artifacts)


def load_probe_task(path: Path) -> dict[str, Any]:
    """Load and validate one schema-1 declarative probe task."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("probe task config must be a schema 1 object")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise ValueError("probe task config needs an id")
    return value


def load_probe_tasks(config_dir: Path = CONFIG_DIR) -> tuple[dict[str, Any], ...]:
    """Load the fixed public task set in stable filename order."""
    return tuple(load_probe_task(path) for path in sorted(config_dir.glob("t*.json")))


def build_admitted_entry(config: dict[str, Any]) -> EntryFixture:
    """Build an admitted one-writer probe graph from its public task config."""
    base = make_entry_fixture()
    old_node = base.graph.node(base.node_id)
    reader = base.reader
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
    limits = config["probe_settings"]
    checks_config = config["checks"]
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

    # Keep this non-admitted record's canary outside the evaluator packet. It detects a
    # bulk private-store dump, not leakage of an admitted evaluator check spec.
    canary_check = CheckContractV1(
        id="privacy_canary_only",
        evaluator_version="deterministic-v1",
        applicability="node_exit_candidate",
        required=False,
        spec={
            "id": "privacy_canary_only",
            "metric": "Q13",
            "kind": "nonempty",
            "method": "deterministic",
            "required": False,
            "path": "scene.txt",
            "private_fixture_canary": PRIVATE_STORE_DUMP_CANARY,
            "unused_author_preference_canary": AUTHOR_PACKET_CANARY,
        },
    )
    reader.private[canary_check.identity()] = canary_check.to_dict()

    reward = RewardContractV1(components=checks_config["components"], incomplete_score=0)
    packet = EvaluatorPacketV1(reward_contract_ref=reward.identity(), check_ids=check_ids)
    reader.private[reward.identity()] = reward.to_dict()
    reader.private[packet.identity()] = packet.to_dict()

    old_contract = old_node.contract
    entry_contract = replace(
        old_contract.entry_contract,
        request_ref=request_ref,
        files_ref=files_ref,
    )
    budgets = replace(
        old_contract.budget_contract,
        max_steps=limits["max_writer_turns"],
        max_tool_calls=limits["max_tool_calls"],
        max_generated_tokens=limits["max_generated_tokens"],
        max_context_tokens=limits["max_context_tokens"],
    )
    completion = replace(
        old_contract.completion_contract,
        required_check_ids=(required_id,),
        required_script_turns=0,
        evaluation_packet_ref=packet.identity(),
    )
    contract = replace(
        old_contract,
        entry=entry_contract,
        budgets=budgets,
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


def persist_entry(store: TaskGraphStore, entry: EntryFixture) -> str:
    for body in entry.reader.public.values():
        store.put_artifact(body)
    for body in entry.reader.private.values():
        store.put_artifact(body, private=True)
    store.persist(entry.graph.instance)
    for artifact in entry.artifacts:
        store.persist_artifact(artifact)
    return store.save_checkpoint(entry.state)


def task_entries(
    store: TaskGraphStore,
    gate: LineageGate,
    bind_entry: Callable[[EntryFixture], EntryFixture],
    *,
    config_dir: Path = CONFIG_DIR,
) -> tuple[ProbeTaskEntry, ...]:
    """Persist public probe tasks and return their core rollout entry records."""
    entries = []
    for config in load_probe_tasks(config_dir):
        fixture = bind_entry(build_admitted_entry(config))
        checkpoint = persist_entry(store, fixture)
        environment = RolloutEnvironment(store, fixture.graph, None, gate, fixture.graph.policy)
        entries.append(ProbeTaskEntry(config["id"], environment, checkpoint))
    return tuple(entries)


__all__ = [
    "AUTHOR_PACKET_CANARY",
    "CONFIG_DIR",
    "PRIVATE_STORE_DUMP_CANARY",
    "EntryFixture",
    "MemoryArtifactReader",
    "ProbeTaskEntry",
    "build_admitted_entry",
    "load_probe_task",
    "load_probe_tasks",
    "make_entry_fixture",
    "persist_entry",
    "task_entries",
]
