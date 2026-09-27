"""Hermetic task-graph entry fixtures backed by the production derive."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from writing_agent.legacy_graph import compile_legacy_scenario
from writing_agent.task_graph import CheckpointV1, domain_hash, load_canonical_json
from writing_agent.task_graph_admission import AdmittedGraphV1
from writing_agent.task_graph_derive_entry import EntryParamsV1, derive_entry
from writing_agent.task_graph_records import (
    AdmissionPolicyV1,
    ContextContentV1,
    ContextRevisionV1,
    MaterializedContextV1,
    materialize_context_nodes,
)
from writing_agent.task_graph_transition import DerivedArtifact


def _scenario() -> dict[str, Any]:
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
    """Hash-indexed, visibility-aware artifact reader with copy-on-read results."""

    def __init__(self, public: dict[str, Any], private: dict[str, Any] | None = None) -> None:
        self.public = dict(public)
        self.private = dict(private or {})
        self.byte_values: dict[str, bytes] = {}
        self.checkpoints: dict[str, CheckpointV1] = {}
        self.context_revisions: dict[str, Any] = {}
        self.context_nodes: dict[str, Any] = {}

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


def make_entry_fixture() -> EntryFixture:
    """Build a complete pure entry and all its deterministic artifacts."""
    bundle = compile_legacy_scenario(_scenario())
    graph = bundle.admission()
    reader = MemoryArtifactReader(
        {identity: _copy(body) for identity, body in bundle.public_artifacts.items()},
        {identity: _copy(body) for identity, body in bundle.private_artifacts.items()},
    )
    rendering = {
        "projection_version": "v1",
        "prefix_id": "root",
        "template_ref": reader.add({"pin": "template"}),
        "tokenizer_ref": reader.add({"pin": "tokenizer"}),
        "tool_schema_ref": reader.add({"pin": "tools"}),
    }
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
    entry = derive_entry(graph, node_id, params, reader)
    state, artifacts = entry.state, entry.artifacts
    for artifact in artifacts:
        if artifact.kind in {"artifact", "private"}:
            (reader.private if artifact.kind == "private" else reader.public)[artifact.ref] = (
                _artifact_body(artifact.value)
            )
        elif artifact.kind == "context_revision":
            reader.context_revisions[artifact.ref] = _artifact_body(artifact.value)
        elif artifact.kind == "context_node":
            reader.context_nodes[artifact.ref] = _artifact_body(artifact.value)
    return EntryFixture(graph, node_id, params, reader, state, artifacts)


__all__ = ["EntryFixture", "MemoryArtifactReader", "make_entry_fixture"]
