"""Pure construction of a task-graph node's entry state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from writing_agent.task_graph import (
    EnvironmentStateV1,
    MessageV1,
    Phase,
    domain_hash,
    tree_hash,
    validate_hash,
)
from writing_agent.task_graph_contracts import (
    writer_tool_schemas,
)
from writing_agent.task_graph_records import (
    ContextContentV1,
    ContextRevisionV1,
    ExternalInputsV1,
    OutcomeV1,
)
from writing_agent.task_graph_transition import ArtifactReader, DerivedArtifact

if TYPE_CHECKING:
    from writing_agent.task_graph_admission import AdmittedGraphV1, AdmittedNodeV1

# Entry prompt is pinned by transition_semantics. Keep it byte-identical to the
# legacy entry prompt while the old and new runtimes coexist.
SYSTEM_PROMPT = """You are a conversational creative-writing collaborator.
Use project files when needed. Follow the latest explicit user decision over stale notes.
Distinguish proposals, drafts, accepted prose, and committed canon. New draft prose does
not automatically authorize canon updates. Preserve character knowledge and deferred reveals.
Write only the requested amount. Make local edits without changing unrelated text.
Use relative workspace paths. Return a final answer when the requested work is complete.
"""
_READ_TOKENIZER = "whitespace-v1"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class EntryParamsV1:
    """Free values pinned by the caller at the start of one rollout."""

    lineage_id: str
    visit_id: str
    versions_ref: str
    provenance_ref: str
    rng_ref: str
    rendering: Mapping[str, Any] | None = None
    rendering_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lineage_id, str) or not self.lineage_id:
            raise ValueError("lineage_id must be nonempty")
        if not isinstance(self.visit_id, str) or not self.visit_id:
            raise ValueError("visit_id must be nonempty")
        for name in ("versions_ref", "provenance_ref", "rng_ref"):
            validate_hash(getattr(self, name))
        validate_hash(self.rendering_ref, optional=True)
        if self.rendering is not None:
            if not isinstance(self.rendering, Mapping):
                raise TypeError("rendering pins must be an object")
            object.__setattr__(self, "rendering", _freeze(self.rendering))


def params_of(state: EnvironmentStateV1) -> EntryParamsV1:
    """Extract caller-controlled entry parameters from a state.

    Rendering lives under the content-addressed context record rather than in
    EnvironmentStateV1. This stores its hash as a locator; derive_entry reads only
    the rendering pins and reconstructs the root context from the admitted node.
    """
    if not isinstance(state, EnvironmentStateV1):
        raise TypeError("state must be EnvironmentStateV1")
    return EntryParamsV1(
        lineage_id=state.position["lineage_id"],
        visit_id=state.position["visit_id"],
        versions_ref=state.versions_ref,
        provenance_ref=state.provenance_ref,
        rng_ref=state.rng_ref,
        rendering_ref=state.context_ref,
    )


def _rendering(params: EntryParamsV1, reader: ArtifactReader) -> Mapping[str, Any]:
    rendering = params.rendering
    if rendering is None:
        if params.rendering_ref is None:
            raise ValueError("entry rendering pins or a context rendering reference are required")
        revision_body = reader.artifact(params.rendering_ref, domain="context_revision")
        revision = (
            revision_body
            if isinstance(revision_body, ContextRevisionV1)
            else ContextRevisionV1.from_dict(dict(revision_body))
        )
        content_ref = revision.content_ref
        seen: set[str] = set()
        while True:
            if content_ref in seen:
                raise ValueError("context content chain contains a cycle")
            seen.add(content_ref)
            content_body = reader.artifact(content_ref, domain="context_content")
            content = (
                content_body
                if isinstance(content_body, ContextContentV1)
                else ContextContentV1.from_dict(dict(content_body))
            )
            if content.rendering is not None:
                rendering = content.rendering
                break
            if content.parent_ref is None:
                raise ValueError("context content root is missing its rendering pins")
            content_ref = content.parent_ref
    required = {
        "projection_version",
        "prefix_id",
        "template_ref",
        "tokenizer_ref",
        "tool_schema_ref",
    }
    if set(rendering) != required:
        raise ValueError("rendering pins have an invalid field set")
    return rendering


def _initial_requirements(node: AdmittedNodeV1, reader: ArtifactReader) -> dict[str, Any]:
    if node.author_packet is not None:
        active = dict(node.author_packet.requirements)
    else:
        source_ref = node.contract.entry_contract.requirement_version
        active: dict[str, str] = {}
        if source_ref is not None:
            source = reader.artifact(source_ref, private=True)
            if isinstance(source, Mapping):
                if "requirements" not in source and "active" not in source:
                    raise ValueError("initial requirement version has no text map")
                requirements = source.get("requirements", source.get("active"))
                if isinstance(requirements, Mapping) and all(
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in requirements.items()
                ):
                    active = dict(requirements)
                else:
                    raise ValueError("initial requirement version must expose a text map")
            else:
                raise ValueError("initial requirement version must be an object")
    return {
        "record_type": "RequirementLedgerV1",
        "schema": 1,
        "active": active,
        "superseded": {},
    }


def _entry(
    graph: AdmittedGraphV1,
    node_id: str,
    params: EntryParamsV1,
    reader: ArtifactReader,
) -> tuple[EnvironmentStateV1, tuple[DerivedArtifact, ...]]:
    node = graph.node(node_id)
    if node.spec.kind != "writer":
        raise ValueError("entry state requires a writer node")
    contract = node.contract
    entry = contract.entry_contract
    files = reader.artifact(entry.files_ref)
    if isinstance(files, Mapping) and files.get("kind") == "legacy-visible-files":
        files = files.get("files")
    if not isinstance(files, Mapping) or any(
        not isinstance(path, str) or not isinstance(text, str) for path, text in files.items()
    ):
        raise TypeError("entry files artifact must map paths to text")
    files = dict(files)

    request = reader.artifact(entry.request_ref)
    if not isinstance(request, Mapping) or not isinstance(request.get("text"), str):
        raise TypeError("entry request artifact must contain text")
    request_text = request["text"]
    rendering = _rendering(params, reader)
    tools = writer_tool_schemas(entry.tool_allowlist, node.interaction_policy)
    messages = (
        MessageV1(
            role="system",
            content=(SYSTEM_PROMPT,),
            origin="system:1",
            trust="instructions",
        ),
        MessageV1(role="user", content=(request_text,), origin="request:1"),
    )

    context_content = ContextContentV1(
        parent_ref=None,
        messages=messages,
        tools=tools,
        rendering=rendering,
    )
    context_revision = ContextRevisionV1(
        content_ref=context_content.identity(),
        event_head=None,
        provenance_refs=(),
    )
    external_inputs = ExternalInputsV1(schema=1, source_refs=graph.instance.source_refs)
    outcome = OutcomeV1(
        schema=1,
        task_status="unknown",
        execution_status="running",
        stop_reason=None,
        reward_status="pending",
        training_eligibility="pending",
        candidate_checkpoint=None,
        requirement_version=None,
        checks=(),
        transition_edge_id=None,
        failed_request_ref=None,
        reward_ref=None,
        eligibility_ref=None,
    )

    requirements = _initial_requirements(node, reader)
    decisions = {
        "record_type": "DecisionLedgerV1",
        "schema": 1,
        "values": {},
        "proposals": {},
    }
    disclosures = {
        "record_type": "DisclosureLedgerV1",
        "schema": 1,
        "decisions": [],
    }
    budget_limits = {
        "writer_turns": contract.budget_contract.max_steps,
        "tool_calls": contract.budget_contract.max_tool_calls,
        "read_tokens": contract.budget_contract.max_read_tokens,
        "storage_bytes": contract.budget_contract.max_total_bytes,
    }
    if contract.interaction_contract.mode == "scripted_author":
        budget_limits["author_calls"] = contract.budget_contract.max_author_calls
    budget = {
        "schema": 1,
        "limits": budget_limits,
        "consumed": {"storage_bytes": sum(len(text.encode("utf-8")) for text in files.values())},
        "read_tokenizer": _READ_TOKENIZER,
    }

    requirements_ref = domain_hash("payload", requirements)
    decisions_ref = domain_hash("payload", decisions)
    disclosures_ref = domain_hash("payload", disclosures)
    budget_ref = domain_hash("payload", budget)
    outcome_ref = domain_hash("payload", outcome.to_wire())
    external_inputs_ref = domain_hash("payload", external_inputs.to_wire())
    state = EnvironmentStateV1(
        instance_ref=graph.instance.identity(),
        position={
            "node_id": node.spec.id,
            "visit_id": params.visit_id,
            "phase": Phase.READY_WRITER.value,
            "entry_contract": node.spec.entry_contract,
            "start_checkpoint": None,
            "loop_counts": {},
            "lineage_id": params.lineage_id,
        },
        files=files,
        tree_hash=tree_hash(files),
        history={
            "head": None,
            "seq": 0,
            "branch_base": None,
            "imported_refs": (),
            "action_ids": (),
            "tool_result_ids": (),
        },
        context_ref=context_revision.identity(),
        requirements_ref=requirements_ref,
        decisions_ref=decisions_ref,
        disclosures_ref=disclosures_ref,
        author_packet_ref=contract.interaction_contract.author_packet_ref,
        versions_ref=params.versions_ref,
        budgets_ref=budget_ref,
        rng_ref=params.rng_ref,
        external_inputs_ref=external_inputs_ref,
        outcome_ref=outcome_ref,
        provenance_ref=params.provenance_ref,
        continuation={
            "tool_queue": (),
            "next_call": 0,
            "author_request": None,
            "check_requests": (),
            "external_requests": (),
            "applied_responses": (),
            "feedback_cursor": 0,
        },
    )
    artifacts = (
        DerivedArtifact(requirements_ref, requirements, "private"),
        DerivedArtifact(decisions_ref, decisions, "artifact"),
        DerivedArtifact(disclosures_ref, disclosures, "artifact"),
        DerivedArtifact(budget_ref, budget, "artifact"),
        DerivedArtifact(outcome_ref, outcome.to_wire(), "artifact"),
        DerivedArtifact(external_inputs_ref, external_inputs.to_wire(), "artifact"),
        DerivedArtifact(context_content.identity(), context_content.to_wire(), "context_content"),
        DerivedArtifact(context_revision.identity(), context_revision.to_wire(), "context"),
    )
    return state, artifacts


def derive_entry(
    graph: AdmittedGraphV1,
    node_id: str,
    params: EntryParamsV1,
    reader: ArtifactReader,
) -> EnvironmentStateV1:
    """Build a node's initial checkpoint state without storage or port access."""
    if not isinstance(params, EntryParamsV1):
        raise TypeError("params must be EntryParamsV1")
    state, _ = _entry(graph, node_id, params, reader)
    return state


def derive_entry_artifacts(
    graph: AdmittedGraphV1,
    node_id: str,
    params: EntryParamsV1,
    reader: ArtifactReader,
) -> tuple[DerivedArtifact, ...]:
    """Return the deterministic entry artifacts that accompany ``derive_entry``."""
    _, artifacts = _entry(graph, node_id, params, reader)
    return artifacts


__all__ = [
    "EntryParamsV1",
    "derive_entry",
    "derive_entry_artifacts",
    "params_of",
]
