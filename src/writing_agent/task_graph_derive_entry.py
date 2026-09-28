"""Pure construction of a task-graph node's entry state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from writing_agent.task_graph import (
    CheckpointV1,
    EnvironmentStateV1,
    MessageV1,
    Phase,
    canonical_bytes,
    domain_hash,
    freeze,
    tree_hash,
    validate_hash,
)
from writing_agent.task_graph_admission import (
    AdmittedGraphV1,
    initial_requirements,
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
from writing_agent.task_graph_transition import (
    ArtifactReader,
    CheckpointChain,
    ContextView,
    DerivedArtifact,
    LineageMode,
    LineageView,
    ToolSpec,
)

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


@dataclass(frozen=True)
class EntryParamsV1:
    """Free values pinned by the caller at the start of one rollout."""

    lineage_id: str
    visit_id: str
    versions_ref: str
    provenance_ref: str
    rng_ref: str
    rendering: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.lineage_id, str) or not self.lineage_id:
            raise ValueError("lineage_id must be nonempty")
        if not isinstance(self.visit_id, str) or not self.visit_id:
            raise ValueError("visit_id must be nonempty")
        for name in ("versions_ref", "provenance_ref", "rng_ref"):
            validate_hash(getattr(self, name))
        if not isinstance(self.rendering, Mapping):
            raise TypeError("rendering pins must be an object")
        object.__setattr__(self, "rendering", freeze(self.rendering))


def params_of(state: EnvironmentStateV1, reader: ArtifactReader) -> EntryParamsV1:
    """Extract caller-controlled entry parameters from a state.

    Rendering is stored in the materialized root context, so it is read through
    the same artifact boundary used by the gate.
    """
    if not isinstance(state, EnvironmentStateV1):
        raise TypeError("state must be EnvironmentStateV1")
    return EntryParamsV1(
        lineage_id=state.position["lineage_id"],
        visit_id=state.position["visit_id"],
        versions_ref=state.versions_ref,
        provenance_ref=state.provenance_ref,
        rng_ref=state.rng_ref,
        rendering=reader.context(state.context_ref).rendering,
    )


def _entry_files(reader: ArtifactReader, files_ref: str) -> dict[str, str]:
    """Unwrap the legacy scenario compiler's visible-files artifact shape."""
    files = reader.artifact(files_ref)
    if isinstance(files, Mapping) and files.get("kind") == "legacy-visible-files":
        files = files.get("files")
    if not isinstance(files, Mapping) or any(
        not isinstance(path, str) or not isinstance(text, str) for path, text in files.items()
    ):
        raise TypeError("entry files artifact must map paths to text")
    return dict(files)


@dataclass(frozen=True)
class EntryV1:
    state: EnvironmentStateV1
    artifacts: tuple[DerivedArtifact, ...]
    view: LineageView


def derive_entry(
    graph: AdmittedGraphV1,
    node_id: str,
    params: EntryParamsV1,
    reader: ArtifactReader,
) -> EntryV1:
    """Build a node's initial state and deterministic artifacts without I/O."""
    if not isinstance(params, EntryParamsV1):
        raise TypeError("params must be EntryParamsV1")
    node = graph.node(node_id)
    if node.spec.kind != "writer":
        raise ValueError("entry state requires a writer node")
    contract = node.contract
    entry = contract.entry_contract
    files = _entry_files(reader, entry.files_ref)

    request = reader.artifact(entry.request_ref)
    if not isinstance(request, Mapping) or not isinstance(request.get("text"), str):
        raise TypeError("entry request artifact must contain text")
    request_text = request["text"]
    rendering = params.rendering
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

    requirements = {
        "record_type": "RequirementLedgerV1",
        "schema": 1,
        "active": initial_requirements(node, reader),
        "superseded": {},
    }
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
    if contract.budget_contract.max_generated_tokens is not None:
        budget_limits["generated_tokens"] = contract.budget_contract.max_generated_tokens
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
        DerivedArtifact(
            requirements_ref, canonical_bytes(requirements), "private", "canonical_json"
        ),
        DerivedArtifact(decisions_ref, canonical_bytes(decisions), "artifact", "canonical_json"),
        DerivedArtifact(
            disclosures_ref, canonical_bytes(disclosures), "artifact", "canonical_json"
        ),
        DerivedArtifact(budget_ref, canonical_bytes(budget), "artifact", "canonical_json"),
        DerivedArtifact(outcome_ref, outcome, "artifact", "record"),
        DerivedArtifact(external_inputs_ref, external_inputs, "artifact", "record"),
        DerivedArtifact(context_content.identity(), context_content, "context_node", "record"),
        DerivedArtifact(
            context_revision.identity(), context_revision, "context_revision", "record"
        ),
    )
    checkpoint_id = CheckpointV1(state=state, event_head=None).identity()
    context = ContextView(
        messages=messages,
        sources=tuple(None for _ in messages),
        tools=tools,
        rendering=rendering,
        content_ref=context_content.identity(),
        revision_ref=context_revision.identity(),
    )
    versions = reader.artifact(params.versions_ref)
    if not isinstance(versions, Mapping) or not isinstance(versions.get("tool_spec"), Mapping):
        raise TypeError("entry execution versions must contain a tool_spec object")
    view = LineageView(
        root_checkpoint_id=checkpoint_id,
        checkpoint_id=checkpoint_id,
        head_event_id=None,
        state=state,
        budget=budget,
        outcome=outcome,
        check_statuses={},
        context=context,
        raw_call_ids=frozenset(),
        call_sources={},
        samples=(),
        ancestry=CheckpointChain(checkpoint_id, context),
        node=node,
        mode=LineageMode.for_node(node),
        tool_spec=ToolSpec(**dict(versions["tool_spec"])),
    )
    return EntryV1(state, artifacts, view)


__all__ = [
    "EntryParamsV1",
    "EntryV1",
    "derive_entry",
    "params_of",
]
