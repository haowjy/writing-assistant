"""Shared constructors and adapters for pure transition derives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from writing_agent.task_graph import (
    CheckpointV1,
    EventV1,
    MessageV1,
    Record,
    canonical_bytes,
    domain_hash,
    load_canonical_json,
    thaw,
    tree_hash,
)
from writing_agent.task_graph import (
    ContextRevisionV1 as BudgetContextRevisionV1,
)
from writing_agent.task_graph_accounting import charge_context_append
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_records import (
    RECORD_TYPES,
    ContextContentV1,
    ContextRevisionV1,
)
from writing_agent.task_graph_transition import (
    ArtifactReader,
    CheckpointChain,
    ContextView,
    DerivedArtifact,
    LineageView,
    Transition,
)
from writing_agent.task_graph_wire import WireRecord

DeriveKey = str | tuple[str, str]


def new_event(
    view: LineageView,
    payload_ref: str,
    *,
    kind: str,
    actor: str,
    audience: tuple[str, ...],
    lineage_id: str | None = None,
) -> EventV1:
    """Build the event for the current head, with an optional member lineage override."""
    if view.head_event_id != view.state.history["head"]:
        raise ProjectionError("lineage view event head differs from its state")
    if view.context.revision_ref != view.state.context_ref:
        raise ProjectionError("lineage view context differs from its state")
    identity = view.state.position["lineage_id"] if lineage_id is None else lineage_id
    return EventV1(
        previous=view.head_event_id,
        seq=view.state.history["seq"] + 1,
        lineage_id=identity,
        rollout_id=identity,
        node_visit_id=view.state.position["visit_id"],
        kind=kind,
        actor=actor,
        audience=audience,
        payload_ref=payload_ref,
        versions_ref=view.state.versions_ref,
        provenance_ref=view.state.provenance_ref,
    )


def next_state(view: LineageView, event: EventV1, **changes: Any):
    """Apply state changes and centrally advance history and the file-tree identity."""
    body = view.state.to_dict()
    history = dict(body["history"])
    history.update(thaw(changes.pop("history", {})))
    history.update(thaw(changes.pop("history_changes", {})))
    history.update(head=event.id, seq=event.seq)
    for field in ("position", "continuation"):
        if field in changes:
            body[field] = {**body[field], **thaw(changes.pop(field))}
    body.update(thaw(changes))
    body["history"] = history
    body["tree_hash"] = tree_hash(body["files"])
    return type(view.state).from_dict(body)


def append_context(
    view: LineageView, messages: tuple[MessageV1, ...], event_id: str
) -> tuple[ContextView, DerivedArtifact, DerivedArtifact, dict[str, Any] | None]:
    """Append visible messages, derive their chained records, and charge existing policy."""
    content = ContextContentV1(
        parent_ref=view.context.content_ref,
        messages=messages,
        tools=None,
        rendering=None,
    )
    revision = ContextRevisionV1(
        content_ref=content.identity(), event_head=event_id, provenance_refs=(event_id,)
    )
    context = ContextView(
        messages=(*view.context.messages, *messages),
        sources=(*view.context.sources, *((event_id,) * len(messages))),
        tools=view.context.tools,
        rendering=view.context.rendering,
        content_ref=content.identity(),
        revision_ref=revision.identity(),
    )
    charged = charge_context_append(
        dict(view.budget),
        BudgetContextRevisionV1(
            messages=context.messages,
            tools=context.tools,
            rendering=context.rendering,
            event_head=event_id,
            provenance_refs=(event_id,),
        ),
    )
    return (
        context,
        payload_artifact(content, "context_node"),
        payload_artifact(revision, "context_revision"),
        charged,
    )


def advance(
    view: LineageView,
    input_record: Any,
    event: EventV1,
    state,
    *,
    artifacts: tuple[DerivedArtifact, ...],
    **view_changes: Any,
) -> Transition:
    """Build the runtime checkpoint and successor view (I5: checkpoints carry no refs)."""
    checkpoint = CheckpointV1(parents=(view.checkpoint_id,), state=state, event_head=event.id)
    context = view_changes.pop("context", view.context)
    budget = view_changes.pop(
        "budget", _changed_value(state.budgets_ref, view.state.budgets_ref, view.budget, artifacts)
    )
    outcome = view_changes.pop(
        "outcome",
        _changed_value(state.outcome_ref, view.state.outcome_ref, view.outcome, artifacts),
    )
    successor = replace(
        view,
        checkpoint_id=checkpoint.identity(),
        head_event_id=event.id,
        state=state,
        budget=budget,
        outcome=outcome,
        context=context,
        ancestry=CheckpointChain(checkpoint.identity(), context, view.ancestry),
        **view_changes,
    )
    return Transition(event, input_record, state, artifacts, successor)


def build_transition(
    view: LineageView,
    input_record: Any,
    *,
    kind: str,
    actor: str,
    audience: tuple[str, ...],
    state_changes: Mapping[str, Any],
    artifacts: tuple[DerivedArtifact, ...] = (),
    include_input: bool = True,
    lineage_id: str | None = None,
    **view_changes: Any,
) -> Transition:
    """Apply a record-driven step without repeating event/state/checkpoint wiring."""
    input_artifact = payload_artifact(input_record)
    event = new_event(
        view,
        input_artifact.ref,
        kind=kind,
        actor=actor,
        audience=audience,
        lineage_id=lineage_id,
    )
    state = next_state(view, event, **state_changes)
    all_artifacts = ((input_artifact,) if include_input else ()) + artifacts
    return advance(view, input_record, event, state, artifacts=all_artifacts, **view_changes)


def payload_artifact(value: Any, kind: str = "artifact") -> DerivedArtifact:
    """Create one identity-checked derived artifact from a typed record or payload body."""
    if isinstance(value, (WireRecord, Record)):
        if kind in {"context_node", "context_revision"}:
            return DerivedArtifact(value.identity(), value, kind, "record")
        body = value.to_wire() if isinstance(value, WireRecord) else value.to_dict()
        return DerivedArtifact(
            domain_hash("payload", body), canonical_bytes(body), kind, "canonical_json"
        )
    if isinstance(value, Mapping):
        body = dict(value)
        record_type = body.get("record_type")
        if isinstance(record_type, str) and record_type in RECORD_TYPES:
            RECORD_TYPES[record_type].from_dict(body)
        return DerivedArtifact(
            domain_hash("payload", body), canonical_bytes(body), kind, "canonical_json"
        )
    raise TypeError("payload artifacts require a typed record or mapping")


def evidence_reader(
    reader: ArtifactReader,
    *,
    summary_ref: str | None = None,
    summary: str | None = None,
    packet_ref: str | None = None,
) -> _EvidenceReader:
    """Adapt both legacy compaction evidence reads and bounded evaluator packet reads."""
    return _EvidenceReader(reader, summary_ref, summary, packet_ref)


class _EvidenceReader:
    def __init__(self, reader, summary_ref, summary, packet_ref):
        self.reader = reader
        self.summary_ref = summary_ref
        self.summary = summary
        self.packet_ref = packet_ref

    def get_artifact(self, identity, **_):
        if identity == self.summary_ref and self.summary is not None:
            return self.summary.encode("utf-8")
        return self.reader.bytes_artifact(identity)

    def load_event(self, identity):
        return self.reader.artifact(identity, domain="event")

    def read_evaluator_packet(self, ref: str) -> Mapping[str, Any]:
        if ref != self.packet_ref:
            raise ProjectionError("evaluator requested an unauthorized packet")
        return self.reader.artifact(ref, private=True)


def _changed_value(ref, old_ref, old_value, artifacts):
    if ref == old_ref:
        return old_value
    for artifact in artifacts:
        if artifact.ref == ref:
            value = artifact.value
            if artifact.value_kind == "record":
                if isinstance(value, (WireRecord, Record)):
                    return value
                raise ValueError("changed view reference has no typed record")
            if artifact.value_kind == "canonical_json":
                body = load_canonical_json(value)
                if isinstance(body, Mapping):
                    record_type = body.get("record_type")
                    codec = RECORD_TYPES.get(record_type) if isinstance(record_type, str) else None
                    if codec is not None:
                        return codec.from_dict(body)
                return body
            raise ValueError("changed view reference has no decodable payload")
    raise ValueError("changed view reference has no derived artifact")
