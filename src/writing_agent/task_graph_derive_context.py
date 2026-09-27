"""Pure derivations for context operations and sealed group-member starts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from writing_agent.task_graph import (
    CheckpointV1,
    EventV1,
    canonical_bytes,
    domain_hash,
)
from writing_agent.task_graph import (
    ContextRevisionV1 as MaterializedContextRevisionV1,
)
from writing_agent.task_graph_compaction import (
    make_record,
    require_quiescent,
    select_context,
    summary_hash,
)
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_records import (
    ContextContentV1,
    ContextOperationInputV1,
    ContextPolicyV1,
    ContextRevisionV1,
    GroupSpecV1,
    MemberStartV1,
)
from writing_agent.task_graph_transition import (
    ArtifactReader,
    CheckpointChain,
    ContextView,
    DerivedArtifact,
    LineageView,
    Transition,
)


class _EvidenceReader:
    """Adapt the hash-addressed reader to the compaction helper's former API."""

    def __init__(self, reader, summary_ref, summary):
        self.reader, self.summary_ref, self.summary = reader, summary_ref, summary

    def get_artifact(self, identity, **_):
        if identity == self.summary_ref:
            return self.summary.encode("utf-8")
        return self.reader.bytes_artifact(identity)

    def load_event(self, identity):
        return self.reader.artifact(identity, domain="event")


def _event(view, payload_ref, *, kind, lineage_id=None):
    identity = lineage_id or view.state.position["lineage_id"]
    return EventV1(
        previous=view.head_event_id,
        seq=view.state.history["seq"] + 1,
        lineage_id=identity,
        rollout_id=identity,
        node_visit_id=view.state.position["visit_id"],
        kind=kind,
        actor="environment",
        audience=("controller", "trainer"),
        payload_ref=payload_ref,
        versions_ref=view.state.versions_ref,
        provenance_ref=view.state.provenance_ref,
    )


def _materialized_context(context, event_head, messages=None):
    return MaterializedContextRevisionV1(
        messages=context.messages if messages is None else messages,
        tools=context.tools,
        rendering=context.rendering,
        event_head=event_head,
        provenance_refs=(event_head,) if event_head else (),
    )


def _transition(view, input_record, event, state, artifacts, context, budget, group):
    checkpoint = CheckpointV1(
        parents=(view.checkpoint_id,),
        state=state,
        event_head=event.id,
        artifact_refs=tuple(sorted(artifact.ref for artifact in artifacts)),
    )
    next_view = replace(
        view,
        checkpoint_id=checkpoint.identity(),
        head_event_id=event.id,
        state=state,
        budget=budget,
        context=context,
        ancestry=CheckpointChain(checkpoint.identity(), context, view.ancestry),
        group=group,
    )
    return Transition(event, input_record, state, artifacts, next_view)


def derive_context_operation(
    view: LineageView,
    operation: ContextOperationInputV1,
    reader: ArtifactReader,
) -> Transition:
    """Apply one caller-selected context policy at a directed quiescent boundary."""
    directive = next_step(view)
    if "context_operation" not in directive.alternatives:
        raise ProjectionError("context operation is not an accepted directive alternative")
    if not isinstance(operation, ContextOperationInputV1):
        raise ProjectionError("context operation input has the wrong record type")
    if view.group is not None and operation.policy_ref != view.group.policy["context_policy_ref"]:
        raise ProjectionError("group context policy differs from the sealed policy")

    try:
        policy = ContextPolicyV1.from_dict(reader.artifact(operation.policy_ref))
        require_quiescent(view.state)
        old = _materialized_context(view.context, view.head_event_id)
        seed = None
        seed_sources: tuple[str | None, ...] = ()
        if policy.operation == "seed":
            assert policy.seed_checkpoint_ref is not None
            seed_context = view.ancestry.context_at(policy.seed_checkpoint_ref)
            seed = _materialized_context(seed_context, None)
            seed_sources = seed_context.sources

        origin = f"{view.state.position['lineage_id']}:context:{view.state.history['seq'] + 1}"
        messages, selected_sources, summary, _ = select_context(
            old,
            view.context.sources,
            policy,
            seed=seed,
            seed_sources=seed_sources,
            summary_origin=origin,
        )
        summary_ref = summary_hash(summary)
        new_materialized = _materialized_context(view.context, view.head_event_id, messages)
        # Recompute the non-persisted evidence record through the same helper used by
        # legacy replay. Its summary bytes are deterministic derived evidence, not an
        # artifact written by this transition.
        evidence_reader = _EvidenceReader(reader, summary_ref, summary)
        _record, budget, selected_sources = make_record(
            evidence_reader,
            view.state,
            old,
            view.context.sources,
            policy,
            operation.policy_ref,
            new_materialized,
            summary_ref,
            dict(view.budget),
            seed=seed,
            seed_sources=seed_sources,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("context operation violates the view or policy") from exc

    content = ContextContentV1(
        parent_ref=None,
        messages=messages,
        tools=old.tools,
        rendering=old.rendering,
    )
    revision = ContextRevisionV1(
        content_ref=content.identity(),
        event_head=view.head_event_id,
        provenance_refs=(view.head_event_id,) if view.head_event_id else (),
    )
    budget_ref = domain_hash("payload", budget)
    input_ref = operation.identity()
    event = _event(view, input_ref, kind="context_changed")
    state = replace(
        view.state,
        context_ref=revision.identity(),
        budgets_ref=budget_ref,
        history={**view.state.history, "head": event.id, "seq": event.seq},
    )
    context = ContextView(
        messages=messages,
        sources=selected_sources,
        tools=old.tools,
        rendering=old.rendering,
        content_ref=content.identity(),
        revision_ref=revision.identity(),
    )
    artifacts = (
        DerivedArtifact(input_ref, operation, "artifact"),
        DerivedArtifact(content.identity(), content, "context_node"),
        DerivedArtifact(revision.identity(), revision, "context_revision"),
        DerivedArtifact(budget_ref, canonical_bytes(budget), "artifact"),
    )
    return _transition(view, operation, event, state, artifacts, context, budget, view.group)


def derive_member_start(
    view: LineageView,
    start: MemberStartV1,
    reader: ArtifactReader,
) -> Transition:
    """Derive the first group-member event from its sealed spec and slot."""
    directive = next_step(view)
    if directive.kind != "sample_writer":
        raise ProjectionError("member start requires an entry writer directive")
    if not isinstance(start, MemberStartV1):
        raise ProjectionError("member start input has the wrong record type")
    if view.group is not None:
        raise ProjectionError("member lineage has already started")

    try:
        spec = GroupSpecV1.from_dict(reader.artifact(start.group_spec_ref))
        if spec.identity() != start.group_spec_ref:
            raise ValueError("group spec ref does not match its canonical payload")
        if not 0 <= start.ordinal < len(spec.members):
            raise ValueError("member ordinal is outside the sealed spec")
        entry_id = spec.environment["entry_checkpoint_id"]
        if (
            view.checkpoint_id != entry_id
            or view.state.identity() != spec.environment["entry_state_hash"]
            or view.state.history["seq"] != 0
            or view.head_event_id is not None
            or view.samples
            or view.state.history["action_ids"]
            or view.state.history["tool_result_ids"]
        ):
            raise ValueError("view is not the sealed unsampled entry checkpoint")
        require_quiescent(view.state)
        member = spec.members[start.ordinal]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("member start violates the sealed group contract") from exc

    seeds = {
        "record_type": "GroupMemberSeedsV1",
        "group_id": spec.group_id,
        "member_id": member.member_id,
        "derivation": member.seed_provenance,
        "writer_seed": member.writer_seed,
        "environment_seed": member.environment_seed,
        "parent_rng_ref": view.state.rng_ref,
    }
    seeds_ref = domain_hash("payload", seeds)
    input_ref = start.identity()
    event = _event(view, input_ref, kind="rollout_started", lineage_id=member.member_id)
    state = replace(
        view.state,
        position={
            **view.state.position,
            "lineage_id": member.member_id,
            "start_checkpoint": entry_id,
        },
        rng_ref=seeds_ref,
        history={
            **view.state.history,
            "head": event.id,
            "seq": event.seq,
            "branch_base": view.head_event_id,
        },
    )
    artifacts = (
        DerivedArtifact(input_ref, start, "artifact"),
        DerivedArtifact(seeds_ref, canonical_bytes(seeds), "artifact"),
    )
    return _transition(view, start, event, state, artifacts, view.context, view.budget, spec)


DERIVES: dict[str, Callable] = {
    "ContextOperationInputV1": derive_context_operation,
    "MemberStartV1": derive_member_start,
}
