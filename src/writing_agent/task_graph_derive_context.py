"""Pure derivations for context operations and sealed group-member starts."""

from __future__ import annotations

from collections.abc import Callable

from writing_agent.task_graph import (
    ContextRevisionV1 as MaterializedContextRevisionV1,
)
from writing_agent.task_graph import (
    domain_hash,
    thaw,
)
from writing_agent.task_graph_compaction import (
    make_record,
    require_quiescent,
    select_context,
    summary_hash,
)
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_derive_common import (
    DeriveKey,
    build_transition,
    evidence_reader,
    payload_artifact,
)
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_record_contracts import ContextPolicyV1, GroupSpecV1
from writing_agent.task_graph_records import (
    ContextContentV1,
    ContextOperationInputV1,
    ContextRevisionV1,
    MemberStartV1,
)
from writing_agent.task_graph_transition import (
    ArtifactReader,
    ContextView,
    LineageView,
    Transition,
)


def _materialized_context(context, event_head, messages=None):
    return MaterializedContextRevisionV1(
        messages=context.messages if messages is None else messages,
        tools=context.tools,
        rendering=context.rendering,
        event_head=event_head,
        provenance_refs=(event_head,) if event_head else (),
    )


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
        compaction_reader = evidence_reader(reader, summary_ref=summary_ref, summary=summary)
        _record, budget, selected_sources = make_record(
            compaction_reader,
            view.state,
            old,
            view.context.sources,
            policy,
            operation.policy_ref,
            new_materialized,
            summary_ref,
            thaw(view.budget),
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
    context = ContextView(
        messages=messages,
        sources=selected_sources,
        tools=old.tools,
        rendering=old.rendering,
        content_ref=content.identity(),
        revision_ref=revision.identity(),
    )
    artifacts = (
        payload_artifact(content, "context_node"),
        payload_artifact(revision, "context_revision"),
        payload_artifact(budget),
    )
    return build_transition(
        view,
        operation,
        kind="context_changed",
        actor="environment",
        audience=("controller", "trainer"),
        state_changes={"context_ref": revision.identity(), "budgets_ref": budget_ref},
        artifacts=artifacts,
        context=context,
        budget=budget,
        group=view.group,
    )


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
    seeds_artifact = payload_artifact(seeds)
    seeds_ref = seeds_artifact.ref
    return build_transition(
        view,
        start,
        kind="rollout_started",
        actor="environment",
        audience=("controller", "trainer"),
        lineage_id=member.member_id,
        state_changes={
            "position": {"lineage_id": member.member_id, "start_checkpoint": entry_id},
            "rng_ref": seeds_ref,
            "history": {"branch_base": view.head_event_id},
        },
        artifacts=(seeds_artifact,),
        context=view.context,
        group=spec,
    )


DERIVES: dict[DeriveKey, Callable] = {
    "ContextOperationInputV1": derive_context_operation,
    "MemberStartV1": derive_member_start,
}
