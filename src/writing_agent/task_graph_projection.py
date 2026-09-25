"""Writer-visible projection of a semantically replayed admitted lineage.

The private event log is never rendered wholesale. Semantic authority belongs to
``task_graph_replay``; this entry point only materializes and verifies the active
visible context from its authorized contributions.
"""

from __future__ import annotations

from writing_agent.task_graph import ContextRevisionV1, EnvironmentStateV1, EventV1
from writing_agent.task_graph_replay import (
    AuthorStage,
    ProjectionError,
    execution_value,
    replay_writer_history,
    validate_result_production,
    validate_writer_effect,
)
from writing_agent.task_graph_sampling import validate_action_trace
from writing_agent.task_graph_store import TaskGraphStore

__all__ = [
    "ProjectionError",
    "execution_value",
    "project_writer_context",
    "validate_action_trace",
    "validate_result_production",
    "validate_writer_effect",
]


def project_writer_context(
    store: TaskGraphStore,
    base_checkpoint_id: str,
    target_checkpoint_id: str,
    *,
    candidate_events: tuple[EventV1, ...] = (),
    candidate_state: EnvironmentStateV1 | None = None,
    source_event_ids: list[str | None] | None = None,
) -> ContextRevisionV1:
    cursor, expected_state = replay_writer_history(
        store,
        base_checkpoint_id,
        target_checkpoint_id,
        candidate_events=candidate_events,
        candidate_state=candidate_state,
    )
    if cursor.reply_stage in {
        AuthorStage.ACK,
        AuthorStage.DISCLOSURE,
        AuthorStage.UPDATE,
        AuthorStage.TURN,
    }:
        raise ProjectionError("author reply transaction ended before its context publication")
    actual = store.load_context(expected_state.context_ref)
    if cursor.state != expected_state:
        raise ProjectionError("semantic history does not reconstruct target state")
    if expected_state.context_ref != cursor.latest_context_ref:
        raise ProjectionError("context revision was changed outside a context event")
    if (
        tuple(cursor.messages) != actual.messages
        or actual.tools != cursor.baseline.tools
        or actual.rendering != cursor.baseline.rendering
    ):
        raise ProjectionError("persisted context differs from authorized event projection")
    if actual.event_head != cursor.last_source:
        raise ProjectionError("context revision lacks latest source-event provenance")
    if cursor.pending != [
        call["call_id"]
        for call in expected_state.continuation["tool_queue"][
            expected_state.continuation["next_call"] :
        ]
    ]:
        raise ProjectionError("pending calls differ from continuation cursor")
    if cursor.action_ids != list(expected_state.history["action_ids"]) or cursor.result_ids != list(
        expected_state.history["tool_result_ids"]
    ):
        raise ProjectionError("logical action/result history differs from visible events")
    if source_event_ids is not None:
        source_event_ids.extend(cursor.message_sources)
    return actual
