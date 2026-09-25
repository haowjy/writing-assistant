"""Writer-visible projection of a semantically replayed admitted lineage.

The private event log is never rendered wholesale. Semantic authority belongs to
``task_graph_replay``; this entry point only materializes and verifies the active
visible context from its authorized contributions.
"""

from __future__ import annotations

from writing_agent.task_graph import ContextRevisionV1, EnvironmentStateV1, EventV1
from writing_agent.task_graph_replay import (
    ProjectionError,
    execution_value,
    replay_writer_history,
    validate_result_production,
    validate_writer_effect,
)
from writing_agent.task_graph_store import TaskGraphStore

__all__ = [
    "ProjectionError",
    "execution_value",
    "project_writer_context",
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
    replay = replay_writer_history(
        store,
        base_checkpoint_id,
        target_checkpoint_id,
        candidate_events=candidate_events,
        candidate_state=candidate_state,
    )
    cursor, expected_state = replay.cursor, replay.state
    actual = store.load_context(expected_state.context_ref)
    if (
        tuple(cursor.messages) != actual.messages
        or actual.tools != cursor.baseline.tools
        or actual.rendering != cursor.baseline.rendering
    ):
        raise ProjectionError("persisted context differs from authorized event projection")
    if actual.event_head != cursor.last_source:
        raise ProjectionError("context revision lacks latest source-event provenance")
    if source_event_ids is not None:
        source_event_ids.extend(cursor.message_sources)
    return actual
