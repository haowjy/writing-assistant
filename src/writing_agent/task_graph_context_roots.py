"""Read-only event ancestry checks for task-graph context roots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from writing_agent.task_graph_errors import ProjectionError


def context_root_changed_after(
    reader: Any,
    head_event_id: str | None,
    sample_event_id: str | None,
) -> bool:
    """Return whether a context event intervenes after a sample or rollout start.

    With a sample event ID, the walk must reach that event. With ``None``, it scans
    through the active lineage's ``rollout_started`` event. Both boundaries are
    fail-closed when absent or outside the active ancestry.
    """
    current = head_event_id
    boundary_found = False
    changed = False
    seen: set[str] = set()

    while current is not None:
        if not isinstance(current, str):
            raise ProjectionError("event.previous: event reference is invalid")
        if current in seen:
            raise ProjectionError("event.previous: lineage event history contains a cycle")
        seen.add(current)

        if sample_event_id is not None and current == sample_event_id:
            boundary_found = True
            break

        try:
            event = reader.artifact(current, domain="event")
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectionError("event.previous: lineage event history is unavailable") from exc
        if not isinstance(event, Mapping):
            raise ProjectionError("event.previous: lineage event is not a record")
        changed |= event.get("kind") == "context_changed"
        if sample_event_id is None and event.get("kind") == "rollout_started":
            boundary_found = True
            break
        previous = event.get("previous")
        if previous is not None and not isinstance(previous, str):
            raise ProjectionError("event.previous: event reference is invalid")
        current = previous

    if not boundary_found:
        boundary = "sample" if sample_event_id is not None else "rollout start"
        raise ProjectionError(f"event.previous: {boundary} is outside the active event ancestry")
    return changed


__all__ = ["context_root_changed_after"]
