"""Compatibility CLI imports for building admitted Phase 8 probe entries."""

from writing_agent.task_graph_probe_tasks import (
    build_admitted_entry,
    load_probe_task,
    load_probe_tasks,
)

__all__ = ["build_admitted_entry", "load_probe_task", "load_probe_tasks"]
