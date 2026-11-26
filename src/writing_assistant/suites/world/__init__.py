"""Suites 1a (World & Pivots) and, from Slice 2, 1b (In-World Roleplay): one SRD engine and
one event-visibility log, two scoreboards.

Twins follow the `Suite` rule in `suites/__init__.py`: `suite_1a.twins` maps a scored
condition to its twin's condition, and the twin is the child episode of that condition.
"""

from __future__ import annotations

from collections.abc import Iterable

from writing_assistant.episode import Episode
from writing_assistant.episode.store import Task, task_rows
from writing_assistant.suites.world import score_1a, srd, tasks_1a


class World1a:
    tools = srd.TOOLS
    twins = tasks_1a.TWINS
    judged_by = score_1a.JUDGED_BY
    about = tasks_1a.ABOUT
    metric_about = score_1a.ABOUT
    text_fields = tasks_1a.TEXT_FIELDS

    def tasks(self, split: str, where: str | None = None) -> Iterable[Task]:
        return task_rows("1a", split, where)

    def setup(self, task: Task, services) -> list[Episode]:
        return tasks_1a.setup(task, services)

    async def next_delivery(self, ep: Episode, last_reply: str) -> tuple[str, str] | None:
        return await tasks_1a.next_delivery(ep, last_reply)

    async def score(self, ep: Episode, services, tally, twin: Episode | None = None) -> dict[str, float]:
        return await score_1a.score(ep, services, tally, twin)


suite_1a = World1a()
