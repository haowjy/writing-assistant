"""Suites 1a (World & Pivots) and, from Slice 2, 1b (In-World Roleplay): one SRD engine and
one event-visibility log, two scoreboards.

`suite_1a.twins` names, for each scored condition, the condition of the twin `score()`
takes: the twin is the episode whose `parent_id` is the scored episode's `ep_id`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from writing_assistant.episode import Episode
from writing_assistant.episode.store import Task, load_rows
from writing_assistant.suites.world import score_1a, srd, tasks_1a

# Written by scripts/build_seed_tasks.py.
TASK_ROWS = Path("/data/jimyao/research/writing-assistant/tasks/1a.jsonl")


class World1a:
    tools = srd.TOOLS
    twins = tasks_1a.TWINS

    def tasks(self, split: str, where: str | None = None) -> Iterable[Task]:
        return load_rows(TASK_ROWS, suite="1a", split=split, where=where)

    def setup(self, task: Task, services) -> list[Episode]:
        return tasks_1a.setup(task, services)

    async def play(self, eps: list[Episode], services, turn) -> list[Episode]:
        return await tasks_1a.play(eps, services, turn)

    async def score(self, ep: Episode, services, twin: Episode | None = None) -> dict[str, float]:
        return await score_1a.score(ep, services, twin)


suite_1a = World1a()
