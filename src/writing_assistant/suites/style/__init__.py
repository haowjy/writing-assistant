"""Suite 3 (Writing Style & Craft): in-context voice from `k` exemplars of an unseen author,
and a scoped line edit answering one craft objection. Task rows come from
scripts/build_seed_tasks.py over the corpus's within-author partition."""

from __future__ import annotations

from collections.abc import Iterable

from writing_assistant.episode import Episode
from writing_assistant.episode.store import Task, task_rows
from writing_assistant.suites.style import score, tasks


class Style:
    tools: dict = {}  # workspace tools only
    twins: dict[str, str] = {}  # no twins: each episode is scored on its own
    judged_by = score.JUDGED_BY

    def tasks(self, split: str, where: str | None = None) -> Iterable[Task]:
        return task_rows("3", split, where)

    def setup(self, task: Task, services) -> list[Episode]:
        return tasks.setup(task, services)

    async def play(self, eps: list[Episode], services, turn) -> list[Episode]:
        return await tasks.play(eps, services, turn)

    async def score(self, ep: Episode, services, twin: Episode | None = None) -> dict[str, float]:
        return await score.score(ep, services, twin)


suite = Style()
