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
    judged_by = score.JUDGED_BY
    about = tasks.ABOUT
    metric_about = score.ABOUT
    text_fields = tasks.TEXT_FIELDS
    rubrics = {"": tuple(tasks.ABOUT)}  # one, unnamed: rows under the bare judge profile

    def tasks(self, split: str, where: str | None = None) -> Iterable[Task]:
        return task_rows("3", split, where)

    def setup(self, task: Task, services, at: Episode | None = None) -> list[Episode]:
        return tasks.setup(task, services)

    async def next_delivery(self, ep: Episode, last_reply: str) -> tuple[str, str] | None:
        return tasks.next_delivery(ep)

    async def score(self, ep: Episode, services, tallies, twin: Episode | None = None) -> dict[str, dict[str, float]]:
        return {"": await score.score(ep, services, tallies[""], twin)}


suite = Style()
