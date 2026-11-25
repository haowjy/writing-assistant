"""Suites: the `Suite` protocol, `Turn`, `Services` and the `SUITES` registry.

`play()` owns the turn schedule and calls `Turn`; cli.py binds a `Driver` to it, so suites
never import drivers. A suite package never imports another suite package.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from writing_assistant.episode import Episode

if TYPE_CHECKING:
    from writing_assistant.corpus import Corpus
    from writing_assistant.episode.store import Task
    from writing_assistant.episode.tools import ToolTable
    from writing_assistant.judges import Judges
    from writing_assistant.judges.embed import ReferenceIndex
    from writing_assistant.llm.client import LLMClient
    from writing_assistant.search.gateway import SearchGateway

# (episode, actor, message) -> reply. The binding in cli.py appends the `deliver` event
# (`Episode.deliver`) before the driver runs, so every delivery is on the record.
Turn = Callable[[Episode, str, str], Awaitable[str]]


@dataclass(frozen=True)
class Services:
    """Built once per run in cli.py; the only way a suite reaches a model, the web or the corpus."""

    llm: LLMClient
    judges: Judges
    search: SearchGateway | None  # None for suites that expose no search tools
    refs: ReferenceIndex | None  # same-split reference maps + train-only novelty index (Suite 2)
    corpus: Corpus


class Suite(Protocol):
    tools: ToolTable  # the suite's static tool table; cli.py hands it to the driver

    def tasks(self, split: str, where: str | None = None) -> Iterable[Task]: ...

    def setup(self, task: Task, services: Services) -> list[Episode]:
        """Resolve task.work_ids through services.corpus, stamp ep.state["work_ids"]; files,
        state, actors, actor_tools; fork twins here when they share no play."""
        ...

    async def play(self, eps: list[Episode], services: Services, turn: Turn) -> list[Episode]:
        """Own the schedule; may fork mid-play; return every episode to save."""
        ...

    async def score(
        self, ep: Episode, services: Services, twin: Episode | None = None
    ) -> dict[str, float]:
        """Pure over a frozen episode; a failed episode is never scored."""
        ...


from writing_assistant.suites.world import suite_1a  # noqa: E402  (last, so suite packages can import Turn and Services from here)

SUITES: dict[str, Suite] = {"1a": suite_1a}
