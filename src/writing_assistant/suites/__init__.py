"""Suites: the `Suite` protocol, `Turn`, `Services`, `Tally` and the `SUITES` registry.

`play()` owns the turn schedule and calls `Turn`; cli.py binds a `Driver` to it, so suites
never import drivers. A suite package never imports another suite package.

- `setup()` maps every model-driven actor to the placeholder `SUBJECT`; cli.py replaces it
  with the endpoint under test before `play()`, and its Turn binding refuses to call a
  placeholder.
- Twins: the twin of an episode is the saved episode whose `parent_id` is that episode's
  `ep_id` and whose `condition` is `suite.twins[episode.condition]`. A condition missing
  from `twins` is scored without a twin.
- `judged_by` maps each metric to the judge use it depends on; `wa-bench report` marks a
  metric ungated until that use's human-label gate has passed. Metrics not listed are
  deterministic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from writing_assistant.episode import Episode
from writing_assistant.judges import Unscored

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

SUBJECT = "subject"  # endpoint placeholder for the model under test; cli.py replaces it


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
    twins: dict[str, str]  # scored condition -> the condition of its twin (a child episode)
    judged_by: dict[str, str]  # metric -> judge use it depends on

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


T = TypeVar("T")


@dataclass
class Tally:
    """Judged items per metric. `ask` returns the judge's value, or None when the judge
    raised `Unscored`, counting why; `into(out)` writes `n_judged.<metric>` and
    `n_unscored.<metric>.<reason>` beside the metrics, so a report can give the refused rate
    beside each metric. A metric with no judged item is absent, never 0.0."""

    judged: Counter[str] = field(default_factory=Counter)
    unscored: Counter[tuple[str, str]] = field(default_factory=Counter)

    async def ask(self, metric: str, judgment: Awaitable[T]) -> T | None:
        try:
            value = await judgment
        except Unscored as e:
            self.unscored[(metric, e.reason)] += 1
            return None
        self.judged[metric] += 1
        return value

    def into(self, out: dict[str, float]) -> dict[str, float]:
        out |= {f"n_judged.{m}": float(n) for m, n in self.judged.items()}
        out |= {f"n_unscored.{m}.{r}": float(n) for (m, r), n in self.unscored.items()}
        return out


def mean(values: Iterable[Any]) -> float | None:
    """Mean of the values that are not None; None when there are none."""
    got = [float(v) for v in values if v is not None]
    return sum(got) / len(got) if got else None


# Last, so suite packages can import SUBJECT, Tally and Turn from here.
from writing_assistant.suites import style  # noqa: E402
from writing_assistant.suites.world import suite_1a  # noqa: E402

SUITES: dict[str, Suite] = {"1a": suite_1a, "3": style.suite}
