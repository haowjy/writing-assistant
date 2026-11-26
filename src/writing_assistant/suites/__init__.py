"""Suites: the `Suite` protocol, `Turn`, `Services`, `Tally` and the `SUITES` registry.

`play()` owns the turn schedule and calls `Turn`; cli.py binds a `Driver` to it, so suites
never import drivers. A suite package never imports another suite package.

- `setup()` maps every model-driven actor to the placeholder `SUBJECT`; cli.py replaces it
  with the endpoint under test before `play()`, and its Turn binding refuses to call a
  placeholder.
- Twins: the twin of an episode is the saved episode whose `parent_id` is that episode's
  `ep_id` and whose `condition` is `suite.twins[episode.condition]`. A condition missing
  from `twins` is scored without a twin. A twin is a baseline, never scored on its own: its
  parent's `score()` judges it once, writes its rate as `twin_<metric>` and subtracts it in
  the `net_*` metric.
- `judged_by` maps each metric to the judge use it depends on; `wa-bench report` marks a
  metric ungated until that use's human-label gate has passed. Metrics not listed are
  deterministic.
- Scoring never reads `Episode.status` (plan/benchmark-code.md §3.1). Each task kind names
  the one artefact its outcome metrics read, through `reply_output` or `file_output` here,
  and a kind that asks for a length passes that length to `reply_output` (a floor below it);
  with none, those functions raise `NoOutput` and the scorer writes no outcome metric, only
  `n_no_output.<reason>`. Bookkeeping metrics (tool calls, rejections) read the log and are
  written whenever the episode ran. A failed episode is never scored (cli.py skips it).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from writing_assistant.episode import Episode

if TYPE_CHECKING:
    from writing_assistant.corpus import Corpus
    from writing_assistant.episode.store import Task
    from writing_assistant.episode.tools import ToolTable
    from writing_assistant.judges import Judge, Judges
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


@dataclass(frozen=True)
class KindAbout:
    """A task kind in plain words, for a reader learning the benchmark (`wa-bench view`). It
    sits beside the kind's `setup` so a change to one is seen next to the other.
    `arms` maps each condition setup forks to what that arm is shown; `twin` says what the
    twin changes and what it controls for (None: no twin)."""

    title: str
    tests: str
    given: str
    kept: str | None
    arms: dict[str, str]
    twin: str | None = None


class Suite(Protocol):
    tools: ToolTable  # the suite's static tool table; cli.py hands it to the driver
    twins: dict[str, str]  # scored condition -> the condition of its twin (a child episode)
    judged_by: dict[str, str]  # metric -> judge use it depends on
    about: dict[str, KindAbout]  # task kind -> what it is, in plain words
    metric_about: dict[str, str]  # metric -> what a value means, in a sentence
    # task kind -> the payload fields that carry work text ("passage.text", "exemplars[].text",
    # "files.*"): what corpus/release.py hashes when a work is not permissive
    text_fields: dict[str, list[str]]

    def tasks(self, split: str, where: str | None = None) -> Iterable[Task]: ...

    def setup(self, task: Task, services: Services) -> list[Episode]:
        """Resolve task.work_ids through services.corpus, stamp ep.state["work_ids"] and the
        task kind's sampling temperature in ep.meta["temperature"]; files, state, actors,
        actor_tools; fork twins here when they share no play."""
        ...

    async def play(self, eps: list[Episode], services: Services, turn: Turn) -> list[Episode]:
        """Own the schedule; may fork mid-play; return every episode to save."""
        ...

    async def score(
        self, ep: Episode, services: Services, tally: Tally, twin: Episode | None = None
    ) -> dict[str, float]:
        """Pure over a frozen episode; a failed episode is never scored. Every judge call
        goes through `tally.ask`, which the caller turns into counts and stored judgments."""
        ...


@dataclass
class Tally:
    """Judged items per metric, and the judgments behind them. `ask(metric, judge, *args)`
    returns the judge's value, or None when the judge left the item unscored, counting why;
    `into(out)` writes `n_judged.<metric>` and `n_unscored.<metric>.<reason>` beside the
    metrics, so a report can give the refused rate beside each metric. A metric with no
    judged item is absent, never 0.0. `judgments` holds every labelable item (question,
    evidence or a/b/context, the judge's stamp, answer or unscored reason, and replies);
    `about` names the episode whose text was judged when it is not the one being scored."""

    judged: Counter[str] = field(default_factory=Counter)
    unscored: Counter[tuple[str, str]] = field(default_factory=Counter)
    judgments: list[dict[str, Any]] = field(default_factory=list)

    async def ask(self, metric: str, judge: Judge, *args: Any, about: str | None = None) -> Any:
        # The slot is taken before the await: judgments keep the order the items were asked
        # in, not the order the judge answered, so a stored id (`<ep_id>#<n>`) names the
        # same item when the run is re-scored.
        item = judge.item(*args)
        if item:
            slot = {"metric": metric, **({"ep_id": about} if about else {}), **item}
            self.judgments.append(slot)
        j = await judge.judge(*args)
        if item:
            slot |= {"model": j.value, "unscored": j.unscored, "judge": judge.stamp, "replies": j.replies}
        if j.unscored:
            self.unscored[(metric, j.unscored)] += 1
            return None
        self.judged[metric] += 1
        return j.value

    def into(self, out: dict[str, float]) -> dict[str, float]:
        out |= {f"n_judged.{m}": float(n) for m, n in self.judged.items()}
        out |= {f"n_unscored.{m}.{r}": float(n) for (m, r), n in self.unscored.items()}
        return out


def mean(values: Iterable[Any]) -> float | None:
    """Mean of the values that are not None; None when there are none."""
    got = [float(v) for v in values if v is not None]
    return sum(got) / len(got) if got else None


# -- what every suite's scorers share --------------------------------------------------------

def paragraphs(text: str) -> list[str]:
    """Paragraphs separated by blank lines (a line of only whitespace is blank), stripped,
    empty ones dropped. The one split every task and scorer uses."""
    out, cur = [], []
    for line in text.splitlines():
        if line.strip():
            cur.append(line)
        elif cur:
            out.append("\n".join(cur).strip())
            cur = []
    if cur:
        out.append("\n".join(cur).strip())
    return out


def norm(text: str) -> str:
    """Whitespace collapsed: how paragraphs are compared, so reflowing changes nothing."""
    return " ".join(text.split())


# What the metrics every prose output gets mean; `readability_*` and `overlap_*` are families.
PROSE_ABOUT = {
    "output_words": "Words in the output the kind scores.",
    "readability_": "How hard the output is to read: sentences, dependency length, stacked nouns, object relatives, rare words.",
    "overlap_": "Longest run, and share, of the output copied from what the model was shown (context) or read with tools.",
    "overlap_flagged": "1 when that overlap crosses the pre-registered copying threshold.",
}


def prose_metrics(out: dict[str, float], text: str, ep: Episode, services: Services) -> None:
    """Every prose output: its length, readability profile and overlap (with the flag)."""
    out["output_words"] = float(len(text.split()))
    if not text.strip():
        return
    out |= {f"readability_{k}": float(v) for k, v in services.judges.readability(text).items()}
    overlap = services.corpus.overlap(text, ep)
    for part, (run, coverage) in overlap.items():
        out[f"overlap_{part}_run"], out[f"overlap_{part}_coverage"] = float(run), float(coverage)
    out["overlap_flagged"] = float(services.corpus.flag(overlap))


# -- the artefact an outcome metric reads ------------------------------------------------------

class NoOutput(Exception):
    """The episode did not produce the artefact a metric reads. `reason`: "truncated" (the
    script did not run to its end, or the last call was cut off at max_tokens or the step
    limit), "refusal" (the host's stop reason), "too_short" (a reply under `FLOOR` of the
    length the task asked for), "empty_output" (nothing written, or only whitespace)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# A reply under this fraction of the length asked for is not the prose asked for: a refusal
# in any wording, or one padded with offered alternatives. Chosen from the saved live runs,
# whose 24 chapters and scenes came in at 0.82-1.04 of the asked length (bench-v0 notes).
FLOOR = 0.5


def _last_delivery(ep: Episode, actor: str) -> int:
    """Event index of the last delivery to `actor`; NoOutput("truncated") if there is none."""
    for ev in reversed(ep.events):
        if ev.kind == "deliver" and ev.actor == actor:
            return ev.idx
    raise NoOutput("truncated")


def reply_output(ep: Episode, actor: str, words: int | None = None) -> str:
    """The reply to the last delivery to `actor`, from a call that ended with `stop`, and at
    least `FLOOR * words` words long when the task asked for `words`."""
    turn = ep.events[_last_delivery(ep, actor)].data["turn_idx"]
    calls = [c for c in ep.calls if c.actor == actor and c.turn_idx == turn]
    if not calls:
        raise NoOutput("truncated")
    last = calls[-1]
    if last.finish_reason == "refusal":
        raise NoOutput("refusal")
    if last.finish_reason != "stop" or last.tool_calls:
        raise NoOutput("truncated")
    text = last.response.strip()
    if not text:
        raise NoOutput("empty_output")
    if words is not None and len(text.split()) < FLOOR * words:
        raise NoOutput("too_short")
    return text


def file_output(ep: Episode, actor: str, path: str) -> str:
    """The text of `path`, which `actor` must have written after its last delivery."""
    after = _last_delivery(ep, actor)
    if not any(e.kind == "file_write" and e.actor == actor and e.data["path"] == path and e.idx > after
               for e in ep.events):
        raise NoOutput("empty_output")
    text = ep.files[path]
    if not text.strip():
        raise NoOutput("empty_output")
    return text


# Last, so suite packages can import SUBJECT, Tally and Turn from here.
from writing_assistant.suites import style  # noqa: E402
from writing_assistant.suites.world import suite_1a  # noqa: E402

SUITES: dict[str, Suite] = {"1a": suite_1a, "3": style.suite}
