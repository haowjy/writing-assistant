"""Suites: the `Suite` protocol, `Turn`, `play`, `Services`, `Tally` and the `SUITES` registry.

A suite keeps no state outside the episode. Its `next_delivery()` is one step of the turn
schedule: given the reply to the last delivery, it does what that reply calls for and names
the next delivery. `play()` here is the only loop over it, and makes each delivery through
`Turn`; `run.Binding` binds the drivers to that, so suites never import drivers. Because the step
reads its place from the episode (`ep.state`, the log), an episode copied when a delivery
returns continues, through the same `play()`, exactly as the original would have. A suite
package never imports another suite package; a module file directly in this directory
(`script.py`, the scripted seats' table; `converse.py`, the conversation step for any number
of seats) is shared by every suite package and imports none.

- A seat (an entry in `Episode.actors`) is `scripted`, which the suite's step speaks for,
  or names who plays it by a cast name: `SUBJECT` for the model under test, any other name
  (`counterpart`, `master`, ...) for another model. `setup()` writes those names and
  nothing overwrites them: the run's cast maps each to an endpoint when the seat is first
  called (`run.Binding`), so one stored episode is played by any cast.
- Twins: the twin of an episode is the one whose `baseline_of` is that episode's `ep_id`;
  `setup()` makes it with `fork(..., baseline=True)`. An episode nothing names is scored
  without a twin. A rollout (`Episode.rollout_of`) has none and gets no `net_*`: its
  source's twin had another conversation, so the difference would not be a matched
  counterfactual. `Episode.parent` says only what an episode is stored after. A twin is a
  baseline, never scored on its own: the `score()` of the episode it names judges it once,
  writes its rate as `twin_<metric>` and subtracts it in the `net_*` metric.
- A suite's scoring is split into **rubrics**, each with its own focus and its own score
  row per episode (stored under the judging `<profile>/<rubric>`). `rubrics` names them and
  the task kinds each is run on by default; a task row may list others
  (`payload["rubrics"]`, `listed_rubrics`). Suites 1a and 3 have one, with an empty name,
  whose rows are stored under the bare profile. `score` is handed a `Tally` per rubric the
  pass wants and returns each one's metrics, so whether two rubrics share a judge call is
  the suite's choice.
- `judged_by` maps each metric to the judge use it depends on; `wa-bench report` marks a
  metric ungated until that use's human-label gate has passed. Metrics not listed are
  deterministic.
- Scoring never reads `Episode.status` (plan/benchmark-code.md §3.1). Each task kind names
  the one artefact its outcome metrics read, through `reply_output` or `file_output` here,
  and a kind that asks for a length passes that length to `reply_output` (a floor below it);
  with none, those functions raise `NoOutput` and the scorer writes no outcome metric, only
  `n_no_output.<reason>`. Bookkeeping metrics (tool calls, rejections) read the log and are
  written whenever the episode ran. A failed episode is never scored (`run.scoring.score_episode` skips it).
"""

from __future__ import annotations

import asyncio
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

# (episode, actor, message): one delivery, made and answered. `run.Binding` appends
# the `deliver` event (`Episode.deliver`) before the driver runs, so every delivery is on the
# record; the reply is read back from the episode (`Episode.last_reply`), never returned.
Turn = Callable[[Episode, str, str], Awaitable[None]]

SUBJECT = "subject"  # the cast name of the model under test: the run's endpoint


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
    tools: ToolTable  # the suite's static tool table; `run.Runner` hands it to the drivers
    judged_by: dict[str, str]  # metric -> judge use it depends on
    about: dict[str, KindAbout]  # task kind -> what it is, in plain words
    metric_about: dict[str, str]  # metric -> what a value means, in a sentence
    # task kind -> the payload fields that carry work text ("passage.text", "exemplars[].text",
    # "files.*"): what corpus/release.py hashes when a work is not permissive
    text_fields: dict[str, list[str]]
    # rubric name -> the task kinds it is run on when a task's row lists none. Every rubric
    # `score` answers to is a key; "" is the one rubric of a suite that names none.
    rubrics: dict[str, tuple[str, ...]]

    def tasks(self, split: str, where: str | None = None) -> Iterable[Task]: ...

    def setup(self, task: Task, services: Services, at: Episode | None = None) -> list[Episode]:
        """Resolve task.work_ids through services.corpus, stamp ep.state["work_ids"] and the
        task kind's sampling temperature in ep.meta["temperature"]; files, state, actors,
        actor_tools; fork every twin here: each episode is then played on its own.

        With `at` (a position: a closed episode as last saved) the task is set up on that
        episode's files: the first episode is `at.begin(...)`, and `setup` writes the task
        into it, grants the new seats the restricted files they may read (`begin` leaves
        each readable by nobody) and copies from `at.state` what the kind carries across.
        `run.situate` stores what is returned; play is then a rollout of it. A suite whose
        tasks stand on their row alone (1a, 3) is never given one."""
        ...

    async def next_delivery(self, ep: Episode, last_reply: str) -> tuple[str, str] | None:
        """One step of the schedule: do what `last_reply` (the reply to the latest delivery,
        "" before the first) calls for, then return the next `(actor, message)` to deliver,
        or None when the episode's play is over. `play()` calls it once before the first
        delivery and once after each delivery returns, never twice for one reply.

        Whatever tells the suite where it is lives in the episode, so the step may keep
        nothing between calls. What a scripted actor does (its line, an edit, a scripted game
        master's tool calls) happens inside the step; only a model actor is delivered to."""
        ...

    async def score(
        self, ep: Episode, services: Services, tallies: dict[str, Tally], twin: Episode | None = None
    ) -> dict[str, dict[str, float]]:
        """Each rubric's metrics, for the rubrics `tallies` names (one `Tally` each). Pure
        over a frozen episode; a failed episode is never scored. Every judge call goes
        through a tally, which the caller turns into counts and stored judgments: one call
        may fill several (`Tally.found`)."""
        ...


def listed_rubrics(suite: Suite, ep: Episode) -> list[str]:
    """The rubrics `ep`'s task lists: its row's `payload["rubrics"]` (a situation keeps its
    row in `state["task"]`), else the suite's defaults for its kind. Names only: a listed
    name the suite has no rubric for is for the judging pass to report."""
    task = ep.state.get("task", {})
    if (listed := task.get("payload", {}).get("rubrics")) is not None:
        return list(listed)
    kind = task.get("kind") or ep.meta.get("row", {}).get("kind")
    return [rubric for rubric, kinds in suite.rubrics.items() if kind in kinds]


async def play(suite: Suite, eps: list[Episode], turn: Turn) -> None:
    """The one loop over `Suite.next_delivery`: a fresh episode and one reopened after a
    delivery returned are played the same way, because the reply each step is given comes
    from the episode. Episodes play concurrently, each with its own sessions; they start in
    list order, so a scripted Turn that never suspends still answers them in order."""

    async def one(ep: Episode) -> None:
        while (step := await suite.next_delivery(ep, ep.last_reply())) is not None:
            await turn(ep, *step)

    await asyncio.gather(*(one(ep) for ep in eps))


def one_message(ep: Episode, actor: str, message: Callable[[Episode], str]) -> tuple[str, str] | None:
    """The step of a task kind that is one message and its reply: deliver `message(ep)`
    unless the episode already holds a delivery."""
    return None if any(e.kind == "deliver" for e in ep.events) else (actor, message(ep))


@dataclass
class Tally:
    """Judged items per metric, and the judgments behind them. `ask(metric, judge, *args)`
    returns the judge's value, or None when the judge left the item unscored, counting why;
    `into(out)` writes `n_judged.<metric>` and `n_unscored.<metric>.<reason>` beside the
    metrics, so a report can give the refused rate beside each metric. A metric with no
    judged item is absent, never 0.0. `judgments` holds every labelable item (question,
    evidence or a/b/context, the judge's stamp, answer or unscored reason, and replies);
    `about` names the episode whose text was judged when it is not the one being scored.
    `stamps` holds the stamp of every judge asked, by use: what a score row says it was
    judged by.

    A rubric that reads a whole rollout records **findings** instead (`found`): several
    from one judge call, or ones no judge was asked for. A finding is a dict with `covers`
    ("task" or {"turn": n}), `seat`, `dimension` ("output" or "thinking"), `criterion`,
    `value` (0, 1 or 2) or `unscored` (why it has none), `cites` and `evidence` (references,
    not text). Its metric is `<dimension>.<criterion>`, and `rates()` gives each one's mean
    on 0 to 1."""

    judged: Counter[str] = field(default_factory=Counter)
    unscored: Counter[tuple[str, str]] = field(default_factory=Counter)
    judgments: list[dict[str, Any]] = field(default_factory=list)
    stamps: dict[str, dict[str, str]] = field(default_factory=dict)

    async def ask(self, metric: str, judge: Judge, *args: Any, about: str | None = None) -> Any:
        # The slot is taken before the await: judgments keep the order the items were asked
        # in, not the order the judge answered, so a stored id (`<ep_id>#<n>`) names the
        # same item when the run is re-scored.
        self.stamps[judge.stamp["use"]] = judge.stamp
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

    def found(self, findings: list[dict[str, Any]], judge: Judge | None = None, replies: list[str] | None = None,
              by_seat: bool = False) -> None:
        """Record findings in the order given. With `judge`, all of them came from one call
        of it: each carries its stamp and `replies`. With none, nobody was asked: each says
        why in `unscored` (`no_thinking`, `not_applicable`). `by_seat` keeps a metric per
        seat (`<dimension>.<criterion>.<seat>`), for a rollout whose subject holds several."""
        if judge:
            self.stamps[judge.stamp["use"]] = judge.stamp
        for finding in findings:
            metric = ".".join([finding["dimension"], finding["criterion"], *([finding["seat"]] if by_seat else [])])
            self.judgments.append({"metric": metric, **finding, **({"judge": judge.stamp, "replies": replies or []} if judge else {})})
            if finding.get("unscored"):
                self.unscored[(metric, finding["unscored"])] += 1
            else:
                self.judged[metric] += 1

    def rates(self) -> dict[str, float]:
        """Each finding metric's mean value on 0 to 1; a metric with no scored finding is absent."""
        values: dict[str, list[int]] = {}
        for j in self.judgments:
            if "criterion" in j and not j.get("unscored"):
                values.setdefault(j["metric"], []).append(j["value"])
        return {metric: sum(vs) / (2 * len(vs)) for metric, vs in values.items()}

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


# Last, so suite packages can import SUBJECT, Tally and one_message from here.
from writing_assistant.suites import situations, style  # noqa: E402
from writing_assistant.suites.world import suite_1a  # noqa: E402

SUITES: dict[str, Suite] = {"1a": suite_1a, "3": style.suite, "situations": situations.suite}
