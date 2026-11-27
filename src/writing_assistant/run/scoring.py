"""Scoring closed episodes: the judging side of a run.

Apart from play on purpose. Nothing that plays episodes or grows a world imports this
module (`tests/test_layering.py`): the module that holds `Runner`, `run/control.py` and
`master/` cannot reach a judge or a score row through it.

An episode has one score row per **judging**: a judge profile and one of its suite's
rubrics (`episode.segments.judging`). `to_score` is the one place that decides what a
scoring has to do, whichever command asks. Rows go into the store beside the episodes, each as
soon as it is made (`score_into`), so a scoring cut short keeps the rows it finished.
"""

from __future__ import annotations

import asyncio
import datetime as dt

from writing_assistant.episode import Episode
from writing_assistant.episode.segments import Store, judging
from writing_assistant.episode.store import LOCAL_DEV, local_only
from writing_assistant.llm.client import AccountError
from writing_assistant.judges import Judges
from writing_assistant.run import code_version
from writing_assistant.run.directory import Refused
from writing_assistant.suites import SUITES, Services, Suite, Tally, listed_rubrics


def to_score(store: Store, profile: str, judges: Judges, *, rubric: str | None = None, pending: bool = False,
             again: bool = False, only: set[str] | None = None,
             loaded: dict[str, Episode] | None = None) -> tuple[dict[str, dict[str, list[str]]], dict[str, int], set[str]]:
    """What a scoring of `store`'s closed episodes under `profile` has to do, for `score`
    on a run directory and on a world alike: (suite -> `ep_id` -> the rubrics to score it
    under, unknown listed name -> how many episodes list it, the suites seen).

    An episode's rubrics are the ones its task lists (`suites.listed_rubrics`), or `rubric`
    for every episode. One is scored when its row does not **stand**: there is none, it says
    the scorer raised, or a judge it was scored by is not the one `profile` names now
    (`judges.stamps()`: use, endpoint, model, prompt version). So an edited prompt, another
    endpoint under the same profile name, and a row left by a pass that had no judge for it
    are each scored again, once; a row no judge was asked for stands. `again` scores
    every listed rubric whatever stands. `pending` takes only the rubrics whose row says
    the scorer raised. `only` keeps to those episodes.

    A listed name the suite has no rubric for is counted and scores nothing: `situate`
    takes any names. A rubric that needs a judge the profile names no endpoint for is
    refused here, before anything is scored (`refuse`). The listing is read from one
    loaded episode per source (`loaded`, or the store), since rollouts list what their
    source lists."""
    current = judges.stamps()
    want: dict[str, dict[str, list[str]]] = {}
    unknown: dict[str, int] = {}
    listed: dict[str, list[str]] = {}
    suites: set[str] = set()
    for ep_id, (name, source, has) in store.judged(profile).items():
        if only is not None and ep_id not in only:
            continue
        suite = SUITES[name]
        suites.add(name)
        if pending:
            names = [r for r, stamps in has.items() if stamps is None and rubric in (None, r)]
        elif rubric is not None:
            names = [rubric]
        else:
            if (source or ep_id) not in listed:
                listed[source or ep_id] = listed_rubrics(suite, loaded[ep_id] if loaded else store.load(ep_id))
            names = listed[source or ep_id]
            for r in set(names) - suite.rubrics.keys():
                unknown[r] = unknown.get(r, 0) + 1
        stale = [r for r in names if r in suite.rubrics
                 and (again or pending or has.get(r) is None or any(current.get(s["use"]) != s for s in has[r]))]
        if stale:
            want.setdefault(name, {})[ep_id] = stale
    for name, wanted in want.items():
        refuse(SUITES[name], {r for rubrics in wanted.values() for r in rubrics}, judges, profile)
    return want, unknown, suites


def refuse(suite: Suite, rubrics: set[str], judges: Judges, profile: str) -> None:
    """Raise `Refused` when one of `rubrics` needs a judge use (`Suite.needs`) that
    `profile` names no endpoint for. Scored anyway, such a rubric would get a row in which
    nothing is scored, and that row would read as the rollout having been judged."""
    current = judges.stamps()
    if missing := sorted((r, use) for r in rubrics for use in suite.needs.get(r, ()) if use not in current):
        raise Refused("; ".join(f"rubric {r!r} is scored by the judge use {use!r}, and [judges.{profile}] in models.toml names "
                                "no endpoint for it" for r, use in missing)
                      + ". Nothing was scored. Name an endpoint there, or pass --rubric for one that needs none")


async def score_episode(suite: Suite, ep: Episode, twin: Episode | None, services: Services,
                        rubrics: list[str]) -> dict[str, tuple[dict, list[dict]]]:
    """rubric -> (score row, the judgments behind it), for each of `rubrics`. A row with
    `error` says the scorer raised, which is what a judge that is down leaves: the episode
    is pending under that judging until it has a row without one (`score --pending`). The
    scorer raising leaves every rubric asked for with such a row, since one call may serve
    several. Every other row is final, a judge's refusal and a use the profile names no
    endpoint for (`unavailable`) included: the same profile would say the same again."""
    base = {"ep_id": ep.ep_id, "task_id": ep.task_id, "condition": ep.condition, "status": ep.status,
            "twin": twin.ep_id if twin else None} | ({"group": ep.group} if ep.group else {})
    rows = {rubric: base | ({"rubric": rubric} if rubric else {}) for rubric in rubrics}
    if ep.status == "failed":
        return {rubric: (row | {"skipped": ep.failure_reason}, []) for rubric, row in rows.items()}
    tallies = {rubric: Tally() for rubric in rubrics}
    try:
        with local_only(ep.split == LOCAL_DEV):
            metrics = await suite.score(ep, services, tallies, twin)
    except AccountError:
        raise
    except Exception as e:  # one episode's scorer failing must not stop the others
        return {rubric: (row | {"error": f"{type(e).__name__}: {e}"}, []) for rubric, row in rows.items()}
    out = {}
    for rubric, tally in tallies.items():
        judgments = [{"id": f"{ep.ep_id}#{n}", "scored": ep.ep_id, "ep_id": ep.ep_id} | j for n, j in enumerate(tally.judgments)]
        out[rubric] = rows[rubric] | {"metrics": tally.into(metrics[rubric]),
                                      "judges": [tally.stamps[use] for use in sorted(tally.stamps)]}, judgments
    return out


async def score_into(store: Store, profile: str, suite: Suite, episodes: list[Episode], services: Services,
                     concurrency: int, want: dict[str, list[str]]) -> list[dict]:
    """Score each episode `want` names under the rubrics it gives (`to_score`), and add each
    row to the store under its judging with its judgments as soon as it is made; the rows
    are returned. Each is stamped with the scoring code, the time and the judges asked. The
    rows an episode had stay in the store, and readers take the new one unless it says the
    scorer raised and an older one does not. A twin is a baseline, judged only through the
    `score()` of the episode it names: its row says whose it is. Twins are paired over all
    of `episodes`; a rollout is the baseline of nothing and has none."""
    refuse(suite, {r for rubrics in want.values() for r in rubrics}, services.judges, profile)  # the one writer of rows
    twins = {ep.baseline_of: ep for ep in episodes if ep.baseline_of is not None}
    sem = asyncio.Semaphore(concurrency)
    code = code_version()

    async def one(ep: Episode, rubrics: list[str]) -> list[dict]:
        if ep.baseline_of is not None:
            scored = {rubric: ({"ep_id": ep.ep_id, "task_id": ep.task_id, "condition": ep.condition, "status": ep.status,
                                "baseline_of": ep.baseline_of} | ({"rubric": rubric} if rubric else {}), []) for rubric in rubrics}
        else:
            async with sem:
                scored = await score_episode(suite, ep, twins.get(ep.ep_id), services, rubrics)
        # To the microsecond: readers take the later of two rows, and one scored again at once must be it.
        stamp = {"code": code, "scored_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}
        for rubric, (row, judgments) in scored.items():
            row |= stamp
            store.put_scores(judging(profile, rubric), [row], judgments)
        return [row for row, _ in scored.values()]

    rows = await asyncio.gather(*(one(ep, want[ep.ep_id]) for ep in episodes if want.get(ep.ep_id)))
    return [row for per_episode in rows for row in per_episode]
