"""Scoring closed episodes: the judging side of a run.

Apart from play on purpose. Nothing that plays episodes or grows a world imports this
module (`tests/test_layering.py`): the module that holds `Runner`, `run/control.py` and
`master/` cannot reach a judge or a score row through it.

An episode has one score row per **judging**: a judge profile and one of its suite's
rubrics (`episode.segments.judging`). Rows go into the store beside the episodes, each as
soon as it is made (`score_into`), so a scoring cut short keeps the rows it finished.
"""

from __future__ import annotations

import asyncio
import datetime as dt

from writing_assistant.episode import Episode
from writing_assistant.episode.segments import Store, judging
from writing_assistant.episode.store import LOCAL_DEV, local_only
from writing_assistant.llm.client import AccountError
from writing_assistant.run import code_version
from writing_assistant.suites import Services, Suite, Tally, listed_rubrics


def wanted(suite: Suite, episodes: list[Episode], rubric: str | None = None) -> tuple[dict[str, list[str]], dict[str, int]]:
    """(`ep_id` -> the rubrics to score it under, unknown name -> how many episodes list it).
    With `rubric`, that one for every episode; with none, each episode's listed rubrics
    (`suites.listed_rubrics`). A listed name the suite has no rubric for scores nothing and
    is counted for the caller to report: `situate` takes any names."""
    want: dict[str, list[str]] = {}
    unknown: dict[str, int] = {}
    for ep in episodes:
        names = [rubric] if rubric is not None else listed_rubrics(suite, ep)
        want[ep.ep_id] = [name for name in names if name in suite.rubrics]
        for name in set(names) - suite.rubrics.keys():
            unknown[name] = unknown.get(name, 0) + 1
    return want, unknown


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
    """Score each episode `want` names under the rubrics it gives (`wanted`), and add each
    row to the store under its judging with its judgments as soon as it is made; the rows
    are returned. Each is stamped with the scoring code, the time and the judges asked. The
    rows an episode had stay in the store, and readers take the new one unless it says the
    scorer raised and an older one does not. A twin is a baseline, judged only through the
    `score()` of the episode it names: its row says whose it is. Twins are paired over all
    of `episodes`; a rollout is the baseline of nothing and has none."""
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
