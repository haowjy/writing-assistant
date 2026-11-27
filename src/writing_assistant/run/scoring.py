"""Scoring closed episodes: the judging side of a run.

Apart from play on purpose. Nothing that plays episodes or grows a world imports this
module (`tests/test_layering.py`): the module that holds `Runner`, `run/control.py` and
`master/` cannot reach a judge or a score row through it.

Score rows go into the store beside the episodes, each as soon as it is made
(`score_into`), so a scoring cut short keeps the rows it finished.
"""

from __future__ import annotations

import asyncio
import datetime as dt

from writing_assistant.episode import Episode
from writing_assistant.episode.segments import Store
from writing_assistant.episode.store import LOCAL_DEV, local_only
from writing_assistant.llm.client import AccountError
from writing_assistant.run import code_version
from writing_assistant.suites import Services, Suite, Tally


async def score_episode(suite: Suite, ep: Episode, twin: Episode | None, services: Services) -> tuple[dict, list[dict]]:
    """(score row, the judgments behind it). A row with `error` says the scorer raised,
    which is what a judge that is down leaves: the episode is pending until it has a row
    without one (`score --pending`). Every other row is final, a judge's refusal and a use
    the profile names no endpoint for (`unavailable`) included: the same profile would say
    the same again."""
    row = {"ep_id": ep.ep_id, "task_id": ep.task_id, "condition": ep.condition, "status": ep.status,
           "twin": twin.ep_id if twin else None} | ({"group": ep.group} if ep.group else {})
    if ep.status == "failed":
        return row | {"skipped": ep.failure_reason}, []
    tally = Tally()
    try:
        with local_only(ep.split == LOCAL_DEV):
            metrics = await suite.score(ep, services, tally, twin)
    except AccountError:
        raise
    except Exception as e:  # one episode's scorer failing must not stop the others
        return row | {"error": f"{type(e).__name__}: {e}"}, []
    judgments = [{"id": f"{ep.ep_id}#{n}", "scored": ep.ep_id, "ep_id": ep.ep_id} | j
                 for n, j in enumerate(tally.judgments)]
    return row | {"metrics": tally.into(metrics), "judges": [tally.stamps[use] for use in sorted(tally.stamps)]}, judgments


async def score_into(store: Store, profile: str, suite: Suite, episodes: list[Episode], services: Services,
                     concurrency: int, only: set[str] | None = None) -> list[dict]:
    """Score every episode, or the ep_ids in `only`, and add each row to the store under
    `profile` with its judgments as soon as it is made; the rows are returned. Each is
    stamped with the scoring code, the time and the judges asked. The rows an episode had
    stay in the store, and readers take the new one unless it says the scorer raised and an
    older one does not. A twin is a baseline,
    judged only through the `score()` of the episode it names: its row says whose it is.
    Twins are paired over all of `episodes`; a rollout is the baseline of nothing and has
    none."""
    twins = {ep.baseline_of: ep for ep in episodes if ep.baseline_of is not None}
    sem = asyncio.Semaphore(concurrency)
    code = code_version()

    async def one(ep: Episode) -> dict:
        if ep.baseline_of is not None:
            row, judgments = {"ep_id": ep.ep_id, "task_id": ep.task_id, "condition": ep.condition, "status": ep.status,
                              "baseline_of": ep.baseline_of}, []
        else:
            async with sem:
                row, judgments = await score_episode(suite, ep, twins.get(ep.ep_id), services)
        # To the microsecond: readers take the later of two rows, and one scored again at once must be it.
        row |= {"code": code, "scored_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}
        store.put_scores(profile, [row], judgments)
        return row

    return list(await asyncio.gather(*(one(ep) for ep in episodes if only is None or ep.ep_id in only)))
