"""Run: play episodes against endpoints, save them, close them and score them.

What a command, and later a generation loop or a trainer, calls to get episodes played; it
parses no arguments and chooses no tasks. A caller opens the run directory and its store,
builds one `Runner` and hands it groups of episodes.

- `Binding` is the `Turn` that `suites.play()` delivers through. The subject's actors go
  through the run's driver; every other model actor goes through `direct`, so a seat the
  episode fixes (its endpoint in `actors`, its limits in `state["sampling"]`) is the same
  whatever harness the subject is run through.
- The subject's actors are the ones `setup()` left on the `SUBJECT` placeholder.
  `bind_subject` maps them to the endpoint under test and records them in
  `meta["subject_actors"]`, so a rollout played against another endpoint rebinds those
  and no other seat.
- `Runner.play` is the one way a group of episodes is played: saved before play and each
  time a delivery returns, closed together, then scored. A task's episodes are one group
  (they close together, so a reader never sees an arm without its twin); a rollout is a
  group of one.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import fcntl
import functools
import json
import os
import subprocess
import traceback
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from writing_assistant.drivers import DRIVERS, Driver, Session
from writing_assistant.drivers.direct import DirectDriver
from writing_assistant.episode import Episode
from writing_assistant.episode.segments import Store
from writing_assistant.episode.store import LOCAL_DEV, RUN_FILE, Task, append_jsonl, local_only
from writing_assistant.llm.client import AccountError
from writing_assistant.suites import SUBJECT, SUITES, Services, Suite, Tally, play

ROOT = Path(__file__).resolve().parents[3]
ERRORS_FILE = "errors.jsonl"


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@functools.cache
def code_version() -> str:
    """The commit this process runs, `-dirty` with uncommitted changes: read once, since
    the code a process loaded does not change under it."""
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@contextmanager
def run_info(run_dir: Path) -> Iterator[dict[str, Any]]:
    """run.json to read and change, every other process kept out until it is written back:
    {} when the directory has none yet, and nothing is written when the block raises or
    changes nothing. This is
    how a run's identity is claimed by one process, and how two runs sharing a directory
    do not lose each other's entries. The lock is on the directory: run.json is replaced
    whole, so a lock on the file would be on one no longer there."""
    fd = os.open(run_dir, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        path = run_dir / RUN_FILE
        was = path.read_text() if path.exists() else "{}"
        info = json.loads(was)
        yield info
        new = json.dumps(info, indent=1, ensure_ascii=False) + "\n"
        if new != was:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(new)
            tmp.replace(path)
    finally:
        os.close(fd)


# -- the subject and the Turn binding ------------------------------------------------------

def subject_actors(ep: Episode) -> list[str]:
    """The actors the model under test plays: the ones recorded when the episode was
    bound, or, before that, the ones still on the placeholder. An episode stored before
    they were recorded (the `base-*` stores) names its run's endpoint in `meta`, and no
    seat but the subject's was then mapped to an endpoint."""
    if "subject_actors" in ep.meta:
        return ep.meta["subject_actors"]
    return [a for a, e in ep.actors.items() if e in (SUBJECT, ep.meta.get("endpoint"))]


def bind_subject(eps: list[Episode], endpoint: str) -> None:
    """Map each episode's subject actors to `endpoint` and record who they are. Of an
    episode bound before (a rollout of a stored one) it rebinds those actors only."""
    for ep in eps:
        subject = subject_actors(ep)
        ep.actors |= dict.fromkeys(subject, endpoint)
        ep.meta["subject_actors"] = subject


class Binding:
    """The `Turn` that `suites.play()` makes each delivery through: (episode, actor, message).
    Logs the delivery with `Episode.deliver`, then runs the actor's driver in the actor's
    session, opening one on first use of each (episode, actor); a fork has its own ep_id, so
    its own session. `driver` plays the subject's actors and `direct` every other one.
    Refuses an actor still mapped to the `SUBJECT` placeholder or to `scripted`."""

    def __init__(self, driver: Driver, direct: Driver) -> None:
        self.driver, self.direct = driver, direct
        self.sessions: dict[tuple[str, str], tuple[Driver, Session]] = {}

    async def __call__(self, ep: Episode, actor: str, message: str) -> None:
        endpoint = ep.actors[actor]
        if endpoint in (SUBJECT, "scripted"):
            raise RuntimeError(f"{ep.ep_id}: actor {actor!r} is {endpoint!r}, not bound to an endpoint")
        key = (ep.ep_id, actor)
        if key not in self.sessions:
            driver = self.driver if actor in subject_actors(ep) else self.direct
            self.sessions[key] = driver, await driver.open(ep, actor, endpoint)
        ep.deliver(actor, message)
        driver, session = self.sessions[key]
        await driver.turn(session, message)

    async def close(self, ep_ids: set[str]) -> None:
        for key in [k for k in self.sessions if k[0] in ep_ids]:
            driver, session = self.sessions.pop(key)
            await driver.close(session)


# -- scoring -------------------------------------------------------------------------------

async def score_episode(suite: Suite, ep: Episode, twin: Episode | None, services: Services) -> tuple[dict, list[dict]]:
    """(scores row, the judgments behind it)."""
    row = {"ep_id": ep.ep_id, "task_id": ep.task_id, "condition": ep.condition, "status": ep.status,
           "twin": twin.ep_id if twin else None}
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
    return row | {"metrics": tally.into(metrics)}, judgments


async def score_all(suite: Suite, episodes: list[Episode], services: Services, concurrency: int,
                    only: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Rows for every episode, or for the ep_ids in `only`, each stamped with the scoring
    code, and all their judgments. A twin is a baseline, judged only through the `score()` of
    the episode it names: its row says whose it is. Twins are paired over all of `episodes`;
    a rollout is the baseline of nothing and has none."""
    twins = {ep.baseline_of: ep for ep in episodes if ep.baseline_of is not None}
    sem = asyncio.Semaphore(concurrency)

    async def one(ep: Episode) -> tuple[dict, list[dict]]:
        if ep.baseline_of is not None:
            return {"ep_id": ep.ep_id, "task_id": ep.task_id, "condition": ep.condition, "status": ep.status,
                    "baseline_of": ep.baseline_of}, []
        async with sem:
            return await score_episode(suite, ep, twins.get(ep.ep_id), services)

    scored = await asyncio.gather(*(one(ep) for ep in episodes if only is None or ep.ep_id in only))
    code = code_version()
    return [row | {"code": code} for row, _ in scored], [j for _, js in scored for j in js]


# -- the runner ----------------------------------------------------------------------------

def row_meta(task: Task) -> dict[str, Any]:
    """The row's scalar fields, stamped on each episode for `--where` and `--group-by`; a
    field marked as work text is not one of them."""
    text = {f.split(".")[0].removesuffix("[]") for f in SUITES[task.suite].text_fields.get(task.kind, [])}
    scalars = {k: v for k, v in task.payload.items() if isinstance(v, str | int | float | bool) and k not in text}
    return scalars | {"kind": task.kind, "split": task.split, "work_ids": ",".join(task.work_ids)}


class Runner:
    """One run's play: a suite, the endpoint under test, the store and directory the
    episodes and their rows go to. `profile` names the judges `services` was built with
    (the rows' files carry it); `score=False` plays and saves only. `counts` adds up what
    happened (episode statuses, `task_errors`, `score_errors`) for the caller to report."""

    def __init__(self, suite: Suite, services: Services, store: Store, run_dir: Path, *, endpoint: str,
                 driver: str, max_tokens: int, profile: str, concurrency: int, score: bool = True) -> None:
        self.suite, self.services, self.store, self.run_dir = suite, services, store, run_dir
        self.endpoint, self.profile, self.scoring = endpoint, profile, score
        self.binding = Binding(DRIVERS[driver](services.llm, suite.tools, max_tokens=max_tokens),
                               DirectDriver(services.llm, suite.tools, max_tokens=max_tokens))
        code = code_version()
        # Stamped on every episode this run binds, so a run resumed after a code change says which is which.
        self.meta = {"driver": driver, "harness": f"{driver}@{code}", "code": code, "endpoint": endpoint,
                     "max_tokens": max_tokens, "run": run_dir.name}
        self.sem = asyncio.Semaphore(concurrency)
        self.counts: Counter[str] = Counter()

    def bind(self, eps: list[Episode]) -> list[Episode]:
        """Make episodes this run's: the subject's actors on its endpoint, its stamps in `meta`."""
        bind_subject(eps, self.endpoint)
        for ep in eps:
            ep.meta |= self.meta
        return eps

    def fresh(self, task: Task) -> list[Episode]:
        """A task row's episodes, bound to this run and stamped with the row's fields."""
        with local_only(task.split == LOCAL_DEV):
            eps = self.bind(self.suite.setup(task, self.services))
        for ep in eps:
            ep.meta["row"] = row_meta(task)
        return eps

    async def play(self, name: str, begin: Callable[[], list[Episode]]) -> list[Episode] | None:
        """Play one group of episodes to the end, close them together and score them, and
        return them. `begin` gives the group once a slot is free: a task's episodes set up
        there and then (`fresh`), or ones already made or stored, to go on with.

        A group that raises in `begin` or in play is recorded under `name` in the errors
        file and counted, and None is returned: its episodes stay open for a later run to
        continue. An `AccountError` is recorded the same way and then raised."""
        async with self.sem:
            eps: list[Episode] = []
            phase = "setup"
            try:
                eps = begin()
                with local_only(any(ep.split == LOCAL_DEV for ep in eps)):
                    self.store.append(*eps)
                    phase = "play"
                    await play(self.suite, eps, self._turn)
            except Exception as e:  # one group failing must not stop the run; a resume takes it up again
                self.counts["task_errors"] += 1
                append_jsonl(self.run_dir / ERRORS_FILE, [{"task_id": name, "phase": phase, "at": now(),
                                                           "error": f"{type(e).__name__}: {e}",
                                                           "trace": traceback.format_exc(limit=-4)}])
                print(f"  {name}: {phase} error: {type(e).__name__}: {str(e)[:200]}", flush=True)
                if isinstance(e, AccountError):
                    raise
                return None
            finally:
                await self.binding.close({ep.ep_id for ep in eps})
            for ep in eps:
                ep.complete()
                self.counts[ep.status] += 1
            self.store.close(*eps)
            if self.scoring:
                await self.score(eps)
            print(f"  {name}: " + ", ".join(f"{ep.condition}={ep.status}" for ep in eps), flush=True)
            return eps

    async def _turn(self, ep: Episode, actor: str, message: str) -> None:
        await self.binding(ep, actor, message)
        self.store.append(ep)

    async def score(self, eps: list[Episode], only: set[str] | None = None) -> None:
        """Score closed episodes (all of them, or the ep_ids in `only`, twins paired over
        all) and add their rows and judgments to the run's files."""
        rows, judgments = await score_all(self.suite, eps, self.services, len(eps) or 1, only)
        append_jsonl(self.run_dir / f"judgments-{self.profile}.jsonl", judgments)
        append_jsonl(self.run_dir / f"scores-{self.profile}.jsonl", rows)
        self.counts["score_errors"] += sum("error" in row for row in rows)
