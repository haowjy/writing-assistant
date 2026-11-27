"""Run: play episodes against endpoints, save them and close them.

What a command, and later a generation loop or a trainer, calls to get episodes played; it
parses no arguments and chooses no tasks. A caller opens the run directory and its store,
builds one `Runner` and hands it groups of episodes. Nothing here scores: judging is a
separate pass over closed episodes (`run/scoring.py`), which this module does not import
(`tests/test_layering.py`).

- A **seat** is an entry in `Episode.actors`; its value is `scripted` or a **cast name**:
  `subject` (the model under test) or any other name the task uses. The run's **cast** maps
  cast names to endpoints: `subject` is the run's endpoint, the rest come from a
  `[cast.<name>]` table of models.toml (`cast_from`). Nothing writes an endpoint into
  `actors`: `Binding` looks a seat's name up when the seat is first called, so the same
  stored episode is played by any cast, and a rollout onto another subject changes no
  other seat. The resolved cast is stamped on every episode a run binds (`meta["cast"]`:
  name -> endpoint and model).
- `Binding` is the `Turn` that `suites.play()` delivers through. The subject's seats go
  through the run's driver; every other model seat goes through `direct`, so it is the
  same whatever harness the subject is run through (its limits in `state["sampling"]`).
- `Runner.play` is the one way episodes are played: saved before play and each time a
  delivery returns, then closed together. A task's episodes are played as one
  (they close together, so a reader never sees an arm without its twin); a rollout alone.
- A **group** is the rollouts one run made of one turn checkpoint in one round
  (`group_id`, `Runner.rollouts`). Its id is derived, so a resumed run finds it again with
  nothing saved, and it is in each member's name, so two runs never make the same name.
- `situate` sets a task up at a position (a closed episode) and stores what the suite's
  `setup` returns as situations: starts, played only as rollouts (`Runner.rollouts`).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import fcntl
import functools
import hashlib
import json
import os
import subprocess
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from writing_assistant.corpus.release import works_in
from writing_assistant.drivers import DRIVERS, Driver, Session
from writing_assistant.drivers.direct import DirectDriver
from writing_assistant.episode import Episode
from writing_assistant.episode.model import norm_path
from writing_assistant.episode.segments import Store
from writing_assistant.episode.store import LOCAL_DEV, RUN_FILE, Task, append_jsonl, local_only
from writing_assistant.llm.client import AccountError
from writing_assistant.suites import SUBJECT, SUITES, Services, Suite, play

ROOT = Path(__file__).resolve().parents[3]
ERRORS_FILE = "errors.jsonl"
LOG_FILE = "log.jsonl"


def resolved(endpoints: dict[str, dict[str, Any]], cast: dict[str, str]) -> dict[str, dict[str, str]]:
    """A cast as it is recorded: each name with its endpoint and that endpoint's model, as a
    judge's stamp has them, so a comparison reads stores and no machine's models.toml. With
    them, what the endpoint's recorded thinking is when models.toml says (`thinking`:
    `trace` or `summary`): a call record does not say, and a judge that reads a turn's
    thinking needs to know which it was given."""
    return {name: {"endpoint": endpoint, "model": endpoints[endpoint]["model"]}
            | ({"thinking": endpoints[endpoint]["thinking"]} if "thinking" in endpoints[endpoint] else {})
            for name, endpoint in sorted(cast.items())}


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Log:
    """The run log: one appended JSON line per event (`at`, `pid`, `event`, then names that
    find the thing in the store), in `log.jsonl` of a run directory and, for what happens
    to a world, `<world>.log.jsonl` beside the world file. A line is written whole and at
    once, so a killed process leaves every event up to its last, and `wa-bench status`
    reads it. Events: started, opened, resumed, closed, failed, pushed, judged, finished.
    No line holds episode text: names, counts, statuses and error types only."""

    def __init__(self, *paths: Path) -> None:
        self.paths, self.began = paths, time.monotonic()

    def __call__(self, event: str, **names: Any) -> None:
        line = {"at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z", "pid": os.getpid(), "event": event} | names
        for path in self.paths:
            append_jsonl(path, [line])

    def finished(self, **names: Any) -> None:
        self("finished", seconds=round(time.monotonic() - self.began, 1), **names)


def group_id(run_id: str, round: str, source: Episode, at: int) -> str:
    """The group of the rollouts the run `run_id` makes, in `round`, of `source`'s turn
    checkpoint `at`. A twin's is its base's, so member `i` of each is the matched pair."""
    key = json.dumps([run_id, round, source.baseline_of or source.ep_id, at])
    return hashlib.sha256(key.encode()).hexdigest()[:12]


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


# -- the cast and the Turn binding ---------------------------------------------------------

def cast_from(config: dict[str, Any], name: str) -> dict[str, str]:
    """The `[cast.<name>]` table of models.toml: cast name -> endpoint, for every model seat
    that is not the subject's. `subject` is the run's endpoint and `scripted` is no model,
    so a table that defines either is refused."""
    tables = config.get("cast", {})
    if name not in tables:
        raise ValueError(f"no [cast.{name}] in models.toml; it has {sorted(tables)}")
    cast = tables[name]
    if reserved := sorted({SUBJECT, "scripted"} & cast.keys()):
        raise ValueError(f"[cast.{name}] defines {', '.join(reserved)}: `{SUBJECT}` is the run's --endpoint and "
                         "`scripted` is a seat the suite speaks for, so a cast names neither")
    if unknown := {n: e for n, e in cast.items() if e not in config["endpoints"]}:
        raise ValueError(f"[cast.{name}] names endpoints models.toml does not have: {unknown}")
    return dict(cast)


class Binding:
    """The `Turn` that `suites.play()` makes each delivery through: (episode, seat, message).
    Logs the delivery with `Episode.deliver`, then runs the seat's driver in the seat's
    session, opening one on first use of each (episode, seat); a fork has its own ep_id, so
    its own session. The session is opened on `cast[ep.actors[seat]]`: `driver` plays the
    subject's seats and `direct` every other one (two drivers only while a run can choose a
    harness for its subject)."""

    def __init__(self, cast: dict[str, str], driver: Driver, direct: Driver) -> None:
        self.cast, self.driver, self.direct = cast, driver, direct
        self.sessions: dict[tuple[str, str], tuple[Driver, Session]] = {}

    async def __call__(self, ep: Episode, actor: str, message: str) -> None:
        key = (ep.ep_id, actor)
        if key not in self.sessions:
            name = ep.actors[actor]
            driver = self.driver if name == SUBJECT else self.direct
            self.sessions[key] = driver, await driver.open(ep, actor, self.cast[name])
        ep.deliver(actor, message)
        driver, session = self.sessions[key]
        await driver.turn(session, message)

    async def close(self, ep_ids: set[str]) -> None:
        for key in [k for k in self.sessions if k[0] in ep_ids]:
            driver, session = self.sessions.pop(key)
            await driver.close(session)


# -- the runner ----------------------------------------------------------------------------

def row_meta(task: Task) -> dict[str, Any]:
    """The row's scalar fields, stamped on each episode for `--where` and `--group-by`; a
    field marked as work text is not one of them."""
    text = {f.split(".")[0].removesuffix("[]") for f in SUITES[task.suite].text_fields.get(task.kind, [])}
    scalars = {k: v for k, v in task.payload.items() if isinstance(v, str | int | float | bool) and k not in text}
    return scalars | {"kind": task.kind, "split": task.split, "work_ids": ",".join(task.work_ids)}


def situate(store: Store, services: Services, suite: str, kind: str, payload: dict[str, Any], at: str, *,
            name: str | None = None, home: Store | None = None, files: dict[str, str] | None = None) -> list[Episode]:
    """Set a task of `suite` up at the position `at` and store its episodes in `store` as
    situations; they are returned. A position is a closed episode (a rollout's end, a
    root, another situation), looked up by name as every reader finds one, so an open
    episode is not found: its files are mid-step. It is read from `home` when another store
    holds it, and its chain is then copied into `store` unlisted.

    The task row is made here: `name` (default `<root's name>/<12 random hex>`) is its
    `task_id` and the first situation's name, and its split and works are the position's.
    It is kept in each situation (`state["task"]`), and its scalar fields in `meta["row"]`
    as a played task's are.

    `files` (path -> text) are laid over the position's files in each situation: its own
    file change, at the next revision of a path the position holds and the first of a new
    one. A path closed at the position stays closed. A row may name the paths its task
    cannot be played without (`payload["requires"]`); one the situation would not hold
    is refused, since the task would be set up on another world than its author meant."""
    held = home or store
    with held.reading():
        if not held.is_closed(at):  # one lookup: a world may hold a hundred thousand names
            raise LookupError(f"{held.path}: no closed episode named {at!r} to set a task up at")
        position = held.load(at)
        world = held.root_of(at)
    task = Task(task_id=name or f"{world}/{uuid.uuid4().hex[:12]}", suite=suite, kind=kind, split=position.split,
                work_ids=sorted(works_in(position.state)), payload=payload)
    with local_only(position.split == LOCAL_DEV):
        eps = SUITES[suite].setup(task, services, position)
    if not eps or eps[0].parent != (position.seg, 0, 0):
        raise ValueError(f"suite {suite} set {kind!r} up from its row alone, not on the position's files "
                         "(`Suite.setup(..., at)` begins with `at.begin`)")
    for ep in eps:
        for path, text in (files or {}).items():
            path = norm_path(path)
            ep.files[path], ep.file_revs[path] = text, ep.file_revs.get(path, 0) + 1
        if missing := sorted(set(payload.get("requires", [])) - ep.files.keys()):
            raise ValueError(f"the row requires {', '.join(missing)}, and the files at {at} are {', '.join(sorted(ep.files))}; "
                             "`--files DIR` lays a folder's files over the position's, or --at names a position that holds them")
        ep.state["task"] = asdict(task)
        ep.meta["row"] = row_meta(task)
    if home:
        home.copy_chain(at, store)
    store.add_starts(*eps)
    return eps


def situated(ep: Episode) -> bool:
    """Whether `ep` plays a task set up at a position (`situate` keeps the row in
    `state["task"]`, and every rollout inherits it). Such a rollout is judged by the
    judging pass alone, never by the command that played it."""
    return "task" in ep.state


class Runner:
    """One run's play: a suite, the endpoint under test and the endpoints of the other cast
    names (`cast`), the store the episodes go to, and the run directory (its name, and
    where errors are recorded). It plays and closes and does nothing else.
    `run_id` is the run's identity (run.json `id`) and `round` the caller's name for this
    pass over a checkpoint (a trainer's policy step): the two name the groups this runner
    makes. `counts` adds up what happened (episode statuses, `task_errors`) for the caller
    to report."""

    def __init__(self, suite: Suite, services: Services, store: Store, run_dir: Path, *, run_id: str, endpoint: str,
                 driver: str, max_tokens: int, concurrency: int, cast: dict[str, str] | None = None,
                 round: str = "0", log: Log | None = None) -> None:
        self.suite, self.services, self.store, self.run_dir = suite, services, store, run_dir
        self.log = log or Log(run_dir / LOG_FILE)
        self.run_id, self.round = run_id, round
        self.cast = {SUBJECT: endpoint} | (cast or {})
        self.binding = Binding(self.cast, DRIVERS[driver](services.llm, suite.tools, max_tokens=max_tokens),
                               DirectDriver(services.llm, suite.tools, max_tokens=max_tokens))
        code = code_version()
        # Stamped on every episode this run binds, so a run resumed after a code change says which is which.
        self.meta = {"driver": driver, "harness": f"{driver}@{code}", "code": code, "endpoint": endpoint,
                     "model": services.llm.endpoints[endpoint]["model"], "max_tokens": max_tokens,
                     "cast": resolved(services.llm.endpoints, self.cast),
                     "run": run_dir.name, "run_id": run_id, "round": round}
        self.sem = asyncio.Semaphore(concurrency)
        self.counts: Counter[str] = Counter()

    def bind(self, eps: list[Episode]) -> list[Episode]:
        """Make episodes this run's: its stamps in `meta`, the cast that will play them
        included. A seat whose cast name this run's cast lacks is refused here, before the
        episode is saved or any seat called."""
        for ep in eps:
            for seat, name in ep.actors.items():
                if name != "scripted" and name not in self.cast:
                    raise ValueError(f"{ep.ep_id}: seat {seat!r} is played by {name!r}, and this run's cast names only "
                                     f"{', '.join(sorted(self.cast))} (--cast picks a [cast.<name>] table of models.toml)")
            ep.meta |= self.meta
        return eps

    def fresh(self, task: Task) -> list[Episode]:
        """A task row's episodes, bound to this run and stamped with the row's fields, and
        with the rubrics it lists when it lists any (`suites.listed_rubrics` reads them)."""
        with local_only(task.split == LOCAL_DEV):
            eps = self.bind(self.suite.setup(task, self.services))
        for ep in eps:
            ep.meta["row"] = row_meta(task)
            if "rubrics" in task.payload:
                ep.meta["rubrics"] = list(task.payload["rubrics"])
        return eps

    def rollouts(self, source: Episode, at: int, n: int) -> list[Episode]:
        """The members the store does not hold yet of this run's group of `n` rollouts of
        `source`, an episode loaded at its turn checkpoint `at`: bound to this run (so
        played by its cast, whatever cast played the source) and stamped with the group and
        its size, to be played each on its own. A group's size
        is fixed when it is first asked for and is per source episode (a base and its twin
        rolled out `n` times each are one group of 2n): another size under the same round
        is refused, since readers tell a whole group from a short one by it."""
        group = group_id(self.run_id, self.round, source, at)
        asked = self.store.groups(group).get(group)
        if asked and asked["size"] != n:
            raise ValueError(f"group {group} (this run's rollouts of {source.baseline_of or source.ep_id}@{at} in round "
                             f"{self.round!r}) was asked for with {asked['size']} rollouts per episode, not {n}; a "
                             "group's size is fixed, and more rollouts are another round (--round)")
        stored = set(self.store.rollouts(source.ep_id, at, group))
        members = [source.rollout(at, i, group) for i in range(n)]
        for ep in members:  # one run's rollouts of two episodes of one name share a name, never a segment
            if ep.ep_id in stored and self.store.chain(ep.ep_id)[-2:-1] != [source.seg]:
                raise ValueError(f"{self.store.path} already holds {ep.ep_id}, a rollout of another episode named "
                                 f"{source.ep_id} (from another run's store); give these rollouts their own --out")
        new = [ep for ep in members if ep.ep_id not in stored]
        for ep in new:
            # The base-* stores were written when a run overwrote its subject's seats with its
            # endpoint and listed them here: give those seats their cast name back.
            ep.actors |= dict.fromkeys(ep.meta.pop("subject_actors", []), SUBJECT)
            ep.meta |= {"group": group, "group_size": n}
        return self.bind(new)

    def unfinished(self) -> dict[str, list[Episode]]:
        """The store's open episodes as they are played on: a task's together under its
        `task_id`, a rollout alone under its own name."""
        out: dict[str, list[Episode]] = {}
        for ep in self.store.episodes(closed=False):
            out.setdefault(ep.ep_id if ep.rollout_of else ep.task_id, []).append(ep)
        return out

    async def play(self, name: str, begin: Callable[[], list[Episode]]) -> list[Episode] | None:
        """Play episodes to the end together, close them together and return them. `begin` gives them once a slot is free: a task's episodes set up there
        and then (`fresh`), or ones already made or stored, to go on with.

        If `begin` or the play raises, that is recorded under `name` in the errors file and
        counted, and None is returned: the episodes stay open for a later run to continue.
        An `AccountError` is recorded the same way and then raised."""
        async with self.sem:
            eps: list[Episode] = []
            phase, private = "setup", False
            try:
                eps = begin()
                private = any(ep.split == LOCAL_DEV for ep in eps)
                for ep in eps:
                    self.log("resumed" if ep.calls else "opened", ep_id=ep.ep_id, **({"group": ep.group} if ep.group else {}))
                with local_only(private):
                    self.store.append(*eps)
                    phase = "play"
                    await play(self.suite, eps, self._turn)
            except Exception as e:  # one play failing must not stop the run; a resume takes it up again
                self.counts["task_errors"] += 1
                append_jsonl(self.run_dir / ERRORS_FILE, [{"task_id": name, "phase": phase, "at": now(),
                                                           "error": f"{type(e).__name__}: {e}",
                                                           "trace": traceback.format_exc(limit=-4)}])
                print(f"  {name}: {phase} error: {type(e).__name__}: {str(e)[:200]}", flush=True)
                # An error's message may quote what was being read: not logged for local_dev text.
                self.log("failed", name=name, phase=phase, error=type(e).__name__ if private else f"{type(e).__name__}: {str(e)[:200]}")
                if isinstance(e, AccountError):
                    raise
                return None
            finally:
                await self.binding.close({ep.ep_id for ep in eps})
            for ep in eps:
                ep.complete()
                self.counts[ep.status] += 1
            self.store.close(*eps)
            for ep in eps:
                self.log("closed", ep_id=ep.ep_id, status=ep.status)
            print(f"  {name}: " + ", ".join(f"{ep.condition}={ep.status}" for ep in eps), flush=True)
            return eps

    async def _turn(self, ep: Episode, actor: str, message: str) -> None:
        await self.binding(ep, actor, message)
        self.store.append(ep)
