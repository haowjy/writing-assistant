"""The controller: groups of rollouts somebody wants, played into the library.

Whoever decides what is played (a benchmark pass, the master, a trainer) hands `play_groups`
a list of `Work` and gets each one's group id back once every group is played in the
library, or an error naming what is not. Where and when it is played is decided here, and
one way is built: on this machine, through a run directory per world.

It plays and nothing else. This module imports neither `judges` nor `run/scoring.py`
(`tests/test_layering.py`), its `Services` are built on the `none` judge profile, and it
reads no score row: whether a group is played is `Store.groups`.

A run directory belongs to one world, suite and run identity (endpoint, cast, token limit,
driver), and its name is made from them, so the same work asked for again after a kill
finds the directory, the run's id and so the same groups, and plays only what is missing.
Its store holds the situations as positions (unlisted, so nothing reads them as episodes
this run played) and the rollouts. Each rollout is pushed to the library as it closes, and
the closed members of the groups asked for are pushed once more on the way out: a push
cut short, or a process killed between a close and its push, is made good by the next
call for the same work. The directory is the call's alone while it runs (`run.lock` held
exclusively), so a second call for the same world and identity is refused.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from writing_assistant.episode import Episode
from writing_assistant.episode.library import Library
from writing_assistant.episode.segments import STORE_FILE, Store
from writing_assistant.llm.client import AccountError
from writing_assistant.run import Runner, cast_from, code_version, group_id, now, run_info, situated
from writing_assistant.run.directory import Refused, identity, open_run, record_usage, services_for
from writing_assistant.suites import SUITES, Services


@dataclass(frozen=True)
class Work:
    """One group somebody wants: `rollouts` rollouts of the situation named `source` in
    `world`. `round` is the caller's name for this pass over it (a trainer's policy step);
    the same work in another round is another group."""

    world: str
    source: str
    rollouts: int
    round: str = "0"


@dataclass(frozen=True)
class Settings:
    """How work is played. Data, from flags and models.toml; it names no judge. `cast` is a
    `[cast.<name>]` table of `models` (None: every model seat is the subject's). `runs` is
    where the run directories go."""

    endpoint: str
    cast: str | None
    max_tokens: int
    concurrency: int
    driver: str = "direct"
    models: Path = Path("models.toml")
    runs: Path = Path("runs")
    held_out_test: bool = False


class Unplayed(RuntimeError):
    """Groups that are not played after two passes. The message names each with its open
    members; `groups` is the work -> group id of the ones that are short."""

    def __init__(self, message: str, groups: dict[Work, str]) -> None:
        super().__init__(message)
        self.groups = groups


async def play_groups(lib: Library, work: list[Work], settings: Settings) -> dict[Work, str]:
    """Play every group of `work` into the library and return each one's group id. Members
    the library's run directory already holds closed are not played again; open ones are
    continued. A group still short after a second pass over what is missing raises
    `Unplayed`; no round is made up here, since a new one would meet the same endpoint and
    code. A world of the `test` split is refused unless `settings.held_out_test`, and
    logged with it. An account the host refuses stops everything (`AccountError`)."""
    services = services_for(settings.models, "none")
    try:
        cast = cast_from(services.llm.config, settings.cast) if settings.cast else {}
        groups: dict[Work, str] = {}
        # A world's situations may be of several suites; a run directory plays one.
        by_run: dict[tuple[str, str], dict[Work, Episode]] = {}
        for world in sorted({w.world for w in work}):
            with lib.open(world) as home, home.reading():
                for w in work:
                    if w.world != world:
                        continue
                    source = home.load(w.source, at=0)
                    if not situated(source):
                        raise Refused(f"{world}: {w.source} is not a situation (a task set up at a position, "
                                      "`wa-bench situate`); only a situation's rollouts are a group")
                    by_run.setdefault((world, source.suite), {})[w] = source
        for (world, suite), sources in by_run.items():
            split = next(iter(sources.values())).split  # the world's: every segment in it has its root's
            groups |= await _play(lib, world, suite, split, sources, cast, services, settings)
        return groups
    except LookupError as e:  # no such world, no such situation
        raise Refused(str(e)) from None
    finally:
        await services.llm.aclose()


async def _play(lib: Library, world: str, suite: str, split: str, sources: dict[Work, Episode], cast: dict[str, str],
                services: Services, settings: Settings) -> dict[Work, str]:
    """One world's work of one suite, through its run directory."""
    ident = identity(services, suite=suite, driver=settings.driver, endpoint=settings.endpoint, split=split,
                     max_tokens=settings.max_tokens, cast=cast, world=world)
    run_dir = settings.runs / f"{world}-{suite}-{hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:10]}"
    # Alone for the whole call: the open rollouts in the directory are then this call's to continue.
    opened = open_run(run_dir, ident, held_out_test=settings.held_out_test, note={"work": len(sources)}, exclusive=True)
    groups = {w: group_id(opened.run_id, w.round, source, 0) for w, source in sources.items()}
    slots = asyncio.Semaphore(settings.concurrency)  # over every round's runner together
    short: dict[Work, list[Episode]] = {}
    store = opened.store
    try:
        with run_info(run_dir) as info:
            info.setdefault("invocations", []).append({"at": now(), "code": code_version(), "work": len(sources)})
        lib.checkout(world, sorted({w.source for w in sources}), store)
        # Two passes, then a look: a member whose play raised stays open, or unstored, for the second.
        for attempt in range(3):
            if attempt:  # reopened, so what the pass before left open is loaded as a new run would load it
                store.db.close()
                store = Store(run_dir / STORE_FILE)
            # One runner per round: a runner stamps its round on what it binds.
            runners = {round: Runner(SUITES[suite], services, store, run_dir, run_id=opened.run_id, endpoint=settings.endpoint,
                                     driver=settings.driver, max_tokens=settings.max_tokens,
                                     concurrency=settings.concurrency, cast=cast, round=round)
                       for round in {w.round for w in sources}}
            short = _missing(store, sources, groups, runners)
            if not short or attempt == 2:
                break
            asked, left = sum(w.rollouts for w in sources), sum(map(len, short.values()))
            print(f"{run_dir}: {left} rollouts to play in {len(short)} groups ({asked - left} of the {asked} asked for are "
                  "closed here)", flush=True)

            async def member(runner: Runner, ep: Episode, store: Store = store) -> None:
                async with slots:
                    if await runner.play(ep.ep_id, lambda: [ep]):
                        lib.push(store, [ep.ep_id], scores=False)

            try:
                async with asyncio.TaskGroup() as tg:  # an AccountError in one cancels the rest
                    for w, eps in short.items():
                        for ep in eps:
                            tg.create_task(member(runners[w.round], ep))
            except* AccountError as stopped:
                raise stopped.exceptions[0] from None
    finally:
        try:
            # Once more, for a member closed and not pushed (this call's or a killed one's), and only
            # this call's groups: the directory keeps every rollout it ever played.
            if closed := [name for w, source in sources.items() for name in store.rollouts(source.ep_id, 0, groups[w])
                          if store.is_closed(name)]:
                lib.push(store, closed, scores=False)
            record_usage(run_dir, services)
        finally:
            store.db.close()
            opened.lock.close()
    if short:
        raise Unplayed("; ".join(f"group {groups[w]} ({w.rollouts} rollouts of {w.source}, round {w.round!r}) is short of "
                                 f"{', '.join(ep.ep_id for ep in eps)}" for w, eps in short.items())
                       + f"; each error is in {run_dir}/errors.jsonl, and the same work asked for again continues them",
                       {w: groups[w] for w in short})
    with lib.open(world) as home:  # the promise is about the library, so that is what is asked
        if unpushed := {w: g for w, g in groups.items() if not home.groups(g).get(g, {}).get("played")}:
            raise Unplayed(f"{world}: played here and not in the library: groups {', '.join(unpushed.values())}", unpushed)
    return groups


def _missing(store: Store, sources: dict[Work, Episode], groups: dict[Work, str], runners: dict[str, Runner]) -> dict[Work, list[Episode]]:
    """Each group's members that are not closed in the run's store: new ones, bound and not
    yet stored, and open ones as they were last saved."""
    out = {}
    for w, source in sources.items():
        open_ = [store.load(name) for name in store.rollouts(source.ep_id, 0, groups[w]) if not store.is_closed(name)]
        if eps := [*runners[w.round].rollouts(source, 0, w.rollouts), *open_]:
            out[w] = eps
    return out
