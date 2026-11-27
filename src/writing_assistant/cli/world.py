"""`wa-bench run --world` and `score --world`: playing and judging a library world.

Two commands on purpose. `run --world NAME` hands a group of each of the world's
situations to the controller (`run.control.play_groups`), which plays them into the world
and scores nothing. `score --world NAME` is the judging pass: it reads the world's closed
rollouts and adds score rows to the world file, whenever and from wherever it is run,
beside a `run --world` that is still pushing included.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from itertools import batched
from pathlib import Path

from writing_assistant.episode.library import Library
from writing_assistant.episode.segments import judging
from writing_assistant.episode.store import matches
from writing_assistant.llm.client import AccountError
from writing_assistant.run.control import Settings, Unplayed, Work, play_groups
from writing_assistant.run.directory import services_for
from writing_assistant.run.scoring import score_into
from writing_assistant.suites import SUITES, listed_rubrics


async def run_world(args: argparse.Namespace) -> int:
    """`run --world NAME --endpoint E [--cast C] --rollouts N [--round R] [--where kind=…]`.
    `--where` tests a situation's name (`task_id`) and its row's scalar fields."""
    from writing_assistant.cli import episode_row

    lib = Library()
    try:
        with lib.open(args.world) as home, home.reading():
            names = home.situations()
            if args.where:  # a row's fields are in the situation, so each is loaded: only when asked
                names = [name for name in names if matches({"task_id": name} | episode_row(home.load(name)), args.where)]
    except (LookupError, sqlite3.Error) as e:
        sys.exit(f"--world {args.world}: {e}")
    if not names:
        sys.exit(f"--world {args.world}: no situation" + (f" matches --where {args.where}" if args.where else
                                                         " in it yet (`wa-bench situate --world` sets a task up at a position)"))
    work = [Work(args.world, name, args.rollouts, args.round) for name in names]
    settings = Settings(endpoint=args.endpoint, cast=args.cast, max_tokens=args.max_tokens, concurrency=args.concurrency,
                        driver=args.driver, models=args.models, runs=Path(args.out or "runs"), held_out_test=args.held_out_test)
    try:
        groups = await play_groups(lib, work, settings)
    except Unplayed as e:
        print(f"--world {args.world}: {e}", file=sys.stderr)
        return 1
    except AccountError as e:
        print(f"stopped: the host refused the account ({e}); nothing more can run until that is fixed. "
              "Closed rollouts are in the world; rerun the same command to continue.", file=sys.stderr)
        return 1
    except ValueError as e:  # a cast that lacks a seat's name, a group asked for again with another size
        sys.exit(f"--world {args.world}: {e}")
    print(f"{args.world}: {len(groups)} groups of {args.rollouts} played ({', '.join(groups.values())}); "
          f"`wa-bench score --world {args.world}` judges them")
    return 0


async def score_world(args: argparse.Namespace) -> int:
    """`score --world NAME --judges P [--rubric R] [--pending]`: the judging pass. Each closed
    rollout of the world is scored under the rubrics its situation lists (or under `R`) that
    have no final row for this profile, and the rows are added to the world file, so a
    second pass adds nothing and a pass beside a `run --world` scores what is closed when it
    looks. `--pending`: only the rubrics whose rows all say the scorer raised.

    What is missing is found by one query over the world's tables (`Store.judged`) and one
    loaded rollout per situation, for its listing; only rollouts to be scored are loaded
    beyond that, a batch at a time. The world's lock is held shared throughout, so a
    restore waits for the pass."""
    from writing_assistant.cli import close, scored

    try:
        judging(args.judges, args.rubric or "")
    except ValueError as e:
        sys.exit(str(e))
    if args.where:
        sys.exit("--where chooses a run directory's episodes; a world is judged whole (--rubric narrows what is scored)")
    services = services_for(args.models, args.judges)
    rows: list[dict] = []
    unknown: Counter[str] = Counter()
    suites: set[str] = set()
    try:
        try:
            with Library().writing(args.world) as store:
                listed: dict[str, list[str]] = {}  # a situation's rollouts list what it lists
                want: dict[str, dict[str, list[str]]] = {}  # suite -> ep_id -> rubrics
                for ep_id, (suite, source, has) in store.judged(args.judges).items():
                    suites.add(suite)
                    rubrics = SUITES[suite].rubrics
                    if args.pending:
                        names = [r for r, final in has.items() if not final and args.rubric in (None, r)]
                    elif args.rubric is not None:
                        names = [args.rubric]
                    else:
                        if (source or ep_id) not in listed:
                            listed[source or ep_id] = listed_rubrics(SUITES[suite], store.load(ep_id))
                        names = listed[source or ep_id]
                        unknown.update(set(names) - rubrics.keys())
                    if missing := [r for r in names if r in rubrics and not has.get(r)]:
                        want.setdefault(suite, {})[ep_id] = missing
                if args.rubric is not None and not any(args.rubric in SUITES[suite].rubrics for suite in suites):
                    sys.exit(f"--rubric {args.rubric}: no suite played in {args.world} ({', '.join(sorted(suites)) or 'none yet'}) "
                             "has a rubric of that name")
                for suite, wanted in want.items():
                    for names in batched(wanted, max(4 * args.concurrency, 32)):  # a large world is never loaded whole
                        episodes = [store.load(name) for name in names]
                        rows += await score_into(store, args.judges, SUITES[suite], episodes, services, args.concurrency, wanted)
        except* AccountError as group:
            sys.exit(f"stopped: the host refused the account ({group.exceptions[0]}); the rows scored before it are in "
                     "the world, and the same command scores the rest")
        finally:
            await close(services)
    except LookupError as e:
        sys.exit(f"--world {args.world}: {e}")
    return scored(f"world {args.world}", args, ", ".join(sorted(suites)), rows, dict(unknown), services)
