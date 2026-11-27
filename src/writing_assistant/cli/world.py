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
from contextlib import ExitStack
from itertools import batched
from pathlib import Path

from writing_assistant.cli.common import close, episode_row, scored, selected
from writing_assistant.episode.library import Busy, Library
from writing_assistant.episode.segments import Store
from writing_assistant.episode.store import matches
from writing_assistant.llm.client import AccountError
from writing_assistant.run import Log, code_version
from writing_assistant.run.control import Settings, Unplayed, Work, play_groups
from writing_assistant.run.directory import services_for
from writing_assistant.run.scoring import score_into
from writing_assistant.suites import SUITES


async def run_world(args: argparse.Namespace) -> int:
    """`run --world NAME --endpoint E [--cast C] --rollouts N [--round R] [--where kind=…]`.
    `--where` tests a situation's name (`task_id`) and its row's scalar fields."""
    lib = Library()
    try:
        with lib.open(args.world) as home, home.reading():
            names = every = home.situations()
            if args.where:  # a row's fields are in the situation, so each is loaded: only when asked
                names = [name for name in names if matches({"task_id": name} | episode_row(home.load(name)), args.where)]
    except (LookupError, sqlite3.Error) as e:
        sys.exit(f"--world {args.world}: {e}")
    if not names:
        sys.exit(f"--world {args.world}: no situation" + (
            f" matches --where {args.where}; it has {', '.join(every)} (--where tests `task_id`, the whole name, and the "
            "row's scalar fields)" if every else " in it yet (`wa-bench situate --world` sets a task up at a position)"))
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
    """`score --world NAME --judges P [--rubric R] [--pending] [--all]`: the judging pass. Each
    closed rollout of the world is scored under the rubrics its situation lists (or under
    `R`) where no row by the profile's present judges stands (`run.scoring.to_score`), and
    the rows are added to the world file. So a second pass adds nothing, a pass beside a
    `run --world` scores what is closed when it looks, and a prompt edit or another judge
    under the same profile name is a pass's work again.

    What is to do is found by one query over the world's tables (`Store.judged`) and one
    loaded rollout per situation, for its listing; only rollouts to be scored are loaded
    beyond that, a batch at a time. The world's lock is held shared throughout, so a
    restore waits for the pass, and a second pass under the same profile is refused while
    this one runs (`Library.judging`)."""
    if args.where:
        sys.exit("--where chooses a run directory's episodes; a world is judged whole (--rubric narrows what is scored)")
    services = services_for(args.models, args.judges)
    rows: list[dict] = []
    unknown: dict[str, int] = {}
    suites: set[str] = set()
    nothing = ""
    lib = Library()
    try:
        try:
            with ExitStack() as held:
                try:  # only the opening: a KeyError from a bug in the pass is a traceback
                    store = held.enter_context(lib.judging(args.world, args.judges))
                except (LookupError, Busy) as e:  # no such world, a pass already running
                    sys.exit(f"--world {args.world}: {e}")
                want, unknown, suites = selected(args, store, services)
                log = Log(lib.log_path(args.world))
                log("started", cmd="score --world", world=args.world, judges=args.judges, rubric=args.rubric, code=code_version(),
                    to_score=sum(map(len, want.values())))
                held.callback(log.finished)
                if not want:
                    nothing = _nothing_to_score(args, store)
                if args.rubric is not None and not any(args.rubric in SUITES[suite].rubrics for suite in suites):
                    sys.exit(f"--rubric {args.rubric}: no suite played in {args.world} ({', '.join(sorted(suites)) or 'none yet'}) "
                             "has a rubric of that name")
                for suite, wanted in want.items():
                    for names in batched(wanted, max(4 * args.concurrency, 32)):  # a large world is never loaded whole
                        episodes = [store.load(name) for name in names]
                        rows += await score_into(store, args.judges, SUITES[suite], episodes, services, args.concurrency, wanted, log)
        except* AccountError as group:
            sys.exit(f"stopped: the host refused the account ({group.exceptions[0]}); the rows scored before it are in "
                     "the world, and the same command scores the rest")
        finally:
            await close(services)
    except sqlite3.Error as e:  # a file that will not open
        sys.exit(f"--world {args.world}: {e}")
    return scored(f"world {args.world}", args, ", ".join(sorted(suites)), rows, unknown, services, nothing)


def _nothing_to_score(args: argparse.Namespace, store: Store) -> str:
    """Why a judging pass found no work, in a sentence: an empty result is otherwise read
    as a pass that did not look."""
    rollouts = len(store.judged(args.judges))
    if not rollouts:
        return (f"nothing to score: no closed rollout in the world yet. `wa-bench run --world {args.world} --endpoint E` "
                "plays its situations")
    if args.pending:
        return f"nothing to score: no row of its {rollouts} closed rollouts says the scorer raised"
    if args.rubric:
        return (f"nothing to score: each of its {rollouts} closed rollouts has a row under {args.judges}/{args.rubric} by the "
                "judges the profile names now. `--all` scores them again")
    return (f"nothing to score: each of its {rollouts} closed rollouts has a row, by the judges the profile names now, under "
            f"every rubric its situation lists (judgings in the world: {', '.join(store.judgings(args.judges)) or 'none'}). "
            "`--all` scores them again, `--rubric R` scores every rollout under one rubric")
