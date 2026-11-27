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
from pathlib import Path

from writing_assistant.episode.library import Library
from writing_assistant.episode.store import matches
from writing_assistant.llm.client import AccountError
from writing_assistant.run.control import Settings, Unplayed, Work, play_groups


async def run_world(args: argparse.Namespace) -> int:
    """`run --world NAME --endpoint E [--cast C] --rollouts N [--round R] [--where kind=…]`."""
    from writing_assistant.cli import episode_row

    lib = Library()
    try:
        with lib.open(args.world) as home, home.reading():
            names = home.situations()
            if args.where:  # a row's fields are in the situation, so each is loaded: only when asked
                names = [name for name in names if matches(episode_row(home.load(name)), args.where)]
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
