"""`wa-bench library`: the worlds kept in the library (`episode/library.py`).

    library create NAME --files DIR [--split S] [--closed PATH ...]
                                                  a new world whose root holds DIR's files; a closed one is
                                                  read by no seat until a task's setup grants it (`reads`)
    library list                                  every world: name, split, what it holds
    library push --run DIR [EP_ID ...]            add a run directory's closed episodes to their world
    library verify NAME                           say what is wrong with a world file, or that nothing is
    library snapshot NAME                         a verified copy under snapshots/ (the two newest are kept)
    library restore NAME                          the newest snapshot that verifies, in the world's place

The library is the folder `WA_LIBRARY` names (default `/cold/writing-assistant/library`).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from writing_assistant.cli.common import folder_files
from writing_assistant.episode import Episode
from writing_assistant.episode.library import DEFAULT_ROOT, Damaged, Library
from writing_assistant.episode.segments import STORE_FILE, Store, has_store
from writing_assistant.episode.store import LOCAL_DEV, SPLITS


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("library", help="worlds: create, list, push, verify, snapshot, restore")
    cmd = p.add_subparsers(dest="library_cmd", required=True)
    c = cmd.add_parser("create", help="a new world from a folder of files")
    c.add_argument("name")
    c.add_argument("--files", required=True, help="a folder; every file under it becomes a file of the world's root")
    c.add_argument("--split", default="dev", choices=[*SPLITS, LOCAL_DEV], help="the world's split, which everything in it has")
    c.add_argument("--closed", nargs="+", default=[], metavar="PATH",
                   help="files of the folder (paths relative to it) no seat may read unless a task's row grants it")
    cmd.add_parser("list", help="every world in the library")
    ps = cmd.add_parser("push", help="add a run directory's closed episodes, with their score rows, to their world")
    ps.add_argument("--run", required=True)
    ps.add_argument("names", nargs="*", metavar="EP_ID", help="default: every closed episode and situation the directory lists")
    for name, text in (("verify", "check a world file: SQLite's integrity check, then every episode loads"),
                       ("snapshot", "a verified copy of a world under snapshots/"),
                       ("restore", "put the newest snapshot that verifies in a world's place; the file there is kept aside")):
        cmd.add_parser(name, help=text).add_argument("name")


def cmd_library(args: argparse.Namespace) -> int:
    lib = Library()
    try:
        return COMMANDS[args.library_cmd](lib, args)
    except (LookupError, ValueError, Damaged, sqlite3.Error) as e:
        sys.exit(f"library {args.library_cmd}: {e}")


def _create(lib: Library, args: argparse.Namespace) -> int:
    files = folder_files("--files", args.files)
    if missing := sorted(set(args.closed) - files.keys()):
        sys.exit(f"--closed {' '.join(missing)}: not under {args.files}; it holds {', '.join(files) or 'no files'}")
    name = lib.create(Episode(ep_id=args.name, task_id=args.name, suite="root", split=args.split, files=files,
                              file_revs=dict.fromkeys(files, 1), state={"file_acl": {path: [] for path in args.closed}}))
    print(f"{lib.path(name)}: world {name}, split {args.split}, {len(files)} files"
          + (f", {len(args.closed)} closed" if args.closed else ""))
    return 0


def _list(lib: Library, args: argparse.Namespace) -> int:
    worlds = lib.worlds()
    if not worlds:
        print(f"{lib.root}: no worlds. WA_LIBRARY sets the library folder (default {DEFAULT_ROOT}); "
              "`wa-bench library create NAME --files DIR` makes a world")
    for world in worlds:
        with lib.open(world.name) as store:
            groups = store.groups()
            print(f"{world.name}\t{world.split}\t{len(store.situations())} situations, {len(store.names())} played episodes, "
                  f"{sum(g['played'] for g in groups.values())}/{len(groups)} groups played\t{world.path}")
    return 0


def _push(lib: Library, args: argparse.Namespace) -> int:
    if not has_store(args.run):
        sys.exit(f"--run {args.run}: no run directory with a {STORE_FILE} there")
    with Store(Path(args.run) / STORE_FILE, readonly=True) as store:
        added = lib.push(store, args.names or None)
    print(f"{args.run}: added " + ", ".join(f"{n} {what}" for what, n in added.items()) if any(added.values())
          else f"{args.run}: nothing to add")
    return 0


def _verify(lib: Library, args: argparse.Namespace) -> int:
    lib.verify(args.name)
    print(f"{lib.path(args.name)}: nothing wrong")
    return 0


def _snapshot(lib: Library, args: argparse.Namespace) -> int:
    print(lib.snapshot(args.name))
    return 0


def _restore(lib: Library, args: argparse.Namespace) -> int:
    aside = lib.restore(args.name)
    print(f"{lib.path(args.name)}: restored from its newest good snapshot; the old file is {aside}. What was added since "
          f"is gone: `wa-bench library push --run DIR` puts rollouts back, then `wa-bench score --world {args.name}`")
    return 0


COMMANDS = {"create": _create, "list": _list, "push": _push, "verify": _verify, "snapshot": _snapshot, "restore": _restore}
