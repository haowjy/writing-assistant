"""`wa-bench library`: the worlds kept in the library (`episode/library.py`).

    library create NAME --files DIR [--split S]   a new world whose root holds DIR's files
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

from writing_assistant.episode import Episode
from writing_assistant.episode.library import Damaged, Library
from writing_assistant.episode.segments import STORE_FILE, Store, has_store
from writing_assistant.episode.store import LOCAL_DEV, SPLITS


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("library", help="worlds: create, list, push, verify, snapshot, restore")
    cmd = p.add_subparsers(dest="library_cmd", required=True)
    c = cmd.add_parser("create", help="a new world from a folder of files")
    c.add_argument("name")
    c.add_argument("--files", required=True, help="a folder; every file under it becomes a file of the world's root")
    c.add_argument("--split", default="dev", choices=[*SPLITS, LOCAL_DEV], help="the world's split, which everything in it has")
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
    folder = Path(args.files)
    if not folder.is_dir():
        sys.exit(f"--files {folder}: no folder there")
    files = {str(p.relative_to(folder)): p.read_text() for p in sorted(folder.rglob("*")) if p.is_file()}
    name = lib.create(Episode(ep_id=args.name, task_id=args.name, suite="root", split=args.split, files=files,
                              file_revs=dict.fromkeys(files, 1)))
    print(f"{lib.path(name)}: world {name}, split {args.split}, {len(files)} files")
    return 0


def _list(lib: Library, args: argparse.Namespace) -> int:
    for world in lib.worlds():
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
    print(f"{args.run}: added " + ", ".join(f"{n} {what}" for what, n in added.items())
          if added else f"{args.run}: no closed episode to push")
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
    print(f"{lib.path(args.name)}: restored from its newest snapshot that verifies; the file that was there is {aside}. "
          "Push again from the run directories that pushed since that snapshot")
    return 0


COMMANDS = {"create": _create, "list": _list, "push": _push, "verify": _verify, "snapshot": _snapshot, "restore": _restore}
