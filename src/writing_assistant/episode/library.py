"""The library: one store file per world, in one folder, and the only ways in and out of it.

    <root>/                                 /cold/writing-assistant/library; WA_LIBRARY overrides
      worlds/<world>.db                     the world: a store (`episode/segments.py`)
      worlds/<world>.lock, <world>.master   empty files that are locked, never read
      snapshots/<world>.<utc>-<4 hex>.db    verified copies; the two newest of a world are kept

A **world** is the tree under one root episode: the root (a start with no parent, whose
`ep_id` is the world's name), the situations set up in it, every rollout any run made of
them and their score rows. Its identity is the root's segment, so any segment names its
world by its chain, and the list of worlds is the folder: there is no table to fall out of
step with the files.

**The library only gains rows.** Every segment, checkpoint and score row has an id that is
unique everywhere and a body that never changes, and only closed episodes and starts are
pushed, so nothing here is ever continued in place. Pushes therefore commute: two at once,
one repeated, or one cut short and run again all end in the same file, and correctness
rests on no lock. `restore` is the one operation that takes rows away, and the one that
keeps everyone else out.

A world file is in SQLite's write-ahead mode (set at `create` and by `restore`; the file
keeps it), so `verify`, `snapshot`, `checkout` and any other reader take one state of it
without making a push wait. It then has `-wal` and `-shm` files beside it, which go when
the last user to close it was a writer and stay, harmlessly, when it was a reader. A
snapshot is one file in rollback mode: nothing writes it again.

Two file locks per world, both dropped by the kernel when their holder dies:

- `<world>.lock` is held shared by everything that reads or writes the world file, from
  before the file is opened until after it is closed (`open`, `writing`, and through them
  `worlds`, `push`, `checkout`, `verify`, `snapshot`), and exclusively by `restore`. There
  is no way to open a world without it: a connection opened while the file is being
  replaced would read the old file through side files the restored one then inherits.
- `<world>.master` is held exclusively by a world's master while it lives (`master_lock`).
  It guards cost (two masters spending on one world), not correctness.

This module knows files, locks and the copy. It knows no suite, licence, model, split rule
or personal text: what may leave the machine is `corpus/release.py`'s to decide.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import os
import re
import sqlite3
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from writing_assistant.episode.model import Episode
from writing_assistant.episode.segments import Store, Unreadable

DEFAULT_ROOT = "/cold/writing-assistant/library"
KEPT = 2  # snapshots of a world
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")  # a world's name is a file name and the first part of its situations' names
SIDE_FILES = ("-journal", "-wal", "-shm")  # what SQLite keeps beside a file a writer has open, or died holding


class Damaged(Exception):
    """A world file or a snapshot that fails `verify`; the message says what is wrong."""


@dataclass(frozen=True)
class World:
    name: str
    split: str
    path: Path


class Library:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.environ.get("WA_LIBRARY") or DEFAULT_ROOT)
        self.worlds_dir, self.snapshots_dir = self.root / "worlds", self.root / "snapshots"

    def path(self, world: str) -> Path:
        if not NAME.fullmatch(world):
            raise ValueError(f"{world!r} is no world's name: letters, digits, '-' and '_', starting with a letter or digit")
        return self.worlds_dir / f"{world}.db"

    # -- the two locks -----------------------------------------------------------------------

    @contextmanager
    def _locked(self, world: str, exclusive: bool = False) -> Iterator[Path]:
        """The world file's path, with `<world>.lock` held (shared, or exclusively) until
        the block ends. A world that does not exist is refused before any file is made."""
        path = self.path(world)
        if not path.exists():
            raise LookupError(f"{self.root}: no world named {world!r}; it has {', '.join(w.name for w in self.worlds()) or 'none'}")
        with open(path.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield path

    @contextmanager
    def master_lock(self, world: str) -> Iterator[None]:
        """Held by the one master working on a world. A second one is refused at once."""
        path = self.path(world)
        if not path.exists():
            raise LookupError(f"{self.root}: no world named {world!r}")
        with open(path.with_suffix(".master"), "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"{world}: another master is alive on this world ({lock.name} is locked)") from None
            yield

    # -- worlds ------------------------------------------------------------------------------

    def create(self, root_ep: Episode) -> str:
        """A new world from its root episode, named by the root's `ep_id`; the name is
        returned. The file is written whole under a temporary name and then linked to its
        own, which fails when the name is taken: a listing never meets a world without its
        root, and of two creates of one name one wins."""
        path = self.path(root_ep.ep_id)
        if root_ep.parent is not None:
            raise ValueError(f"{root_ep.ep_id}: a world's root has no parent; this episode was forked from another")
        self.worlds_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.worlds_dir / f".{root_ep.ep_id}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with Store(tmp) as store:
                store.add_starts(root_ep)
                # Write-ahead mode, which the file keeps: a reader then never holds a writer up.
                # In rollback mode a push waits for a whole `verify` or `snapshot` (44 s and 55 s
                # measured on a 100,000-rollout world, against the writer's 60 s timeout).
                store.db.execute("PRAGMA journal_mode = WAL")
            try:
                os.link(tmp, path)
            except FileExistsError:
                raise ValueError(f"{self.root}: a world named {root_ep.ep_id!r} already exists") from None
        finally:
            tmp.unlink(missing_ok=True)
        return root_ep.ep_id

    def worlds(self) -> list[World]:
        """Every world, read from the files: its name, its split (the root's, which every
        segment in it has) and its path."""
        out = []
        for found in sorted(self.worlds_dir.glob("*.db")):
            with self.open(found.stem) as store:
                out.append(World(found.stem, _root(store)[1], found))
        return out

    @contextmanager
    def open(self, world: str) -> Iterator[Store]:
        """The world's store, read-only, with the world's lock held shared until the block
        ends: a restore waits for the reader, and a reader for a restore. So keep the block
        as short as the reading."""
        with self._locked(world) as path, Store(path, readonly=True) as store:
            yield store

    @contextmanager
    def writing(self, world: str) -> Iterator[Store]:
        """The world's store open for writing, with the world's lock held shared: how
        anything but a restore adds rows to it (a push, a judging pass)."""
        with self._locked(world) as path, Store(path) as store:
            yield store

    # -- in and out --------------------------------------------------------------------------

    def checkout(self, world: str, names: list[str], into: Store) -> None:
        """Copy the named episodes' chains into a worker's store, unlisted and without score
        rows: positions to fork from. Listed there they would read as episodes that store's
        run played, to be scored, reported and resumed. The world is read in one read
        transaction, so what arrives is one state of it whatever is being pushed meanwhile."""
        with self.open(world) as store, store.reading():
            for name in names:
                store.copy_chain(name, into)

    def push(self, src: Store, names: list[str] | None = None, scores: bool = True) -> Counter[str]:
        """Add closed episodes of `src` to their world, each with its chain, listing and score
        rows, in one transaction; what was added is returned (`episodes`, `segments`,
        `checkpoints`, `scores`), all zeros when the world held everything. `names` None:
        every closed episode and every start `src` lists. An open episode is refused by
        name, and so is a segment whose root is not the world's: the world is the one the
        first name's root names. `src` is read in one read transaction (a live run may be
        writing it), so the names and what is copied are one state of it. `scores=False`
        leaves the score rows behind and reads none (the controller, which only plays)."""
        with src.reading():
            names = [*src.names(), *src.starts()] if names is None else names
            if not names:
                return Counter()
            # Asked of each name, not of the whole listing: a run pushes each rollout as it closes.
            if still_open := [n for n in names if not src.is_closed(n)]:
                raise ValueError(f"{src.path}: {', '.join(still_open)} cannot be pushed: only a closed episode this store "
                                 "lists goes to the library, and nothing there is ever continued")
            roots = {name: src.chain(name)[0] for name in names}
            world = src.db.execute("SELECT ep_id FROM segments WHERE seg = ?", (roots[names[0]],)).fetchone()[0]
            with self.writing(world) as store:
                root = _root(store)[0]
                if foreign := [name for name, seg in roots.items() if seg != root]:
                    raise ValueError(f"{', '.join(foreign)}: not of the world {world!r} (another root); one push goes to "
                                     "one world")
                return src.copy(names, store, scores)

    # -- keeping a world whole ---------------------------------------------------------------

    def verify(self, world: str) -> None:
        """Raise `Damaged` saying what is wrong with the world file; return when nothing is:
        SQLite's integrity check passes and every episode it lists loads, starts included.
        What a dead writer left in the log beside the file is no damage: no reader sees it."""
        with self._locked(world) as path:
            _verify(path)

    def snapshot(self, world: str) -> Path:
        """A verified copy of the world under a new name in `snapshots/`, which is returned;
        afterwards all but the newest `KEPT` of the world are deleted. The world is verified
        first, so a file that fails never becomes a snapshot, and the copy is verified
        before it gets its name. Two snapshots at once each write their own name."""
        with self._locked(world) as path:
            _verify(path)
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.snapshots_dir / f".{world}.{uuid.uuid4().hex[:8]}.tmp"
            try:
                # SQLite's backup: one state of the file, whatever is pushed while it runs.
                with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)) as src, \
                        closing(sqlite3.connect(tmp)) as dst:
                    src.backup(dst)
                    dst.execute("PRAGMA journal_mode = DELETE")  # the copy took the world's write-ahead mode with its header
                _verify(tmp)
                taken = self.snapshots_dir / f"{world}.{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:4]}.db"
                os.link(tmp, taken)
            finally:
                tmp.unlink(missing_ok=True)
            for side in SIDE_FILES:  # none is expected; one left by a reader of the temporary file would be litter
                tmp.with_name(tmp.name + side).unlink(missing_ok=True)
            for old in self.snapshots(world)[KEPT:]:
                old.unlink(missing_ok=True)  # another snapshot may be deleting the same ones
            return taken

    def snapshots(self, world: str) -> list[Path]:
        """The world's snapshots, newest first (their names sort by the time taken)."""
        return sorted((p for p in self.snapshots_dir.glob(f"{world}.*.db") if p.name.count(".") == 2), reverse=True)

    def restore(self, world: str) -> Path:
        """Put the newest snapshot that verifies in the world's place and return where the
        file that was there went (`<world>.db.damaged-<utc>`). Every other user of the
        world is kept out, and waited for, by the world's lock.

        A world's `-wal` file holds what was committed and not yet folded into the world
        file, and whatever a writer that died left half-written; SQLite applies it to
        whatever file has that name the next time it is opened. So the live file is first
        opened for writing and closed, which folds the log in and removes it, and whatever
        side file is still there afterwards (the file was too damaged to open) is moved
        aside with it: none may be left to be applied to the restored file."""
        with self._locked(world, exclusive=True) as path:
            good = None
            for taken in self.snapshots(world):
                try:
                    _verify(taken)
                except Damaged:
                    continue
                good = taken
                break
            if good is None:
                raise Damaged(f"{world}: no snapshot of it verifies ({len(self.snapshots(world))} in {self.snapshots_dir}); "
                              "the world file is left as it is")
            _settle(path)
            aside = path.with_name(f"{path.name}.damaged-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:4]}")
            for suffix in SIDE_FILES:
                if (side := path.with_name(path.name + suffix)).exists():
                    side.rename(aside.with_name(aside.name + suffix))
            tmp = path.with_name(f".{world}.{uuid.uuid4().hex[:8]}.tmp")
            try:
                with closing(sqlite3.connect(f"file:{good}?mode=ro", uri=True)) as src, closing(sqlite3.connect(tmp)) as dst:
                    src.backup(dst)
                    dst.execute("PRAGMA journal_mode = WAL")
                os.link(path, aside)  # the name always points at a file: the old one until the new one replaces it
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
            return aside


def _root(store: Store) -> tuple[str, str]:
    """(seg, split) of the store's root: its one start with no parent."""
    roots = store.db.execute("SELECT s.seg, s.split FROM episodes e JOIN segments s ON s.seg = e.seg "
                             "WHERE e.start AND s.parent IS NULL").fetchall()
    if len(roots) != 1:
        raise Damaged(f"{store.path}: a world file holds one root, and this one lists {len(roots)}")
    return roots[0]


def _settle(path: Path) -> None:
    """`restore`'s first step, with every other user of the world locked out. When the file
    has a side file, open it for writing, commit nothing and close it: SQLite then folds in
    what was committed, discards what a dead writer left half-written, and removes the side
    files. It waits five seconds at most: the lock leaves nobody else to hold the file, and
    one too damaged to open says so at once or never.
    A file with no side file is not opened: a write transaction on an emptied one would
    give it a header."""
    if not any(path.with_name(path.name + suffix).exists() for suffix in SIDE_FILES):
        return
    try:
        with closing(sqlite3.connect(path, isolation_level=None, timeout=5)) as live:
            live.execute("BEGIN IMMEDIATE")
            live.execute("COMMIT")
    except sqlite3.DatabaseError:
        pass


def _verify(path: Path) -> None:
    """Raise `Damaged` unless the store file at `path` passes SQLite's integrity check and
    every episode it lists loads."""
    try:
        with Store(path, readonly=True) as store, store.reading():
            problems = [r[0] for r in store.db.execute("PRAGMA integrity_check").fetchall()]
            if problems != ["ok"]:
                raise Damaged(f"{path}: SQLite's integrity check: {'; '.join(problems[:5])}")
            _root(store)
            for name in [*store.starts(), *store.names(), *store.names(closed=False)]:
                try:
                    store.load(name)
                except Exception as e:  # whatever a damaged body raises: the point is which episode
                    raise Damaged(f"{path}: episode {name} does not load: {type(e).__name__}: {e}") from e
    except (sqlite3.DatabaseError, Unreadable) as e:
        raise Damaged(f"{path}: {type(e).__name__}: {e}") from e
