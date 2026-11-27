"""The episode store: one SQLite file per store, every episode a segment of a tree.

A **segment** is one episode's own part of its log. It has a parent (the segment it was
forked from, and how many of that segment's events and calls it took) or none, and a row of
**checkpoints** appended in order. A checkpoint holds the events and calls added since the
one before and what changed in the tip (state, actors, actor_tools, status, failure_reason);
a call's messages are stored as what follows those of the same actor's call before it.
Checkpoint 0 is the segment's **start**: it also holds what the segment changed at birth
against its parent at the fork position, namely the files it set and the events it replaced
(a twin's rules file, a swapped event), and its tip whole. Loading walks the parent chain:
the parent's log up to the fork position, then the start, then the checkpoints; files are
the nearest start's files plus the `file_write` events since.

Checkpoints are of two kinds. A **turn** checkpoint (`append`) is taken before play and each
time a delivery returns, before the suite has seen the reply: play can continue or be forked
from it, and from nothing else. The **closing** checkpoint (`close`) is taken once, when
play has ended; it holds what the last step logged and the final status, and it is what
makes an episode visible to readers. `Suite.next_delivery` runs once per reply, so
continuing from a closing checkpoint would run the last step twice: `load(name, at=...)`
refuses one, and so does `append` on a closed episode.

A store's **episodes** are the segments its `episodes` table lists, by `ep_id` (unique in a
store; `seg` is unique everywhere). An ancestor nothing listed (the base both POV arms are
forked from, a chain copied in from another store) sits in the same file unlisted. Every
store is self-contained: `copy_chain` brings a segment and its ancestors into another store,
and because `seg` is unique everywhere copying twice, or merging stores written on two
machines, is the same idempotent operation. It refuses the one case that is not: a
checkpoint both stores hold with different bodies, each having continued the segment.

What the caller owes the store:
- `append` only before play or when a delivery has returned; `close` when play has ended.
- A fork of a stored episode is saved from one of its stored positions: `append` refuses a
  fork made past the parent's last checkpoint. (A parent nothing has saved is saved with
  its fork, as it then stands.)
- After an episode's first save its files change only through `Episode.write_file`. Before
  it they may be set with no event (what `setup` mounts), and the start records them.
- State is plain JSON.

A store also holds the **score rows** of its episodes, each with the judge items behind it
(`put_scores`), so an episode travels with its scores: `copy` carries a segment, its
ancestors, its listing and its rows in one transaction. Rows are only ever added, like
checkpoints: scoring an episode again adds a row. Rows are kept per **judging**: a judge
profile and a rubric, written `<profile>/<rubric>`, or the bare profile for a suite's one
unnamed rubric (`judging`, `split_judging`: the only two places the name is put together
or taken apart; the column is still called `profile`). Every reader takes one row of an
episode under a judging, by one rule (`_current`): the newest that does not say the scorer
raised (`error`), or the newest of all when every one does. So a re-score that failed (a
judge was down) hides no good row, and which row is read never depends on the order rows
arrived in. The row read may therefore be older than the last attempt; it says which code
scored it and when. A row's id is worked out from its content, so the same row arriving
twice is one row.

A **start** is an episode stored once and never played in place (`add_starts`): saved,
closed and marked in one transaction. A world's root is one, and so is every **situation**,
a task set up on the files a closed episode ended with (`Episode.begin`), which is a start
that has a parent. Play of a start is always a rollout of its checkpoint 0. The listing
functions readers use (`names`, `episodes`) leave starts out; `starts` and `situations` list
them.

A **rollout** is a fork of a stored turn checkpoint that says so (`rollout_of`), and the
rollouts one run made of one checkpoint in one round are a **group** (`grp`; the id is the
caller's, `run.group_id`). `groups` says which are played, and `settled` which of those are
scored under one judging.

SQLite gives one writer at a time, on one machine: several processes may append to one
store, and each `append`/`close` is one transaction. A file this class creates is in the
default rollback mode, so a store nobody is writing is one file, and a reader makes a
writer's commit wait. A library world is created in write-ahead mode instead
(`episode/library.py`): its long readers must not hold pushes up. Two writers continuing the same segment is
an error the second one gets. A `local_dev` or `test` segment never shares a store with
another split's (`corpus/release.py` relies on it, and filters what leaves besides).
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import weakref
import zlib
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from writing_assistant.episode.model import CallRecord, Episode, Event, files_at
from writing_assistant.episode.store import LOCAL_DEV

STORE_FILE = "store.db"  # in a run directory, beside run.json
HELD_APART = (LOCAL_DEV, "test")  # splits whose segments share a store with no other split's
JSONL_FILE = "episodes.jsonl"  # what a run directory held before the store; read by nothing


class Unreadable(Exception):
    """A run directory or store file this build will not read or write. The message is for
    the person at the command line: `wa-bench` exits with it and no traceback."""

# The on-disk format is how many of these steps a file has had (PRAGMA user_version). Each
# step only adds, so a writer brings an older file up to date in place; a change that cannot
# be made by adding is a new store with no reader for the old one.
_STEPS = ((
    """CREATE TABLE segments (
        seg TEXT PRIMARY KEY, parent TEXT, parent_events INTEGER, parent_calls INTEGER,
        ep_id TEXT NOT NULL, task_id TEXT NOT NULL, suite TEXT NOT NULL, split TEXT NOT NULL,
        condition TEXT NOT NULL, baseline_of TEXT, rollout_of TEXT, rollout_at INTEGER, meta TEXT NOT NULL)""",
    "CREATE INDEX segments_split ON segments (split)",
    # events, calls: how many the episode holds after this checkpoint, inherited ones included
    """CREATE TABLE checkpoints (
        seg TEXT NOT NULL, n INTEGER NOT NULL, kind TEXT NOT NULL, events INTEGER NOT NULL,
        calls INTEGER NOT NULL, body BLOB NOT NULL, PRIMARY KEY (seg, n))""",
    "CREATE TABLE episodes (ep_id TEXT PRIMARY KEY, seg TEXT NOT NULL UNIQUE)",
), (  # 2: groups, and the score rows beside the episodes
    "ALTER TABLE segments ADD COLUMN grp TEXT",  # a rollout's group; NULL otherwise
    "CREATE INDEX segments_rollout ON segments (rollout_of, rollout_at, grp)",
    # final: 0 for a row that says the scorer raised (`error`)
    """CREATE TABLE scores (
        id TEXT PRIMARY KEY, ep_id TEXT NOT NULL, profile TEXT NOT NULL, final INT NOT NULL, scored_at TEXT NOT NULL,
        row TEXT NOT NULL)""",
    "CREATE INDEX scores_episode ON scores (ep_id, profile, final, scored_at)",
    # The judge items that count in one score row (`score` is its id), in the order asked,
    # packed together as a checkpoint's body is: one row's items repeat each other's evidence.
    "CREATE TABLE judgments (score TEXT PRIMARY KEY, body BLOB NOT NULL)",
), (  # 3: episodes stored as a start and never played (`add_starts`): 1 on those, NULL otherwise
    "ALTER TABLE episodes ADD COLUMN start INTEGER",
))
FORMAT = len(_STEPS)
_SEGMENT = ("seg, parent, parent_events, parent_calls, ep_id, task_id, suite, split, condition, baseline_of, "
            "rollout_of, rollout_at, meta, grp")


def judging(profile: str, rubric: str = "") -> str:
    """The name score rows of `rubric` judged by `profile` are stored under."""
    if not profile:
        raise ValueError("a judging needs a judge profile")
    for what, name in (("judge profile", profile), ("rubric", rubric)):
        if "/" in name:
            raise ValueError(f"{what} {name!r} holds '/', which is what separates the two in a judging's name")
    return f"{profile}/{rubric}" if rubric else profile


def split_judging(name: str) -> tuple[str, str]:
    """(judge profile, rubric) of a judging; the rubric is "" for a bare profile."""
    profile, _, rubric = name.partition("/")
    return profile, rubric


def _current(s: str) -> str:
    """SQL: of the score rows the alias `s` ranges over, the one readers take for its
    episode under its judging. Final before not, then the later; two rows of one time are
    told apart by id, so every reader on every machine takes the same one."""
    return (f"{s}.id = (SELECT id FROM scores WHERE ep_id = {s}.ep_id AND profile = {s}.profile "
            "ORDER BY final DESC, scored_at DESC, id DESC LIMIT 1)")


_CLOSED = "EXISTS (SELECT 1 FROM checkpoints c WHERE c.seg = e.seg AND c.kind = 'closing')"
_PLAYED = "e.start IS NULL"  # of the listing `e`: not a start, so an episode some run plays or played


@dataclass
class _Tip:
    """Where this process last saw a segment it may append to."""

    n: int  # its last checkpoint
    events: int
    calls: int
    tip: dict[str, Any]  # the tip as saved there: what the next checkpoint's change is against
    listed: bool
    ep: weakref.ref[Episode]  # the object those counts are of: another one of the same segment is not its sequel


class Store:
    """`with Store(path) as store:` opens the file, creating it when it is not there and
    bringing one of an older format up to this one; `readonly=True` opens an existing one
    and writes nothing, not even a lock file, so it refuses an older format."""

    def __init__(self, path: str | Path, readonly: bool = False) -> None:
        self.path = Path(path)
        target = f"file:{quote(str(self.path))}?mode=ro" if readonly else f"file:{quote(str(self.path))}"
        self.db = sqlite3.connect(target, uri=True, isolation_level=None, timeout=60)
        self._tips: dict[str, _Tip] = {}  # the open segments this process saved or loaded at their last checkpoint
        # (seg, events, calls) -> what a fork made there inherits. A segment's log only grows,
        # so a prefix of it never changes.
        self._prefixes: dict[tuple[str, int, int], tuple[list[Event], list[CallRecord], dict[str, list]]] = {}
        try:
            if readonly:
                try:
                    version = self.db.execute("PRAGMA user_version").fetchone()[0]
                except sqlite3.OperationalError as e:
                    if "readonly" not in str(e):
                        raise
                    # SQLite must roll the dead writer's journal back before anything can be read.
                    raise Unreadable(f"{self.path}: a writer died in the middle of a checkpoint, and a reader may not "
                                       "repair the file; the next writer does (`wa-bench run` on this directory)") from e
                if version < FORMAT:
                    raise Unreadable(
                        f"{self.path}: store format {version}; this build reads format {FORMAT}, and a reader changes no "
                        "file. Opening it once for writing brings it up to date in place: `wa-bench score --run` on its "
                        "directory, or `wa-bench run` on it")
            else:
                # The version is read inside the write transaction: of two writers opening an
                # old file together, the second waits and then finds nothing left to add.
                with self._write():
                    version = self.db.execute("PRAGMA user_version").fetchone()[0]
                    if version == 0 and self.db.execute("SELECT 1 FROM sqlite_master").fetchone():
                        raise Unreadable(f"{self.path}: an SQLite file that is not an episode store")
                    for step in _STEPS[version:]:
                        for statement in step:
                            self.db.execute(statement)
                    if version < FORMAT:
                        self.db.execute(f"PRAGMA user_version = {FORMAT}")
            if version > FORMAT:
                raise Unreadable(f"{self.path}: store format {version}, written by a later build; this one reads and "
                                   f"writes format {FORMAT}")
        except BaseException:
            self.db.close()
            raise

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.db.close()

    @contextmanager
    def _write(self) -> Iterator[None]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    # -- writing ---------------------------------------------------------------------------

    def append(self, *eps: Episode) -> None:
        """A turn checkpoint of each episode, all in one transaction: its start when the
        store has not seen it (first saving, unlisted, any ancestor nothing saved yet), else
        what it gained since its last checkpoint. An episode that gained nothing gets none."""
        self._checkpoint(eps, "turn")

    def close(self, *eps: Episode) -> None:
        """The closing checkpoint of each episode, all in one transaction, so a task's
        episodes become visible to readers together. Nothing can be appended after it."""
        self._checkpoint(eps, "closing")

    def _checkpoint(self, eps: tuple[Episode, ...], kind: str) -> None:
        tips: dict[str, _Tip | None] = {}  # None: closed in this transaction
        with self._write():
            for ep in eps:
                self._save(ep, kind, True, tips)
        for seg, tip in tips.items():
            if tip is None:
                self._tips.pop(seg, None)
            else:
                self._tips[seg] = tip

    def add_starts(self, *eps: Episode) -> None:
        """Store each episode as a **start**: saved as it stands, closed at once and marked,
        all in one transaction. Nothing plays a start in place (`append` refuses a closed
        episode); play is a rollout of its checkpoint 0, which is why it is saved before it
        is closed: a closing checkpoint cannot be forked. A start is no played episode, so
        `names` and `episodes` leave it out and no reader scores, reports or resumes it;
        `starts` lists them."""
        tips: dict[str, _Tip | None] = {}
        with self._write():
            for ep in eps:
                self._save(ep, "turn", True, tips)
                self._save(ep, "closing", True, tips)
                self.db.execute("UPDATE episodes SET start = 1 WHERE seg = ?", (ep.seg,))

    def _save(self, ep: Episode, kind: str, listed: bool, tips: dict[str, _Tip | None]) -> None:
        last = tips[ep.seg] if ep.seg in tips else self._tips.get(ep.seg)
        body: dict[str, Any] = {}
        if last is None:
            last = _Tip(-1, ep.parent[1] if ep.parent else 0, ep.parent[2] if ep.parent else 0, {}, False, weakref.ref(ep))
            body = self._start(ep, tips)
        elif last.ep() is not ep:
            raise ValueError(f"{self.path}: {ep.ep_id} is being continued here by another Episode object; one loaded at "
                             "an earlier checkpoint is played on by forking it")
        if listed and not last.listed:
            try:
                self.db.execute("INSERT INTO episodes (ep_id, seg) VALUES (?, ?)", (ep.ep_id, ep.seg))
            except sqlite3.IntegrityError:
                raise ValueError(f"{self.path}: another episode named {ep.ep_id!r} is already stored here (the same task "
                                 "set up twice, or the same fork made twice)") from None
        tip = json.loads(json.dumps({"state": ep.state, "actors": ep.actors, "actor_tools": ep.actor_tools,
                                     "status": ep.status, "failure_reason": ep.failure_reason}))
        now = _Tip(last.n, len(ep.events), len(ep.calls), tip, last.listed or listed, last.ep)
        if kind == "closing" or last.n < 0 or (now.events, now.calls, tip) != (last.events, last.calls, last.tip):
            now.n += 1
            body |= {"events": [_event_out(e) for e in ep.events[last.events:]],
                     "calls": _calls_out(ep.calls, last.calls), "tip": _diff(last.tip, tip)}
            try:
                self.db.execute("INSERT INTO checkpoints (seg, n, kind, events, calls, body) VALUES (?, ?, ?, ?, ?, ?)",
                                (ep.seg, now.n, kind, now.events, now.calls, _pack(body)))
            except sqlite3.IntegrityError:
                raise ValueError(f"{self.path}: checkpoint {now.n} of {ep.ep_id} is already stored: "
                                 "another writer continued this episode") from None
        tips[ep.seg] = None if kind == "closing" else now

    def _start(self, ep: Episode, tips: dict[str, _Tip | None]) -> dict[str, Any]:
        """Create `ep`'s segment, its unsaved ancestors first, and return what its start
        holds beside a checkpoint's own parts: the files it set and the events it replaced,
        against its parent at the fork position."""
        if self.db.execute("SELECT 1 FROM segments WHERE seg = ?", (ep.seg,)).fetchone():
            if self.db.execute("SELECT 1 FROM checkpoints WHERE seg = ? AND kind = 'closing'", (ep.seg,)).fetchone():
                raise ValueError(f"{self.path}: {ep.ep_id} is closed; nothing is appended to a closed episode")
            raise ValueError(f"{self.path}: {ep.ep_id} is stored here, and this is not the Episode this process loaded "
                             "at its last checkpoint: it cannot be appended to (fork it to play on from an earlier one)")
        files = files_at(ep, ep.parent[1] if ep.parent else 0)
        replaced: dict[str, list] = {}
        if ep.parent:
            seg, n_events, n_calls = ep.parent
            if seg not in tips and not self.db.execute("SELECT 1 FROM segments WHERE seg = ?", (seg,)).fetchone():
                if ep.forked_from is None:
                    raise ValueError(f"{self.path}: {ep.ep_id} was forked from segment {seg}, which is not stored here")
                self._save(ep.forked_from, "turn", False, tips)
            # Only a parent's stored log is a fixed thing to be stored after: a fork made past
            # it would load with whatever the parent's next checkpoint, or a resume, put there.
            events, calls = self.db.execute("SELECT events, calls FROM checkpoints WHERE seg = ? ORDER BY n DESC LIMIT 1",
                                            (seg,)).fetchone()
            if n_events > events or n_calls > calls:
                raise ValueError(f"{self.path}: {ep.ep_id} was forked from its parent at {n_events} events and {n_calls} "
                                 f"calls, past the {events} events and {calls} calls stored of it; checkpoint the "
                                 "parent first, or fork one of its stored checkpoints")
            if ep.forked_from is not None:
                was_events, was_files = ep.forked_from.events[:n_events], files_at(ep.forked_from, n_events)
            else:
                was_events, _, was_files = self._rebuild(seg, cut=(n_events, n_calls))[:3]
            replaced = {str(i): _event_out(e) for i, (e, was) in enumerate(zip(ep.events[:n_events], was_events, strict=True))
                        if e is not was and e != was}
            files = {p: files.get(p) for p in files.keys() | was_files.keys() if files.get(p) != was_files.get(p)}
        self._insert_segment((ep.seg, *(ep.parent or (None, None, None)), ep.ep_id, ep.task_id, ep.suite, ep.split,
                              ep.condition, ep.baseline_of, *(ep.rollout_of or (None, None)),
                              json.dumps(ep.meta, ensure_ascii=False), ep.group))
        return {"files": files, "replaced": replaced}

    def _insert_segment(self, row: tuple) -> None:
        ep_id, split = row[4], row[7]
        other = self.db.execute("SELECT split FROM segments WHERE split != ? LIMIT 1", (split,)).fetchone()
        if other and (split in HELD_APART or other[0] in HELD_APART):
            raise ValueError(f"{self.path}: holds {other[0]!r} segments; a {split!r} one ({ep_id}) may not join them "
                             f"({' and '.join(HELD_APART)} segments share a store with no other split's)")
        self.db.execute(f"INSERT INTO segments ({_SEGMENT}) VALUES ({', '.join('?' * len(row))})", row)

    # -- reading ---------------------------------------------------------------------------

    @contextmanager
    def reading(self) -> Iterator[None]:
        """One read transaction: everything read inside sees the file as it stood at the
        first read, whatever another process appends meanwhile. In rollback mode a commit
        waits for it to end (up to the writer's 60 s timeout), so it is kept short."""
        if self.db.in_transaction:  # already inside one: the outer block is the transaction
            yield
            return
        self.db.execute("BEGIN")
        try:
            yield
        finally:
            self.db.execute("COMMIT")

    def names(self, closed: bool = True) -> list[str]:
        """`ep_id` of the store's closed episodes, or of its open ones, in the order they were
        first saved. Starts are not among them."""
        rows = self.db.execute(
            f"SELECT e.ep_id FROM episodes e WHERE {_PLAYED} AND {'' if closed else 'NOT '}{_CLOSED} ORDER BY e.rowid")
        return [r[0] for r in rows.fetchall()]

    def is_closed(self, name: str) -> bool:
        """Whether the store lists an episode of this name and it is closed (played to its
        end, or a start): one lookup, for a caller that has the name and not the listing."""
        return bool(self.db.execute(f"SELECT 1 FROM episodes e WHERE e.ep_id = ? AND {_CLOSED}", (name,)).fetchone())

    def root_of(self, name: str) -> str:
        """The `ep_id` of the first segment in the episode's chain: in a world, the world."""
        return self.db.execute("SELECT ep_id FROM segments WHERE seg = ?", (self.chain(name)[0],)).fetchone()[0]

    def starts(self) -> list[str]:
        """`ep_id` of the episodes stored as starts (`add_starts`), in the order stored."""
        return [r[0] for r in self.db.execute("SELECT ep_id FROM episodes WHERE start ORDER BY rowid").fetchall()]

    def situations(self) -> list[str]:
        """The starts that have a parent: each a task set up at a position (`run.situate`).
        A start with none is a root, the material a world begins from."""
        rows = self.db.execute("SELECT e.ep_id FROM episodes e JOIN segments s ON s.seg = e.seg "
                               "WHERE e.start AND s.parent IS NOT NULL ORDER BY e.rowid")
        return [r[0] for r in rows.fetchall()]

    def episodes(self, closed: bool = True) -> Iterator[Episode]:
        """The store's closed episodes, which is what a reader wants; `closed=False` gives
        the open ones instead, each at its last turn checkpoint and ready to be continued."""
        for name in self.names(closed):
            yield self.load(name)

    def load(self, name: str, at: int | None = None) -> Episode:
        """The episode named `name` (a listed `ep_id`, or a `seg`) as last saved. With `at`,
        as it stood at that turn checkpoint (0 is its start), to continue or fork from: a
        closing checkpoint is refused. Continue the last checkpoint of an open episode by
        appending to what this returns, any other by forking it. An episode saved before play
        and each time a delivery returned has turn checkpoint n after its n-th delivery."""
        seg = self._seg(name)
        events, calls, files, tip, n, kind, head = self._rebuild(seg, at=at)
        (_, parent, parent_events, parent_calls, ep_id, task_id, suite, split, condition, baseline_of,
         rollout_of, rollout_at, meta, group) = head
        if at is not None and n != at:
            raise LookupError(f"{self.path}: {ep_id} has no checkpoint {at} (its last is {n})")
        if at is not None and kind == "closing":
            raise ValueError(f"{self.path}: checkpoint {at} of {ep_id} is its closing checkpoint; play cannot "
                             f"continue from the end of an episode (its turn checkpoints are 0 to {at - 1})")
        ep = Episode(ep_id=ep_id, task_id=task_id, suite=suite, split=split, seg=seg,
                     parent=(parent, parent_events, parent_calls) if parent else None, baseline_of=baseline_of,
                     rollout_of=(rollout_of, rollout_at) if rollout_of else None, group=group,
                     condition=condition, meta=json.loads(meta), events=events, calls=calls,
                     files={p: text for p, (text, _) in files.items()}, file_revs={p: rev for p, (_, rev) in files.items()},
                     **copy.deepcopy(tip))
        # The episode can be appended to when it is at its last checkpoint, open, and no Episode
        # this process loaded or saved earlier is still continuing the segment.
        held = self._tips.get(seg)
        if kind == "turn" and (held is None or held.ep() is None) and \
                n == self.db.execute("SELECT MAX(n) FROM checkpoints WHERE seg = ?", (seg,)).fetchone()[0]:
            listed = bool(self.db.execute("SELECT 1 FROM episodes WHERE seg = ?", (seg,)).fetchone())
            self._tips[seg] = _Tip(n, len(events), len(calls), tip, listed, weakref.ref(ep))
        return ep

    def chain(self, name: str) -> list[str]:
        """The episode's segment and its ancestors', root first."""
        seg, out = self._seg(name), []
        while seg:
            out.append(seg)
            seg = self.db.execute("SELECT parent FROM segments WHERE seg = ?", (seg,)).fetchone()[0]
        return out[::-1]

    def snapshots(self, seg: str) -> tuple[str, list[dict[str, Any]]]:
        """One segment's split and the state at each of its checkpoints: what a release rule
        tests before the segment leaves the machine."""
        split = self.db.execute("SELECT split FROM segments WHERE seg = ?", (seg,)).fetchone()[0]
        tip: dict[str, Any] = {}
        states = []
        for (body,) in self.db.execute("SELECT body FROM checkpoints WHERE seg = ? ORDER BY n", (seg,)).fetchall():
            _patch(tip, _unpack(body)["tip"])
            states.append(copy.deepcopy(tip["state"]))
        return split, states

    def copy_chain(self, name: str, into: Store) -> str:
        """Copy the episode's segment and every ancestor into another store, whole, skipping
        what is already there, and return its `seg`. It arrives as an unlisted ancestor to
        fork from (`copy` lists it too). A checkpoint the other store holds with a different
        body is refused, and nothing is copied: both stores continued one open segment."""
        with into._write():
            return self._copy_chain(name, into, None, Counter(), set())

    def _copy_chain(self, name: str, into: Store, listing: tuple[str, int | None] | None, added: Counter[str],
                    copied: set[str]) -> str:
        """`listing`: the episode's (`ep_id`, `start`) in this store's `episodes`, to list
        it the same way there; None leaves it unlisted. `added` counts what the other store
        gained, and `copied` holds the segments this transaction has already brought over:
        the members of a group share every ancestor, which is then walked once."""
        chain = self.chain(name)
        for seg in chain:
            if seg in copied:
                continue
            copied.add(seg)
            row = self.db.execute(f"SELECT {_SEGMENT} FROM segments WHERE seg = ?", (seg,)).fetchone()
            if into.db.execute("SELECT 1 FROM segments WHERE seg = ?", (seg,)).fetchone() is None:
                into._insert_segment(row)
                added["segments"] += 1
            have = dict(into.db.execute("SELECT n, body FROM checkpoints WHERE seg = ?", (seg,)).fetchall())
            for n, kind, events, calls, body in self.db.execute(
                    "SELECT n, kind, events, calls, body FROM checkpoints WHERE seg = ? ORDER BY n", (seg,)).fetchall():
                if n not in have:
                    into.db.execute("INSERT INTO checkpoints (seg, n, kind, events, calls, body) VALUES (?, ?, ?, ?, ?, ?)",
                                    (seg, n, kind, events, calls, body))
                    added["checkpoints"] += 1
                elif have[n] != body and _unpack(have[n]) != _unpack(body):  # two zlib builds pack one body differently
                    raise ValueError(f"{into.path}: checkpoint {n} of {row[4]} is stored there with another body: "
                                     f"it and {self.path} each continued this episode, and only one of them can be kept")
        if listing:
            ep_id, start = listing
            have = into.db.execute("SELECT seg FROM episodes WHERE ep_id = ?", (ep_id,)).fetchone()
            if have is None:
                into.db.execute("INSERT INTO episodes (ep_id, seg, start) VALUES (?, ?, ?)", (ep_id, chain[-1], start))
                added["episodes"] += 1
            elif have[0] != chain[-1]:
                raise ValueError(f"{into.path}: another episode named {ep_id!r} is already stored there")
        return chain[-1]

    def copy(self, names: list[str], into: Store, scores: bool = True) -> Counter[str]:
        """Copy each named episode into another store as it is here: its segment and every
        ancestor (`copy_chain`), its listing (a start stays one), and every score row of it
        with its judge items, skipping the rows already there. A name that is an unlisted segment's `seg`
        arrives unlisted and brings no rows. This store is read in one read transaction, so
        what is copied is one state of a file another process may be writing, and the other
        is written in one transaction: an episode is there with its rows or not at all.
        Returns what the other store gained: `episodes`, `segments`, `checkpoints`, `scores`.
        `scores=False` copies the episodes alone and reads no score row: what play does."""
        added: Counter[str] = Counter(episodes=0, segments=0, checkpoints=0, scores=0)
        copied: set[str] = set()
        with self.reading(), into._write():
            for name in names:
                listed = self.db.execute("SELECT ep_id, start FROM episodes WHERE seg = ?", (self._seg(name),)).fetchone()
                self._copy_chain(name, into, listed, added, copied)
                if listed and scores:
                    for judging_, row, items in self.db.execute(
                            "SELECT s.profile, s.row, j.body FROM scores s LEFT JOIN judgments j ON j.score = s.id "
                            "WHERE s.ep_id = ? ORDER BY s.rowid", listed[:1]).fetchall():
                        added["scores"] += into._add_score(judging_, row, items)
        return added

    def rollouts(self, source: str, at: int | None = None, group: str | None = None) -> list[str]:
        """`ep_id` of the stored rollouts of the episode named `source`, open ones included:
        of every turn checkpoint and group, or of the ones given."""
        tests = {"rollout_at": at, "grp": group}
        rows = self.db.execute(
            "SELECT e.ep_id FROM episodes e JOIN segments s ON s.seg = e.seg WHERE s.rollout_of = ?"
            + "".join(f" AND s.{column} = ?" for column, value in tests.items() if value is not None) + " ORDER BY e.rowid",
            (source, *(value for value in tests.values() if value is not None)))
        return [r[0] for r in rows.fetchall()]

    # -- score rows ------------------------------------------------------------------------

    def put_scores(self, judging: str, rows: list[dict[str, Any]], judgments: list[dict[str, Any]] = ()) -> None:
        """Add score rows under a judging, each with the judge items that count in it
        (`judgments` whose `scored` is the row's `ep_id`, in the order given), all in one
        transaction. A row carries `ep_id` and `scored_at`. Nothing is replaced: the rows an
        episode had stay, and `_current` says which is read."""
        behind: dict[str, list[dict[str, Any]]] = {}
        for j in judgments:
            behind.setdefault(j["scored"], []).append(j)
        with self._write():
            for row in rows:
                items = behind.get(row["ep_id"])
                self._add_score(judging, json.dumps(row, ensure_ascii=False), _pack(items) if items else None)

    def _add_score(self, judging: str, text: str, items: bytes | None) -> int:
        """One row arriving, as its JSON text, with its judge items packed (None: it has
        none). Its id, episode, time and whether it is final are worked out from the text
        here, never taken from the file it came from, and a row already held is skipped with
        its items. Returns how many rows that added (1 or 0)."""
        row = json.loads(text)
        id_ = hashlib.sha256(f"{judging}\0{text}".encode()).hexdigest()[:32]
        added = self.db.execute(
            "INSERT OR IGNORE INTO scores (id, ep_id, profile, final, scored_at, row) VALUES (?, ?, ?, ?, ?, ?)",
            (id_, row["ep_id"], judging, "error" not in row, row["scored_at"], text))
        if added.rowcount and items:
            self.db.execute("INSERT INTO judgments (score, body) VALUES (?, ?)", (id_, items))
        return added.rowcount

    def judgings(self, profile: str | None = None) -> list[str]:
        """The judgings this store holds score rows under: all of them, or those of one
        judge profile (its unnamed rubric first)."""
        names = [r[0] for r in self.db.execute("SELECT DISTINCT profile FROM scores ORDER BY profile").fetchall()]
        return [n for n in names if profile is None or split_judging(n)[0] == profile]

    def judged(self, profile: str) -> dict[str, tuple[str, str | None, dict[str, list[dict[str, str]] | None]]]:
        """Every closed episode some run played, in the order first saved -> (its suite, the
        episode it is a rollout of, rubric -> the judges' stamps of the row readers take)
        under one judge profile. A rubric whose row says the scorer raised is there as None,
        one never scored is absent, and a row no judge was asked for has no stamps. One
        query in one read state, with no episode loaded: what lets a judging pass find the
        little it has to do in a large world."""
        rows = self.db.execute(
            f"""SELECT e.ep_id, s.suite, s.rollout_of, sc.profile, sc.final, json_extract(sc.row, '$.judges')
                FROM episodes e JOIN segments s ON s.seg = e.seg
                LEFT JOIN scores sc ON sc.ep_id = e.ep_id AND {_current("sc")}
                WHERE {_PLAYED} AND {_CLOSED} ORDER BY e.rowid""")
        out: dict[str, tuple[str, str | None, dict[str, list[dict[str, str]] | None]]] = {}
        parsed: dict[str | None, list[dict[str, str]]] = {None: []}  # a world's rows carry few distinct stamps
        for ep_id, suite, source, name, final, stamps in rows.fetchall():
            has = out.setdefault(ep_id, (suite, source, {}))[2]
            if name is not None and split_judging(name)[0] == profile:
                if stamps not in parsed:
                    parsed[stamps] = json.loads(stamps)
                has[split_judging(name)[1]] = parsed[stamps] if final else None
        return out

    def scores(self, judging: str) -> dict[str, dict[str, Any]]:
        """`ep_id` -> its score row under `judging` (`_current`)."""
        rows = self.db.execute(f"SELECT s.ep_id, s.row FROM scores s WHERE s.profile = ? AND {_current('s')} ORDER BY s.rowid",
                               (judging,))
        return {ep_id: json.loads(row) for ep_id, row in rows.fetchall()}

    def judgments(self, judging: str | None = None, scored: str | None = None, superseded: bool = False) -> list[dict[str, Any]]:
        """The judge items behind the score rows readers take (`_current`), in the order they
        were asked within a row: under one judging or every one, of every episode or of the
        one named `scored`. `superseded` gives the items of an episode's other rows as
        well: everything `copy` would carry."""
        tests = {"s.profile": judging, "s.ep_id": scored}
        rows = self.db.execute(
            "SELECT j.body FROM judgments j JOIN scores s ON s.id = j.score WHERE " + ("1" if superseded else _current("s"))
            + "".join(f" AND {c} = ?" for c, v in tests.items() if v is not None) + " ORDER BY s.rowid",
            tuple(v for v in tests.values() if v is not None))
        return [item for r in rows.fetchall() for item in _unpack(r[0])]

    def groups(self, group: str | None = None) -> dict[str, dict[str, Any]]:
        """Every group of rollouts stored here, or the one named: group id -> `size` (the
        members asked for of each source episode, `meta["group_size"]`), `sources` (source
        `ep_id` -> how many of its members are closed) and `played`: every source episode in
        it (a base, and its twin when the twin was rolled out too) has `size` closed members.
        A member that never closed is not counted, so a group short of its size is never
        played, whoever copied it here. Read from the episodes and their segments alone:
        whether anything was scored is `settled`'s question."""
        rows = self.db.execute(
            f"""SELECT s.grp, s.rollout_of, MAX(json_extract(s.meta, '$.group_size')), SUM({_CLOSED})
                FROM episodes e JOIN segments s ON s.seg = e.seg
                WHERE s.grp {'= ?' if group else 'IS NOT NULL'} GROUP BY s.grp, s.rollout_of ORDER BY MIN(e.rowid)""",
            (group,) if group else ())
        found: dict[str, dict[str, Any]] = {}
        for grp, source, size, closed in rows.fetchall():
            found.setdefault(grp, {"size": size, "sources": {}})["sources"][source] = closed
        for g in found.values():
            g["played"] = all(n == g["size"] for n in g["sources"].values())
        return found

    def settled(self, judging: str, group: str | None = None) -> dict[str, dict[str, Any]]:
        """`groups`, asked of one judging: group id -> `size`, `sources` (source `ep_id` -> how
        many of its closed members have a final row under `judging`) and `settled`. A
        member's row is the one every reader takes (`_current`), and it is final unless it
        says the scorer raised (`error`), which is what a judge that was down leaves.

        A group is settled when it is played, every member has such a row and no two of the
        rows were scored by different judges for one use (the rows' `judges` stamps: a
        profile is a name in one machine's models.toml)."""
        stamps: dict[str, dict[str, set[str]]] = {}
        with self.reading():  # the groups and their rows are one state: a group may be arriving
            found = {grp: {"size": g["size"], "sources": dict.fromkeys(g["sources"], 0), "settled": g["played"]}
                     for grp, g in self.groups(group).items()}
            # CROSS JOIN fixes the order: the group's segments first, then each one's rows. Left to
            # itself SQLite starts from every score row in the file (1.4 s for one group of a
            # 100,000-rollout world, against 10 ms).
            rows = self.db.execute(
                f"""SELECT s.grp, s.rollout_of, sc.row FROM segments s CROSS JOIN episodes e ON e.seg = s.seg
                    CROSS JOIN scores sc ON sc.ep_id = e.ep_id AND sc.profile = ? AND sc.final AND {_current("sc")}
                    WHERE s.grp {'= ?' if group else 'IS NOT NULL'} AND {_CLOSED}""",
                (judging, *([group] if group else []))).fetchall()
        for grp, source, row in rows:
            found[grp]["sources"][source] += 1
            for stamp in json.loads(row).get("judges", []):
                stamps.setdefault(grp, {}).setdefault(stamp["use"], set()).add(json.dumps(stamp, sort_keys=True))
        for grp, g in found.items():
            g["settled"] = (g["settled"] and all(n == g["size"] for n in g["sources"].values())
                            and all(len(judges) == 1 for judges in stamps.get(grp, {}).values()))
        return found

    def _seg(self, name: str) -> str:
        row = (self.db.execute("SELECT seg FROM episodes WHERE ep_id = ?", (name,)).fetchone()
               or self.db.execute("SELECT seg FROM segments WHERE seg = ?", (name,)).fetchone())
        if row is None:
            raise LookupError(f"{self.path}: no episode or segment {name!r}")
        return row[0]

    def _rebuild(self, seg: str, at: int | None = None, cut: tuple[int, int] | None = None) -> tuple:
        """(events, calls, files, tip, n, kind, segment row) of a segment after checkpoint `at`
        (its last when None). `files` maps path -> [text, rev]. With `cut = (events, calls)`:
        what a fork made at that position inherits, the tip then being of no use."""
        if cut and (seg, *cut) in self._prefixes:
            events, calls, files = self._prefixes[seg, *cut]
            return list(events), list(calls), dict(files)
        head = self.db.execute(f"SELECT {_SEGMENT} FROM segments WHERE seg = ?", (seg,)).fetchone()
        own = head[2] or 0  # where the segment's own events begin
        events, calls, files = self._rebuild(head[1], cut=(head[2], head[3]))[:3] if head[1] else ([], [], {})
        tip: dict[str, Any] = {}
        n, kind = -1, ""
        query = "SELECT n, kind, body FROM checkpoints WHERE seg = ?" + (" AND n <= ?" if at is not None else "") + " ORDER BY n"
        for n, kind, body in self.db.execute(query, (seg,) if at is None else (seg, at)).fetchall():
            if cut and n > 0 and len(events) >= cut[0] and len(calls) >= cut[1]:
                break  # a fork takes the whole start, whatever its position, and no more log than it needs
            part = _unpack(body)
            if n == 0:
                for i, event in part["replaced"].items():
                    events[int(i)] = Event(int(i), *event)
                for path, file in part["files"].items():
                    files.pop(path, None) if file is None else files.update({path: file})
            events += [Event(len(events) + i, *e) for i, e in enumerate(part["events"])]
            for call in part["calls"]:
                keep, then = call["messages"]
                calls.append(CallRecord(**call | {"messages": _messages_before(calls, len(calls), call["actor"])[:keep] + then}))
            _patch(tip, part["tip"])
        if cut:
            del events[cut[0]:], calls[cut[1]:]
        for event in events[own:]:
            if event.kind == "file_write":
                files[event.data["path"]] = [event.data["after"], event.data["rev"]]
        if cut:
            self._prefixes[seg, *cut] = (list(events), list(calls), dict(files))
            return events, calls, files
        return events, calls, files, tip, n, kind, head


# -- a checkpoint's body ---------------------------------------------------------------------

def _pack(body: Any) -> bytes:
    return zlib.compress(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6)


def _unpack(blob: bytes) -> Any:
    return json.loads(zlib.decompress(blob))


def _event_out(event: Event) -> list:
    return [event.actor, event.kind, event.data, event.visible_to]  # its idx is its place in the log


def _calls_out(calls: list[CallRecord], start: int) -> list[dict[str, Any]]:
    """`calls[start:]` as stored. A conversation only grows, so a call's messages are kept as
    [how many of the same actor's previous call's messages they begin with, the rest]: a
    checkpoint then costs what the turn added, not the whole history again."""
    out = []
    for i in range(start, len(calls)):
        call, before = calls[i], _messages_before(calls, i, calls[i].actor)
        keep = next((k for k, (a, b) in enumerate(zip(before, call.messages)) if a != b), min(len(before), len(call.messages)))
        out.append(asdict(replace(call, messages=[])) | {"messages": [keep, call.messages[keep:]]})
    return out


def _messages_before(calls: list[CallRecord], i: int, actor: str) -> list[dict[str, Any]]:
    """The messages of `actor`'s last call before `calls[i]`; [] when it has none."""
    return next((calls[j].messages for j in range(i - 1, -1, -1) if calls[j].actor == actor), [])


def _diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """What turns `old` into `new`: `set` (keys added or given another value), `in` (keys
    whose dict changed inside, as a diff of their own) and `del`; a part with nothing is
    left out, so an unchanged dict is {}."""
    out: dict[str, Any] = {"set": {}, "in": {}, "del": [k for k in old if k not in new]}
    for key, value in new.items():
        if key not in old:
            out["set"][key] = value
        elif old[key] != value:
            both = isinstance(value, dict) and isinstance(old[key], dict)
            out["in" if both else "set"][key] = _diff(old[key], value) if both else value
    return {part: change for part, change in out.items() if change}


def _patch(into: dict[str, Any], diff: dict[str, Any]) -> None:
    for key in diff.get("del", []):
        del into[key]
    into.update(diff.get("set", {}))
    for key, inner in diff.get("in", {}).items():
        _patch(into[key], inner)


def has_store(run_dir: str | Path) -> bool:
    """Whether a run directory has a store yet. One written before the store is refused: it
    is not an empty run, and a store made beside its file would be a second run in it."""
    run_dir = Path(run_dir)
    if (run_dir / JSONL_FILE).exists():
        raise Unreadable(f"{run_dir}: holds {JSONL_FILE}: this run directory predates the episode store ({STORE_FILE}), "
                         "and this build has no reader for it. It is not resumed, scored, reported, viewed or synced; "
                         "play its tasks again into a new directory.")
    return (run_dir / STORE_FILE).exists()


def read_run[T](run_dir: str | Path, read: Callable[[Store], T], empty: T) -> T:
    """What `read` makes of a run directory's store, opened read-only; `empty` when the
    run has no store yet."""
    if not has_store(run_dir):
        return empty
    with Store(Path(run_dir) / STORE_FILE, readonly=True) as store:
        return read(store)


def open_episodes(run_dir: str | Path) -> list[str]:
    """The names of a run directory's open episodes: a live run's, in play, or what a killed
    run left for the next `run` to continue. Readers show none of them, and say so."""
    return read_run(run_dir, lambda store: store.names(closed=False), [])


def closed_episodes(run_dir: str | Path) -> list[Episode]:
    """The closed episodes of a run directory's store; none when the run has no store yet."""
    return read_run(run_dir, lambda store: list(store.episodes()), [])
