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
machines, is the same idempotent operation.

What the caller owes the store:
- `append` only before play or when a delivery has returned; `close` when play has ended.
- After an episode's first save its files change only through `Episode.write_file`. Before
  it they may be set with no event (what `setup` mounts), and the start records them.
- State is plain JSON.

SQLite gives one writer at a time, on one machine: several processes may append to one
store, and each `append`/`close` is one transaction. The file is in the default rollback
mode, so a store nobody is writing is one file. Two writers continuing the same segment is
an error the second one gets. A `local_dev` or `test` segment never shares a store with
another split's (`corpus/release.py` relies on it, and filters what leaves besides).
"""

from __future__ import annotations

import copy
import json
import sqlite3
import weakref
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from writing_assistant.episode.model import CallRecord, Episode, Event
from writing_assistant.episode.store import LOCAL_DEV

STORE_FILE = "store.db"  # in a run directory, beside run.json
# The on-disk format (PRAGMA user_version). A store written under another number is refused:
# there is no reader for any format but this one.
FORMAT = 1
HELD_APART = (LOCAL_DEV, "test")  # splits whose segments share a store with no other split's

_SCHEMA = (
    """CREATE TABLE segments (
        seg TEXT PRIMARY KEY, parent TEXT, parent_events INTEGER, parent_calls INTEGER,
        ep_id TEXT NOT NULL, task_id TEXT NOT NULL, suite TEXT NOT NULL, split TEXT NOT NULL,
        condition TEXT NOT NULL, baseline_of TEXT, rollout_of TEXT, rollout_at INTEGER, meta TEXT NOT NULL)""",
    "CREATE INDEX segments_split ON segments (split)",
    "CREATE INDEX segments_parent ON segments (parent)",
    # events, calls: how many the episode holds after this checkpoint, inherited ones included
    """CREATE TABLE checkpoints (
        seg TEXT NOT NULL, n INTEGER NOT NULL, kind TEXT NOT NULL, events INTEGER NOT NULL,
        calls INTEGER NOT NULL, body BLOB NOT NULL, PRIMARY KEY (seg, n))""",
    "CREATE TABLE episodes (ep_id TEXT PRIMARY KEY, seg TEXT NOT NULL UNIQUE)",
)
_SEGMENT = ("seg, parent, parent_events, parent_calls, ep_id, task_id, suite, split, condition, baseline_of, "
            "rollout_of, rollout_at, meta")
_CLOSED = "EXISTS (SELECT 1 FROM checkpoints c WHERE c.seg = e.seg AND c.kind = 'closing')"


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
    """`with Store(path) as store:` opens the file, creating it when it is not there;
    `readonly=True` opens an existing one and writes nothing, not even a lock file."""

    def __init__(self, path: str | Path, readonly: bool = False) -> None:
        self.path = Path(path)
        target = f"file:{quote(str(self.path))}?mode=ro" if readonly else f"file:{quote(str(self.path))}"
        self.db = sqlite3.connect(target, uri=True, isolation_level=None, timeout=60)
        self._tips: dict[str, _Tip] = {}  # the open segments this process saved or loaded at their last checkpoint
        # (seg, events, calls) -> what a fork made there inherits. A segment's log only grows,
        # so a prefix of it never changes.
        self._prefixes: dict[tuple[str, int, int], tuple[list[Event], list[CallRecord], dict[str, list]]] = {}
        try:
            if not readonly:
                with self._write():
                    if not self.db.execute("SELECT 1 FROM sqlite_master").fetchone():
                        for statement in _SCHEMA:
                            self.db.execute(statement)
                        self.db.execute(f"PRAGMA user_version = {FORMAT}")
            try:
                version = self.db.execute("PRAGMA user_version").fetchone()[0]
            except sqlite3.OperationalError as e:
                if not readonly or "readonly" not in str(e):
                    raise
                # SQLite must roll the dead writer's journal back before anything can be read.
                raise RuntimeError(f"{self.path}: a writer died in the middle of a checkpoint, and a reader may not "
                                   "repair the file; the next writer does (`wa-bench run` on this directory)") from e
            if version != FORMAT:
                raise ValueError(f"{self.path}: store format {version}; this build reads and writes format {FORMAT} only")
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
        files = _files_at(ep, ep.parent[1] if ep.parent else 0)
        replaced: dict[str, list] = {}
        if ep.parent:
            seg, n_events, n_calls = ep.parent
            if seg not in tips and not self.db.execute("SELECT 1 FROM segments WHERE seg = ?", (seg,)).fetchone():
                if ep.forked_from is None:
                    raise ValueError(f"{self.path}: {ep.ep_id} was forked from segment {seg}, which is not stored here")
                self._save(ep.forked_from, "turn", False, tips)
            if ep.forked_from is not None:
                was_events, was_files = ep.forked_from.events[:n_events], _files_at(ep.forked_from, n_events)
            else:
                was_events, _, was_files = self._rebuild(seg, cut=(n_events, n_calls))[:3]
            replaced = {str(i): _event_out(e) for i, (e, was) in enumerate(zip(ep.events[:n_events], was_events, strict=True))
                        if e is not was and e != was}
            files = {p: files.get(p) for p in files.keys() | was_files.keys() if files.get(p) != was_files.get(p)}
        self._insert_segment((ep.seg, *(ep.parent or (None, None, None)), ep.ep_id, ep.task_id, ep.suite, ep.split,
                              ep.condition, ep.baseline_of, *(ep.rollout_of or (None, None)),
                              json.dumps(ep.meta, ensure_ascii=False)))
        return {"files": files, "replaced": replaced}

    def _insert_segment(self, row: tuple) -> None:
        ep_id, split = row[4], row[7]
        other = self.db.execute("SELECT split FROM segments WHERE split != ? LIMIT 1", (split,)).fetchone()
        if other and (split in HELD_APART or other[0] in HELD_APART):
            raise ValueError(f"{self.path}: holds {other[0]!r} segments; a {split!r} one ({ep_id}) may not join them "
                             f"({' and '.join(HELD_APART)} segments share a store with no other split's)")
        self.db.execute(f"INSERT INTO segments ({_SEGMENT}) VALUES ({', '.join('?' * len(row))})", row)

    # -- reading ---------------------------------------------------------------------------

    def names(self, closed: bool = True) -> list[str]:
        """`ep_id` of the store's closed episodes, or of its open ones, in the order they were
        first saved."""
        rows = self.db.execute(f"SELECT e.ep_id FROM episodes e WHERE {'' if closed else 'NOT '}{_CLOSED} ORDER BY e.rowid")
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
         rollout_of, rollout_at, meta) = head
        if at is not None and n != at:
            raise LookupError(f"{self.path}: {ep_id} has no checkpoint {at} (its last is {n})")
        if at is not None and kind == "closing":
            raise ValueError(f"{self.path}: checkpoint {at} of {ep_id} is its closing checkpoint; play cannot "
                             f"continue from the end of an episode (its turn checkpoints are 0 to {at - 1})")
        ep = Episode(ep_id=ep_id, task_id=task_id, suite=suite, split=split, seg=seg,
                     parent=(parent, parent_events, parent_calls) if parent else None, baseline_of=baseline_of,
                     rollout_of=(rollout_of, rollout_at) if rollout_of else None,
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

    def baseline(self, name: str) -> Episode | None:
        """The twin the episode's `net_*` metrics subtract, when it is stored here and closed:
        the fork of the episode that names it in `baseline_of`. A rollout has the twin of the
        episode it replays, which is the segment it is stored after."""
        seg, ep_id, parent, rollout_of = self.db.execute(
            "SELECT seg, ep_id, parent, rollout_of FROM segments WHERE seg = ?", (self._seg(name),)).fetchone()
        while rollout_of is not None:
            seg, ep_id, parent, rollout_of = self.db.execute(
                "SELECT seg, ep_id, parent, rollout_of FROM segments WHERE seg = ?", (parent,)).fetchone()
        twin = self.db.execute(f"SELECT e.seg FROM segments e WHERE e.parent = ? AND e.baseline_of = ? AND {_CLOSED}",
                               (seg, ep_id)).fetchone()
        return self.load(twin[0]) if twin else None

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

    def copy_chain(self, name: str, into: Store, listed: bool = False) -> str:
        """Copy the episode's segment and every ancestor into another store, whole, skipping
        what is already there, and return its `seg`. `listed` also lists it there as one of
        that store's episodes; otherwise it arrives as an unlisted ancestor to fork from."""
        chain = self.chain(name)
        with into._write():
            for seg in chain:
                row = self.db.execute(f"SELECT {_SEGMENT} FROM segments WHERE seg = ?", (seg,)).fetchone()
                if into.db.execute("SELECT 1 FROM segments WHERE seg = ?", (seg,)).fetchone() is None:
                    into._insert_segment(row)
                into.db.executemany(
                    "INSERT OR IGNORE INTO checkpoints (seg, n, kind, events, calls, body) VALUES (?, ?, ?, ?, ?, ?)",
                    self.db.execute("SELECT seg, n, kind, events, calls, body FROM checkpoints WHERE seg = ?", (seg,)).fetchall())
            if listed:
                ep_id = row[4]
                have = into.db.execute("SELECT seg FROM episodes WHERE ep_id = ?", (ep_id,)).fetchone()
                if have is None:
                    into.db.execute("INSERT INTO episodes (ep_id, seg) VALUES (?, ?)", (ep_id, chain[-1]))
                elif have[0] != chain[-1]:
                    raise ValueError(f"{into.path}: another episode named {ep_id!r} is already stored there")
        return chain[-1]

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

def _pack(body: dict[str, Any]) -> bytes:
    return zlib.compress(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6)


def _unpack(blob: bytes) -> dict[str, Any]:
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


def _files_at(ep: Episode, n_events: int) -> dict[str, list]:
    """path -> [text, rev] as the episode's files stood when it held `n_events` events: its
    files now, with every later `file_write` undone."""
    files = {path: [text, ep.file_revs[path]] for path, text in ep.files.items()}
    for event in reversed(ep.events[n_events:]):
        if event.kind == "file_write":
            data = event.data
            if data["before"] is None:
                del files[data["path"]]
            else:
                files[data["path"]] = [data["before"], data["rev"] - 1]
    return files


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


def closed_episodes(run_dir: str | Path) -> list[Episode]:
    """The closed episodes of a run directory's store; none when the run has no store yet."""
    path = Path(run_dir) / STORE_FILE
    if not path.exists():
        return []
    with Store(path, readonly=True) as store:
        return list(store.episodes())
