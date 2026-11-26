"""Event, CallRecord, Episode: the Episode is the only state.

Suite state is plain JSON in `Episode.state`; files are immutable strings in a dict, so a
fork is a copy of a few containers. An episode holds no history of itself: what it was at an
earlier point is kept by the store (`episode/segments.py`), which saves it as a segment of a
tree and needs `seg`, `parent` and `forked_from` here for that.

Conventions other packages rely on:
- `state["file_acl"]`: `{path or "dir/": [actor, ...]}`. A covered path is reachable only by
  the listed actors (the longest matching key wins); uncovered paths are open to everyone.
- `state["system_prompts"]`: `{actor: prompt}`, read by drivers at `open()`.
- An event's `data["redacted"]`, when present, is what a `redacted` perceiver sees.
- Who wrote a text is on the record: a `deliver` event's message and a file's content before
  its first `file_write` came from the benchmark; `file_write` events and `tool` events
  carry the actor that caused them.
"""

from __future__ import annotations

import copy
import posixpath
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal

EventKind = Literal["tool", "say", "deliver", "file_write", "world", "decision", "swap"]
Visibility = Literal["full", "redacted", "existence"]


@dataclass(frozen=True)
class Event:
    """Never mutate `data` in place: a fork shares Event objects with its parent."""

    idx: int
    actor: str
    kind: EventKind
    data: dict[str, Any]
    visible_to: dict[str, str]  # perceiver -> "full" | "redacted" | "existence"; key "*" = everyone


@dataclass
class CallRecord:
    """One model call. `tools_advertised` holds bare tool names (`episode_read`): a harness
    driver strips its MCP prefix (`mcp__episode__`) before recording."""

    actor: str = ""
    turn_idx: int | None = None  # the delivery this call served; None outside a delivery
    # How many events the episode held when the call was appended: the results of its tool
    # calls are the actor's `tool` events from there on.
    at_event: int = 0
    endpoint: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    response: str = ""  # the reply text only: never the model's reasoning
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {id, name, arguments}
    tools_advertised: list[str] = field(default_factory=list)
    # input_tokens, output_tokens, cost_usd (when the host reports it); {} if unreported
    usage: dict[str, float] = field(default_factory=dict)
    # Why the model stopped: "stop", "length", "tool_calls", "refusal", "unparseable_tool_call"
    # (the server could not parse its tool call: llm.client.UnparseableToolCall), or the host's own word;
    # "" when the host did not say. A refusal or a length cut-off is not an empty reply.
    finish_reason: str = ""
    reasoning: str = ""  # thinking the host returned beside the reply; saved, never scored or replayed
    ext: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeView:
    """What one actor may perceive: masked events (renumbered, so gaps reveal nothing) and
    the files the ACL lets it reach."""

    actor: str
    events: list[Event]
    files: dict[str, str]
    file_revs: dict[str, int]


def norm_path(path: str) -> str:
    """Canonical workspace path; ValueError for anything outside the workspace.

    Every ACL check goes through this, so `./secrets.json` and `a/../secrets.json` are
    `secrets.json`."""
    if not isinstance(path, str) or not path.strip() or "\\" in path or "\0" in path:
        raise ValueError(f"bad path {path!r}")
    norm = posixpath.normpath(path)
    if norm.startswith("/") or norm == "." or norm == ".." or norm.startswith("../"):
        raise ValueError(f"bad path {path!r}: paths are relative and stay inside the workspace")
    return norm


@dataclass
class Episode:
    ep_id: str
    task_id: str
    suite: str
    split: Literal["train", "dev", "test", "local_dev"]
    # Storage identity, unique in every store; `ep_id` is the readable name scores, judgments
    # and sampling seeds use, unique within one store.
    seg: str = field(default_factory=lambda: uuid.uuid4().hex)
    # (seg, events, calls) of the episode this one was forked from and how much of its log it
    # took. It says what this episode is stored after and nothing else: twins are `baseline_of`.
    parent: tuple[str, int, int] | None = None
    # The parent itself while this process still holds it, so that saving a fork can save an
    # ancestor nothing else saved; None on an episode loaded from a store.
    forked_from: Episode | None = field(default=None, repr=False, compare=False)
    baseline_of: str | None = None  # ep_id of the episode this one is the twin (the baseline) of
    # (ep_id, turn checkpoint) of the stored episode this one replays from: set by `rollout()`.
    # Readers keep rollouts apart from a run's task episodes.
    rollout_of: tuple[str, int] | None = None
    # The group a rollout is a member of: the rollouts one run made of one checkpoint in one
    # round (the id is `run.group_id`'s). None on anything that is not a rollout.
    group: str | None = None
    condition: str = "base"
    meta: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    file_revs: dict[str, int] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    actors: dict[str, str] = field(default_factory=dict)  # actor -> endpoint name or "scripted"
    actor_tools: dict[str, list[str]] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)
    status: Literal["running", "completed", "truncated", "failed"] = "running"
    failure_reason: str | None = None

    # -- recording -------------------------------------------------------------------------

    def log(self, actor: str, kind: EventKind, data: dict[str, Any], visible_to: dict[str, str]) -> Event:
        ev = Event(len(self.events), actor, kind, data, visible_to)
        self.events.append(ev)
        return ev

    def deliver(self, actor: str, message: str) -> int:
        """Record one delivery to a model actor and return its `turn_idx`; the Turn binding
        calls this before the driver runs. The delivery snapshots the actor's tools, which
        `complete()` requires a call of this turn to have advertised."""
        turn_idx = sum(1 for e in self.events if e.kind == "deliver")
        data = {"turn_idx": turn_idx, "message": message, "tools": list(self.actor_tools.get(actor, []))}
        self.log(actor, "deliver", data, {actor: "full"})
        return turn_idx

    def current_turn(self, actor: str) -> int:
        """`turn_idx` of the latest delivery to `actor` (what a driver stamps on its calls)."""
        for ev in reversed(self.events):
            if ev.kind == "deliver" and ev.actor == actor:
                return ev.data["turn_idx"]
        raise LookupError(f"no delivery to {actor!r} in {self.ep_id}")

    def last_reply(self) -> str:
        """The reply to the latest delivery: the text of the last call made for it. "" before
        the first delivery, and when no call answered it."""
        for ev in reversed(self.events):
            if ev.kind == "deliver":
                turn = (ev.actor, ev.data["turn_idx"])
                return next((c.response for c in reversed(self.calls) if (c.actor, c.turn_idx) == turn), "")
        return ""

    def write_file(self, actor: str, path: str, text: str, visible_to: dict[str, str] | None = None) -> int:
        """Write a file (no ACL or revision check: callers acting for a model go through
        `tools.py`) and log a `file_write` event with before/after. Returns the new rev."""
        path = norm_path(path)
        rev = self.file_revs.get(path, 0) + 1
        data = {"path": path, "before": self.files.get(path), "after": text, "rev": rev}
        self.files[path] = text
        self.file_revs[path] = rev
        self.log(actor, "file_write", data, {actor: "full"} if visible_to is None else visible_to)
        return rev

    # -- access ----------------------------------------------------------------------------

    def readable(self, actor: str, path: str) -> bool:
        """Whether the file ACL in `state["file_acl"]` lets `actor` reach `path`."""
        path = norm_path(path)
        best, allowed = -1, True
        for key, actors in self.state.get("file_acl", {}).items():
            is_dir = key.endswith("/")
            key = norm_path(key) + ("/" if is_dir else "")
            if (path == key or (is_dir and path.startswith(key))) and len(key) > best:
                best, allowed = len(key), actor in actors
        return allowed

    def view(self, actor: str) -> EpisodeView:
        events = []
        for ev in self.events:
            mode = ev.visible_to.get(actor) or ev.visible_to.get("*")
            if mode is None:
                continue
            idx = len(events)
            if mode == "full":
                events.append(Event(idx, ev.actor, ev.kind, ev.data, {actor: mode}))
            elif mode == "redacted":
                data = {"redacted": ev.data["redacted"]} if "redacted" in ev.data else {}
                events.append(Event(idx, ev.actor, ev.kind, data, {actor: mode}))
            elif mode == "existence":
                events.append(Event(idx, "", ev.kind, {}, {actor: mode}))
            else:
                raise ValueError(f"event {ev.idx}: unknown visibility {mode!r}")
        files = {p: t for p, t in self.files.items() if self.readable(actor, p)}
        return EpisodeView(actor, events, files, {p: self.file_revs[p] for p in files})

    # -- branching and outcome -------------------------------------------------------------

    def fork(self, condition: str, swap: dict[str, Any] | None = None, *, baseline: bool = False) -> Episode:
        """A copy that shares nothing mutable with this episode except Event and CallRecord
        objects (never mutated). Its id is `<parent>~<condition>`. `baseline` makes the copy
        this episode's twin: the baseline its `net_*` metrics subtract.

        `swap = {"event_idx": i, "data": {...}, "state": {...}, "restricted": [seat, ...]}`
        replaces event i's data and deep-merges `state` leaf by leaf. It refuses an event that
        a restricted seat (the seats whose ignorance the twin tests) or everyone ("*") could
        perceive."""
        child = Episode(
            ep_id=f"{self.ep_id}~{condition}",
            task_id=self.task_id,
            suite=self.suite,
            split=self.split,
            parent=(self.seg, len(self.events), len(self.calls)),
            forked_from=self,
            baseline_of=self.ep_id if baseline else None,
            condition=condition,
            meta=copy.deepcopy(self.meta),
            events=list(self.events),
            files=dict(self.files),
            file_revs=dict(self.file_revs),
            state=copy.deepcopy(self.state),
            actors=dict(self.actors),
            actor_tools={a: list(t) for a, t in self.actor_tools.items()},
            calls=list(self.calls),
        )
        if swap is not None:
            i = swap["event_idx"]
            ev = child.events[i]
            seen_by = set(ev.visible_to) & ({"*"} | set(swap["restricted"]))
            if seen_by:
                raise ValueError(f"swap of event {i}: perceived by {sorted(seen_by)}")
            child.events[i] = replace(ev, data=copy.deepcopy(swap["data"]))
            _deep_merge(child.state, copy.deepcopy(swap.get("state", {})))
        return child

    def begin(self, ep_id: str, task_id: str, suite: str) -> Episode:
        """A new episode on this one's files: what a new task set up at this position (a
        closed, stored episode) starts as. It takes the files, their revisions and the
        split, and nothing of the session: no events, calls, actors or meta, so its seats
        start new conversations whatever they are named. It is stored after this episode's
        start (`parent`), so its own start holds only the files this episode's play changed.

        Its state is empty but for the file access list, kept with every restricted path
        readable by nobody: the list names the old task's seats, and a path it does not
        cover is open to everyone, so dropping it would open every restricted file to every
        new seat. The new task's `setup` grants its seats what they may read."""
        closed = {path: [] for path in self.state.get("file_acl", {})}
        return Episode(ep_id=ep_id, task_id=task_id, suite=suite, split=self.split, parent=(self.seg, 0, 0),
                       files=dict(self.files), file_revs=dict(self.file_revs),
                       state={"file_acl": closed} if closed else {})

    def rollout(self, at: int, i: int, group: str) -> Episode:
        """Member `i` of the group `group` of rollouts of this episode as the store loaded it
        at its turn checkpoint `at`: a fork under the same condition to play on from there,
        named `<ep_id>@<at>.<group>.<i>`. A group belongs to one run, so two runs' rollouts
        of one checkpoint never share a name, and each samples with its own seed. It has no
        twin and is scored on its own metrics only. A twin's rollout is the same thing: it
        replays the control checkpoint and is the baseline of nothing."""
        child = self.fork(self.condition)
        child.ep_id, child.rollout_of, child.group = f"{self.ep_id}@{at}.{group}.{i}", (self.ep_id, at), group
        return child

    def fail(self, reason: str) -> None:
        self.status = "failed"
        self.failure_reason = reason

    def complete(self) -> None:
        """Positive check: every delivery has at least one call by that actor, for that
        `turn_idx`, advertising every tool the actor held at delivery. Otherwise the episode
        fails `mcp_unattested`. A failed episode stays failed; a truncated one stays truncated."""
        if self.status == "failed":
            return
        for ev in self.events:
            if ev.kind != "deliver":
                continue
            need = set(ev.data["tools"])
            if not any(
                c.actor == ev.actor and c.turn_idx == ev.data["turn_idx"] and need <= set(c.tools_advertised)
                for c in self.calls
            ):
                self.fail("mcp_unattested")
                return
        if self.status == "running":
            self.status = "completed"


def strings(value: Any) -> list[str]:
    """Every string inside a JSON value (dict values and list items, recursively)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def _deep_merge(into: dict[str, Any], new: dict[str, Any]) -> None:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            _deep_merge(into[k], v)
        else:
            into[k] = v
