"""Event, CallRecord, Episode: the Episode is the only state.

Suite state is plain JSON in `Episode.state`; files are immutable strings in a dict, so a
fork is a copy of a few containers. A saved episode (`to_dict`) is the whole state.

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
from dataclasses import asdict, dataclass, field, replace
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
    parent_id: str | None = None
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

    def fork(self, condition: str, swap: dict[str, Any] | None = None) -> Episode:
        """A copy that shares nothing mutable with this episode except Event and CallRecord
        objects (never mutated). Its id is `<parent>~<condition>`.

        `swap = {"event_idx": i, "data": {...}, "state": {...}, "restricted": [seat, ...]}`
        replaces event i's data and deep-merges `state` leaf by leaf. It refuses an event that
        a restricted seat (the seats whose ignorance the twin tests) or everyone ("*") could
        perceive."""
        child = Episode(
            ep_id=f"{self.ep_id}~{condition}",
            task_id=self.task_id,
            suite=self.suite,
            split=self.split,
            parent_id=self.ep_id,
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

    # -- serialisation ---------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Episode:
        d = dict(d)
        d["events"] = [Event(**e) for e in d["events"]]
        d["calls"] = [CallRecord(**c) for c in d["calls"]]
        return cls(**d)


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
