"""Event, CallRecord, Episode: the Episode is the only state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventKind = Literal["tool", "say", "deliver", "file_write", "world", "decision", "swap"]


@dataclass(frozen=True)
class Event:
    idx: int
    actor: str
    kind: EventKind
    data: dict[str, Any]
    visible_to: dict[str, str]  # perceiver -> "full" | "redacted" | "existence"; key "*" = everyone


@dataclass
class CallRecord:
    """One model call. `tools_advertised` holds bare tool names (`episode_read`)."""

    actor: str = ""
    turn_idx: int | None = None
    endpoint: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    response: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tools_advertised: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    ext: dict[str, Any] = field(default_factory=dict)


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
    actors: dict[str, str] = field(default_factory=dict)
    actor_tools: dict[str, list[str]] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)
    status: Literal["running", "completed", "truncated", "failed"] = "running"
    failure_reason: str | None = None
