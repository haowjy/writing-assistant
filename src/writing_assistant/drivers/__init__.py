"""Drivers: one harness session per (episode, actor). Drivers know nothing about suites.

A driver is constructed by cli.py with the model client and the suite's tool table; `open`
gets the episode, so everything per-episode (actor tools, the actor's system prompt in
`ep.state["system_prompts"][actor]`) is read from it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from writing_assistant.episode import Episode


@dataclass(frozen=True)
class Session:
    slot: str  # serving-app slot (harness drivers); "" for in-process drivers
    ep_id: str
    actor: str
    handle: Any  # driver-private: process handle, conversation history, the Episode


class Driver(Protocol):
    async def open(self, ep: Episode, actor: str, endpoint: str) -> Session: ...

    async def turn(self, session: Session, message: str) -> str:
        """One delivery; the model calls episode_* tools until it replies."""
        ...

    async def close(self, session: Session) -> None: ...


DRIVERS: dict[str, Callable[..., Driver]] = {}
