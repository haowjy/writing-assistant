"""Drivers: one harness session per (episode, actor). Drivers know nothing about suites.

A driver is constructed by cli.py with the model client and the suite's tool table; `open`
gets the episode, so everything per-episode (actor tools, the actor's system prompt in
`ep.state["system_prompts"][actor]`, the sampling temperature in `ep.meta["temperature"]`)
is read from it, and so is the conversation so far when the episode is already under way: a
driver keeps nothing between turns that it could not rebuild from the episode.
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

    async def turn(self, session: Session, message: str) -> None:
        """One delivery; the model calls episode_* tools until it replies. Every model call
        is appended to `ep.calls`, and the reply is the last one's text (`Episode.last_reply`)."""
        ...

    async def close(self, session: Session) -> None: ...


from writing_assistant.drivers.direct import DirectDriver  # noqa: E402  (needs Session above)

# name -> constructor; cli.py builds the chosen one with the client and the suite's tools.
DRIVERS: dict[str, Callable[..., Driver]] = {"direct": DirectDriver}
