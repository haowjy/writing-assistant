"""Fakes shared by every package's tests: no network, no GPU.

- `fake_llm(replies, ...)` -> FakeLLM, the `LLMClient` surface (`chat`, `embed`, `is_local`).
- `fake_provider(...)` -> FakeProvider, a search `Provider`.
- `fake_turn(replies, tools)` -> a `Turn` over the real direct driver and a FakeLLM.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from writing_assistant.drivers import Session
from writing_assistant.drivers.direct import DirectDriver
from writing_assistant.episode import CallRecord, Episode
from writing_assistant.episode.tools import ToolTable
from writing_assistant.search.providers import Doc, Hit

# A scripted reply: plain text, or {"text": ..., "tool_calls": [{"name": ..., "arguments": {...}}]},
# or a callable (messages, tools, endpoint) -> one of those.
Reply = str | dict[str, Any] | Callable[[list[dict], list[dict] | None, str], Any]


class FakeLLM:
    """Scripted `LLMClient`. `replies` is one queue, or a dict endpoint -> queue."""

    def __init__(
        self,
        replies: list[Reply] | dict[str, list[Reply]] = (),
        *,
        vectors: dict[str, list[float]] | None = None,
        dim: int = 16,
        local: tuple[str, ...] = (),
    ) -> None:
        self.replies = {k: list(v) for k, v in replies.items()} if isinstance(replies, dict) else list(replies)
        self.vectors = vectors or {}
        self.dim = dim
        self.local = set(local)
        self.requests: list[dict[str, Any]] = []  # every chat() call, for prompt assertions

    def _next(self, endpoint: str) -> Reply:
        queue = self.replies[endpoint] if isinstance(self.replies, dict) else self.replies
        if not queue:
            raise AssertionError(f"FakeLLM ran out of scripted replies for {endpoint!r}")
        return queue.pop(0)

    async def chat(self, messages, tools=None, *, endpoint, temperature, max_tokens, seed=None) -> CallRecord:
        self.requests.append(
            {"messages": list(messages), "tools": tools, "endpoint": endpoint, "temperature": temperature}
        )
        reply = self._next(endpoint)
        if callable(reply):
            reply = reply(messages, tools, endpoint)
        if isinstance(reply, str):
            reply = {"text": reply}
        calls = [
            {"id": c.get("id", f"call_{i}"), "name": c["name"], "arguments": c.get("arguments", {})}
            for i, c in enumerate(reply.get("tool_calls", []))
        ]
        text = reply.get("text", "")
        return CallRecord(
            endpoint=endpoint,
            messages=list(messages),
            response=text,
            tool_calls=calls,
            tools_advertised=[t["name"] for t in tools or []],
            usage={"input_tokens": sum(len(str(m.get("content") or "")) for m in messages), "output_tokens": len(text)},
        )

    async def embed(self, texts: list[str], *, endpoint: str) -> np.ndarray:
        """`vectors[text]` when given, else a unit vector seeded by (endpoint, text): same text, same vector."""
        rows = []
        for t in texts:
            if t in self.vectors:
                rows.append(np.asarray(self.vectors[t], dtype=np.float32))
                continue
            seed = int.from_bytes(hashlib.sha256(f"{endpoint}\0{t}".encode()).digest()[:8], "big")
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)

    def is_local(self, endpoint: str) -> bool:
        return endpoint in self.local


class FakeProvider:
    """Scripted `Provider`. `fetch` returns None for unknown ids, as a blocked upstream does."""

    def __init__(
        self,
        name: str = "fake",
        kind: str = "web",
        *,
        hits: dict[str, list[Hit]] | None = None,
        docs: dict[str, Doc] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.name, self.kind = name, kind
        self.hits, self.docs, self.delay = hits or {}, docs or {}, delay
        self.searches: list[tuple[str, dict]] = []  # upstream calls, for coalescing tests
        self.fetches: list[str] = []

    async def search(self, query: str, **facets: Any) -> list[Hit]:
        self.searches.append((query, facets))
        await asyncio.sleep(self.delay)
        return list(self.hits.get(query, []))

    async def fetch(self, canonical_id: str) -> Doc | None:
        self.fetches.append(canonical_id)
        await asyncio.sleep(self.delay)
        return self.docs.get(canonical_id)


def _fake_turn(replies: list[Reply] | dict[str, list[Reply]], tools: ToolTable | None = None, **driver_kw: Any):
    """A `Turn` bound the way cli.py binds one: a session per (episode, actor), the delivery
    logged with `Episode.deliver`, then the real `DirectDriver` over a `FakeLLM` (so tool calls
    run through `dispatch_tool` and every call is attested). `replies` follow FakeLLM: one
    queue, or a dict keyed by endpoint (`ep.actors[actor]`); a tool-using turn takes one reply
    per model call. `turn.llm` exposes the requests, to assert on what a model was shown."""
    llm = FakeLLM(replies)
    driver = DirectDriver(llm, tools or {}, **driver_kw)
    sessions: dict[tuple[str, str], Session] = {}

    async def turn(ep: Episode, actor: str, message: str) -> str:
        key = (ep.ep_id, actor)
        if key not in sessions:
            sessions[key] = await driver.open(ep, actor, ep.actors[actor])
        ep.deliver(actor, message)
        return await driver.turn(sessions[key], message)

    turn.llm = llm
    return turn


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def fake_provider():
    return FakeProvider


@pytest.fixture
def fake_turn():
    return _fake_turn
