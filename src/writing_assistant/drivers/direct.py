"""In-process tool loop over `LLMClient.chat`: the driver for 1-2 turn tasks, scripted actors
and tests. The session holds the conversation while it is open, and `open()` rebuilds it
from the episode, so a session opened on an episode already under way (a fork, or one
reopened after a delivery) goes on with what the actor was last sent. The
sampling temperature is the episode's (`ep.meta["temperature"]`, set by the suite's
`setup()`); `max_tokens` and `max_steps` are the run's, unless the episode fixes them for
the actor in `ep.state["sampling"][actor]`: a seat that is not the model under test keeps
its limits whatever run plays the episode."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

from writing_assistant.drivers import Session
from writing_assistant.episode import Episode
from writing_assistant.episode.tools import WORKSPACE_TOOLS, ToolTable, dispatch_tool
from writing_assistant.llm.client import LLMClient, assistant_message


@dataclass
class _Conversation:
    ep: Episode
    endpoint: str
    seed: int
    temperature: float
    max_tokens: int
    max_steps: int
    messages: list[dict[str, Any]]


class DirectDriver:
    def __init__(
        self,
        client: LLMClient,
        tools: ToolTable,
        *,
        max_tokens: int = 4096,
        max_steps: int = 16,
    ) -> None:
        self.client, self.tools = client, tools
        self.max_tokens, self.max_steps = max_tokens, max_steps

    async def open(self, ep: Episode, actor: str, endpoint: str) -> Session:
        # Seeded by the episode and actor, so a rerun of the same episode samples the same way.
        seed = int.from_bytes(hashlib.sha256(f"{ep.ep_id}\0{actor}".encode()).digest()[:4], "big") >> 1
        limits = ep.state.get("sampling", {}).get(actor, {})
        conv = _Conversation(ep, endpoint, seed, ep.meta["temperature"], limits.get("max_tokens", self.max_tokens),
                             limits.get("max_steps", self.max_steps), _conversation(ep, actor))
        return Session(slot="", ep_id=ep.ep_id, actor=actor, handle=conv)

    async def turn(self, session: Session, message: str) -> None:
        """Run the tool loop for one delivery (already logged by the Turn binding) until the
        model replies without a tool call. Past `max_steps` calls, when the model is cut off at
        `max_tokens` before replying, or when the server could not parse its tool call, the
        episode is truncated."""
        conv: _Conversation = session.handle
        ep, actor = conv.ep, session.actor
        turn_idx = ep.current_turn(actor)
        table = WORKSPACE_TOOLS | self.tools
        schemas = [table[name][0] for name in ep.actor_tools.get(actor, []) if name in table]
        conv.messages.append({"role": "user", "content": message})
        for _ in range(conv.max_steps):
            rec = await self.client.chat(
                conv.messages,
                schemas or None,
                endpoint=conv.endpoint,
                temperature=conv.temperature,
                max_tokens=conv.max_tokens,
                seed=conv.seed,
            )
            rec = replace(rec, actor=actor, turn_idx=turn_idx, at_event=len(ep.events))
            ep.calls.append(rec)
            conv.messages.append(assistant_message(rec))
            if not rec.tool_calls:
                if rec.finish_reason == "length":
                    _truncate(ep, f"{actor} was cut off at max_tokens={conv.max_tokens}")
                elif rec.finish_reason == "unparseable_tool_call":
                    _truncate(ep, f"{actor} made a tool call whose arguments were not valid JSON")
                return
            for call in rec.tool_calls:
                result = await dispatch_tool(ep, actor, call["name"], call["arguments"], self.tools)
                conv.messages.append(_tool_message(call, result))
        _truncate(ep, f"{actor} exceeded {conv.max_steps} model calls in one turn")

    async def close(self, session: Session) -> None:
        session.handle.messages.clear()


def _tool_message(call: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)}


def _conversation(ep: Episode, actor: str) -> list[dict[str, Any]]:
    """What `actor` has been sent and has answered so far. Before its first call that is its
    system prompt. After, it is the last call's messages, that call's reply and the results
    of the reply's tool calls: a turn stopped at `max_steps` ends on tool calls whose results
    reached no later call, and they are the actor's `tool` events since that call."""
    last = next((c for c in reversed(ep.calls) if c.actor == actor), None)
    if last is None:
        system = ep.state.get("system_prompts", {}).get(actor)
        return [{"role": "system", "content": system}] if system else []
    messages = [*last.messages, assistant_message(last)]
    if last.tool_calls:
        results = [e for e in ep.events[last.at_event:] if e.kind == "tool" and e.actor == actor]
        if [e.data["name"] for e in results] != [c["name"] for c in last.tool_calls]:
            raise ValueError(f"{ep.ep_id}: the log does not hold the results of {actor}'s last tool calls")
        messages += [_tool_message(call, e.data["result"]) for call, e in zip(last.tool_calls, results)]
    return messages


def _truncate(ep: Episode, reason: str) -> None:
    if ep.status == "running":
        ep.status = "truncated"
        ep.failure_reason = reason
