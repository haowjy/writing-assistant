"""In-process tool loop over `LLMClient.chat`: the driver for 1-2 turn tasks, scripted actors
and tests. The conversation lives in the session, so a fork opens a new session."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
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
    messages: list[dict[str, Any]] = field(default_factory=list)


class DirectDriver:
    def __init__(
        self,
        client: LLMClient,
        tools: ToolTable,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_steps: int = 16,
    ) -> None:
        self.client, self.tools = client, tools
        self.temperature, self.max_tokens, self.max_steps = temperature, max_tokens, max_steps

    async def open(self, ep: Episode, actor: str, endpoint: str) -> Session:
        # Seeded by the episode and actor, so a rerun of the same episode samples the same way.
        seed = int.from_bytes(hashlib.sha256(f"{ep.ep_id}\0{actor}".encode()).digest()[:4], "big") >> 1
        conv = _Conversation(ep, endpoint, seed)
        system = ep.state.get("system_prompts", {}).get(actor)
        if system:
            conv.messages.append({"role": "system", "content": system})
        return Session(slot="", ep_id=ep.ep_id, actor=actor, handle=conv)

    async def turn(self, session: Session, message: str) -> str:
        """Run the tool loop for one delivery (already logged by the Turn binding) until the
        model replies without a tool call. Past `max_steps` calls the episode is truncated."""
        conv: _Conversation = session.handle
        ep, actor = conv.ep, session.actor
        turn_idx = ep.current_turn(actor)
        table = WORKSPACE_TOOLS | self.tools
        schemas = [table[name][0] for name in ep.actor_tools.get(actor, []) if name in table]
        conv.messages.append({"role": "user", "content": message})
        rec = None
        for _ in range(self.max_steps):
            rec = await self.client.chat(
                conv.messages,
                schemas or None,
                endpoint=conv.endpoint,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=conv.seed,
            )
            rec = replace(rec, actor=actor, turn_idx=turn_idx)
            ep.calls.append(rec)
            conv.messages.append(assistant_message(rec))
            if not rec.tool_calls:
                return rec.response
            for call in rec.tool_calls:
                result = await dispatch_tool(ep, actor, call["name"], call["arguments"], self.tools)
                conv.messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)}
                )
        if ep.status == "running":
            ep.status = "truncated"
            ep.failure_reason = f"{actor} exceeded {self.max_steps} model calls in one turn"
        return rec.response if rec else ""

    async def close(self, session: Session) -> None:
        session.handle.messages.clear()
