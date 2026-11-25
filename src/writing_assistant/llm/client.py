"""`LLMClient`: chat and embeddings over endpoints named in models.toml. Client only; the
relay that harnesses call lives in drivers/serve.py.

Messages and tools use one neutral shape whatever the endpoint:
- messages are OpenAI chat messages: `system` / `user` / `assistant` (optional
  `tool_calls: [{id, type: "function", function: {name, arguments: <JSON text>}}]`) /
  `tool` (`tool_call_id`, `content`). `assistant_message(record)` turns a CallRecord back
  into the assistant message to append.
- a tool is `{"name", "description", "parameters"}` (the schema in a tool table).
A `CallRecord` comes back with `actor=""` and `turn_idx=None`; a driver fills them in.
Server responses are untrusted: missing `usage` gives `{}`, tool-call arguments may arrive as
JSON text or an object, and arguments that are not valid JSON are kept as the raw string (the
tool's validator then refuses them).
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from writing_assistant.episode import CallRecord

ANTHROPIC_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
EMBED_BATCH = 64
TIMEOUT = 600.0  # seconds; local models on small GPUs are slow; per endpoint: `timeout = ...`


class LLMClient:
    def __init__(self, models_toml: str | Path) -> None:
        self.endpoints: dict[str, dict[str, Any]] = tomllib.loads(Path(models_toml).read_text())["endpoints"]
        self._http: httpx.AsyncClient | None = None

    def is_local(self, endpoint: str) -> bool:
        """Whether the endpoint may receive `local_dev` text (`local = true`). The check
        belongs to whoever sends that text."""
        return bool(self._endpoint(endpoint).get("local", False))

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        endpoint: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> CallRecord:
        ep = self._endpoint(endpoint)
        if ep["kind"] == "openai":
            adapter = _openai_compat
        elif ep["kind"] == "anthropic":
            adapter = _anthropic
        else:
            raise ValueError(f"endpoint {endpoint!r} is kind {ep['kind']!r}, not a chat endpoint")
        rec = await adapter(self._client(), ep, self._headers(ep), messages, tools, temperature, max_tokens, seed)
        rec.endpoint = endpoint
        rec.messages = list(messages)
        rec.tools_advertised = [t["name"] for t in tools or []]
        return rec

    async def embed(self, texts: list[str], *, endpoint: str) -> np.ndarray:
        ep = self._endpoint(endpoint)
        if ep["kind"] != "openai-embeddings":
            raise ValueError(f"endpoint {endpoint!r} is kind {ep['kind']!r}, not openai-embeddings")
        rows: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            rows += await _openai_embeddings(self._client(), ep, self._headers(ep), texts[i : i + EMBED_BATCH])
        return np.asarray(rows, dtype=np.float32).reshape(len(texts), -1) if rows else np.zeros((0, 0), np.float32)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _endpoint(self, name: str) -> dict[str, Any]:
        if name not in self.endpoints:
            raise KeyError(f"unknown endpoint {name!r}; models.toml has {sorted(self.endpoints)}")
        return self.endpoints[name]

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=TIMEOUT)
        return self._http

    @staticmethod
    def _headers(ep: dict[str, Any]) -> dict[str, str]:
        var = ep.get("api_key_env", "")
        if not var:
            return {}
        key = os.environ.get(var)
        if not key:
            raise RuntimeError(f"endpoint model {ep['model']!r} needs ${var}, which is not set")
        if ep["kind"] == "anthropic":
            return {"x-api-key": key}
        return {"Authorization": f"Bearer {key}"}


def assistant_message(rec: CallRecord) -> dict[str, Any]:
    """The assistant message a CallRecord adds to the conversation (neutral shape)."""
    msg: dict[str, Any] = {"role": "assistant", "content": rec.response or None}
    if rec.tool_calls:
        msg["tool_calls"] = [
            {
                "id": c["id"],
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"]),
                },
            }
            for c in rec.tool_calls
        ]
    return msg


def _parse_args(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


async def _post(http: httpx.AsyncClient, url: str, headers: dict, body: dict, timeout: float) -> dict[str, Any]:
    resp = await http.post(url, json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


async def _openai_compat(http, ep, headers, messages, tools, temperature, max_tokens, seed) -> CallRecord:
    body: dict[str, Any] = {
        "model": ep["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        body["seed"] = seed
    if tools:
        body["tools"] = [{"type": "function", "function": t} for t in tools]
    data = await _post(http, ep["base_url"].rstrip("/") + "/chat/completions", headers, body, ep.get("timeout", TIMEOUT))
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise ValueError(f"{ep['model']}: response has no message: {str(data)[:300]}")
    msg = choices[0]["message"]
    calls = [
        {
            "id": c.get("id") or f"call_{i}",
            "name": c["function"]["name"],
            "arguments": _parse_args(c["function"].get("arguments")),
        }
        for i, c in enumerate(msg.get("tool_calls") or [])
    ]
    usage = data.get("usage") or {}
    return CallRecord(
        response=msg.get("content") or "",
        tool_calls=calls,
        usage={k: usage[src] for k, src in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")) if usage.get(src) is not None},
    )


async def _openai_embeddings(http, ep, headers, texts) -> list[list[float]]:
    body = {"model": ep["model"], "input": texts}
    data = await _post(http, ep["base_url"].rstrip("/") + "/embeddings", headers, body, ep.get("timeout", TIMEOUT))
    rows = sorted(data.get("data") or [], key=lambda d: d["index"])
    if len(rows) != len(texts):
        raise ValueError(f"{ep['model']}: {len(rows)} embeddings for {len(texts)} texts")
    return [r["embedding"] for r in rows]


def _to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Neutral messages -> (system, Messages API messages); adjacent same-role turns merge,
    so the tool results of one assistant turn travel in one user message."""
    system, out = [], []
    for m in messages:
        if m["role"] == "system":
            system.append(m["content"])
            continue
        if m["role"] == "tool":
            role, blocks = "user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}]
        else:
            role, blocks = m["role"], [{"type": "text", "text": m["content"]}] if m.get("content") else []
            for c in m.get("tool_calls") or []:
                args = _parse_args(c["function"]["arguments"])
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                               "input": args if isinstance(args, dict) else {}})
        if out and out[-1]["role"] == role:
            out[-1]["content"] += blocks
        else:
            out.append({"role": role, "content": blocks})
    return "\n\n".join(system), out


async def _anthropic(http, ep, headers, messages, tools, temperature, max_tokens, seed) -> CallRecord:
    # No temperature or seed: current Claude models reject sampling parameters (400), and the
    # Messages API has no seed. Claude runs are unpinned, as they are through Claude Code.
    system, msgs = _to_anthropic(messages)
    body: dict[str, Any] = {"model": ep["model"], "max_tokens": max_tokens, "messages": msgs}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
    url = ep.get("base_url", ANTHROPIC_URL).rstrip("/") + "/messages"
    data = await _post(http, url, headers | {"anthropic-version": ANTHROPIC_VERSION}, body, ep.get("timeout", TIMEOUT))
    if not isinstance(data.get("content"), list):
        raise ValueError(f"{ep['model']}: response has no content: {str(data)[:300]}")
    usage = data.get("usage") or {}
    return CallRecord(
        response="".join(b.get("text", "") for b in data["content"] if b.get("type") == "text"),
        tool_calls=[
            {"id": b["id"], "name": b["name"], "arguments": b.get("input", {})}
            for b in data["content"]
            if b.get("type") == "tool_use"
        ],
        usage={k: usage[k] for k in ("input_tokens", "output_tokens") if usage.get(k) is not None},
    )
