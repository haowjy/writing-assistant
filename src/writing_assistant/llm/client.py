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

Thinking never reaches `CallRecord.response`: a host's `reasoning_content` and an inline
`<think>...</think>` block both go to `CallRecord.reasoning`, so scorers, judges and the next
actor see only the reply, and the saved call still has the thinking.

`local_only()` is the one guard on `local_dev` text: inside it, every chat and embed call to
an endpoint without `local = true` raises `LocalOnly` before anything is sent.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from writing_assistant.episode import CallRecord

log = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
EMBED_BATCH = 64
TIMEOUT = 600.0  # seconds; local models on small GPUs are slow; per endpoint: `timeout = ...`
CONCURRENCY = 16  # requests in flight per endpoint; per endpoint: `concurrency = ...`
RETRIES = 6  # attempts on 429, 5xx and transport errors, backing off 2, 4, 8, ... s (or Retry-After)
# Each adapter's stop words mapped to one vocabulary; anything else is kept as the host said it.
_FINISH = {
    "stop": "stop", "end_turn": "stop", "stop_sequence": "stop", "eos": "stop",
    "length": "length", "max_tokens": "length", "model_length": "length",
    "tool_calls": "tool_calls", "tool_use": "tool_calls", "function_call": "tool_calls",
    "content_filter": "refusal", "refusal": "refusal",
}

_LOCAL_ONLY: contextvars.ContextVar[bool] = contextvars.ContextVar("local_only", default=False)


class LocalOnly(RuntimeError):
    """A hosted endpoint was asked to receive `local_dev` text."""


@contextmanager
def local_only(active: bool = True) -> Iterator[None]:
    """Within this block (and the asyncio tasks it starts), only `local = true` endpoints
    may be called. cli.py enters it around everything that handles a `local_dev` row."""
    token = _LOCAL_ONLY.set(active or _LOCAL_ONLY.get())
    try:
        yield
    finally:
        _LOCAL_ONLY.reset(token)


class LLMClient:
    def __init__(self, models_toml: str | Path) -> None:
        # models.toml is parsed here once; judges_from and cli.py read `config`.
        self.config: dict[str, Any] = tomllib.loads(Path(models_toml).read_text())
        self.endpoints: dict[str, dict[str, Any]] = self.config["endpoints"]
        self._http: httpx.AsyncClient | None = None
        self._slots: dict[str, asyncio.Semaphore] = {}

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
        ep = self._guard(endpoint)
        if ep["kind"] == "openai":
            adapter = _openai_compat
        elif ep["kind"] == "anthropic":
            adapter = _anthropic
        else:
            raise ValueError(f"endpoint {endpoint!r} is kind {ep['kind']!r}, not a chat endpoint")
        async with self._slot(endpoint):
            rec = await adapter(self._client(), ep, self._headers(ep), messages, tools, temperature, max_tokens, seed)
        rec.endpoint = endpoint
        rec.messages = list(messages)
        rec.tools_advertised = [t["name"] for t in tools or []]
        return rec

    async def embed(self, texts: list[str], *, endpoint: str) -> np.ndarray:
        ep = self._guard(endpoint)
        if ep["kind"] != "openai-embeddings":
            raise ValueError(f"endpoint {endpoint!r} is kind {ep['kind']!r}, not openai-embeddings")
        rows: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            async with self._slot(endpoint):
                rows += await _openai_embeddings(self._client(), ep, self._headers(ep), texts[i : i + EMBED_BATCH])
        return np.asarray(rows, dtype=np.float32).reshape(len(texts), -1) if rows else np.zeros((0, 0), np.float32)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _guard(self, name: str) -> dict[str, Any]:
        ep = self._endpoint(name)
        if _LOCAL_ONLY.get() and not ep.get("local", False):
            raise LocalOnly(f"endpoint {name!r} is hosted (no `local = true`) and this call carries local_dev text")
        return ep

    def _slot(self, name: str) -> asyncio.Semaphore:
        if name not in self._slots:
            self._slots[name] = asyncio.Semaphore(self.endpoints[name].get("concurrency", CONCURRENCY))
        return self._slots[name]

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
    """POST with a slow retry on 429, 5xx and transport errors; any other status raises at
    once. Errors name the URL and status only, never the headers (they carry the key)."""
    for attempt in range(1, RETRIES + 1):
        wait = 2.0**attempt
        try:
            resp = await http.post(url, json=body, headers=headers, timeout=timeout)
        except httpx.TransportError as e:
            if attempt == RETRIES:
                raise
            log.warning("%s: %s; retry %d in %.0fs", url, type(e).__name__, attempt, wait)
            await asyncio.sleep(wait)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == RETRIES:
                resp.raise_for_status()
            after = resp.headers.get("retry-after", "")
            wait = float(after) if after.replace(".", "", 1).isdigit() else wait
            log.warning("%s: HTTP %d; retry %d in %.0fs", url, resp.status_code, attempt, wait)
            await asyncio.sleep(wait)
            continue
        if resp.is_error:
            raise httpx.HTTPStatusError(f"HTTP {resp.status_code} from {url}: {resp.text[:300]}", request=resp.request, response=resp)
        return resp.json()
    raise AssertionError("unreachable")


def split_reasoning(content: str, reasoning: str = "") -> tuple[str, str]:
    """(reply, reasoning). Text after the last `</think>` is the reply (some servers drop the
    opening tag); a `<think>` never closed means the reply was cut off while thinking, so
    the whole text is reasoning and the reply is empty."""
    if "</think>" in content:
        head, content = content.rsplit("</think>", 1)
        reasoning = "\n\n".join(x for x in (reasoning, head.replace("<think>", "", 1).strip()) if x)
    elif content.lstrip().startswith("<think>"):
        reasoning = "\n\n".join(x for x in (reasoning, content.lstrip()[len("<think>"):].strip()) if x)
        content = ""
    return content.strip(), reasoning


def _usage(usage: dict[str, Any], names: tuple[tuple[str, str], ...]) -> dict[str, float]:
    out = {k: usage[src] for k, src in names if isinstance(usage.get(src), int | float)}
    if isinstance(usage.get("estimated_cost"), int | float):  # DeepInfra reports the call's price
        out["cost_usd"] = usage["estimated_cost"]
    return out


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
    body |= ep.get("params", {})  # per-endpoint extras, e.g. chat_template_kwargs
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
    reply, reasoning = split_reasoning(msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning") or "")
    finish = choices[0].get("finish_reason") or ""
    return CallRecord(
        response=reply,
        tool_calls=calls,
        usage=_usage(data.get("usage") or {}, (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"))),
        finish_reason=_FINISH.get(finish, finish),
        reasoning=reasoning,
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
    finish = data.get("stop_reason") or ""
    return CallRecord(
        response="".join(b.get("text", "") for b in data["content"] if b.get("type") == "text"),
        reasoning="".join(b.get("thinking", "") for b in data["content"] if b.get("type") == "thinking"),
        finish_reason=_FINISH.get(finish, finish),
        tool_calls=[
            {"id": b["id"], "name": b["name"], "arguments": b.get("input", {})}
            for b in data["content"]
            if b.get("type") == "tool_use"
        ],
        usage=_usage(data.get("usage") or {}, (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"))),
    )
