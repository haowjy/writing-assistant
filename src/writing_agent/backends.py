"""Interchangeable inference clients. No model weights required by the core package."""

import copy
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Completion:
    message: dict
    usage: dict = field(default_factory=dict)


class Backend(Protocol):
    """Completion clients may emit transport/model diagnostics before returning."""

    def complete(
        self, messages: list[dict], tools: list[dict], *, emit=lambda event: None
    ) -> Completion: ...


class ScriptedBackend:
    """Fixture playback for harness tests only; never a model-quality baseline."""

    def __init__(self, responses: list[dict]):
        self.responses = iter(copy.deepcopy(responses))

    def complete(
        self, messages: list[dict], tools: list[dict], *, emit=lambda event: None
    ) -> Completion:
        try:
            return Completion(next(self.responses))
        except StopIteration as exc:
            raise ValueError("Scripted fixture exhausted before a final answer") from exc


class ChatServerBackend:
    """Chat-completions transport for a separately managed server such as vLLM."""

    def __init__(self, config: dict):
        self.config = config

    def complete(
        self, messages: list[dict], tools: list[dict], *, emit=lambda event: None
    ) -> Completion:
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config.get("temperature", 0),
            "max_tokens": self.config.get("max_tokens", 1024),
            "seed": self.config.get("seed", 42),
        }
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.get("api_key_env", "CWA_API_KEY"))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.config["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.get("timeout", 120)) as response:
            result = json.load(response)
        choice = result["choices"][0]
        if choice.get("finish_reason") not in ("stop", "tool_calls"):
            raise ValueError(f"Incomplete generation: {choice.get('finish_reason')}")
        raw = choice["message"]
        # Keep only portable conversation fields; avoid passing server-specific metadata back.
        message = {key: raw[key] for key in ("role", "content", "tool_calls") if key in raw}
        return Completion(message, result.get("usage", {}))
