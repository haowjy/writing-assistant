"""Serial paid judgments with a persistent reservation ledger and no automatic retries."""

import fcntl
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from writing_agent.catalog import fingerprint, save_json


class AnthropicGrader:
    def __init__(self, destination: Path, *, api_key: str, budget_usd=10.0):
        if not api_key or budget_usd <= 0:
            raise ValueError("API key and positive budget required")
        self.destination = destination
        self.api_key = api_key
        self.limit = int(budget_usd * 1_000_000)

    def _request(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/" + path,
            data=json.dumps(payload).encode(),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Never log headers, credentials or provider response bodies.
            raise RuntimeError(f"Anthropic HTTP {exc.code}; no automatic retry") from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("Anthropic network failure; reservation retained") from None

    def grade(self, prompt: str, *, model="claude-sonnet-4-6", max_tokens=4096) -> dict:
        if model != "claude-sonnet-4-6" or not 1 <= max_tokens <= 4096:
            raise ValueError("This price schedule covers Sonnet 4.6, up to 4096 output tokens")
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        key = fingerprint(payload)
        self.destination.mkdir(parents=True, exist_ok=True)
        with (self.destination / "ledger.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger_path = self.destination / "ledger.json"
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
            result_path = self.destination / (key + ".json")
            if key in ledger:
                if ledger[key]["status"] == "completed":
                    return json.loads(result_path.read_text())
                raise RuntimeError("Prior judgment unresolved; review ledger before retrying")
            count = self._request(
                "messages/count_tokens", {"model": model, "messages": payload["messages"]}
            )
            # Reserve conservative text-input bound plus the maximum output charge.
            bound = max(count["input_tokens"] + 1024, len(prompt.encode()) + 1024)
            reserve = bound * 3 + max_tokens * 15  # micro-USD, $3/$15 per million tokens
            total = sum(x["charged_or_reserved_micro_usd"] for x in ledger.values())
            if total + reserve > self.limit:
                raise RuntimeError("Paid grading budget exhausted before request")
            ledger[key] = {
                "status": "reserved",
                "charged_or_reserved_micro_usd": reserve,
                "counted_input_tokens": count["input_tokens"],
                "input_price_per_million": 3,
                "output_price_per_million": 15,
            }
            save_json(ledger_path, ledger)
            response = self._request("messages", payload)
            usage = response["usage"]
            cost = usage["input_tokens"] * 3 + usage["output_tokens"] * 15
            result = {
                "request": payload,
                "response": response,
                "cost_micro_usd": cost,
                "text": "\n".join(b["text"] for b in response["content"] if b["type"] == "text"),
            }
            save_json(result_path, result)
            if cost > reserve:
                raise RuntimeError("Charge exceeded reservation; halt for ledger review")
            ledger[key] = {
                **ledger[key],
                "status": "completed",
                "charged_or_reserved_micro_usd": cost,
                "usage": usage,
            }
            save_json(ledger_path, ledger)
            return result


def from_env(destination: Path, env_file: Path, *, budget_usd=10.0) -> AnthropicGrader:
    from dotenv import dotenv_values

    key = os.environ.get("ANTHROPIC_API_KEY") or dotenv_values(env_file).get("ANTHROPIC_API_KEY")
    return AnthropicGrader(destination, api_key=key, budget_usd=budget_usd)
