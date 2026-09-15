"""Pinned GLM/Reka JSON calls with shared, persistent paid-call accounting."""

import fcntl
import json
import os
import time
import urllib.error
import urllib.request
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from writing_agent.catalog import fingerprint, save_json
from writing_agent.grading import JUDGE_SCHEMA, validate_judgment

MODEL = "z-ai/glm-5.3"
PROVIDER = "reka/fp8"
# Routing ceilings, not a promise about current promotional prices.
INPUT_PRICE = Decimal("1.20")
OUTPUT_PRICE = Decimal("4.00")
FEE_FACTOR = Decimal("1.055")


def _micro_usd(value):
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid cost")
    return int((amount * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


class OpenRouterClient:
    """Each call has a fresh system/user context; failures never trigger paid retries.

    Share one destination across task authors, reviewers and output judges when
    they share a budget. The ledger includes the platform-fee allowance. Raw
    responses (including reasoning) are retained before parsing their content.
    """

    def __init__(self, destination: Path, *, api_key: str, budget_usd: float):
        self.limit = _micro_usd(budget_usd)
        if not api_key or self.limit <= 0:
            raise ValueError("OpenRouter API key and positive budget are required")
        self.destination = destination
        self.api_key = api_key

    def _request(self, payload):
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenRouter HTTP {exc.code}; reservation retained") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            raise RuntimeError("OpenRouter response unavailable; reservation retained") from None

    def json_call(self, instructions: str, data: dict, *, role: str, max_tokens=8192):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 16384:
            raise ValueError("Output budget must be 1–16384 tokens including thinking")
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
            ],
            "provider": {
                "only": [PROVIDER],
                "allow_fallbacks": False,
                "quantizations": ["fp8"],
                "max_price": {"prompt": float(INPUT_PRICE), "completion": float(OUTPUT_PRICE)},
            },
            "reasoning": {"effort": "high", "exclude": False},
            "max_tokens": max_tokens,
            "temperature": 0.6 if role == "task_author" else 0,
            "stream": False,
        }
        identity = fingerprint({"payload": payload, "role": role, "version": 1})
        self.destination.mkdir(parents=True, exist_ok=True)
        with (self.destination / "ledger.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger_path = self.destination / "ledger.json"
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
            result_path = self.destination / (identity + ".json")
            if identity in ledger:
                if ledger[identity]["status"] != "completed":
                    raise RuntimeError("Prior call unresolved; inspect ledger before retrying")
                record = json.loads(result_path.read_text())
                if fingerprint(record) != ledger[identity]["result_hash"]:
                    raise ValueError("Cached API response hash mismatch")
            else:
                if any(item["status"] != "completed" for item in ledger.values()):
                    raise RuntimeError(
                        "Unresolved API reservation; inspect ledger before continuing"
                    )
                # Byte bound plus generous chat overhead; text-only requests, no server tools.
                bound = len(json.dumps(payload["messages"], ensure_ascii=False).encode()) + 4096
                reserve = int(
                    (
                        (bound * INPUT_PRICE + max_tokens * OUTPUT_PRICE) * FEE_FACTOR
                    ).to_integral_value(rounding=ROUND_CEILING)
                )
                spent = sum(item["charged_or_reserved_micro_usd"] for item in ledger.values())
                if spent + reserve > self.limit:
                    raise RuntimeError("Shared API budget exhausted before request")
                ledger[identity] = {
                    "status": "reserved",
                    "role": role,
                    "charged_or_reserved_micro_usd": reserve,
                }
                save_json(ledger_path, ledger)
                started = time.perf_counter()
                response = self._request(payload)
                record = {
                    "identity": identity,
                    "request": payload,
                    "response": response,
                    "role": role,
                    "latency_seconds": time.perf_counter() - started,
                }
                # Retain even malformed or interrupted model outputs for inspection.
                save_json(result_path, record)
                usage = response.get("usage", {})
                if "cost" in usage:
                    cost = _micro_usd(usage["cost"])
                    cost_basis = "provider_reported"
                else:
                    tokens = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
                    if any(type(value) is not int or value < 0 for value in tokens):
                        raise RuntimeError("Missing API usage; reservation retained")
                    cost = int(
                        (tokens[0] * INPUT_PRICE + tokens[1] * OUTPUT_PRICE).to_integral_value(
                            rounding=ROUND_CEILING
                        )
                    )
                    cost_basis = "routing_ceiling_estimate"
                accounted = int((cost * FEE_FACTOR).to_integral_value(rounding=ROUND_CEILING))
                record.update(
                    cost_micro_usd=cost, cost_basis=cost_basis, accounted_micro_usd=accounted
                )
                save_json(result_path, record)
                if accounted > reserve:
                    ledger[identity].update(
                        status="overrun",
                        charged_or_reserved_micro_usd=accounted,
                    )
                    save_json(ledger_path, ledger)
                    raise RuntimeError("API cost exceeded reservation; halt for review")
                ledger[identity].update(
                    status="completed",
                    charged_or_reserved_micro_usd=accounted,
                    result_hash=fingerprint(record),
                )
                save_json(ledger_path, ledger)
        response = record["response"]
        if response.get("provider", "").lower() not in {"reka", "reka ai"}:
            raise ValueError("Unverified response provider; raw response saved")
        returned_model = response.get("model", "")
        if returned_model != MODEL and not returned_model.startswith(MODEL + "-2026"):
            raise ValueError("Unexpected response model; raw response saved")
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Incomplete model response; raw response saved, no retry")
        value = json.loads(choice["message"]["content"])
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object; raw response saved")
        return {**record, "value": value}


def from_env(destination: Path, env_file: Path, *, budget_usd: float):
    from dotenv import dotenv_values

    key = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(env_file).get("OPENROUTER_API_KEY")
    return OpenRouterClient(destination, api_key=key, budget_usd=budget_usd)


class GLMGrader:
    """Use the existing blinded packet, rubric schema and scorecard application."""

    def __init__(self, client: OpenRouterClient):
        self.client = client

    def grade(self, packet: dict) -> dict:
        instructions = Path(__file__).with_name("grader_instructions.md").read_text()
        try:
            result = self.client.json_call(
                instructions,
                {"packet": packet, "response_schema": JUDGE_SCHEMA},
                role="output_judge",
                max_tokens=8192,
            )
            validate_judgment(packet, result["value"])
            return {
                "status": "ok",
                "identity": result["identity"],
                "judge": MODEL,
                "judge_provider": PROVIDER,
                "packet_hash": fingerprint(packet),
                "judgment": result["value"],
                "human_review": "pending",
                "cost_micro_usd": result["cost_micro_usd"],
            }
        except (ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
            return {
                "status": "grader_error",
                "reason": type(exc).__name__,
                "packet_hash": fingerprint(packet),
                "judge": MODEL,
                "judge_provider": PROVIDER,
            }
