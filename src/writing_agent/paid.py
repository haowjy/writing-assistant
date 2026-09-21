"""Pinned DeepSeek JSON calls with shared, persistent paid-call accounting."""

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

MODEL = "deepseek-flash"
PROVIDER = "deepseek"
ENDPOINT = "https://api.deepseek.com/chat/completions"
# Micro-USD per token (= USD per million). Reservations and estimates use the peak ceiling.
INPUT_CACHE_HIT_PRICE = Decimal("0.003")
INPUT_CACHE_MISS_PRICE = Decimal("0.15")
OUTPUT_PRICE = Decimal("0.60")
PEAK_FACTOR = Decimal("2")


def _micro_usd(value):
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid cost")
    return int((amount * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


def _json_instructions(instructions: str) -> str:
    if "json" in instructions.lower():
        return instructions
    return instructions + "\nReturn a json object."


def accepted_model(returned: str) -> bool:
    return returned == MODEL or returned.startswith(MODEL + "-")


class TransportFailure(RuntimeError):
    def __init__(self, message, *, http_status=None, body=""):
        super().__init__(message)
        self.http_status = http_status
        self.body = body


class DeepSeekTransport:
    """Direct DeepSeek V4.1 Flash chat completions; no external routing layer."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def payload(self, instructions: str, data: dict, *, role: str, max_tokens: int, thinking: bool):
        return {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": _json_instructions(instructions)},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
            ],
            "thinking": {"type": "enabled" if thinking else "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "temperature": 0.6 if role == "task_author" else 0,
            "stream": False,
        }

    def request(self, payload):
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        raw = b""
        status = None
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read() if exc.fp is not None else b""
            raise TransportFailure(
                f"DeepSeek HTTP {exc.code}; reservation retained",
                http_status=exc.code,
                body=raw.decode("utf-8", errors="replace"),
            ) from None
        except (urllib.error.URLError, TimeoutError):
            raise TransportFailure("DeepSeek response unavailable; reservation retained") from None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise TransportFailure(
                "DeepSeek response unavailable; reservation retained",
                http_status=status,
                body=raw.decode("utf-8", errors="replace"),
            ) from None

    def reserved_micro_usd(self, payload, max_tokens: int) -> int:
        bound = len(json.dumps(payload["messages"], ensure_ascii=False).encode()) + 4096
        return int(
            (
                (bound * INPUT_CACHE_MISS_PRICE + max_tokens * OUTPUT_PRICE) * PEAK_FACTOR
            ).to_integral_value(rounding=ROUND_CEILING)
        )

    def charge(self, usage) -> tuple[int, int, str]:
        if not isinstance(usage, dict):
            raise RuntimeError("Missing API usage; reservation retained")
        completion = usage.get("completion_tokens")
        hit = usage.get("prompt_cache_hit_tokens")
        miss = usage.get("prompt_cache_miss_tokens")
        if hit is None and miss is None:
            prompt = usage.get("prompt_tokens")
            if type(prompt) is not int or prompt < 0:
                raise RuntimeError("Missing API usage; reservation retained")
            hit, miss = 0, prompt
        if (
            type(hit) is not int
            or hit < 0
            or type(miss) is not int
            or miss < 0
            or type(completion) is not int
            or completion < 0
        ):
            raise RuntimeError("Missing API usage; reservation retained")
        cost = int(
            (
                (
                    hit * INPUT_CACHE_HIT_PRICE
                    + miss * INPUT_CACHE_MISS_PRICE
                    + completion * OUTPUT_PRICE
                )
                * PEAK_FACTOR
            ).to_integral_value(rounding=ROUND_CEILING)
        )
        return cost, cost, "peak_ceiling_estimate"

    def verify(self, response: dict) -> None:
        if not accepted_model(response.get("model", "")):
            raise ValueError("Unexpected response model; raw response saved")


class PaidClient:
    """Each call has a fresh system/user context; failures never trigger paid retries.

    Share one destination across task authors, reviewers and output judges when
    they share a budget. Reservations use the peak cache-miss ceiling. Raw
    responses (including reasoning) are retained before parsing their content.
    """

    def __init__(self, destination: Path, *, api_key: str, budget_usd: float, transport=None):
        self.limit = _micro_usd(budget_usd)
        if not api_key or self.limit <= 0:
            raise ValueError("API key and positive budget are required")
        self.destination = destination
        self.transport = transport or DeepSeekTransport(api_key)

    def json_call(
        self, instructions: str, data: dict, *, role: str, max_tokens=8192, thinking=True
    ):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
            raise ValueError("Output budget must be 1–32768 tokens including thinking")
        if type(thinking) is not bool:
            raise ValueError("thinking must be a boolean")
        payload = self.transport.payload(
            instructions, data, role=role, max_tokens=max_tokens, thinking=thinking
        )
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
                if any(
                    item["status"] not in {"completed", "abandoned"} for item in ledger.values()
                ):
                    raise RuntimeError(
                        "Unresolved API reservation; inspect ledger before continuing"
                    )
                reserve = self.transport.reserved_micro_usd(payload, max_tokens)
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
                try:
                    response = self.transport.request(payload)
                except Exception as exc:
                    save_json(
                        self.destination / (identity + ".error.json"),
                        {
                            "identity": identity,
                            "http_status": getattr(exc, "http_status", None),
                            "body": getattr(exc, "body", ""),
                            "error": str(exc),
                        },
                    )
                    raise
                record = {
                    "identity": identity,
                    "request": payload,
                    "response": response,
                    "role": role,
                    "latency_seconds": time.perf_counter() - started,
                }
                # Retain even malformed or interrupted model outputs for inspection.
                save_json(result_path, record)
                cost, accounted, cost_basis = self.transport.charge(response.get("usage", {}))
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
        self.transport.verify(response)
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Incomplete model response; raw response saved, no retry")
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty model content; raw response saved, no retry")
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON content; raw response saved, no retry") from None
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object; raw response saved")
        return {**record, "value": value}


def reconcile_abandoned(destination: Path, identity: str, *, reason: str) -> None:
    """Move a stuck reservation to a terminal key; retain the charge; free the identity."""
    if not isinstance(identity, str) or not identity:
        raise ValueError("Expected a call identity")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Abandonment reason is required")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    abandoned_key = f"{identity}#abandoned"
    with (destination / "ledger.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger_path = destination / "ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        if identity not in ledger:
            entry = ledger.get(abandoned_key)
            if isinstance(entry, dict) and entry.get("status") == "abandoned":
                return
            raise RuntimeError("Identity is not in a reconcilable reserved state")
        entry = ledger[identity]
        if entry.get("status") != "reserved" or abandoned_key in ledger:
            raise RuntimeError("Identity is not in a reconcilable reserved state")
        ledger[abandoned_key] = {
            "status": "abandoned",
            "role": entry["role"],
            "charged_or_reserved_micro_usd": entry["charged_or_reserved_micro_usd"],
            "reason": reason,
        }
        del ledger[identity]
        save_json(ledger_path, ledger)


def from_env(destination: Path, env_file: Path, *, budget_usd: float):
    from dotenv import dotenv_values

    key = os.environ.get("DEEPSEEK_API_KEY") or dotenv_values(env_file).get("DEEPSEEK_API_KEY")
    return PaidClient(destination, api_key=key, budget_usd=budget_usd)


class OutputGrader:
    """Use the existing blinded packet, rubric schema and scorecard application."""

    def __init__(self, client: PaidClient):
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
