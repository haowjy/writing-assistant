"""Paid-call accounting and admission boundaries, without network or models."""

import copy
import io
import json
import tempfile
import unittest
import urllib.error
from decimal import ROUND_CEILING
from pathlib import Path
from unittest.mock import patch

from writing_agent.catalog import fingerprint
from writing_agent.grading import apply_judgment, grading_packet
from writing_agent.paid import (
    ENDPOINT,
    INPUT_CACHE_HIT_PRICE,
    INPUT_CACHE_MISS_PRICE,
    MODEL,
    OUTPUT_PRICE,
    PEAK_FACTOR,
    DeepSeekTransport,
    OutputGrader,
    PaidClient,
    from_env,
    reconcile_abandoned,
)
from writing_agent.suite import load_scenarios
from writing_agent.task_authoring import REVIEW_GATES, author_tasks, validate_task
from writing_agent.task_generation import prepare_task_requests


def fixture():
    source = {
        "id": "book",
        "work_id": "book",
        "role": "train",
        "provenance": "human",
        "text": "Mara waits at the river.",
        "parents": [],
        "sha256": fingerprint("Mara waits at the river."),
        "terms": {"license": "test fixture"},
    }
    variations = {
        k: ["one", "two"]
        for k in ("genres", "styles", "tropes", "situations", "continuity_challenges")
    }
    request = prepare_task_requests(
        [source], ["book"], excluded_source_groups=set(), variation_catalog=variations, count=1
    )["requests"][0]
    request["assignment"]["family"] = "F1"
    request["assignment"]["stage_families"] = ["F1"]
    body = {k: v for k, v in request.items() if k not in {"id", "status", "request_hash"}}
    request["request_hash"] = fingerprint(body)
    candidate = {
        "visible": {
            "brief": "Mara waits at the river. Continue with prose only.",
            "initial_files": {},
            "followups": [],
            "tools": [],
            "budgets": {
                "max_steps": 8,
                "max_tool_calls": 0,
                "max_read_tokens": 0,
                "max_total_bytes": 4096,
            },
            "prose": [{"id": "scene", "kind": "reply", "turn": 0}],
        },
        "labels": {
            "checks": [
                {
                    "id": "delivery",
                    "metric": "Q1",
                    "required": True,
                    "method": "deterministic",
                    "kind": "nonempty",
                    "artifact": "scene",
                }
            ],
            "rubrics": {
                key: {
                    "description": "Follow the request.",
                    "range": [1, 5],
                    "anchors": {str(i): "Fixture" for i in range(1, 6)},
                }
                for key in ("Q1", "Q2", "Q13")
            },
        },
        "branch_contract": {
            key: []
            for key in (
                "source_invariants",
                "accepted_departures",
                "open_questions",
                "allowed_alternatives",
            )
        },
        "evidence": [{"claim": "Mara waits.", "quote": "Mara waits", "source_id": "book"}],
        "stage_families": ["F1"],
        "realized_variation": {"genre": "source"},
        "review_notes": "A short continuation.",
    }
    review = {
        "verdict": "accept",
        "uncertainty": "Fixture",
        "summary": "Grounded",
        "gates": {
            k: {"passed": True, "evidence": ["Mara waits"], "rationale": "Fixture"}
            for k in REVIEW_GATES
        },
    }
    return [source], request, candidate, review


def response(value, *, model=MODEL, usage=None, content=None):
    return {
        "model": model,
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 20,
        }
        if usage is None
        else usage,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(value) if content is None else content,
                    "reasoning_content": "retained",
                },
            }
        ],
    }


class PaidCallTests(unittest.TestCase):
    def test_route_cache_and_shared_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = PaidClient(Path(tmp), api_key="secret", budget_usd=0.05)
            with patch.object(
                client.transport, "request", return_value=response({"ok": True})
            ) as call:
                first = client.json_call("System", {}, role="task_author")
                self.assertEqual(first, client.json_call("System", {}, role="task_author"))
                self.assertEqual(call.call_count, 1)
                self.assertEqual(first["request"]["model"], MODEL)
                self.assertNotIn("provider", first["request"])
                self.assertNotIn("reasoning", first["request"])
                self.assertEqual(first["request"]["thinking"], {"type": "enabled"})
                self.assertEqual(first["request"]["response_format"], {"type": "json_object"})
                client.limit = first["accounted_micro_usd"]
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    client.json_call("Review", {}, role="task_reviewer")
                self.assertEqual(call.call_count, 1)
            self.assertNotIn("secret", "".join(p.read_text() for p in Path(tmp).glob("*.json")))

    def test_failed_transport_blocks_paid_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = PaidClient(Path(tmp), api_key="secret", budget_usd=10)
            with patch.object(
                client.transport, "request", side_effect=RuntimeError("timeout")
            ) as call:
                with self.assertRaises(RuntimeError):
                    client.json_call("System", {}, role="task_author")
                with self.assertRaisesRegex(RuntimeError, "unresolved"):
                    client.json_call("System", {}, role="task_author")
                self.assertEqual(call.call_count, 1)

    def test_invalid_content_is_charged_and_cached(self):
        bad = response({}, content="unfinished JSON")
        with tempfile.TemporaryDirectory() as tmp:
            client = PaidClient(Path(tmp), api_key="secret", budget_usd=10)
            with patch.object(client.transport, "request", return_value=bad) as call:
                for _ in range(2):
                    with self.assertRaises(ValueError):
                        client.json_call("System", {}, role="task_author")
                self.assertEqual(call.call_count, 1)
            ledger = json.loads((Path(tmp) / "ledger.json").read_text())
            entry = next(iter(ledger.values()))
            self.assertEqual(entry["status"], "completed")
            self.assertGreater(entry["charged_or_reserved_micro_usd"], 0)

    def test_overrun_halts_subsequent_calls(self):
        huge = response(
            {"ok": True},
            usage={"prompt_tokens": 8, "completion_tokens": 10_000_000},
        )
        with tempfile.TemporaryDirectory() as tmp:
            client = PaidClient(Path(tmp), api_key="secret", budget_usd=10)
            with patch.object(client.transport, "request", return_value=huge) as call:
                with self.assertRaisesRegex(RuntimeError, "exceeded reservation"):
                    client.json_call("System", {}, role="task_author", max_tokens=8)
                with self.assertRaisesRegex(RuntimeError, r"(?i)unresolved"):
                    client.json_call("System", {}, role="task_author", max_tokens=8)
                with self.assertRaisesRegex(RuntimeError, r"(?i)unresolved"):
                    client.json_call("Review", {}, role="task_reviewer")
                self.assertEqual(call.call_count, 1)
            ledger = json.loads((Path(tmp) / "ledger.json").read_text())
            self.assertEqual(next(iter(ledger.values()))["status"], "overrun")

    def test_missing_usage_retains_reservation(self):
        cases = (
            {},
            {"prompt_tokens": 12},
            {"completion_tokens": 3},
            {"prompt_tokens": 12, "completion_tokens": 3, "prompt_cache_hit_tokens": 4},
        )
        for usage in cases:
            with self.subTest(usage=usage), tempfile.TemporaryDirectory() as tmp:
                client = PaidClient(Path(tmp), api_key="secret", budget_usd=10)
                bad = response({"ok": True}, usage=usage)
                with patch.object(client.transport, "request", return_value=bad) as call:
                    with self.assertRaisesRegex(RuntimeError, "Missing API usage"):
                        client.json_call("System", {}, role="task_author")
                    with self.assertRaisesRegex(RuntimeError, "unresolved"):
                        client.json_call("System", {}, role="task_author")
                    self.assertEqual(call.call_count, 1)
                ledger = json.loads((Path(tmp) / "ledger.json").read_text())
                self.assertEqual(next(iter(ledger.values()))["status"], "reserved")

    def test_from_env_fails_closed_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("OTHER=1\n")
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
                with self.assertRaises(ValueError):
                    from_env(Path(tmp) / "calls", env_file, budget_usd=1)

    def test_peak_reservation_blocks_small_budget_before_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            transport = DeepSeekTransport("secret")
            payload = transport.payload(
                "System", {}, role="task_author", max_tokens=64, thinking=True
            )
            reserve = transport.reserved_micro_usd(payload, 64)
            client = PaidClient(Path(tmp), api_key="secret", budget_usd=1)
            client.limit = reserve - 1
            ok = response({"ok": True})
            with patch.object(client.transport, "request", return_value=ok) as call:
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    client.json_call("System", {}, role="task_author", max_tokens=64)
                self.assertEqual(call.call_count, 0)
            client.limit = reserve
            with patch.object(client.transport, "request", return_value=ok) as call:
                result = client.json_call("System", {}, role="task_author", max_tokens=64)
                self.assertEqual(call.call_count, 1)
            self.assertLessEqual(result["accounted_micro_usd"], reserve)

    def test_transport_failure_writes_inspectable_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = PaidClient(Path(tmp), api_key="secret", budget_usd=10)
            error = urllib.error.HTTPError(
                ENDPOINT,
                502,
                "Bad Gateway",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"bad gateway"}'),
            )
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "HTTP 502"):
                    client.json_call("System", {}, role="task_author")
            ledger = json.loads((Path(tmp) / "ledger.json").read_text())
            identity = next(iter(ledger))
            self.assertEqual(ledger[identity]["status"], "reserved")
            saved = json.loads((Path(tmp) / (identity + ".error.json")).read_text())
            self.assertEqual(saved["identity"], identity)
            self.assertEqual(saved["http_status"], 502)
            self.assertIn("bad gateway", saved["body"])
            with patch("urllib.request.urlopen", side_effect=error) as call:
                with self.assertRaisesRegex(RuntimeError, "unresolved"):
                    client.json_call("System", {}, role="task_author")
                self.assertEqual(call.call_count, 0)

    def test_abandoned_does_not_block_new_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            (destination / "ledger.json").write_text(
                json.dumps(
                    {
                        "prior#abandoned": {
                            "status": "abandoned",
                            "role": "task_author",
                            "charged_or_reserved_micro_usd": 1000,
                            "reason": "process killed",
                        }
                    }
                )
            )
            client = PaidClient(destination, api_key="secret", budget_usd=10)
            with patch.object(
                client.transport, "request", return_value=response({"ok": True})
            ) as call:
                result = client.json_call("System", {}, role="task_author")
            self.assertEqual(call.call_count, 1)
            self.assertEqual(result["value"], {"ok": True})
            ledger = json.loads((destination / "ledger.json").read_text())
            self.assertEqual(ledger["prior#abandoned"]["status"], "abandoned")
            self.assertEqual(ledger["prior#abandoned"]["charged_or_reserved_micro_usd"], 1000)
            self.assertGreater(
                sum(item["charged_or_reserved_micro_usd"] for item in ledger.values()), 1000
            )

    def test_reconcile_abandoned_allows_retry_and_keeps_spend(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            client = PaidClient(destination, api_key="secret", budget_usd=10)
            with patch.object(
                client.transport, "request", side_effect=RuntimeError("killed")
            ) as call:
                with self.assertRaises(RuntimeError):
                    client.json_call("System", {}, role="task_author")
                with self.assertRaisesRegex(RuntimeError, "unresolved"):
                    client.json_call("System", {}, role="task_author")
                self.assertEqual(call.call_count, 1)
            ledger = json.loads((destination / "ledger.json").read_text())
            identity = next(iter(ledger))
            reserved = ledger[identity]["charged_or_reserved_micro_usd"]
            self.assertEqual(ledger[identity]["status"], "reserved")
            reconcile_abandoned(destination, identity, reason="process killed")
            reconcile_abandoned(destination, identity, reason="process killed again")
            ledger = json.loads((destination / "ledger.json").read_text())
            abandoned_key = identity + "#abandoned"
            self.assertNotIn(identity, ledger)
            self.assertEqual(ledger[abandoned_key]["status"], "abandoned")
            self.assertEqual(ledger[abandoned_key]["charged_or_reserved_micro_usd"], reserved)
            self.assertEqual(ledger[abandoned_key]["reason"], "process killed")
            self.assertEqual(
                sum(item["charged_or_reserved_micro_usd"] for item in ledger.values()), reserved
            )
            with patch.object(
                client.transport, "request", return_value=response({"ok": True})
            ) as call:
                retried = client.json_call("System", {}, role="task_author")
            self.assertEqual(call.call_count, 1)
            self.assertEqual(retried["value"], {"ok": True})
            ledger = json.loads((destination / "ledger.json").read_text())
            self.assertEqual(ledger[abandoned_key]["charged_or_reserved_micro_usd"], reserved)
            self.assertEqual(ledger[identity]["status"], "completed")
            self.assertGreater(
                sum(item["charged_or_reserved_micro_usd"] for item in ledger.values()), reserved
            )
            with self.assertRaisesRegex(RuntimeError, "reconcilable"):
                reconcile_abandoned(destination, identity, reason="must not reduce spend")


class TaskAdmissionTests(unittest.TestCase):
    def test_compile_private_separation_and_resume(self):
        catalog, request, candidate, review = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = PaidClient(root / "calls", api_key="secret", budget_usd=10)
            with patch.object(
                client.transport,
                "request",
                side_effect=[response(candidate), response(review)],
            ) as call:
                for _ in range(2):
                    result = author_tasks(
                        [request], catalog, "Author", [], root / "tasks", client=client
                    )
                    self.assertEqual(result["accepted"], 1)
                self.assertEqual(call.call_count, 2)
            payloads = [
                json.loads(path.read_text())["request"]
                for path in (root / "calls").glob("*.json")
                if path.name != "ledger.json"
            ]
            self.assertEqual({p["thinking"]["type"] for p in payloads}, {"disabled"})
            self.assertEqual({p["max_tokens"] for p in payloads}, {16384, 8192})
            loaded = load_scenarios(root / "tasks/compiled")
            self.assertEqual(loaded[0]["review_status"], "model_reviewed")
            self.assertNotIn("branch_contract", loaded[0]["visible"])
            outcome = root / "tasks/tasks" / (request["id"] + ".json")
            changed = json.loads(outcome.read_text())
            changed["status"] = "invalid"
            outcome.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "outcome hash"):
                author_tasks([request], catalog, "Author", [], root / "tasks", client=client)

    def test_interruption_exports_partial_collection_and_resumes_cached_author(self):
        catalog, request, candidate, review = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = PaidClient(root / "calls", api_key="secret", budget_usd=10)
            original = client.json_call

            def interrupt_review(instructions, data, **kwargs):
                if kwargs["role"] == "task_reviewer":
                    raise RuntimeError("budget exhausted")
                return original(instructions, data, **kwargs)

            with patch.object(
                client.transport, "request", return_value=response(candidate)
            ) as author:
                with patch.object(client, "json_call", side_effect=interrupt_review):
                    stopped = author_tasks(
                        [request], catalog, "Author", [], root / "tasks", client=client
                    )
                self.assertEqual(stopped["status"], "interrupted")
                self.assertEqual(stopped["compiled"], 0)
                self.assertEqual(author.call_count, 1)
            with patch.object(
                client.transport, "request", return_value=response(review)
            ) as reviewer:
                resumed = author_tasks(
                    [request], catalog, "Author", [], root / "tasks", client=client
                )
                self.assertEqual(resumed["accepted"], 1)
                self.assertEqual(reviewer.call_count, 1)

    def test_reject_bad_evidence_and_missing_stage_prose(self):
        _, request, candidate, _ = fixture()
        for invalid in ("evidence", "prose"):
            changed = copy.deepcopy(candidate)
            if invalid == "evidence":
                changed["evidence"][0]["quote"] = "invented quote"
            else:
                changed["visible"]["prose"] = []
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_task(request, changed)

    def test_extra_top_level_keys_are_ignored(self):
        _, request, candidate, _ = fixture()
        candidate = copy.deepcopy(candidate)
        candidate["type"] = "task"
        candidate["review_notes_extra"] = "ignored"
        validate_task(request, candidate)

    def test_evidence_quote_whitespace_normalization(self):
        _, request, candidate, _ = fixture()
        request = copy.deepcopy(request)
        source = request["packet"]["source"]
        source["text"] = "till he had\n    dust in his throat"
        source["sha256"] = fingerprint(source["text"])
        candidate = copy.deepcopy(candidate)
        candidate["evidence"][0]["quote"] = "till he had dust in his throat"
        validate_task(request, candidate)
        candidate["evidence"][0]["quote"] = "till he had dust in her throat"
        with self.assertRaisesRegex(ValueError, "Evidence must cite exact source quotes"):
            validate_task(request, candidate)

    def test_starting_kb_navigation(self):
        _, request, candidate, _ = fixture()
        candidate["visible"]["initial_files"] = {"kb/index.md": "[Missing](missing.md)"}
        with self.assertRaisesRegex(ValueError, "Starting KB"):
            validate_task(request, candidate)
        candidate["visible"]["initial_files"] = {"kb/index.md": "# Mara\nAt the river."}
        validate_task(request, candidate)


class OutputGraderTests(unittest.TestCase):
    def test_judgment_applies_and_failure_leaves_score_pending(self):
        packet = {"rubrics": {"Q1": {"range": [1, 5]}}, "checks": []}
        judgment = {
            "assessments": [
                {
                    "metric": "Q1",
                    "value": 4,
                    "evidence": ["selected_prose: Mara"],
                    "rationale": "Follows the request",
                    "uncertainty": "Low",
                    "dimensions": [],
                }
            ],
            "checks": [],
        }
        card = {
            "scores": {"Q1": {"status": "pending"}},
            "checks": [],
            "status": "completed",
            "artifacts": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            client = PaidClient(Path(tmp), api_key="secret", budget_usd=10)
            with patch.object(client.transport, "request", return_value=response(judgment)):
                result = OutputGrader(client).grade(packet)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(apply_judgment(card, packet, result)["scores"]["Q1"]["value"], 4)
            with patch.object(client, "json_call", side_effect=RuntimeError("budget")):
                failed = OutputGrader(client).grade(packet)
            self.assertEqual(
                apply_judgment(card, packet, failed)["scores"]["Q1"]["status"], "pending"
            )


class MultiStagePacketTests(unittest.TestCase):
    def test_earlier_planning_and_file_versions_reach_judge(self):
        _, request, candidate, _ = fixture()
        scenario = validate_task(request, candidate)
        result = {
            "status": "completed",
            "output": "Final prose",
            "trace": [],
            "turns": [
                {"output": "Earlier plan", "snapshot": {"plan.md": "Draft"}, "message_index": 1}
            ],
        }
        card = {
            "artifacts": [],
            "scores": {k: {"status": "pending"} for k in scenario["labels"]["rubrics"]},
        }
        packet = grading_packet(scenario, result, card)
        self.assertEqual(packet["turn_outputs"][0]["output"], "Earlier plan")
        self.assertEqual(packet["turn_outputs"][0]["snapshot"], {"plan.md": "Draft"})
        self.assertNotIn("message_index", packet["turn_outputs"][0])


class DeepSeekAdapterTests(unittest.TestCase):
    def test_direct_flash_json_request_shape(self):
        transport = DeepSeekTransport("secret")
        payload = transport.payload(
            "Reply with json.", {"n": 1}, role="output_judge", max_tokens=64, thinking=False
        )
        self.assertEqual(payload["model"], MODEL)
        self.assertNotIn("provider", payload)
        self.assertNotIn("reasoning", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("json", payload["messages"][0]["content"].lower())
        missing = transport.payload("System", {}, role="task_author", max_tokens=8, thinking=True)
        self.assertIn("json", missing["messages"][0]["content"].lower())
        self.assertEqual(missing["thinking"], {"type": "enabled"})

        class FakeHTTP:
            status = 200

            def __enter__(self):
                return self

            def read(self):
                return b'{"ok": true}'

            def __exit__(self, *args):
                return False

        with patch("urllib.request.urlopen", return_value=FakeHTTP()) as opener:
            transport.request(payload)
        request = opener.call_args[0][0]
        self.assertEqual(request.full_url, ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        sent = json.loads(request.data.decode())
        self.assertEqual(sent["model"], MODEL)
        self.assertNotIn("provider", sent)
        self.assertEqual(sent["thinking"], {"type": "disabled"})

    def test_non_flash_model_is_rejected_without_retry(self):
        bad = response({"ok": True}, model="deepseek-v4-pro")
        with tempfile.TemporaryDirectory() as tmp:
            client = PaidClient(Path(tmp), api_key="secret", budget_usd=10)
            with patch.object(client.transport, "request", return_value=bad) as call:
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, "Unexpected response model"):
                        client.json_call("System", {}, role="task_author")
                self.assertEqual(call.call_count, 1)
            saved = [p for p in Path(tmp).glob("*.json") if p.name != "ledger.json"]
            self.assertEqual(
                json.loads(saved[0].read_text())["response"]["model"], "deepseek-v4-pro"
            )

    def test_flash_alias_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = PaidClient(Path(tmp), api_key="secret", budget_usd=10)
            alias = response({"ok": True}, model=MODEL + "-20260315")
            with patch.object(client.transport, "request", return_value=alias):
                result = client.json_call("System", {}, role="task_reviewer")
            self.assertEqual(result["value"], {"ok": True})

    def test_empty_and_non_object_json_fail_closed(self):
        cases = (("", "Empty model content"), ("[]", "JSON object"))
        for content, message in cases:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                client = PaidClient(Path(tmp), api_key="secret", budget_usd=10)
                bad = response({}, content=content)
                with patch.object(client.transport, "request", return_value=bad) as call:
                    with self.assertRaisesRegex(ValueError, message):
                        client.json_call("System", {}, role="task_author")
                    with self.assertRaisesRegex(ValueError, message):
                        client.json_call("System", {}, role="task_author")
                    self.assertEqual(call.call_count, 1)

    def test_cache_hit_tokens_use_cache_price(self):
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
        }
        transport = DeepSeekTransport("secret")
        cost, accounted, basis = transport.charge(usage)
        miss_only, _, _ = transport.charge({"prompt_tokens": 1000, "completion_tokens": 10})
        expected = int(
            (
                (800 * INPUT_CACHE_HIT_PRICE + 200 * INPUT_CACHE_MISS_PRICE + 10 * OUTPUT_PRICE)
                * PEAK_FACTOR
            ).to_integral_value(rounding=ROUND_CEILING)
        )
        self.assertEqual(basis, "peak_ceiling_estimate")
        self.assertEqual(cost, expected)
        self.assertLess(cost, miss_only)
        with tempfile.TemporaryDirectory() as tmp:
            client = PaidClient(Path(tmp), api_key="secret", budget_usd=10)
            with patch.object(
                client.transport, "request", return_value=response({"ok": True}, usage=usage)
            ):
                result = client.json_call("System", {}, role="task_reviewer")
            self.assertEqual(result["accounted_micro_usd"], accounted)
            ledger = json.loads((Path(tmp) / "ledger.json").read_text())
            self.assertEqual(
                next(iter(ledger.values()))["charged_or_reserved_micro_usd"], accounted
            )
