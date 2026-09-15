"""Paid-call accounting and admission boundaries, without network or models."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from writing_agent.catalog import fingerprint
from writing_agent.grading import apply_judgment, grading_packet
from writing_agent.openrouter import MODEL, GLMGrader, OpenRouterClient
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


def response(value):
    return {
        "provider": "Reka",
        "model": MODEL,
        "usage": {"cost": 0.001},
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(value), "reasoning": "retained"},
            }
        ],
    }


class PaidCallTests(unittest.TestCase):
    def test_route_cache_and_shared_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = OpenRouterClient(Path(tmp), api_key="secret", budget_usd=0.05)
            with patch.object(client, "_request", return_value=response({"ok": True})) as call:
                first = client.json_call("System", {}, role="task_author")
                self.assertEqual(first, client.json_call("System", {}, role="task_author"))
                self.assertEqual(call.call_count, 1)
                self.assertEqual(first["request"]["provider"]["only"], ["reka/fp8"])
                self.assertFalse(first["request"]["provider"]["allow_fallbacks"])
                client.limit = 1055
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    client.json_call("Review", {}, role="task_reviewer")
                self.assertEqual(call.call_count, 1)
            self.assertNotIn("secret", "".join(p.read_text() for p in Path(tmp).glob("*.json")))

    def test_failed_transport_blocks_paid_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = OpenRouterClient(Path(tmp), api_key="secret", budget_usd=10)
            with patch.object(client, "_request", side_effect=RuntimeError("timeout")) as call:
                with self.assertRaises(RuntimeError):
                    client.json_call("System", {}, role="task_author")
                with self.assertRaisesRegex(RuntimeError, "unresolved"):
                    client.json_call("System", {}, role="task_author")
                self.assertEqual(call.call_count, 1)

    def test_invalid_content_is_charged_and_cached(self):
        bad = response({})
        bad["choices"][0]["message"]["content"] = "unfinished JSON"
        with tempfile.TemporaryDirectory() as tmp:
            client = OpenRouterClient(Path(tmp), api_key="secret", budget_usd=10)
            with patch.object(client, "_request", return_value=bad) as call:
                for _ in range(2):
                    with self.assertRaises(ValueError):
                        client.json_call("System", {}, role="task_author")
                self.assertEqual(call.call_count, 1)
            ledger = json.loads((Path(tmp) / "ledger.json").read_text())
            self.assertEqual(next(iter(ledger.values()))["charged_or_reserved_micro_usd"], 1055)


class TaskAdmissionTests(unittest.TestCase):
    def test_compile_private_separation_and_resume(self):
        catalog, request, candidate, review = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = OpenRouterClient(root / "calls", api_key="secret", budget_usd=10)
            with patch.object(
                client, "_request", side_effect=[response(candidate), response(review)]
            ) as call:
                for _ in range(2):
                    result = author_tasks(
                        [request], catalog, "Author", [], root / "tasks", client=client
                    )
                    self.assertEqual(result["accepted"], 1)
                self.assertEqual(call.call_count, 2)
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
            client = OpenRouterClient(root / "calls", api_key="secret", budget_usd=10)
            original = client.json_call

            def interrupt_review(instructions, data, **kwargs):
                if kwargs["role"] == "task_reviewer":
                    raise RuntimeError("budget exhausted")
                return original(instructions, data, **kwargs)

            with patch.object(client, "_request", return_value=response(candidate)) as author:
                with patch.object(client, "json_call", side_effect=interrupt_review):
                    stopped = author_tasks(
                        [request], catalog, "Author", [], root / "tasks", client=client
                    )
                self.assertEqual(stopped["status"], "interrupted")
                self.assertEqual(stopped["compiled"], 0)
                self.assertEqual(author.call_count, 1)
            with patch.object(client, "_request", return_value=response(review)) as reviewer:
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
            client = OpenRouterClient(Path(tmp), api_key="secret", budget_usd=10)
            with patch.object(client, "_request", return_value=response(judgment)):
                result = GLMGrader(client).grade(packet)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(apply_judgment(card, packet, result)["scores"]["Q1"]["value"], 4)
            with patch.object(client, "json_call", side_effect=RuntimeError("budget")):
                failed = GLMGrader(client).grade(packet)
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
