"""Protect the paid-call boundary without network requests or real credentials."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from writing_agent.anthropic_grading import AnthropicGrader


class BudgetTests(unittest.TestCase):
    def test_completed_call_is_cached_and_actual_cost_replaces_reservation(self):
        with tempfile.TemporaryDirectory() as temp:
            grader = AnthropicGrader(Path(temp), api_key="fixture-not-a-key")
            grader._request = Mock(
                side_effect=[
                    {"input_tokens": 10},
                    {
                        "usage": {"input_tokens": 10, "output_tokens": 20},
                        "content": [{"type": "text", "text": "A judgment"}],
                        "stop_reason": "end_turn",
                    },
                ]
            )
            first = grader.grade("A prompt")
            self.assertEqual(first, grader.grade("A prompt"))
            self.assertEqual(grader._request.call_count, 2)
            ledger = json.loads((Path(temp) / "ledger.json").read_text())
            self.assertEqual(sum(v["charged_or_reserved_micro_usd"] for v in ledger.values()), 330)
            self.assertNotIn("fixture-not-a-key", (Path(temp) / "ledger.json").read_text())

    def test_insufficient_budget_prevents_paid_request(self):
        with tempfile.TemporaryDirectory() as temp:
            grader = AnthropicGrader(Path(temp), api_key="fixture", budget_usd=0.001)
            grader._request = Mock(return_value={"input_tokens": 10})
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                grader.grade("A prompt")
            self.assertEqual(grader._request.call_count, 1)
            self.assertEqual(grader._request.call_args.args[0], "messages/count_tokens")

    def test_uncertain_request_retains_reservation_and_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            grader = AnthropicGrader(Path(temp), api_key="fixture")
            grader._request = Mock(side_effect=[{"input_tokens": 10}, RuntimeError("network")])
            with self.assertRaisesRegex(RuntimeError, "network"):
                grader.grade("A prompt")
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                grader.grade("A prompt")
            self.assertEqual(grader._request.call_count, 2)
            ledger = json.loads((Path(temp) / "ledger.json").read_text())
            self.assertGreater(sum(v["charged_or_reserved_micro_usd"] for v in ledger.values()), 0)
