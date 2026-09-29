"""Phase-3 artifact contract compatibility tests."""

from __future__ import annotations

import unittest

from writing_agent.task_graph_contracts import BudgetContractV1


class BudgetContractTests(unittest.TestCase):
    def test_budget_contract_owns_usage_charged_limits(self) -> None:
        required = {
            "max_steps": 1,
            "max_total_bytes": 1,
            "max_graph_hops": 1,
            "max_visits": 1,
        }
        self.assertEqual(BudgetContractV1(**required).usage_charged_limits(), {})
        self.assertEqual(
            BudgetContractV1(
                **required,
                max_generated_tokens=32,
                max_total_tokens=64,
            ).usage_charged_limits(),
            {"generated_tokens": 32, "total_tokens": 64},
        )

    def test_absent_token_limit_keeps_the_legacy_wire_form(self) -> None:
        budget = BudgetContractV1(
            max_steps=1,
            max_total_bytes=1,
            max_graph_hops=1,
            max_visits=1,
        )

        wire = budget.to_dict()

        self.assertNotIn("max_generated_tokens", wire)
        self.assertNotIn("max_total_tokens", wire)
        self.assertEqual(BudgetContractV1.from_dict(wire), budget)
        with self.assertRaises(ValueError):
            BudgetContractV1.from_dict({**wire, "max_generated_tokens": None})
        with self.assertRaises(ValueError):
            BudgetContractV1.from_dict({**wire, "max_total_tokens": None})

    def test_total_token_limit_is_optional_and_omitted_when_absent(self) -> None:
        budget = BudgetContractV1(
            max_steps=1,
            max_total_bytes=1,
            max_graph_hops=1,
            max_visits=1,
            max_total_tokens=1536,
        )

        wire = budget.to_dict()

        self.assertEqual(wire["max_total_tokens"], 1536)
        self.assertEqual(BudgetContractV1.from_dict(wire), budget)


if __name__ == "__main__":
    unittest.main()
