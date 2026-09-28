"""Phase-3 artifact contract compatibility tests."""

from __future__ import annotations

import unittest

from writing_agent.task_graph_contracts import BudgetContractV1


class BudgetContractTests(unittest.TestCase):
    def test_absent_token_limit_keeps_the_legacy_wire_form(self) -> None:
        budget = BudgetContractV1(
            max_steps=1,
            max_total_bytes=1,
            max_graph_hops=1,
            max_visits=1,
        )

        wire = budget.to_dict()

        self.assertNotIn("max_generated_tokens", wire)
        self.assertEqual(BudgetContractV1.from_dict(wire), budget)
        with self.assertRaises(ValueError):
            BudgetContractV1.from_dict({**wire, "max_generated_tokens": None})


if __name__ == "__main__":
    unittest.main()
