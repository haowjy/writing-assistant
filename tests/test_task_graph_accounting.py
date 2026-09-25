import unittest

from writing_agent.task_graph_accounting import (
    exhausted_stop_reason,
    sampled_usage_charge,
    tool_error,
    tool_result_charge,
)


class AccountingTests(unittest.TestCase):
    def test_sampled_usage_counts_parent_total_once_and_orders_overrun(self):
        original = {
            "schema": 1,
            "limits": {"generated_tokens": 3, "total_tokens": 10},
            "consumed": {"generated_tokens": 2, "total_tokens": 8},
            "read_tokenizer": "whitespace-v1",
        }
        charged, exceeded = sampled_usage_charge(
            original,
            {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 3,
                "details": {"cache_read_tokens": 2},
            },
        )
        self.assertEqual(exceeded, "generated_tokens")
        self.assertEqual(charged["consumed"]["generated_tokens"], 4)
        self.assertEqual(charged["consumed"]["total_tokens"], 11)
        self.assertNotIn("model_calls", original["consumed"])

    def test_tool_error_precedence_and_charge(self):
        budget = {
            "schema": 1,
            "limits": {"tool_calls": 0, "author_calls": 0},
            "consumed": {"tool_calls": 0},
            "read_tokenizer": "whitespace-v1",
        }
        self.assertEqual(
            tool_error(budget, "bad syntax", "ask_author"),
            {"ok": False, "valid": True, "error": "Tool-call budget exceeded"},
        )
        charged, evidence = tool_result_charge(budget, {"a": "old"}, {"a": "new"}, 0)
        self.assertEqual(evidence["attempted_tool_calls"], 1)
        self.assertEqual(evidence["tool_calls"], 0)
        self.assertEqual(charged["consumed"]["storage_bytes"], 3)
        self.assertNotIn("attempted_tool_calls", budget["consumed"])

    def test_exhausted_reason_uses_protocol_order(self):
        budget = {
            "limits": {"writer_turns": 1, "generated_tokens": 2, "context_bytes": 3},
            "consumed": {"writer_turns": 1, "generated_tokens": 2, "context_bytes": 3},
        }
        self.assertEqual(exhausted_stop_reason(budget), "writer_budget")
        budget["consumed"]["writer_turns"] = 0
        self.assertEqual(exhausted_stop_reason(budget), "generated_tokens_budget")
        budget["consumed"]["generated_tokens"] = 0
        self.assertEqual(exhausted_stop_reason(budget), "context_budget")


if __name__ == "__main__":
    unittest.main()
