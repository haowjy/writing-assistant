"""Committed tool-outcome derivation stays code-based and independent of model stacks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.task_graph_rollout_fixtures import build_rollout_fixture, run_slice
from writing_agent.task_graph_calls import PROTOCOL_SHAPED_REJECTION_CODES
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_tool_outcomes import read_member_tool_outcomes


def _call(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _sample(*, content: str = "", calls: tuple[dict, ...] = ()) -> SampleResult:
    return SampleResult({"role": "assistant", "content": content, "tool_calls": list(calls)})


class TaskGraphToolOutcomeTests(unittest.TestCase):
    def test_reader_distinguishes_model_argument_error_from_duplicate_id_protocol_error(self):
        samples = (
            _sample(calls=(_call("ask_author", {"question": "missing fields"}, "ask-1"),)),
            _sample(
                calls=(
                    _call("read_file", {"path": "draft.txt"}, "same-id"),
                    _call("read_file", {"path": "draft.txt"}, "same-id"),
                )
            ),
            _sample(
                calls=(
                    _call(
                        "write_file",
                        {"path": "draft.txt", "content": "The revised draft."},
                        "write-1",
                    ),
                )
            ),
            _sample(content="The revision is ready for review."),
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_rollout_fixture(
                Path(temporary) / "fixture", mode="slice", sample_results=samples
            )
            final = run_slice(fixture)
            start_view = fixture.gate.view(fixture.store, fixture.checkpoint_ids[0])
            final_view = fixture.gate.view(fixture.store, final.checkpoint_id)
            outcomes = read_member_tool_outcomes(start_view, final_view)

        by_text = {
            result["text"]: result["code"]
            for call in outcomes["calls"]
            if isinstance((result := call["result"]), dict)
        }
        self.assertEqual(
            by_text["ask_author needs exact structured arguments"],
            "ask_author_arguments_shape",
        )
        self.assertEqual(by_text["Duplicate tool call id"], "duplicate_id")
        self.assertEqual(outcomes["protocol_shaped_rejection_count"], 1)
        self.assertIn("duplicate_id", PROTOCOL_SHAPED_REJECTION_CODES)
        self.assertEqual(outcomes["counts_by_code"]["ok"], 2)
        self.assertEqual(outcomes["counts_by_code"]["ask_author_arguments_shape"], 1)
        self.assertTrue(outcomes["files_changed"])
        self.assertEqual(outcomes["changed_paths"], ["draft.txt"])


if __name__ == "__main__":
    unittest.main()
