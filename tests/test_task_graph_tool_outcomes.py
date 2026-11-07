"""Committed tool-outcome derivation stays code-based and independent of model stacks."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.task_graph_rollout_fixtures import build_rollout_fixture, run_slice
from writing_agent.task_graph_calls import PROTOCOL_SHAPED_REJECTION_CODES
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_tool_outcomes import ToolOutcomeError, read_member_tool_outcomes


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

    def _incomplete_call_lineage(self, root, *, token_limit=False, calls=()):
        from tests.test_task_graph_v2_writer import _bound_native_lineage, _turn
        from writing_agent.task_graph_gate import StoreArtifactReader
        from writing_agent.task_graph_records import EnvironmentStepV1

        fixture, _session, _spec, runtime = _bound_native_lineage(
            root, mode="context_token_limited", max_tokens=4
        )
        start = fixture.env.verify(runtime)
        if token_limit:
            turn = _turn(
                start,
                StoreArtifactReader(fixture.store),
                generated_ids=(10, 11, 12, 13),
                termination_kind="token_limit",
                stop_token_id=None,
                limit="decision",
                calls=calls,
                content="",
            )
        else:
            turn = _turn(
                start,
                StoreArtifactReader(fixture.store),
                generated_ids=(1,),
                stop_token_id=1,
                calls=calls,
                content="",
            )
        committed = fixture.env.commit(runtime, turn)
        published = fixture.env.commit(
            committed.runtime,
            EnvironmentStepV1(directive={"kind": "publish_reward"}),
        )
        return fixture, start, fixture.env.verify(published.runtime)

    def test_final_incomplete_action_reports_unexecuted_calls_as_model_behavior(self):
        tool_call = _call("write_file", {"path": "draft.txt", "content": "x"}, "write-1")
        with tempfile.TemporaryDirectory() as temporary:
            _fixture, start, final = self._incomplete_call_lineage(
                Path(temporary) / "unterminated",
                calls=(tool_call,),
            )
            outcomes = read_member_tool_outcomes(start, final)
            self.assertEqual(final.outcome.stop_reason, "unterminated_tool_call")
            self.assertEqual(
                outcomes["calls"],
                [
                    {
                        "call_id": next(iter(final.call_sources)),
                        "name": "write_file",
                        "result": {"code": "not_executed_incomplete"},
                    }
                ],
            )
            self.assertEqual(outcomes["protocol_shaped_rejection_count"], 0)

        with tempfile.TemporaryDirectory() as temporary:
            _fixture, start, final = self._incomplete_call_lineage(
                Path(temporary) / "token-limit",
                token_limit=True,
                calls=(tool_call,),
            )
            outcomes = read_member_tool_outcomes(start, final)
            self.assertEqual(final.outcome.stop_reason, "decision_token_limit")
            self.assertEqual(outcomes["calls"][0]["result"], {"code": "not_executed_incomplete"})

    def test_unpaired_call_on_nonfinal_action_stays_an_error_and_results_are_sorted(self):
        calls = tuple(
            _call("write_file", {"path": f"{index}.txt", "content": "x"}, f"write-{index}")
            for index in range(12)
        )
        with tempfile.TemporaryDirectory() as temporary:
            _fixture, start, final = self._incomplete_call_lineage(
                Path(temporary) / "multiple-calls",
                calls=calls,
            )
            outcomes = read_member_tool_outcomes(start, final)
            call_ids = [item["call_id"] for item in outcomes["calls"]]
            self.assertEqual(
                call_ids,
                [
                    f"{start.state.position['lineage_id']}:call:0:{index}"
                    for index in range(len(calls))
                ],
            )

            later_sample = replace(final.samples[-1], action_id="later:action:0")
            nonfinal_action = replace(final, samples=(*final.samples, later_sample))
            with self.assertRaises(ToolOutcomeError):
                read_member_tool_outcomes(start, nonfinal_action)


if __name__ == "__main__":
    unittest.main()
