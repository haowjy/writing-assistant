"""Boundary checks for native V2 termination and terminal outcomes."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.test_task_graph_v2_writer import (
    _assert_path,
    _bound_native_lineage,
    _native_view,
    _turn,
    with_budget,
)
from writing_agent.task_graph_derive_writer import derive_writer_turn_v2
from writing_agent.task_graph_gate import StoreArtifactReader
from writing_agent.task_graph_records import OutcomeV1


class V2TerminationTests(unittest.TestCase):
    def setUp(self):
        self.fixture, self.view, _manifest = _native_view()
        self.reader = self.fixture.reader

    def test_zero_generation_context_limit_rejects_nonempty_message(self):
        context_view = with_budget(
            self.view,
            self.reader,
            limits={"context_tokens": 2},
        )
        bad_content = _turn(
            context_view,
            self.reader,
            input_ids=(10, 11, 12),
            generated_ids=(),
            termination_kind="context_limit",
            stop_token_id=None,
            limit="context",
            content="never generated",
        )
        _assert_path(self, context_view, bad_content, self.reader, "input.message:")

        bad_call = _turn(
            context_view,
            self.reader,
            input_ids=(10, 11, 12),
            generated_ids=(),
            termination_kind="context_limit",
            stop_token_id=None,
            limit="context",
            content="",
            calls=(
                {
                    "id": "never-generated",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": {"path": "x", "content": "y"},
                    },
                },
            ),
        )
        _assert_path(self, context_view, bad_call, self.reader, "input.message:")

        bad_raw_output = replace(
            _turn(
                context_view,
                self.reader,
                input_ids=(10, 11, 12),
                generated_ids=(),
                termination_kind="context_limit",
                stop_token_id=None,
                limit="context",
                content="",
            ),
            raw_output_ref="f" * 64,
        )
        _assert_path(self, context_view, bad_raw_output, self.reader, "input.raw_output_ref:")

        # Equal limits choose the first term: decision, then generated budget, then context.
        tied_view = with_budget(
            self.view,
            self.reader,
            limits={"generated_tokens": 4, "context_tokens": 6},
        )
        tied = _turn(
            tied_view,
            self.reader,
            input_ids=(10, 11),
            generated_ids=(12, 13, 14, 15),
            termination_kind="token_limit",
            stop_token_id=None,
            limit="decision",
        )
        self.assertEqual(
            derive_writer_turn_v2(tied_view, tied, self.reader).view.outcome.stop_reason,
            "decision_token_limit",
        )

    def test_each_termination_outcome_is_published_in_terminal_outcome_v1(self):
        tool_call = {
            "id": "write-1",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": {"path": "draft.txt", "content": "x"},
            },
        }
        cases = (
            (
                "unterminated_tool_call",
                "context_token_limited",
                4,
                {"generated_ids": (1,), "stop_token_id": 1, "calls": (tool_call,), "content": ""},
            ),
            (
                "unterminated_final_answer",
                "context_token_limited",
                4,
                {"generated_ids": (50,), "stop_token_id": 50, "content": "answer"},
            ),
            (
                "decision_token_limit",
                "context_token_limited",
                4,
                {
                    "generated_ids": (10, 11, 12, 13),
                    "termination_kind": "token_limit",
                    "stop_token_id": None,
                    "limit": "decision",
                },
            ),
            (
                "generated_tokens_budget",
                "token_limited",
                200,
                {
                    "generated_ids": tuple(range(200, 300)),
                    "termination_kind": "token_limit",
                    "stop_token_id": None,
                    "limit": "generated_budget",
                },
            ),
            (
                "context_tokens_budget",
                "context_token_limited",
                4,
                {
                    "input_ids": tuple(range(10, 108)),
                    "generated_ids": (200, 201),
                    "termination_kind": "token_limit",
                    "stop_token_id": None,
                    "limit": "context",
                },
            ),
            (
                "context_tokens_budget",
                "context_token_limited",
                4,
                {
                    "input_ids": tuple(range(200, 301)),
                    "generated_ids": (),
                    "termination_kind": "context_limit",
                    "stop_token_id": None,
                    "limit": "context",
                    "content": "",
                },
            ),
        )

        with tempfile.TemporaryDirectory() as root:
            for ordinal, (reason, mode, max_tokens, turn_args) in enumerate(cases):
                with self.subTest(reason=reason, ordinal=ordinal):
                    fixture, _session, _spec, runtime = _bound_native_lineage(
                        Path(root) / f"lineage-{ordinal}",
                        mode=mode,
                        max_tokens=max_tokens,
                    )
                    view = fixture.env.verify(runtime)
                    turn = _turn(view, StoreArtifactReader(fixture.store), **turn_args)
                    prior_checkpoint = view.checkpoint_id

                    committed = fixture.env.commit(runtime, turn)
                    committed_view = fixture.env.verify(committed.runtime)
                    outcome = committed_view.outcome
                    self.assertIsInstance(outcome, OutcomeV1)
                    self.assertEqual(outcome.task_status, "incomplete")
                    self.assertEqual(outcome.execution_status, "valid")
                    self.assertEqual(outcome.stop_reason, reason)
                    self.assertEqual(outcome.candidate_checkpoint, prior_checkpoint)
                    self.assertEqual(committed_view.state.position["phase"], "terminal")
                    self.assertEqual(
                        fixture.store.get_artifact(committed_view.state.outcome_ref),
                        outcome.to_wire(),
                    )


if __name__ == "__main__":
    unittest.main()
