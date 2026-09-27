"""Tests for immutable transition values and stable mismatch paths."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace

from tests.task_graph_fixtures import make_entry_fixture
from writing_agent.task_graph import MessageV1
from writing_agent.task_graph_records import OutcomeV1
from writing_agent.task_graph_transition import (
    DERIVE,
    CheckpointChain,
    ContextView,
    DerivedArtifact,
    LineageMode,
    LineageView,
    ToolSpec,
    first_difference,
)


class TransitionValueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = make_entry_fixture()
        self.context = ContextView(
            messages=(MessageV1(role="user", content=("brief",), origin="request:1"),),
            sources=(None,),
            tools=({"function": {"parameters": {"required": ["path"]}}},),
            rendering={"pins": {"template": "a" * 64}},
            content_ref="a" * 64,
            revision_ref="b" * 64,
        )
        self.mode = LineageMode(
            interaction="scripted_author",
            ask_semantics=True,
            feedback_rules=({"prerequisite_check_ids": ["check"]},),
            evaluation=False,
            reward=None,
        )
        self.outcome = replace(
            OutcomeV1.from_dict(self.fixture.reader.artifact(self.fixture.state.outcome_ref)),
            checks=({"request_ref": "e" * 64, "result_ref": None},),
        )
        self.view = LineageView(
            root_checkpoint_id="c" * 64,
            checkpoint_id="d" * 64,
            head_event_id=None,
            state=self.fixture.state,
            budget={"consumed": {"turns": 1}},
            outcome=self.outcome,
            check_statuses={"check": "pass"},
            context=self.context,
            raw_call_ids=frozenset({"raw-call"}),
            call_sources={},
            samples=(),
            ancestry=CheckpointChain("c" * 64, self.context),
            node=self.fixture.graph.node(self.fixture.node_id),
            mode=self.mode,
            tool_spec=ToolSpec(128_000, 4096),
            group={"policy": {"refs": ["e" * 64]}},
        )

    def test_views_freeze_nested_containers(self) -> None:
        mutations = (
            lambda: self.context.tools[0]["function"]["parameters"]["required"].append("x"),
            lambda: self.context.rendering["pins"].__setitem__("template", "x"),
            lambda: self.context.sources.append("a" * 64),
            lambda: self.mode.feedback_rules[0]["prerequisite_check_ids"].append("other"),
            lambda: self.view.budget["consumed"].__setitem__("turns", 2),
            lambda: self.view.outcome.checks[0].__setitem__("request_ref", "f" * 64),
            lambda: self.view.check_statuses.__setitem__("check", "fail"),
            lambda: self.view.state.position.__setitem__("phase", "terminal"),
            lambda: self.view.node.contract.entry.__setitem__("request_ref", "f" * 64),
            lambda: self.view.ancestry.context.rendering.__setitem__("prefix_id", "other"),
            lambda: self.view.raw_call_ids.add("new-call"),
            lambda: self.view.call_sources.__setitem__("new", object()),
            lambda: self.view.samples.append(object()),
            lambda: self.view.group["policy"]["refs"].append("f" * 64),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate), self.assertRaises((TypeError, AttributeError)):
                mutate()

        with self.assertRaises(FrozenInstanceError):
            self.context.content_ref = "f" * 64
        with self.assertRaises(FrozenInstanceError):
            self.view.checkpoint_id = "f" * 64
        with self.assertRaises(FrozenInstanceError):
            self.mode.ask_semantics = False

    def test_lineage_mode_and_unimplemented_derive_registry(self) -> None:
        mode = LineageMode.for_node(self.fixture.graph.node(self.fixture.node_id))
        self.assertEqual(mode.interaction, "none")
        self.assertFalse(mode.ask_semantics)
        self.assertFalse(mode.evaluation)
        self.assertIsNone(mode.reward)
        self.assertEqual(DERIVE, {})
        with self.assertRaises(TypeError):
            DERIVE["WriterTurnV1"] = object()

    def test_derived_artifact_freezes_nested_value(self) -> None:
        artifact = DerivedArtifact(
            ref="a" * 64,
            value={"nested": [{"value": "safe"}]},
            kind="private",
        )
        with self.assertRaises(TypeError):
            artifact.value["nested"][0]["value"] = "changed"

    def test_first_difference_handles_mappings_lists_records_and_ref_body(self) -> None:
        self.assertEqual(
            first_difference(
                {"budget": {"consumed": {"turns": 1}}},
                {"budget": {"consumed": {"turns": 2}}},
            ),
            "state.budget.consumed.turns",
        )
        self.assertEqual(first_difference({"items": [1, 2]}, {"items": [1, 3]}), "state.items[1]")
        self.assertIsNone(first_difference(self.fixture.state, self.fixture.state.to_dict()))
        candidate = dict(self.fixture.state.to_dict())
        candidate["position"] = {**candidate["position"], "phase": "checking"}
        self.assertEqual(
            first_difference(self.fixture.state, candidate),
            "state.position.phase",
        )
        self.assertEqual(
            first_difference(
                {"context_ref": "a" * 64},
                {"context_ref": {"messages": []}},
            ),
            "state.context_ref",
        )


if __name__ == "__main__":
    unittest.main()
