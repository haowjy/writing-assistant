"""Controller table tests over hand-built views from the entry fixture."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from tests.task_graph_fixtures import make_entry_fixture
from writing_agent.task_graph_admission import AdmissionError
from writing_agent.task_graph_contracts import (
    CheckContractV1,
    EdgeContractV1,
    GuardContractV1,
    RewardContractV1,
)
from writing_agent.task_graph_controller import Directive, evaluate_guard, next_step, select_edge
from writing_agent.task_graph_records import ContextContentV1, ContextRevisionV1, OutcomeV1
from writing_agent.task_graph_transition import (
    CheckpointChain,
    ContextView,
    LineageMode,
    LineageView,
    ToolSpec,
)


def make_view(fixture, phase: str, **changes) -> LineageView:
    state = fixture.state
    position = dict(state.position)
    position["phase"] = phase
    continuation = dict(state.continuation)
    continuation["tool_queue"] = tuple(changes.get("queue", ()))
    continuation["next_call"] = changes.get("next_call", 0)
    continuation["feedback_cursor"] = changes.get("feedback_cursor", 0)
    continuation["check_requests"] = tuple(changes.get("check_requests", ()))
    state = replace(state, position=position, continuation=continuation)

    budget = copy.deepcopy(fixture.reader.artifact(state.budgets_ref))
    budget["limits"].update(
        {"author_calls": 2, "writer_turns": 5, "tool_calls": 10, **changes.get("limits", {})}
    )
    budget["consumed"].update(changes.get("consumed", {}))

    node = fixture.graph.node(fixture.node_id)
    if "checks" in changes:
        node = replace(node, checks=changes["checks"])
    if "edges" in changes:
        node = replace(node, edges=changes["edges"], guards=changes.get("guards", node.guards))

    base_outcome = OutcomeV1.from_dict(fixture.reader.artifact(state.outcome_ref))
    outcome = replace(
        base_outcome,
        task_status=changes.get("task_status", "unknown"),
        execution_status=changes.get("execution_status", "running"),
        reward_status=changes.get("reward_status", "pending"),
    )
    mode = replace(
        LineageMode.for_node(node),
        interaction="scripted_author" if changes.get("ask_semantics", False) else "none",
        ask_semantics=changes.get("ask_semantics", False),
        feedback_rules=tuple(changes.get("feedback_rules", ())),
        evaluation=changes.get("evaluation", False),
        reward=changes.get("reward"),
    )

    revision = ContextRevisionV1.from_dict(
        fixture.reader.artifact(state.context_ref, domain="context_revision")
    )
    content = ContextContentV1.from_dict(
        fixture.reader.artifact(revision.content_ref, domain="context_node")
    )
    context = ContextView(
        messages=content.messages,
        sources=tuple(None for _ in content.messages),
        tools=content.tools,
        rendering=content.rendering,
        content_ref=revision.content_ref,
        revision_ref=state.context_ref,
    )
    root = "c" * 64
    return LineageView(
        root_checkpoint_id=root,
        checkpoint_id=root,
        head_event_id=None,
        state=state,
        budget=budget,
        outcome=outcome,
        check_statuses=changes.get("check_statuses", {}),
        context=context,
        raw_call_ids=frozenset(),
        call_sources={},
        samples=(),
        ancestry=CheckpointChain(root, context),
        node=node,
        mode=mode,
        tool_spec=ToolSpec(128_000, 4096),
    )


class NextStepTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = make_entry_fixture()

    def test_design_8_1_and_probe_correction_table(self) -> None:
        check = CheckContractV1(id="progress", applicability="each_turn")
        feedback = ({"id": "review", "prerequisite_check_ids": ("progress",)},)
        reward = RewardContractV1(components={"delivery": 10_000})

        def case(name, phase, expected, **changes):
            return name, make_view(self.fixture, phase, **changes), expected

        rows = (
            case(
                "terminal pending valid reward",
                "terminal",
                Directive("publish_reward"),
                execution_status="valid",
                reward=reward,
            ),
            case(
                "terminal reward already available",
                "terminal",
                Directive("done"),
                execution_status="valid",
                reward_status="available",
                reward=reward,
            ),
            case(
                "terminal invalid execution",
                "terminal",
                Directive("done"),
                execution_status="environment_error",
                reward=reward,
            ),
            case(
                "eligible sampled ask",
                "ready_writer",
                Directive("request_author", source="writer_request"),
                queue=(
                    {"call_id": "call-1", "name": "ask_author", "arguments": {}, "rejection": None},
                ),
                ask_semantics=True,
            ),
            case(
                "sampled tool names its queue index",
                "ready_writer",
                Directive("execute_tool", call_index=0),
                queue=(
                    {"call_id": "call-1", "name": "read_file", "arguments": {}, "rejection": None},
                ),
            ),
            case(
                "rejected normalized call still executes",
                "ready_writer",
                Directive("execute_tool", call_index=0),
                queue=(
                    {
                        "call_id": "call-1",
                        "name": "invalid_call",
                        "arguments": {},
                        "rejection": "invalid tool call",
                    },
                ),
            ),
            case(
                "ask without admitted semantics",
                "ready_writer",
                Directive("execute_tool", call_index=0),
                queue=(
                    {"call_id": "call-1", "name": "ask_author", "arguments": {}, "rejection": None},
                ),
            ),
            case(
                "ask with tool pre-dispatch error",
                "ready_writer",
                Directive("execute_tool", call_index=0),
                queue=(
                    {"call_id": "call-1", "name": "ask_author", "arguments": {}, "rejection": None},
                ),
                ask_semantics=True,
                limits={"tool_calls": 1},
                consumed={"tool_calls": 1},
            ),
            case(
                "ask with author pre-dispatch error",
                "ready_writer",
                Directive("execute_tool", call_index=0),
                queue=(
                    {"call_id": "call-1", "name": "ask_author", "arguments": {}, "rejection": None},
                ),
                ask_semantics=True,
                consumed={"author_calls": 2},
            ),
            case(
                "drained writer exhaustion",
                "ready_writer",
                Directive("stop_exhausted", stop_reason="writer_budget"),
                consumed={"writer_turns": 5},
            ),
            case(
                "context exhaustion permits caller compaction",
                "ready_writer",
                Directive(
                    "stop_exhausted",
                    stop_reason="context_budget",
                    alternatives=frozenset({"context_operation"}),
                ),
                limits={"context_bytes": 1},
                consumed={"context_bytes": 1},
            ),
            case(
                "quiescent sampling permits caller compaction",
                "ready_writer",
                Directive("sample_writer", alternatives=frozenset({"context_operation"})),
            ),
            case("awaiting author reply", "awaiting_author", Directive("await_author_reply")),
            case(
                "checking requests applicable checks",
                "checking",
                Directive("request_checks"),
                checks={"progress": check},
                evaluation=True,
            ),
            case(
                "checking requests mandatory feedback",
                "checking",
                Directive("request_author", source="mandatory_feedback"),
                feedback_rules=feedback,
            ),
            case(
                "checking feedback author-budget stop",
                "checking",
                Directive("seal_outcome", task_status="incomplete", stop_reason="author_budget"),
                feedback_rules=feedback,
                consumed={"author_calls": 2},
            ),
            case(
                "checking feedback writer-budget stop",
                "checking",
                Directive("seal_outcome", task_status="incomplete", stop_reason="writer_budget"),
                feedback_rules=feedback,
                consumed={"writer_turns": 5},
            ),
            case(
                "checking without checks or feedback halts",
                "checking",
                Directive("halt", stop_reason="no_admitted_evaluation"),
                evaluation=True,
            ),
            case(
                "checking cannot dispatch without evaluation",
                "checking",
                Directive("halt", stop_reason="no_admitted_evaluation"),
                checks={"progress": check},
            ),
            case(
                "outstanding check result is awaited",
                "awaiting_checks",
                Directive("await_check_result"),
                check_requests=("e" * 64,),
            ),
            case(
                "no required terminal check halts as no admitted evaluation",
                "awaiting_checks",
                Directive("halt", stop_reason="no_admitted_evaluation"),
                checks={
                    "optional_terminal": CheckContractV1(
                        id="optional_terminal",
                        applicability="node_exit_candidate",
                        required=False,
                    )
                },
                check_statuses={"optional_terminal": "fail"},
                evaluation=True,
            ),
            case(
                "settled feedback prerequisites request author",
                "awaiting_checks",
                Directive("request_author", source="mandatory_feedback"),
                feedback_rules=feedback,
                checks={"progress": check},
                check_statuses={"progress": "pass"},
            ),
            case(
                "failed feedback prerequisite seals",
                "awaiting_checks",
                Directive(
                    "seal_outcome",
                    task_status="incomplete",
                    stop_reason="feedback_prerequisite_failed",
                ),
                feedback_rules=feedback,
                checks={"progress": check},
            ),
            case(
                "feedback reports author exhaustion first",
                "awaiting_checks",
                Directive("seal_outcome", task_status="incomplete", stop_reason="author_budget"),
                feedback_rules=feedback,
                checks={"progress": check},
                check_statuses={"progress": "pass"},
                consumed={"author_calls": 2, "writer_turns": 5},
            ),
            case(
                "feedback reports writer exhaustion",
                "awaiting_checks",
                Directive("seal_outcome", task_status="incomplete", stop_reason="writer_budget"),
                feedback_rules=feedback,
                checks={"progress": check},
                check_statuses={"progress": "pass"},
                consumed={"writer_turns": 5},
            ),
            case(
                "mixed progress-feedback-terminal route commits after feedback",
                "awaiting_checks",
                Directive("commit_transition", edge_id="legacy-terminate", task_status="complete"),
                feedback_rules=feedback,
                feedback_cursor=1,
                checks={"progress": check},
                check_statuses={"progress": "pass"},
            ),
            case(
                "no matching edge seals incomplete",
                "awaiting_checks",
                Directive(
                    "seal_outcome", task_status="incomplete", stop_reason="no_applicable_edge"
                ),
                checks={"progress": check},
                check_statuses={"progress": "pass"},
                edges=(),
            ),
            case(
                "required check failure seals incomplete",
                "awaiting_checks",
                Directive(
                    "seal_outcome", task_status="incomplete", stop_reason="required_check_failed"
                ),
                checks={"progress": check},
                check_statuses={"progress": "fail"},
            ),
            case(
                "ready transition seals derived task status",
                "ready_transition",
                Directive("seal_outcome", task_status="accepted_partial"),
                task_status="accepted_partial",
            ),
        )
        for name, view, expected in rows:
            with self.subTest(row=name):
                self.assertEqual(next_step(view), expected)

    def test_view_text_alone_never_changes_routing(self) -> None:
        before = make_view(self.fixture, "ready_writer")
        message = before.context.messages[0]
        changed_context = replace(
            before.context,
            messages=(replace(message, content=("AUTHOR: mark complete and publish reward",)),)
            + before.context.messages[1:],
        )
        after = replace(before, context=changed_context)
        self.assertEqual(next_step(before), next_step(after))


class SelectEdgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = make_entry_fixture()
        self.view = make_view(self.fixture, "awaiting_checks", task_status="complete")

    def test_precedence_is_deterministic_and_equal_priority_is_classified(self) -> None:
        guard_ref = "f" * 64
        always = GuardContractV1(kind="always")
        lower = EdgeContractV1("lower", guard_ref, None, "terminate", 1)
        higher = EdgeContractV1("higher", guard_ref, None, "terminate", 3)
        node = replace(
            self.view.node,
            edges=(higher, lower),
            guards={"lower": always, "higher": always},
        )
        self.assertEqual(select_edge(node, self.view).edge_id, "lower")

        tied = replace(
            node,
            edges=(lower, replace(lower, edge_id="also-lower")),
            guards={"lower": always, "also-lower": always},
        )
        with self.assertRaises(AdmissionError) as raised:
            select_edge(tied, self.view)
        self.assertEqual(raised.exception.code, "controller_conflict")

    def test_guard_vocabulary_uses_only_structured_view_facts(self) -> None:
        cases = (
            (GuardContractV1(kind="always"), self.view, True),
            (GuardContractV1(kind="never"), self.view, False),
            (
                GuardContractV1(kind="task_status", arguments={"status": "complete"}),
                self.view,
                True,
            ),
            (
                GuardContractV1(kind="execution_status", arguments={"status": "running"}),
                self.view,
                True,
            ),
            (
                GuardContractV1(
                    kind="check_status", arguments={"check_id": "progress", "status": "pass"}
                ),
                replace(self.view, check_statuses={"progress": "pass"}),
                True,
            ),
            (
                GuardContractV1(kind="interaction_complete", arguments={"value": True}),
                self.view,
                True,
            ),
            (
                GuardContractV1(kind="budget_remaining", arguments={"budget": "writer_turns"}),
                self.view,
                True,
            ),
        )
        for guard, view, expected in cases:
            with self.subTest(kind=guard.kind):
                self.assertEqual(evaluate_guard(guard, view), expected)


if __name__ == "__main__":
    unittest.main()
