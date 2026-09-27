"""Pure sequencing and guard evaluation for verified task-graph views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from writing_agent.task_graph_accounting import exhausted_stop_reason, tool_error
from writing_agent.task_graph_admission import AdmissionError, AdmittedNodeV1
from writing_agent.task_graph_contracts import EdgeContractV1, GuardContractV1

DirectiveKind = Literal[
    "sample_writer",
    "execute_tool",
    "request_author",
    "await_author_reply",
    "request_checks",
    "await_check_result",
    "commit_transition",
    "seal_outcome",
    "stop_exhausted",
    "publish_reward",
    "halt",
    "done",
]
CONTEXT_LIMITS = frozenset({"context_budget", "context_storage_budget"})
REQUIRED_CHECK_SCOPES = frozenset({"each_turn", "node_exit_candidate"})


@dataclass(frozen=True)
class Directive:
    """The next legal sequencer action, with any caller alternatives."""

    kind: DirectiveKind
    call_index: int | None = None
    source: Literal["writer_request", "mandatory_feedback"] | None = None
    edge_id: str | None = None
    task_status: str | None = None
    stop_reason: str | None = None
    alternatives: frozenset[str] = frozenset()


def evaluate_guard(
    guard: GuardContractV1,
    view: Any,
    *,
    task_status: str | None = None,
) -> bool:
    """Evaluate the complete admitted guard vocabulary against structured view fields."""
    status = view.outcome.task_status if task_status is None else task_status
    arguments = guard.arguments
    if guard.kind == "always":
        return True
    if guard.kind == "never":
        return False
    if guard.kind == "task_status":
        return status == arguments["status"]
    if guard.kind == "execution_status":
        return view.outcome.execution_status == arguments["status"]
    if guard.kind == "check_status":
        return view.check_statuses.get(arguments["check_id"]) == arguments["status"]
    if guard.kind == "interaction_complete":
        return _interaction_complete(view) is arguments["value"]
    if guard.kind == "budget_remaining":
        budget = arguments["budget"]
        limits = view.budget["limits"]
        consumed = view.budget["consumed"]
        return limits.get(budget, 0) - consumed.get(budget, 0) > 0
    raise AssertionError(f"admission allowed unknown guard {guard.kind}")


def select_edge(
    node: AdmittedNodeV1,
    view: Any,
    *,
    task_status: str | None = None,
) -> EdgeContractV1 | None:
    """Return the unique highest-priority matching edge, or None when none match."""
    matching = [
        edge
        for edge in node.edges
        if evaluate_guard(node.guards[edge.edge_id], view, task_status=task_status)
    ]
    if not matching:
        return None
    priorities = [edge.precedence if edge.precedence is not None else 0 for edge in matching]
    if len(set(priorities)) != len(priorities):
        raise AdmissionError("controller_conflict", "matching edges have equal precedence")
    return matching[priorities.index(min(priorities))]


def next_step(view: Any) -> Directive:
    """Derive one routing directive solely from the verified structured lineage view."""
    phase = view.state.position["phase"]
    if phase == "terminal":
        if (
            view.outcome.reward_status == "pending"
            and view.outcome.execution_status == "valid"
            and view.mode.reward is not None
        ):
            return Directive("publish_reward")
        return Directive("done")

    if phase == "ready_writer":
        continuation = view.state.continuation
        queue = continuation["tool_queue"]
        call_index = continuation["next_call"]
        if call_index < len(queue):
            call = queue[call_index]
            name = call["name"]
            pre_dispatch_error = tool_error(view.budget, call.get("rejection"), name)
            if name == "ask_author" and view.mode.ask_semantics and pre_dispatch_error is None:
                return Directive("request_author", source="writer_request")
            return Directive("execute_tool", call_index=call_index)

        reason = exhausted_stop_reason(view.budget)
        if reason is not None:
            alternatives = (
                frozenset({"context_operation"}) if reason in CONTEXT_LIMITS else frozenset()
            )
            return Directive("stop_exhausted", stop_reason=reason, alternatives=alternatives)
        return Directive("sample_writer", alternatives=frozenset({"context_operation"}))

    if phase == "awaiting_author":
        return Directive("await_author_reply")

    feedback_cursor = view.state.continuation["feedback_cursor"]
    feedback_remaining = feedback_cursor < len(view.mode.feedback_rules)
    if phase == "checking":
        checks = _applicable_checks(view, feedback_cursor)
        if checks and view.mode.evaluation:
            return Directive("request_checks")
        if not checks and feedback_remaining:
            if _feedback_budgets_allow(view):
                return Directive("request_author", source="mandatory_feedback")
            return Directive(
                "seal_outcome",
                task_status="incomplete",
                stop_reason=_feedback_budget_reason(view),
            )
        return Directive("halt", stop_reason="no_admitted_evaluation")

    if phase == "awaiting_checks":
        if view.state.continuation["check_requests"]:
            return Directive("await_check_result")
        if feedback_remaining:
            rule = view.mode.feedback_rules[feedback_cursor]
            prerequisites_pass = all(
                view.check_statuses.get(check_id) == "pass"
                for check_id in rule["prerequisite_check_ids"]
            )
            if not prerequisites_pass:
                return Directive(
                    "seal_outcome",
                    task_status="incomplete",
                    stop_reason="feedback_prerequisite_failed",
                )
            if not _feedback_budgets_allow(view):
                return Directive(
                    "seal_outcome",
                    task_status="incomplete",
                    stop_reason=_feedback_budget_reason(view),
                )
            return Directive("request_author", source="mandatory_feedback")

        task_status = (
            "accepted_partial"
            if view.node.contract.completion_contract.accepted_partial
            else "complete"
        )
        required_checks = _required_terminal_checks(view)
        if not required_checks:
            # §8.1's no-admitted-evaluation row governs an empty terminal set.
            return Directive("halt", stop_reason="no_admitted_evaluation")
        if not all(view.check_statuses.get(check.id) == "pass" for check in required_checks):
            return Directive(
                "seal_outcome", task_status="incomplete", stop_reason="required_check_failed"
            )
        edge = select_edge(view.node, view, task_status=task_status)
        if edge is None:
            return Directive(
                "seal_outcome", task_status="incomplete", stop_reason="no_applicable_edge"
            )
        return Directive("commit_transition", edge_id=edge.edge_id, task_status=task_status)

    if phase == "ready_transition":
        return Directive("seal_outcome", task_status=view.outcome.task_status)

    raise AssertionError(f"unhandled admitted phase {phase}")


def _interaction_complete(view: Any) -> bool:
    return view.state.continuation["feedback_cursor"] >= len(view.mode.feedback_rules)


def _applicable_checks(view: Any, feedback_cursor: int) -> tuple:
    feedback = view.mode.feedback_rules
    if feedback_cursor < len(feedback):
        scope = f"before_feedback:{feedback[feedback_cursor]['id']}"
        applicable = {"each_turn", scope}
    else:
        applicable = {"each_turn", "node_exit_candidate"}
    return tuple(check for check in view.node.checks.values() if check.applicability in applicable)


def _feedback_budgets_allow(view: Any) -> bool:
    consumed = view.budget["consumed"]
    limits = view.budget["limits"]
    return all(
        consumed.get(counter, 0) < limits.get(counter, 0)
        for counter in ("author_calls", "writer_turns")
    )


def _feedback_budget_reason(view: Any) -> str:
    consumed = view.budget["consumed"]
    limits = view.budget["limits"]
    if consumed.get("author_calls", 0) >= limits.get("author_calls", 0):
        return "author_budget"
    return "writer_budget"


def _required_terminal_checks(view: Any) -> tuple:
    return tuple(
        check
        for check in view.node.checks.values()
        if check.required and check.applicability in REQUIRED_CHECK_SCOPES
    )


__all__ = ["Directive", "evaluate_guard", "next_step", "select_edge"]
