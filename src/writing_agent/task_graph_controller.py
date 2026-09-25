"""Deterministic environment-owned routing for admitted task graphs.

The controller consumes trusted state and check results only.  Author utterance text
is carried for auditability but is deliberately absent from every routing predicate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from writing_agent.task_graph_admission import AdmissionError, AdmittedGraphV1, AdmittedNodeV1
from writing_agent.task_graph_contracts import (
    EXECUTION_STATUSES,
    TASK_STATUSES,
    GuardContractV1,
)

REWARD_STATUSES = frozenset({"pending", "available", "unavailable"})
TRAINING_ELIGIBILITY = frozenset({"pending", "eligible", "ineligible"})
PHASES = frozenset(
    {
        "ready_writer",
        "checking",
        "awaiting_author",
        "awaiting_checks",
        "ready_transition",
        "terminal",
    }
)


@dataclass(frozen=True)
class OutcomeStatusV1:
    """Orthogonal outcome axes; no field is inferred from another."""

    task_status: str = "unknown"
    execution_status: str = "running"
    stop_reason: str | None = None
    reward_status: str = "pending"
    training_eligibility: str = "pending"
    schema: int = 1

    def __post_init__(self) -> None:
        if type(self.schema) is not int or self.schema != 1:
            raise ValueError("unsupported outcome status schema")
        if self.task_status not in TASK_STATUSES:
            raise ValueError("unsupported task status")
        if self.execution_status not in EXECUTION_STATUSES:
            raise ValueError("unsupported execution status")
        if self.stop_reason is not None and (
            not isinstance(self.stop_reason, str) or not self.stop_reason
        ):
            raise ValueError("stop_reason must be null or a nonempty string")
        if self.reward_status not in REWARD_STATUSES:
            raise ValueError("unsupported reward status")
        if self.training_eligibility not in TRAINING_ELIGIBILITY:
            raise ValueError("unsupported training eligibility")


@dataclass(frozen=True)
class ControllerViewV1:
    node_id: str
    phase: str
    outcome: OutcomeStatusV1
    writer_turn_complete: bool = False
    pending_author_request: str | None = None
    outstanding_checks: tuple[str, ...] = ()
    check_status: Mapping[str, str] = field(default_factory=dict)
    interaction_complete: bool = False
    continuation_allowed: bool = False
    budgets_remaining: Mapping[str, int] = field(default_factory=dict)
    author_utterance: str | None = None
    schema: int = 1

    def __post_init__(self) -> None:
        if type(self.schema) is not int or self.schema != 1:
            raise ValueError("unsupported controller view schema")
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id is required")
        if self.phase not in PHASES:
            raise ValueError("unsupported controller phase")
        if not isinstance(self.outcome, OutcomeStatusV1):
            raise TypeError("outcome must be OutcomeStatusV1")
        for value, label in (
            (self.writer_turn_complete, "writer_turn_complete"),
            (self.interaction_complete, "interaction_complete"),
            (self.continuation_allowed, "continuation_allowed"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{label} must be bool")
        if self.pending_author_request is not None and (
            not isinstance(self.pending_author_request, str) or not self.pending_author_request
        ):
            raise ValueError("pending_author_request must be null or a nonempty reference")
        if not isinstance(self.outstanding_checks, tuple) or any(
            not isinstance(value, str) or not value for value in self.outstanding_checks
        ):
            raise TypeError("outstanding_checks must be an array of ids")
        if len(self.outstanding_checks) != len(set(self.outstanding_checks)):
            raise ValueError("outstanding_checks must be unique")
        if not isinstance(self.check_status, Mapping) or any(
            not isinstance(key, str) or not key or value not in {"pass", "fail", "unavailable"}
            for key, value in self.check_status.items()
        ):
            raise ValueError("check_status must map ids to pass/fail/unavailable")
        if not isinstance(self.budgets_remaining, Mapping) or any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in self.budgets_remaining.items()
        ):
            raise ValueError("budgets_remaining must contain nonnegative integers")
        if self.author_utterance is not None and not isinstance(self.author_utterance, str):
            raise TypeError("author_utterance must be text or null")
        object.__setattr__(self, "check_status", MappingProxyType(dict(self.check_status)))
        object.__setattr__(
            self, "budgets_remaining", MappingProxyType(dict(self.budgets_remaining))
        )


@dataclass(frozen=True)
class ControllerDirectiveV1:
    kind: str
    edge_id: str | None = None
    request_ref: str | None = None
    reason: str | None = None
    schema: int = 1

    def __post_init__(self) -> None:
        if type(self.schema) is not int or self.schema != 1:
            raise ValueError("unsupported controller directive schema")
        if self.kind not in {
            "request_author",
            "continue_writer",
            "propose_edge",
            "stop_incomplete",
            "wait_checks",
        }:
            raise ValueError("unsupported controller directive")
        if self.kind == "propose_edge" and not self.edge_id:
            raise ValueError("propose_edge requires edge_id")
        if self.kind != "propose_edge" and self.edge_id is not None:
            raise ValueError("only propose_edge may name an edge")
        if self.kind == "request_author" and not self.request_ref:
            raise ValueError("request_author requires request_ref")
        if self.kind != "request_author" and self.request_ref is not None:
            raise ValueError("only request_author may name a request")


def evaluate_guard(
    guard: GuardContractV1,
    view: ControllerViewV1,
    *,
    task_status: str | None = None,
) -> bool:
    """Evaluate the complete v1 guard vocabulary without generated code or models."""
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
        return view.check_status.get(arguments["check_id"]) == arguments["status"]
    if guard.kind == "interaction_complete":
        return view.interaction_complete is arguments["value"]
    if guard.kind == "budget_remaining":
        return view.budgets_remaining.get(arguments["budget"], 0) > 0
    raise AssertionError(f"admission allowed unknown guard {guard.kind}")


class DeterministicControllerV1:
    VERSION = "deterministic-v1"

    def __init__(self, graph: AdmittedGraphV1):
        if not isinstance(graph, AdmittedGraphV1):
            raise TypeError("graph must be admitted before controller construction")
        self.graph = graph

    def next(self, view: ControllerViewV1) -> ControllerDirectiveV1:
        """Return one directive from trusted state; never parse author prose."""
        node = self.graph.node(view.node_id)
        if view.outcome.execution_status not in {"running", "valid"}:
            return ControllerDirectiveV1(
                "stop_incomplete",
                reason=view.outcome.stop_reason or view.outcome.execution_status,
            )
        if view.phase == "awaiting_checks" or view.outstanding_checks:
            return ControllerDirectiveV1("wait_checks")
        if view.phase == "awaiting_author" or view.pending_author_request is not None:
            if node.contract.interaction_contract.mode == "none":
                return ControllerDirectiveV1(
                    "stop_incomplete", reason="author_interaction_not_permitted"
                )
            if view.pending_author_request is None:
                return ControllerDirectiveV1("stop_incomplete", reason="missing_author_request")
            if view.budgets_remaining.get("author_calls", 0) < 1:
                return ControllerDirectiveV1("stop_incomplete", reason="author_budget")
            return ControllerDirectiveV1("request_author", request_ref=view.pending_author_request)
        if view.phase == "ready_writer":
            return self._writer_directive(view)
        if view.phase == "checking":
            if not view.writer_turn_complete:
                return self._writer_directive(view)
            required = node.contract.completion_contract.required_check_ids
            required_statuses = [view.check_status.get(check_id) for check_id in required]
            if any(status is None or status == "unavailable" for status in required_statuses):
                return ControllerDirectiveV1("wait_checks")
            passed = all(status == "pass" for status in required_statuses)
            if passed and view.interaction_complete:
                status = (
                    "accepted_partial"
                    if node.contract.completion_contract.accepted_partial
                    else "complete"
                )
                return self._transition(node, view, task_status=status)
            if self._can_continue(node, view):
                return ControllerDirectiveV1("continue_writer")
            return ControllerDirectiveV1(
                "stop_incomplete",
                reason="required_check_failed" if not passed else "interaction_incomplete",
            )
        if view.phase == "ready_transition":
            return self._transition(node, view)
        if view.phase == "terminal":
            return ControllerDirectiveV1(
                "stop_incomplete", reason=view.outcome.stop_reason or "already_terminal"
            )
        raise AssertionError(f"unhandled admitted phase {view.phase}")

    @staticmethod
    def _writer_directive(view: ControllerViewV1) -> ControllerDirectiveV1:
        if view.budgets_remaining.get("writer_turns", 0) < 1:
            return ControllerDirectiveV1("stop_incomplete", reason="writer_budget")
        return ControllerDirectiveV1("continue_writer")

    @staticmethod
    def _can_continue(node: AdmittedNodeV1, view: ControllerViewV1) -> bool:
        return (
            node.contract.completion_contract.repair_turns > 0
            and view.continuation_allowed
            and view.budgets_remaining.get("writer_turns", 0) > 0
        )

    @staticmethod
    def _transition(
        node: AdmittedNodeV1,
        view: ControllerViewV1,
        *,
        task_status: str | None = None,
    ) -> ControllerDirectiveV1:
        matching = [
            edge
            for edge in node.edges
            if evaluate_guard(node.guards[edge.edge_id], view, task_status=task_status)
        ]
        if not matching:
            if DeterministicControllerV1._can_continue(node, view):
                return ControllerDirectiveV1("continue_writer")
            return ControllerDirectiveV1("stop_incomplete", reason="no_applicable_edge")
        # Admission proved unique precedence whenever multiple guards can match.
        matching.sort(key=lambda edge: edge.precedence or 0)
        if len(matching) > 1 and matching[0].precedence == matching[1].precedence:
            raise AdmissionError(
                "controller_conflict", "admitted guards produced ambiguous precedence"
            )
        return ControllerDirectiveV1("propose_edge", edge_id=matching[0].edge_id)
