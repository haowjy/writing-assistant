"""Pure checks, transition, terminal, and reward derives for wire-v1 outcomes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from writing_agent.task_graph_accounting import exhausted_stop_reason
from writing_agent.task_graph_contracts import CheckContractV1
from writing_agent.task_graph_controller import Directive, applicable_checks, next_step, select_edge
from writing_agent.task_graph_derive_common import (
    DeriveKey,
    build_transition,
    evidence_reader,
    payload_artifact,
)
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_evaluation import (
    FAMILIES,
    EvaluationRequestV1,
    verify_evaluation_evidence,
)
from writing_agent.task_graph_records import (
    EnvironmentStepV1,
    EvaluatorResultV1,
    OutcomeV1,
)
from writing_agent.task_graph_sampling import CURRENT_ELIGIBILITY
from writing_agent.task_graph_transition import (
    ArtifactReader,
    DerivedArtifact,
    LineageView,
    Transition,
)


def _build_outcome(view: LineageView, **changes: Any) -> tuple[OutcomeV1, DerivedArtifact]:
    outcome = replace(view.outcome, **changes)
    return outcome, payload_artifact(outcome)


def _step(view: LineageView, step: EnvironmentStepV1, kind: str) -> Directive:
    if not isinstance(step, EnvironmentStepV1):
        raise ProjectionError("outcome derive requires an environment step")
    directive = next_step(view)
    if directive.kind != kind or step != EnvironmentStepV1.of(directive):
        raise ProjectionError("environment step differs from the current directive")
    return directive


def _check_request(
    view: LineageView, ref: str, reader: ArtifactReader
) -> tuple[Mapping[str, Any], CheckContractV1]:
    request = reader.artifact(ref, private=True)
    if not isinstance(request, Mapping) or request.get("record_type") != "CheckRequestV1":
        raise ProjectionError("outstanding check request has wrong type")
    check = view.node.checks.get(request.get("check_id"))
    packet_ref = view.node.contract.completion_contract.evaluation_packet_ref
    expected = {
        "target_checkpoint": view.outcome.candidate_checkpoint,
        "requirement_version": view.outcome.requirement_version,
        "check_contract_hash": check.identity() if check else None,
        "evaluator_packet_ref": packet_ref,
    }
    if (
        check is None
        or packet_ref is None
        or any(request.get(key) != value for key, value in expected.items())
    ):
        raise ProjectionError("check request differs from the admitted candidate")
    return request, check


def _result_statuses(view: LineageView) -> tuple[dict[str, str], list[str]]:
    statuses = {key: value for key, value in view.check_statuses.items() if value is not None}
    refs = [row["result_ref"] for row in view.outcome.checks if row["result_ref"] is not None]
    return statuses, refs


def derive_check_request(
    view: LineageView, step: EnvironmentStepV1, reader: ArtifactReader
) -> Transition:
    _step(view, step, "request_checks")
    checks = applicable_checks(view, view.state.continuation["feedback_cursor"])
    packet_ref = view.node.contract.completion_contract.evaluation_packet_ref
    if not checks or packet_ref is None:
        raise ProjectionError("no admitted checks are applicable")

    purpose = (
        "progress"
        if view.state.continuation["feedback_cursor"] < len(view.mode.feedback_rules)
        else "completion"
    )
    request_prefix = f"{view.state.position['lineage_id']}:check:{view.state.history['seq']}:"
    request_artifacts = tuple(
        payload_artifact(
            {
                "record_type": "CheckRequestV1",
                "schema": 1,
                "request_id": request_prefix + check.id,
                "target_checkpoint": view.checkpoint_id,
                "requirement_version": view.state.requirements_ref,
                "check_contract_hash": check.identity(),
                "evaluator_packet_ref": packet_ref,
                "check_id": check.id,
                "purpose": purpose,
            },
            "private",
        )
        for check in checks
    )
    check_refs = tuple(item.ref for item in request_artifacts)
    outcome, outcome_artifact = _build_outcome(
        view,
        candidate_checkpoint=view.checkpoint_id,
        requirement_version=view.state.requirements_ref,
        checks=tuple({"request_ref": ref, "result_ref": None} for ref in check_refs),
        transition_edge_id=None,
    )
    continuation = view.state.to_dict()["continuation"]
    continuation["check_requests"] = list(check_refs)
    check_statuses = dict(view.check_statuses)
    check_statuses.update({check.id: None for check in checks})
    return build_transition(
        view,
        step,
        kind="external_requested",
        actor="environment",
        audience=("controller", "evaluator", "trainer"),
        state_changes={
            "position": {"phase": "awaiting_checks"},
            "continuation": continuation,
            "outcome_ref": outcome_artifact.ref,
        },
        artifacts=(*request_artifacts, outcome_artifact),
        outcome=outcome,
        check_statuses=check_statuses,
    )


def derive_check_result(
    view: LineageView, result: EvaluatorResultV1, reader: ArtifactReader
) -> Transition:
    if not isinstance(result, EvaluatorResultV1) or next_step(view).kind != "await_check_result":
        raise ProjectionError("no check result is currently requested")
    pending = view.state.continuation["check_requests"]
    if not pending or result.request_ref != pending[0]:
        raise ProjectionError("check result is stale, duplicate, or out of order")
    request, check = _check_request(view, result.request_ref, reader)
    family = next(
        (item for item in FAMILIES.values() if item.check_version == check.evaluator_version), None
    )
    if family is None:
        raise ProjectionError("check has no admitted evaluator family")
    target_ref = request["target_checkpoint"]
    target = reader.checkpoint(target_ref)
    packet_ref = request["evaluator_packet_ref"]
    packet = reader.artifact(packet_ref, private=True)
    evaluation_request = EvaluationRequestV1.create(
        family.name,
        target_ref,
        check,
        packet_ref,
        target.state.files,
        evaluator_packet=packet,
    )
    evidence = reader.artifact(result.evidence_ref)
    verified = verify_evaluation_evidence(
        evaluation_request, evidence, evidence_reader(reader, packet_ref=packet_ref)
    )
    if verified.family != family.name or verified.status != result.status:
        raise ProjectionError("evaluator result contradicts admitted evidence")

    checks = [dict(row) for row in view.outcome.checks]
    index = next(
        (index for index, row in enumerate(checks) if row["request_ref"] == result.request_ref),
        None,
    )
    if index is None or checks[index]["result_ref"] is not None:
        raise ProjectionError("check result does not fill one pending outcome row")
    result_artifact = payload_artifact(result)
    checks[index]["result_ref"] = result_artifact.ref
    outcome, outcome_artifact = _build_outcome(view, checks=checks)
    continuation = view.state.to_dict()["continuation"]
    continuation["check_requests"] = list(pending[1:])
    continuation["applied_responses"] = [
        *continuation["applied_responses"],
        request["request_id"],
    ]
    check_statuses = dict(view.check_statuses)
    check_statuses[check.id] = result.status
    return build_transition(
        view,
        result,
        kind="check_recorded",
        actor="evaluator",
        audience=("controller", "evaluator", "trainer"),
        state_changes={
            "position": {"phase": "awaiting_checks"},
            "continuation": continuation,
            "outcome_ref": outcome_artifact.ref,
        },
        artifacts=(outcome_artifact,),
        outcome=outcome,
        check_statuses=check_statuses,
    )


def derive_transition(
    view: LineageView, step: EnvironmentStepV1, reader: ArtifactReader
) -> Transition:
    directive = _step(view, step, "commit_transition")
    edge = select_edge(view.node, view, task_status=directive.task_status)
    if directive.task_status is None or edge is None or edge.effect != "terminate":
        raise ProjectionError("transition does not select an admitted terminal edge")
    outcome, outcome_artifact = _build_outcome(
        view,
        task_status=directive.task_status,
        transition_edge_id=edge.edge_id,
    )
    return build_transition(
        view,
        step,
        kind="transition_committed",
        actor="environment",
        audience=("controller", "evaluator", "trainer"),
        state_changes={
            "position": {"phase": "ready_transition"},
            "outcome_ref": outcome_artifact.ref,
        },
        artifacts=(outcome_artifact,),
        outcome=outcome,
    )


def derive_seal(view: LineageView, step: EnvironmentStepV1, reader: ArtifactReader) -> Transition:
    directive = _step(view, step, "seal_outcome")
    if directive.task_status is None:
        raise ProjectionError("outcome is not ready to seal")
    outcome, outcome_artifact = _build_outcome(
        view,
        task_status=directive.task_status,
        execution_status="valid",
        stop_reason=directive.stop_reason,
        reward_status="pending",
        training_eligibility="pending",
    )
    return build_transition(
        view,
        step,
        kind="termination_recorded",
        actor="environment",
        audience=("controller", "evaluator", "trainer"),
        state_changes={
            "position": {"phase": "terminal"},
            "outcome_ref": outcome_artifact.ref,
        },
        artifacts=(outcome_artifact,),
        outcome=outcome,
    )


def derive_exhausted_stop(
    view: LineageView, step: EnvironmentStepV1, reader: ArtifactReader
) -> Transition:
    _step(view, step, "stop_exhausted")
    reason = exhausted_stop_reason(view.budget)
    if reason is None or step.directive["stop_reason"] != reason:
        raise ProjectionError("writer exhaustion does not authorize this stop")
    outcome, outcome_artifact = _build_outcome(
        view,
        task_status="incomplete",
        execution_status="valid",
        stop_reason=reason,
        candidate_checkpoint=view.checkpoint_id,
        requirement_version=view.state.requirements_ref,
        checks=(),
        transition_edge_id=None,
        reward_status="pending",
        training_eligibility="pending",
    )
    return build_transition(
        view,
        step,
        kind="termination_recorded",
        actor="writer_runtime",
        audience=("controller", "evaluator", "trainer"),
        state_changes={
            "position": {"phase": "terminal"},
            "outcome_ref": outcome_artifact.ref,
        },
        artifacts=(outcome_artifact,),
        outcome=outcome,
        check_statuses={},
    )


def derive_reward(view: LineageView, step: EnvironmentStepV1, reader: ArtifactReader) -> Transition:
    _step(view, step, "publish_reward")
    contract = view.mode.reward
    if contract is None:
        raise ProjectionError("outcome has no admitted reward contract")
    statuses, result_refs = _result_statuses(view)
    completed = view.outcome.task_status in {"complete", "accepted_partial"}
    if completed and set(contract.components) - set(statuses):
        raise ProjectionError("reward components lack terminal check evidence")
    components = {
        check_id: {
            "weight": weight,
            "earned": weight if statuses.get(check_id) == "pass" else 0,
            "status": statuses.get(check_id, "not_run"),
        }
        for check_id, weight in contract.components.items()
    }
    numerator = (
        sum(component["earned"] for component in components.values())
        if completed
        else contract.incomplete_score
    )
    eligibility = CURRENT_ELIGIBILITY.training_wire(view.state.outcome_ref)
    eligibility_artifact = payload_artifact(eligibility)
    reward = {
        "record_type": "RewardV1",
        "schema": 1,
        "terminal_outcome_ref": view.state.outcome_ref,
        "reward_contract_ref": contract.identity(),
        "candidate_checkpoint": view.outcome.candidate_checkpoint,
        "check_result_refs": result_refs,
        "components": components,
        "numerator": numerator,
        "normalization": contract.normalization,
        "availability": "available",
        "eligibility_ref": eligibility_artifact.ref,
    }
    reward_artifact = payload_artifact(reward)
    outcome, outcome_artifact = _build_outcome(
        view,
        reward_status="available",
        training_eligibility=CURRENT_ELIGIBILITY.training_status,
        reward_ref=reward_artifact.ref,
        eligibility_ref=eligibility_artifact.ref,
    )
    return build_transition(
        view,
        step,
        kind="reward_recorded",
        actor="evaluator",
        audience=("controller", "evaluator", "trainer"),
        state_changes={
            "position": {"phase": "terminal"},
            "outcome_ref": outcome_artifact.ref,
        },
        artifacts=(eligibility_artifact, reward_artifact, outcome_artifact),
        outcome=outcome,
    )


DERIVES: dict[DeriveKey, Callable] = {
    ("EnvironmentStepV1", "request_checks"): derive_check_request,
    ("EnvironmentStepV1", "commit_transition"): derive_transition,
    ("EnvironmentStepV1", "seal_outcome"): derive_seal,
    ("EnvironmentStepV1", "stop_exhausted"): derive_exhausted_stop,
    ("EnvironmentStepV1", "publish_reward"): derive_reward,
    "EvaluatorResultV1": derive_check_result,
}


__all__ = [
    "DERIVES",
    "derive_check_request",
    "derive_check_result",
    "derive_transition",
    "derive_seal",
    "derive_exhausted_stop",
    "derive_reward",
]
