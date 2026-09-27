"""Pure checks, transition, terminal, and reward derives for wire-v1 outcomes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Annotated, Any, ClassVar

from writing_agent.task_graph import CheckpointV1, EventV1, domain_hash
from writing_agent.task_graph_accounting import exhausted_stop_reason
from writing_agent.task_graph_contracts import CheckContractV1
from writing_agent.task_graph_controller import Directive, next_step, select_edge
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_evaluation import (
    FAMILIES,
    EvaluationRequestV1,
    verify_evaluation_evidence,
)
from writing_agent.task_graph_records import (
    RECORD_TYPES,
    EnvironmentStepV1,
    EvaluatorResultV1,
    OutcomeV1,
    _WireRecord,
)
from writing_agent.task_graph_sampling import CURRENT_ELIGIBILITY
from writing_agent.task_graph_transition import (
    ArtifactReader,
    CheckpointChain,
    DerivedArtifact,
    LineageView,
    Transition,
)
from writing_agent.task_graph_wire import DictOf, JsonValue


@dataclass(frozen=True)
class _PayloadArtifact(_WireRecord):
    body: Annotated[Mapping[str, Any], DictOf(JsonValue())]
    RECORD_TYPE: ClassVar[str | None] = None

    def to_wire(self) -> dict[str, Any]:
        return dict(self.body)


@dataclass(frozen=True)
class _ReaderEvidenceResolver:
    reader: ArtifactReader
    packet_ref: str

    def read_evaluator_packet(self, ref: str) -> Mapping[str, Any]:
        if ref != self.packet_ref:
            raise ProjectionError("evaluator requested an unauthorized packet")
        return self.reader.artifact(ref, private=True)


def _artifact(value: _WireRecord | Mapping[str, Any], kind: str) -> DerivedArtifact:
    typed = isinstance(value, _WireRecord)
    body = value.to_wire() if typed else dict(value)
    record_type = body.get("record_type")
    if not typed and record_type in RECORD_TYPES:
        RECORD_TYPES[record_type].from_dict(body)
    ref = value.identity() if typed and value.RECORD_TYPE else domain_hash("payload", body)
    return DerivedArtifact(ref, value if typed else _PayloadArtifact(body), kind)


def _build_outcome(view: LineageView, **changes: Any) -> tuple[OutcomeV1, DerivedArtifact]:
    outcome = replace(view.outcome, **changes)
    return outcome, _artifact(outcome, "artifact")


def _step(view: LineageView, step: EnvironmentStepV1, kind: str) -> Directive:
    if not isinstance(step, EnvironmentStepV1):
        raise ProjectionError("outcome derive requires an environment step")
    directive = next_step(view)
    expected = {"kind": kind}
    if kind == "commit_transition":
        expected["edge_id"] = directive.edge_id
    elif kind == "seal_outcome":
        expected.update(task_status=directive.task_status, stop_reason=directive.stop_reason)
    elif kind == "stop_exhausted":
        expected["stop_reason"] = directive.stop_reason
    if directive.kind != kind or step.directive != expected:
        raise ProjectionError("environment step differs from the current directive")
    return directive


def _applicable_checks(view: LineageView) -> tuple[CheckContractV1, ...]:
    cursor = view.state.continuation["feedback_cursor"]
    if cursor < len(view.mode.feedback_rules):
        applicable = {"each_turn", f"before_feedback:{view.mode.feedback_rules[cursor]['id']}"}
    else:
        applicable = {"each_turn", "node_exit_candidate"}
    return tuple(check for check in view.node.checks.values() if check.applicability in applicable)


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


def _transition(
    view: LineageView,
    input_record: EnvironmentStepV1 | EvaluatorResultV1,
    outcome: OutcomeV1,
    outcome_artifact: DerivedArtifact,
    *,
    kind: str,
    actor: str,
    phase: str,
    continuation: Mapping[str, Any] | None = None,
    check_statuses: Mapping[str, str | None] | None = None,
    extra_artifacts: tuple[DerivedArtifact, ...] = (),
) -> Transition:
    state = view.state
    input_artifact = _artifact(input_record, "artifact")
    event = EventV1(
        previous=state.history["head"],
        seq=state.history["seq"] + 1,
        lineage_id=state.position["lineage_id"],
        node_visit_id=state.position["visit_id"],
        kind=kind,
        actor=actor,
        audience=("controller", "evaluator", "trainer"),
        payload_ref=input_artifact.ref,
        versions_ref=state.versions_ref,
        provenance_ref=state.provenance_ref,
    )
    next_state = replace(
        state,
        position={**state.position, "phase": phase},
        history={**state.history, "head": event.id, "seq": event.seq},
        outcome_ref=outcome_artifact.ref,
        continuation=state.continuation if continuation is None else continuation,
    )
    checkpoint = CheckpointV1(
        parents=(view.checkpoint_id,), state=next_state, event_head=event.id, artifact_refs=()
    )
    next_view = replace(
        view,
        checkpoint_id=checkpoint.identity(),
        head_event_id=event.id,
        state=next_state,
        outcome=outcome,
        check_statuses=view.check_statuses if check_statuses is None else check_statuses,
        ancestry=CheckpointChain(checkpoint.identity(), view.context, view.ancestry),
    )
    return Transition(
        event,
        input_record,
        next_state,
        (input_artifact, *extra_artifacts, outcome_artifact),
        next_view,
    )


def derive_check_request(
    view: LineageView, step: EnvironmentStepV1, reader: ArtifactReader
) -> Transition:
    _step(view, step, "request_checks")
    checks = _applicable_checks(view)
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
        _artifact(
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
    return _transition(
        view,
        step,
        outcome,
        outcome_artifact,
        kind="external_requested",
        actor="environment",
        phase="awaiting_checks",
        continuation=continuation,
        check_statuses=check_statuses,
        extra_artifacts=request_artifacts,
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
        evaluation_request,
        evidence,
        _ReaderEvidenceResolver(reader, packet_ref),
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
    result_artifact = _artifact(result, "artifact")
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
    return _transition(
        view,
        result,
        outcome,
        outcome_artifact,
        kind="check_recorded",
        actor="evaluator",
        phase="awaiting_checks",
        continuation=continuation,
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
    return _transition(
        view,
        step,
        outcome,
        outcome_artifact,
        kind="transition_committed",
        actor="environment",
        phase="ready_transition",
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
    return _transition(
        view,
        step,
        outcome,
        outcome_artifact,
        kind="termination_recorded",
        actor="environment",
        phase="terminal",
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
    return _transition(
        view,
        step,
        outcome,
        outcome_artifact,
        kind="termination_recorded",
        actor="writer_runtime",
        phase="terminal",
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
    eligibility_artifact = _artifact(eligibility, "artifact")
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
    reward_artifact = _artifact(reward, "artifact")
    outcome, outcome_artifact = _build_outcome(
        view,
        reward_status="available",
        training_eligibility=CURRENT_ELIGIBILITY.training_status,
        reward_ref=reward_artifact.ref,
        eligibility_ref=eligibility_artifact.ref,
    )
    return _transition(
        view,
        step,
        outcome,
        outcome_artifact,
        kind="reward_recorded",
        actor="evaluator",
        phase="terminal",
        extra_artifacts=(eligibility_artifact, reward_artifact),
    )


def derive_environment_step(
    view: LineageView, step: EnvironmentStepV1, reader: ArtifactReader
) -> Transition:
    if not isinstance(step, EnvironmentStepV1):
        raise ProjectionError("outcome derive requires an environment step")
    derives = {
        "request_checks": derive_check_request,
        "commit_transition": derive_transition,
        "seal_outcome": derive_seal,
        "stop_exhausted": derive_exhausted_stop,
        "publish_reward": derive_reward,
    }
    try:
        derive = derives[step.directive["kind"]]
    except KeyError as exc:
        raise ProjectionError("unsupported environment step") from exc
    return derive(view, step, reader)


DERIVES: dict[str, Callable] = {
    "EnvironmentStepV1": derive_environment_step,
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
    "derive_environment_step",
]
