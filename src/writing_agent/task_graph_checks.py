"""Frozen deterministic file checks for the scripted-author graph slice."""

from __future__ import annotations

from collections.abc import Mapping

from writing_agent.task_graph import file_hash
from writing_agent.task_graph_contracts import (
    CheckContractV1,
    InteractionPolicyV1,
    NodeContractV1,
)
from writing_agent.task_graph_environment import WriterRuntimeError


def deterministic_check(check: CheckContractV1, files: Mapping[str, str]) -> tuple[str, dict]:
    """Evaluate only the strict admitted file-target vocabulary."""
    spec = check.spec
    path = spec["path"]
    text = files.get(path)
    if text is None:
        return "fail", {"path": path, "subject_hash": None, "reason": "missing_path"}
    kind = spec["kind"]
    if kind == "nonempty":
        passed = bool(text.strip())
    elif kind == "contains":
        passed = spec["text"].casefold() in text.casefold()
    elif kind == "excludes":
        passed = spec["text"].casefold() not in text.casefold()
    elif kind == "excludes_all":
        passed = not any(item.casefold() in text.casefold() for item in spec["texts"])
    elif kind == "word_range":
        passed = spec["min"] <= len(text.split()) <= spec["max"]
    elif kind == "exact":
        passed = text == spec["text"]
    else:
        raise ValueError("check kind was not admitted for deterministic evaluator")
    return ("pass" if passed else "fail"), {
        "path": path,
        "subject_hash": file_hash(text),
        "reason": None,
    }


def applicable_checks(node, feedback_cursor: int) -> tuple[CheckContractV1, ...]:
    feedback = node.interaction_policy.mandatory_feedback
    if feedback_cursor < len(feedback):
        scope = f"before_feedback:{feedback[feedback_cursor]}"
        return tuple(
            check for check in node.checks.values() if check.applicability in {"each_turn", scope}
        )
    return tuple(
        check
        for check in node.checks.values()
        if check.applicability in {"each_turn", "node_exit_candidate"}
    )


class DeterministicChecksV1:
    def __init__(self, writer, dependencies=None):
        self.writer = writer
        self.dependencies = dependencies or writer.dependencies
        self.environment = self.dependencies.environment
        self.store = writer.store

    def request_checks(self, runtime):
        node, _ = self.writer.validate_runtime(runtime)
        state = runtime.state
        if node.contract.interaction_contract.mode != "scripted_author":
            raise WriterRuntimeError("strict checks require scripted-author admission")
        if state.position["phase"] != "checking" or state.continuation["next_call"] != len(
            state.continuation["tool_queue"]
        ):
            raise WriterRuntimeError("checks require a quiescent completed writer turn")
        checks = applicable_checks(node, state.continuation["feedback_cursor"])
        if not checks:
            raise WriterRuntimeError("no applicable checks at this boundary")
        request_refs = []
        for check in checks:
            request = {
                "record_type": "CheckRequestV1",
                "schema": 1,
                "request_id": f"{self.writer.rollout_id}:check:{state.history['seq']}:{check.id}",
                "target_checkpoint": runtime.checkpoint_id,
                "requirement_version": state.requirements_ref,
                "check_contract_hash": check.identity(),
                "evaluator_packet_ref": node.contract.completion_contract.evaluation_packet_ref,
                "check_id": check.id,
                "purpose": "progress"
                if state.continuation["feedback_cursor"]
                < len(node.interaction_policy.mandatory_feedback)
                else "completion",
            }
            request_refs.append(self.store.put_artifact(request, private=True))
        record = {
            "record_type": "CheckBatchV1",
            "schema": 1,
            "target_checkpoint": runtime.checkpoint_id,
            "request_refs": request_refs,
        }
        continuation = state.to_dict()["continuation"]
        continuation["check_requests"] = request_refs
        position = state.to_dict()["position"]
        position["phase"] = "awaiting_checks"
        return self.environment.publish_record(
            runtime,
            "external_requested",
            "environment",
            record=record,
            changes={"continuation": continuation, "position": position},
            extra_refs=tuple(request_refs),
            restore_prefix="check",
            result_effect=True,
        )

    def check_next(self, runtime):
        node, _ = self.writer.validate_runtime(runtime)
        state = runtime.state
        if state.position["phase"] != "awaiting_checks" or not state.continuation["check_requests"]:
            raise WriterRuntimeError("no outstanding frozen check request")
        request_ref = state.continuation["check_requests"][0]
        request = self.store.get_artifact(request_ref, private=True)
        target = self.store.load_checkpoint(request["target_checkpoint"])
        check = node.checks[request["check_id"]]
        status, evidence = self.dependencies.evaluator.evaluate(check, target.state.files)
        evidence_ref = self.store.put_artifact(
            {
                "record_type": "DeterministicCheckEvidenceV1",
                "schema": 1,
                "target_checkpoint": request["target_checkpoint"],
                "check_contract_hash": check.identity(),
                "evaluator_packet_ref": request["evaluator_packet_ref"],
                "evidence": evidence,
                "status": status,
            }
        )
        result = {
            "record_type": "CheckResultV1",
            "schema": 1,
            "request_ref": request_ref,
            "request_id": request["request_id"],
            "target_checkpoint": request["target_checkpoint"],
            "requirement_version": request["requirement_version"],
            "check_contract_hash": check.identity(),
            "evaluator_packet_ref": request["evaluator_packet_ref"],
            "status": status,
            "evidence_ref": evidence_ref,
        }
        continuation = state.to_dict()["continuation"]
        continuation["check_requests"] = list(state.continuation["check_requests"][1:])
        continuation["applied_responses"] = [
            *state.continuation["applied_responses"],
            request["request_id"],
        ]
        return self.environment.publish_record(
            runtime,
            "check_recorded",
            "evaluator",
            record=result,
            changes={"continuation": continuation},
            extra_refs=(evidence_ref,),
            restore_prefix="check",
            result_effect=True,
        )


def _contracts(store, state):
    instance = store.load_instance(state.instance_ref)
    spec = next(node for node in instance.nodes if node.id == state.position["node_id"])
    contract = NodeContractV1.from_dict(store.get_artifact(spec.entry_contract))
    policy = InteractionPolicyV1.from_dict(
        store.get_artifact(contract.interaction_contract.interaction_policy_ref)
    )
    checks = tuple(
        CheckContractV1.from_dict(store.get_artifact(ref, private=True))
        for ref in (*contract.mandatory_checks, *contract.optional_checks)
    )
    return contract, policy, checks


def validate_check_batch_effect(store, before, after, event, effect, entry) -> None:
    from writing_agent.task_graph_projection import ProjectionError

    record = store.get_artifact(entry["record_ref"])
    if (
        not isinstance(record, dict)
        or set(record) != {"record_type", "schema", "target_checkpoint", "request_refs"}
        or record["record_type"] != "CheckBatchV1"
        or record["schema"] != 1
    ):
        raise ProjectionError("check batch has wrong record type")
    target_id = record["target_checkpoint"]
    target = store.load_checkpoint(target_id)
    contract, policy, checks = _contracts(store, before)
    if before.continuation["feedback_cursor"] < len(policy.mandatory_feedback):
        scope = (
            f"before_feedback:{policy.mandatory_feedback[before.continuation['feedback_cursor']]}"
        )
        expected_checks = [c for c in checks if c.applicability in {"each_turn", scope}]
        purpose = "progress"
    else:
        expected_checks = [
            c for c in checks if c.applicability in {"each_turn", "node_exit_candidate"}
        ]
        purpose = "completion"
    expected_requests = []
    for check in expected_checks:
        expected_requests.append(
            {
                "record_type": "CheckRequestV1",
                "schema": 1,
                "request_id": f"{event.rollout_id}:check:{before.history['seq']}:{check.id}",
                "target_checkpoint": target_id,
                "requirement_version": before.requirements_ref,
                "check_contract_hash": check.identity(),
                "evaluator_packet_ref": contract.completion_contract.evaluation_packet_ref,
                "check_id": check.id,
                "purpose": purpose,
            }
        )
    actual_requests = [store.get_artifact(ref, private=True) for ref in record["request_refs"]]
    continuation = before.to_dict()["continuation"]
    continuation["check_requests"] = record["request_refs"]
    position = before.to_dict()["position"]
    position["phase"] = "awaiting_checks"
    if (
        not expected_checks
        or target.state != before
        or target.event_head != before.history["head"]
        or before.position["phase"] != "checking"
        or before.continuation["next_call"] != len(before.continuation["tool_queue"])
        or before.continuation["check_requests"]
        or actual_requests != expected_requests
        or event.actor != "environment"
        or "writer" in event.audience
        or set(effect["set"]) != {"position", "continuation", "external_inputs_ref"}
        or effect["history_set"]
        or effect["file_delta"]
        or after.to_dict()["position"] != position
        or after.to_dict()["continuation"] != continuation
        or after.files != before.files
        or after.context_ref != before.context_ref
        or after.budgets_ref != before.budgets_ref
        or after.outcome_ref != before.outcome_ref
    ):
        raise ProjectionError("check batch does not freeze the exact quiescent candidate")


def validate_check_result_effect(store, before, after, event, effect, entry) -> None:
    from writing_agent.task_graph_projection import ProjectionError

    record = store.get_artifact(entry["record_ref"])
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "record_type",
            "schema",
            "request_ref",
            "request_id",
            "target_checkpoint",
            "requirement_version",
            "check_contract_hash",
            "evaluator_packet_ref",
            "status",
            "evidence_ref",
        }
        or record["record_type"] != "CheckResultV1"
        or record["schema"] != 1
    ):
        raise ProjectionError("check result has wrong record type")
    pending = before.continuation["check_requests"]
    if not pending or record["request_ref"] != pending[0]:
        raise ProjectionError("check result does not match next outstanding request")
    request = store.get_artifact(pending[0], private=True)
    target = store.load_checkpoint(request["target_checkpoint"])
    check = CheckContractV1.from_dict(
        store.get_artifact(request["check_contract_hash"], private=True)
    )
    status, evidence = deterministic_check(check, target.state.files)
    expected_evidence = {
        "record_type": "DeterministicCheckEvidenceV1",
        "schema": 1,
        "target_checkpoint": request["target_checkpoint"],
        "check_contract_hash": check.identity(),
        "evaluator_packet_ref": request["evaluator_packet_ref"],
        "evidence": evidence,
        "status": status,
    }
    expected_result = {
        "record_type": "CheckResultV1",
        "schema": 1,
        "request_ref": pending[0],
        "request_id": request["request_id"],
        "target_checkpoint": request["target_checkpoint"],
        "requirement_version": request["requirement_version"],
        "check_contract_hash": check.identity(),
        "evaluator_packet_ref": request["evaluator_packet_ref"],
        "status": status,
        "evidence_ref": record["evidence_ref"],
    }
    continuation = before.to_dict()["continuation"]
    continuation["check_requests"] = list(pending[1:])
    continuation["applied_responses"] = [
        *before.continuation["applied_responses"],
        request["request_id"],
    ]
    if (
        record != expected_result
        or store.get_artifact(record["evidence_ref"]) != expected_evidence
        or request["requirement_version"] != before.requirements_ref
        or before.position["phase"] != "awaiting_checks"
        or event.actor != "evaluator"
        or "writer" in event.audience
        or set(effect["set"]) != {"continuation", "external_inputs_ref"}
        or effect["history_set"]
        or effect["file_delta"]
        or after.to_dict()["continuation"] != continuation
        or after.position != before.position
        or after.files != before.files
        or after.context_ref != before.context_ref
        or after.budgets_ref != before.budgets_ref
        or after.outcome_ref != before.outcome_ref
    ):
        raise ProjectionError("check result is not bound to frozen target/evidence")
