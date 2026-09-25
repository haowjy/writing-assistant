"""Environment-owned terminal routing and exact deterministic reward records."""

from __future__ import annotations

from types import MappingProxyType

from writing_agent.task_graph_controller import (
    ControllerViewV1,
    OutcomeStatusV1,
    evaluate_guard,
)
from writing_agent.task_graph_environment import WriterRuntimeError
from writing_agent.task_graph_sampling import CURRENT_ELIGIBILITY


def current_check_results(store, state):
    log = store.get_artifact(state.external_inputs_ref)
    batches = [
        store.get_artifact(entry["record_ref"])
        for entry in log["entries"]
        if entry["kind"] == "external_requested"
        and "public" in store.artifact_visibilities(entry["record_ref"])
    ]
    if not batches:
        return None, {}
    batch = batches[-1]
    if batch.get("record_type") != "CheckBatchV1":
        raise WriterRuntimeError("latest external check request has wrong type")
    results = {}
    for entry in log["entries"]:
        if entry["kind"] != "check_recorded":
            continue
        result = store.get_artifact(entry["record_ref"])
        if result["request_ref"] in batch["request_refs"]:
            if result["request_ref"] in results:
                raise WriterRuntimeError("conflicting duplicate check result")
            results[result["request_ref"]] = (entry["record_ref"], result)
    return batch, results


def _check_summary(store, state, node):
    batch, results = current_check_results(store, state)
    if batch is None or len(results) != len(batch["request_refs"]):
        raise WriterRuntimeError("terminal decision needs every frozen check result")
    statuses = {}
    refs = []
    for request_ref in batch["request_refs"]:
        request = store.get_artifact(request_ref, private=True)
        result_ref, result = results[request_ref]
        if result["status"] == "unavailable":
            raise WriterRuntimeError("unavailable check cannot settle terminal outcome")
        statuses[request["check_id"]] = result["status"]
        refs.append(result_ref)
    mandatory = [
        check.id
        for check in node.checks.values()
        if check.required and check.applicability in {"each_turn", "node_exit_candidate"}
    ]
    relevant = [check_id for check_id in mandatory if check_id in statuses]
    if not relevant or len(relevant) != len(mandatory):
        raise WriterRuntimeError("terminal checks do not cover every mandatory check")
    return batch, statuses, refs


def _required_passes(node, statuses):
    return all(
        statuses.get(check.id) == "pass"
        for check in node.checks.values()
        if check.required and check.applicability in {"each_turn", "node_exit_candidate"}
    )


def _feedback_stop_evidence(store, state, node, candidate_hint=None):
    from writing_agent.task_graph_checks import applicable_checks

    cursor = state.continuation["feedback_cursor"]
    if cursor >= len(node.script.feedback) or state.continuation["check_requests"]:
        raise WriterRuntimeError("no outstanding feedback obligation can be terminalized")
    rule = node.script.feedback[cursor]
    progress_checks = applicable_checks(node, cursor)
    if progress_checks and state.position["phase"] != "awaiting_checks":
        raise WriterRuntimeError("progress checks have not run")
    if not progress_checks and state.position["phase"] != "checking":
        raise WriterRuntimeError("feedback has no completed-turn boundary")
    if progress_checks:
        batch, results = current_check_results(store, state)
        if batch is None or len(results) != len(batch["request_refs"]):
            raise WriterRuntimeError("progress results are incomplete")
        candidate = batch["target_checkpoint"]
        refs = [results[request_ref][0] for request_ref in batch["request_refs"]]
        statuses = {
            store.get_artifact(request_ref, private=True)["check_id"]: results[request_ref][1][
                "status"
            ]
            for request_ref in batch["request_refs"]
        }
        if any(status == "unavailable" for status in statuses.values()):
            raise WriterRuntimeError("unavailable progress check cannot settle feedback")
    else:
        candidate = candidate_hint
        refs = []
        statuses = {}
        if candidate is None:
            raise WriterRuntimeError("feedback stop needs its quiescent candidate")
    budget = store.get_artifact(state.budgets_ref)
    if any(statuses.get(check_id) != "pass" for check_id in rule["prerequisite_check_ids"]):
        reason = "feedback_prerequisite_failed"
    elif budget["consumed"].get("author_calls", 0) >= budget["limits"]["author_calls"]:
        reason = "author_budget"
    elif budget["consumed"].get("writer_turns", 0) >= budget["limits"]["writer_turns"]:
        reason = "writer_budget"
    else:
        raise WriterRuntimeError("feedback can proceed; incomplete stop is not authorized")
    return candidate, refs, reason


def applicable_feedback_checks(node, state):
    from writing_agent.task_graph_checks import applicable_checks

    return applicable_checks(node, state.continuation["feedback_cursor"])


class ScriptedTerminalV1:
    def __init__(self, writer):
        self.writer = writer
        self.store = writer.store

    def transition(self, runtime):
        node, _ = self.writer.validate_runtime(runtime)
        state = runtime.state
        if state.position["phase"] != "awaiting_checks" or state.continuation["check_requests"]:
            raise WriterRuntimeError("transition needs all required check results")
        if state.continuation["feedback_cursor"] != len(node.interaction_policy.mandatory_feedback):
            raise WriterRuntimeError("mandatory feedback remains")
        batch, statuses, refs = _check_summary(self.store, state, node)
        if not _required_passes(node, statuses):
            raise WriterRuntimeError("failing checks cannot authorize a complete transition")
        old_budget = self.store.get_artifact(state.budgets_ref)
        remaining = {
            key: max(0, value - old_budget["consumed"].get(key, 0))
            for key, value in old_budget["limits"].items()
        }
        task_status = (
            "accepted_partial" if node.contract.completion_contract.accepted_partial else "complete"
        )
        view = ControllerViewV1(
            node_id=node.spec.id,
            phase="ready_transition",
            outcome=OutcomeStatusV1(task_status=task_status, execution_status="valid"),
            check_status=MappingProxyType(statuses),
            interaction_complete=True,
            budgets_remaining=MappingProxyType(remaining),
        )
        matching = [edge for edge in node.edges if evaluate_guard(node.guards[edge.edge_id], view)]
        if not matching:
            raise WriterRuntimeError("no applicable completion edge")
        matching.sort(key=lambda edge: edge.precedence or 0)
        edge = matching[0]
        if edge.effect != "terminate":
            raise WriterRuntimeError("this single-node slice requires a terminate edge")
        record = {
            "record_type": "TransitionDecisionV1",
            "schema": 1,
            "candidate_checkpoint": batch["target_checkpoint"],
            "edge_id": edge.edge_id,
            "effect": edge.effect,
            "task_status": task_status,
            "check_result_refs": refs,
            "requirement_version": state.requirements_ref,
        }
        position = state.to_dict()["position"]
        position["phase"] = "ready_transition"
        return self.writer.environment.publish_record(
            runtime,
            "transition_committed",
            "environment",
            record=record,
            changes={"position": position},
            extra_refs=tuple(refs),
            restore_prefix="terminal",
        )

    def terminal_outcome(self, runtime):
        node, _ = self.writer.validate_runtime(runtime)
        state = runtime.state
        if state.continuation["feedback_cursor"] < len(node.script.feedback):
            raise WriterRuntimeError("mandatory feedback requires delivery or verified stop")
        if state.position["phase"] not in {"ready_transition", "awaiting_checks"}:
            raise WriterRuntimeError("terminal outcome requires settled checks or transition")
        batch, statuses, refs = _check_summary(self.store, state, node)
        transition_ref = None
        task_status = "incomplete"
        stop_reason = "required_check_failed"
        if state.position["phase"] == "ready_transition":
            log = self.store.get_artifact(state.external_inputs_ref)
            transition_ref = next(
                self.store.get_artifact(entry["record_ref"])
                for entry in reversed(log["entries"])
                if entry["kind"] == "transition_committed"
            )
            task_status = transition_ref["task_status"]
            stop_reason = None
        elif _required_passes(node, statuses):
            raise WriterRuntimeError("passing checks require a transition decision")
        outcome = {
            "record_type": "TerminalOutcomeV1",
            "schema": 1,
            "candidate_checkpoint": batch["target_checkpoint"],
            "task_status": task_status,
            "execution_status": "valid",
            "stop_reason": stop_reason,
            "reward_status": "pending",
            "training_eligibility": "pending",
            "check_result_refs": refs,
            "transition_edge_id": transition_ref["edge_id"] if transition_ref else None,
            "requirement_version": state.requirements_ref,
        }
        outcome_ref = self.store.put_artifact(outcome)
        position = state.to_dict()["position"]
        position["phase"] = "terminal"
        record = {
            "record_type": "TerminalOutcomeCommitV1",
            "schema": 1,
            "outcome_ref": outcome_ref,
        }
        return self.writer.environment.publish_record(
            runtime,
            "termination_recorded",
            "environment",
            record=record,
            changes={"position": position, "outcome_ref": outcome_ref},
            extra_refs=(outcome_ref,),
            restore_prefix="terminal",
        )

    def stop_incomplete(self, runtime):
        """Seal a verifiable feedback prerequisite or budget failure."""
        node, _ = self.writer.validate_runtime(runtime)
        state = runtime.state
        candidate, refs, reason = _feedback_stop_evidence(
            self.store, state, node, runtime.checkpoint_id
        )
        outcome = {
            "record_type": "TerminalOutcomeV1",
            "schema": 1,
            "candidate_checkpoint": candidate,
            "task_status": "incomplete",
            "execution_status": "valid",
            "stop_reason": reason,
            "reward_status": "pending",
            "training_eligibility": "pending",
            "check_result_refs": refs,
            "transition_edge_id": None,
            "requirement_version": state.requirements_ref,
        }
        outcome_ref = self.store.put_artifact(outcome)
        position = state.to_dict()["position"]
        position["phase"] = "terminal"
        return self.writer.environment.publish_record(
            runtime,
            "termination_recorded",
            "environment",
            record={
                "record_type": "TerminalOutcomeCommitV1",
                "schema": 1,
                "outcome_ref": outcome_ref,
            },
            changes={"position": position, "outcome_ref": outcome_ref},
            extra_refs=(outcome_ref,),
            restore_prefix="terminal",
        )

    def reward(self, runtime):
        node, _ = self.writer.validate_runtime(runtime)
        state = runtime.state
        if state.position["phase"] != "terminal":
            raise WriterRuntimeError("reward requires a terminal outcome")
        outcome = self.store.get_artifact(state.outcome_ref)
        if outcome.get("record_type") != "TerminalOutcomeV1":
            raise WriterRuntimeError("reward has no immutable terminal outcome")
        contract = node.reward_contract
        statuses = {}
        for ref in outcome["check_result_refs"]:
            result = self.store.get_artifact(ref)
            request = self.store.get_artifact(result["request_ref"], private=True)
            statuses[request["check_id"]] = result["status"]
        if outcome["task_status"] in {"complete", "accepted_partial"} and (
            set(contract.components) - set(statuses)
        ):
            raise WriterRuntimeError("reward components lack terminal check evidence")
        components = {
            check_id: {
                "weight": weight,
                "earned": weight if statuses.get(check_id) == "pass" else 0,
                "status": statuses.get(check_id, "not_run"),
            }
            for check_id, weight in contract.components.items()
        }
        numerator = (
            sum(item["earned"] for item in components.values())
            if outcome["task_status"] in {"complete", "accepted_partial"}
            else contract.incomplete_score
        )
        eligibility = CURRENT_ELIGIBILITY.training_wire(state.outcome_ref)
        eligibility_ref = self.store.put_artifact(eligibility)
        reward = {
            "record_type": "RewardV1",
            "schema": 1,
            "terminal_outcome_ref": state.outcome_ref,
            "reward_contract_ref": contract.identity(),
            "candidate_checkpoint": outcome["candidate_checkpoint"],
            "check_result_refs": outcome["check_result_refs"],
            "components": components,
            "numerator": numerator,
            "normalization": contract.normalization,
            "availability": "available",
            "eligibility_ref": eligibility_ref,
        }
        reward_ref = self.store.put_artifact(reward)
        availability = {
            "record_type": "RewardAvailabilityV1",
            "schema": 1,
            "terminal_outcome_ref": state.outcome_ref,
            "reward_ref": reward_ref,
            "eligibility_ref": eligibility_ref,
            "reward_status": "available",
            "training_eligibility": "ineligible",
        }
        availability_ref = self.store.put_artifact(availability)
        record = {
            "record_type": "RewardPublicationV1",
            "schema": 1,
            "availability_ref": availability_ref,
        }
        return self.writer.environment.publish_record(
            runtime,
            "external_response",
            "evaluator",
            record=record,
            changes={"outcome_ref": availability_ref},
            extra_refs=(reward_ref, availability_ref, eligibility_ref),
            restore_prefix="terminal",
        )


def _admitted_node(store, state):
    from writing_agent.task_graph_admission import StoreArtifactResolver, admit_graph

    graph = admit_graph(store.load_instance(state.instance_ref), StoreArtifactResolver(store))
    return graph.node(state.position["node_id"])


def validate_terminal_effect(store, before, after, event, effect, entry):
    """Recalculate edge, terminal outcome, and reward at publish and replay time."""
    from writing_agent.task_graph_projection import ProjectionError

    node = _admitted_node(store, before)
    record = store.get_artifact(entry["record_ref"])
    if (
        effect["file_delta"]
        or effect["history_set"]
        or after.files != before.files
        or after.context_ref != before.context_ref
        or after.budgets_ref != before.budgets_ref
        or after.continuation != before.continuation
        or after.requirements_ref != before.requirements_ref
        or after.decisions_ref != before.decisions_ref
        or after.disclosures_ref != before.disclosures_ref
        or event.actor not in {"environment", "evaluator"}
        or "writer" in event.audience
    ):
        raise ProjectionError("terminal producer changed execution state")
    if event.kind == "transition_committed":
        batch, statuses, refs = _check_summary(store, before, node)
        if not _required_passes(node, statuses):
            raise ProjectionError("transition committed despite failing check")
        if before.continuation["feedback_cursor"] != len(
            node.interaction_policy.mandatory_feedback
        ):
            raise ProjectionError("transition skipped mandatory feedback")
        old_budget = store.get_artifact(before.budgets_ref)
        remaining = {
            key: max(0, value - old_budget["consumed"].get(key, 0))
            for key, value in old_budget["limits"].items()
        }
        task_status = (
            "accepted_partial" if node.contract.completion_contract.accepted_partial else "complete"
        )
        view = ControllerViewV1(
            node_id=node.spec.id,
            phase="ready_transition",
            outcome=OutcomeStatusV1(task_status=task_status, execution_status="valid"),
            check_status=MappingProxyType(statuses),
            interaction_complete=True,
            budgets_remaining=MappingProxyType(remaining),
        )
        matching = [edge for edge in node.edges if evaluate_guard(node.guards[edge.edge_id], view)]
        matching.sort(key=lambda edge: edge.precedence or 0)
        if not matching or matching[0].effect != "terminate":
            raise ProjectionError("transition names no admitted terminal edge")
        expected_record = {
            "record_type": "TransitionDecisionV1",
            "schema": 1,
            "candidate_checkpoint": batch["target_checkpoint"],
            "edge_id": matching[0].edge_id,
            "effect": "terminate",
            "task_status": task_status,
            "check_result_refs": refs,
            "requirement_version": before.requirements_ref,
        }
        position = before.to_dict()["position"]
        position["phase"] = "ready_transition"
        if (
            before.position["phase"] != "awaiting_checks"
            or before.continuation["check_requests"]
            or record != expected_record
            or event.actor != "environment"
            or set(effect["set"]) != {"position", "external_inputs_ref"}
            or after.to_dict()["position"] != position
            or after.outcome_ref != before.outcome_ref
        ):
            raise ProjectionError("transition decision differs from frozen guards")
        return
    if event.kind == "termination_recorded":
        recorded_outcome = store.get_artifact(after.outcome_ref)
        transition = None
        if before.continuation["feedback_cursor"] < len(node.script.feedback):
            candidate, refs, stop_reason = _feedback_stop_evidence(
                store, before, node, recorded_outcome.get("candidate_checkpoint")
            )
            if not applicable_feedback_checks(node, before):
                target = store.load_checkpoint(candidate)
                if target.state != before or target.event_head != before.history["head"]:
                    raise ProjectionError("feedback stop did not freeze current candidate")
            task_status = "incomplete"
        else:
            batch, statuses, refs = _check_summary(store, before, node)
            candidate = batch["target_checkpoint"]
            if before.position["phase"] == "ready_transition":
                log = store.get_artifact(before.external_inputs_ref)
                matches = [
                    store.get_artifact(item["record_ref"])
                    for item in log["entries"]
                    if item["kind"] == "transition_committed"
                ]
                if len(matches) != 1:
                    raise ProjectionError("terminal outcome needs one committed transition")
                transition = matches[0]
                task_status = transition["task_status"]
                stop_reason = None
            else:
                task_status = "incomplete"
                stop_reason = "required_check_failed"
                if _required_passes(node, statuses):
                    raise ProjectionError("passing candidate was terminalized incomplete")
        expected_outcome = {
            "record_type": "TerminalOutcomeV1",
            "schema": 1,
            "candidate_checkpoint": candidate,
            "task_status": task_status,
            "execution_status": "valid",
            "stop_reason": stop_reason,
            "reward_status": "pending",
            "training_eligibility": "pending",
            "check_result_refs": refs,
            "transition_edge_id": transition["edge_id"] if transition else None,
            "requirement_version": before.requirements_ref,
        }
        position = before.to_dict()["position"]
        position["phase"] = "terminal"
        if (
            before.position["phase"] not in {"ready_transition", "awaiting_checks", "checking"}
            or before.continuation["check_requests"]
            or record
            != {
                "record_type": "TerminalOutcomeCommitV1",
                "schema": 1,
                "outcome_ref": after.outcome_ref,
            }
            or store.get_artifact(after.outcome_ref) != expected_outcome
            or event.actor != "environment"
            or set(effect["set"]) != {"position", "outcome_ref", "external_inputs_ref"}
            or after.to_dict()["position"] != position
        ):
            raise ProjectionError("terminal outcome is not causally bound to checks")
        return
    if event.kind == "external_response":
        outcome = store.get_artifact(before.outcome_ref)
        if outcome.get("record_type") != "TerminalOutcomeV1":
            raise ProjectionError("reward preceded terminal outcome")
        contract = node.reward_contract
        statuses = {}
        for result_ref in outcome["check_result_refs"]:
            result = store.get_artifact(result_ref)
            request = store.get_artifact(result["request_ref"], private=True)
            statuses[request["check_id"]] = result["status"]
        if outcome["task_status"] in {"complete", "accepted_partial"} and (
            set(contract.components) - set(statuses)
        ):
            raise ProjectionError("reward lacks declared component evidence")
        components = {
            check_id: {
                "weight": weight,
                "earned": weight if statuses.get(check_id) == "pass" else 0,
                "status": statuses.get(check_id, "not_run"),
            }
            for check_id, weight in contract.components.items()
        }
        numerator = (
            sum(item["earned"] for item in components.values())
            if outcome["task_status"] in {"complete", "accepted_partial"}
            else contract.incomplete_score
        )
        availability = store.get_artifact(after.outcome_ref)
        eligibility = store.get_artifact(availability["eligibility_ref"])
        reward = store.get_artifact(availability["reward_ref"])
        if (
            availability
            != {
                "record_type": "RewardAvailabilityV1",
                "schema": 1,
                "terminal_outcome_ref": before.outcome_ref,
                "reward_ref": availability["reward_ref"],
                "eligibility_ref": availability["eligibility_ref"],
                "reward_status": "available",
                "training_eligibility": "ineligible",
            }
            or eligibility != CURRENT_ELIGIBILITY.training_wire(before.outcome_ref)
            or reward
            != {
                "record_type": "RewardV1",
                "schema": 1,
                "terminal_outcome_ref": before.outcome_ref,
                "reward_contract_ref": contract.identity(),
                "candidate_checkpoint": outcome["candidate_checkpoint"],
                "check_result_refs": outcome["check_result_refs"],
                "components": components,
                "numerator": numerator,
                "normalization": contract.normalization,
                "availability": "available",
                "eligibility_ref": availability["eligibility_ref"],
            }
            or record
            != {
                "record_type": "RewardPublicationV1",
                "schema": 1,
                "availability_ref": after.outcome_ref,
            }
            or before.position["phase"] != "terminal"
            or after.position != before.position
            or event.actor != "evaluator"
            or set(effect["set"]) != {"outcome_ref", "external_inputs_ref"}
        ):
            raise ProjectionError("reward arithmetic or terminal provenance is false")
        return
    raise ProjectionError("unsupported terminal event kind")
