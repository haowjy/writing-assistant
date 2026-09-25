"""Deterministic scripted-author producer for opted-in graph nodes.

All choices come from admitted, immutable contracts. This module does not call an
author model, interpret prose, run checks, or decide completion.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from writing_agent.task_graph import MessageV1, canonical_json
from writing_agent.task_graph_accounting import charge_tool_attempt
from writing_agent.task_graph_environment import WriterRuntimeError


def validate_ask_shape(arguments: Mapping[str, Any]) -> None:
    """Reject malformed structured control syntax before any author operation."""
    if not isinstance(arguments, Mapping) or set(arguments) != {
        "question",
        "decision_ids",
        "proposals",
        "option_refs",
    }:
        raise ValueError("ask_author needs exact structured arguments")
    if not isinstance(arguments["question"], str) or not arguments["question"].strip():
        raise ValueError("ask_author question must be nonempty text")
    ids = arguments["decision_ids"]
    if (
        not isinstance(ids, (list, tuple))
        or not ids
        or any(not isinstance(x, str) or not x for x in ids)
    ):
        raise ValueError("ask_author decision_ids must be nonempty text IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("ask_author repeats a decision ID")
    proposals = arguments["proposals"]
    if not isinstance(proposals, (list, tuple)) or any(
        not isinstance(item, Mapping)
        or set(item) != {"id", "text"}
        or any(not isinstance(value, str) or not value for value in item.values())
        for item in proposals
    ):
        raise ValueError("ask_author proposals need exact id/text pairs")
    proposal_ids = [item["id"] for item in proposals]
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("ask_author repeats a proposal ID")
    refs = arguments["option_refs"]
    if not isinstance(refs, (list, tuple)) or any(
        not isinstance(ref, str) or not ref for ref in refs
    ):
        raise ValueError("ask_author option_refs must be text IDs")
    if len(refs) != len(set(refs)):
        raise ValueError("ask_author repeats an option reference")
    canonical_json(arguments)


def validate_ask_semantics(arguments, node, decisions) -> None:
    validate_ask_shape(arguments)
    policy = node.interaction_policy
    if policy is None:
        raise ValueError("ask_author is unavailable")
    if len(arguments["question"]) > policy.max_question_chars:
        raise ValueError("ask_author question exceeds limit")
    if len(arguments["decision_ids"]) > policy.max_decisions_per_request:
        raise ValueError("ask_author has too many decision IDs")
    declared = {item["id"] for item in policy.public_decisions}
    if set(arguments["decision_ids"]) - declared:
        raise ValueError("ask_author names undeclared decision ID")
    if len(arguments["proposals"]) > policy.max_proposals or any(
        len(item["text"]) > policy.max_proposal_chars or len(item["id"]) > 256
        for item in arguments["proposals"]
    ):
        raise ValueError("ask_author proposals exceed limits")
    if len(arguments["option_refs"]) > policy.max_proposals or any(
        len(ref) > 256 for ref in arguments["option_refs"]
    ):
        raise ValueError("ask_author option references exceed limits")
    prior = decisions.get("proposals", {})
    available = {item["id"] for item in arguments["proposals"]} | set(prior)
    if set(arguments["option_refs"]) - available:
        raise ValueError("ask_author references an undeclared proposal")
    if {item["id"] for item in arguments["proposals"]} & set(prior):
        raise ValueError("ask_author redefines a prior proposal")


def frozen_prerequisite_results(store, state):
    from writing_agent.task_graph_terminal import current_check_results

    _, results = current_check_results(store, state)
    frozen = {}
    for request_ref, (result_ref, result) in results.items():
        request = store.get_artifact(request_ref, private=True)
        frozen[request["check_id"]] = {"result_ref": result_ref, "status": result["status"]}
    return frozen


def resolve_script_reply(script, request, decisions, disclosures):
    """Pure exact routing. A missing selector is infrastructure coverage failure."""
    values = json.loads(canonical_json(decisions))
    ledger = json.loads(canonical_json(disclosures))
    for proposal in request["arguments"]["proposals"]:
        values["proposals"][proposal["id"]] = {
            "text": proposal["text"],
            "action_id": request["action_id"],
        }
    utterances = []
    selected = {}
    for public_id in request["decision_ids"]:
        rule = script.answers[public_id]
        prior = values["values"].get(public_id)
        if prior is not None:
            utterances.append(prior["utterance"])
            selected[public_id] = prior["selected_proposal_id"]
            continue
        if any(
            request["prerequisite_results"].get(check_id, {}).get("status") != "pass"
            for check_id in rule["prerequisite_check_ids"]
        ):
            raise ScriptCoverageError("script answer prerequisites were not satisfied")
        options = request["arguments"]["option_refs"]
        selected_id = None
        if rule["mode"] == "declared_option_id":
            if rule["selector"] not in options:
                raise ScriptCoverageError("script selector does not match declared options")
            selected_id = rule["selector"]
        elif rule["mode"] == "declared_option_position":
            if rule["selector"] >= len(options):
                raise ScriptCoverageError("script option selector is exhausted")
            selected_id = options[rule["selector"]]
        utterances.append(rule["utterance"])
        selected[public_id] = selected_id
        values["values"][public_id] = {
            "value": rule["value"],
            "utterance": rule["utterance"],
            "selected_proposal_id": selected_id,
            "request_id": request["request_id"],
        }
        ledger["decisions"].append(public_id)
    utterance = "\n".join(utterances)
    if not utterance:
        raise ScriptCoverageError("script has no answer for declared request")
    reply = {
        "record_type": "ScriptedAuthorReplyV1",
        "schema": 1,
        "request_ref": request["request_ref"],
        "request_id": request["request_id"],
        "utterance": utterance,
        "decision_ids": request["decision_ids"],
        "selected_proposals": selected,
    }
    return values, ledger, reply


class ScriptedAuthorRuntimeV1:
    def __init__(self, writer):
        self.writer = writer
        self.store = writer.store

    def _node(self, runtime):
        node, budget = self.writer.validate_runtime(runtime)
        if node.contract.interaction_contract.mode != "scripted_author":
            raise WriterRuntimeError("not an admitted scripted-author node")
        if runtime.state.author_packet_ref != node.contract.interaction_contract.author_packet_ref:
            raise WriterRuntimeError("runtime author packet differs from admitted author packet")
        return node, budget

    def request(self, runtime, call, action):
        """Commit the request after its writer action, before resolving a reply."""
        node, budget = self._node(runtime)
        state = runtime.state
        if state.position["phase"] != "ready_writer" or state.continuation["author_request"]:
            raise WriterRuntimeError("author request requires a ready, unanswered control call")
        if (
            budget["consumed"].get("author_calls", 0)
            >= node.contract.budget_contract.max_author_calls
        ):
            raise WriterRuntimeError("author-call budget exhausted")
        if budget["consumed"].get("tool_calls", 0) >= budget["limits"]["tool_calls"]:
            raise WriterRuntimeError("tool-call budget exhausted")
        decisions = self.store.get_artifact(state.decisions_ref, expected_domain="payload")
        validate_ask_semantics(call["arguments"], node, decisions)
        request_id = f"{self.writer.rollout_id}:author:{budget['consumed'].get('author_calls', 0)}"
        ids = call["arguments"]["decision_ids"]
        ordered = [
            item["id"] for item in node.interaction_policy.public_decisions if item["id"] in ids
        ]
        request = {
            "record_type": "AuthorRequestV1",
            "schema": 1,
            "request_id": request_id,
            "source": "writer_request",
            "action_id": action["action_id"],
            "call_id": call["call_id"],
            "feedback_id": None,
            "arguments": call["arguments"],
            "decision_ids": ordered,
            "prerequisite_results": frozen_prerequisite_results(self.store, state),
            "requirement_version": state.requirements_ref,
            "script_ref": node.contract.interaction_contract.script_ref,
            "author_packet_ref": state.author_packet_ref,
        }
        request_ref = self.store.put_artifact(request, private=True)
        next_budget = json.loads(canonical_json(budget))
        next_budget["consumed"]["author_calls"] = next_budget["consumed"].get("author_calls", 0) + 1
        budget_ref = self.store.put_artifact(next_budget)
        position = state.to_dict()["position"]
        position["phase"] = "awaiting_author"
        continuation = state.to_dict()["continuation"]
        continuation["author_request"] = request_ref
        return self.writer.environment.publish_record(
            runtime,
            "external_requested",
            "environment",
            record_ref=request_ref,
            changes={"position": position, "continuation": continuation, "budgets_ref": budget_ref},
            audience=("controller", "trainer"),
            extra_refs=(budget_ref,),
            restore_prefix="scripted",
            result_effect=True,
        )

    def reply(self, runtime):
        """Resolve one committed request without any live provider or model call."""
        node, budget = self._node(runtime)
        state = runtime.state
        if state.position["phase"] != "awaiting_author" or not state.continuation["author_request"]:
            raise WriterRuntimeError("no committed author request to reply to")
        request_ref = state.continuation["author_request"]
        request = self.store.get_artifact(request_ref, private=True)
        if request["source"] == "mandatory_feedback":
            return self._feedback_reply(runtime, request_ref, request)
        script = node.script
        decisions = self.store.get_artifact(state.decisions_ref)
        disclosures = self.store.get_artifact(state.disclosures_ref)
        if (
            not isinstance(decisions, dict)
            or set(decisions) != {"record_type", "schema", "values", "proposals"}
            or decisions["record_type"] != "DecisionLedgerV1"
            or decisions["schema"] != 1
            or not isinstance(disclosures, dict)
            or set(disclosures) != {"record_type", "schema", "decisions"}
            or disclosures["record_type"] != "DisclosureLedgerV1"
            or disclosures["schema"] != 1
        ):
            raise WriterRuntimeError("scripted-author ledgers are not typed")
        try:
            values, ledger, reply = resolve_script_reply(
                script, {**request, "request_ref": request_ref}, decisions, disclosures
            )
        except ScriptCoverageError:
            return self._coverage_failure(runtime, request_ref)
        utterance = reply["utterance"]
        decisions_ref = self.store.put_artifact(values)
        disclosures_ref = self.store.put_artifact(ledger)
        reply_ref = self.store.put_artifact(reply)
        ack_id = f"{self.writer.rollout_id}:tool_result:{len(state.history['tool_result_ids'])}"
        ack_message = MessageV1(
            role="tool",
            call_id=request["call_id"],
            origin=request["action_id"],
            content=(
                {
                    "type": "tool_result",
                    "call_id": request["call_id"],
                    "content": {
                        "ok": True,
                        "valid": True,
                        "result": {"status": "author_reply_follows"},
                    },
                },
            ),
        )
        author_message = MessageV1(
            role="user",
            origin=request["request_id"],
            content=({"type": "text", "text": utterance},),
        )
        batch = self.writer.environment.batch(runtime, restore_prefix="scripted")
        next_budget, _ = charge_tool_attempt(budget)
        budget_ref = self.store.put_artifact(next_budget)
        continuation = batch.current.to_dict()["continuation"]
        continuation["next_call"] += 1
        ack_record = {
            "record_type": "AuthorToolAckV1",
            "schema": 1,
            "request_ref": request_ref,
            "result_id": ack_id,
            "call_id": request["call_id"],
            "action_id": request["action_id"],
        }
        ack_event, _ = batch.append_record(
            "tool_result",
            "environment",
            record=ack_record,
            message=ack_message,
            changes={"continuation": continuation, "budgets_ref": budget_ref},
            history={"tool_result_ids": [*state.history["tool_result_ids"], ack_id]},
        )
        disclosure_record = {
            "record_type": "DecisionDisclosureV1",
            "schema": 1,
            "request_ref": request_ref,
            "reply_ref": reply_ref,
            "decisions_ref": decisions_ref,
            "disclosures_ref": disclosures_ref,
        }
        batch.append_record(
            "decision_disclosed",
            "environment",
            record=disclosure_record,
            changes={"decisions_ref": decisions_ref, "disclosures_ref": disclosures_ref},
        )
        position = batch.current.to_dict()["position"]
        position["phase"] = "ready_writer"
        continuation = batch.current.to_dict()["continuation"]
        continuation["author_request"] = None
        turn_record = {
            "record_type": "AuthorTurnV1",
            "schema": 1,
            "request_ref": request_ref,
            "reply_ref": reply_ref,
        }
        turn_event, _ = batch.append_record(
            "author_turn",
            "author",
            record=turn_record,
            message=author_message,
            changes={"position": position, "continuation": continuation},
        )
        batch.append_visible((ack_message, author_message), turn_event)
        return batch.publish(
            ack_event,
            ack_event.payload_ref,
            extra_refs=(reply_ref, decisions_ref, disclosures_ref, budget_ref),
        )

    def _coverage_failure(self, runtime, request_ref):
        state = runtime.state
        outcome = {
            "record_type": "InfrastructureInvalidV1",
            "schema": 1,
            "task_status": "unknown",
            "execution_status": "simulator_error",
            "stop_reason": "unsupported_script_coverage",
            "reward_status": "unavailable",
            "training_eligibility": "ineligible",
            "request_ref": request_ref,
        }
        outcome_ref = self.store.put_artifact(outcome)
        record = {
            "record_type": "ScriptCoverageFailureV1",
            "schema": 1,
            "request_ref": request_ref,
            "outcome_ref": outcome_ref,
        }
        position = state.to_dict()["position"]
        position["phase"] = "terminal"
        continuation = state.to_dict()["continuation"]
        continuation["author_request"] = None
        return self.writer.environment.publish_record(
            runtime,
            "termination_recorded",
            "environment",
            record=record,
            changes={
                "position": position,
                "continuation": continuation,
                "outcome_ref": outcome_ref,
            },
            audience=("controller", "trainer"),
            extra_refs=(outcome_ref,),
            restore_prefix="scripted",
            result_effect=True,
        )

    def request_feedback(self, runtime):
        """Issue one frozen feedback item after its declared progress prerequisites."""
        from writing_agent.task_graph_checks import applicable_checks
        from writing_agent.task_graph_terminal import current_check_results

        node, budget = self._node(runtime)
        state = runtime.state
        cursor = state.continuation["feedback_cursor"]
        rules = node.script.feedback
        if (
            cursor >= len(rules)
            or state.position["phase"] not in {"checking", "awaiting_checks"}
            or state.continuation["check_requests"]
        ):
            raise WriterRuntimeError("no ready mandatory feedback item")
        rule = rules[cursor]
        progress_checks = applicable_checks(node, cursor)
        if (progress_checks and state.position["phase"] != "awaiting_checks") or (
            not progress_checks and state.position["phase"] != "checking"
        ):
            raise WriterRuntimeError("feedback cannot skip or repeat progress checks")
        batch, results = current_check_results(self.store, state)
        statuses = {}
        if batch is not None:
            for request_ref, (_, result) in results.items():
                request = self.store.get_artifact(request_ref, private=True)
                statuses[request["check_id"]] = result["status"]
        if any(statuses.get(check_id) != "pass" for check_id in rule["prerequisite_check_ids"]):
            raise WriterRuntimeError("mandatory feedback prerequisites did not pass")
        if budget["consumed"].get("author_calls", 0) >= budget["limits"]["author_calls"]:
            raise WriterRuntimeError("author budget cannot deliver mandatory feedback")
        if budget["consumed"].get("writer_turns", 0) >= budget["limits"]["writer_turns"]:
            raise WriterRuntimeError("writer budget cannot continue after mandatory feedback")
        request_id = f"{self.writer.rollout_id}:author:{budget['consumed'].get('author_calls', 0)}"
        request = {
            "record_type": "AuthorRequestV1",
            "schema": 1,
            "request_id": request_id,
            "source": "mandatory_feedback",
            "action_id": None,
            "call_id": None,
            "feedback_id": rule["id"],
            "arguments": None,
            "decision_ids": [],
            "prerequisite_results": frozen_prerequisite_results(self.store, state),
            "requirement_version": state.requirements_ref,
            "script_ref": node.contract.interaction_contract.script_ref,
            "author_packet_ref": state.author_packet_ref,
        }
        request_ref = self.store.put_artifact(request, private=True)
        next_budget = json.loads(canonical_json(budget))
        next_budget["consumed"]["author_calls"] = next_budget["consumed"].get("author_calls", 0) + 1
        budget_ref = self.store.put_artifact(next_budget)
        position = state.to_dict()["position"]
        position["phase"] = "awaiting_author"
        continuation = state.to_dict()["continuation"]
        continuation["author_request"] = request_ref
        return self.writer.environment.publish_record(
            runtime,
            "external_requested",
            "environment",
            record_ref=request_ref,
            changes={"position": position, "continuation": continuation, "budgets_ref": budget_ref},
            audience=("controller", "trainer"),
            extra_refs=(budget_ref,),
            restore_prefix="scripted",
            result_effect=True,
        )

    def _feedback_reply(self, runtime, request_ref, request):
        node, _ = self._node(runtime)
        state = runtime.state
        cursor = state.continuation["feedback_cursor"]
        rule = node.script.feedback[cursor]
        if request["feedback_id"] != rule["id"]:
            raise WriterRuntimeError("feedback request cursor differs from admitted script")
        reply = {
            "record_type": "ScriptedAuthorReplyV1",
            "schema": 1,
            "request_ref": request_ref,
            "request_id": request["request_id"],
            "utterance": rule["utterance"],
            "decision_ids": [],
            "selected_proposals": {},
        }
        reply_ref = self.store.put_artifact(reply)
        author_message = MessageV1(
            role="user",
            origin=request["request_id"],
            content=({"type": "text", "text": rule["utterance"]},),
        )
        batch = self.writer.environment.batch(runtime, restore_prefix="scripted")

        if rule["requirement_update_ref"] is not None:
            from writing_agent.task_graph_contracts import RequirementUpdateV1

            update = RequirementUpdateV1.from_dict(
                self.store.get_artifact(rule["requirement_update_ref"], private=True)
            )
            ledger = self.store.get_artifact(batch.current.requirements_ref)
            if ledger.get("record_type") != "RequirementLedgerV1" or (
                update.supersedes not in ledger["active"]
            ):
                raise WriterRuntimeError("preauthorized requirement has no active predecessor")
            next_ledger = json.loads(canonical_json(ledger))
            old = next_ledger["active"].pop(update.supersedes)
            next_ledger["superseded"][update.supersedes] = old
            next_ledger["active"][update.id] = update.replacement
            requirement_ref = self.store.put_artifact(next_ledger)
            batch.append_record(
                "requirements_changed",
                "environment",
                record={
                    "record_type": "RequirementSupersessionV1",
                    "schema": 1,
                    "request_ref": request_ref,
                    "update_ref": rule["requirement_update_ref"],
                    "before_ref": batch.current.requirements_ref,
                    "after_ref": requirement_ref,
                },
                changes={"requirements_ref": requirement_ref},
            )
        position = batch.current.to_dict()["position"]
        position["phase"] = "ready_writer"
        continuation = batch.current.to_dict()["continuation"]
        continuation["author_request"] = None
        continuation["feedback_cursor"] += 1
        turn_event, _ = batch.append_record(
            "author_turn",
            "author",
            record={
                "record_type": "AuthorTurnV1",
                "schema": 1,
                "request_ref": request_ref,
                "reply_ref": reply_ref,
            },
            message=author_message,
            changes={"position": position, "continuation": continuation},
        )
        batch.append_visible((author_message,), turn_event)
        return batch.publish(turn_event, turn_event.payload_ref, extra_refs=(reply_ref,))


class ScriptCoverageError(RuntimeError):
    """Admitted script cannot cover an actual declared request; not a writer failure."""
