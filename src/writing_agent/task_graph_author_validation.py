"""Semantic validation for durable scripted-author effects.

The projection walk invokes these rules before publication and on restore/replay.
"""

from __future__ import annotations

import json

from writing_agent.task_graph import MessageV1, canonical_json
from writing_agent.task_graph_contracts import InteractionPolicyV1, NodeContractV1
from writing_agent.task_graph_scripted import (
    ScriptCoverageError,
    frozen_prerequisite_results,
    resolve_script_reply,
    validate_ask_semantics,
)


def validate_author_request_effect(store, before, after, event, effect, entry) -> None:
    """The same deterministic request semantics run before publication and on restore."""
    from writing_agent.task_graph_projection import ProjectionError

    old_author_calls = store.get_artifact(before.budgets_ref)["consumed"].get("author_calls", 0)
    if (
        event.actor != "environment"
        or "writer" in event.audience
        or before.position["phase"] not in {"ready_writer", "checking", "awaiting_checks"}
        or after.position["phase"] != "awaiting_author"
        or set(effect["set"]) != {"position", "continuation", "budgets_ref", "external_inputs_ref"}
        or effect["history_set"]
        or effect["file_delta"]
        or before.files != after.files
        or before.context_ref != after.context_ref
        or before.author_packet_ref != after.author_packet_ref
        or before.requirements_ref != after.requirements_ref
        or before.decisions_ref != after.decisions_ref
        or before.disclosures_ref != after.disclosures_ref
        or before.outcome_ref != after.outcome_ref
    ):
        raise ProjectionError("author request exceeded environment authority")
    previous_continuation = before.to_dict()["continuation"]
    if before.continuation["author_request"] is not None or (
        after.to_dict()["continuation"]
        != {**previous_continuation, "author_request": after.continuation["author_request"]}
    ):
        raise ProjectionError("author request changed unrelated continuation")
    previous_position = before.to_dict()["position"]
    if after.to_dict()["position"] != {**previous_position, "phase": "awaiting_author"}:
        raise ProjectionError("author request changed unrelated position")
    if entry["record_ref"] != after.continuation["author_request"]:
        raise ProjectionError("author request log and continuation disagree")
    request = store.get_artifact(entry["record_ref"], private=True)
    if (
        not isinstance(request, dict)
        or set(request)
        != {
            "record_type",
            "schema",
            "request_id",
            "source",
            "action_id",
            "call_id",
            "feedback_id",
            "arguments",
            "decision_ids",
            "prerequisite_results",
            "requirement_version",
            "script_ref",
            "author_packet_ref",
        }
        or request["record_type"] != "AuthorRequestV1"
        or request["schema"] != 1
    ):
        raise ProjectionError("author request record has wrong type")
    instance = store.load_instance(before.instance_ref)
    spec = next((node for node in instance.nodes if node.id == before.position["node_id"]), None)
    if spec is None:
        raise ProjectionError("author request names unknown node")
    contract = NodeContractV1.from_dict(store.get_artifact(spec.entry_contract))
    interaction = contract.interaction_contract
    if interaction.mode != "scripted_author" or (
        request["script_ref"] != interaction.script_ref
        or request["author_packet_ref"] != interaction.author_packet_ref
        or request["author_packet_ref"] != before.author_packet_ref
        or request["requirement_version"] != before.requirements_ref
        or request["prerequisite_results"] != frozen_prerequisite_results(store, before)
    ):
        raise ProjectionError("author request differs from admitted role contract")
    policy = InteractionPolicyV1.from_dict(store.get_artifact(interaction.interaction_policy_ref))

    class _Node:
        interaction_policy = policy

    if request["source"] == "writer_request":
        request_budget = store.get_artifact(before.budgets_ref)
        if (
            request_budget["consumed"].get("tool_calls", 0)
            >= request_budget["limits"]["tool_calls"]
        ):
            raise ProjectionError("author request exceeded tool-call budget")
        try:
            validate_ask_semantics(
                request["arguments"], _Node(), store.get_artifact(before.decisions_ref)
            )
        except (TypeError, ValueError) as exc:
            raise ProjectionError("author request carries invalid ask_author arguments") from exc
        ids = request["arguments"]["decision_ids"]
        ordered = [item["id"] for item in policy.public_decisions if item["id"] in ids]
        cursor = before.continuation["next_call"]
        queue = before.continuation["tool_queue"]
        if (
            before.position["phase"] != "ready_writer"
            or len(queue) != 1
            or cursor != 0
            or cursor >= len(queue)
            or canonical_json(queue[cursor])
            != canonical_json(
                {
                    "call_id": request["call_id"],
                    "name": "ask_author",
                    "arguments": request["arguments"],
                }
            )
            or request["decision_ids"] != ordered
            or request["feedback_id"] is not None
            or request["action_id"] != before.history["action_ids"][-1]
        ):
            raise ProjectionError("author request does not bind pending writer call")
    elif request["source"] == "mandatory_feedback":
        from writing_agent.task_graph_contracts import CheckContractV1, ScriptedAuthorV1
        from writing_agent.task_graph_terminal import current_check_results

        script = ScriptedAuthorV1.from_dict(store.get_artifact(request["script_ref"], private=True))
        cursor = before.continuation["feedback_cursor"]
        if cursor >= len(script.feedback):
            raise ProjectionError("feedback cursor exceeds admitted script")
        rule = script.feedback[cursor]
        progress_checks = [
            check
            for check_ref in (*contract.mandatory_checks, *contract.optional_checks)
            if (
                check := CheckContractV1.from_dict(store.get_artifact(check_ref, private=True))
            ).applicability
            in {"each_turn", f"before_feedback:{rule['id']}"}
        ]
        batch, results = current_check_results(store, before)
        statuses = {}
        if batch is not None:
            for check_ref, (_, result) in results.items():
                check_request = store.get_artifact(check_ref, private=True)
                statuses[check_request["check_id"]] = result["status"]
        old_budget = store.get_artifact(before.budgets_ref)
        if (
            before.position["phase"] not in {"checking", "awaiting_checks"}
            or (bool(progress_checks) and before.position["phase"] != "awaiting_checks")
            or (not progress_checks and before.position["phase"] != "checking")
            or before.continuation["check_requests"]
            or request["feedback_id"] != rule["id"]
            or request["action_id"] is not None
            or request["call_id"] is not None
            or request["arguments"] is not None
            or request["decision_ids"] != []
            or any(statuses.get(check_id) != "pass" for check_id in rule["prerequisite_check_ids"])
            or old_budget["consumed"].get("writer_turns", 0) >= old_budget["limits"]["writer_turns"]
        ):
            raise ProjectionError("mandatory feedback route or prerequisites are false")
    else:
        raise ProjectionError("unsupported author request source")
    if request["request_id"] != f"{event.rollout_id}:author:{old_author_calls}":
        raise ProjectionError("author request ordinal is false")
    old_budget = store.get_artifact(before.budgets_ref)
    expected_budget = json.loads(canonical_json(old_budget))
    expected_budget["consumed"]["author_calls"] = (
        expected_budget["consumed"].get("author_calls", 0) + 1
    )
    if (
        expected_budget["consumed"]["author_calls"] > contract.budget_contract.max_author_calls
        or store.get_artifact(after.budgets_ref) != expected_budget
    ):
        raise ProjectionError("author request has false budget accounting")


def validate_author_ack_effect(store, before, after, event, effect, entry, message) -> None:
    from writing_agent.task_graph_projection import ProjectionError

    request_ref = before.continuation["author_request"]
    if request_ref is None:
        raise ProjectionError("author acknowledgement has no outstanding request")
    request = store.get_artifact(request_ref, private=True)
    record = store.get_artifact(entry["record_ref"])
    expected_id = f"{event.rollout_id}:tool_result:{len(before.history['tool_result_ids'])}"
    expected_message = MessageV1(
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
    old_budget = store.get_artifact(before.budgets_ref)
    expected_budget = json.loads(canonical_json(old_budget))
    consumed = expected_budget["consumed"]
    consumed["attempted_tool_calls"] = consumed.get("attempted_tool_calls", 0) + 1
    if consumed.get("tool_calls", 0) >= expected_budget["limits"]["tool_calls"]:
        raise ProjectionError("author acknowledgement exceeded tool-call budget")
    consumed["tool_calls"] = consumed.get("tool_calls", 0) + 1
    continuation = before.to_dict()["continuation"]
    continuation["next_call"] += 1
    if (
        event.actor != "environment"
        or "writer" not in event.audience
        or set(effect["set"]) != {"continuation", "budgets_ref", "external_inputs_ref"}
        or set(effect["history_set"]) != {"tool_result_ids"}
        or effect["file_delta"]
        or record
        != {
            "record_type": "AuthorToolAckV1",
            "schema": 1,
            "request_ref": request_ref,
            "result_id": expected_id,
            "call_id": request["call_id"],
            "action_id": request["action_id"],
        }
        or message != expected_message
        or after.to_dict()["continuation"] != continuation
        or list(after.history["tool_result_ids"])
        != [*before.history["tool_result_ids"], expected_id]
        or after.position != before.position
        or after.files != before.files
        or after.context_ref != before.context_ref
        or store.get_artifact(after.budgets_ref) != expected_budget
    ):
        raise ProjectionError("author acknowledgement has false authority or pairing")


def validate_decision_disclosure_effect(store, before, after, event, effect, entry) -> None:
    from writing_agent.task_graph_contracts import ScriptedAuthorV1
    from writing_agent.task_graph_projection import ProjectionError

    request_ref = before.continuation["author_request"]
    if request_ref is None:
        raise ProjectionError("decision disclosure has no outstanding author request")
    record = store.get_artifact(entry["record_ref"])
    request = store.get_artifact(request_ref, private=True)
    script = ScriptedAuthorV1.from_dict(store.get_artifact(request["script_ref"], private=True))
    old_decisions = store.get_artifact(before.decisions_ref)
    old_disclosures = store.get_artifact(before.disclosures_ref)
    expected_decisions, expected_disclosures, expected_reply = resolve_script_reply(
        script, {**request, "request_ref": request_ref}, old_decisions, old_disclosures
    )
    if (
        not isinstance(record, dict)
        or set(record)
        != {"record_type", "schema", "request_ref", "reply_ref", "decisions_ref", "disclosures_ref"}
        or record["record_type"] != "DecisionDisclosureV1"
        or record["schema"] != 1
        or record["request_ref"] != request_ref
        or record["decisions_ref"] != after.decisions_ref
        or record["disclosures_ref"] != after.disclosures_ref
        or store.get_artifact(record["reply_ref"]) != expected_reply
        or store.get_artifact(after.decisions_ref) != expected_decisions
        or store.get_artifact(after.disclosures_ref) != expected_disclosures
        or event.actor != "environment"
        or "writer" in event.audience
        or set(effect["set"]) != {"decisions_ref", "disclosures_ref", "external_inputs_ref"}
        or effect["history_set"]
        or effect["file_delta"]
        or after.files != before.files
        or after.context_ref != before.context_ref
        or after.continuation != before.continuation
        or after.position != before.position
        or after.budgets_ref != before.budgets_ref
    ):
        raise ProjectionError("decision disclosure differs from frozen script or authority")


def _feedback_utterance(store, request, before):
    from writing_agent.task_graph_contracts import ScriptedAuthorV1

    script = ScriptedAuthorV1.from_dict(store.get_artifact(request["script_ref"], private=True))
    cursor = before.continuation["feedback_cursor"]
    rule = script.feedback[cursor]
    if request["feedback_id"] != rule["id"]:
        raise ValueError("feedback cursor and request id disagree")
    return rule["utterance"]


def validate_requirement_update_effect(store, before, after, event, effect, entry) -> None:
    from writing_agent.task_graph_contracts import RequirementUpdateV1, ScriptedAuthorV1
    from writing_agent.task_graph_projection import ProjectionError

    request_ref = before.continuation["author_request"]
    if request_ref is None:
        raise ProjectionError("requirement update has no feedback request")
    request = store.get_artifact(request_ref, private=True)
    script = ScriptedAuthorV1.from_dict(store.get_artifact(request["script_ref"], private=True))
    cursor = before.continuation["feedback_cursor"]
    if request["source"] != "mandatory_feedback" or cursor >= len(script.feedback):
        raise ProjectionError("requirement update is not a declared feedback effect")
    rule = script.feedback[cursor]
    update_ref = rule["requirement_update_ref"]
    if update_ref is None:
        raise ProjectionError("feedback has no authorized requirement update")
    update = RequirementUpdateV1.from_dict(store.get_artifact(update_ref, private=True))
    old_ledger = store.get_artifact(before.requirements_ref)
    if (
        not isinstance(old_ledger, dict)
        or set(old_ledger) != {"record_type", "schema", "active", "superseded"}
        or old_ledger["record_type"] != "RequirementLedgerV1"
        or old_ledger["schema"] != 1
        or update.supersedes not in old_ledger["active"]
        or update.id in old_ledger["active"]
    ):
        raise ProjectionError("requirement ledger has no uniquely active predecessor")
    next_ledger = json.loads(canonical_json(old_ledger))
    old_value = next_ledger["active"].pop(update.supersedes)
    next_ledger["superseded"][update.supersedes] = old_value
    next_ledger["active"][update.id] = update.replacement
    record = store.get_artifact(entry["record_ref"])
    if (
        record
        != {
            "record_type": "RequirementSupersessionV1",
            "schema": 1,
            "request_ref": request_ref,
            "update_ref": update_ref,
            "before_ref": before.requirements_ref,
            "after_ref": after.requirements_ref,
        }
        or store.get_artifact(after.requirements_ref) != next_ledger
        or event.actor != "environment"
        or "writer" in event.audience
        or set(effect["set"]) != {"requirements_ref", "external_inputs_ref"}
        or effect["history_set"]
        or effect["file_delta"]
        or after.files != before.files
        or after.context_ref != before.context_ref
        or after.position != before.position
        or after.continuation != before.continuation
        or after.budgets_ref != before.budgets_ref
        or after.decisions_ref != before.decisions_ref
        or after.disclosures_ref != before.disclosures_ref
        or after.outcome_ref != before.outcome_ref
    ):
        raise ProjectionError("requirement supersession is not preauthorized")


def validate_coverage_failure_effect(store, before, after, event, effect, entry) -> None:
    from writing_agent.task_graph_contracts import ScriptedAuthorV1
    from writing_agent.task_graph_projection import ProjectionError

    request_ref = before.continuation["author_request"]
    if request_ref is None or before.position["phase"] != "awaiting_author":
        raise ProjectionError("coverage failure has no committed author request")
    request = store.get_artifact(request_ref, private=True)
    if request["source"] != "writer_request":
        raise ProjectionError("feedback cannot claim unsupported writer ask coverage")
    script = ScriptedAuthorV1.from_dict(store.get_artifact(request["script_ref"], private=True))
    try:
        resolve_script_reply(
            script,
            {**request, "request_ref": request_ref},
            store.get_artifact(before.decisions_ref),
            store.get_artifact(before.disclosures_ref),
        )
    except ScriptCoverageError:
        pass
    else:
        raise ProjectionError("covered request cannot be infrastructure-invalid")
    expected_outcome = {
        "record_type": "InfrastructureInvalidV1",
        "schema": 1,
        "task_status": "unknown",
        "execution_status": "simulator_error",
        "stop_reason": "unsupported_script_coverage",
        "reward_status": "unavailable",
        "training_eligibility": "ineligible",
        "request_ref": request_ref,
    }
    continuation = before.to_dict()["continuation"]
    continuation["author_request"] = None
    position = before.to_dict()["position"]
    position["phase"] = "terminal"
    record = store.get_artifact(entry["record_ref"])
    if (
        record
        != {
            "record_type": "ScriptCoverageFailureV1",
            "schema": 1,
            "request_ref": request_ref,
            "outcome_ref": after.outcome_ref,
        }
        or store.get_artifact(after.outcome_ref) != expected_outcome
        or event.actor != "environment"
        or "writer" in event.audience
        or set(effect["set"]) != {"position", "continuation", "outcome_ref", "external_inputs_ref"}
        or effect["history_set"]
        or effect["file_delta"]
        or after.to_dict()["continuation"] != continuation
        or after.to_dict()["position"] != position
        or after.files != before.files
        or after.context_ref != before.context_ref
        or after.budgets_ref != before.budgets_ref
        or after.requirements_ref != before.requirements_ref
        or after.decisions_ref != before.decisions_ref
        or after.disclosures_ref != before.disclosures_ref
    ):
        raise ProjectionError("unsupported script coverage record has false authority")


def validate_author_turn_effect(store, before, after, event, effect, entry, message) -> None:
    from writing_agent.task_graph_projection import ProjectionError

    request_ref = before.continuation["author_request"]
    if request_ref is None:
        raise ProjectionError("author turn has no outstanding request")
    request = store.get_artifact(request_ref, private=True)
    record = store.get_artifact(entry["record_ref"])
    reply = store.get_artifact(record["reply_ref"])
    expected_message = MessageV1(
        role="user",
        origin=request["request_id"],
        content=({"type": "text", "text": reply["utterance"]},),
    )
    continuation = before.to_dict()["continuation"]
    continuation["author_request"] = None
    if request["source"] == "mandatory_feedback":
        continuation["feedback_cursor"] += 1
        from writing_agent.task_graph_contracts import ScriptedAuthorV1

        script = ScriptedAuthorV1.from_dict(store.get_artifact(request["script_ref"], private=True))
        update_expected = (
            script.feedback[before.continuation["feedback_cursor"]]["requirement_update_ref"]
            is not None
        )
        if (before.requirements_ref != request["requirement_version"]) is not update_expected:
            raise ProjectionError("feedback requirement update was skipped or invented")
    else:
        log = store.get_artifact(before.external_inputs_ref)
        disclosures = [
            store.get_artifact(item["record_ref"])
            for item in log["entries"]
            if item["kind"] == "decision_disclosed"
        ]
        if not disclosures or disclosures[-1]["reply_ref"] != record["reply_ref"]:
            raise ProjectionError("author turn is not paired with disclosed decision")
    position = before.to_dict()["position"]
    position["phase"] = "ready_writer"
    if (
        event.actor != "author"
        or "writer" not in event.audience
        or set(effect["set"]) != {"position", "continuation", "external_inputs_ref"}
        or effect["history_set"]
        or effect["file_delta"]
        or record
        != {
            "record_type": "AuthorTurnV1",
            "schema": 1,
            "request_ref": request_ref,
            "reply_ref": record["reply_ref"],
        }
        or reply.get("request_ref") != request_ref
        or reply.get("request_id") != request["request_id"]
        or reply.get("record_type") != "ScriptedAuthorReplyV1"
        or reply.get("schema") != 1
        or (
            request["source"] == "mandatory_feedback"
            and (
                reply.get("decision_ids") != []
                or reply.get("selected_proposals") != {}
                or reply.get("utterance") != _feedback_utterance(store, request, before)
            )
        )
        or message != expected_message
        or after.to_dict()["continuation"] != continuation
        or after.to_dict()["position"] != position
        or after.files != before.files
        or after.context_ref != before.context_ref
        or after.budgets_ref != before.budgets_ref
        or after.decisions_ref != before.decisions_ref
        or after.disclosures_ref != before.disclosures_ref
        or after.requirements_ref != before.requirements_ref
        or after.outcome_ref != before.outcome_ref
    ):
        raise ProjectionError("author text attempted to change environment state")
