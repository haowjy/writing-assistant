"""Pure scripted-author policy used by derives and the scripted gatherer."""

from __future__ import annotations

import json

from writing_agent.task_graph import canonical_json
from writing_agent.task_graph_calls import validate_ask_shape
from writing_agent.task_graph_records import AuthorReplyV1


class ScriptCoverageError(ValueError):
    """An admitted author script cannot cover the requested interaction."""


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


def resolve_script_reply(script, request, decisions, disclosures):
    """Resolve an admitted script against a request without consulting prose."""
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


def scripted_author_reply(script, request, request_ref, decisions, disclosures) -> AuthorReplyV1:
    """Build the canonical typed reply for an admitted scripted-author source."""
    bound_request = {**dict(request), "request_ref": request_ref}
    if bound_request.get("source") == "mandatory_feedback":
        rule = next(
            (item for item in script.feedback if item["id"] == bound_request["feedback_id"]),
            None,
        )
        if rule is None:
            raise ScriptCoverageError("script lacks the requested feedback response")
        return AuthorReplyV1(request_ref, "answered", rule["utterance"], (), {})

    try:
        _, _, result = resolve_script_reply(
            script, bound_request, dict(decisions), dict(disclosures)
        )
    except ScriptCoverageError:
        return AuthorReplyV1(request_ref, "unsupported_coverage", None, (), {})
    selected = {
        decision: [] if proposal is None else [proposal]
        for decision, proposal in result["selected_proposals"].items()
    }
    return AuthorReplyV1(
        request_ref,
        "answered",
        result["utterance"],
        result["decision_ids"],
        selected,
    )
