"""Closed semantic replay of an admitted task-graph writer lineage.

Typed cursor and event handlers validate authority before publication and on
restore. Handlers emit only authorized writer-visible contributions; the private
event log is never rendered wholesale.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from writing_agent.task_graph import (
    CheckpointV1,
    ContextRevisionV1,
    EnvironmentStateV1,
    EventV1,
    MessageV1,
    canonical_bytes,
    canonical_json,
    context_content_hash,
    domain_hash,
)
from writing_agent.task_graph_accounting import (
    charge_context_append,
    exhausted_stop_reason,
    observation_read_tokens,
    sampled_usage_charge,
    tool_error,
    tool_result_charge,
)
from writing_agent.task_graph_sampling import (
    _ACTION_RECORD_FIELDS,
    _STOP_RECORD_FIELDS,
    ProjectionError,
    _validate_prepared_request,
    validate_action_trace,
)
from writing_agent.task_graph_store import TaskGraphStore

_WRITER_EFFECT_FIELDS = {
    "writer_action": (
        {"continuation", "position", "budgets_ref", "external_inputs_ref"},
        {"action_ids"},
    ),
    "tool_result": (
        {"continuation", "position", "budgets_ref", "external_inputs_ref"},
        {"tool_result_ids"},
    ),
    "budget_charged": ({"position", "budgets_ref", "outcome_ref", "external_inputs_ref"}, set()),
    "termination_recorded": ({"position", "outcome_ref", "external_inputs_ref"}, set()),
}
_RESULT_RECORD_FIELDS = {
    "record_type",
    "result_id",
    "call_id",
    "action_id",
    "observation",
    "file_delta",
    "before_execution_hash",
    "after_execution_hash",
    "budget_charge",
    "loss_eligibility",
}


def validate_writer_effect(before, after, event, effect, *, store=None) -> None:
    """Enforce Phase 4 field ownership and phase boundaries.

    The generic Phase 2 reducer intentionally does not know event authority. This
    check is one part of the shared history-aware projection contract.
    """
    allowed_set, allowed_history = _WRITER_EFFECT_FIELDS[event.kind]
    if set(effect["set"]) != allowed_set or set(effect["history_set"]) != allowed_history:
        raise ProjectionError("writer event changes fields outside its authority")
    old = before.to_dict()
    new = after.to_dict()
    for key in old:
        if key in {"history", "position", "continuation", "files", "tree_hash"}:
            continue
        if key not in allowed_set and old[key] != new[key]:
            raise ProjectionError(f"writer event changed unrelated {key}")
    for key in old["history"]:
        if key in {"head", "seq", *allowed_history}:
            continue
        if old["history"][key] != new["history"][key]:
            raise ProjectionError(f"writer event changed unrelated history {key}")
    for key in old["position"]:
        if key != "phase" and old["position"][key] != new["position"][key]:
            raise ProjectionError(f"writer event changed unrelated position {key}")
    for key in old["continuation"]:
        if (
            key not in {"tool_queue", "next_call"}
            and old["continuation"][key] != new["continuation"][key]
        ):
            raise ProjectionError(f"writer event changed unrelated continuation {key}")
    if event.kind != "tool_result" and (old["files"] != new["files"] or effect["file_delta"]):
        raise ProjectionError("non-tool writer event changed files")
    if event.kind == "tool_result" and old["files"] == new["files"] and effect["file_delta"]:
        raise ProjectionError("tool result has false file delta")
    if (
        event.kind == "tool_result"
        and old["continuation"]["tool_queue"] != new["continuation"]["tool_queue"]
    ):
        raise ProjectionError("tool result replaced its queue")
    if (
        event.kind in {"budget_charged", "termination_recorded"}
        and old["continuation"] != new["continuation"]
    ):
        raise ProjectionError("writer stop changed continuation")
    if old["position"]["phase"] != "ready_writer":
        raise ProjectionError("writer event has illegal pre-phase")
    if store is not None:
        prior_outcome = store.get_artifact(before.outcome_ref, expected_domain="payload")
        if isinstance(prior_outcome, dict) and (
            prior_outcome.get("task_status") not in {None, "unknown", "incomplete"}
            or prior_outcome.get("execution_status") not in {None, "running", "valid"}
            or prior_outcome.get("reward_status") not in {None, "pending"}
            or prior_outcome.get("training_eligibility") not in {None, "pending"}
            or prior_outcome.get("stop_reason") is not None
        ):
            raise ProjectionError("writer event has illegal pre-status")
    previous_outcome = before.outcome_ref
    if event.kind in {"writer_action", "tool_result"} and after.outcome_ref != previous_outcome:
        raise ProjectionError("writer action or result changed outcome status")
    if event.kind == "writer_action":
        expected = "ready_writer" if new["continuation"]["tool_queue"] else "checking"
        if (
            old["continuation"]["next_call"] != len(old["continuation"]["tool_queue"])
            or new["position"]["phase"] != expected
        ):
            raise ProjectionError("writer action has illegal phase or pending queue")
    elif event.kind == "tool_result":
        if new["position"]["phase"] != "ready_writer" or old["continuation"]["next_call"] >= len(
            old["continuation"]["tool_queue"]
        ):
            raise ProjectionError("tool result has illegal phase or empty queue")
    else:
        if new["position"]["phase"] != "terminal" or old["continuation"]["next_call"] != len(
            old["continuation"]["tool_queue"]
        ):
            raise ProjectionError("writer stop has illegal phase or pending queue")


def _writer_stop_outcome(before, candidate_checkpoint, reason, *, phase5):
    outcome = {
        "schema": 1,
        "task_status": "incomplete",
        "execution_status": "valid",
        "stop_reason": reason,
        "reward_status": "pending",
        "training_eligibility": "pending",
    }
    if phase5:
        outcome.update(
            record_type="TerminalOutcomeV1",
            candidate_checkpoint=candidate_checkpoint,
            check_result_refs=[],
            transition_edge_id=None,
            requirement_version=before.requirements_ref,
        )
    return outcome


def validate_result_production(
    store, before, after, old_context, new_context, record, message, effect, action
) -> None:
    """Reject contradictory result metadata before its authority is published."""
    cursor = before.continuation["next_call"]
    call = before.continuation["tool_queue"][cursor]
    if (
        record.get("record_type") != "WriterToolResultV1"
        or set(record) != _RESULT_RECORD_FIELDS
        or record["loss_eligibility"] != {"tool_observation": False}
        or record["action_id"] != action["action_id"]
        or message.origin != action["action_id"]
        or record["call_id"] != call["call_id"]
        or message.call_id != call["call_id"]
        or record["result_id"] != after.history["tool_result_ids"][-1]
        or after.continuation["next_call"] != cursor + 1
    ):
        raise ProjectionError("tool result has false action, call, or cursor")
    delta = {
        path: {"before": before.files.get(path), "after": after.files.get(path)}
        for path in sorted(set(before.files) | set(after.files))
        if before.files.get(path) != after.files.get(path)
    }
    if record["file_delta"] != delta or effect["file_delta"] != delta:
        raise ProjectionError("tool result has false file delta")
    old_budget = store.get_artifact(before.budgets_ref, expected_domain="payload")
    new_budget = store.get_artifact(after.budgets_ref, expected_domain="payload")
    expected_error = tool_error(old_budget, action["validation_error"], call["name"])
    if call["name"] == "ask_author" and expected_error is None:
        raise ProjectionError("eligible ask_author was incorrectly drained as an error")
    if expected_error is not None and record["observation"] != expected_error:
        raise ProjectionError("tool error contradicts budget or syntax precedence")
    try:
        read_charge = observation_read_tokens(
            record["observation"], call["name"], old_budget["read_tokenizer"]
        )
    except ValueError as exc:
        raise ProjectionError(str(exc)) from exc
    expected_budget, expected_charge = tool_result_charge(
        old_budget, before.files, after.files, read_charge
    )
    if record["budget_charge"] != expected_charge or new_budget != expected_budget:
        raise ProjectionError("tool result has false budget charge")
    if (
        record["before_execution_hash"] != execution_value(store, before, old_context, old_budget)
        or record["after_execution_hash"] != execution_value(store, after, new_context, new_budget)
        or message.role != "tool"
        or message.loss_eligible
        or len(message.content) != 1
        or canonical_json(message.content[0])
        != canonical_json(
            {"type": "tool_result", "call_id": call["call_id"], "content": record["observation"]}
        )
    ):
        raise ProjectionError("tool result has false message or execution fingerprint")


def _value_only(value):
    if isinstance(value, Mapping):
        return {
            key: _value_only(item)
            for key, item in value.items()
            if not (key.endswith("_ref") or key.endswith("_refs") or "provenance" in key)
        }
    if isinstance(value, (tuple, list)):
        return [_value_only(item) for item in value]
    if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return "<identity>"
    return value


def execution_value(store, state, context, budget) -> str:
    return domain_hash(
        "state",
        {
            "files": state.files,
            "position": {
                key: state.position[key]
                for key in ("node_id", "visit_id", "phase", "loop_counts", "lineage_id")
            },
            "messages": [message.to_dict() for message in context.messages],
            "tool_queue": state.continuation["tool_queue"],
            "next_call": state.continuation["next_call"],
            "action_ids": state.history["action_ids"],
            "tool_result_ids": state.history["tool_result_ids"],
            "budgets": budget,
            "requirements": _value_only(store.get_artifact(state.requirements_ref)),
            "decisions": _value_only(store.get_artifact(state.decisions_ref)),
            "disclosures": _value_only(store.get_artifact(state.disclosures_ref)),
            "outcome": _value_only(store.get_artifact(state.outcome_ref)),
            "rng": _value_only(store.get_artifact(state.rng_ref)),
        },
    )


def _is_writer_stop(store: TaskGraphStore, event: EventV1, state: EnvironmentStateV1) -> bool:
    if event.kind not in {"budget_charged", "termination_recorded"}:
        return False
    if event.actor == "writer_runtime":
        return True
    # An indexed Phase 4 stop cannot evade its actor contract by changing only
    # the actor. Phase 5 uses the same log but has its own typed terminal producer.
    if state.author_packet_ref is not None:
        return False
    log = store.get_artifact(state.external_inputs_ref, expected_domain="payload")
    return (
        isinstance(log, dict)
        and log.get("record_type") == "WriterRuntimeLogV1"
        and log.get("rollout_id") == event.rollout_id
        and isinstance(log.get("entries"), list)
        and bool(log["entries"])
        and isinstance(log["entries"][-1], dict)
        and type(log["entries"][-1].get("seq")) is int
        and log["entries"][-1].get("seq") == event.seq
        and log["entries"][-1].get("kind") == event.kind
    )


def _writer_log_entry(store, state, event, seen_entries, *, has_message):
    log = store.get_artifact(state.external_inputs_ref, expected_domain="payload")
    keys = (
        {"seq", "kind", "record_ref", "message_ref"}
        if has_message
        else {"seq", "kind", "record_ref"}
    )
    if (
        not isinstance(log, dict)
        or set(log) != {"record_type", "rollout_id", "entries"}
        or log["record_type"] != "WriterRuntimeLogV1"
        or log["rollout_id"] != event.rollout_id
        or not isinstance(log["entries"], list)
        or len(log["entries"]) != len(seen_entries) + 1
        # Python equality treats True as 1; the retained witness must be byte-identical.
        or canonical_bytes(log["entries"][:-1]) != canonical_bytes(seen_entries)
        or not isinstance(log["entries"][-1], dict)
        or set(log["entries"][-1]) != keys
        or type(log["entries"][-1]["seq"]) is not int
        or log["entries"][-1]["seq"] < 0
        or log["entries"][-1]["seq"] != event.seq
        or log["entries"][-1]["kind"] != event.kind
    ):
        raise ProjectionError("writer event lacks its exact runtime-log entry")
    return log["entries"][-1]


class AuthorStage(StrEnum):
    REQUEST = "request"
    ACK = "ack"
    DISCLOSURE = "disclosure"
    UPDATE = "update"
    TURN = "turn"


@dataclass(frozen=True)
class VisibleContribution:
    """Only handler-authorized messages and their causal sources reach projection."""

    messages: tuple[MessageV1, ...]
    sources: tuple[str | None, ...]
    last_source: str | None
    replace: bool = False


@dataclass
class ReplayCursor:
    store: TaskGraphStore
    base: CheckpointV1
    baseline: ContextRevisionV1
    base_checkpoint_id: str
    checkpoint_ancestry: set[str]
    phase5: bool
    messages: list[MessageV1]
    message_sources: list[str | None]
    pending: list[str]
    seen_calls: set[str]
    action_ids: list[str]
    result_ids: list[str]
    last_source: str | None
    latest_context_ref: str
    state: EnvironmentStateV1
    call_sources: dict[str, tuple[str, dict, dict]]
    seen_entries: list[dict]
    reply_stage: AuthorStage | None = None


class SemanticReplayEngine(ReplayCursor):
    """Closed semantic dispatcher for an admitted writer lineage."""

    HANDLERS = MappingProxyType(
        {
            "rollout_started": "_rollout_started",
            "external_requested": "_external_requested",
            "check_recorded": "_check_recorded",
            "transition_committed": "_terminal_event",
            "termination_recorded": "_terminal_event",
            "external_response": "_terminal_event",
            "tool_result": "_visible_writer",
            "decision_disclosed": "_decision_disclosed",
            "requirements_changed": "_requirements_changed",
            "author_turn": "_author_turn",
            "context_changed": "_context_changed",
            "budget_charged": "_sampled_stop",
            "writer_action": "_visible_writer",
        }
    )

    def step(self, event: EventV1) -> None:
        store = self.store
        effect = store.get_artifact(event.payload_ref, expected_domain="payload")
        before = self.state
        state = store._apply_recorded_effect_body(before, event, effect)
        self.state = state
        if self.phase5 and state.author_packet_ref != self.base.state.author_packet_ref:
            raise ProjectionError("event replaced the admitted author capability")
        handler = self.HANDLERS.get(event.kind)
        if handler is None:
            raise ProjectionError("unsupported event in admitted writer lineage")
        if event.kind != "rollout_started" and self.reply_stage in {
            AuthorStage.ACK,
            AuthorStage.DISCLOSURE,
            AuthorStage.UPDATE,
            AuthorStage.TURN,
        }:
            expected = {
                AuthorStage.ACK: "decision_disclosed",
                AuthorStage.DISCLOSURE: "author_turn",
                AuthorStage.UPDATE: "author_turn",
                AuthorStage.TURN: "context_changed",
            }[self.reply_stage]
            if event.kind != expected:
                raise ProjectionError("author reply transaction is incomplete or reordered")
        writer_stop = _is_writer_stop(store, event, state)
        if event.kind in {"writer_action", "tool_result"} or writer_stop:
            if not (event.kind == "tool_result" and before.position["phase"] == "awaiting_author"):
                validate_writer_effect(before, state, event, effect, store=store)
        if event.kind == "tool_result" and before.position["phase"] == "awaiting_author":
            handler = "_author_ack"
        elif event.kind == "termination_recorded" and writer_stop:
            handler = "_exhausted_stop"
        elif event.kind == "budget_charged" and not writer_stop:
            raise ProjectionError("unsupported event in admitted writer lineage")
        if (
            event.kind
            in {
                "external_requested",
                "check_recorded",
                "transition_committed",
                "external_response",
                "decision_disclosed",
                "requirements_changed",
                "author_turn",
            }
            and before.author_packet_ref is None
        ):
            raise ProjectionError("unsupported Phase 5 event in admitted writer lineage")
        if (
            event.kind == "termination_recorded"
            and not writer_stop
            and before.author_packet_ref is None
        ):
            raise ProjectionError("unsupported event in admitted writer lineage")
        contribution = getattr(self, handler)(event, before, state, effect)
        if contribution is not None:
            if contribution.replace:
                self.messages = list(contribution.messages)
                self.message_sources = list(contribution.sources)
            else:
                self.messages.extend(contribution.messages)
                self.message_sources.extend(contribution.sources)
            self.last_source = contribution.last_source

    def _rollout_started(self, event, before, state, effect) -> None:
        store = self.store
        position = state.position
        seeds = store.get_artifact(state.rng_ref)
        if (
            event.actor != "environment"
            or "writer" in event.audience
            or before.identity() != self.base.state.identity()
            or (event.node_visit_id != before.position["visit_id"])
            or (event.versions_ref != before.versions_ref)
            or (event.provenance_ref != before.provenance_ref)
            or (before.position["phase"] != "ready_writer")
            or before.history["action_ids"]
            or before.history["tool_result_ids"]
            or (set(effect["set"]) != {"position", "rng_ref"})
            or (set(effect["history_set"]) != {"branch_base"})
            or effect["file_delta"]
            or (
                position
                != {
                    **before.position,
                    "lineage_id": event.rollout_id,
                    "start_checkpoint": self.base_checkpoint_id,
                }
            )
            or (state.history["branch_base"] != before.history["head"])
            or (state.rng_ref == before.rng_ref)
            or (seeds.get("record_type") != "GroupMemberSeedsV1")
            or (seeds.get("member_id") != event.rollout_id)
            or (seeds.get("parent_rng_ref") != before.rng_ref)
            or (type(seeds.get("writer_seed")) is not int)
            or (type(seeds.get("environment_seed")) is not int)
        ):
            raise ProjectionError("invalid isolated group rollout start")
        return

    def _external_requested(self, event, before, state, effect) -> None:
        store = self.store
        entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=False)
        if "private" in store.artifact_visibilities(entry["record_ref"]):
            from writing_agent.task_graph_author_validation import validate_author_request_effect

            validate_author_request_effect(store, before, state, event, effect, entry)
            self.reply_stage = AuthorStage.REQUEST
        else:
            from writing_agent.task_graph_checks import validate_check_batch_effect

            validate_check_batch_effect(store, before, state, event, effect, entry)
        self.seen_entries.append(entry)
        return

    def _check_recorded(self, event, before, state, effect) -> None:
        store = self.store
        from writing_agent.task_graph_checks import validate_check_result_effect

        entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=False)
        validate_check_result_effect(store, before, state, event, effect, entry)
        self.seen_entries.append(entry)
        return

    def _terminal_event(self, event, before, state, effect) -> None:
        store = self.store
        entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=False)
        record = store.get_artifact(entry["record_ref"])
        if record.get("record_type") == "ScriptCoverageFailureV1":
            if event.kind != "termination_recorded":
                raise ProjectionError("coverage failure used the wrong event kind")
            from writing_agent.task_graph_author_validation import validate_coverage_failure_effect

            validate_coverage_failure_effect(store, before, state, event, effect, entry)
        else:
            from writing_agent.task_graph_terminal import validate_terminal_effect

            validate_terminal_effect(store, before, state, event, effect, entry)
        self.seen_entries.append(entry)
        return

    def _author_ack(self, event, before, state, effect) -> VisibleContribution:
        store = self.store
        if self.reply_stage != AuthorStage.REQUEST:
            raise ProjectionError("author acknowledgement is out of order")
        from writing_agent.task_graph_author_validation import validate_author_ack_effect

        entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=True)
        message = MessageV1.from_dict(
            store.get_artifact(entry["message_ref"], expected_domain="message")
        )
        validate_author_ack_effect(store, before, state, event, effect, entry, message)
        if not self.pending or self.pending.pop(0) != message.call_id:
            raise ProjectionError("author tool acknowledgement is unpaired")
        self.result_ids.append(f"{event.rollout_id}:tool_result:{len(self.result_ids)}")
        self.seen_entries.append(entry)
        self.reply_stage = AuthorStage.ACK
        return VisibleContribution((message,), (event.id,), event.id)

    def _decision_disclosed(self, event, before, state, effect) -> None:
        store = self.store
        if self.reply_stage != AuthorStage.ACK:
            raise ProjectionError("decision disclosure lacks author acknowledgement")
        from writing_agent.task_graph_author_validation import validate_decision_disclosure_effect

        entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=False)
        validate_decision_disclosure_effect(store, before, state, event, effect, entry)
        self.seen_entries.append(entry)
        self.reply_stage = AuthorStage.DISCLOSURE
        return

    def _requirements_changed(self, event, before, state, effect) -> None:
        store = self.store
        if self.reply_stage != AuthorStage.REQUEST:
            raise ProjectionError("feedback update is out of order")
        from writing_agent.task_graph_author_validation import validate_requirement_update_effect

        entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=False)
        validate_requirement_update_effect(store, before, state, event, effect, entry)
        self.seen_entries.append(entry)
        self.reply_stage = AuthorStage.UPDATE
        return

    def _author_turn(self, event, before, state, effect) -> VisibleContribution:
        store = self.store
        if before.continuation["author_request"] is None:
            raise ProjectionError("author turn has no outstanding request")
        request = store.get_artifact(before.continuation["author_request"], private=True)
        required_stage = AuthorStage.DISCLOSURE if request["source"] == "writer_request" else None
        if request["source"] == "mandatory_feedback":
            required_stage = (
                AuthorStage.UPDATE
                if self.reply_stage == AuthorStage.UPDATE
                else AuthorStage.REQUEST
            )
        if self.reply_stage != required_stage:
            raise ProjectionError("author turn lacks complete preceding effects")
        if request["source"] == "writer_request" and (
            before.continuation["next_call"] != len(before.continuation["tool_queue"])
            or self.pending
        ):
            raise ProjectionError("author turn cannot clear an undrained control call")
        from writing_agent.task_graph_author_validation import validate_author_turn_effect

        entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=True)
        message = MessageV1.from_dict(
            store.get_artifact(entry["message_ref"], expected_domain="message")
        )
        validate_author_turn_effect(store, before, state, event, effect, entry, message)
        self.seen_entries.append(entry)
        self.reply_stage = AuthorStage.TURN
        return VisibleContribution((message,), (event.id,), event.id)

    def _context_changed(self, event, before, state, effect) -> VisibleContribution | None:
        store = self.store
        from writing_agent.task_graph_projection import project_writer_context

        if event.actor != "environment" or "writer" in event.audience:
            raise ProjectionError("context change has invalid actor or audience")
        if effect.get("artifact_type") != "Phase2RecordedEffectV1":
            raise ProjectionError("context event has no replayable effect")
        context_operation = state.external_inputs_ref != before.external_inputs_ref
        if (self.phase5 or context_operation) and event.audience != ("controller", "trainer"):
            raise ProjectionError("context event has false audience ownership")
        old_budget = store.get_artifact(before.budgets_ref, expected_domain="payload")
        tracked_context = (
            not context_operation
            and isinstance(old_budget, dict)
            and isinstance(old_budget.get("limits"), dict)
            and ("context_bytes" in old_budget["limits"])
        )
        expected_set = {"context_ref"}
        if context_operation:
            expected_set.update({"budgets_ref", "external_inputs_ref"})
        elif tracked_context:
            expected_set.add("budgets_ref")
        if (
            set(effect["set"]) != expected_set
            or effect["history_set"]
            or effect["file_delta"]
            or (state.context_ref == before.context_ref)
            or (state.files != before.files)
            or (state.continuation != before.continuation)
        ):
            raise ProjectionError("context event has an unrelated execution effect")
        self.latest_context_ref = effect["set"].get("context_ref")
        if not isinstance(self.latest_context_ref, str):
            raise ProjectionError("context event lacks a revision")
        revision = store.load_context(self.latest_context_ref)
        if context_operation:
            if (
                event.node_visit_id != before.position["visit_id"]
                or event.versions_ref != before.versions_ref
                or event.provenance_ref != before.provenance_ref
            ):
                raise ProjectionError("context operation event has false causal ownership")
            from writing_agent.task_graph_compaction import (
                ContextOperationV1,
                ContextPolicyV1,
                make_record,
                require_quiescent,
            )

            if self.reply_stage is not None:
                raise ProjectionError("context operation interrupted an author transaction")
            require_quiescent(before, pending=tuple(self.pending))
            entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=False)
            record = store.get_artifact(entry["record_ref"], expected_domain="payload")
            if not isinstance(record, dict) or record.get("record_type") != "ContextOperationV1":
                raise ProjectionError("context operation lacks its typed record")
            ContextOperationV1.from_dict(record)
            policy_ref = record.get("policy_ref")
            if not isinstance(policy_ref, str):
                raise ProjectionError("context operation lacks a policy")
            policy = ContextPolicyV1.from_dict(store.get_artifact(policy_ref))
            seed = None
            seed_sources: list[str | None] = []
            if policy.operation == "seed":
                seed_id = policy.seed_checkpoint_ref
                if seed_id not in self.checkpoint_ancestry:
                    raise ProjectionError("named seed is not an ancestor checkpoint")
                seed_checkpoint = store.load_checkpoint(seed_id)
                if seed_checkpoint.state.history["seq"] > before.history["seq"]:
                    raise ProjectionError("named seed is newer than the operation")
                seed = store.load_context(seed_checkpoint.state.context_ref)
                project_writer_context(
                    store, self.base_checkpoint_id, seed_id, source_event_ids=seed_sources
                )
            expected_record, expected_budget, selected_sources = make_record(
                store,
                before,
                ContextRevisionV1(
                    messages=tuple(self.messages),
                    tools=self.baseline.tools,
                    rendering=self.baseline.rendering,
                ),
                tuple(self.message_sources),
                policy,
                policy_ref,
                revision,
                record.get("summary_ref"),
                store.get_artifact(before.budgets_ref, expected_domain="payload"),
                seed=seed,
                seed_sources=tuple(seed_sources),
                recorded_summary=record.get("summary_text"),
            )
            if (
                canonical_bytes(record) != canonical_bytes(expected_record)
                or canonical_bytes(store.get_artifact(state.budgets_ref, expected_domain="payload"))
                != canonical_bytes(expected_budget)
                or state.files != before.files
                or (state.requirements_ref != before.requirements_ref)
                or (state.decisions_ref != before.decisions_ref)
                or (state.disclosures_ref != before.disclosures_ref)
                or (state.position != before.position)
                or (state.outcome_ref != before.outcome_ref)
            ):
                raise ProjectionError("context operation has false selection, charge or authority")
            selected_message_sources = [
                event.id if policy.operation == "compact" and index == 2 else source
                for index, source in enumerate(selected_sources)
            ]
            self.seen_entries.append(entry)
        elif (
            revision.event_head != self.last_source
            or revision.provenance_refs != (self.last_source,)
            or revision.messages != tuple(self.messages)
        ):
            raise ProjectionError("context revision has false source-event provenance")
        if not context_operation:
            expected_budget = charge_context_append(old_budget, revision)
            if expected_budget is None:
                if state.budgets_ref != before.budgets_ref:
                    raise ProjectionError("unmetered context change altered budget")
            elif (
                store.get_artifact(state.budgets_ref, expected_domain="payload") != expected_budget
            ):
                raise ProjectionError("context append has false budget charge")
        if self.reply_stage == AuthorStage.TURN:
            self.reply_stage = None
        if context_operation:
            return VisibleContribution(
                tuple(revision.messages),
                tuple(selected_message_sources),
                before.history["head"],
                replace=True,
            )
        return None

    def _sampled_stop(self, event, before, state, effect) -> None:
        store = self.store
        entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=False)
        record = store.get_artifact(entry["record_ref"], expected_domain="payload")
        if (
            not isinstance(record, dict)
            or set(record) != _STOP_RECORD_FIELDS
            or record.get("record_type") != "WriterSampledBudgetStopV1"
        ):
            raise ProjectionError("sampled stop log names an invalid record")
        trace = store.get_artifact(record["trace_ref"], expected_domain="payload")
        validate_action_trace(
            store,
            record,
            trace,
            f"{event.rollout_id}:action:{len(self.action_ids)}",
            context_content_hash(
                tuple(self.messages), tools=self.baseline.tools, rendering=self.baseline.rendering
            ),
            self.latest_context_ref,
            self.baseline.rendering,
        )
        prepared_ref = record["prepared_request_ref"]
        if prepared_ref is not None:
            _validate_prepared_request(
                store,
                prepared_ref,
                trace.get("context_content_hash"),
                self.latest_context_ref,
                self.baseline.rendering,
                record["request_ref"],
            )
        usage = record["usage"]
        old_budget = store.get_artifact(before.budgets_ref, expected_domain="payload")
        new_budget = store.get_artifact(state.budgets_ref, expected_domain="payload")
        expected, exceeded = sampled_usage_charge(old_budget, usage)
        outcome = store.get_artifact(state.outcome_ref, expected_domain="payload")
        if (
            event.actor != "writer_runtime"
            or "writer" in event.audience
            or trace.get("action_id") != f"{event.rollout_id}:action:{len(self.action_ids)}"
            or (
                trace.get("context_content_hash")
                != context_content_hash(
                    tuple(self.messages),
                    tools=self.baseline.tools,
                    rendering=self.baseline.rendering,
                )
            )
            or (trace.get("context_revision_ref") != self.latest_context_ref)
            or (trace.get("exact_request_ref") != record["request_ref"])
            or (trace.get("raw_output_ref") != record["raw_output_ref"])
            or (exceeded is None)
            or (record["reason"] != f"{exceeded}_budget")
            or (not isinstance(outcome, dict))
            or (
                outcome
                != _writer_stop_outcome(
                    before,
                    outcome.get("candidate_checkpoint"),
                    record["reason"],
                    phase5=self.phase5,
                )
            )
            or (
                self.phase5
                and (
                    outcome["candidate_checkpoint"] not in self.checkpoint_ancestry
                    or store.load_checkpoint(outcome["candidate_checkpoint"]).state != before
                )
            )
            or (new_budget != expected)
            or (state.files != before.files)
            or (state.context_ref != before.context_ref)
            or (state.continuation != before.continuation)
            or (state.position["phase"] != "terminal")
            or effect["file_delta"]
        ):
            raise ProjectionError("sampled budget stop has false accounting or authority")
        self.seen_entries.append(entry)
        return

    def _exhausted_stop(self, event, before, state, effect) -> None:
        store = self.store
        entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=False)
        record = store.get_artifact(entry["record_ref"], expected_domain="payload")
        old_budget = store.get_artifact(before.budgets_ref, expected_domain="payload")
        outcome = store.get_artifact(state.outcome_ref, expected_domain="payload")
        reason = exhausted_stop_reason(old_budget)
        if (
            reason is None
            or not isinstance(outcome, dict)
            or outcome
            != _writer_stop_outcome(
                before, outcome.get("candidate_checkpoint"), reason, phase5=self.phase5
            )
            or (
                self.phase5
                and (
                    outcome["candidate_checkpoint"] not in self.checkpoint_ancestry
                    or store.load_checkpoint(outcome["candidate_checkpoint"]).state != before
                )
            )
            or (event.actor != "writer_runtime")
            or ("writer" in event.audience)
            or (record != {"record_type": "WriterExhaustedStopV1", "reason": reason})
            or (state.position["phase"] != "terminal")
            or (state.files != before.files)
            or (state.context_ref != before.context_ref)
            or (state.continuation != before.continuation)
            or (state.budgets_ref != before.budgets_ref)
            or effect["file_delta"]
        ):
            raise ProjectionError("termination is not an exhausted writer stop")
        self.seen_entries.append(entry)
        return

    def _visible_writer(self, event, before, state, effect) -> VisibleContribution:
        store = self.store
        expected_actor = "writer" if event.kind == "writer_action" else "environment"
        if event.actor != expected_actor or "writer" not in event.audience:
            raise ProjectionError("writer-visible event has invalid actor or audience")
        if effect.get("artifact_type") != "Phase2RecordedEffectV1":
            raise ProjectionError("writer event has no replayable effect")
        entry = _writer_log_entry(store, state, event, self.seen_entries, has_message=True)
        self.seen_entries.append(entry)
        record = store.get_artifact(entry["record_ref"], expected_domain="payload")
        message = MessageV1.from_dict(
            store.get_artifact(entry["message_ref"], expected_domain="message")
        )
        if event.kind == "writer_action":
            if (
                self.pending
                or not isinstance(record, dict)
                or set(record) != _ACTION_RECORD_FIELDS
                or (record.get("record_type") != "WriterActionV1")
            ):
                raise ProjectionError("writer action starts before prior calls finish")
            if (
                message.role != "assistant"
                or not message.loss_eligible
                or message.origin != record["action_id"]
            ):
                raise ProjectionError("writer action message identity is invalid")
            trace = store.get_artifact(record["trace_ref"], expected_domain="payload")
            validate_action_trace(
                store,
                record,
                trace,
                f"{event.rollout_id}:action:{len(self.action_ids)}",
                context_content_hash(
                    tuple(self.messages),
                    tools=self.baseline.tools,
                    rendering=self.baseline.rendering,
                ),
                self.latest_context_ref,
                self.baseline.rendering,
                message,
            )
            expected_context = context_content_hash(
                tuple(self.messages), tools=self.baseline.tools, rendering=self.baseline.rendering
            )
            if trace.get("context_content_hash") != expected_context:
                raise ProjectionError("writer trace names a different sampling context")
            if trace.get("context_revision_ref") != self.latest_context_ref:
                raise ProjectionError("writer trace names a different context revision")
            if canonical_json(trace.get("rendering")) != canonical_json(self.baseline.rendering):
                raise ProjectionError("writer trace rendering pins differ from context")
            prepared_ref = trace.get("prepared_request_ref")
            if prepared_ref is not None:
                _validate_prepared_request(
                    store,
                    prepared_ref,
                    expected_context,
                    self.latest_context_ref,
                    self.baseline.rendering,
                    trace.get("exact_request_ref"),
                )
            calls = [part for part in message.content if part["type"] == "tool_call"]
            if [part["id"] for part in calls] != [call["call_id"] for call in record["calls"]]:
                raise ProjectionError("action syntax and call metadata differ")
            queue = effect["set"].get("continuation", {}).get("tool_queue", ())
            if canonical_json(
                [
                    {"call_id": part["id"], "name": part["name"], "arguments": part["arguments"]}
                    for part in calls
                ]
            ) != canonical_json(queue):
                raise ProjectionError("committed tool queue differs from assistant syntax")
            if record["action_id"] in self.action_ids:
                raise ProjectionError("duplicate writer action logical ID")
            if record["action_id"] != f"{event.rollout_id}:action:{len(self.action_ids)}":
                raise ProjectionError("writer action ordinal is false")
            old_budget = store.get_artifact(before.budgets_ref, expected_domain="payload")
            new_budget = store.get_artifact(state.budgets_ref, expected_domain="payload")
            usage = record["usage"]
            expected_budget, _ = sampled_usage_charge(old_budget, usage)
            if (
                new_budget != expected_budget
                or state.files != before.files
                or effect["file_delta"]
                or (state.continuation["next_call"] != 0)
                or (list(state.history["action_ids"]) != [*self.action_ids, record["action_id"]])
            ):
                raise ProjectionError("writer action state or budget charge is false")
            self.action_ids.append(record["action_id"])
            for index, part in enumerate(calls):
                if part["id"] in self.seen_calls:
                    raise ProjectionError("duplicate logical call ID")
                validation_error = record["calls"][index]["validation_error"]
                if (queue[index]["name"] == "invalid_call") is not (validation_error is not None):
                    raise ProjectionError("call validity contradicts its queued tool name")
                self.seen_calls.add(part["id"])
                self.pending.append(part["id"])
                self.call_sources[part["id"]] = (
                    record["action_id"],
                    queue[index],
                    record["calls"][index],
                )
        else:
            if (
                not isinstance(record, dict)
                or set(record) != _RESULT_RECORD_FIELDS
                or record.get("record_type") != "WriterToolResultV1"
            ):
                raise ProjectionError("tool event metadata has wrong type")
            if (
                message.role != "tool"
                or message.loss_eligible
                or message.call_id != record["call_id"]
            ):
                raise ProjectionError("tool observation has invalid role or loss eligibility")
            if not self.pending or self.pending.pop(0) != message.call_id:
                raise ProjectionError("tool observation is unpaired or out of order")
            source = self.call_sources.get(message.call_id)
            if source is None or record["action_id"] != source[0] or message.origin != source[0]:
                raise ProjectionError("tool observation has the wrong action origin")
            cursor = before.continuation["next_call"]
            if (
                canonical_json(before.continuation["tool_queue"][cursor])
                != canonical_json(source[1])
                or state.continuation["next_call"] != cursor + 1
                or state.continuation["tool_queue"] != before.continuation["tool_queue"]
                or (record["result_id"] != f"{event.rollout_id}:tool_result:{len(self.result_ids)}")
                or (
                    list(state.history["tool_result_ids"])
                    != [*self.result_ids, record["result_id"]]
                )
            ):
                raise ProjectionError("tool result queue, cursor or ordinal is false")
            if record["result_id"] in self.result_ids:
                raise ProjectionError("duplicate tool result logical ID")
            self.result_ids.append(record["result_id"])
            old_context = ContextRevisionV1(
                messages=tuple(self.messages),
                tools=self.baseline.tools,
                rendering=self.baseline.rendering,
            )
            new_context = ContextRevisionV1(
                messages=(*self.messages, message),
                tools=self.baseline.tools,
                rendering=self.baseline.rendering,
            )
            validate_result_production(
                store,
                before,
                state,
                old_context,
                new_context,
                record,
                message,
                effect,
                {"action_id": source[0], "validation_error": source[2]["validation_error"]},
            )
        return VisibleContribution((message,), (event.id,), event.id)


def replay_writer_history(
    store: TaskGraphStore,
    base_checkpoint_id: str,
    target_checkpoint_id: str,
    *,
    candidate_events: tuple[EventV1, ...] = (),
    candidate_state: EnvironmentStateV1 | None = None,
) -> tuple[SemanticReplayEngine, EnvironmentStateV1]:
    """Replay the stored suffix and optional staged batch with identical authority."""
    from writing_agent.task_graph_contracts import NodeContractV1

    base = store.load_checkpoint(base_checkpoint_id)
    target = store.load_checkpoint(target_checkpoint_id)
    if base.state.instance_ref != target.state.instance_ref:
        raise ProjectionError("projection crosses graph instances")
    ancestor = target
    ancestry = {target_checkpoint_id}
    while ancestor.identity() != base_checkpoint_id:
        if not ancestor.parents:
            raise ProjectionError("target checkpoint is not descended from projection base")
        ancestor = store.load_checkpoint(ancestor.parents[0])
        ancestry.add(ancestor.identity())
    baseline = store.load_context(base.state.context_ref)
    instance = store.load_instance(base.state.instance_ref)
    spec = next(
        (node for node in instance.nodes if node.id == base.state.position["node_id"]), None
    )
    if spec is None:
        raise ProjectionError("projection entry names an unknown node")
    contract = NodeContractV1.from_dict(store.get_artifact(spec.entry_contract))
    phase5 = contract.interaction_contract.mode == "scripted_author"
    if phase5 and base.state.author_packet_ref != contract.interaction_contract.author_packet_ref:
        raise ProjectionError("entry lacks its admitted author capability")
    messages = list(baseline.messages)
    if (
        len(messages) < 2
        or messages[0].role != "system"
        or messages[0].trust != "instructions"
        or messages[1].role != "user"
    ):
        raise ProjectionError("entry context lacks system instructions and node request")
    if any(message.loss_eligible for message in messages):
        raise ProjectionError("seed context cannot contain writer targets")
    pending: list[str] = []
    seen_calls: set[str] = set()
    for message in messages:
        for part in message.content:
            if part["type"] == "tool_call":
                if message.role != "assistant" or part["id"] in seen_calls:
                    raise ProjectionError("invalid seeded tool-call pairing")
                pending.append(part["id"])
                seen_calls.add(part["id"])
            elif part["type"] == "tool_result":
                if message.role != "tool" or not pending or pending.pop(0) != part["call_id"]:
                    raise ProjectionError("seeded tool result is unmatched")
    if pending:
        raise ProjectionError("seed context ends with an incomplete tool exchange")
    events: list[EventV1] = []
    event_cursor = target.event_head
    while event_cursor != base.event_head:
        if event_cursor is None:
            raise ProjectionError("target does not descend from projection base")
        event = store.load_event(event_cursor)
        events.append(event)
        event_cursor = event.previous
    events.reverse()
    if candidate_events:
        if candidate_state is None:
            raise ProjectionError("candidate events require a proposed final state")
        events.extend(candidate_events)
    elif candidate_state is not None:
        raise ProjectionError("candidate state requires proposed events")
    cursor = SemanticReplayEngine(
        store=store,
        base=base,
        baseline=baseline,
        base_checkpoint_id=base_checkpoint_id,
        checkpoint_ancestry=ancestry,
        phase5=phase5,
        messages=messages,
        message_sources=[None] * len(messages),
        pending=pending,
        seen_calls=seen_calls,
        action_ids=list(base.state.history["action_ids"]),
        result_ids=list(base.state.history["tool_result_ids"]),
        last_source=baseline.event_head,
        latest_context_ref=base.state.context_ref,
        state=base.state,
        call_sources={},
        seen_entries=[],
    )
    for event in events:
        cursor.step(event)
    return cursor, candidate_state if candidate_state is not None else target.state
