"""Versioned sampling wire codecs, evidence binding, and eligibility policy.

The V1 wire dictionaries are retained byte-for-byte for approved stored identities.
Typed values are immutable; decoding is the only place that interprets duplicated
sampling fields. Opaque adapter claims never establish native token eligibility.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from writing_agent.task_graph import MessageV1, canonical_bytes, canonical_json, validate_hash
from writing_agent.task_graph_calls import ToolQueueEntry, parse_calls
from writing_agent.task_graph_errors import (
    ProjectionError,
    VerifiedMessagesStaleError,
    WriterRuntimeError,
)
from writing_agent.task_graph_records import (
    OutcomeV1,
    WriterRequestV1,
    WriterTurnV1,
    decode_canonical_value,
)

NATIVE_TRACE_REASON = "native token alignment and loss masks are not implemented in Phase 4"
_ACTION_RECORD_FIELDS = {
    "record_type",
    "action_id",
    "trace_ref",
    "request_ref",
    "prepared_request_ref",
    "raw_output_ref",
    "logprob_ref",
    "calls",
    "usage",
    "model",
    "seed",
    "loss_eligibility",
}
_STOP_RECORD_FIELDS = {
    "record_type",
    "action_id",
    "reason",
    "usage",
    "model",
    "seed",
    "parsed_message_json",
    "trace_ref",
    "request_ref",
    "prepared_request_ref",
    "raw_output_ref",
    "logprob_ref",
}
_TRACE_FIELDS = {
    "record_type",
    "action_id",
    "context_content_hash",
    "context_revision_ref",
    "rendering",
    "exact_request_ref",
    "prepared_request_ref",
    "raw_output_ref",
    "raw_output_evidence",
    "logprob_ref",
    "adapter_trace",
    "token_evidence",
    "logprob_evidence",
    "usage",
    "model",
    "seed",
    "native_on_policy_eligible",
    "reason",
}


@dataclass(frozen=True)
class PreparedRequestV1:
    kind: str
    context_content_hash: str
    context_revision_ref: str
    rendering_json: str
    payload_ref: str | None

    @classmethod
    def from_wire(cls, value: Any) -> PreparedRequestV1:
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "record_type",
                "context_content_hash",
                "context_revision_ref",
                "rendering",
                "payload_ref",
            }
            or value["record_type"] not in {"PreparedWriterRequestV1", "VerifiedWriterMessagesV1"}
        ):
            raise ProjectionError("prepared request has wrong schema")
        if not isinstance(value["rendering"], dict):
            raise ProjectionError("prepared request rendering has wrong schema")
        return cls(
            value["record_type"],
            value["context_content_hash"],
            value["context_revision_ref"],
            canonical_json(value["rendering"]),
            value["payload_ref"],
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "record_type": self.kind,
            "context_content_hash": self.context_content_hash,
            "context_revision_ref": self.context_revision_ref,
            "rendering": json.loads(self.rendering_json),
            "payload_ref": self.payload_ref,
        }

    def bind(
        self,
        store,
        context_hash: str,
        context_ref: str,
        rendering: Mapping,
        payload_ref: str | None,
    ) -> None:
        if self.to_wire() != {
            "record_type": self.kind,
            "context_content_hash": context_hash,
            "context_revision_ref": context_ref,
            "rendering": dict(rendering),
            "payload_ref": payload_ref,
        }:
            raise ProjectionError("prepared request contradicts action trace")
        if self.kind == "VerifiedWriterMessagesV1":
            payload = store.get_artifact(payload_ref, expected_domain="payload")
            messages = [m.to_dict() for m in store.load_context(context_ref).messages]
            if not isinstance(payload, dict) or canonical_bytes(
                payload.get("messages")
            ) != canonical_bytes(messages):
                raise VerifiedMessagesStaleError(
                    "verified request messages differ from current context"
                )


@dataclass(frozen=True)
class AdapterEvidenceV1:
    """Opaque adapter metadata with normalized references and claims."""

    body_json: str
    model: str | None
    seed: int | None
    logprob_ref: str | None
    has_tokens: bool

    @classmethod
    def from_wire(cls, value: Mapping[str, Any] | None) -> AdapterEvidenceV1 | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ProjectionError("adapter trace must be an object")
        return cls(
            canonical_json(value),
            value.get("model"),
            value.get("seed"),
            value.get("per_token_logprobs_ref"),
            "generated_token_ids" in value,
        )

    def to_wire(self) -> dict[str, Any]:
        return json.loads(self.body_json)


@dataclass(frozen=True)
class EligibilityDecisionV1:
    native_on_policy_eligible: bool = False
    trace_reason: str = NATIVE_TRACE_REASON
    training_status: str = "ineligible"
    training_reason: str = "native_action_trace_unavailable"

    def training_wire(self, outcome_ref: str) -> dict[str, Any]:
        return {
            "record_type": "TrainingEligibilityV1",
            "schema": 1,
            "terminal_outcome_ref": outcome_ref,
            "status": self.training_status,
            "reason": self.training_reason,
        }


CURRENT_ELIGIBILITY = EligibilityDecisionV1()


def _decode_training_eligibility(value: Any, outcome_ref: str) -> EligibilityDecisionV1:
    try:
        matches = canonical_bytes(value) == canonical_bytes(
            CURRENT_ELIGIBILITY.training_wire(outcome_ref)
        )
    except (TypeError, ValueError) as exc:
        raise ProjectionError("training eligibility has invalid wire value") from exc
    if not matches:
        raise ProjectionError("training eligibility contradicts current native policy")
    return CURRENT_ELIGIBILITY


def make_prepared_request(context, payload_ref: str, *, verified: bool) -> PreparedRequestV1:
    return PreparedRequestV1(
        "VerifiedWriterMessagesV1" if verified else "PreparedWriterRequestV1",
        context.content_hash,
        context.identity(),
        canonical_json(context.rendering),
        payload_ref,
    )


def _validate_adapter_claims(store, trace: Mapping[str, Any] | None) -> None:
    if trace is None:
        return
    if not isinstance(trace, Mapping):
        raise ProjectionError("trace metadata must be an object")
    if "per_token_logprobs" in trace:
        raise ProjectionError("native logprob arrays require a binary artifact reference")
    if "per_token_logprobs_ref" in trace:
        validate_hash(trace["per_token_logprobs_ref"])
        store.get_artifact(trace["per_token_logprobs_ref"], expected_domain="payload:bytes")
    if "generated_token_ids" in trace and (
        not isinstance(trace["generated_token_ids"], list)
        or any(type(token) is not int or token < 0 for token in trace["generated_token_ids"])
    ):
        raise ProjectionError("generated token IDs must be nonnegative integers")
    typed_logprobs = {"per_token_logprobs_codec", "per_token_logprobs_shape"}
    if typed_logprobs & trace.keys():
        tokens = trace.get("generated_token_ids")
        shape = trace.get("per_token_logprobs_shape")
        if (
            not typed_logprobs <= trace.keys()
            or "per_token_logprobs_ref" not in trace
            or trace["per_token_logprobs_codec"] != "f32-le"
            or not isinstance(shape, list)
            or len(shape) != 1
            or type(shape[0]) is not int
            or shape[0] < 0
            or not isinstance(tokens, list)
            or len(tokens) != shape[0]
            or len(
                store.get_artifact(trace["per_token_logprobs_ref"], expected_domain="payload:bytes")
            )
            != 4 * shape[0]
        ):
            raise ProjectionError("binary logprob metadata contradicts token evidence")


def make_sampling_evidence(
    store,
    action_id: str,
    context,
    request_ref: str | None,
    prepared_request_ref: str | None,
    raw_output_ref: str | None,
    usage: Mapping[str, Any],
    adapter_trace: Mapping[str, Any] | None,
) -> SamplingEvidenceV1:
    _validate_adapter_claims(store, adapter_trace)
    adapter = AdapterEvidenceV1.from_wire(adapter_trace)
    return SamplingEvidenceV1(
        action_id=action_id,
        context_content_hash=context.content_hash,
        context_revision_ref=context.identity(),
        rendering_json=canonical_json(context.rendering),
        exact_request_ref=request_ref,
        prepared_request_ref=prepared_request_ref,
        raw_output_ref=raw_output_ref,
        logprob_ref=adapter.logprob_ref if adapter else None,
        usage_json=canonical_json(usage),
        model=adapter.model if adapter else None,
        seed=adapter.seed if adapter else None,
        adapter=adapter,
    )


@dataclass(frozen=True)
class SamplingEvidenceV1:
    """Typed normalized fields around the frozen WriterActionTraceV1 wire shape."""

    action_id: str
    context_content_hash: str
    context_revision_ref: str
    rendering_json: str
    exact_request_ref: str | None
    prepared_request_ref: str | None
    raw_output_ref: str | None
    logprob_ref: str | None
    usage_json: str
    model: str | None
    seed: int | None
    adapter: AdapterEvidenceV1 | None
    eligibility: EligibilityDecisionV1 = CURRENT_ELIGIBILITY

    @classmethod
    def from_wire(cls, value: Any) -> SamplingEvidenceV1:
        if (
            not isinstance(value, dict)
            or set(value) != _TRACE_FIELDS
            or value.get("record_type") != "WriterActionTraceV1"
        ):
            raise ProjectionError("writer action trace has wrong type")
        if (
            value["native_on_policy_eligible"] is not False
            or value["reason"] != CURRENT_ELIGIBILITY.trace_reason
        ):
            raise ProjectionError("writer action trace has wrong eligibility")
        instance = cls(
            value["action_id"],
            value["context_content_hash"],
            value["context_revision_ref"],
            canonical_json(value["rendering"]),
            value["exact_request_ref"],
            value["prepared_request_ref"],
            value["raw_output_ref"],
            value["logprob_ref"],
            canonical_json(value["usage"]),
            value["model"],
            value["seed"],
            AdapterEvidenceV1.from_wire(value["adapter_trace"]),
        )
        if instance.to_wire() != value:
            raise ProjectionError("writer action trace evidence contradicts references")
        return instance

    def to_wire(self) -> dict[str, Any]:
        return {
            "record_type": "WriterActionTraceV1",
            "action_id": self.action_id,
            "context_content_hash": self.context_content_hash,
            "context_revision_ref": self.context_revision_ref,
            "rendering": json.loads(self.rendering_json),
            "exact_request_ref": self.exact_request_ref,
            "prepared_request_ref": self.prepared_request_ref,
            "raw_output_ref": self.raw_output_ref,
            "usage": json.loads(self.usage_json),
            "model": self.model,
            "seed": self.seed,
            "raw_output_evidence": "supplied" if self.raw_output_ref is not None else "missing",
            "logprob_ref": self.logprob_ref,
            "adapter_trace": self.adapter.to_wire() if self.adapter else None,
            "token_evidence": "supplied" if self.adapter and self.adapter.has_tokens else "missing",
            "logprob_evidence": "supplied" if self.logprob_ref is not None else "missing",
            "native_on_policy_eligible": self.eligibility.native_on_policy_eligible,
            "reason": self.eligibility.trace_reason,
        }


@dataclass(frozen=True)
class SamplingRecordV1:
    kind: str
    action_id: str
    request_ref: str | None
    prepared_request_ref: str | None
    raw_output_ref: str | None
    logprob_ref: str | None
    usage_json: str
    model: str | None
    seed: int | None

    @classmethod
    def from_wire(cls, record: Mapping[str, Any]) -> SamplingRecordV1:
        return cls(
            record["record_type"],
            record["action_id"],
            record["request_ref"],
            record["prepared_request_ref"],
            record["raw_output_ref"],
            record["logprob_ref"],
            canonical_json(record["usage"]),
            record["model"],
            record["seed"],
        )


@dataclass(frozen=True)
class BoundSamplingV1:
    record: SamplingRecordV1
    evidence: SamplingEvidenceV1
    prepared: PreparedRequestV1 | None
    eligibility: EligibilityDecisionV1


@dataclass(frozen=True)
class ActionSamplingBindingV1:
    store: Any
    record: Mapping[str, Any]
    trace: Mapping[str, Any]
    action_id: str
    context_content_hash: str
    context_revision_ref: str
    rendering: Mapping[str, Any]
    message: Any = None


@dataclass(frozen=True)
class WriterTurnSamplingBindingV1:
    """Bind re-cut sampling evidence to a typed turn and its active context."""

    turn: WriterTurnV1
    context: Any
    reader: Any


@dataclass(frozen=True)
class BoundWriterTurnSamplingV1:
    turn: WriterTurnV1
    prepared: WriterRequestV1 | None
    eligibility: EligibilityDecisionV1 = CURRENT_ELIGIBILITY


@dataclass(frozen=True)
class TrainingEligibilityBindingV1:
    wire: Any
    outcome_ref: str


def decode_and_bind_sampling(
    binding: ActionSamplingBindingV1 | WriterTurnSamplingBindingV1 | TrainingEligibilityBindingV1,
) -> BoundSamplingV1 | BoundWriterTurnSamplingV1 | EligibilityDecisionV1:
    """The only public decoder for old and re-cut sampling claims."""
    if isinstance(binding, TrainingEligibilityBindingV1):
        return _decode_training_eligibility(binding.wire, binding.outcome_ref)
    if isinstance(binding, WriterTurnSamplingBindingV1):
        return _decode_writer_turn_sampling(binding.turn, binding.context, binding.reader)
    if not isinstance(binding, ActionSamplingBindingV1):
        raise TypeError("unsupported sampling binding request")
    return _decode_action_sampling(
        binding.store,
        binding.record,
        binding.trace,
        binding.action_id,
        binding.context_content_hash,
        binding.context_revision_ref,
        binding.rendering,
        binding.message,
    )


def bind_group_sampling_claims(
    policy: Mapping[str, str],
    writer_seed: int,
    trace: Mapping[str, Any],
    claims: Any,
    *,
    model_id: str | None = None,
    context_content_hash: str | None = None,
    context_revision_ref: str | None = None,
    rendering: Mapping[str, Any] | None = None,
) -> None:
    """Compare adapter claims with caller-pinned group and active-view values.

    Keyword expectations are supplied by the new derive. Their ``None`` defaults
    preserve the old group caller, which still binds against the legacy trace.
    """
    if not isinstance(claims, Mapping):
        return
    for field, expected in policy.items():
        if field in claims and canonical_bytes(claims[field]) != canonical_bytes(expected):
            raise ProjectionError(f"writer sample used a different {field}")
    if "policy_ref" in claims and canonical_bytes(claims["policy_ref"]) != canonical_bytes(
        policy["behavior_policy_ref"]
    ):
        raise ProjectionError("writer sample used a different behavior policy")
    for field, expected in (
        ("seed", writer_seed),
        ("model", trace.get("model") if model_id is None else model_id),
        (
            "context_content_hash",
            trace.get("context_content_hash")
            if context_content_hash is None
            else context_content_hash,
        ),
        (
            "context_revision_ref",
            trace.get("context_revision_ref")
            if context_revision_ref is None
            else context_revision_ref,
        ),
        ("rendering", trace.get("rendering") if rendering is None else rendering),
    ):
        if (
            expected is not None
            and field in claims
            and canonical_bytes(claims[field]) != canonical_bytes(expected)
        ):
            raise ProjectionError(f"writer sample request/adapter changed {field}")


def bind_group_writer_sampling(view, turn: WriterTurnV1, reader) -> None:
    """Bind group sampling claims to the active sealed member and context."""
    spec = view.group
    lineage_id = view.state.position["lineage_id"]
    member = next((item for item in spec.members if item.member_id == lineage_id), None)
    if member is None:
        raise ProjectionError("writer lineage is absent from its sealed group spec")
    try:
        model = reader.artifact(spec.policy["model_ref"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("sealed group model is unavailable") from exc
    if not isinstance(model, Mapping) or not isinstance(model.get("model_id"), str):
        raise ProjectionError("sealed group model has no model ID")
    claims = turn.adapter_trace
    bind_group_sampling_claims(
        spec.policy,
        member.writer_seed,
        claims or {},
        claims or {},
        model_id=model["model_id"],
        context_content_hash=view.context.content_ref,
        context_revision_ref=view.context.revision_ref,
        rendering=view.context.rendering,
    )


def usage_overrun_outcome(
    outcome: OutcomeV1, checkpoint_ref: str, requirements_ref: str, reason: str
):
    return replace(
        outcome,
        task_status="incomplete",
        execution_status="valid",
        stop_reason=reason,
        candidate_checkpoint=checkpoint_ref,
        requirement_version=requirements_ref,
    )


def parse_writer_turn_calls(
    view, turn: WriterTurnV1, reader, ask_semantics=None
) -> list[ToolQueueEntry]:
    """Parse a sampled turn's calls against its active allowlist and prior raw IDs."""
    ordinal = turn.action_id.removeprefix(f"{view.state.position['lineage_id']}:action:")
    if not ordinal.isdecimal():
        raise ProjectionError("sampled action ID has an invalid ordinal")
    id_prefix = f"{view.state.position['lineage_id']}:call:{ordinal}"
    if not turn.message.tool_calls_was_list:
        return [ToolQueueEntry(f"{id_prefix}:0", "invalid_call", {}, "tool_calls must be an array")]
    prior = _prior_raw_call_ids(view, turn.action_id, reader)
    if view.mode.ask_semantics:
        if ask_semantics is None:
            raise ProjectionError("ask semantics are required for this writer turn")
        decisions = reader.artifact(view.state.decisions_ref)

        def ask(arguments):
            ask_semantics(arguments, view.node, decisions)
    else:
        ask = None

    try:
        return parse_calls(
            turn.message,
            id_prefix=id_prefix,
            allowed=frozenset(view.node.contract.entry_contract.tool_allowlist),
            prior_raw_ids=prior,
            ask_semantics=ask,
        )
    except WriterRuntimeError as exc:
        raise ProjectionError("sampled calls violate the canonical call envelope") from exc


def _prior_raw_call_ids(view, action_id: str, reader) -> frozenset[str]:
    prior: set[str] = set()
    for sample in view.samples:
        if sample.action_id == action_id:
            break
        if sample.outcome != "action":
            continue
        try:
            raw = reader.artifact(sample.turn_ref)
            turn = raw if isinstance(raw, WriterTurnV1) else WriterTurnV1.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectionError("writer sample history cannot be decoded") from exc
        if turn.message.tool_calls_was_list:
            prior.update(_raw_call_ids(turn))
    return frozenset(prior)


def _raw_call_ids(turn: WriterTurnV1) -> set[str]:
    found: set[str] = set()
    for item in turn.message.calls:
        if not item["bounded"]:
            continue
        raw = decode_canonical_value(item["value"])
        if not isinstance(raw, Mapping):
            continue
        raw_id = raw.get("id")
        if (
            isinstance(raw_id, str)
            and raw_id
            and not any(character.isspace() for character in raw_id)
            and raw_id.isprintable()
        ):
            found.add(raw_id)
    return found


def updated_raw_call_ids(prior: frozenset[str], turn: WriterTurnV1) -> frozenset[str]:
    return (
        prior if not turn.message.tool_calls_was_list else frozenset((*prior, *_raw_call_ids(turn)))
    )


def assistant_message(
    action_id: str, content: str, turn: WriterTurnV1, queue: list[ToolQueueEntry]
) -> MessageV1:
    parts: list[dict[str, Any]] = []
    if content:
        parts.append({"type": "text", "text": content})
    for index, call in enumerate(queue):
        raw = (
            turn.message.calls[index]["value"]
            if turn.message.tool_calls_was_list and index < len(turn.message.calls)
            else turn.message.calls
        )
        if call.rejection is not None:
            parts.append({"type": "invalid_tool_call", "id": call.call_id, "raw": raw})
        else:
            parts.append(
                {
                    "type": "tool_call",
                    "id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )
    return MessageV1(role="assistant", content=tuple(parts), origin=action_id, loss_eligible=True)


def requires_usage_evidence(budget: Mapping[str, Any], usage: Mapping[str, Any]) -> bool:
    limits = budget["limits"]
    return ("generated_tokens" in limits and "completion_tokens" not in usage) or (
        "total_tokens" in limits
        and "total_tokens" not in usage
        and not {"prompt_tokens", "completion_tokens"} <= usage.keys()
    )


def _decode_writer_turn_sampling(
    turn: WriterTurnV1, context: Any, reader: Any
) -> BoundWriterTurnSamplingV1:
    if not isinstance(turn, WriterTurnV1):
        raise ProjectionError("sampling input is not a WriterTurnV1")
    if turn.context_revision_ref != context.revision_ref:
        raise ProjectionError("writer turn context revision differs from the active context")

    adapter = turn.adapter_trace
    if adapter is not None:
        if not isinstance(adapter, Mapping) or "native_on_policy_eligible" in adapter:
            raise ProjectionError("adapter trace may not claim native eligibility")
        tokens_present = "generated_token_ids" in adapter
        tokens = adapter.get("generated_token_ids")
        if tokens_present and (
            not isinstance(tokens, (tuple, list))
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ProjectionError("generated token IDs must be nonnegative integers")
        if tokens_present and (
            type(turn.usage.get("completion_tokens")) is not int
            or turn.usage["completion_tokens"] != len(tokens)
        ):
            raise ProjectionError("completion usage differs from generated token count")

        logprob_fields = {
            "per_token_logprobs_ref",
            "per_token_logprobs_codec",
            "per_token_logprobs_shape",
        }
        present = logprob_fields & set(adapter)
        if present and present != logprob_fields:
            raise ProjectionError("logprobs require a reference, codec and shape")
        if present:
            shape = adapter["per_token_logprobs_shape"]
            if (
                "generated_token_ids" not in adapter
                or adapter["per_token_logprobs_codec"] != "f32-le"
                or not isinstance(shape, (tuple, list))
                or len(shape) != 1
                or type(shape[0]) is not int
                or shape[0] != len(tokens)
            ):
                raise ProjectionError("logprob shape is not aligned with generated tokens")
            try:
                logprobs = reader.bytes_artifact(adapter["per_token_logprobs_ref"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProjectionError("logprob byte artifact is unavailable") from exc
            if not isinstance(logprobs, bytes) or len(logprobs) != 4 * len(tokens):
                raise ProjectionError("logprob byte artifact has the wrong shape")
        elif "per_token_logprobs_ref" in adapter:
            # The wire codec also rejects this partial triplet; keep the decoder strict
            # for direct callers that provide an intentionally forged instance.
            raise ProjectionError("ref-only logprobs are not typed evidence")

    prepared = None
    if turn.prepared_request_ref is not None:
        try:
            body = reader.artifact(turn.prepared_request_ref)
            prepared = (
                body if isinstance(body, WriterRequestV1) else WriterRequestV1.from_dict(body)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectionError("prepared writer request cannot be decoded") from exc
        if (
            prepared.context_revision_ref != context.revision_ref
            or prepared.context_revision_ref != turn.context_revision_ref
            or prepared.payload_ref != turn.request_ref
        ):
            raise ProjectionError("prepared writer request is not bound to the sampled turn")
        if prepared.verified_messages:
            try:
                payload = reader.artifact(prepared.payload_ref)
            except (KeyError, TypeError, ValueError) as exc:
                raise ProjectionError("verified request payload is unavailable") from exc
            messages = [message.to_dict() for message in context.messages]
            if not isinstance(payload, Mapping) or canonical_bytes(
                payload.get("messages")
            ) != canonical_bytes(messages):
                raise ProjectionError("verified request messages differ from the active context")

    return BoundWriterTurnSamplingV1(turn, prepared)


def _decode_action_sampling(
    store, record, trace, action_id, context_hash, context_ref, rendering, message=None
) -> BoundSamplingV1:
    """Bind every duplicated sampling claim to the owning action and request."""
    evidence = SamplingEvidenceV1.from_wire(trace)
    if record.get("record_type") == "WriterActionV1":
        if set(record) != _ACTION_RECORD_FIELDS or message is None:
            raise ProjectionError("writer action record has wrong schema")
        if message.role != "assistant" or not message.loss_eligible or message.origin != action_id:
            raise ProjectionError("writer action message has wrong role or origin")
        expected_loss = {
            "assistant_text": any(part["type"] == "text" for part in message.content),
            "tool_syntax": any(part["type"] == "tool_call" for part in message.content),
            "assistant_ending": True,
            "system": False,
            "user": False,
            "author": False,
            "tool_observation": False,
            "seed": False,
            "environment": False,
        }
        if record["loss_eligibility"] != expected_loss:
            raise ProjectionError("writer action loss eligibility contradicts message")
    elif record.get("record_type") == "WriterSampledBudgetStopV1":
        if set(record) != _STOP_RECORD_FIELDS:
            raise ProjectionError("sampled stop record has wrong schema")
    else:
        raise ProjectionError("trace has no owning action or sampled stop")
    claims = {
        "action_id": action_id,
        "context_content_hash": context_hash,
        "context_revision_ref": context_ref,
        "exact_request_ref": record["request_ref"],
        "prepared_request_ref": record["prepared_request_ref"],
        "raw_output_ref": record["raw_output_ref"],
        "logprob_ref": record["logprob_ref"],
        "usage": record["usage"],
        "model": record["model"],
        "seed": record["seed"],
    }
    if record["action_id"] != action_id or any(
        trace.get(key) != value for key, value in claims.items()
    ):
        raise ProjectionError("writer trace contradicts its action")
    for key in ("request_ref", "raw_output_ref"):
        if record[key] is not None:
            store.get_artifact(record[key])
    if record["logprob_ref"] is not None:
        store.get_artifact(record["logprob_ref"], expected_domain="payload:bytes")
    if canonical_json(trace.get("rendering")) != canonical_json(rendering):
        raise ProjectionError("writer trace rendering differs from context")
    adapter = trace.get("adapter_trace")
    _validate_adapter_claims(store, adapter)
    if adapter is None and trace["logprob_ref"] is not None:
        raise ProjectionError("adapter trace lost its logprob reference")
    if adapter is not None and (
        not isinstance(adapter, dict)
        or any(key in adapter and adapter[key] != trace[key] for key in ("model", "seed", "usage"))
        or adapter.get("per_token_logprobs_ref") != trace["logprob_ref"]
        or ("per_token_logprobs_ref" in adapter) != (trace["logprob_ref"] is not None)
        or "native_on_policy_eligible" in adapter
    ):
        raise ProjectionError("adapter trace contradicts sampling claims")
    if trace.get("raw_output_evidence") != (
        "supplied" if record["raw_output_ref"] is not None else "missing"
    ):
        raise ProjectionError("raw output evidence contradicts reference")
    if trace.get("logprob_evidence") != (
        "supplied" if record["logprob_ref"] is not None else "missing"
    ):
        raise ProjectionError("logprob evidence contradicts reference")
    if trace.get("token_evidence") != (
        "supplied" if isinstance(adapter, dict) and "generated_token_ids" in adapter else "missing"
    ):
        raise ProjectionError("token evidence contradicts adapter trace")
    prepared_ref = record["prepared_request_ref"]
    prepared = None
    if prepared_ref is not None:
        prepared = _decode_and_bind_prepared_request(
            store, prepared_ref, context_hash, context_ref, rendering, record["request_ref"]
        )
    return BoundSamplingV1(
        SamplingRecordV1.from_wire(record), evidence, prepared, evidence.eligibility
    )


def _decode_and_bind_prepared_request(
    store, ref, context_hash, context_ref, rendering, payload_ref
):
    prepared = PreparedRequestV1.from_wire(store.get_artifact(ref, expected_domain="payload"))
    prepared.bind(store, context_hash, context_ref, rendering, payload_ref)
    return prepared
