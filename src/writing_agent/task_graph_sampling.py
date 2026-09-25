"""Versioned sampling wire codecs, evidence binding, and eligibility policy.

The V1 wire dictionaries are retained byte-for-byte for approved stored identities.
Typed values are immutable; decoding is the only place that interprets duplicated
sampling fields. Opaque adapter claims never establish native token eligibility.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from writing_agent.task_graph import canonical_bytes, canonical_json, validate_hash


class ProjectionError(ValueError):
    """A causal event chain cannot be safely rendered as writer context."""


class VerifiedMessagesStaleError(ProjectionError):
    """A verified request no longer describes the active visible messages."""


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


def decode_training_eligibility(value: Any, outcome_ref: str) -> EligibilityDecisionV1:
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


def bind_group_sampling_claims(
    policy: Mapping[str, str], writer_seed: int, trace: Mapping[str, Any], claims: Any
) -> None:
    """Apply sealed group policy only to claims an adapter actually made."""
    if not isinstance(claims, dict):
        return
    for field in (
        "model_ref",
        "behavior_policy_ref",
        "tokenizer_ref",
        "template_ref",
        "adapter_ref",
        "decoding_ref",
        "context_policy_ref",
    ):
        if field in claims and claims[field] != policy[field]:
            raise ProjectionError(f"writer sample used a different {field}")
    if "policy_ref" in claims and claims["policy_ref"] != policy["behavior_policy_ref"]:
        raise ProjectionError("writer sample used a different behavior policy")
    for field, expected in (
        ("seed", writer_seed),
        ("model", trace["model"]),
        ("context_content_hash", trace["context_content_hash"]),
        ("context_revision_ref", trace["context_revision_ref"]),
        ("rendering", trace["rendering"]),
    ):
        if field in claims and canonical_bytes(claims[field]) != canonical_bytes(expected):
            raise ProjectionError(f"writer sample request/adapter changed {field}")


def decode_and_bind_sampling(
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
