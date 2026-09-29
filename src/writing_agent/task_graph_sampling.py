"""Writer-turn codec helpers and sampling policy shared by producers and derives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from writing_agent.task_graph import canonical_bytes
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_records import WriterTurnV1

NATIVE_TRACE_REASON = "native token alignment and loss masks are not implemented in Phase 4"


class ArtifactSink(Protocol):
    """The two artifact writes needed by sampling adapters."""

    def put_artifact(self, value: Any, *, private: bool = False) -> str: ...

    def put_bytes_artifact(self, value: bytes) -> str: ...


def persist_logprob_trace(
    store: ArtifactSink, result: Any, trace: dict[str, Any]
) -> dict[str, Any]:
    """Validate and persist aligned binary logprobs into their shared trace shape."""
    if result.logprobs is None:
        return trace
    if "per_token_logprobs_ref" in trace or "per_token_logprobs" in trace:
        raise ValueError("sample supplied duplicate logprob evidence")
    tokens = trace.get("generated_token_ids")
    if (
        not isinstance(tokens, list)
        or any(type(token) is not int or token < 0 for token in tokens)
        or len(tokens) != result.logprobs.shape[0]
    ):
        raise ValueError("binary logprobs do not align with sampled tokens")
    trace["per_token_logprobs_ref"] = store.put_bytes_artifact(result.logprobs.data)
    trace["per_token_logprobs_codec"] = result.logprobs.codec
    trace["per_token_logprobs_shape"] = list(result.logprobs.shape)
    return trace


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


def bind_group_sampling_claims(
    policy: Mapping[str, str],
    writer_seed: int,
    claims: Mapping[str, Any],
    *,
    model_id: str,
) -> None:
    """Compare present adapter claims with the member's sealed sampling policy."""
    if not isinstance(claims, Mapping):
        return
    for field, expected in policy.items():
        if field in claims and canonical_bytes(claims[field]) != canonical_bytes(expected):
            raise ProjectionError(f"writer sample used a different {field}")
    if "policy_ref" in claims and canonical_bytes(claims["policy_ref"]) != canonical_bytes(
        policy["behavior_policy_ref"]
    ):
        raise ProjectionError("writer sample used a different behavior policy")
    for field, expected in (("seed", writer_seed), ("model", model_id)):
        if field in claims and canonical_bytes(claims[field]) != canonical_bytes(expected):
            raise ProjectionError(f"writer sample request/adapter changed {field}")


def decode_writer_turn_sampling(turn: WriterTurnV1, context: Any, reader: Any) -> None:
    """Bind a typed writer turn and its trace claims to the active context."""
    if not isinstance(turn, WriterTurnV1):
        raise ProjectionError("input.record_type: sampling input is not a WriterTurnV1")
    if turn.context_revision_ref != context.revision_ref:
        raise ProjectionError(
            "input.context_revision_ref: writer turn differs from the active context"
        )

    adapter = turn.adapter_trace
    if adapter is not None:
        if not isinstance(adapter, Mapping) or "native_on_policy_eligible" in adapter:
            raise ProjectionError("input.adapter_trace: adapter may not claim native eligibility")
        for field, expected in (
            ("context_revision_ref", context.revision_ref),
            ("context_content_hash", context.content_ref),
            ("rendering", context.rendering),
        ):
            if field in adapter and canonical_bytes(adapter[field]) != canonical_bytes(expected):
                raise ProjectionError(f"input.adapter_trace.{field}: differs from active context")
        tokens_present = "generated_token_ids" in adapter
        tokens = adapter.get("generated_token_ids")
        if tokens_present and (
            type(turn.usage.get("completion_tokens")) is not int
            or turn.usage["completion_tokens"] != len(tokens)
        ):
            raise ProjectionError("input.usage.completion_tokens: token count mismatch")

        logprob_fields = {
            "per_token_logprobs_ref",
            "per_token_logprobs_codec",
            "per_token_logprobs_shape",
        }
        present = logprob_fields & set(adapter)
        if present and present != logprob_fields:
            raise ProjectionError("input.adapter_trace.per_token_logprobs_ref: triplet required")
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
                raise ProjectionError("input.adapter_trace.per_token_logprobs_shape: not aligned")
            try:
                logprobs = reader.bytes_artifact(adapter["per_token_logprobs_ref"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProjectionError(
                    "input.adapter_trace.per_token_logprobs_ref: unavailable"
                ) from exc
            if not isinstance(logprobs, bytes) or len(logprobs) != 4 * len(tokens):
                raise ProjectionError(
                    "input.adapter_trace.per_token_logprobs_ref: byte shape differs"
                )
