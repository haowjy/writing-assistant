"""Writer-turn codec helpers and sampling policy shared by producers and derives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from writing_agent.task_graph import canonical_bytes
from writing_agent.task_graph_context_roots import context_root_changed_after
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_native_contracts import require_native_manifest_binding
from writing_agent.task_graph_record_contracts import GroupSpecV1
from writing_agent.task_graph_records import (
    RECORD_TYPES,
    RendererDescriptorV1,
    RuntimeManifestV1,
    RuntimeManifestV2,
    WriterTurnV1,
    WriterTurnV2,
    decode_runtime_manifest,
)
from writing_agent.task_graph_token_ledger import decode_u32_token_ids


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


def bind_group_sampling_claims(
    policy: Mapping[str, str],
    writer_seed: int,
    claims: Mapping[str, Any],
    *,
    model_id: str,
    require_context_claims: bool = True,
) -> None:
    """Compare present adapter claims with the member's sealed sampling policy."""
    if not isinstance(claims, Mapping):
        raise ProjectionError("group writer sample must include adapter claims")
    context_claims = {"context_revision_ref", "context_content_hash", "rendering"}
    if require_context_claims and not context_claims <= claims.keys():
        raise ProjectionError("group writer sample omits active context claims")
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


def sampling_usage_requirements(budget: Mapping[str, Any]) -> frozenset[str]:
    """Return usage fields required by the active token-budget limits."""
    limits = budget["limits"]
    fields: set[str] = set()
    if "generated_tokens" in limits:
        fields.add("completion_tokens")
    if "total_tokens" in limits:
        fields.add("total_tokens")
    if "context_tokens" in limits:
        fields.update(("prompt_tokens", "completion_tokens"))
    return frozenset(fields)


def decode_and_bind_sampling(
    turn: WriterTurnV1 | WriterTurnV2,
    context: Any,
    group: GroupSpecV1 | None,
    samples: tuple[Any, ...],
    head_event_id: str | None,
    budget: Mapping[str, Any],
    lineage_id: str,
    reader: Any,
) -> RuntimeManifestV1 | RuntimeManifestV2 | None:
    """Dispatch sampling evidence by record type and bind it to its pinned context."""
    record_type = getattr(turn, "RECORD_TYPE", None)
    if record_type == WriterTurnV1.RECORD_TYPE and isinstance(turn, WriterTurnV1):
        manifest = _sampling_manifest(group, reader) if group is not None else None
        if isinstance(manifest, RuntimeManifestV2):
            raise ProjectionError("input.record_type: WriterTurnV1 cannot use RuntimeManifestV2")
        _decode_v1_sampling(turn, context, reader)
        return manifest
    if record_type != WriterTurnV2.RECORD_TYPE or not isinstance(turn, WriterTurnV2):
        raise ProjectionError("input.record_type: sampling input is not a writer-turn record")
    if group is None:
        raise ProjectionError("input.record_type: WriterTurnV2 requires a sealed V2 manifest")
    manifest = _sampling_manifest(group, reader)
    if not isinstance(manifest, RuntimeManifestV2):
        raise ProjectionError("input.record_type: WriterTurnV2 requires RuntimeManifestV2")
    _decode_v2_sampling(
        turn, context, group, manifest, samples, head_event_id, budget, lineage_id, reader
    )
    return manifest


def _sampling_manifest(
    group: GroupSpecV1, reader: Any
) -> RuntimeManifestV1 | RuntimeManifestV2 | None:
    try:
        body = reader.artifact(group.policy["adapter_ref"])
        if body.get("record_type") not in {
            RuntimeManifestV1.RECORD_TYPE,
            RuntimeManifestV2.RECORD_TYPE,
        }:
            if group.runner_mode == "fixture" and group.training_mode is None:
                return None
            raise ValueError("sealed group artifact is not a runtime manifest")
        return decode_runtime_manifest(body)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ProjectionError(
            "input.sampling_pins.manifest_ref: sealed manifest is invalid"
        ) from exc


def _decode_v1_sampling(turn: WriterTurnV1, context: Any, reader: Any) -> None:
    """Bind the existing V1 trace shape to its active context."""
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


def _decode_v2_sampling(
    turn: WriterTurnV2,
    context: Any,
    group: GroupSpecV1,
    manifest: RuntimeManifestV2,
    samples: tuple[Any, ...],
    head_event_id: str | None,
    budget: Mapping[str, Any],
    lineage_id: str,
    reader: Any,
) -> None:
    """Check V2 evidence, pin identity, prior-token chaining and derived termination."""
    if turn.context_revision_ref != context.revision_ref:
        raise ProjectionError("input.context_revision_ref: writer turn differs from active context")
    trace = turn.adapter_trace
    if trace is not None:
        for field, expected in (
            ("context_revision_ref", context.revision_ref),
            ("context_content_hash", context.content_ref),
            ("rendering", context.rendering),
        ):
            if field in trace and canonical_bytes(trace[field]) != canonical_bytes(expected):
                raise ProjectionError(f"input.adapter_trace.{field}: differs from active context")

    usage = turn.usage
    if usage.get("prompt_tokens") != turn.input_token_count:
        raise ProjectionError("input.input_token_count: differs from usage.prompt_tokens")
    if usage.get("completion_tokens") != turn.generated_token_count:
        raise ProjectionError("input.generated_token_count: differs from usage.completion_tokens")
    total_tokens = turn.input_token_count + turn.generated_token_count
    if "total_tokens" in usage and usage["total_tokens"] != total_tokens:
        raise ProjectionError("input.usage.total_tokens: differs from token ledger counts")
    if usage.get("prefill_tokens") != turn.input_token_count:
        raise ProjectionError("input.usage.prefill_tokens: differs from input_token_count")
    if usage.get("cached_input_tokens") != 0:
        raise ProjectionError("input.usage.cached_input_tokens: native prefill cannot be cached")
    if turn.logprobs["shape"] != (turn.generated_token_count,):
        raise ProjectionError("input.logprobs.shape: differs from generated_token_count")

    input_ids = _read_token_ids(
        reader, turn.input_token_ids_ref, turn.input_token_count, path="input.input_token_ids_ref"
    )
    generated_ids = _read_token_ids(
        reader,
        turn.generated_token_ids_ref,
        turn.generated_token_count,
        path="input.generated_token_ids_ref",
    )
    try:
        logprobs = reader.bytes_artifact(turn.logprobs["ref"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("input.logprobs.ref: unavailable") from exc
    if not isinstance(logprobs, bytes) or len(logprobs) != 4 * turn.generated_token_count:
        raise ProjectionError("input.logprobs.ref: byte shape differs from generated_token_count")

    member = next((item for item in group.members if item.member_id == lineage_id), None)
    if member is None:
        raise ProjectionError("input.sampling_pins.seed: writer lineage is not a group member")
    expected_pins = {
        "manifest_ref": group.policy["adapter_ref"],
        "behavior_policy_ref": group.policy["behavior_policy_ref"],
        "decoding_ref": group.policy["decoding_ref"],
        "renderer_ref": manifest.renderer.identity(),
        "seed": member.writer_seed,
    }
    for field, expected in expected_pins.items():
        if turn.sampling_pins[field] != expected:
            raise ProjectionError(f"input.sampling_pins.{field}: differs from sealed policy")
    require_native_manifest_binding(
        manifest,
        group.policy["adapter_ref"],
        policy=group.policy,
        rendering=context.rendering,
        require_capabilities=group.training_mode == "native",
    )

    prior_sample = next(iter(reversed(samples)), None)
    context_changed = False
    if prior_sample is not None:
        try:
            context_changed = context_root_changed_after(
                reader, head_event_id, prior_sample.event_id
            )
        except ProjectionError as exc:
            raise ProjectionError(
                "input.input_token_ids_ref: prior sample event ancestry is invalid"
            ) from exc
    if prior_sample is not None and not context_changed:
        previous_turn = _read_previous_turn(reader, prior_sample.turn_ref)
        if not isinstance(previous_turn, WriterTurnV2):
            raise ProjectionError(
                "input.record_type: V2 sampling follows a V1 turn on the same root"
            )
        previous_input = _read_token_ids(
            reader,
            previous_turn.input_token_ids_ref,
            previous_turn.input_token_count,
            path="input.input_token_ids_ref",
        )
        previous_generated = _read_token_ids(
            reader,
            previous_turn.generated_token_ids_ref,
            previous_turn.generated_token_count,
            path="input.generated_token_ids_ref",
        )
        expected_prefix = previous_input + previous_generated
        if input_ids[: len(expected_prefix)] != expected_prefix:
            raise ProjectionError("input.input_token_ids_ref: prior generated prefix differs")

    allowed, limit = _allowed_tokens(turn, manifest, budget)
    _validate_termination(turn, generated_ids, manifest, allowed, limit)


def _read_token_ids(reader: Any, ref: str, count: int, *, path: str) -> tuple[int, ...]:
    try:
        data = reader.bytes_artifact(ref)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectionError(f"{path}: unavailable") from exc
    try:
        return decode_u32_token_ids(data, count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectionError(f"{path}: byte count differs from declared token count") from exc


def _read_previous_turn(reader: Any, ref: str) -> WriterTurnV1 | WriterTurnV2:
    try:
        body = reader.artifact(ref)
        record_type = body.get("record_type")
        codec = RECORD_TYPES.get(record_type)
        if codec in (WriterTurnV1, WriterTurnV2):
            return codec.from_dict(body)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ProjectionError("input.input_token_ids_ref: previous turn is invalid") from exc
    raise ProjectionError("input.record_type: previous sample is not a writer turn")


def _allowed_tokens(
    turn: WriterTurnV2, manifest: RuntimeManifestV2, budget: Mapping[str, Any]
) -> tuple[int, str]:
    limits = budget["limits"]
    consumed = budget["consumed"]
    generated_limit = limits.get("generated_tokens")
    context_limit = limits.get("context_tokens")
    if generated_limit is not None and type(generated_limit) is not int:
        raise ProjectionError("state.budgets_ref.limits.generated_tokens: invalid committed limit")
    generated_consumed = consumed.get("generated_tokens", 0)
    if type(generated_consumed) is not int:
        raise ProjectionError("state.budgets_ref.consumed.generated_tokens: invalid counter")
    if context_limit is not None and type(context_limit) is not int:
        raise ProjectionError("state.budgets_ref.limits.context_tokens: invalid committed limit")
    context_remaining = (
        float("inf") if context_limit is None else context_limit - turn.input_token_count
    )
    generated_remaining = (
        float("inf") if generated_limit is None else generated_limit - generated_consumed
    )
    candidates = (
        ("decision", manifest.decoding.max_tokens_per_decision),
        ("generated_budget", generated_remaining),
        ("context", context_remaining),
    )
    allowed = min(value for _, value in candidates)
    limiting_term = next(name for name, value in candidates if value == allowed)
    return allowed, limiting_term


def _validate_termination(
    turn: WriterTurnV2,
    generated_ids: tuple[int, ...],
    manifest: RuntimeManifestV2,
    allowed: int,
    limiting_term: str,
) -> None:
    termination = turn.termination
    kind = termination["kind"]
    stop_ids = frozenset(manifest.renderer.stop_token_ids)
    has_stop = any(token in stop_ids for token in generated_ids)
    if kind == "native_stop":
        last_is_stop = bool(generated_ids) and generated_ids[-1] in stop_ids
        if not (
            1 <= len(generated_ids) <= allowed
            and last_is_stop
            and termination["stop_token_id"] == generated_ids[-1]
            and not any(token in stop_ids for token in generated_ids[:-1])
        ):
            raise ProjectionError("input.termination: native_stop does not match token counts")
        return
    if kind == "token_limit":
        if not (
            len(generated_ids) == allowed >= 1
            and not has_stop
            and termination["limit"] == limiting_term
        ):
            raise ProjectionError("input.termination: token_limit does not match token counts")
        return
    if kind == "context_limit" and not (
        allowed <= 0 and limiting_term == "context" and not generated_ids
    ):
        raise ProjectionError("input.termination: context_limit does not match token counts")
    if kind == "context_limit":
        if turn.raw_output_ref is not None:
            raise ProjectionError("input.raw_output_ref: zero-generation context_limit has output")
        if turn.message.content not in (None, "") or turn.message.calls:
            raise ProjectionError("input.message: zero-generation context_limit must be empty")
    if kind != "context_limit":
        raise ProjectionError("input.termination.kind: unsupported termination")


_TOKEN_LIMIT_STOP_REASONS = {
    "decision": "decision_token_limit",
    "generated_budget": "generated_tokens_budget",
    "context": "context_tokens_budget",
}
TERMINATION_STOP_REASONS = frozenset(
    {
        *_TOKEN_LIMIT_STOP_REASONS.values(),
        "unterminated_tool_call",
        "unterminated_final_answer",
        "unparsed_tool_call",
    }
)


def termination_stop_reason(turn: WriterTurnV2, renderer: RendererDescriptorV1) -> str | None:
    """Map a validated native termination and sampled message to its writer outcome."""
    termination = turn.termination
    kind = termination["kind"]
    if kind == "token_limit":
        return _TOKEN_LIMIT_STOP_REASONS[termination["limit"]]
    if kind == "context_limit":
        return "context_tokens_budget"
    if turn.native_parse_failed is True:
        return "unparsed_tool_call"
    has_tool_calls = turn.message.tool_calls_was_list and bool(turn.message.calls)
    tool_response_stop = renderer.tool_response_stop_token_id
    if has_tool_calls and termination["stop_token_id"] != tool_response_stop:
        return "unterminated_tool_call"
    if not has_tool_calls and termination["stop_token_id"] == tool_response_stop:
        return "unterminated_final_answer"
    return None
