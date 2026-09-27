"""Strict wire codecs for the transition-seam task-graph runtime.

These records are deliberately separate from the legacy task-graph runtime.  They
provide canonical, closed decoders and an explicit reference-edge registry; no
producer uses them until the transition core is introduced.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, ClassVar

from writing_agent.task_graph import (
    MessageV1,
    _logical_id,
    _utf8,
    canonical_bytes,
    canonical_json,
    domain_hash,
    load_canonical_json,
    safe_path,
    validate_hash,
)
from writing_agent.task_graph_contracts import EXECUTION_STATUSES, TASK_STATUSES

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NONCANONICAL_MARKERS = {
    "float": {"$noncanonical", "repr"},
    "bytes": {"$noncanonical", "hex"},
    "tuple": {"$noncanonical", "items"},
    "mapping": {"$noncanonical", "items"},
    "surrogate-string": {"$noncanonical", "codepoints"},
    "cycle": {"$noncanonical"},
    "limit": {"$noncanonical"},
    "unsupported": {"$noncanonical", "type"},
}
_TRACE_REF_KEYS = (
    "model_ref",
    "behavior_policy_ref",
    "tokenizer_ref",
    "template_ref",
    "adapter_ref",
    "decoding_ref",
    "context_policy_ref",
    "policy_ref",
)
_TRACE_LOGPROB_KEYS = frozenset(
    {
        "per_token_logprobs_ref",
        "per_token_logprobs_codec",
        "per_token_logprobs_shape",
    }
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _wire_value(value: Any) -> None:
    """Require JSON-origin containers and reject floats before value equality."""
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("wire object keys must be strings")
            _wire_value(item)
    elif type(value) is list:
        for item in value:
            _wire_value(item)
    elif type(value) in (str, int, bool) or value is None:
        return
    else:
        raise TypeError("wire values must use canonical JSON types")


def _json_value(value: Any) -> Any:
    if isinstance(value, _WireRecord):
        return value.to_wire()
    if isinstance(value, MessageV1):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _object(value: Any, label: str, *, required: set[str], optional: set[str] = frozenset()):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)}")
    return value


def _string(value: Any, label: str, *, nonempty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    _utf8(value, label)
    if nonempty and not value:
        raise ValueError(f"{label} must be nonempty")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} is below its minimum")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    return value


def _array(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be an array")
    return tuple(value)


def _canonical_value(value: Any) -> None:
    """Validate the intake codec's JSON value and noncanonical marker forms."""
    if value is None or type(value) in (str, int, bool):
        if type(value) is str:
            _utf8(value)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _canonical_value(item)
        return
    if not isinstance(value, Mapping):
        raise TypeError("value is not a canonical intake value")
    if any(type(key) is not str for key in value):
        raise TypeError("canonical object keys must be strings")
    if "$noncanonical" in value:
        marker = value["$noncanonical"]
        if type(marker) is not str or marker not in _NONCANONICAL_MARKERS:
            raise ValueError("unknown or forged noncanonical marker")
        if set(value) != _NONCANONICAL_MARKERS[marker]:
            raise ValueError("noncanonical marker has the wrong key set")
        if marker == "float":
            _string(value["repr"], "float marker repr")
        elif marker == "bytes":
            _string(value["hex"], "bytes marker hex")
            if len(value["hex"]) % 2 or any(c not in "0123456789abcdef" for c in value["hex"]):
                raise ValueError("bytes marker must contain lowercase hexadecimal")
        elif marker in {"tuple", "mapping"}:
            items = _array(value["items"], "marker items")
            if marker == "tuple":
                for item in items:
                    _canonical_value(item)
            else:
                for pair in items:
                    pair = _array(pair, "mapping item")
                    if len(pair) != 2:
                        raise ValueError("mapping item must be a key/value pair")
                    _canonical_value(pair[0])
                    _canonical_value(pair[1])
        elif marker == "surrogate-string":
            points = _array(value["codepoints"], "surrogate codepoints")
            for point in points:
                _integer(point, "surrogate codepoint", minimum=0)
                if point > 0x10FFFF:
                    raise ValueError("surrogate codepoint is outside Unicode")
        elif marker == "unsupported":
            _string(value["type"], "unsupported type", nonempty=True)
        return
    for item in value.values():
        _canonical_value(item)


def _json_shape(value: Any) -> None:
    """Validate ordinary canonical JSON without intake-marker interpretation."""
    if value is None or type(value) in (str, int, bool):
        if type(value) is str:
            _utf8(value)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _json_shape(item)
        return
    if not isinstance(value, Mapping):
        raise TypeError("value is not canonical JSON")
    for key, item in value.items():
        _string(key, "canonical object key")
        _json_shape(item)


def _message(value: Any) -> MessageV1:
    if isinstance(value, MessageV1):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("context messages must be message records")
    return MessageV1.from_dict(dict(value))


def _schema(value: Any) -> int:
    if type(value) is not int or value != 1:
        raise ValueError("unsupported wire schema")
    return value


@dataclass(frozen=True)
class _WireRecord:
    """Base for strict, frozen wire values with optional payload record types."""

    RECORD_TYPE: ClassVar[str | None] = None
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType({})

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(self, item.name, _freeze(getattr(self, item.name)))
        # This walk also rejects noncanonical floats or unsupported Python values.
        canonical_bytes(self.to_wire())
        self.validate()

    def validate(self) -> None:
        pass

    def to_wire(self) -> dict[str, Any]:
        body = {item.name: _json_value(getattr(self, item.name)) for item in fields(self)}
        if self.RECORD_TYPE is not None:
            body = {"record_type": self.RECORD_TYPE, **body}
        return body

    def to_dict(self) -> dict[str, Any]:
        """Match the value-record serialization API while naming its wire clearly."""
        return self.to_wire()

    def to_json(self) -> str:
        return canonical_json(self.to_wire())

    def identity(self) -> str:
        if self.RECORD_TYPE is None:
            raise TypeError(f"{type(self).__name__} does not have a record identity")
        return domain_hash("payload", self.to_wire())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if type(value) is not dict:
            raise TypeError("wire record must be a plain object")
        _wire_value(value)
        expected = {item.name for item in fields(cls)}
        if cls.RECORD_TYPE is not None:
            expected.add("record_type")
            if value.get("record_type") != cls.RECORD_TYPE:
                raise ValueError("record_type does not match codec")
        missing, unknown = expected - set(value), set(value) - expected
        if missing:
            raise ValueError(f"record is missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(f"record has unknown fields: {sorted(unknown)}")
        kwargs = {item.name: value[item.name] for item in fields(cls)}
        record = cls(**kwargs)
        if canonical_bytes(record.to_wire()) != canonical_bytes(value):
            raise ValueError("record did not preserve its canonical wire value")
        return record

    @classmethod
    def from_json(cls, data: str | bytes):
        raw = data.encode("utf-8", "strict") if isinstance(data, str) else data
        if not isinstance(raw, bytes):
            raise TypeError("JSON input must be text or bytes")
        record = cls.from_dict(load_canonical_json(raw))
        if canonical_bytes(record.to_wire()) != raw:
            raise ValueError("decoded record did not preserve canonical bytes")
        return record


@dataclass(frozen=True)
class SampledMessageV1(_WireRecord):
    content: Any
    tool_calls_was_list: bool
    calls: Any

    def validate(self) -> None:
        _canonical_value(self.content)
        _boolean(self.tool_calls_was_list, "tool_calls_was_list")
        if self.tool_calls_was_list:
            for call in _array(self.calls, "calls"):
                call = _object(call, "call", required={"bounded", "value"})
                bounded = _boolean(call["bounded"], "bounded")
                if bounded:
                    _canonical_value(call["value"])
                elif call["value"] != {"$noncanonical": "bounded-call"}:
                    raise ValueError("unbounded calls require the bounded-call marker")
        else:
            _canonical_value(self.calls)


@dataclass(frozen=True)
class WriterTurnV1(_WireRecord):
    action_id: str
    context_revision_ref: str
    request_ref: str | None
    prepared_request_ref: str | None
    raw_output_ref: str | None
    usage: Mapping[str, Any]
    adapter_trace: Mapping[str, Any] | None
    message: SampledMessageV1 | Mapping[str, Any]
    RECORD_TYPE: ClassVar[str] = "WriterTurnV1"
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "context_revision_ref": "context_revision",
            "request_ref": "artifact|bytes",
            "prepared_request_ref": "artifact",
            "raw_output_ref": "artifact|bytes",
            "adapter_trace.per_token_logprobs_ref": "bytes",
            **{f"adapter_trace.{key}": "artifact" for key in _TRACE_REF_KEYS},
        }
    )

    def __post_init__(self) -> None:
        if isinstance(self.message, Mapping):
            object.__setattr__(self, "message", SampledMessageV1.from_dict(dict(self.message)))
        super().__post_init__()

    def validate(self) -> None:
        _logical_id(self.action_id, "action id")
        validate_hash(self.context_revision_ref)
        validate_hash(self.request_ref, optional=True)
        validate_hash(self.prepared_request_ref, optional=True)
        validate_hash(self.raw_output_ref, optional=True)
        if not isinstance(self.usage, Mapping):
            raise TypeError("usage must be an object")
        usage = self.usage
        for key, value in usage.items():
            _string(key, "usage key", nonempty=True)
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}:
                _integer(value, f"usage.{key}")
            else:
                _json_shape(value)
        if self.adapter_trace is not None:
            if not isinstance(self.adapter_trace, Mapping):
                raise TypeError("adapter_trace must be an object")
            trace = self.adapter_trace
            if "per_token_logprobs" in trace or "native_on_policy_eligible" in trace:
                raise ValueError("adapter trace contains a forbidden field")
            if "model" in trace:
                _string(trace["model"], "adapter_trace.model")
            if "seed" in trace:
                _integer(trace["seed"], "adapter_trace.seed", minimum=None)
            if "generated_token_ids" in trace:
                for token in _array(trace["generated_token_ids"], "generated_token_ids"):
                    _integer(token, "generated token id")
            for key in _TRACE_REF_KEYS:
                if key in trace:
                    validate_hash(trace[key])
            logprob_fields = _TRACE_LOGPROB_KEYS & set(trace)
            if logprob_fields and logprob_fields != _TRACE_LOGPROB_KEYS:
                raise ValueError("logprob reference, codec, and shape must appear together")
            if logprob_fields:
                validate_hash(trace["per_token_logprobs_ref"])
                if trace["per_token_logprobs_codec"] != "f32-le":
                    raise ValueError("unsupported logprob codec")
                for dimension in _array(trace["per_token_logprobs_shape"], "logprob shape"):
                    _integer(dimension, "logprob shape dimension")
            known = {
                "model",
                "seed",
                "generated_token_ids",
                *_TRACE_REF_KEYS,
                *_TRACE_LOGPROB_KEYS,
            }
            for key, value in trace.items():
                if key not in known:
                    _json_shape(value)
        if not isinstance(self.message, SampledMessageV1):
            raise TypeError("message must be SampledMessageV1")


@dataclass(frozen=True)
class WriterRequestV1(_WireRecord):
    context_revision_ref: str
    payload_ref: str
    verified_messages: bool
    RECORD_TYPE: ClassVar[str] = "WriterRequestV1"
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {"context_revision_ref": "context_revision", "payload_ref": "artifact|bytes"}
    )

    def validate(self) -> None:
        validate_hash(self.context_revision_ref)
        validate_hash(self.payload_ref)
        _boolean(self.verified_messages, "verified_messages")


@dataclass(frozen=True)
class ToolObservationV1(_WireRecord):
    call_id: str
    dispatch: Mapping[str, Any] | None
    RECORD_TYPE: ClassVar[str] = "ToolObservationV1"

    def validate(self) -> None:
        _logical_id(self.call_id, "call id")
        if self.dispatch is None:
            return
        dispatch = _object(self.dispatch, "dispatch", required={"spec", "observation", "effect"})
        spec = _object(
            dispatch["spec"], "dispatch spec", required={"max_file_bytes", "max_workspace_bytes"}
        )
        _integer(spec["max_file_bytes"], "max_file_bytes")
        _integer(spec["max_workspace_bytes"], "max_workspace_bytes")
        observation = _object(
            dispatch["observation"],
            "tool observation",
            required={"ok", "valid"},
            optional={"result", "error"},
        )
        _boolean(observation["ok"], "observation.ok")
        _boolean(observation["valid"], "observation.valid")
        if "result" in observation:
            _json_shape(observation["result"])
        if "error" in observation:
            _string(observation["error"], "observation.error")
        if not isinstance(dispatch["effect"], Mapping):
            raise TypeError("tool effect must be an object")
        for path, change in dispatch["effect"].items():
            safe_path(path)
            change = _object(change, "file effect", required={"before", "after"})
            for key in ("before", "after"):
                if change[key] is not None:
                    _string(change[key], f"effect.{key}")


@dataclass(frozen=True)
class AuthorReplyV1(_WireRecord):
    request_ref: str
    status: str
    utterance: str | None
    decision_ids: tuple[str, ...] | list[str]
    selected_proposals: Mapping[str, Any]
    RECORD_TYPE: ClassVar[str] = "AuthorReplyV1"
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType({"request_ref": "private"})

    def validate(self) -> None:
        validate_hash(self.request_ref)
        if self.status not in {"answered", "unsupported_coverage"}:
            raise ValueError("unsupported author reply status")
        if self.utterance is not None:
            _string(self.utterance, "utterance")
        if (self.status == "answered") != (self.utterance is not None):
            raise ValueError("utterance must be present only for an answered reply")
        decisions = _array(self.decision_ids, "decision_ids")
        for decision in decisions:
            _logical_id(decision, "decision id")
        if len(decisions) != len(set(decisions)):
            raise ValueError("decision ids must be unique")
        if not isinstance(self.selected_proposals, Mapping):
            raise TypeError("selected_proposals must be an object")
        for decision, proposals in self.selected_proposals.items():
            _logical_id(decision, "proposal decision id")
            if decision not in decisions:
                raise ValueError("proposal selection names an undisclosed decision")
            for proposal in _array(proposals, "selected proposals"):
                _logical_id(proposal, "proposal id")


@dataclass(frozen=True)
class EvaluatorResultV1(_WireRecord):
    request_ref: str
    status: str
    evidence_ref: str
    RECORD_TYPE: ClassVar[str] = "EvaluatorResultV1"
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {"request_ref": "private", "evidence_ref": "artifact"}
    )

    def validate(self) -> None:
        validate_hash(self.request_ref)
        validate_hash(self.evidence_ref)
        if self.status not in {"pass", "fail"}:
            raise ValueError("unsupported evaluator status")


@dataclass(frozen=True)
class ContextOperationInputV1(_WireRecord):
    policy_ref: str
    RECORD_TYPE: ClassVar[str] = "ContextOperationInputV1"
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType({"policy_ref": "artifact"})

    def validate(self) -> None:
        validate_hash(self.policy_ref)


@dataclass(frozen=True)
class EnvironmentStepV1(_WireRecord):
    directive: Mapping[str, Any]
    RECORD_TYPE: ClassVar[str] = "EnvironmentStepV1"

    def validate(self) -> None:
        if not isinstance(self.directive, Mapping):
            raise TypeError("directive must be an object")
        kind = self.directive.get("kind")
        shapes = {
            "request_author": ({"kind", "source"}, {"writer_request", "mandatory_feedback"}),
            "request_checks": ({"kind"}, None),
            "commit_transition": ({"kind", "edge_id"}, None),
            "seal_outcome": ({"kind", "task_status", "stop_reason"}, None),
            "stop_exhausted": ({"kind", "stop_reason"}, None),
            "publish_reward": ({"kind"}, None),
        }
        if kind not in shapes:
            raise ValueError("unsupported environment directive")
        required, choices = shapes[kind]
        directive = _object(self.directive, "directive", required=required)
        if kind == "request_author" and directive["source"] not in choices:
            raise ValueError("unsupported author request source")
        if kind == "commit_transition":
            _logical_id(directive["edge_id"], "edge id")
        if kind == "seal_outcome":
            if directive["task_status"] not in TASK_STATUSES:
                raise ValueError("unsupported task status")
            if directive["stop_reason"] is not None:
                _string(directive["stop_reason"], "stop reason")
        if kind == "stop_exhausted":
            _string(directive["stop_reason"], "stop reason", nonempty=True)


@dataclass(frozen=True)
class MemberStartV1(_WireRecord):
    group_spec_ref: str
    ordinal: int
    RECORD_TYPE: ClassVar[str] = "MemberStartV1"
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType({"group_spec_ref": "artifact"})

    def validate(self) -> None:
        validate_hash(self.group_spec_ref)
        _integer(self.ordinal, "ordinal")


@dataclass(frozen=True)
class ExternalInputsV1(_WireRecord):
    schema: int
    source_refs: tuple[str, ...] | list[str]
    RECORD_TYPE: ClassVar[str] = "ExternalInputsV1"
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType({"source_refs[]": "artifact"})

    def validate(self) -> None:
        _schema(self.schema)
        for identity in _array(self.source_refs, "source_refs"):
            validate_hash(identity)


@dataclass(frozen=True)
class AdmissionPolicyV1(_WireRecord):
    schema: int
    writer_family: str | None
    allowed_tools: tuple[str, ...] | list[str]
    controller_versions: tuple[str, ...] | list[str]
    check_versions: tuple[str, ...] | list[str]
    RECORD_TYPE: ClassVar[str] = "AdmissionPolicyV1"

    def validate(self) -> None:
        _schema(self.schema)
        if self.writer_family is not None:
            _string(self.writer_family, "writer_family", nonempty=True)
        for key in ("allowed_tools", "controller_versions", "check_versions"):
            values = _array(getattr(self, key), key)
            for value in values:
                _string(value, f"{key} item", nonempty=True)
            if tuple(values) != tuple(sorted(set(values))):
                raise ValueError(f"{key} must be sorted and unique")

    @classmethod
    def from_admission_policy(cls, policy: Any) -> AdmissionPolicyV1:
        return cls(
            schema=1,
            writer_family=policy.writer_family,
            allowed_tools=sorted(policy.allowed_tools),
            controller_versions=sorted(policy.controller_versions),
            check_versions=sorted(policy.check_versions),
        )


@dataclass(frozen=True)
class OutcomeV1(_WireRecord):
    schema: int
    task_status: str
    execution_status: str
    stop_reason: str | None
    reward_status: str
    training_eligibility: str
    candidate_checkpoint: str | None
    requirement_version: str | None
    checks: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]
    transition_edge_id: str | None
    failed_request_ref: str | None
    reward_ref: str | None
    eligibility_ref: str | None
    RECORD_TYPE: ClassVar[str] = "OutcomeV1"
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "candidate_checkpoint": "checkpoint",
            "requirement_version": "private",
            "checks[].request_ref": "private",
            "checks[].result_ref": "artifact",
            "failed_request_ref": "private",
            "reward_ref": "artifact",
            "eligibility_ref": "artifact",
        }
    )

    def validate(self) -> None:
        _schema(self.schema)
        if self.task_status not in TASK_STATUSES:
            raise ValueError("unsupported task status")
        if self.execution_status not in EXECUTION_STATUSES:
            raise ValueError("unsupported execution status")
        if self.reward_status not in {"pending", "available", "unavailable"}:
            raise ValueError("unsupported reward status")
        if self.training_eligibility not in {"pending", "eligible", "ineligible"}:
            raise ValueError("unsupported training eligibility")
        if self.stop_reason is not None:
            _string(self.stop_reason, "stop_reason")
        if self.transition_edge_id is not None:
            _logical_id(self.transition_edge_id, "transition edge id")
        for key in (
            "candidate_checkpoint",
            "requirement_version",
            "failed_request_ref",
            "reward_ref",
            "eligibility_ref",
        ):
            validate_hash(getattr(self, key), optional=True)
        for check in _array(self.checks, "checks"):
            check = _object(check, "outcome check", required={"request_ref", "result_ref"})
            validate_hash(check["request_ref"])
            validate_hash(check["result_ref"], optional=True)


@dataclass(frozen=True)
class ContextContentV1(_WireRecord):
    parent_ref: str | None
    messages: tuple[MessageV1 | Mapping[str, Any], ...] | list[MessageV1 | Mapping[str, Any]]
    tools: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None
    rendering: Mapping[str, Any] | None
    RECORD_TYPE: ClassVar[str] = None
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "parent_ref": "context_node",
            "rendering.template_ref": "artifact",
            "rendering.tokenizer_ref": "artifact",
            "rendering.tool_schema_ref": "artifact",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(_message(item) for item in self.messages))
        super().__post_init__()

    def validate(self) -> None:
        validate_hash(self.parent_ref, optional=True)
        for message in self.messages:
            if not isinstance(message, MessageV1):
                raise TypeError("messages must be MessageV1 records")
        is_root = self.parent_ref is None
        if is_root and (self.tools is None or self.rendering is None):
            raise ValueError("root content must carry tools and rendering")
        if not is_root and (self.tools is not None or self.rendering is not None):
            raise ValueError("root content alone must carry tools and rendering")
        if self.tools is not None:
            for tool in _array(self.tools, "tools"):
                if not isinstance(tool, Mapping):
                    raise TypeError("tool definitions must be objects")
        if self.rendering is not None:
            rendering = _object(
                self.rendering,
                "rendering",
                required={
                    "projection_version",
                    "prefix_id",
                    "template_ref",
                    "tokenizer_ref",
                    "tool_schema_ref",
                },
            )
            _string(rendering["projection_version"], "projection_version", nonempty=True)
            _logical_id(rendering["prefix_id"], "prefix id")
            for key in ("template_ref", "tokenizer_ref", "tool_schema_ref"):
                validate_hash(rendering[key])

    def identity(self) -> str:
        return domain_hash("context_content", self.to_wire())


@dataclass(frozen=True)
class ContextRevisionV1(_WireRecord):
    content_ref: str
    event_head: str | None
    provenance_refs: tuple[str, ...] | list[str]
    RECORD_TYPE: ClassVar[str] = None
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "content_ref": "context_node",
            "event_head": "event",
            "provenance_refs[]": "event",
        }
    )

    def validate(self) -> None:
        validate_hash(self.content_ref)
        validate_hash(self.event_head, optional=True)
        for identity in _array(self.provenance_refs, "provenance_refs"):
            validate_hash(identity)

    def identity(self) -> str:
        return domain_hash("context", self.to_wire())


RECORD_TYPES: Mapping[str, type[_WireRecord]] = MappingProxyType(
    {
        record.RECORD_TYPE: record
        for record in (
            WriterTurnV1,
            WriterRequestV1,
            ToolObservationV1,
            AuthorReplyV1,
            EvaluatorResultV1,
            ContextOperationInputV1,
            EnvironmentStepV1,
            MemberStartV1,
            ExternalInputsV1,
            OutcomeV1,
            AdmissionPolicyV1,
        )
    }
)

# Known legacy payload names remain opaque during coexistence. The old runtime
# does not publish records from RECORD_TYPES and its closure contract predates
# typed payload edges; unknown names still fail closed.
LEGACY_PAYLOAD_RECORD_TYPES = frozenset(
    {
        "AuthorRequestV1",
        "AuthorToolAckV1",
        "AuthorTurnV1",
        "CheckBatchV1",
        "CheckRequestV1",
        "CheckResultV1",
        "ContextOperationV1",
        "ContextPolicyV1",
        "DecisionDisclosureV1",
        "DecisionLedgerV1",
        "DeterministicCheckEvidenceV1",
        "DisclosureLedgerV1",
        "EvaluatorPacketV1",
        "FixtureFileCountEvidenceV1",
        "GroupExecutionFailureV1",
        "GroupAdvantageV1",
        "GroupDecisionV1",
        "GroupMemberSeedsV1",
        "GroupMemberResultV1",
        "GroupSegmentCreditV1",
        "GroupSpecV1",
        "GroupScriptedTerminalV1",
        "InfrastructureInvalidV1",
        "PreparedWriterRequestV1",
        "RequirementLedgerV1",
        "RequirementSupersessionV1",
        "RewardAvailabilityV1",
        "RewardPublicationV1",
        "RewardV1",
        "RuntimeManifestV1",
        "RuntimePortDescriptorV1",
        "ScriptCoverageFailureV1",
        "ScriptedAuthorReplyV1",
        "TerminalOutcomeCommitV1",
        "TerminalOutcomeV1",
        "TrainingEligibilityV1",
        "TransitionDecisionV1",
        "TranscriptReviewEvidenceV1",
        "VerifiedWriterMessagesV1",
        "WriterActionTraceV1",
        "WriterActionV1",
        "WriterExhaustedStopV1",
        "WriterRuntimeLogV1",
        "WriterSampledBudgetStopV1",
        "WriterToolResultV1",
    }
)

# Registry edge paths are deliberately separate from type dispatch: context
# records have closure kinds, not payload ``record_type`` fields.
RECORD_EDGES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        **{record_type: codec.REFS for record_type, codec in RECORD_TYPES.items()},
        "context_node": ContextContentV1.REFS,
        "context_revision": ContextRevisionV1.REFS,
    }
)
ALL_RECORD_CODECS: Mapping[str, type[_WireRecord]] = MappingProxyType(
    {
        **dict(RECORD_TYPES),
        "SampledMessageV1": SampledMessageV1,
        "context_node": ContextContentV1,
        "context_revision": ContextRevisionV1,
    }
)


def _path_values(record: _WireRecord, path: str) -> tuple[Any, ...]:
    values: tuple[Any, ...] = (record,)
    for segment in path.split("."):
        expand = segment.endswith("[]")
        key = segment[:-2] if expand else segment
        next_values = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, Mapping):
                child = value.get(key)
            else:
                child = getattr(value, key, None)
            if expand:
                if child is None:
                    continue
                next_values.extend(_array(child, key))
            else:
                next_values.append(child)
        values = tuple(next_values)
    return values


def record_reference_edges(
    record_type: str, body: Mapping[str, Any]
) -> tuple[tuple[str, str], ...]:
    """Decode one registered payload record and return its declared direct edges."""
    try:
        codec = RECORD_TYPES[record_type]
        ref_paths = RECORD_EDGES[record_type]
    except KeyError as exc:
        raise ValueError(f"unregistered task-graph record_type: {record_type!r}") from exc
    record = codec.from_dict(dict(body))
    references = []
    for path, kind in ref_paths.items():
        for identity in _path_values(record, path):
            if identity is None:
                continue
            validate_hash(identity)
            references.append((kind, identity))
    return tuple(references)


def context_reference_edges(record: _WireRecord) -> tuple[tuple[str, str], ...]:
    """Return closure edges for a new chained-context record."""
    kind = "context_node" if isinstance(record, ContextContentV1) else "context_revision"
    references = []
    for path, edge_kind in RECORD_EDGES[kind].items():
        for identity in _path_values(record, path):
            if identity is not None:
                validate_hash(identity)
                references.append((edge_kind, identity))
    return tuple(references)


@dataclass(frozen=True)
class MaterializedContextV1:
    """Read-only flattened view of a chained context revision."""

    messages: tuple[MessageV1, ...]
    tools: tuple[Mapping[str, Any], ...]
    rendering: Mapping[str, Any]


def materialize_context_nodes(
    revision: ContextRevisionV1, nodes: Mapping[str, ContextContentV1]
) -> MaterializedContextV1:
    """Materialize a verified revision from its already-loaded content nodes."""
    chunks: list[tuple[MessageV1, ...]] = []
    identity = revision.content_ref
    visited: set[str] = set()
    root: ContextContentV1 | None = None
    while True:
        if identity in visited:
            raise ValueError("context content chain contains a cycle")
        visited.add(identity)
        try:
            node = nodes[identity]
        except KeyError as exc:
            raise ValueError(f"missing context content node: {identity}") from exc
        chunks.append(node.messages)
        if node.parent_ref is None:
            root = node
            break
        identity = node.parent_ref
    assert root is not None and root.tools is not None and root.rendering is not None
    messages = tuple(message for chunk in reversed(chunks) for message in chunk)
    return MaterializedContextV1(messages, root.tools, root.rendering)


def is_sha256_string(value: Any) -> bool:
    """Small helper shared with the `REFS` completeness test."""
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None
