"""Strict wire-v1 records, payload registry, and chained-context materialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, ClassVar

import writing_agent.task_graph_group_records  # noqa: F401 - register group payload records
import writing_agent.task_graph_record_contracts  # noqa: F401 - populate the wire registry
from writing_agent import task_graph_errors
from writing_agent.task_graph import (
    EXECUTION_STATUSES,
    TASK_STATUSES,
    MaterializedContextV1,
    MessageV1,
    domain_hash,
    safe_path,
)
from writing_agent.task_graph_payloads import payload_record_codecs
from writing_agent.task_graph_wire import (
    Bool,
    CanonicalIntake,
    DictOf,
    Enum,
    Hash,
    Int,
    JsonValue,
    KindUnion,
    ListOf,
    MessageValue,
    PayloadCodec,
    RecordOf,
    Str,
    UnionOf,
    WireRecord,
    obj,
    obj_opt,
    record_class,
    registered_record_classes,
    validate_canonical_value,
    validate_codec,
)

_PAYLOAD_CODECS = payload_record_codecs()

_TRACE_REF_KEYS = (
    "model_ref behavior_policy_ref tokenizer_ref template_ref adapter_ref decoding_ref "
    "context_policy_ref policy_ref"
).split()
_TRACE_LOGPROB_KEYS = frozenset(
    "per_token_logprobs_ref per_token_logprobs_codec per_token_logprobs_shape".split()
)
_JSON = JsonValue()
_TEXT = Str()
_RENDERING_SCHEMA = obj(
    projection_version=Str(nonempty=True),
    prefix_id=Str(nonempty=True, logical=True),
    template_ref=Hash("artifact"),
    tokenizer_ref=Hash("artifact"),
    tool_schema_ref=Hash("artifact"),
)
_TRACE_SCHEMA = obj_opt(
    {},
    {
        "model": _TEXT,
        "seed": Int(minimum=None),
        "generated_token_ids": ListOf(Int()),
        **{key: Hash("artifact") for key in _TRACE_REF_KEYS},
        "context_revision_ref": Hash("context_revision"),
        "context_content_hash": Hash(None),
        "per_token_logprobs_ref": Hash("bytes"),
        "per_token_logprobs_codec": Enum(frozenset({"f32-le"})),
        "per_token_logprobs_shape": ListOf(Int()),
    },
    extra=_JSON,
)
_USAGE_SCHEMA = obj_opt(
    {},
    {key: Int() for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
    extra=_JSON,
)
NATIVE_RUNTIME_CAPABILITIES = frozenset(
    {"usage_reporting", "native_token_ledger", "sampled_logprobs"}
)
_NATIVE_USAGE_SCHEMA = obj_opt(
    {},
    {
        key: Int()
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prefill_tokens",
            "cached_input_tokens",
        )
    },
    extra=_JSON,
)
_V2_TRACE_SCHEMA = obj_opt(
    {},
    {
        key: item
        for key, item in _TRACE_SCHEMA.optional.items()
        if key
        not in {
            "generated_token_ids",
            "per_token_logprobs_ref",
            "per_token_logprobs_codec",
            "per_token_logprobs_shape",
        }
    },
    extra=_JSON,
)
_NATIVE_LOGPROBS_SCHEMA = obj(
    ref=Hash("bytes"),
    codec=Enum(frozenset({"f32-le"})),
    shape=ListOf(Int()),
)
_NATIVE_TERMINATION_SCHEMA = obj(
    kind=Enum(frozenset({"native_stop", "token_limit", "context_limit"})),
    stop_token_id=UnionOf((Int(), type(None))),
    limit=UnionOf((Enum(frozenset({"decision", "generated_budget", "context"})), type(None))),
)
_SAMPLING_PINS_SCHEMA = obj(
    manifest_ref=Hash("artifact"),
    behavior_policy_ref=Hash("artifact"),
    decoding_ref=Hash("artifact"),
    renderer_ref=Hash("artifact"),
    seed=UnionOf((Int(), type(None))),
)
_TOOL_DISPATCH = obj(
    spec=obj(max_file_bytes=Int(minimum=1), max_workspace_bytes=Int(minimum=1)),
    observation=obj_opt({"ok": Bool(), "valid": Bool()}, {"result": _JSON, "error": _TEXT}),
    effect=DictOf(obj(before=Str(optional=True), after=Str(optional=True))),
)


@dataclass(frozen=True)
class SampledMessageV1(WireRecord):
    content: Annotated[Any, CanonicalIntake()]
    tool_calls_was_list: Annotated[bool, Bool()]
    calls: Annotated[Any, _JSON]
    reasoning: Annotated[Any, CanonicalIntake()] = None
    thinking: Annotated[Any, CanonicalIntake()] = None
    reasoning_content: Annotated[Any, CanonicalIntake()] = None

    def check(self) -> None:
        if not self.tool_calls_was_list:
            validate_canonical_value(self.calls)
            return
        if not isinstance(self.calls, (tuple, list)):
            raise TypeError("calls must be an array")
        for call in self.calls:
            if not isinstance(call, Mapping) or set(call) != {"bounded", "value"}:
                raise ValueError("call must contain exactly bounded and value")
            if type(call["bounded"]) is not bool:
                raise TypeError("bounded must be a boolean")
            if call["bounded"]:
                validate_canonical_value(call["value"])
            elif call["value"] != {"$noncanonical": "bounded-call"}:
                raise ValueError("unbounded calls require the bounded-call marker")

    OMIT_NONE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"reasoning", "thinking", "reasoning_content"}
    )


@dataclass(frozen=True)
class WriterTurnV1(WireRecord):
    action_id: Annotated[str, Str(nonempty=True, logical=True)]
    context_revision_ref: Annotated[str, Hash("context_revision")]
    raw_output_ref: Annotated[str | None, Hash("artifact|bytes", optional=True)]
    usage: Annotated[Mapping[str, Any], _USAGE_SCHEMA]
    adapter_trace: Annotated[Mapping[str, Any] | None, UnionOf((_TRACE_SCHEMA, type(None)))]
    message: Annotated[SampledMessageV1 | Mapping[str, Any], RecordOf(SampledMessageV1)]
    RECORD_TYPE: ClassVar[str] = "WriterTurnV1"

    def check(self) -> None:
        if self.adapter_trace is None:
            return
        if {"per_token_logprobs", "native_on_policy_eligible"} & set(self.adapter_trace):
            raise ValueError("adapter trace contains a forbidden field")
        if any(
            key.endswith("_ref")
            and key not in {*_TRACE_REF_KEYS, "context_revision_ref", "per_token_logprobs_ref"}
            for key in self.adapter_trace
        ):
            raise ValueError("unknown adapter reference claim")
        present = _TRACE_LOGPROB_KEYS & set(self.adapter_trace)
        if present and present != _TRACE_LOGPROB_KEYS:
            raise ValueError("logprob reference, codec, and shape must appear together")


@dataclass(frozen=True)
class WriterTurnV2(WireRecord):
    """Tokenizer-free sampled turn evidence for the native training seam."""

    action_id: Annotated[str, Str(nonempty=True, logical=True)]
    context_revision_ref: Annotated[str, Hash("context_revision")]
    raw_output_ref: Annotated[str | None, Hash("artifact|bytes", optional=True)]
    usage: Annotated[Mapping[str, Any], _NATIVE_USAGE_SCHEMA]
    adapter_trace: Annotated[Mapping[str, Any] | None, UnionOf((_V2_TRACE_SCHEMA, type(None)))]
    message: Annotated[SampledMessageV1 | Mapping[str, Any], RecordOf(SampledMessageV1)]
    input_token_ids_ref: Annotated[str, Hash("bytes")]
    input_token_count: Annotated[int, Int()]
    generated_token_ids_ref: Annotated[str, Hash("bytes")]
    generated_token_count: Annotated[int, Int()]
    logprobs: Annotated[Mapping[str, Any], _NATIVE_LOGPROBS_SCHEMA]
    termination: Annotated[Mapping[str, Any], _NATIVE_TERMINATION_SCHEMA]
    sampling_pins: Annotated[Mapping[str, Any], _SAMPLING_PINS_SCHEMA]
    RECORD_TYPE: ClassVar[str] = "WriterTurnV2"

    def check(self) -> None:
        if len(self.logprobs["shape"]) != 1:
            raise ValueError("native logprob shape must be one-dimensional")
        if self.termination["kind"] == "native_stop":
            if self.termination["stop_token_id"] is None or self.termination["limit"] is not None:
                raise ValueError("native stop termination requires only a stop token")
        elif self.termination["stop_token_id"] is not None or self.termination["limit"] is None:
            raise ValueError("limit termination requires only a limit")
        if self.termination["kind"] == "context_limit" and self.termination["limit"] != "context":
            raise ValueError("context limit termination requires the context limit")
        if self.adapter_trace is not None and (
            {"generated_token_ids", *_TRACE_LOGPROB_KEYS, "native_on_policy_eligible"}
            & set(self.adapter_trace)
        ):
            raise ValueError("native adapter trace contains a forbidden claim")


RENDERER_STOP_TOKEN_ROLES = ("eos", "end_of_turn", "tool_response")


@dataclass(frozen=True)
class RendererDescriptorV1(WireRecord):
    implementation: Annotated[str, Str(nonempty=True)]
    template_ref: Annotated[str, Hash("artifact")]
    tokenizer_ref: Annotated[str, Hash("artifact")]
    tool_schema_ref: Annotated[str, Hash("artifact")]
    stop_token_ids: Annotated[tuple[int, ...] | list[int], ListOf(Int(), unique=True)]
    enable_thinking: Annotated[bool, Bool()]
    suffix_rules_version: Annotated[str, Str(nonempty=True)]
    RECORD_TYPE: ClassVar[str] = "RendererDescriptorV1"

    @property
    def tool_response_stop_token_id(self) -> int:
        """Return the renderer-pinned token ID assigned the tool-response role."""
        return self.stop_token_ids[RENDERER_STOP_TOKEN_ROLES.index("tool_response")]

    def check(self) -> None:
        if (
            self.implementation != "gemma4-native-append-v1"
            or self.stop_token_ids != (1, 106, 50)
            or self.enable_thinking
        ):
            raise ValueError("renderer descriptor differs from the pinned native renderer")


@dataclass(frozen=True)
class TokenizerDescriptorV1(WireRecord):
    model_id: Annotated[str, Str(nonempty=True)]
    revision: Annotated[str, Str(nonempty=True)]
    files_sha256: Annotated[Mapping[str, str], DictOf(Hash(None))]
    RECORD_TYPE: ClassVar[str] = "TokenizerDescriptorV1"

    def check(self) -> None:
        if not self.files_sha256 or any(not name for name in self.files_sha256):
            raise ValueError("tokenizer descriptor requires pinned files")


@dataclass(frozen=True)
class DecodingDescriptorV1(WireRecord):
    temperature: Annotated[int, Int(equals=1)]
    top_p: Annotated[int, Int(equals=1)]
    top_k: Annotated[int, Int(equals=0)]
    processors: Annotated[tuple[str, ...] | list[str], ListOf(Str(nonempty=True), unique=True)]
    max_tokens_per_decision: Annotated[int, Int(minimum=1)]
    seed_rule: Annotated[str, Str(nonempty=True)]
    logprob_convention: Annotated[str, Str(nonempty=True)]
    trainer_ratio: Annotated[str, Str(nonempty=True)]
    RECORD_TYPE: ClassVar[str] = "DecodingDescriptorV1"

    def check(self) -> None:
        if self.seed_rule != "writer_seed ⊕ action ordinal (sha256-domain-v1)":
            raise ValueError("unsupported decoding seed rule")
        if self.logprob_convention != "log_softmax(model logits after model softcap), fp32":
            raise ValueError("unsupported sampled-logprob convention")
        if self.trainer_ratio != "recomputed, num_iterations=1":
            raise ValueError("unsupported trainer ratio policy")


@dataclass(frozen=True)
class TrainingAdmissionV1(WireRecord):
    group_id: Annotated[str, Hash(None)]
    decision_ref: Annotated[str, Hash("artifact")]
    batch_ref: Annotated[str, Hash("artifact")]
    audit_version: Annotated[str, Str(nonempty=True)]
    renderer_ref: Annotated[str, Hash(None)]
    tokenizer_descriptor_ref: Annotated[str, Hash(None)]
    adapter_hash_before: Annotated[str, Hash(None)]
    adapter_hash_after: Annotated[str, Hash(None)]
    members: Annotated[
        tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
        ListOf(
            obj(
                member_id=Str(nonempty=True, logical=True),
                status=Enum(frozenset({"admitted", "refused"})),
                failed_check=UnionOf((Str(nonempty=True), type(None))),
            ),
            min_items=1,
        ),
    ]
    RECORD_TYPE: ClassVar[str] = "TrainingAdmissionV1"

    def check(self) -> None:
        ids = tuple(member["member_id"] for member in self.members)
        if len(ids) != len(set(ids)):
            raise ValueError("training admission members must be unique")
        if any(
            (member["status"] == "admitted") != (member["failed_check"] is None)
            for member in self.members
        ):
            raise ValueError("training admission failure must match member status")


@dataclass(frozen=True)
class ToolObservationV1(WireRecord):
    call_id: Annotated[str, Str(nonempty=True, logical=True)]
    dispatch: Annotated[Mapping[str, Any] | None, UnionOf((type(None), _TOOL_DISPATCH))]
    RECORD_TYPE: ClassVar[str] = "ToolObservationV1"

    def check(self) -> None:
        if self.dispatch is None:
            return
        for path, change in self.dispatch["effect"].items():
            safe_path(path)
            if change["before"] == change["after"]:
                raise ValueError("tool effect must omit unchanged paths")


@dataclass(frozen=True)
class AuthorReplyV1(WireRecord):
    request_ref: Annotated[str, Hash("private")]
    status: Annotated[str, Enum(frozenset({"answered", "unsupported_coverage"}))]
    utterance: Annotated[str | None, Str(optional=True)]
    decision_ids: Annotated[
        tuple[str, ...] | list[str], ListOf(Str(nonempty=True, logical=True), unique=True)
    ]
    selected_proposals: Annotated[
        Mapping[str, Any], DictOf(ListOf(Str(nonempty=True, logical=True), unique=True))
    ]
    RECORD_TYPE: ClassVar[str] = "AuthorReplyV1"

    def check(self) -> None:
        if (self.status == "answered") != (self.utterance is not None):
            raise ValueError("utterance must be present only for an answered reply")
        if set(self.selected_proposals) - set(self.decision_ids):
            raise ValueError("proposal selection names an undisclosed decision")


@dataclass(frozen=True)
class EvaluatorResultV1(WireRecord):
    request_ref: Annotated[str, Hash("private")]
    status: Annotated[str, Enum(frozenset({"pass", "fail"}))]
    evidence_ref: Annotated[str, Hash("artifact")]
    RECORD_TYPE: ClassVar[str] = "EvaluatorResultV1"


@dataclass(frozen=True)
class ContextOperationInputV1(WireRecord):
    policy_ref: Annotated[str, Hash("artifact")]
    RECORD_TYPE: ClassVar[str] = "ContextOperationInputV1"


@dataclass(frozen=True)
class EnvironmentStepV1(WireRecord):
    directive: Annotated[
        Mapping[str, Any],
        KindUnion(
            "kind",
            MappingProxyType(
                {
                    "request_author": obj(
                        kind=Enum(frozenset({"request_author"})),
                        source=Enum(frozenset({"writer_request", "mandatory_feedback"})),
                    ),
                    "request_checks": obj(kind=Enum(frozenset({"request_checks"}))),
                    "commit_transition": obj(
                        kind=Enum(frozenset({"commit_transition"})),
                        edge_id=Str(nonempty=True, logical=True),
                    ),
                    "seal_outcome": obj(
                        kind=Enum(frozenset({"seal_outcome"})),
                        task_status=Enum(frozenset(TASK_STATUSES)),
                        stop_reason=Str(nonempty=True, optional=True),
                    ),
                    "stop_exhausted": obj(
                        kind=Enum(frozenset({"stop_exhausted"})),
                        stop_reason=Str(nonempty=True),
                    ),
                    "publish_reward": obj(kind=Enum(frozenset({"publish_reward"}))),
                }
            ),
        ),
    ]
    RECORD_TYPE: ClassVar[str] = "EnvironmentStepV1"

    @classmethod
    def of(cls, directive: Any) -> EnvironmentStepV1:
        """Encode the controller's directive in its exact environment-step shape."""
        body: dict[str, Any] = {"kind": directive.kind}
        if directive.kind == "request_author":
            body["source"] = directive.source
        elif directive.kind == "commit_transition":
            body["edge_id"] = directive.edge_id
        elif directive.kind == "seal_outcome":
            body.update(task_status=directive.task_status, stop_reason=directive.stop_reason)
        elif directive.kind == "stop_exhausted":
            body["stop_reason"] = directive.stop_reason
        return cls(directive=body)


@dataclass(frozen=True)
class MemberStartV1(WireRecord):
    group_spec_ref: Annotated[str, Hash("artifact")]
    ordinal: Annotated[int, Int()]
    RECORD_TYPE: ClassVar[str] = "MemberStartV1"


@dataclass(frozen=True)
class ExternalInputsV1(WireRecord):
    schema: Annotated[int, Int(equals=1)]
    source_refs: Annotated[tuple[str, ...] | list[str], ListOf(Hash("artifact"))]
    RECORD_TYPE: ClassVar[str] = "ExternalInputsV1"


@dataclass(frozen=True)
class AdmissionPolicyV1(WireRecord):
    schema: Annotated[int, Int(equals=1)]
    writer_family: Annotated[str | None, Str(nonempty=True, optional=True)]
    allowed_tools: Annotated[tuple[str, ...] | list[str], ListOf(Str(nonempty=True))]
    controller_versions: Annotated[tuple[str, ...] | list[str], ListOf(Str(nonempty=True))]
    check_versions: Annotated[tuple[str, ...] | list[str], ListOf(Str(nonempty=True))]
    RECORD_TYPE: ClassVar[str] = "AdmissionPolicyV1"

    @classmethod
    def from_admission_policy(cls, policy: Any) -> AdmissionPolicyV1:
        return cls(
            schema=1,
            writer_family=policy.writer_family,
            allowed_tools=sorted(policy.allowed_tools),
            controller_versions=sorted(policy.controller_versions),
            check_versions=sorted(policy.check_versions),
        )

    def to_admission_policy(self, policy_type: Any) -> Any:
        return policy_type(
            writer_family=self.writer_family,
            allowed_tools=frozenset(self.allowed_tools),
            controller_versions=frozenset(self.controller_versions),
            check_versions=frozenset(self.check_versions),
        )

    def check(self) -> None:
        for name in ("allowed_tools", "controller_versions", "check_versions"):
            values = getattr(self, name)
            if tuple(values) != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")


@dataclass(frozen=True)
class OutcomeV1(WireRecord):
    schema: Annotated[int, Int(equals=1)]
    task_status: Annotated[str, Enum(frozenset(TASK_STATUSES))]
    execution_status: Annotated[str, Enum(frozenset(EXECUTION_STATUSES))]
    stop_reason: Annotated[str | None, Str(optional=True)]
    reward_status: Annotated[str, Enum(frozenset({"pending", "available", "unavailable"}))]
    training_eligibility: Annotated[
        str,
        Enum(frozenset({"pending", "eligible", "ineligible", "structurally_eligible"})),
    ]
    candidate_checkpoint: Annotated[str | None, Hash("checkpoint", optional=True)]
    requirement_version: Annotated[str | None, Hash("private", optional=True)]
    checks: Annotated[
        tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
        ListOf(obj(request_ref=Hash("private"), result_ref=Hash("artifact", optional=True))),
    ]
    transition_edge_id: Annotated[str | None, Str(logical=True, optional=True)]
    failed_request_ref: Annotated[str | None, Hash("private", optional=True)]
    reward_ref: Annotated[str | None, Hash("artifact", optional=True)]
    eligibility_ref: Annotated[str | None, Hash("artifact", optional=True)]
    RECORD_TYPE: ClassVar[str] = "OutcomeV1"


@dataclass(frozen=True)
class RuntimePortDescriptorV1(WireRecord):
    schema: Annotated[int, Int(equals=1)]
    role: Annotated[str, Enum(frozenset({"sampling", "environment", "tools", "evaluator"}))]
    implementation: Annotated[str, Str(nonempty=True)]
    version: Annotated[str, Str(nonempty=True)]
    configuration: Annotated[Mapping[str, Any], DictOf(_JSON)]
    RECORD_TYPE: ClassVar[str] = "RuntimePortDescriptorV1"


@dataclass(frozen=True)
class RuntimeManifestV1(WireRecord):
    schema: Annotated[int, Int(equals=1)]
    ports: Annotated[
        tuple[RuntimePortDescriptorV1 | Mapping[str, Any], ...]
        | list[RuntimePortDescriptorV1 | Mapping[str, Any]],
        ListOf(RecordOf(RuntimePortDescriptorV1), min_items=4, max_items=4),
    ]
    RECORD_TYPE: ClassVar[str] = "RuntimeManifestV1"

    def check(self) -> None:
        roles = tuple(port.role for port in self.ports)
        if set(roles) != {"sampling", "environment", "tools", "evaluator"} or len(roles) != 4:
            raise ValueError("runtime manifest requires one descriptor for each port")
        sampler = next(port for port in self.ports if port.role == "sampling")
        capabilities = sampler.configuration.get("capabilities", ())
        if (
            not isinstance(capabilities, (tuple, list))
            or set(capabilities) - {"usage_reporting"}
            or tuple(capabilities) != tuple(sorted(set(capabilities)))
        ):
            raise ValueError("RuntimeManifestV1 capabilities must remain V1-closed")


@dataclass(frozen=True)
class RuntimeManifestV2(WireRecord):
    schema: Annotated[int, Int(equals=2)]
    ports: Annotated[
        tuple[RuntimePortDescriptorV1 | Mapping[str, Any], ...]
        | list[RuntimePortDescriptorV1 | Mapping[str, Any]],
        ListOf(RecordOf(RuntimePortDescriptorV1), min_items=4, max_items=4),
    ]
    capabilities: Annotated[
        tuple[str, ...] | list[str],
        ListOf(Enum(NATIVE_RUNTIME_CAPABILITIES), unique=True),
    ]
    renderer: Annotated[RendererDescriptorV1 | Mapping[str, Any], RecordOf(RendererDescriptorV1)]
    tokenizer: Annotated[TokenizerDescriptorV1 | Mapping[str, Any], RecordOf(TokenizerDescriptorV1)]
    decoding: Annotated[DecodingDescriptorV1 | Mapping[str, Any], RecordOf(DecodingDescriptorV1)]
    RECORD_TYPE: ClassVar[str] = "RuntimeManifestV2"

    def check(self) -> None:
        roles = tuple(port.role for port in self.ports)
        if set(roles) != {"sampling", "environment", "tools", "evaluator"} or len(roles) != 4:
            raise ValueError("runtime manifest requires one descriptor for each port")
        if tuple(self.capabilities) != tuple(sorted(NATIVE_RUNTIME_CAPABILITIES)):
            raise ValueError("RuntimeManifestV2 must declare all native sampling capabilities")
        if self.renderer.tokenizer_ref != self.tokenizer.identity():
            raise ValueError("renderer tokenizer pin differs from tokenizer descriptor")

        sampler = next(port for port in self.ports if port.role == "sampling")
        port_capabilities = sampler.configuration.get("capabilities", ())
        if not isinstance(port_capabilities, (tuple, list)):
            raise TypeError("sampling port capabilities must be an array")
        if set(port_capabilities) - NATIVE_RUNTIME_CAPABILITIES:
            raise ValueError("sampling port declares an unsupported capability")
        if tuple(port_capabilities) != tuple(sorted(set(port_capabilities))):
            raise ValueError("sampling port capabilities must be sorted and unique")


RuntimeManifest = RuntimeManifestV1 | RuntimeManifestV2


def decode_runtime_manifest(value: Mapping[str, Any]) -> RuntimeManifest:
    """Decode the closed set of runtime manifest record types."""
    record_type = value.get("record_type")
    if record_type == RuntimeManifestV1.RECORD_TYPE:
        return RuntimeManifestV1.from_dict(dict(value))
    if record_type == RuntimeManifestV2.RECORD_TYPE:
        return RuntimeManifestV2.from_dict(dict(value))
    raise ValueError("runtime adapter pin is not a registered manifest")


@dataclass(frozen=True)
class ContextContentV1(WireRecord):
    """One immutable node in the writer-visible context chain."""

    parent_ref: Annotated[str | None, Hash("context_node", optional=True)]
    messages: Annotated[
        tuple[MessageV1 | Mapping[str, Any], ...] | list[MessageV1 | Mapping[str, Any]],
        ListOf(MessageValue()),
    ]
    tools: Annotated[
        tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
        UnionOf((type(None), ListOf(obj_opt({}, extra=_JSON)))),
    ]
    rendering: Annotated[
        Mapping[str, Any] | None,
        UnionOf((type(None), _RENDERING_SCHEMA)),
    ]
    EDGE_TYPE: ClassVar[str] = "context_node"

    def check(self) -> None:
        is_root = self.parent_ref is None
        if (self.tools is None) != (self.rendering is None) or is_root != (self.tools is not None):
            raise ValueError("root content alone must carry tools and rendering")

    def identity(self) -> str:
        return domain_hash("context_content", self.to_wire())


@dataclass(frozen=True)
class ContextRevisionV1(WireRecord):
    """Immutable reference to one materialized context node and its source history."""

    content_ref: Annotated[str, Hash("context_node")]
    event_head: Annotated[str | None, Hash("event", optional=True)]
    provenance_refs: Annotated[tuple[str, ...] | list[str], ListOf(Hash("event"))]
    EDGE_TYPE: ClassVar[str] = "context_revision"

    def identity(self) -> str:
        return domain_hash("context", self.to_wire())


RECORD_TYPES: Mapping[str, type[WireRecord] | PayloadCodec] = MappingProxyType(
    {
        **{
            record.RECORD_TYPE: record
            for record in registered_record_classes()
            if record.RECORD_TYPE
        },
        **dict(_PAYLOAD_CODECS),
    }
)

RECORD_EDGES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        **{
            record.RECORD_TYPE or record.EDGE_TYPE or record.__name__: record.REFS
            for record in registered_record_classes()
        },
        **{record_type: codec.REFS for record_type, codec in _PAYLOAD_CODECS},
    }
)


def record_reference_edges(
    record_type: str, body: Mapping[str, Any]
) -> tuple[tuple[str, str], ...]:
    """Validate one registered payload and return its direct declared edges."""
    return tuple(
        (edge, identity) for _path, edge, identity in record_reference_paths(record_type, body)
    )


def record_reference_paths(
    record_type: str, body: Mapping[str, Any]
) -> tuple[tuple[str, str, str], ...]:
    """Validate a record and return each reference with its codec field path."""
    codec = RECORD_TYPES.get(record_type)
    if codec is None:
        try:
            codec = record_class(record_type)
        except KeyError as exc:
            raise ValueError(f"unregistered task-graph record: {record_type!r}") from exc
    if isinstance(codec, type):
        record = codec.from_dict(dict(body))
        edges = record._wire_edges
    else:
        edges = validate_codec(codec, body)
    return tuple((path, edge, identity) for path, edge, identity in edges)


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
    if root is None or root.tools is None or root.rendering is None:
        raise task_graph_errors.MaterializationError("context chain has no materializable root")
    messages = tuple(message for chunk in reversed(chunks) for message in chunk)
    return MaterializedContextV1(messages, root.tools, root.rendering)
