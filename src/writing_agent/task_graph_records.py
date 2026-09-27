"""Strict wire-v1 records, payload registry, and chained-context materialization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, ClassVar

from writing_agent.task_graph import (
    MessageV1,
    domain_hash,
    safe_path,
)
from writing_agent.task_graph_contracts import EXECUTION_STATUSES, TASK_STATUSES
from writing_agent.task_graph_errors import MaterializationError
from writing_agent.task_graph_payloads import PAYLOAD_RECORD_CODECS
from writing_agent.task_graph_record_contracts import (
    POLICY_FIELDS as POLICY_FIELDS,
)
from writing_agent.task_graph_record_contracts import (
    CompactionError as CompactionError,
)
from writing_agent.task_graph_record_contracts import (
    ContextPolicyV1,
    ExecutionVersionsV1,
    GroupMemberSpecV1,
    GroupSpecV1,
)
from writing_agent.task_graph_record_contracts import (
    GroupError as GroupError,
)
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
    _edge_values,
    _f,
    _o,
    _validate_codec,
    _WireRecord,
    validate_canonical_value,
)
from writing_agent.task_graph_wire import (
    Obj as Obj,
)
from writing_agent.task_graph_wire import (
    decode_canonical_value as decode_canonical_value,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEMANTICS_V1 = "task-graph-derive-v1"
_TRACE_REF_KEYS = (
    "model_ref behavior_policy_ref tokenizer_ref template_ref adapter_ref decoding_ref "
    "context_policy_ref policy_ref"
).split()
_TRACE_LOGPROB_KEYS = frozenset(
    "per_token_logprobs_ref per_token_logprobs_codec per_token_logprobs_shape".split()
)
_JSON = JsonValue()
_TEXT = Str()
_TRACE_SCHEMA = _o(
    {},
    {
        "model": _TEXT,
        "seed": Int(minimum=None),
        "generated_token_ids": ListOf(Int()),
        **{key: Hash("artifact") for key in _TRACE_REF_KEYS},
        "per_token_logprobs_ref": Hash("bytes"),
        "per_token_logprobs_codec": Enum(frozenset({"f32-le"})),
        "per_token_logprobs_shape": ListOf(Int()),
    },
    extra=_JSON,
)
_RENDERING_SCHEMA = _f(
    projection_version=Str(nonempty=True),
    prefix_id=Str(nonempty=True, logical=True),
    template_ref=Hash("artifact"),
    tokenizer_ref=Hash("artifact"),
    tool_schema_ref=Hash("artifact"),
)
_USAGE_SCHEMA = _o(
    {},
    {key: Int() for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
    extra=_JSON,
)
_TOOL_DISPATCH = _f(
    spec=_f(max_file_bytes=Int(minimum=1), max_workspace_bytes=Int(minimum=1)),
    observation=_o({"ok": Bool(), "valid": Bool()}, {"result": _JSON, "error": _TEXT}),
    effect=DictOf(_f(before=Str(optional=True), after=Str(optional=True))),
)


@dataclass(frozen=True)
class SampledMessageV1(_WireRecord):
    content: Annotated[Any, CanonicalIntake()]
    tool_calls_was_list: Annotated[bool, Bool()]
    calls: Annotated[Any, _JSON]

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


@dataclass(frozen=True)
class WriterTurnV1(_WireRecord):
    action_id: Annotated[str, Str(nonempty=True, logical=True)]
    context_revision_ref: Annotated[str, Hash("context_revision")]
    request_ref: Annotated[str | None, Hash("artifact|bytes", optional=True)]
    prepared_request_ref: Annotated[str | None, Hash("artifact", optional=True)]
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
        present = _TRACE_LOGPROB_KEYS & set(self.adapter_trace)
        if present and present != _TRACE_LOGPROB_KEYS:
            raise ValueError("logprob reference, codec, and shape must appear together")


@dataclass(frozen=True)
class WriterRequestV1(_WireRecord):
    context_revision_ref: Annotated[str, Hash("context_revision")]
    payload_ref: Annotated[str, Hash("artifact|bytes")]
    verified_messages: Annotated[bool, Bool()]
    RECORD_TYPE: ClassVar[str] = "WriterRequestV1"


@dataclass(frozen=True)
class ToolObservationV1(_WireRecord):
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
class AuthorReplyV1(_WireRecord):
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
class EvaluatorResultV1(_WireRecord):
    request_ref: Annotated[str, Hash("private")]
    status: Annotated[str, Enum(frozenset({"pass", "fail"}))]
    evidence_ref: Annotated[str, Hash("artifact")]
    RECORD_TYPE: ClassVar[str] = "EvaluatorResultV1"


@dataclass(frozen=True)
class ContextOperationInputV1(_WireRecord):
    policy_ref: Annotated[str, Hash("artifact")]
    RECORD_TYPE: ClassVar[str] = "ContextOperationInputV1"


@dataclass(frozen=True)
class EnvironmentStepV1(_WireRecord):
    directive: Annotated[
        Mapping[str, Any],
        KindUnion(
            "kind",
            MappingProxyType(
                {
                    "request_author": _f(
                        kind=Enum(frozenset({"request_author"})),
                        source=Enum(frozenset({"writer_request", "mandatory_feedback"})),
                    ),
                    "request_checks": _f(kind=Enum(frozenset({"request_checks"}))),
                    "commit_transition": _f(
                        kind=Enum(frozenset({"commit_transition"})),
                        edge_id=Str(nonempty=True, logical=True),
                    ),
                    "seal_outcome": _f(
                        kind=Enum(frozenset({"seal_outcome"})),
                        task_status=Enum(frozenset(TASK_STATUSES)),
                        stop_reason=Str(optional=True),
                    ),
                    "stop_exhausted": _f(
                        kind=Enum(frozenset({"stop_exhausted"})),
                        stop_reason=Str(nonempty=True),
                    ),
                    "publish_reward": _f(kind=Enum(frozenset({"publish_reward"}))),
                }
            ),
        ),
    ]
    RECORD_TYPE: ClassVar[str] = "EnvironmentStepV1"

    def check(self) -> None:
        if self.directive["kind"] == "seal_outcome" and self.directive["stop_reason"] == "":
            raise ValueError("stop reason must be nonempty when present")


@dataclass(frozen=True)
class MemberStartV1(_WireRecord):
    group_spec_ref: Annotated[str, Hash("artifact")]
    ordinal: Annotated[int, Int()]
    RECORD_TYPE: ClassVar[str] = "MemberStartV1"


@dataclass(frozen=True)
class ExternalInputsV1(_WireRecord):
    schema: Annotated[int, Int(equals=1)]
    source_refs: Annotated[tuple[str, ...] | list[str], ListOf(Hash("artifact"))]
    RECORD_TYPE: ClassVar[str] = "ExternalInputsV1"


@dataclass(frozen=True)
class AdmissionPolicyV1(_WireRecord):
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

    def check(self) -> None:
        for name in ("allowed_tools", "controller_versions", "check_versions"):
            values = getattr(self, name)
            if tuple(values) != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")


@dataclass(frozen=True)
class OutcomeV1(_WireRecord):
    schema: Annotated[int, Int(equals=1)]
    task_status: Annotated[str, Enum(frozenset(TASK_STATUSES))]
    execution_status: Annotated[str, Enum(frozenset(EXECUTION_STATUSES))]
    stop_reason: Annotated[str | None, Str(optional=True)]
    reward_status: Annotated[str, Enum(frozenset({"pending", "available", "unavailable"}))]
    training_eligibility: Annotated[str, Enum(frozenset({"pending", "eligible", "ineligible"}))]
    candidate_checkpoint: Annotated[str | None, Hash("checkpoint", optional=True)]
    requirement_version: Annotated[str | None, Hash("private", optional=True)]
    checks: Annotated[
        tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
        ListOf(_f(request_ref=Hash("private"), result_ref=Hash("artifact", optional=True))),
    ]
    transition_edge_id: Annotated[str | None, Str(logical=True, optional=True)]
    failed_request_ref: Annotated[str | None, Hash("private", optional=True)]
    reward_ref: Annotated[str | None, Hash("artifact", optional=True)]
    eligibility_ref: Annotated[str | None, Hash("artifact", optional=True)]
    RECORD_TYPE: ClassVar[str] = "OutcomeV1"


@dataclass(frozen=True)
class ContextContentV1(_WireRecord):
    parent_ref: Annotated[str | None, Hash("context_node", optional=True)]
    messages: Annotated[
        tuple[MessageV1 | Mapping[str, Any], ...] | list[MessageV1 | Mapping[str, Any]],
        ListOf(MessageValue()),
    ]
    tools: Annotated[
        tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
        UnionOf((type(None), ListOf(_o({}, extra=_JSON)))),
    ]
    rendering: Annotated[Mapping[str, Any] | None, UnionOf((type(None), _RENDERING_SCHEMA))]
    RECORD_TYPE: ClassVar[str | None] = None

    def check(self) -> None:
        is_root = self.parent_ref is None
        if (self.tools is None) != (self.rendering is None) or is_root != (self.tools is not None):
            raise ValueError("root content alone must carry tools and rendering")

    def identity(self) -> str:
        return domain_hash("context_content", self.to_wire())


@dataclass(frozen=True)
class ContextRevisionV1(_WireRecord):
    content_ref: Annotated[str, Hash("context_node")]
    event_head: Annotated[str | None, Hash("event", optional=True)]
    provenance_refs: Annotated[tuple[str, ...] | list[str], ListOf(Hash("event"))]
    RECORD_TYPE: ClassVar[str | None] = None

    def identity(self) -> str:
        return domain_hash("context", self.to_wire())


_WIRE_CLASSES = (
    SampledMessageV1,
    WriterTurnV1,
    WriterRequestV1,
    ToolObservationV1,
    AuthorReplyV1,
    EvaluatorResultV1,
    ContextOperationInputV1,
    EnvironmentStepV1,
    MemberStartV1,
    ExternalInputsV1,
    AdmissionPolicyV1,
    OutcomeV1,
    ContextContentV1,
    ContextRevisionV1,
    GroupMemberSpecV1,
    GroupSpecV1,
    ContextPolicyV1,
    ExecutionVersionsV1,
)
RECORD_TYPES: Mapping[str, type[_WireRecord] | PayloadCodec] = MappingProxyType(
    {
        **{record.RECORD_TYPE: record for record in _WIRE_CLASSES if record.RECORD_TYPE},
        **dict(PAYLOAD_RECORD_CODECS),
        "ExecutionVersionsV1": ExecutionVersionsV1,
    }
)

SHARED_WIRE_V1_RECORD_TYPES = frozenset(
    "AuthorRequestV1 CheckRequestV1 DecisionLedgerV1 DisclosureLedgerV1 RequirementLedgerV1 "
    "RewardV1 TrainingEligibilityV1 GroupMemberSeedsV1".split()
)

LEGACY_PAYLOAD_RECORD_TYPES = frozenset(
    "AuthorToolAckV1 AuthorTurnV1 CheckBatchV1 CheckResultV1 ContextOperationV1 "
    "DecisionDisclosureV1 DeterministicCheckEvidenceV1 FixtureFileCountEvidenceV1 "
    "GroupExecutionFailureV1 GroupAdvantageV1 GroupDecisionV1 GroupMemberResultV1 "
    "GroupSegmentCreditV1 GroupScriptedTerminalV1 InfrastructureInvalidV1 "
    "PreparedWriterRequestV1 RequirementSupersessionV1 RewardAvailabilityV1 "
    "RewardPublicationV1 RuntimeManifestV1 ScriptCoverageFailureV1 ScriptedAuthorReplyV1 "
    "TerminalOutcomeCommitV1 TerminalOutcomeV1 TransitionDecisionV1 "
    "TranscriptReviewEvidenceV1 VerifiedWriterMessagesV1 WriterActionTraceV1 WriterActionV1 "
    "WriterExhaustedStopV1 WriterRuntimeLogV1 WriterSampledBudgetStopV1 WriterToolResultV1".split()
)

RECORD_EDGES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        **{record_type: codec.REFS for record_type, codec in RECORD_TYPES.items()},
        "context_node": ContextContentV1.REFS,
        "context_revision": ContextRevisionV1.REFS,
    }
)
ALL_RECORD_CODECS: Mapping[str, Any] = MappingProxyType(
    {
        **dict(RECORD_TYPES),
        "SampledMessageV1": SampledMessageV1,
        "GroupMemberSpecV1": GroupMemberSpecV1,
        "ExecutionVersionsV1": ExecutionVersionsV1,
        "context_node": ContextContentV1,
        "context_revision": ContextRevisionV1,
    }
)


def record_reference_edges(
    record_type: str, body: Mapping[str, Any]
) -> tuple[tuple[str, str], ...]:
    """Validate one registered payload and return its direct declared edges."""
    try:
        codec = ALL_RECORD_CODECS[record_type]
        schema = codec.FIELD_SPEC if isinstance(codec, type) else codec.fields
    except KeyError as exc:
        raise ValueError(f"unregistered task-graph record: {record_type!r}") from exc
    if isinstance(codec, type):
        value = codec.from_dict(dict(body)).to_wire()
    else:
        _validate_codec(codec, body)
        value = body
    return _edge_values(schema, value)


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
    if root is None or root.tools is None or root.rendering is None:
        raise MaterializationError("context chain has no materializable root")
    messages = tuple(message for chunk in reversed(chunks) for message in chunk)
    return MaterializedContextV1(messages, root.tools, root.rendering)


def is_sha256_string(value: Any) -> bool:
    """Small helper shared with the ``REFS`` completeness test."""
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None
