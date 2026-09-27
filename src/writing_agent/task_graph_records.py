"""Strict wire codecs for the transition-seam task-graph runtime.

These records are deliberately separate from the legacy task-graph runtime.  They
provide canonical, closed decoders and an explicit reference-edge registry; no
producer uses them until the transition core is introduced.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, ClassVar

from writing_agent.task_graph import (
    MessageV1,
    _freeze,
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
from writing_agent.task_graph_errors import MaterializationError

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEMANTICS_V1 = "task-graph-derive-v1"
_INTAKE_MARKER_FIELDS = {
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


def _decode_canonical_value(value: Any) -> Any:
    """Validate and decode exactly one canonical value produced by adapter intake."""
    if value is None or type(value) in (str, int, bool):
        if type(value) is str:
            _utf8(value)
        return value
    if isinstance(value, list):
        return [_decode_canonical_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_canonical_value(item) for item in value)
    if not isinstance(value, Mapping):
        raise TypeError("value is not a canonical intake value")
    if any(type(key) is not str for key in value):
        raise TypeError("canonical object keys must be strings")
    if "$noncanonical" in value:
        marker = value["$noncanonical"]
        if type(marker) is not str or marker not in _INTAKE_MARKER_FIELDS:
            raise ValueError("unknown or forged noncanonical marker")
        if set(value) != _INTAKE_MARKER_FIELDS[marker]:
            raise ValueError("noncanonical marker has the wrong key set")
        if marker == "float":
            _string(value["repr"], "float marker repr")
            try:
                decoded = float(value["repr"])
            except (ValueError, OverflowError) as exc:
                raise ValueError("float marker representation is invalid") from exc
            if repr(decoded) != value["repr"]:
                raise ValueError("float marker representation is not canonical")
            return decoded
        elif marker == "bytes":
            _string(value["hex"], "bytes marker hex")
            try:
                decoded = bytes.fromhex(value["hex"])
            except ValueError as exc:
                raise ValueError("bytes marker contains invalid hexadecimal") from exc
            if decoded.hex() != value["hex"]:
                raise ValueError("bytes marker hexadecimal is not canonical")
            return decoded
        elif marker in {"tuple", "mapping"}:
            items = _array(value["items"], "marker items")
            if marker == "tuple":
                return tuple(_decode_canonical_value(item) for item in items)
            else:
                decoded_mapping = {}
                for pair in items:
                    pair = _array(pair, "mapping item")
                    if len(pair) != 2:
                        raise ValueError("mapping item must be a key/value pair")
                    key = _decode_canonical_value(pair[0])
                    item = _decode_canonical_value(pair[1])
                    try:
                        if key in decoded_mapping:
                            raise ValueError("mapping marker has duplicate decoded keys")
                        decoded_mapping[key] = item
                    except TypeError as exc:
                        raise ValueError("mapping marker keys must be hashable") from exc
                return decoded_mapping
        elif marker == "surrogate-string":
            points = _array(value["codepoints"], "surrogate codepoints")
            decoded_points = []
            for point in points:
                _integer(point, "surrogate codepoint", minimum=0)
                if point > 0x10FFFF:
                    raise ValueError("surrogate codepoint is outside Unicode")
                decoded_points.append(chr(point))
            return "".join(decoded_points)
        elif marker == "unsupported":
            _string(value["type"], "unsupported type", nonempty=True)
            return object()
        return object()
    return {key: _decode_canonical_value(item) for key, item in value.items()}


def validate_canonical_value(value: Any) -> None:
    """Validate a canonical intake value without preserving adapter-only values."""
    _decode_canonical_value(value)


def decode_canonical_value(value: Any) -> Any:
    """Decode a value already carried by a validated :class:`SampledMessageV1`."""
    return _decode_canonical_value(value)


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
class Hash:
    """A hash field; ``edge=None`` explicitly means a hash, but not a ref."""

    edge: str | None
    optional: bool = False


@dataclass(frozen=True)
class Str:
    nonempty: bool = False
    logical: bool = False


@dataclass(frozen=True)
class Int:
    minimum: int | None = 0
    equals: int | None = None


@dataclass(frozen=True)
class Bool:
    pass


@dataclass(frozen=True)
class JsonValue:
    pass


@dataclass(frozen=True)
class Enum:
    values: frozenset[str]


@dataclass(frozen=True)
class ListOf:
    item: Any
    unique: bool = False


@dataclass(frozen=True)
class DictOf:
    value: Any


@dataclass(frozen=True)
class Obj:
    required: Mapping[str, Any]
    optional: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    extra: Any | None = None


@dataclass(frozen=True)
class Open:
    """An arbitrary object containing only canonical JSON values."""


@dataclass(frozen=True)
class UnionOf:
    options: tuple[Any, ...]


@dataclass(frozen=True)
class KindUnion:
    discriminator: str
    variants: Mapping[str, Obj]


@dataclass(frozen=True)
class CanonicalIntake:
    """A value using the adapter-intake marker vocabulary."""


@dataclass(frozen=True)
class MessageValue:
    """A wire ``MessageV1`` object from the task-graph base layer."""


@dataclass(frozen=True)
class RecordOf:
    record: type[_WireRecord]


@dataclass(frozen=True)
class _PayloadCodec:
    record_type: str
    fields: Obj
    tagged: bool = True

    @property
    def REFS(self) -> Mapping[str, str]:
        return _derived_refs(self.fields)

    def from_dict(self, body: Mapping[str, Any]) -> dict[str, Any]:
        _validate_codec(self, body)
        return dict(body)


def _validate_spec(spec: Any, value: Any, label: str) -> None:
    if isinstance(spec, Hash):
        validate_hash(value, optional=spec.optional)
    elif isinstance(spec, Str):
        _string(value, label, nonempty=spec.nonempty)
        if spec.logical:
            _logical_id(value, label)
    elif isinstance(spec, Int):
        _integer(value, label, minimum=spec.minimum)
        if spec.equals is not None and value != spec.equals:
            raise ValueError(f"{label} must equal {spec.equals}")
    elif isinstance(spec, Bool):
        _boolean(value, label)
    elif isinstance(spec, Enum):
        if type(value) is not str or value not in spec.values:
            raise ValueError(f"{label} has an unsupported value")
    elif isinstance(spec, ListOf):
        values = _array(value, label)
        for index, item in enumerate(values):
            _validate_spec(spec.item, item, f"{label}[{index}]")
        if spec.unique and len(values) != len(set(values)):
            raise ValueError(f"{label} must be unique")
    elif isinstance(spec, DictOf):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be an object")
        for key, item in value.items():
            _string(key, f"{label} key")
            _validate_spec(spec.value, item, f"{label}.{key}")
    elif isinstance(spec, Obj):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be an object")
        keys = set(value)
        required = set(spec.required)
        optional = set(spec.optional)
        if required - keys:
            raise ValueError(f"{label} is missing fields: {sorted(required - keys)}")
        unknown = keys - required - optional
        if unknown and spec.extra is None:
            raise ValueError(f"{label} has unknown fields: {sorted(unknown)}")
        for key, item_spec in spec.required.items():
            _validate_spec(item_spec, value[key], f"{label}.{key}")
        for key, item_spec in spec.optional.items():
            if key in value:
                _validate_spec(item_spec, value[key], f"{label}.{key}")
        if unknown:
            _validate_spec(spec.extra, {key: value[key] for key in unknown}, label)
    elif isinstance(spec, Open):
        _json_shape(value)
    elif isinstance(spec, JsonValue):
        _json_shape(value)
    elif isinstance(spec, UnionOf):
        failures = []
        for option in spec.options:
            try:
                _validate_spec(option, value, label)
                return
            except (TypeError, ValueError) as exc:
                failures.append(exc)
        raise ValueError(f"{label} does not match a permitted shape") from failures[-1]
    elif isinstance(spec, KindUnion):
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a tagged object")
        kind = value.get(spec.discriminator)
        try:
            variant = spec.variants[kind]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{label} has an unknown {spec.discriminator}") from exc
        _validate_spec(variant, value, label)
    elif isinstance(spec, CanonicalIntake):
        validate_canonical_value(value)
    elif isinstance(spec, MessageValue):
        if isinstance(value, MessageV1):
            return
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a MessageV1 object")
        MessageV1.from_dict(dict(value))
    elif isinstance(spec, RecordOf):
        if isinstance(value, spec.record):
            return
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a {spec.record.__name__} record")
        spec.record.from_dict(dict(value))
    elif spec is type(None):
        if value is not None:
            raise TypeError(f"{label} must be null")
    else:
        raise TypeError(f"unsupported field specification for {label}: {spec!r}")


def _derived_refs(spec: Obj) -> Mapping[str, str]:
    edges: dict[str, str] = {}

    def walk(value_spec: Any, path: str) -> None:
        if isinstance(value_spec, Hash):
            if value_spec.edge is not None:
                edges[path] = value_spec.edge
        elif isinstance(value_spec, ListOf):
            walk(value_spec.item, f"{path}[]")
        elif isinstance(value_spec, DictOf):
            walk(value_spec.value, f"{path}{{}}")
        elif isinstance(value_spec, Obj):
            for key, child in (*value_spec.required.items(), *value_spec.optional.items()):
                walk(child, f"{path}.{key}" if path else key)
        elif isinstance(value_spec, UnionOf):
            for option in value_spec.options:
                walk(option, path)
        elif isinstance(value_spec, KindUnion):
            for option in value_spec.variants.values():
                walk(option, path)
        elif isinstance(value_spec, RecordOf):
            edges.update(
                {
                    f"{path}.{key}" if path else key: kind
                    for key, kind in value_spec.record.REFS.items()
                }
            )

    walk(spec, "")
    return MappingProxyType(edges)


def _validate_codec(codec: _PayloadCodec, body: Mapping[str, Any]) -> None:
    if type(body) is not dict:
        raise TypeError("wire record must be a plain object")
    _wire_value(body)
    required = set(codec.fields.required)
    optional = set(codec.fields.optional)
    if codec.tagged:
        if body.get("record_type") != codec.record_type:
            raise ValueError("record_type does not match codec")
    actual = set(body) - ({"record_type"} if codec.tagged else set())
    if required - actual or actual - required - optional:
        raise ValueError("record has missing or unknown fields")
    _validate_spec(codec.fields, {key: body[key] for key in actual}, codec.record_type)


def _edge_values(spec: Any, value: Any) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    if isinstance(spec, Hash):
        if value is not None and spec.edge is not None:
            found.append((spec.edge, value))
    elif isinstance(spec, ListOf):
        for item in value:
            found.extend(_edge_values(spec.item, item))
    elif isinstance(spec, DictOf):
        for item in value.values():
            found.extend(_edge_values(spec.value, item))
    elif isinstance(spec, Obj):
        for key, child in (*spec.required.items(), *spec.optional.items()):
            if key in value:
                found.extend(_edge_values(child, value[key]))
    elif isinstance(spec, UnionOf):
        for option in spec.options:
            if isinstance(option, Obj):
                try:
                    _validate_spec(option, value, "union value")
                except (TypeError, ValueError):
                    continue
                found.extend(_edge_values(option, value))
                break
    elif isinstance(spec, KindUnion):
        found.extend(_edge_values(spec.variants[value[spec.discriminator]], value))
    elif isinstance(spec, RecordOf):
        body = value.to_wire() if isinstance(value, spec.record) else value
        found.extend(_edge_values(spec.record.FIELD_SPEC, body))
    return tuple(found)


@dataclass(frozen=True)
class _WireRecord:
    """Base for strict, frozen wire values with optional payload record types."""

    RECORD_TYPE: ClassVar[str | None] = None
    FIELD_SPEC: ClassVar[Obj]
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType({})

    def __post_init__(self) -> None:
        self.validate()
        canonical_bytes(self.to_wire())
        for item in fields(self):
            object.__setattr__(self, item.name, _freeze(getattr(self, item.name)))
        canonical_bytes(self.to_wire())

    def validate(self) -> None:
        _validate_schema_fields(self)

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

    def __post_init__(self) -> None:
        if isinstance(self.message, Mapping):
            object.__setattr__(self, "message", SampledMessageV1.from_dict(dict(self.message)))
        super().__post_init__()


@dataclass(frozen=True)
class WriterRequestV1(_WireRecord):
    context_revision_ref: str
    payload_ref: str
    verified_messages: bool
    RECORD_TYPE: ClassVar[str] = "WriterRequestV1"


@dataclass(frozen=True)
class ToolObservationV1(_WireRecord):
    call_id: str
    dispatch: Mapping[str, Any] | None
    RECORD_TYPE: ClassVar[str] = "ToolObservationV1"


@dataclass(frozen=True)
class AuthorReplyV1(_WireRecord):
    request_ref: str
    status: str
    utterance: str | None
    decision_ids: tuple[str, ...] | list[str]
    selected_proposals: Mapping[str, Any]
    RECORD_TYPE: ClassVar[str] = "AuthorReplyV1"


@dataclass(frozen=True)
class EvaluatorResultV1(_WireRecord):
    request_ref: str
    status: str
    evidence_ref: str
    RECORD_TYPE: ClassVar[str] = "EvaluatorResultV1"


@dataclass(frozen=True)
class ContextOperationInputV1(_WireRecord):
    policy_ref: str
    RECORD_TYPE: ClassVar[str] = "ContextOperationInputV1"


@dataclass(frozen=True)
class EnvironmentStepV1(_WireRecord):
    directive: Mapping[str, Any]
    RECORD_TYPE: ClassVar[str] = "EnvironmentStepV1"


@dataclass(frozen=True)
class MemberStartV1(_WireRecord):
    group_spec_ref: str
    ordinal: int
    RECORD_TYPE: ClassVar[str] = "MemberStartV1"


@dataclass(frozen=True)
class ExternalInputsV1(_WireRecord):
    schema: int
    source_refs: tuple[str, ...] | list[str]
    RECORD_TYPE: ClassVar[str] = "ExternalInputsV1"


@dataclass(frozen=True)
class AdmissionPolicyV1(_WireRecord):
    schema: int
    writer_family: str | None
    allowed_tools: tuple[str, ...] | list[str]
    controller_versions: tuple[str, ...] | list[str]
    check_versions: tuple[str, ...] | list[str]
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


@dataclass(frozen=True)
class ContextContentV1(_WireRecord):
    parent_ref: str | None
    messages: tuple[MessageV1 | Mapping[str, Any], ...] | list[MessageV1 | Mapping[str, Any]]
    tools: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None
    rendering: Mapping[str, Any] | None
    RECORD_TYPE: ClassVar[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(_message(item) for item in self.messages))
        super().__post_init__()

    def identity(self) -> str:
        return domain_hash("context_content", self.to_wire())


@dataclass(frozen=True)
class ContextRevisionV1(_WireRecord):
    content_ref: str
    event_head: str | None
    provenance_refs: tuple[str, ...] | list[str]
    RECORD_TYPE: ClassVar[str] = None

    def identity(self) -> str:
        return domain_hash("context", self.to_wire())


class GroupError(ValueError):
    """A sealed group contract or immutable group record is invalid."""


class CompactionError(ValueError):
    """A context operation is not safe or its recorded evidence is false."""


POLICY_FIELDS = frozenset(
    {
        "model_ref",
        "behavior_policy_ref",
        "tokenizer_ref",
        "template_ref",
        "adapter_ref",
        "decoding_ref",
        "simulator_ref",
        "context_policy_ref",
        "controller_ref",
        "rng_derivation_version",
    }
)


def _group_hash(value: Any) -> str:
    return domain_hash("payload", value)


def _group_seed(group_seed: int, role: str, ordinal: int | None = None) -> int:
    import hashlib

    material = canonical_bytes(["GroupSeedV1", group_seed, role, ordinal])
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@dataclass(frozen=True)
class GroupMemberSpecV1(_WireRecord):
    member_id: str = ""
    ordinal: int = -1
    writer_seed: int = -1
    environment_seed: int = -1
    seed_provenance: str = "sha256-domain-v1"


@dataclass(frozen=True)
class GroupSpecV1(_WireRecord):
    group_id: str = ""
    group_sequence: int = -1
    group_seed: int = -1
    runner_mode: str = "real"
    environment: Mapping[str, Any] | None = None
    policy: Mapping[str, str] | None = None
    members: tuple[GroupMemberSpecV1 | Mapping[str, Any], ...] = ()
    RECORD_TYPE: ClassVar[str] = "GroupSpecV1"

    @property
    def record_type(self) -> str:
        return self.RECORD_TYPE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "members",
            tuple(
                member
                if isinstance(member, GroupMemberSpecV1)
                else GroupMemberSpecV1.from_dict(dict(member))
                for member in self.members
            ),
        )
        super().__post_init__()


@dataclass(frozen=True)
class ContextPolicyV1(_WireRecord):
    operation: str
    retained_exchanges: int = 0
    seed_name: str | None = None
    seed_checkpoint_ref: str | None = None
    summarizer_version: str | None = None
    max_summary_chars: int | None = None
    max_operations: int = 32
    max_context_bytes: int = 1_000_000
    max_context_storage_bytes: int = 8_000_000
    schema: int = 1
    RECORD_TYPE: ClassVar[str] = "ContextPolicyV1"


@dataclass(frozen=True)
class ExecutionVersionsV1(_WireRecord):
    schema: int
    transition_semantics: str
    admission_policy_ref: str
    tool_spec: Mapping[str, Any]


def _f(**fields: Any) -> Obj:
    return Obj(required=fields)


def _o(
    required: Mapping[str, Any],
    optional: Mapping[str, Any] = MappingProxyType({}),
    *,
    extra=None,
):
    return Obj(required=required, optional=optional, extra=extra)


def _h(edge: str | None, *, optional: bool = False) -> Hash:
    return Hash(edge, optional)


def _s(*, nonempty: bool = False, logical: bool = False) -> Str:
    return Str(nonempty, logical)


def _i(minimum: int | None = 0, *, equals: int | None = None) -> Int:
    return Int(minimum, equals)


def _l(item: Any, *, unique: bool = False) -> ListOf:
    return ListOf(item, unique)


def _enum(*values: str) -> Enum:
    return Enum(frozenset(values))


_TEXT = _s()
_BOOL = Bool()
_JSON = JsonValue()
_INTAKE = CanonicalIntake()
_NULLABLE_TEXT = UnionOf((_TEXT, type(None)))
_NULLABLE_LOGICAL = UnionOf((_s(logical=True), type(None)))
_TOKEN_IDS = _l(_i())
_TRACE_REFS = {key: _h("artifact") for key in _TRACE_REF_KEYS}
_TRACE_SCHEMA = _o(
    {},
    {
        "model": _TEXT,
        "seed": _i(None),
        "generated_token_ids": _TOKEN_IDS,
        **_TRACE_REFS,
        "per_token_logprobs_ref": _h("bytes"),
        "per_token_logprobs_codec": _enum("f32-le"),
        "per_token_logprobs_shape": _l(_i()),
    },
    extra=Open(),
)
_RENDERING_SCHEMA = _f(
    projection_version=_s(nonempty=True),
    prefix_id=_s(nonempty=True, logical=True),
    template_ref=_h("artifact"),
    tokenizer_ref=_h("artifact"),
    tool_schema_ref=_h("artifact"),
)


def _sampled_rules(record: _WireRecord) -> None:
    if not record.tool_calls_was_list:
        validate_canonical_value(record.calls)
        return
    for call in _array(record.calls, "calls"):
        call = _object(call, "call", required={"bounded", "value"})
        _boolean(call["bounded"], "bounded")
        if call["bounded"]:
            validate_canonical_value(call["value"])
        elif call["value"] != {"$noncanonical": "bounded-call"}:
            raise ValueError("unbounded calls require the bounded-call marker")


def _writer_turn_rules(record: _WireRecord) -> None:
    trace = record.adapter_trace
    if trace is None:
        return
    if "per_token_logprobs" in trace or "native_on_policy_eligible" in trace:
        raise ValueError("adapter trace contains a forbidden field")
    present = _TRACE_LOGPROB_KEYS & set(trace)
    if present and present != _TRACE_LOGPROB_KEYS:
        raise ValueError("logprob reference, codec, and shape must appear together")


def _tool_observation_rules(record: _WireRecord) -> None:
    if record.dispatch is None:
        return
    for path, change in record.dispatch["effect"].items():
        safe_path(path)
        if change["before"] == change["after"]:
            raise ValueError("tool effect must omit unchanged paths")


def _author_reply_rules(record: _WireRecord) -> None:
    if (record.status == "answered") != (record.utterance is not None):
        raise ValueError("utterance must be present only for an answered reply")
    decisions = set(record.decision_ids)
    if set(record.selected_proposals) - decisions:
        raise ValueError("proposal selection names an undisclosed decision")


def _directive_rules(record: _WireRecord) -> None:
    directive = record.directive
    if directive["kind"] == "seal_outcome" and directive["task_status"] not in TASK_STATUSES:
        raise ValueError("unsupported task status")
    if directive["kind"] == "seal_outcome" and directive["stop_reason"] == "":
        raise ValueError("stop reason must be nonempty when present")


def _schema_one(record: _WireRecord) -> None:
    _schema(record.schema)


def _admission_policy_rules(record: _WireRecord) -> None:
    for name in ("allowed_tools", "controller_versions", "check_versions"):
        values = getattr(record, name)
        if tuple(values) != tuple(sorted(set(values))):
            raise ValueError(f"{name} must be sorted and unique")


def _context_content_rules(record: _WireRecord) -> None:
    is_root = record.parent_ref is None
    if (record.tools is None) != (record.rendering is None) or is_root != (
        record.tools is not None
    ):
        raise ValueError("root content alone must carry tools and rendering")


def _group_member_rules(record: _WireRecord) -> None:
    if record.seed_provenance != "sha256-domain-v1":
        raise GroupError("unknown member seed derivation")


def _group_spec_rules(record: _WireRecord) -> None:
    if record.environment is None or record.policy is None:
        raise GroupError("group spec requires its environment and policy")
    if record.group_sequence < 0 or record.group_seed < 0 or not 2 <= len(record.members) <= 64:
        raise GroupError("group size must be 2..64 and seed nonnegative")
    if record.runner_mode not in {"real", "fixture"}:
        raise GroupError("unknown group runner mode")
    if set(record.policy) != POLICY_FIELDS:
        raise GroupError("incomplete policy contract")
    expected = _group_hash(
        [
            "GroupIdV1",
            record.group_sequence,
            record.environment,
            record.policy,
            record.group_seed,
            record.runner_mode,
            len(record.members),
        ]
    )
    if record.group_id != expected:
        raise GroupError("group ID does not bind its contract and sequence")
    if tuple(member.ordinal for member in record.members) != tuple(range(len(record.members))):
        raise GroupError("member slots are not canonical")
    for member in record.members:
        if member.member_id != f"grp-{record.group_id[:24]}-{member.ordinal:02d}":
            raise GroupError("member ID does not bind its ordinal")
        if member.writer_seed != _group_seed(record.group_seed, "writer", member.ordinal):
            raise GroupError("writer seed derivation mismatch")
        if member.environment_seed != _group_seed(record.group_seed, "environment"):
            raise GroupError("environment seed derivation mismatch")
    if len({member.writer_seed for member in record.members}) != len(record.members):
        raise GroupError("writer streams are not distinct")


def _context_policy_rules(record: _WireRecord) -> None:
    _schema(record.schema)
    if record.operation == "compact":
        if (
            record.summarizer_version != "visible-text-v1"
            or type(record.max_summary_chars) is not int
            or record.max_summary_chars < 0
            or record.seed_name is not None
            or record.seed_checkpoint_ref is not None
        ):
            raise CompactionError("compact requires the fixed visible-text-v1 algorithm")
    elif record.summarizer_version is not None or record.max_summary_chars is not None:
        raise CompactionError("only compact may configure the summarizer")
    if record.operation == "seed":
        if not record.seed_name or any(
            char.isspace() or ord(char) < 0x20 for char in record.seed_name
        ):
            raise CompactionError("seed requires a named immutable prefix")
    elif record.seed_name is not None or record.seed_checkpoint_ref is not None:
        raise CompactionError("only seed may name a prefix")
    if record.operation != "compact" and record.retained_exchanges:
        raise CompactionError("only compact may retain a tail")


_CLASS_FIELD_SPECS: dict[type[_WireRecord], Obj] = {
    SampledMessageV1: _f(
        content=_INTAKE,
        tool_calls_was_list=_BOOL,
        calls=_JSON,
    ),
    WriterTurnV1: _f(
        action_id=_s(nonempty=True, logical=True),
        context_revision_ref=_h("context_revision"),
        request_ref=_h("artifact|bytes", optional=True),
        prepared_request_ref=_h("artifact", optional=True),
        raw_output_ref=_h("artifact|bytes", optional=True),
        usage=_o(
            {},
            {key: _i() for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
            extra=Open(),
        ),
        adapter_trace=UnionOf((_TRACE_SCHEMA, type(None))),
        message=RecordOf(SampledMessageV1),
    ),
    WriterRequestV1: _f(
        context_revision_ref=_h("context_revision"),
        payload_ref=_h("artifact|bytes"),
        verified_messages=_BOOL,
    ),
    ToolObservationV1: _f(
        call_id=_s(nonempty=True, logical=True),
        dispatch=UnionOf(
            (
                type(None),
                _f(
                    spec=_f(max_file_bytes=_i(1), max_workspace_bytes=_i(1)),
                    observation=_o(
                        {"ok": _BOOL, "valid": _BOOL},
                        {"result": _JSON, "error": _TEXT},
                    ),
                    effect=DictOf(
                        _f(
                            before=UnionOf((_TEXT, type(None))),
                            after=UnionOf((_TEXT, type(None))),
                        )
                    ),
                ),
            )
        ),
    ),
    AuthorReplyV1: _f(
        request_ref=_h("private"),
        status=_enum("answered", "unsupported_coverage"),
        utterance=UnionOf((_TEXT, type(None))),
        decision_ids=_l(_s(nonempty=True, logical=True), unique=True),
        selected_proposals=DictOf(_l(_s(nonempty=True, logical=True), unique=True)),
    ),
    EvaluatorResultV1: _f(
        request_ref=_h("private"),
        status=_enum("pass", "fail"),
        evidence_ref=_h("artifact"),
    ),
    ContextOperationInputV1: _f(policy_ref=_h("artifact")),
    EnvironmentStepV1: _f(
        directive=KindUnion(
            "kind",
            {
                "request_author": _f(
                    kind=_enum("request_author"),
                    source=_enum("writer_request", "mandatory_feedback"),
                ),
                "request_checks": _f(kind=_enum("request_checks")),
                "commit_transition": _f(
                    kind=_enum("commit_transition"), edge_id=_s(nonempty=True, logical=True)
                ),
                "seal_outcome": _f(
                    kind=_enum("seal_outcome"),
                    task_status=_TEXT,
                    stop_reason=UnionOf((_TEXT, type(None))),
                ),
                "stop_exhausted": _f(kind=_enum("stop_exhausted"), stop_reason=_s(nonempty=True)),
                "publish_reward": _f(kind=_enum("publish_reward")),
            },
        )
    ),
    MemberStartV1: _f(group_spec_ref=_h("artifact"), ordinal=_i()),
    ExternalInputsV1: _f(schema=_i(1, equals=1), source_refs=_l(_h("artifact"))),
    AdmissionPolicyV1: _f(
        schema=_i(1, equals=1),
        writer_family=UnionOf((_s(nonempty=True), type(None))),
        allowed_tools=_l(_s(nonempty=True)),
        controller_versions=_l(_s(nonempty=True)),
        check_versions=_l(_s(nonempty=True)),
    ),
    OutcomeV1: _f(
        schema=_i(1, equals=1),
        task_status=_enum(*TASK_STATUSES),
        execution_status=_enum(*EXECUTION_STATUSES),
        stop_reason=UnionOf((_TEXT, type(None))),
        reward_status=_enum("pending", "available", "unavailable"),
        training_eligibility=_enum("pending", "eligible", "ineligible"),
        candidate_checkpoint=_h("checkpoint", optional=True),
        requirement_version=_h("private", optional=True),
        checks=_l(_f(request_ref=_h("private"), result_ref=_h("artifact", optional=True))),
        transition_edge_id=_NULLABLE_LOGICAL,
        failed_request_ref=_h("private", optional=True),
        reward_ref=_h("artifact", optional=True),
        eligibility_ref=_h("artifact", optional=True),
    ),
    ContextContentV1: _f(
        parent_ref=_h("context_node", optional=True),
        messages=_l(MessageValue()),
        tools=UnionOf((type(None), _l(_o({}, extra=_JSON)))),
        rendering=UnionOf(
            (
                type(None),
                _o(
                    _RENDERING_SCHEMA.required,
                ),
            )
        ),
    ),
    ContextRevisionV1: _f(
        content_ref=_h("context_node"),
        event_head=_h("event", optional=True),
        provenance_refs=_l(_h("event")),
    ),
    GroupMemberSpecV1: _f(
        member_id=_s(nonempty=True, logical=True),
        ordinal=_i(),
        writer_seed=_i(),
        environment_seed=_i(),
        seed_provenance=_s(nonempty=True),
    ),
    GroupSpecV1: _f(
        group_id=_h(None),
        group_sequence=_i(),
        group_seed=_i(),
        runner_mode=_enum("real", "fixture"),
        environment=_f(
            entry_checkpoint_id=_h("checkpoint"),
            entry_state_hash=_h(None),
            entry_tree_hash=_h(None),
            instance_hash=_h(None),
            graph_hash=_h(None),
            node_id=_s(nonempty=True, logical=True),
            node_visit_id=_s(nonempty=True, logical=True),
            node_contract_hash=_h(None),
            controller_contract_hash=_h(None),
            check_contracts_hash=_h(None),
            reward_contract_hash=_h(None, optional=True),
            simulator_contract_hash=_h(None, optional=True),
            source_refs_hash=_h(None),
            request_refs_hash=_h(None),
            visible_prefix_hash=_h(None),
            context_revision_ref=_h("context_revision"),
            context_messages_hash=_h(None),
            rendering_hash=_h(None),
            tool_schemas_hash=_h(None),
            budget_ref=_h("artifact"),
            budget_hash=_h(None),
            versions_ref=_h("artifact"),
            versions_hash=_h(None),
            author_packet_ref=_h("private", optional=True),
            requirements_ref=_h("private"),
            decisions_ref=_h("artifact"),
            disclosures_ref=_h("artifact"),
            external_inputs_ref=_h("artifact"),
            rng_ref=_h("artifact"),
            outcome_ref=_h("artifact"),
            provenance_ref=_h("artifact"),
            continuation_hash=_h(None),
            horizon=_s(nonempty=True),
        ),
        policy=_f(
            model_ref=_h("artifact"),
            behavior_policy_ref=_h("artifact"),
            tokenizer_ref=_h("artifact"),
            template_ref=_h("artifact"),
            adapter_ref=_h("artifact"),
            decoding_ref=_h("artifact"),
            simulator_ref=_h("artifact"),
            context_policy_ref=_h("artifact"),
            controller_ref=_h("artifact"),
            rng_derivation_version=_s(nonempty=True),
        ),
        members=_l(RecordOf(GroupMemberSpecV1)),
    ),
    ContextPolicyV1: _f(
        operation=_enum("carry", "seed", "drop", "compact"),
        retained_exchanges=_i(),
        seed_name=UnionOf((_NULLABLE_TEXT, type(None))),
        seed_checkpoint_ref=_h("checkpoint", optional=True),
        summarizer_version=UnionOf((_NULLABLE_TEXT, type(None))),
        max_summary_chars=UnionOf((_i(), type(None))),
        max_operations=_i(),
        max_context_bytes=_i(),
        max_context_storage_bytes=_i(),
        schema=_i(1, equals=1),
    ),
    ExecutionVersionsV1: _f(
        schema=_i(1, equals=1),
        transition_semantics=_enum(SEMANTICS_V1),
        admission_policy_ref=_h("artifact"),
        tool_spec=_f(max_file_bytes=_i(1), max_workspace_bytes=_i(1)),
    ),
}


def _validate_schema_fields(record: _WireRecord) -> None:
    spec = record.FIELD_SPEC
    values = {item.name: getattr(record, item.name) for item in fields(record)}
    if set(values) != set(spec.required):
        raise TypeError(f"{type(record).__name__} field declarations differ from its spec")
    _validate_spec(spec, values, type(record).__name__)
    for rule in _CLASS_RULES.get(type(record), ()):
        rule(record)


_CLASS_RULES: dict[type[_WireRecord], tuple[Any, ...]] = {
    SampledMessageV1: (_sampled_rules,),
    WriterTurnV1: (_writer_turn_rules,),
    ToolObservationV1: (_tool_observation_rules,),
    AuthorReplyV1: (_author_reply_rules,),
    EnvironmentStepV1: (_directive_rules,),
    ExternalInputsV1: (_schema_one,),
    AdmissionPolicyV1: (_schema_one, _admission_policy_rules),
    OutcomeV1: (_schema_one,),
    ContextContentV1: (_context_content_rules,),
    GroupMemberSpecV1: (_group_member_rules,),
    GroupSpecV1: (_group_spec_rules,),
    ContextPolicyV1: (_context_policy_rules,),
    ExecutionVersionsV1: (_schema_one,),
}

for _record, _field_spec in _CLASS_FIELD_SPECS.items():
    _record.FIELD_SPEC = _field_spec
    _record.REFS = _derived_refs(_field_spec)


def _payload(record_type: str, **field_specs: Any) -> _PayloadCodec:
    return _PayloadCodec(record_type, _f(**field_specs))


_AUTHOR_PREREQUISITES = DictOf(_f(result_ref=_h("artifact"), status=_enum("pass", "fail")))
_REWARD_COMPONENTS = DictOf(_f(weight=_i(), earned=_i(), status=_enum("pass", "fail", "not_run")))
_PAYLOAD_RECORD_CODECS: Mapping[str, _PayloadCodec] = MappingProxyType(
    {
        "DecisionLedgerV1": _payload(
            "DecisionLedgerV1",
            schema=_i(1, equals=1),
            values=DictOf(_JSON),
            proposals=DictOf(_JSON),
        ),
        "DisclosureLedgerV1": _payload(
            "DisclosureLedgerV1", schema=_i(1, equals=1), decisions=_l(_JSON)
        ),
        "AuthorRequestV1": _payload(
            "AuthorRequestV1",
            schema=_i(1, equals=1),
            request_id=_s(nonempty=True, logical=True),
            source=_enum("writer_request", "mandatory_feedback"),
            action_id=_NULLABLE_LOGICAL,
            call_id=_NULLABLE_LOGICAL,
            feedback_id=_NULLABLE_LOGICAL,
            arguments=UnionOf((_JSON, type(None))),
            decision_ids=_l(_s(nonempty=True, logical=True)),
            prerequisite_results=_AUTHOR_PREREQUISITES,
            requirement_version=_h("artifact|private"),
            script_ref=_h("private"),
            author_packet_ref=_h("private", optional=True),
        ),
        "CheckRequestV1": _payload(
            "CheckRequestV1",
            schema=_i(1, equals=1),
            request_id=_s(nonempty=True, logical=True),
            target_checkpoint=_h("checkpoint"),
            requirement_version=_h("private"),
            check_contract_hash=_h("private"),
            evaluator_packet_ref=_h("private", optional=True),
            check_id=_s(nonempty=True, logical=True),
            purpose=_enum("progress", "completion"),
        ),
        "RewardV1": _payload(
            "RewardV1",
            schema=_i(1, equals=1),
            terminal_outcome_ref=_h("artifact"),
            reward_contract_ref=_h("private"),
            candidate_checkpoint=_h("checkpoint", optional=True),
            check_result_refs=_l(_h("artifact")),
            components=_REWARD_COMPONENTS,
            numerator=_i(),
            normalization=_i(1),
            availability=_enum("available", "unavailable"),
            eligibility_ref=_h("artifact"),
        ),
        "TrainingEligibilityV1": _payload(
            "TrainingEligibilityV1",
            schema=_i(1, equals=1),
            terminal_outcome_ref=_h("artifact"),
            status=_enum("eligible", "ineligible"),
            reason=_s(nonempty=True),
        ),
        "GroupMemberSeedsV1": _payload(
            "GroupMemberSeedsV1",
            group_id=_h(None),
            member_id=_s(nonempty=True, logical=True),
            derivation=_s(nonempty=True),
            writer_seed=_i(),
            environment_seed=_i(),
            parent_rng_ref=_h("artifact"),
        ),
        "RequirementLedgerV1": _payload(
            "RequirementLedgerV1",
            schema=_i(1, equals=1),
            active=DictOf(_TEXT),
            superseded=DictOf(_TEXT),
        ),
    }
)

_EXECUTION_VERSIONS_CODEC = _PayloadCodec(
    "ExecutionVersionsV1", _CLASS_FIELD_SPECS[ExecutionVersionsV1], tagged=False
)


_WIRE_CLASSES = tuple(codec for codec in _CLASS_FIELD_SPECS if codec.RECORD_TYPE is not None)
RECORD_TYPES: Mapping[str, type[_WireRecord] | _PayloadCodec] = MappingProxyType(
    {
        **{record.RECORD_TYPE: record for record in _WIRE_CLASSES},
        **dict(_PAYLOAD_RECORD_CODECS),
        "ExecutionVersionsV1": _EXECUTION_VERSIONS_CODEC,
    }
)

SHARED_WIRE_V1_RECORD_TYPES = frozenset(
    {
        "AuthorRequestV1",
        "CheckRequestV1",
        "DecisionLedgerV1",
        "DisclosureLedgerV1",
        "RequirementLedgerV1",
        "RewardV1",
        "TrainingEligibilityV1",
        "GroupMemberSeedsV1",
    }
)

# These names are records the new runtime does not write and which have not
# yet earned an L0 codec. GroupDecisionV1 remains because the group coordinator
# persists its to_dict() payload; RuntimePortDescriptorV1 is nested only.
LEGACY_PAYLOAD_RECORD_TYPES = frozenset(
    {
        "AuthorToolAckV1",
        "AuthorTurnV1",
        "CheckBatchV1",
        "CheckResultV1",
        "ContextOperationV1",
        "DecisionDisclosureV1",
        "DeterministicCheckEvidenceV1",
        "FixtureFileCountEvidenceV1",
        "GroupExecutionFailureV1",
        "GroupAdvantageV1",
        "GroupDecisionV1",
        "GroupMemberResultV1",
        "GroupSegmentCreditV1",
        "GroupScriptedTerminalV1",
        "InfrastructureInvalidV1",
        "PreparedWriterRequestV1",
        "RequirementSupersessionV1",
        "RewardAvailabilityV1",
        "RewardPublicationV1",
        "RuntimeManifestV1",
        "ScriptCoverageFailureV1",
        "ScriptedAuthorReplyV1",
        "TerminalOutcomeCommitV1",
        "TerminalOutcomeV1",
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
        record = codec.from_dict(dict(body))
        value = record.to_wire()
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
    """Small helper shared with the `REFS` completeness test."""
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None
