"""Immutable Phase 1 task-graph records and canonical identities.

This module intentionally contains data contracts only.  It has no store, model,
filesystem worker, or runtime loop.  Records are frozen values and can be encoded
as canonical JSON v1 for content-addressed storage.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar

CANONICAL_VERSION = 1
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_DOMAINS = {
    "file": b"task-graph:file:v1\0",
    "tree": b"task-graph:tree:v1\0",
    "state": b"task-graph:state:v1\0",
    "event": b"task-graph:event:v1\0",
    "context": b"task-graph:context:v1\0",
    "context_content": b"task-graph:context-content:v1\0",
    "checkpoint": b"task-graph:checkpoint:v1\0",
    "commit": b"task-graph:commit:v1\0",
    "instance": b"task-graph:instance:v1\0",
    "lineage": b"task-graph:lineage:v1\0",
    "message": b"task-graph:message:v1\0",
    # Payloads are immutable event inputs (action traces and tool results).
    # They are intentionally separate from event identities: an event binds a
    # payload by reference, while the payload can be hashed before that event.
    "payload": b"task-graph:payload:v1\0",
}
_DOMAINS.update({f"{name}:bytes": tag + b"bytes\0" for name, tag in tuple(_DOMAINS.items())})
DOMAIN_TAGS = MappingProxyType(_DOMAINS)

EVENT_KINDS = frozenset(
    {
        "rollout_started",
        "writer_action",
        "tool_result",
        "author_turn",
        "external_requested",
        "request_entered",
        "seed_attached",
        "requirements_changed",
        "decision_disclosed",
        "check_recorded",
        "transition_committed",
        "termination_recorded",
        "context_changed",
        "fetch_recorded",
        "external_response",
        "budget_charged",
        "reward_recorded",
    }
)
TASK_STATUSES = frozenset({"complete", "accepted_partial", "incomplete", "unknown"})
EXECUTION_STATUSES = frozenset(
    {
        "running",
        "valid",
        "interrupted",
        "environment_error",
        "backend_error",
        "simulator_error",
        "controller_error",
        "external_cancelled",
    }
)


class Phase(StrEnum):
    """Phases used by the transition-seam controller."""

    READY_WRITER = "ready_writer"
    CHECKING = "checking"
    AWAITING_AUTHOR = "awaiting_author"
    AWAITING_CHECKS = "awaiting_checks"
    READY_TRANSITION = "ready_transition"
    TERMINAL = "terminal"


def utf8(value: str, label: str = "string") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8") from exc
    return value


def logical_id(value: str, label: str = "logical id") -> str:
    value = utf8(value, label)
    if not value or any(ch.isspace() or ord(ch) < 0x20 for ch in value):
        raise ValueError(f"invalid {label}")
    return value


def _action_id(value: str) -> str:
    """Validate a stable logical action identifier, not an event hash."""
    value = logical_id(value, "action id")
    if _HASH_RE.fullmatch(value):
        raise ValueError("action id must be logical, not a SHA-256 hash")
    return value


def _tool_result_id(value: str) -> str:
    """Validate a stable logical tool-result identifier, not an event hash.

    Tool results are indexed by their logical result ID (the call/ordinal
    identity assigned by the rollout).  The result event's SHA-256 is a
    separate reference and must not be substituted here.
    """
    value = logical_id(value, "tool result id")
    if _HASH_RE.fullmatch(value):
        raise ValueError("tool result id must be logical, not a SHA-256 hash")
    return value


def _json_value(value: Any) -> Any:
    """Convert frozen records to JSON-compatible values without losing order."""
    kind = type(value)
    if kind is str or kind is int or kind is bool or value is None:
        return value
    if is_dataclass(value):
        if not isinstance(value, Record):
            raise TypeError("arbitrary dataclasses are not canonical values")
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def _wire_value(value: Any) -> None:
    """Reject Python-only constructor values at the wire boundary.

    JSON arrays are lists and objects are mappings with string keys.  Tuples,
    record instances, and other conveniences belong to direct Python
    construction; accepting them here would make ``from_dict`` normalize a
    non-wire value into a different byte representation.
    """
    if isinstance(value, Mapping):
        if type(value) is not dict:
            raise TypeError("wire objects must be plain dictionaries")
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("wire object keys must be strings")
            _wire_value(item)
    elif isinstance(value, list):
        for item in value:
            _wire_value(item)
    elif type(value) in (str, int, bool) or value is None:
        return
    else:
        raise TypeError("wire values must be JSON objects, arrays, or scalars")


def canonical_json(value: Any) -> str:
    """Return canonical JSON v1 (compact UTF-8-safe text, without a newline)."""
    if isinstance(value, Record) and type(value).to_dict is Record.to_dict:
        return value._canonical().decode("utf-8")
    text = _dumps(_json_value(value))
    _strict_utf8(text)
    return text


def _dumps(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _strict_utf8(text: str) -> bytes:
    try:
        return text.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise ValueError("string must be valid UTF-8") from exc


def _no_float(text: str) -> Any:
    raise ValueError("canonical JSON does not permit floats")


def canonical_bytes(value: Any) -> bytes:
    if isinstance(value, Record) and type(value).to_dict is Record.to_dict:
        return value._canonical()
    return canonical_json(value).encode("utf-8", "strict")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_canonical_json(data: str | bytes) -> Any:
    """Load canonical JSON and reject duplicate keys, floats, and noncanonical bytes."""
    try:
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    except (UnicodeEncodeError, TypeError) as exc:
        raise ValueError("invalid canonical JSON") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_float=_no_float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid canonical JSON") from exc
    # json.loads yields only dict/list/str/int/bool/None with string keys and
    # (with parse_float) no floats, so it can be re-serialized directly.
    if _strict_utf8(_dumps(value)) != raw:
        raise ValueError("JSON is not canonical JSON v1")
    return value


def domain_hash(domain: str, value: Any) -> str:
    if domain == "file":
        if isinstance(value, str):
            # Keep the legacy spelling only for the one unambiguous file
            # codec. Structured values must not be silently treated as bytes.
            return file_hash(value)
        raise TypeError("the file domain requires text; use file_hash(text)")
    if domain.endswith(":bytes"):
        raise ValueError("binary domains require domain_hash_bytes")
    try:
        tag = _DOMAINS[domain]
    except KeyError as exc:
        raise ValueError(f"unknown identity domain: {domain}") from exc
    return hashlib.sha256(tag + canonical_bytes(value)).hexdigest()


def domain_hash_bytes(domain: str, value: bytes) -> str:
    """Hash an opaque binary artifact under an explicit ``:bytes`` domain.

    ``domain`` is the base domain name (for example ``"payload"``); this
    helper appends ``":bytes"`` itself. Passing a name that already ends in
    ``":bytes"`` is rejected. This is not the file codec: text files must use
    :func:`file_hash`, which hashes their exact UTF-8 bytes under the file
    domain. Structured JSON and opaque bytes intentionally use separate tags,
    so callers must choose this helper rather than accidentally hashing bytes
    as JSON.
    """
    try:
        tag = _DOMAINS[f"{domain}:bytes"]
    except KeyError as exc:
        raise ValueError(f"unknown identity domain: {domain}") from exc
    if not isinstance(value, bytes):
        raise TypeError("binary identity input must be bytes")
    return hashlib.sha256(tag + value).hexdigest()


def validate_hash(value: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError("expected a lowercase SHA-256 hash")
    return value


def safe_path(path: str) -> str:
    """Validate a canonical relative POSIX path (without normalising it)."""
    utf8(path, "path")
    if not isinstance(path, str) or not path or "\\" in path or path.startswith("/"):
        raise ValueError("unsafe relative path")
    pieces = path.split("/")
    if any(piece in {"", ".", ".."} for piece in pieces):
        raise ValueError("unsafe relative path")
    if "\x00" in path:
        raise ValueError("NUL is not allowed in paths")
    return path


def validate_file_tree(files: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(files, Mapping):
        raise TypeError("files must be a mapping")
    result: dict[str, str] = {}
    for path, text in files.items():
        safe_path(path)
        if not isinstance(text, str):
            raise TypeError("file contents must be text")
        try:
            text.encode("utf-8", "strict")
        except UnicodeEncodeError as exc:
            raise ValueError("file contents must be valid UTF-8") from exc
        if path in result:
            raise ValueError("duplicate file path")
        result[path] = text
    paths = sorted(result)
    # Paths are deliberately case-sensitive (the target runtime is Linux).
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if other.startswith(path + "/"):
                raise ValueError("a file cannot also be a directory prefix")
            if not other.startswith(path):
                break
    return result


def file_hash(text: str) -> str:
    """Hash the exact UTF-8 file bytes (not a JSON representation)."""
    if not isinstance(text, str):
        raise TypeError("file content must be text")
    raw = text.encode("utf-8", "strict")
    return hashlib.sha256(_DOMAINS["file"] + raw).hexdigest()


def tree_hash(files: Mapping[str, str]) -> str:
    files = validate_file_tree(files)
    entries = [[path, file_hash(files[path])] for path in sorted(files)]
    return domain_hash("tree", entries)


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(v) for v in value)
    return value


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Record:
    schema: int = 1
    DOMAIN: ClassVar[str] = "state"

    def __post_init__(self) -> None:
        if type(self.schema) is not int or self.schema != CANONICAL_VERSION:
            raise ValueError(f"unsupported {type(self).__name__} schema")
        # One walk: _json_value rejects floats/unsupported values/non-string
        # keys; the strict encode of its serialization rejects lone surrogates.
        _strict_utf8(_dumps(_json_value(self)))
        for field in fields(self):
            if field.name != "schema":
                object.__setattr__(self, field.name, freeze(getattr(self, field.name)))
        self.validate()

    def validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        # _json_value already builds fresh dict/list containers.
        return _json_value(self)

    # Records are frozen after __post_init__ (every field is frozen by freeze
    # into tuples/MappingProxyType over private dicts, or is itself a frozen
    # record), so their canonical bytes and identity are pure functions of the
    # instance. The cache lives in the instance __dict__, outside dataclass
    # fields, so eq/hash/repr and wire bytes are unchanged.
    def _canonical(self) -> bytes:
        memoizable = type(self).to_dict is Record.to_dict
        cached = self.__dict__.get("_canonical_cache") if memoizable else None
        if cached is not None:
            return cached
        encoded = canonical_json(self.to_dict()).encode("utf-8", "strict")
        if memoizable:
            self.__dict__["_canonical_cache"] = encoded
        return encoded

    def to_json(self) -> str:
        return canonical_json(self)

    def identity(self) -> str:
        cached = self.__dict__.get("_identity_cache")
        if cached is not None:
            return cached
        memoizable = type(self).to_dict is Record.to_dict
        if memoizable and not isinstance(self, (EventV1, LineageRefV1)):
            cached = hashlib.sha256(_DOMAINS[self.DOMAIN] + self._canonical()).hexdigest()
            self.__dict__["_identity_cache"] = cached
            return cached
        body = self.to_dict()
        # EventV1 carries its address for interchange, but an identity never
        # hashes the field that contains that identity.
        if isinstance(self, EventV1):
            body.pop("id", None)
        if isinstance(self, LineageRefV1):
            # expected_head is a compare-and-swap request, not lineage authority.
            body.pop("expected_head", None)
        identity = domain_hash(self.DOMAIN, body)
        if memoizable:
            self.__dict__["_identity_cache"] = identity
        return identity

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping):
            raise TypeError("record must be an object")
        _wire_value(value)
        allowed = {field.name for field in fields(cls)}
        required = {field.name for field in fields(cls)}
        missing = required - set(value)
        if missing:
            raise ValueError(f"missing required fields: {sorted(missing)}")
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        kwargs = dict(value)
        record = cls(**kwargs)
        # Constructors intentionally remain ergonomic for Python callers, but
        # the wire decoder must be byte-preserving and may not normalize
        # shorthands (for example, a text string into a typed message part).
        if record.to_dict() != dict(value):
            raise ValueError("wire record is not already in canonical typed form")
        return record

    @classmethod
    def from_json(cls, data: str | bytes):
        if isinstance(data, str):
            raw = data.encode("utf-8", "strict")
        elif isinstance(data, bytes):
            raw = data
        else:
            raise TypeError("JSON input must be str or bytes")
        record = cls.from_dict(load_canonical_json(raw))
        if record._canonical() != raw:
            raise ValueError("decoded record did not preserve canonical bytes")
        return record


def _hash_tuple(values: Any, *, optional: bool = False) -> None:
    if not isinstance(values, (tuple, list)):
        raise TypeError("expected an array")
    for value in values:
        validate_hash(value, optional=optional)


def _logical_tuple(values: Any, validator, label: str) -> None:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{label} must be an array")
    for value in values:
        validator(value)


@dataclass(frozen=True)
class NodeSpecV1(Record):
    id: str = ""
    kind: str = "writer"
    families: tuple[str, ...] = ()
    entry_contract: str = ""
    exits: tuple[Mapping[str, Any], ...] = ()
    DOMAIN: ClassVar[str] = "instance"

    def validate(self) -> None:
        logical_id(self.id, "node id")
        if self.kind not in {"writer", "environment"}:
            raise ValueError("invalid node kind")
        if not isinstance(self.families, tuple) or any(
            not isinstance(family, str) for family in self.families
        ):
            raise TypeError("node families must be strings")
        if self.kind == "environment" and self.families:
            raise ValueError("environment nodes cannot declare families")
        validate_hash(self.entry_contract)
        if not isinstance(self.exits, tuple) or any(
            not isinstance(exit_spec, Mapping) for exit_spec in self.exits
        ):
            raise TypeError("node exits must be objects")


@dataclass(frozen=True)
class GraphInstanceV1(Record):
    template_ref: str = ""
    entry_node: str = ""
    nodes: tuple[NodeSpecV1 | Mapping[str, Any], ...] = ()
    source_refs: tuple[str, ...] = ()
    request_refs: tuple[str, ...] = ()
    requirements_ref: str | None = None
    budgets: Mapping[str, int] = dataclass_field(default_factory=dict)
    DOMAIN: ClassVar[str] = "instance"

    def __post_init__(self) -> None:
        nodes = tuple(
            node if isinstance(node, NodeSpecV1) else NodeSpecV1(**dict(node))
            for node in self.nodes
        )
        object.__setattr__(self, "nodes", nodes)
        super().__post_init__()

    def validate(self) -> None:
        logical_id(self.entry_node, "entry node")
        validate_hash(self.template_ref)
        # Validate the container before iterating: strings/mappings are
        # iterable Python values but are not JSON arrays and must never be
        # interpreted as a sequence of hash references.
        _hash_tuple(self.source_refs)
        _hash_tuple(self.request_refs)
        validate_hash(self.requirements_ref, optional=True)
        if not isinstance(self.budgets, Mapping):
            raise TypeError("budgets must be an object")
        if any(
            not isinstance(k, str) or not isinstance(v, int) or isinstance(v, bool) or v < 0
            for k, v in self.budgets.items()
        ):
            raise ValueError("budgets must be nonnegative integer values")
        ids = []
        for node in self.nodes:
            ids.append(node.id)
        if ids and (len(ids) != len(set(ids)) or self.entry_node not in ids):
            raise ValueError("graph nodes must be unique and include entry_node")


@dataclass(frozen=True)
class MessageV1(Record):
    role: str = "user"
    content: tuple[Any, ...] = ()
    call_id: str | None = None
    origin: str = ""
    trust: str = "untrusted_data"
    loss_eligible: bool = False
    DOMAIN: ClassVar[str] = "message"

    def __post_init__(self) -> None:
        parts = []
        for part in self.content:
            if isinstance(part, str):
                parts.append({"type": "text", "text": part})
            elif isinstance(part, Mapping):
                item = dict(part)
                kind = item.get("type")
                if (
                    kind == "text"
                    and set(item) == {"type", "text"}
                    and isinstance(item["text"], str)
                ):
                    parts.append(item)
                elif kind == "tool_call" and set(item) == {"type", "id", "name", "arguments"}:
                    logical_id(item["id"], "tool call id")
                    logical_id(item["name"], "tool name")
                    if not isinstance(item["arguments"], Mapping):
                        raise TypeError("tool call arguments must be an object")
                    parts.append(item)
                elif kind == "tool_result" and set(item) == {"type", "call_id", "content"}:
                    logical_id(item["call_id"], "tool result call id")
                    if not isinstance(item["content"], (str, Mapping, list, tuple)):
                        raise TypeError("tool result content has invalid shape")
                    parts.append(item)
                elif kind == "invalid_tool_call" and set(item) == {"type", "id", "raw"}:
                    logical_id(item["id"], "invalid tool call id")
                    _json_value(item["raw"])
                    parts.append(item)
                else:
                    raise ValueError("invalid message part shape")
            else:
                raise TypeError("message parts must be text or typed objects")
        object.__setattr__(self, "content", tuple(parts))
        super().__post_init__()

    def validate(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("invalid message role")
        if not isinstance(self.content, tuple):
            raise TypeError("message content must be an array")
        if self.call_id is not None and (not isinstance(self.call_id, str) or not self.call_id):
            raise ValueError("invalid call_id")
        if not self.origin:
            raise ValueError("message origin is required")
        logical_id(self.origin, "message origin")
        if self.call_id is not None:
            logical_id(self.call_id, "call_id")
        if self.trust not in {"instructions", "untrusted_data"}:
            raise ValueError("invalid message trust")
        if not isinstance(self.loss_eligible, bool):
            raise TypeError("loss_eligible must be bool")
        for part in self.content:
            if not isinstance(part, Mapping) or not isinstance(part.get("type"), str):
                raise ValueError("invalid message part shape")


@dataclass(frozen=True)
class EventV1(Record):
    id: str | None = None
    previous: str | None = None
    seq: int = 1
    lineage_id: str = ""
    rollout_id: str | None = None
    node_visit_id: str | None = None
    kind: str = ""
    actor: str = "environment"
    audience: tuple[str, ...] = ()
    payload_ref: str = ""
    caused_by: tuple[str, ...] = ()
    versions_ref: str = ""
    provenance_ref: str = ""
    DOMAIN: ClassVar[str] = "event"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.id is None:
            body = self.to_dict()
            body.pop("id", None)
            object.__setattr__(self, "id", domain_hash(self.DOMAIN, body))
        elif self.id != self.identity():
            raise ValueError("event id does not match its canonical body")

    def validate(self) -> None:
        validate_hash(self.id, optional=True)
        validate_hash(self.previous, optional=True)
        if not isinstance(self.seq, int) or isinstance(self.seq, bool) or self.seq < 1:
            raise ValueError("event sequence must be positive")
        if self.previous is None and self.seq != 1:
            raise ValueError("an event without a predecessor must have sequence 1")
        if self.previous is not None and self.seq <= 1:
            raise ValueError("an event with a predecessor must have sequence greater than 1")
        logical_id(self.lineage_id, "event lineage_id")
        if self.kind not in EVENT_KINDS:
            raise ValueError("unknown event kind")
        if self.actor not in {"writer", "author", "environment", "evaluator", "writer_runtime"}:
            raise ValueError("invalid event actor")
        for value, label in (
            (self.rollout_id, "rollout_id"),
            (self.node_visit_id, "node_visit_id"),
        ):
            if value is not None:
                logical_id(value, label)
        if (
            not isinstance(self.audience, tuple)
            or not self.audience
            or any(not isinstance(audience, str) for audience in self.audience)
        ):
            raise TypeError("event audience must be an array")
        if len(set(self.audience)) != len(self.audience):
            raise ValueError("event audience must be nonempty and unique")
        if tuple(sorted(self.audience)) != self.audience:
            raise ValueError("event audience must be sorted")
        if any(
            a not in {"writer", "author", "controller", "evaluator", "trainer"}
            for a in self.audience
        ):
            raise ValueError("invalid audience")
        validate_hash(self.payload_ref)
        _hash_tuple(self.caused_by)
        validate_hash(self.versions_ref)
        validate_hash(self.provenance_ref)


@dataclass(frozen=True)
class ContextContentV1:
    """One immutable node in the writer-visible context chain."""

    parent_ref: str | None
    messages: tuple[MessageV1 | Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...] | None
    rendering: Mapping[str, Any] | None
    DOMAIN: ClassVar[str] = "context_content"
    RECORD_TYPE: ClassVar[str | None] = None
    EDGE_TYPE: ClassVar[str] = "context_node"
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "parent_ref": "context_node",
            "rendering.template_ref": "artifact",
            "rendering.tokenizer_ref": "artifact",
            "rendering.tool_schema_ref": "artifact",
        }
    )

    def __post_init__(self) -> None:
        messages = tuple(
            message if isinstance(message, MessageV1) else MessageV1.from_dict(dict(message))
            for message in self.messages
        )
        object.__setattr__(self, "messages", messages)
        if self.tools is not None:
            object.__setattr__(self, "tools", tuple(freeze(tool) for tool in self.tools))
        if self.rendering is not None:
            object.__setattr__(self, "rendering", freeze(self.rendering))
        object.__setattr__(self, "parent_ref", self.parent_ref)
        self.validate()

    def validate(self) -> None:
        validate_hash(self.parent_ref, optional=True)
        if (self.tools is None) != (self.rendering is None) or (self.parent_ref is None) != (
            self.tools is not None
        ):
            raise ValueError("root context content alone must carry tools and rendering")
        if not isinstance(self.messages, tuple) or any(
            not isinstance(message, MessageV1) for message in self.messages
        ):
            raise TypeError("context messages must be MessageV1 records")
        if self.tools is not None and any(not isinstance(tool, Mapping) for tool in self.tools):
            raise TypeError("context tools must be objects")
        if self.rendering is not None:
            required = {
                "projection_version",
                "prefix_id",
                "template_ref",
                "tokenizer_ref",
                "tool_schema_ref",
            }
            if not isinstance(self.rendering, Mapping) or set(self.rendering) != required:
                raise ValueError("context rendering has the wrong shape")
            utf8(self.rendering["projection_version"], "projection version")
            logical_id(self.rendering["prefix_id"], "prefix id")
            for name in ("template_ref", "tokenizer_ref", "tool_schema_ref"):
                validate_hash(self.rendering[name])

    def to_wire(self) -> dict[str, Any]:
        return {
            "parent_ref": self.parent_ref,
            "messages": [message.to_dict() for message in self.messages],
            "tools": None if self.tools is None else thaw(self.tools),
            "rendering": None if self.rendering is None else thaw(self.rendering),
        }

    to_dict = to_wire

    def identity(self) -> str:
        return domain_hash(self.DOMAIN, self.to_wire())

    def to_json(self) -> str:
        return canonical_json(self.to_wire())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContextContentV1:
        if not isinstance(value, Mapping) or set(value) != {
            "parent_ref",
            "messages",
            "tools",
            "rendering",
        }:
            raise ValueError("context content has the wrong schema")
        result = cls(
            parent_ref=value["parent_ref"],
            messages=tuple(value["messages"]),
            tools=None if value["tools"] is None else tuple(value["tools"]),
            rendering=value["rendering"],
        )
        if result.to_wire() != dict(value):
            raise ValueError("context content is not canonical")
        return result

    @classmethod
    def from_json(cls, data: str | bytes) -> ContextContentV1:
        raw = data.encode("utf-8", "strict") if isinstance(data, str) else data
        result = cls.from_dict(load_canonical_json(raw))
        if canonical_bytes(result.to_wire()) != raw:
            raise ValueError("context content is not canonical")
        return result


@dataclass(frozen=True)
class ContextRevisionV1:
    """Immutable reference to one materialized context node and its source history."""

    content_ref: str
    event_head: str | None
    provenance_refs: tuple[str, ...]
    DOMAIN: ClassVar[str] = "context"
    RECORD_TYPE: ClassVar[str | None] = None
    EDGE_TYPE: ClassVar[str] = "context_revision"
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "content_ref": "context_node",
            "event_head": "event",
            "provenance_refs[]": "event",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance_refs", tuple(self.provenance_refs))
        self.validate()

    def validate(self) -> None:
        validate_hash(self.content_ref)
        validate_hash(self.event_head, optional=True)
        _hash_tuple(self.provenance_refs)

    def to_wire(self) -> dict[str, Any]:
        return {
            "content_ref": self.content_ref,
            "event_head": self.event_head,
            "provenance_refs": list(self.provenance_refs),
        }

    to_dict = to_wire

    def identity(self) -> str:
        return domain_hash(self.DOMAIN, self.to_wire())

    def to_json(self) -> str:
        return canonical_json(self.to_wire())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContextRevisionV1:
        if not isinstance(value, Mapping) or set(value) != {
            "content_ref",
            "event_head",
            "provenance_refs",
        }:
            raise ValueError("context revision has the wrong schema")
        result = cls(value["content_ref"], value["event_head"], tuple(value["provenance_refs"]))
        if result.to_wire() != dict(value):
            raise ValueError("context revision is not canonical")
        return result

    @classmethod
    def from_json(cls, data: str | bytes) -> ContextRevisionV1:
        raw = data.encode("utf-8", "strict") if isinstance(data, str) else data
        result = cls.from_dict(load_canonical_json(raw))
        if canonical_bytes(result.to_wire()) != raw:
            raise ValueError("context revision is not canonical")
        return result


@dataclass(frozen=True)
class MaterializedContextV1:
    """Flattened, read-only projection of a chained context revision."""

    messages: tuple[MessageV1, ...]
    tools: tuple[Mapping[str, Any], ...]
    rendering: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [message.to_dict() for message in self.messages],
            "tools": thaw(self.tools),
            "rendering": thaw(self.rendering),
        }


@dataclass(frozen=True)
class EnvironmentStateV1(Record):
    instance_ref: str = ""
    position: Mapping[str, Any] = dataclass_field(default_factory=dict)
    files: Mapping[str, str] = dataclass_field(default_factory=dict)
    tree_hash: str = ""
    history: Mapping[str, Any] = dataclass_field(default_factory=dict)
    context_ref: str = ""
    requirements_ref: str = ""
    decisions_ref: str = ""
    disclosures_ref: str = ""
    author_packet_ref: str | None = None
    versions_ref: str = ""
    budgets_ref: str = ""
    rng_ref: str = ""
    external_inputs_ref: str = ""
    outcome_ref: str = ""
    provenance_ref: str = ""
    continuation: Mapping[str, Any] = dataclass_field(
        default_factory=lambda: {"tool_queue": (), "next_call": 0}
    )
    in_flight_effects: tuple[Any, ...] = ()
    DOMAIN: ClassVar[str] = "state"

    POSITION_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "node_id",
            "visit_id",
            "phase",
            "entry_contract",
            "start_checkpoint",
            "loop_counts",
            "lineage_id",
        }
    )
    HISTORY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"head", "seq", "branch_base", "imported_refs", "action_ids", "tool_result_ids"}
    )
    CONTINUATION_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "tool_queue",
            "next_call",
            "author_request",
            "check_requests",
            "external_requests",
            "applied_responses",
            "feedback_cursor",
        }
    )

    def validate(self) -> None:
        if not isinstance(self.position, Mapping) or set(self.position) != self.POSITION_FIELDS:
            raise TypeError("position must be an object")
        if not isinstance(self.history, Mapping) or set(self.history) != self.HISTORY_FIELDS:
            raise TypeError("history must be an object")
        if (
            not isinstance(self.continuation, Mapping)
            or set(self.continuation) != self.CONTINUATION_FIELDS
        ):
            raise TypeError("continuation must be an object")
        for ref in (
            self.instance_ref,
            self.context_ref,
            self.requirements_ref,
            self.decisions_ref,
            self.disclosures_ref,
            self.versions_ref,
            self.budgets_ref,
            self.rng_ref,
            self.external_inputs_ref,
            self.outcome_ref,
            self.provenance_ref,
        ):
            validate_hash(ref)
        validate_hash(self.author_packet_ref, optional=True)
        files = validate_file_tree(self.files)
        if self.tree_hash != tree_hash(files):
            raise ValueError("tree_hash does not match files")
        logical_id(self.position["node_id"], "node id")
        logical_id(self.position["visit_id"], "visit id")
        logical_id(self.position["lineage_id"], "position lineage id")
        validate_hash(self.position["entry_contract"])
        validate_hash(self.position["start_checkpoint"], optional=True)
        loop_counts = self.position["loop_counts"]
        if not isinstance(loop_counts, Mapping):
            raise TypeError("loop_counts must be an object")
        for loop_id, count in loop_counts.items():
            logical_id(loop_id, "loop id")
            if type(count) is not int or count < 0:
                raise ValueError("loop counts must be nonnegative integers")
        phase = self.position.get("phase")
        if phase not in {
            "ready_writer",
            "checking",
            "awaiting_author",
            "awaiting_checks",
            "ready_transition",
            "terminal",
        }:
            raise ValueError("invalid state phase")
        history_head = self.history["head"]
        validate_hash(history_head, optional=True)
        if type(self.history["seq"]) is not int or self.history["seq"] < 0:
            raise ValueError("invalid history sequence")
        if history_head is None and self.history["seq"] != 0:
            raise ValueError("empty history must have sequence 0")
        if history_head is not None and self.history["seq"] < 1:
            raise ValueError("nonempty history must have a positive sequence")
        validate_hash(self.history["branch_base"], optional=True)
        _hash_tuple(self.history["imported_refs"])
        _logical_tuple(self.history["action_ids"], _action_id, "action_ids")
        _logical_tuple(self.history["tool_result_ids"], _tool_result_id, "tool_result_ids")
        next_call = self.continuation["next_call"]
        if type(next_call) is not int or next_call < 0:
            raise ValueError("invalid continuation cursor")
        queue = self.continuation["tool_queue"]
        if not isinstance(queue, tuple):
            raise TypeError("tool_queue must be an array")
        if next_call > len(queue):
            raise ValueError("continuation cursor exceeds tool queue")
        call_ids: set[str] = set()
        for call in queue:
            if not isinstance(call, Mapping):
                raise TypeError("tool_queue entries must be objects")
            legacy_fields = {"call_id", "name", "arguments"}
            rejection_fields = {*legacy_fields, "rejection"}
            if frozenset(call) not in {
                frozenset(legacy_fields),
                frozenset(rejection_fields),
            }:
                raise ValueError("invalid tool call shape")
            call_id = logical_id(call["call_id"], "tool call id")
            if call_id in call_ids:
                raise ValueError("continuation tool_queue call IDs must be unique")
            call_ids.add(call_id)
            logical_id(call["name"], "tool name")
            if not isinstance(call["arguments"], Mapping):
                raise TypeError("tool arguments must be an object")
            if "rejection" in call:
                rejection = call["rejection"]
                if rejection is not None and not isinstance(rejection, str):
                    raise TypeError("tool rejection must be text or null")
                if rejection is not None and (
                    call["name"] != "invalid_call" or call["arguments"] != {}
                ):
                    raise ValueError("rejected calls must use invalid_call and empty arguments")
                if rejection is None and call["name"] == "invalid_call":
                    raise ValueError("invalid_call requires a rejection")
        validate_hash(self.continuation["author_request"], optional=True)
        refs = self.continuation["check_requests"]
        if not isinstance(refs, tuple):
            raise TypeError("check_requests must be an array")
        _hash_tuple(refs)
        for key in ("external_requests", "applied_responses"):
            refs = self.continuation[key]
            if not isinstance(refs, tuple):
                raise TypeError(f"{key} must be an array")
            for request_id in refs:
                logical_id(request_id, f"{key} id")
        if (
            type(self.continuation["feedback_cursor"]) is not int
            or self.continuation["feedback_cursor"] < 0
        ):
            raise ValueError("invalid feedback cursor")
        if not isinstance(self.in_flight_effects, tuple) or self.in_flight_effects:
            raise ValueError("checkpoint state cannot contain in-flight effects")


@dataclass(frozen=True)
class CheckpointV1(Record):
    parents: tuple[str, ...] = ()
    state: EnvironmentStateV1 | Mapping[str, Any] = None  # type: ignore[assignment]
    event_head: str | None = None
    artifact_refs: tuple[str, ...] = ()
    DOMAIN: ClassVar[str] = "checkpoint"

    def __post_init__(self) -> None:
        if not isinstance(self.state, EnvironmentStateV1):
            object.__setattr__(self, "state", EnvironmentStateV1.from_dict(self.state))
        super().__post_init__()

    def validate(self) -> None:
        if len(self.parents) > 1:
            raise ValueError("v1 checkpoints have at most one parent")
        _hash_tuple(self.parents)
        state = (
            self.state
            if isinstance(self.state, EnvironmentStateV1)
            else EnvironmentStateV1.from_dict(self.state)
        )
        validate_hash(self.event_head, optional=True)
        validate_hash(state.history.get("head"), optional=True)
        if state.history.get("head") != self.event_head:
            raise ValueError("checkpoint event_head disagrees with state history")
        _hash_tuple(self.artifact_refs)


@dataclass(frozen=True)
class CommitV1(Record):
    parent_commit: str | None = None
    events: tuple[str, ...] = ()
    checkpoint: str = ""
    DOMAIN: ClassVar[str] = "commit"

    def validate(self) -> None:
        validate_hash(self.parent_commit, optional=True)
        _hash_tuple(self.events)
        validate_hash(self.checkpoint)


@dataclass(frozen=True)
class LineageRefV1(Record):
    lineage_id: str = ""
    head_commit: str | None = None
    expected_head: str | None = None
    DOMAIN: ClassVar[str] = "lineage"

    def validate(self) -> None:
        logical_id(self.lineage_id, "lineage_id")
        validate_hash(self.head_commit, optional=True)
        validate_hash(self.expected_head, optional=True)


__all__ = [
    "CANONICAL_VERSION",
    "DOMAIN_TAGS",
    "EVENT_KINDS",
    "canonical_json",
    "canonical_bytes",
    "load_canonical_json",
    "domain_hash",
    "domain_hash_bytes",
    "validate_hash",
    "utf8",
    "logical_id",
    "freeze",
    "thaw",
    "Record",
    "safe_path",
    "validate_file_tree",
    "file_hash",
    "tree_hash",
    "GraphInstanceV1",
    "NodeSpecV1",
    "MessageV1",
    "EventV1",
    "ContextRevisionV1",
    "ContextContentV1",
    "MaterializedContextV1",
    "EnvironmentStateV1",
    "CheckpointV1",
    "CommitV1",
    "LineageRefV1",
]
