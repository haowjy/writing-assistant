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
    "checkpoint": b"task-graph:checkpoint:v1\0",
    "commit": b"task-graph:commit:v1\0",
    "instance": b"task-graph:instance:v1\0",
    "lineage": b"task-graph:lineage:v1\0",
    "message": b"task-graph:message:v1\0",
}
_DOMAINS.update({f"{name}:bytes": tag + b"bytes\0" for name, tag in tuple(_DOMAINS.items())})
DOMAIN_TAGS = MappingProxyType(_DOMAINS)

EVENT_KINDS = frozenset(
    {
        "rollout_started",
        "writer_action",
        "tool_result",
        "author_request",
        "author_response",
        "context_revision",
        "check_result",
        "transition",
        "rollout_finished",
        "environment_error",
    }
)


def _utf8(value: str, label: str = "string") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be valid UTF-8") from exc
    return value


def _logical_id(value: str, label: str = "logical id") -> str:
    value = _utf8(value, label)
    if not value or any(ch.isspace() or ord(ch) < 0x20 for ch in value):
        raise ValueError(f"invalid {label}")
    return value


def _validate_utf8(value: Any) -> None:
    """Validate every string and object key recursively, without normalizing."""
    if isinstance(value, str):
        _utf8(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _utf8(key, "object key")
            _validate_utf8(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_utf8(item)


def _reject_float(value: Any) -> None:
    if isinstance(value, float):
        raise ValueError("canonical JSON does not permit floats")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            _reject_float(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_float(item)


def _json_value(value: Any) -> Any:
    """Convert frozen records to JSON-compatible values without losing order."""
    if is_dataclass(value):
        if not isinstance(value, _Record):
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


def canonical_json(value: Any) -> str:
    """Return canonical JSON v1 (compact UTF-8-safe text, without a newline)."""
    _validate_utf8(value)
    value = _json_value(value)
    _reject_float(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def canonical_bytes(value: Any) -> bytes:
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
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid canonical JSON") from exc
    _reject_float(value)
    if canonical_bytes(value) != raw:
        raise ValueError("JSON is not canonical JSON v1")
    return value


def domain_hash(domain: str, value: Any) -> str:
    if domain.endswith(":bytes"):
        raise ValueError("binary domains require domain_hash_bytes")
    try:
        tag = _DOMAINS[domain]
    except KeyError as exc:
        raise ValueError(f"unknown identity domain: {domain}") from exc
    return hashlib.sha256(tag + canonical_bytes(value)).hexdigest()


def domain_hash_bytes(domain: str, value: bytes) -> str:
    """Hash exact bytes under an explicitly binary-only domain.

    Structured JSON and raw bytes intentionally use separate tags.  Callers must
    opt into a ``:bytes`` domain; accidentally hashing the same bytes as JSON is
    therefore impossible.
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
    _utf8(path, "path")
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


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class _Record:
    schema: int = 1
    DOMAIN: ClassVar[str] = "state"

    def __post_init__(self) -> None:
        if type(self.schema) is not int or self.schema != CANONICAL_VERSION:
            raise ValueError(f"unsupported {type(self).__name__} schema")
        _validate_utf8(self.to_dict())
        for field in fields(self):
            if field.name != "schema":
                object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))
        _reject_float(self.to_dict())
        self.validate()

    def validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        return _thaw(_json_value(self))

    def to_json(self) -> str:
        return canonical_json(self)

    def identity(self) -> str:
        body = self.to_dict()
        # EventV1 carries its address for interchange, but an identity never
        # hashes the field that contains that identity.
        if isinstance(self, EventV1):
            body.pop("id", None)
        if isinstance(self, LineageRefV1):
            # expected_head is a compare-and-swap request, not lineage authority.
            body.pop("expected_head", None)
        return domain_hash(self.DOMAIN, body)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping):
            raise TypeError("record must be an object")
        allowed = {field.name for field in fields(cls)}
        required = {field.name for field in fields(cls)}
        missing = required - set(value)
        if missing:
            raise ValueError(f"missing required fields: {sorted(missing)}")
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        kwargs = dict(value)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, data: str | bytes):
        return cls.from_dict(load_canonical_json(data))


def _hash_tuple(values: Any, *, optional: bool = False) -> None:
    if not isinstance(values, (tuple, list)):
        raise TypeError("expected an array")
    for value in values:
        validate_hash(value, optional=optional)


@dataclass(frozen=True)
class NodeSpecV1(_Record):
    id: str = ""
    kind: str = "writer"
    families: tuple[str, ...] = ()
    entry_contract: str = ""
    exits: tuple[Mapping[str, Any], ...] = ()
    DOMAIN: ClassVar[str] = "instance"

    def validate(self) -> None:
        _logical_id(self.id, "node id")
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
class GraphInstanceV1(_Record):
    template_ref: str = ""
    entry_node: str = ""
    nodes: tuple[NodeSpecV1 | Mapping[str, Any], ...] = ()
    source_refs: tuple[str, ...] = ()
    request_refs: tuple[str, ...] = ()
    requirements_ref: str | None = None
    budgets: Mapping[str, int] = MappingProxyType({})
    DOMAIN: ClassVar[str] = "instance"

    def __post_init__(self) -> None:
        nodes = tuple(
            node if isinstance(node, NodeSpecV1) else NodeSpecV1(**dict(node))
            for node in self.nodes
        )
        object.__setattr__(self, "nodes", nodes)
        super().__post_init__()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping):
            raise TypeError("record must be an object")
        allowed = {field.name for field in fields(cls)}
        missing = allowed - set(value)
        unknown = set(value) - allowed
        if missing or unknown:
            raise ValueError(
                f"missing required fields: {sorted(missing)}"
                if missing
                else f"unknown fields: {sorted(unknown)}"
            )
        nodes = tuple(NodeSpecV1.from_dict(node) for node in value["nodes"])
        return cls(**{**dict(value), "nodes": nodes})

    def validate(self) -> None:
        _logical_id(self.entry_node, "entry node")
        validate_hash(self.template_ref)
        for ref in (*self.source_refs, *self.request_refs):
            validate_hash(ref)
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
class MessageV1(_Record):
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
                    _logical_id(item["id"], "tool call id")
                    _logical_id(item["name"], "tool name")
                    if not isinstance(item["arguments"], Mapping):
                        raise TypeError("tool call arguments must be an object")
                    parts.append(item)
                elif kind == "tool_result" and set(item) == {"type", "call_id", "content"}:
                    _logical_id(item["call_id"], "tool result call id")
                    if not isinstance(item["content"], (str, Mapping, list, tuple)):
                        raise TypeError("tool result content has invalid shape")
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
        _logical_id(self.origin, "message origin")
        if self.call_id is not None:
            _logical_id(self.call_id, "call_id")
        if self.trust not in {"instructions", "untrusted_data"}:
            raise ValueError("invalid message trust")
        if not isinstance(self.loss_eligible, bool):
            raise TypeError("loss_eligible must be bool")
        for part in self.content:
            if not isinstance(part, Mapping) or not isinstance(part.get("type"), str):
                raise ValueError("invalid message part shape")


@dataclass(frozen=True)
class EventV1(_Record):
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
        _logical_id(self.lineage_id, "event lineage_id")
        if self.kind not in EVENT_KINDS:
            raise ValueError("unknown event kind")
        if self.actor not in {"writer", "author", "environment", "evaluator"}:
            raise ValueError("invalid event actor")
        for value, label in (
            (self.rollout_id, "rollout_id"),
            (self.node_visit_id, "node_visit_id"),
        ):
            if value is not None:
                _logical_id(value, label)
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


_DEFAULT_RENDERING = MappingProxyType(
    {
        "projection_version": "v1",
        "prefix_id": "root",
        "template_ref": "0" * 64,
        "tokenizer_ref": "0" * 64,
        "tool_schema_ref": "0" * 64,
    }
)


@dataclass(frozen=True)
class ContextContentV1(_Record):
    """Provenance-free writer input identity."""

    messages: tuple[MessageV1 | Mapping[str, Any], ...] = ()
    tools: tuple[Mapping[str, Any], ...] = ()
    projection_version: str = "v1"
    prefix_id: str = "root"
    template_ref: str = "0" * 64
    tokenizer_ref: str = "0" * 64
    tool_schema_ref: str = "0" * 64
    DOMAIN: ClassVar[str] = "context"

    @classmethod
    def from_revision(cls, revision: ContextRevisionV1) -> ContextContentV1:
        if not isinstance(revision, ContextRevisionV1):
            raise TypeError("revision must be ContextRevisionV1")
        pins = revision.rendering
        return cls(
            messages=revision.messages,
            tools=revision.tools,
            projection_version=pins["projection_version"],
            prefix_id=pins["prefix_id"],
            template_ref=pins["template_ref"],
            tokenizer_ref=pins["tokenizer_ref"],
            tool_schema_ref=pins["tool_schema_ref"],
        )

    def __post_init__(self) -> None:
        messages = tuple(
            message if isinstance(message, MessageV1) else MessageV1(**dict(message))
            for message in self.messages
        )
        object.__setattr__(self, "messages", messages)
        super().__post_init__()

    def validate(self) -> None:
        if not self.messages:
            raise ValueError("context content requires messages")
        _utf8(self.projection_version, "projection version")
        _logical_id(self.prefix_id, "prefix id")
        validate_hash(self.template_ref)
        validate_hash(self.tokenizer_ref)
        validate_hash(self.tool_schema_ref)
        if not isinstance(self.tools, tuple):
            raise TypeError("tools must be an array")
        for tool in self.tools:
            if not isinstance(tool, Mapping):
                raise TypeError("tool definitions must be objects")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        obj = super().from_dict(value)
        for message in value["messages"]:
            MessageV1.from_dict(message)
        return obj


def context_content_hash(
    messages: tuple[MessageV1 | Mapping[str, Any], ...],
    *,
    tools: tuple[Mapping[str, Any], ...] = (),
    rendering: Mapping[str, str] | None = None,
) -> str:
    pins = _DEFAULT_RENDERING if rendering is None else rendering
    required = {
        "projection_version",
        "prefix_id",
        "template_ref",
        "tokenizer_ref",
        "tool_schema_ref",
    }
    if not isinstance(pins, Mapping) or set(pins) != required:
        raise ValueError("rendering must contain exactly the pinned inputs")
    content = ContextContentV1(
        messages=messages,
        tools=tools,
        projection_version=pins["projection_version"],
        prefix_id=pins["prefix_id"],
        template_ref=pins["template_ref"],
        tokenizer_ref=pins["tokenizer_ref"],
        tool_schema_ref=pins["tool_schema_ref"],
    )
    return content.identity()


@dataclass(frozen=True)
class ContextRevisionV1(_Record):
    messages: tuple[MessageV1 | Mapping[str, Any], ...] = ()
    tools: tuple[Mapping[str, Any], ...] = ()
    content_hash: str | None = None
    event_head: str | None = None
    provenance_refs: tuple[str, ...] = ()
    rendering: Mapping[str, str] = _DEFAULT_RENDERING
    DOMAIN: ClassVar[str] = "context"

    def __post_init__(self) -> None:
        messages = tuple(
            message if isinstance(message, MessageV1) else MessageV1(**dict(message))
            for message in self.messages
        )
        object.__setattr__(self, "messages", messages)
        super().__post_init__()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        obj = super().from_dict(value)
        # super() already validates the outer envelope; nested messages must be
        # full wire records rather than constructor shorthand.
        for message in value["messages"]:
            MessageV1.from_dict(message)
        return obj

    def validate(self) -> None:
        expected = context_content_hash(self.messages, tools=self.tools, rendering=self.rendering)
        if self.content_hash is None:
            object.__setattr__(self, "content_hash", expected)
        elif self.content_hash != expected:
            raise ValueError("content_hash does not match context content")
        validate_hash(self.content_hash)
        validate_hash(self.event_head, optional=True)
        _hash_tuple(self.provenance_refs)
        for message in self.messages:
            if not isinstance(message, MessageV1):
                MessageV1.from_dict(message)
        if not isinstance(self.tools, tuple) or any(
            not isinstance(tool, Mapping) for tool in self.tools
        ):
            raise TypeError("tools must be objects")
        if not isinstance(self.rendering, Mapping):
            raise TypeError("rendering must be an object")
        required = {
            "projection_version",
            "prefix_id",
            "template_ref",
            "tokenizer_ref",
            "tool_schema_ref",
        }
        if set(self.rendering) != required:
            raise ValueError("rendering must contain exactly the pinned inputs")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.rendering.items()):
            raise ValueError("rendering values must be strings")
        _utf8(self.rendering["projection_version"], "projection version")
        _logical_id(self.rendering["prefix_id"], "prefix id")
        for key in ("template_ref", "tokenizer_ref", "tool_schema_ref"):
            validate_hash(self.rendering[key])


@dataclass(frozen=True)
class EnvironmentStateV1(_Record):
    instance_ref: str = ""
    position: Mapping[str, Any] = MappingProxyType({})
    files: Mapping[str, str] = MappingProxyType({})
    tree_hash: str = ""
    history: Mapping[str, Any] = MappingProxyType({})
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
    continuation: Mapping[str, Any] = MappingProxyType({"tool_queue": (), "next_call": 0})
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
        _logical_id(self.position["node_id"], "node id")
        _logical_id(self.position["visit_id"], "visit id")
        _logical_id(self.position["lineage_id"], "position lineage id")
        validate_hash(self.position["entry_contract"])
        validate_hash(self.position["start_checkpoint"], optional=True)
        loop_counts = self.position["loop_counts"]
        if not isinstance(loop_counts, Mapping):
            raise TypeError("loop_counts must be an object")
        for loop_id, count in loop_counts.items():
            _logical_id(loop_id, "loop id")
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
        for key in ("imported_refs", "action_ids", "tool_result_ids"):
            _hash_tuple(self.history[key])
        next_call = self.continuation["next_call"]
        if type(next_call) is not int or next_call < 0:
            raise ValueError("invalid continuation cursor")
        queue = self.continuation["tool_queue"]
        if not isinstance(queue, tuple):
            raise TypeError("tool_queue must be an array")
        if next_call > len(queue):
            raise ValueError("continuation cursor exceeds tool queue")
        for call in queue:
            if not isinstance(call, Mapping):
                raise TypeError("tool_queue entries must be objects")
            if set(call) != {"call_id", "name", "arguments"}:
                raise ValueError("invalid tool call shape")
            _logical_id(call["call_id"], "tool call id")
            _logical_id(call["name"], "tool name")
            if not isinstance(call["arguments"], Mapping):
                raise TypeError("tool arguments must be an object")
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
                _logical_id(request_id, f"{key} id")
        if (
            type(self.continuation["feedback_cursor"]) is not int
            or self.continuation["feedback_cursor"] < 0
        ):
            raise ValueError("invalid feedback cursor")
        if not isinstance(self.in_flight_effects, tuple) or self.in_flight_effects:
            raise ValueError("checkpoint state cannot contain in-flight effects")


@dataclass(frozen=True)
class CheckpointV1(_Record):
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
class CommitV1(_Record):
    parent_commit: str | None = None
    events: tuple[str, ...] = ()
    checkpoint: str = ""
    DOMAIN: ClassVar[str] = "commit"

    def validate(self) -> None:
        validate_hash(self.parent_commit, optional=True)
        _hash_tuple(self.events)
        validate_hash(self.checkpoint)


@dataclass(frozen=True)
class LineageRefV1(_Record):
    lineage_id: str = ""
    head_commit: str | None = None
    expected_head: str | None = None
    DOMAIN: ClassVar[str] = "lineage"

    def validate(self) -> None:
        _logical_id(self.lineage_id, "lineage_id")
        validate_hash(self.head_commit, optional=True)
        validate_hash(self.expected_head, optional=True)


def record_hash(record: _Record) -> str:
    if not isinstance(record, _Record):
        raise TypeError("record_hash expects a task-graph record")
    return record.identity()


# Compatibility aliases for the pre-review prototype; new callers should use
# the canonical names above.  They are intentionally behavior-identical, not
# parallel identity implementations.
canonicalize = canonical_json
canonical_load = load_canonical_json
identity_hash = domain_hash
canonical_hash = domain_hash
hash_bytes = domain_hash_bytes
validate_path = safe_path
hash_file = file_hash
hash_tree = tree_hash
GraphInstance = GraphInstanceV1
Event = EventV1
Message = MessageV1
ContextRevision = ContextRevisionV1
EnvironmentState = EnvironmentStateV1
Checkpoint = CheckpointV1
Commit = CommitV1
LineageRef = LineageRefV1


__all__ = [
    "CANONICAL_VERSION",
    "DOMAIN_TAGS",
    "EVENT_KINDS",
    "canonical_json",
    "canonical_bytes",
    "load_canonical_json",
    "domain_hash",
    "domain_hash_bytes",
    "identity_hash",
    "canonical_hash",
    "hash_bytes",
    "record_hash",
    "validate_hash",
    "safe_path",
    "validate_path",
    "validate_file_tree",
    "canonicalize",
    "canonical_load",
    "file_hash",
    "tree_hash",
    "hash_file",
    "hash_tree",
    "GraphInstanceV1",
    "NodeSpecV1",
    "MessageV1",
    "EventV1",
    "ContextRevisionV1",
    "ContextContentV1",
    "context_content_hash",
    "GraphInstance",
    "Event",
    "Message",
    "ContextRevision",
    "EnvironmentState",
    "Checkpoint",
    "Commit",
    "LineageRef",
    "EnvironmentStateV1",
    "CheckpointV1",
    "CommitV1",
    "LineageRefV1",
]
