"""L0 vocabulary and mechanics for strict, reference-aware task-graph wires."""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, get_args, get_origin

from writing_agent.task_graph import (
    MessageV1,
    canonical_bytes,
    canonical_json,
    domain_hash,
    freeze,
    load_canonical_json,
    logical_id,
    utf8,
    validate_hash,
)


@dataclass(frozen=True)
class Hash:
    """A hash field; ``edge=None`` marks identity data that is not a store ref."""

    edge: str | None
    optional: bool = False


@dataclass(frozen=True)
class Str:
    nonempty: bool = False
    logical: bool = False
    optional: bool = False
    no_whitespace_or_controls: bool = False


@dataclass(frozen=True)
class Int:
    minimum: int | None = 0
    equals: int | None = None
    optional: bool = False


@dataclass(frozen=True)
class Bool:
    pass


@dataclass(frozen=True)
class TrueOnly:
    """A presence claim whose only permitted boolean value is true."""


@dataclass(frozen=True)
class JsonValue:
    """Any ordinary canonical JSON value (not the adapter-intake marker dialect)."""


@dataclass(frozen=True)
class Enum:
    values: frozenset[str]


@dataclass(frozen=True)
class ListOf:
    item: Any
    unique: bool = False
    min_items: int | None = None
    max_items: int | None = None


@dataclass(frozen=True)
class DictOf:
    value: Any


@dataclass(frozen=True)
class Obj:
    required: Mapping[str, Any]
    optional: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    extra: Any | None = None


def obj(**required: Any) -> Obj:
    """Build one closed object shape without a parallel field table."""
    return Obj(MappingProxyType(required))


def obj_opt(
    required: Mapping[str, Any], optional: Mapping[str, Any] | None = None, *, extra: Any = None
) -> Obj:
    return Obj(
        MappingProxyType(dict(required)),
        MappingProxyType(dict(optional or {})),
        extra,
    )


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
    """A task-graph ``MessageV1`` value."""


@dataclass(frozen=True)
class RecordOf:
    record: type[WireRecord]


@dataclass(frozen=True)
class PayloadCodec:
    """Closed codec for one tagged payload shape that has no Python record class."""

    record_type: str
    fields: Obj
    tagged: bool = True

    @property
    def REFS(self) -> Mapping[str, str]:
        return _derived_refs(self.fields)

    def from_dict(self, body: Mapping[str, Any]) -> dict[str, Any]:
        validate_codec(self, body)
        return dict(body)


def _canonical_json_value(value: Any) -> None:
    if value is None or type(value) in (str, int, bool):
        if type(value) is str:
            utf8(value)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _canonical_json_value(item)
        return
    if not isinstance(value, Mapping):
        raise TypeError("value is not canonical JSON")
    for key, item in value.items():
        if type(key) is not str:
            raise TypeError("canonical object keys must be strings")
        utf8(key, "canonical object key")
        _canonical_json_value(item)


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


def _decode_canonical_value(value: Any) -> Any:
    """Validate and decode exactly one canonical value produced by adapter intake."""
    if value is None or type(value) in (str, int, bool):
        if type(value) is str:
            utf8(value)
        return value
    if isinstance(value, list):
        return [_decode_canonical_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_canonical_value(item) for item in value)
    if not isinstance(value, Mapping):
        raise TypeError("value is not a canonical intake value")
    if any(type(key) is not str for key in value):
        raise TypeError("canonical object keys must be strings")
    if "$noncanonical" not in value:
        return {key: _decode_canonical_value(item) for key, item in value.items()}

    marker = value["$noncanonical"]
    if type(marker) is not str or marker not in _INTAKE_MARKER_FIELDS:
        raise ValueError("unknown or forged noncanonical marker")
    if set(value) != _INTAKE_MARKER_FIELDS[marker]:
        raise ValueError("noncanonical marker has the wrong key set")
    if marker == "float":
        representation = value["repr"]
        if type(representation) is not str:
            raise TypeError("float marker repr must be a string")
        utf8(representation, "float marker repr")
        try:
            decoded = float(representation)
        except (ValueError, OverflowError) as exc:
            raise ValueError("float marker representation is invalid") from exc
        if repr(decoded) != representation:
            raise ValueError("float marker representation is not canonical")
        return decoded
    if marker == "bytes":
        hexadecimal = value["hex"]
        if type(hexadecimal) is not str:
            raise TypeError("bytes marker hex must be a string")
        utf8(hexadecimal, "bytes marker hex")
        try:
            decoded = bytes.fromhex(hexadecimal)
        except ValueError as exc:
            raise ValueError("bytes marker contains invalid hexadecimal") from exc
        if decoded.hex() != hexadecimal:
            raise ValueError("bytes marker hexadecimal is not canonical")
        return decoded
    if marker in {"tuple", "mapping"}:
        items = value["items"]
        if not isinstance(items, (list, tuple)):
            raise TypeError("marker items must be an array")
        if marker == "tuple":
            return tuple(_decode_canonical_value(item) for item in items)
        decoded_mapping = {}
        for pair in items:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("mapping item must be a key/value pair")
            key, item = (_decode_canonical_value(part) for part in pair)
            try:
                if key in decoded_mapping:
                    raise ValueError("mapping marker has duplicate decoded keys")
                decoded_mapping[key] = item
            except TypeError as exc:
                raise ValueError("mapping marker keys must be hashable") from exc
        return decoded_mapping
    if marker == "surrogate-string":
        points = value["codepoints"]
        if not isinstance(points, (tuple, list)):
            raise TypeError("surrogate codepoints must be an array")
        decoded_points = []
        for point in points:
            if type(point) is not int or point < 0 or point > 0x10FFFF:
                raise ValueError("surrogate codepoint is outside Unicode")
            decoded_points.append(chr(point))
        return "".join(decoded_points)
    if marker == "unsupported":
        type_name = value["type"]
        if type(type_name) is not str or not type_name:
            raise ValueError("unsupported type must be nonempty text")
        utf8(type_name, "unsupported type")
        return object()
    return object()


def validate_canonical_value(value: Any) -> None:
    """Validate a canonical intake value without preserving adapter-only values."""
    _decode_canonical_value(value)


def decode_canonical_value(value: Any) -> Any:
    """Decode a value already carried by a validated :class:`SampledMessageV1`."""
    return _decode_canonical_value(value)


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
    elif value is None or type(value) in (str, int, bool):
        return
    else:
        raise TypeError("wire values must use canonical JSON types")


def _json_value(value: Any) -> Any:
    if isinstance(value, WireRecord):
        return value.to_wire()
    if isinstance(value, MessageV1):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


_NO_VALUE = object()


def _child_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _walk_spec(
    spec: Any,
    value: Any,
    label: str,
    path: str = "",
    edges: list[tuple[str, str, Any]] | None = None,
) -> Any:
    """Normalize one schema node and collect its declared edges in the same walk."""
    if edges is None:
        edges = []
    if value is _NO_VALUE and (
        isinstance(
            spec,
            (Str, Int, Bool, TrueOnly, Enum, JsonValue, CanonicalIntake, MessageValue),
        )
        or spec is type(None)
    ):
        return value
    if isinstance(spec, Hash):
        if value is not _NO_VALUE:
            validate_hash(value, optional=spec.optional)
            if value is not None and spec.edge is not None:
                edges.append((path, spec.edge, value))
        elif spec.edge is not None:
            edges.append((path, spec.edge, _NO_VALUE))
        return value
    elif isinstance(spec, (Str, Int, Bool, Enum)):
        if value is None and getattr(spec, "optional", False):
            return None
        if isinstance(spec, Str):
            if type(value) is not str:
                raise TypeError(f"{label} must be a string")
            utf8(value, label)
            if spec.nonempty and not value:
                raise ValueError(f"{label} must be nonempty")
            if spec.no_whitespace_or_controls and any(
                char.isspace() or ord(char) < 0x20 for char in value
            ):
                raise ValueError(f"{label} cannot contain whitespace or control characters")
            if spec.logical:
                logical_id(value, label)
        elif isinstance(spec, Int):
            if type(value) is not int:
                raise TypeError(f"{label} must be an integer")
            if spec.minimum is not None and value < spec.minimum:
                raise ValueError(f"{label} is below its minimum")
            if spec.equals is not None and value != spec.equals:
                raise ValueError(f"{label} must equal {spec.equals}")
        elif isinstance(spec, Bool):
            if type(value) is not bool:
                raise TypeError(f"{label} must be a boolean")
        elif type(value) is not str or value not in spec.values:
            raise ValueError(f"{label} has an unsupported value")
        return value
    elif isinstance(spec, TrueOnly):
        if value is not True:
            raise ValueError(f"{label} may only be true when present")
        return value
    elif isinstance(spec, ListOf):
        if value is _NO_VALUE:
            _walk_spec(spec.item, _NO_VALUE, label, f"{path}[]", edges)
            return value
        if not isinstance(value, (tuple, list)):
            raise TypeError(f"{label} must be an array")
        if spec.min_items is not None and len(value) < spec.min_items:
            raise ValueError(f"{label} has fewer than {spec.min_items} items")
        if spec.max_items is not None and len(value) > spec.max_items:
            raise ValueError(f"{label} has more than {spec.max_items} items")
        result = [
            _walk_spec(spec.item, item, f"{label}[{index}]", f"{path}[]", edges)
            for index, item in enumerate(value)
        ]
        result = tuple(result)
        if spec.unique and len(result) != len(set(result)):
            raise ValueError(f"{label} must be unique")
        return result
    elif isinstance(spec, DictOf):
        if value is _NO_VALUE:
            _walk_spec(spec.value, _NO_VALUE, label, f"{path}{{}}", edges)
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be an object")
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} keys must be strings")
            utf8(key, f"{label} key")
            result[key] = _walk_spec(spec.value, item, f"{label}.{key}", f"{path}{{}}", edges)
        return result
    elif isinstance(spec, Obj):
        if value is _NO_VALUE:
            for key, child_spec in (*spec.required.items(), *spec.optional.items()):
                _walk_spec(child_spec, _NO_VALUE, label, _child_path(path, key), edges)
            if spec.extra is not None:
                _walk_spec(spec.extra, _NO_VALUE, label, f"{path}{{}}", edges)
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be an object")
        if any(type(key) is not str for key in value):
            raise TypeError(f"{label} keys must be strings")
        keys = set(value)
        required = set(spec.required)
        optional = set(spec.optional)
        missing = required - keys
        unknown = keys - required - optional
        if missing:
            raise ValueError(f"{label} is missing fields: {sorted(missing)}")
        if unknown and spec.extra is None:
            raise ValueError(f"{label} has unknown fields: {sorted(unknown)}")
        result = dict(value)
        for key, item_spec in (*spec.required.items(), *spec.optional.items()):
            if key in value:
                result[key] = _walk_spec(
                    item_spec, value[key], f"{label}.{key}", _child_path(path, key), edges
                )
        if unknown:
            for key in unknown:
                result[key] = _walk_spec(
                    spec.extra, value[key], f"{label}.{key}", f"{path}{{}}", edges
                )
        return result
    elif isinstance(spec, JsonValue):
        _canonical_json_value(value)
        return value
    elif isinstance(spec, UnionOf):
        if value is _NO_VALUE:
            for option in spec.options:
                _walk_spec(option, _NO_VALUE, label, path, edges)
            return value
        failures = []
        for option in spec.options:
            initial_edge_count = len(edges)
            try:
                return _walk_spec(option, value, label, path, edges)
            except (TypeError, ValueError) as exc:
                del edges[initial_edge_count:]
                failures.append(exc)
        if not failures:
            raise ValueError(f"{label} has no permitted shape")
        raise ValueError(f"{label} does not match a permitted shape") from failures[-1]
    elif isinstance(spec, KindUnion):
        if value is _NO_VALUE:
            for option in spec.variants.values():
                _walk_spec(option, _NO_VALUE, label, path, edges)
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a tagged object")
        try:
            variant = spec.variants[value.get(spec.discriminator)]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{label} has an unknown {spec.discriminator}") from exc
        return _walk_spec(variant, value, label, path, edges)
    elif isinstance(spec, CanonicalIntake):
        validate_canonical_value(value)
        return value
    elif isinstance(spec, MessageValue):
        if isinstance(value, MessageV1):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a MessageV1 object")
        return MessageV1.from_dict(dict(value))
    elif isinstance(spec, RecordOf):
        if value is _NO_VALUE:
            for edge_path, edge in spec.record.REFS.items():
                edges.append((_child_path(path, edge_path), edge, _NO_VALUE))
            return value
        if isinstance(value, spec.record):
            record = value
        elif isinstance(value, Mapping):
            record = spec.record.from_dict(dict(value))
        else:
            raise TypeError(f"{label} must be a {spec.record.__name__} record")
        edges.extend(
            (_child_path(path, edge_path), edge, ref) for edge_path, edge, ref in record._wire_edges
        )
        return record
    elif spec is type(None):
        if value is not None:
            raise TypeError(f"{label} must be null")
        return None
    else:
        raise TypeError(f"unsupported field specification for {label}: {spec!r}")


def _derived_refs(spec: Obj) -> Mapping[str, str]:
    found: list[tuple[str, str, Any]] = []
    _walk_spec(spec, _NO_VALUE, "field specification", edges=found)
    return MappingProxyType({path: edge for path, edge, _ in found})


def validate_codec(
    codec: PayloadCodec, body: Mapping[str, Any]
) -> tuple[tuple[str, str, Any], ...]:
    if type(body) is not dict:
        raise TypeError("wire record must be a plain object")
    _wire_value(body)
    if codec.tagged and body.get("record_type") != codec.record_type:
        raise ValueError("record_type does not match codec")
    candidate = (
        {key: value for key, value in body.items() if key != "record_type"}
        if codec.tagged
        else body
    )
    edges: list[tuple[str, str, Any]] = []
    _walk_spec(codec.fields, candidate, codec.record_type, edges=edges)
    return tuple(edges)


_RECORD_CLASSES: dict[str, type[WireRecord]] = {}


def registered_record_classes() -> tuple[type[WireRecord], ...]:
    """Return the record classes registered by their declarations."""
    return tuple(_RECORD_CLASSES.values())


def record_class(record_name: str) -> type[WireRecord]:
    """Resolve a registered record class, including untagged domain records."""
    return _RECORD_CLASSES[record_name]


class WireRecord:
    """Base for strict, frozen wire values with optional payload record types."""

    RECORD_TYPE: ClassVar[str | None] = None
    EDGE_TYPE: ClassVar[str | None] = None
    OMIT_NONE_FIELDS: ClassVar[frozenset[str]] = frozenset()
    FIELD_SPEC: ClassVar[Obj] = Obj(MappingProxyType({}))
    REFS: ClassVar[Mapping[str, str]] = MappingProxyType({})
    OMIT_NONE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        annotations = typing.get_type_hints(cls, include_extras=True)
        specs: dict[str, Any] = {}
        for name in cls.__dict__.get("__annotations__", {}):
            annotation = annotations[name]
            if get_origin(annotation) is ClassVar:
                continue
            if get_origin(annotation) is not Annotated:
                raise TypeError(f"{cls.__name__}.{name} must declare its wire spec with Annotated")
            args = get_args(annotation)
            if len(args) != 2:
                raise TypeError(f"{cls.__name__}.{name} must have exactly one wire spec")
            specs[name] = args[1]
        omitted = cls.OMIT_NONE_FIELDS
        if omitted - specs.keys():
            raise TypeError(f"{cls.__name__}.OMIT_NONE_FIELDS names an unknown field")
        cls.FIELD_SPEC = Obj(
            MappingProxyType({name: spec for name, spec in specs.items() if name not in omitted}),
            MappingProxyType({name: spec for name, spec in specs.items() if name in omitted}),
        )
        cls.REFS = _derived_refs(cls.FIELD_SPEC)
        registry_name = cls.RECORD_TYPE or cls.EDGE_TYPE or cls.__name__
        if registry_name in _RECORD_CLASSES:
            raise TypeError(f"duplicate task-graph record registration: {registry_name}")
        _RECORD_CLASSES[registry_name] = cls

    def __post_init__(self) -> None:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        edges: list[tuple[str, str, Any]] = []
        values = _walk_spec(self.FIELD_SPEC, values, type(self).__name__, edges=edges)
        self._finish(values, tuple(edges))

    def _finish(self, values: Mapping[str, Any], edges: tuple[tuple[str, str, Any], ...]) -> None:
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.check()
        for item in fields(self):
            object.__setattr__(self, item.name, freeze(getattr(self, item.name)))
        object.__setattr__(self, "_wire_edges", tuple(edges))
        canonical_bytes(self.to_wire())

    def check(self) -> None:
        """Apply record-local cross-field invariants."""

    def to_wire(self) -> dict[str, Any]:
        body = {
            item.name: _json_value(getattr(self, item.name))
            for item in fields(self)
            if item.name not in self.OMIT_NONE_FIELDS or getattr(self, item.name) is not None
        }
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
        original = value
        if cls.RECORD_TYPE is not None:
            if value.get("record_type") != cls.RECORD_TYPE:
                raise ValueError("record_type does not match codec")
            value = {key: item for key, item in value.items() if key != "record_type"}
        fields_by_name = {item.name: item for item in fields(cls)}
        missing_optional = (set(fields_by_name) - set(value)) & cls.OMIT_NONE_FIELDS
        value = {
            **value,
            **{name: fields_by_name[name].default for name in missing_optional},
        }
        edges: list[tuple[str, str, Any]] = []
        normalized = _walk_spec(cls.FIELD_SPEC, value, cls.__name__, edges=edges)
        record = object.__new__(cls)
        record._finish(normalized, tuple(edges))
        if canonical_bytes(record.to_wire()) != canonical_bytes(original):
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
