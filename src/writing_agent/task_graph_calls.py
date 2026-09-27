"""Canonical writer-message intake, deterministic call parsing, and tool effects.

The public functions in this module form a pure seam: raw adapter values are
canonicalized once, recorded values parse to the same queue on replay, and file
effects are checked against the call and pinned limits.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from writing_agent.task_graph import canonical_json, safe_path, validate_file_tree
from writing_agent.task_graph_accounting import READ_TOOLS
from writing_agent.task_graph_errors import AdapterContractError, WriterRuntimeError

_MAX_ARGUMENT_BYTES = 65_536
_MAX_CALL_EVIDENCE_BYTES = 131_072
_MAX_INTAKE_NODES = 4_096
_MAX_INTAKE_DEPTH = 64
_ENVELOPE_KEYS = {"id", "type", "function"}
_FUNCTION_KEYS = {"name", "arguments"}


def _valid_utf8(value: str) -> bool:
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return False
    return True


def _tag(name: str, **fields: Any) -> dict[str, Any]:
    return {"$noncanonical": name, **fields}


def _encode_value(
    value: Any,
    active: set[int] | None = None,
    depth: int = 0,
    budget: list[int] | None = None,
) -> Any:
    """Encode adapter values as canonical JSON, retaining non-JSON type tags."""
    active = set() if active is None else active
    budget = [_MAX_INTAKE_NODES] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or depth > _MAX_INTAKE_DEPTH:
        return _tag("limit")
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        if _valid_utf8(value):
            return value
        return _tag("surrogate-string", codepoints=[ord(character) for character in value])
    if type(value) is float:
        return _tag("float", repr=repr(value))
    if isinstance(value, bytes):
        return _tag("bytes", hex=value.hex())
    if isinstance(value, tuple):
        if id(value) in active:
            return _tag("cycle")
        active.add(id(value))
        items = [_encode_value(item, active, depth + 1, budget) for item in value]
        active.remove(id(value))
        return _tag("tuple", items=items)
    if isinstance(value, list):
        if id(value) in active:
            return _tag("cycle")
        active.add(id(value))
        items = [_encode_value(item, active, depth + 1, budget) for item in value]
        active.remove(id(value))
        return items
    if isinstance(value, dict):
        if id(value) in active:
            return _tag("cycle")
        active.add(id(value))
        keys_are_canonical = all(
            isinstance(key, str) and _valid_utf8(key) and key != "$noncanonical" for key in value
        )
        if keys_are_canonical:
            result = {
                key: _encode_value(item, active, depth + 1, budget) for key, item in value.items()
            }
        else:
            result = _tag(
                "mapping",
                items=[
                    [
                        _encode_value(key, active, depth + 1, budget),
                        _encode_value(item, active, depth + 1, budget),
                    ]
                    for key, item in value.items()
                ],
            )
        active.remove(id(value))
        return result
    return _tag("unsupported", type=type(value).__name__)


def _decode_value(value: Any) -> Any:
    """Undo intake's typed markers for parser decisions, never for persistence."""
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, dict):
        marker = value.get("$noncanonical")
        if marker == "surrogate-string" and set(value) == {"$noncanonical", "codepoints"}:
            return "".join(chr(point) for point in value["codepoints"])
        if marker == "float" and set(value) == {"$noncanonical", "repr"}:
            try:
                return float(value["repr"])
            except (TypeError, ValueError, OverflowError):
                return 0.0
        if marker == "bytes" and set(value) == {"$noncanonical", "hex"}:
            try:
                return bytes.fromhex(value["hex"])
            except (TypeError, ValueError):
                return b""
        if marker == "tuple" and set(value) == {"$noncanonical", "items"}:
            return tuple(_decode_value(item) for item in value["items"])
        if marker == "mapping" and set(value) == {"$noncanonical", "items"}:
            return {_decode_value(pair[0]): _decode_value(pair[1]) for pair in value["items"]}
        if marker in {"cycle", "limit", "unsupported"}:
            return object()
        return {key: _decode_value(item) for key, item in value.items()}
    return value


def _bounded_call(value: Any) -> bool:
    """Bound one raw call before encoding, matching the legacy evidence boundary."""
    stack = [(value, 0)]
    remaining_bytes = _MAX_CALL_EVIDENCE_BYTES
    remaining_nodes = _MAX_INTAKE_NODES
    seen: set[int] = set()
    while stack:
        item, depth = stack.pop()
        remaining_nodes -= 1
        if remaining_nodes < 0 or depth > _MAX_INTAKE_DEPTH:
            return False
        if isinstance(item, (dict, list, tuple)):
            if len(item) > remaining_nodes:
                return False
            identity = id(item)
            # Match the established evidence bound: repeated containers are opaque too.
            if identity in seen:
                return False
            seen.add(identity)
            children = item.items() if isinstance(item, dict) else item
            if isinstance(item, dict):
                for key, child in children:
                    stack.append((key, depth + 1))
                    stack.append((child, depth + 1))
            else:
                stack.extend((child, depth + 1) for child in children)
        elif isinstance(item, str):
            remaining_bytes -= len(item.encode("utf-8", "backslashreplace"))
        elif isinstance(item, bytes):
            remaining_bytes -= len(item)
        elif item is None or type(item) in {bool, int, float}:
            remaining_bytes -= 64
        else:
            return False
        if remaining_bytes < 0:
            return False
    return True


def _is_intake_record(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("$sampled_message_v1") is True
        and set(value) == {"$sampled_message_v1", "content", "tool_calls_was_list", "calls"}
    )


def intake_message(message: Any) -> dict[str, Any]:
    """Return the canonical, replayable SampledMessageV1 form of an adapter message."""
    if _is_intake_record(message):
        return message
    if not isinstance(message, dict):
        message = {}
    raw_calls = message.get("tool_calls", [])
    was_list = isinstance(raw_calls, list)
    if was_list:
        calls = [
            {"bounded": True, "value": _encode_value(call)}
            if _bounded_call(call)
            else {"bounded": False, "value": _tag("bounded-call")}
            for call in raw_calls
        ]
    else:
        calls = _encode_value(raw_calls)
    return {
        "$sampled_message_v1": True,
        "content": _encode_value(message.get("content")),
        "tool_calls_was_list": was_list,
        "calls": calls,
    }


@dataclass(frozen=True)
class ToolQueueEntry:
    """A parsed queue item in the transition-seam wire shape."""

    call_id: str
    name: str
    arguments: Any
    rejection: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not isinstance(self.name, str):
            raise TypeError("tool queue identifiers and names must be text")
        if self.rejection is not None and not isinstance(self.rejection, str):
            raise TypeError("tool queue rejection must be text or null")
        if self.rejection is not None and (self.name != "invalid_call" or self.arguments != {}):
            raise ValueError("rejected calls must use invalid_call and empty arguments")
        if self.rejection is None and self.name == "invalid_call":
            raise ValueError("invalid_call requires a rejection")

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "rejection": self.rejection,
        }


@dataclass(frozen=True)
class _RawCall:
    raw: Any
    bounded: bool
    raw_id: Any
    function: Any
    name: Any
    arguments: Any
    arguments_undecodable: bool
    duplicate_id: bool

    @classmethod
    def read(cls, raw: Any, seen: set[str]) -> _RawCall:
        envelope = raw if isinstance(raw, dict) else {}
        function = envelope.get("function")
        fields = function if isinstance(function, dict) else {}
        raw_id, name, arguments = envelope.get("id"), fields.get("name"), fields.get("arguments")
        undecodable = False
        if isinstance(arguments, str):
            try:
                if len(arguments.encode("utf-8", "surrogatepass")) > _MAX_ARGUMENT_BYTES:
                    raise ValueError("tool arguments JSON exceeds size limit")
                arguments = json.loads(
                    arguments,
                    object_pairs_hook=_unique_pairs,
                    parse_constant=lambda _: (_ for _ in ()).throw(
                        ValueError("invalid JSON constant")
                    ),
                )
            except (ValueError, TypeError, UnicodeError, RecursionError):
                undecodable = True
        valid_id = _clean_text(raw_id) and raw_id.isprintable()
        duplicate = bool(valid_id and raw_id in seen)
        if valid_id and not duplicate:
            seen.add(raw_id)
        return cls(raw, True, raw_id, function, name, arguments, undecodable, duplicate)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _clean_text(value: Any, *, allow_control: bool = True) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and _valid_utf8(value)
        and not any(ch.isspace() or (not allow_control and ord(ch) < 0x20) for ch in value)
    )


def _oversized(call: _RawCall) -> bool:
    raw_args = call.function.get("arguments") if isinstance(call.function, dict) else None
    return (
        (isinstance(call.raw_id, str) and len(call.raw_id) > 256)
        or (isinstance(call.name, str) and len(call.name) > 256)
        or (
            isinstance(raw_args, dict)
            and (
                len(raw_args) > 64
                or sum(
                    len(value.encode("utf-8", "surrogatepass"))
                    for value in raw_args.values()
                    if isinstance(value, str)
                )
                > _MAX_ARGUMENT_BYTES
            )
        )
    )


def _ask_shape(call: _RawCall) -> str | None:
    """Keep ask_author's deterministic syntax checks in this parser boundary."""
    arguments = call.arguments
    if not isinstance(arguments, Mapping) or set(arguments) != {
        "question",
        "decision_ids",
        "proposals",
        "option_refs",
    }:
        return "ask_author needs exact structured arguments"
    question = arguments["question"]
    if not isinstance(question, str) or not question.strip():
        return "ask_author question must be nonempty text"
    ids = arguments["decision_ids"]
    if (
        not isinstance(ids, (list, tuple))
        or not ids
        or any(not isinstance(item, str) or not item for item in ids)
    ):
        return "ask_author decision_ids must be nonempty text IDs"
    if len(ids) != len(set(ids)):
        return "ask_author repeats a decision ID"
    proposals = arguments["proposals"]
    if not isinstance(proposals, (list, tuple)) or any(
        not isinstance(item, Mapping)
        or set(item) != {"id", "text"}
        or any(not isinstance(value, str) or not value for value in item.values())
        for item in proposals
    ):
        return "ask_author proposals need exact id/text pairs"
    proposal_ids = [item["id"] for item in proposals]
    if len(proposal_ids) != len(set(proposal_ids)):
        return "ask_author repeats a proposal ID"
    refs = arguments["option_refs"]
    if not isinstance(refs, (list, tuple)) or any(
        not isinstance(ref, str) or not ref for ref in refs
    ):
        return "ask_author option_refs must be text IDs"
    if len(refs) != len(set(refs)):
        return "ask_author repeats an option reference"
    try:
        canonical_json(arguments)
    except (TypeError, ValueError) as exc:
        return str(exc)
    return None


def _text_arguments(call: _RawCall) -> bool:
    return all(
        isinstance(key, str) and _valid_utf8(key) and isinstance(value, str) and _valid_utf8(value)
        for key, value in call.arguments.items()
    )


def _unsafe_path(call: _RawCall) -> bool:
    if "path" not in call.arguments:
        return False
    path = call.arguments["path"]
    listing = call.name in {"list_dir", "search"}
    if listing and path == ".":
        return False
    try:
        safe_path(path[:-1] if listing and path.endswith("/") else path)
    except (TypeError, ValueError):
        return True
    return False


Rule = Callable[[_RawCall], str | None]


def _when(predicate: Callable[[_RawCall], bool], reason: str) -> Rule:
    return lambda call: reason if predicate(call) else None


CALL_RULES: tuple[Rule, ...] = (
    _when(lambda call: not call.bounded, "Tool call envelope exceeds size or nesting limit"),
    _when(
        lambda call: (
            not isinstance(call.raw, dict)
            or set(call.raw) != _ENVELOPE_KEYS
            or call.raw.get("type") != "function"
        ),
        "Invalid tool call envelope",
    ),
    _when(_oversized, "Tool call envelope exceeds size limit"),
    _when(
        lambda call: not isinstance(call.function, dict) or set(call.function) != _FUNCTION_KEYS,
        "Invalid tool function envelope",
    ),
    _when(
        lambda call: not (_clean_text(call.raw_id) and call.raw_id.isprintable()),
        "Tool call needs an id",
    ),
    _when(lambda call: call.duplicate_id, "Duplicate tool call id"),
    _when(lambda call: not _clean_text(call.name, allow_control=False), "Invalid tool function"),
    _when(lambda call: call.arguments_undecodable, "Invalid tool arguments JSON"),
    _when(lambda call: not isinstance(call.arguments, dict), "Tool arguments must be an object"),
    lambda call: _ask_shape(call) if call.name == "ask_author" else None,
    _when(
        lambda call: call.name != "ask_author" and not _text_arguments(call),
        "Tool arguments must be an object of UTF-8 strings",
    ),
    _when(_unsafe_path, "Unsafe or noncanonical workspace path"),
)


def _names_ask_author(raw: Any) -> bool:
    return (
        isinstance(raw, dict)
        and isinstance(raw.get("function"), dict)
        and raw["function"].get("name") == "ask_author"
    )


def parse_calls(
    message: Mapping[str, Any],
    *,
    id_prefix: str,
    allowed: frozenset[str],
    prior_raw_ids: Iterable[str] = (),
    ask_semantics: Callable[[dict[str, Any]], None] | None = None,
) -> list[ToolQueueEntry]:
    """Parse calls only from canonical intake; first matching rule owns rejection."""
    if not _is_intake_record(message):
        raise ValueError("message is not a canonical sampled-message intake")
    if not isinstance(message["tool_calls_was_list"], bool) or not message["tool_calls_was_list"]:
        raise WriterRuntimeError("tool_calls must be an array")
    if not isinstance(message["calls"], list):
        raise ValueError("canonical sampled-message calls must be a list")

    decoded_calls: list[tuple[bool, Any]] = []
    for item in message["calls"]:
        if not isinstance(item, dict) or set(item) != {"bounded", "value"}:
            raise ValueError("invalid canonical sampled-message call")
        if type(item["bounded"]) is not bool:
            raise ValueError("canonical sampled-message bounded flag must be boolean")
        decoded_calls.append((item["bounded"], _decode_value(item["value"])))

    seen = set(prior_raw_ids)
    mixed = len(decoded_calls) > 1 and any(
        bounded and _names_ask_author(raw) for bounded, raw in decoded_calls
    )
    entries = []
    for index, (bounded, raw) in enumerate(decoded_calls):
        if bounded:
            call = _RawCall.read(raw, seen)
        else:
            call = _RawCall(None, False, None, None, None, None, False, False)
        reason = next((rule_reason for rule in CALL_RULES if (rule_reason := rule(call))), None)
        if reason is None and call.name not in allowed:
            reason = "Tool is not available in this condition"
        if mixed:
            reason = "Mixed control and file-tool batch is forbidden"
        if reason is None and call.name == "ask_author" and ask_semantics is not None:
            try:
                ask_semantics(call.arguments)
            except (TypeError, ValueError) as exc:
                reason = str(exc)
        entries.append(
            ToolQueueEntry(
                f"{id_prefix}:{index}",
                call.name if reason is None else "invalid_call",
                call.arguments if reason is None else {},
                reason,
            )
        )
    return entries


def apply_effect(before_files: Mapping[str, str], effect: Mapping[str, Any]) -> dict[str, str]:
    """Apply a file delta after requiring every claimed prior value to match."""
    try:
        files = validate_file_tree(before_files)
        if not isinstance(effect, Mapping):
            raise ValueError("file effect must be a mapping")
        for path, change in effect.items():
            safe_path(path)
            if not isinstance(change, Mapping) or set(change) != {"before", "after"}:
                raise ValueError("invalid file effect entry")
            prior, after = change["before"], change["after"]
            if (prior is not None and not isinstance(prior, str)) or (
                after is not None and not isinstance(after, str)
            ):
                raise ValueError("file effect content must be text or null")
            if (path in files) != (prior is not None) or files.get(path) != prior:
                raise ValueError("file effect precondition does not match current files")
            if prior == after:
                raise ValueError("file effect must contain only changed paths")
            if after is None:
                files.pop(path)
            else:
                files[path] = after
        return validate_file_tree(files)
    except (TypeError, ValueError) as exc:
        raise AdapterContractError("tool file effect violates its precondition") from exc


def _limit(value: Any) -> bool:
    return type(value) is int and value >= 0


def tool_effect_contract(
    name: str,
    arguments: Mapping[str, Any],
    before: Mapping[str, str],
    files_after: Mapping[str, str],
    observation_ok: bool,
    *,
    max_file_bytes: int,
    max_workspace_bytes: int,
    storage_bytes_limit: int,
) -> None:
    """Raise ``AdapterContractError`` unless a snapshot exactly follows the call."""
    try:
        if type(observation_ok) is not bool:
            raise ValueError("tool observation status must be boolean")
        if not all(
            _limit(limit) for limit in (max_file_bytes, max_workspace_bytes, storage_bytes_limit)
        ):
            raise ValueError("tool limits must be nonnegative integers")
        if max_workspace_bytes > storage_bytes_limit:
            raise ValueError("tool workspace limit exceeds the pinned storage budget")
        old = validate_file_tree(before)
        new = validate_file_tree(files_after)
        if not isinstance(arguments, Mapping):
            raise ValueError("tool arguments must be a mapping")
        if name not in READ_TOOLS | {"write_file", "patch_file"}:
            raise ValueError("tool has no declared file-effect contract")
        if any(len(text.encode("utf-8")) > max_file_bytes for text in new.values()):
            raise ValueError("tool result exceeds the per-file limit")
        if sum(len(text.encode("utf-8")) for text in new.values()) > max_workspace_bytes:
            raise ValueError("tool result exceeds the workspace limit")
        if sum(len(text.encode("utf-8")) for text in new.values()) > storage_bytes_limit:
            raise ValueError("tool result exceeds the storage budget")

        if not observation_ok:
            expected = old
        elif name in READ_TOOLS:
            expected = old
        elif name == "write_file":
            path, content = arguments.get("path"), arguments.get("content")
            safe_path(path)
            if not isinstance(content, str):
                raise ValueError("write_file content must be text")
            expected = {**old, path: content}
        elif name == "patch_file":
            path, old_text, new_text = (
                arguments.get("path"),
                arguments.get("old"),
                arguments.get("new"),
            )
            safe_path(path)
            if not isinstance(old_text, str) or not old_text or not isinstance(new_text, str):
                raise ValueError("patch_file requires nonempty old and text new values")
            if path not in old or old[path].count(old_text) != 1:
                raise ValueError("patch_file old text must occur exactly once")
            expected = {**old, path: old[path].replace(old_text, new_text, 1)}
        else:
            raise ValueError("tool has no declared file-effect contract")

        # A successful writer operation may not create an invalid file tree.
        expected = validate_file_tree(expected)
        if new != expected:
            raise ValueError("tool snapshot does not match the requested file effect")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AdapterContractError("tool result violates its declared effect contract") from exc


__all__ = [
    "CALL_RULES",
    "ToolQueueEntry",
    "apply_effect",
    "intake_message",
    "parse_calls",
    "tool_effect_contract",
]
