"""Workspace tools as a static table, and `dispatch_tool`, the one way a model's tool call
reaches an episode.

A table maps a name to `(schema, fn)`: `schema` is `{"name", "description", "parameters"}`
(`parameters` a JSON Schema object), `fn(ep, actor, args)` returns a result dict (or an
awaitable of one) and reports an expected failure as `{"ok": False, "error": ...}`; an
exception a tool raises anyway is logged and becomes `{"ok": False, "error": "tool error:
..."}`, so nothing raises into the harness. Every constraint `validate` checks is stated in
the schema, and every constraint a tool checks itself is stated in its description
(kb/training-lessons.md).

A path the file ACL hides from an actor looks like one that is not there: reads say "no
such file", a write with a revision says the file is at rev 0, and creating it is refused
with the message a path blocked by a file or directory gets.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import regex

from writing_assistant.episode.model import Episode, norm_path

log = logging.getLogger(__name__)

ToolFn = Callable[[Episode, str, dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]
ToolTable = dict[str, tuple[dict[str, Any], ToolFn]]

GREP_LIMIT = 50
GREP_SECONDS = 2.0  # one episode_grep call's whole budget: a backtracking pattern cannot stall the process

PATH = {
    "type": "string",
    "minLength": 1,
    "description": "Workspace path, relative and '/'-separated (e.g. 'chapters/01.md'); no '..' or leading '/'.",
}
REV = {
    "type": "integer",
    "minimum": 0,
    "description": "The file's current revision from episode_read or episode_list; 0 to create a new file.",
}


def _err(error: str) -> dict[str, Any]:
    return {"ok": False, "error": error}


def _reachable(ep: Episode, actor: str, path: str) -> tuple[str | None, dict | None]:
    """(normalised path, None) or (None, error). A file the ACL hides reads as missing."""
    try:
        path = norm_path(path)
    except ValueError as e:
        return None, _err(str(e))
    if path not in ep.files or not ep.readable(actor, path):
        return None, _err(f"no such file: {path}")
    return path, None


def _read(ep: Episode, actor: str, args: dict) -> dict:
    path, error = _reachable(ep, actor, args["path"])
    if error:
        return error
    return {"ok": True, "path": path, "text": ep.files[path], "rev": ep.file_revs[path]}


def _write(ep: Episode, actor: str, args: dict) -> dict:
    try:
        path = norm_path(args["path"])
    except ValueError as e:
        return _err(str(e))
    hidden = not ep.readable(actor, path)
    current = 0 if hidden else ep.file_revs.get(path, 0)
    if args["rev"] != current:
        return _err(f"stale revision for {path}: it is at rev {current}, you sent {args['rev']}; read it again")
    parents = {path[:i] for i, c in enumerate(path) if c == "/"}
    if hidden or parents & ep.files.keys() or any(p.startswith(path + "/") for p in ep.files):
        return _err(f"cannot create {path}: a file or directory is in the way")
    return {"ok": True, "path": path, "rev": ep.write_file(actor, path, args["text"])}


def _edit(ep: Episode, actor: str, args: dict) -> dict:
    path, error = _reachable(ep, actor, args["path"])
    if error:
        return error
    if args["rev"] != ep.file_revs[path]:
        return _err(f"stale revision for {path}: it is at rev {ep.file_revs[path]}, you sent {args['rev']}; read it again")
    text = ep.files[path]
    n = text.count(args["old"])
    if n != 1:
        return _err(f"'old' occurs {n} times in {path}; it must occur exactly once")
    return {"ok": True, "path": path, "rev": ep.write_file(actor, path, text.replace(args["old"], args["new"], 1))}


def _grep(ep: Episode, actor: str, args: dict) -> dict:
    try:
        pattern = regex.compile(args["pattern"])
    except regex.error as e:
        return _err(f"bad pattern: {e}")
    prefix = args.get("prefix", "")
    matches = []
    deadline = time.monotonic() + GREP_SECONDS
    for path in sorted(ep.files):
        if not path.startswith(prefix) or not ep.readable(actor, path):
            continue
        for n, line in enumerate(ep.files[path].splitlines(), 1):
            try:
                found = pattern.search(line, timeout=max(deadline - time.monotonic(), 0.001))
            except TimeoutError:
                return _err(f"pattern took longer than {GREP_SECONDS:g} s; use a simpler one")
            if found:
                matches.append({"path": path, "line": n, "text": line})
                if len(matches) == GREP_LIMIT:
                    return {"ok": True, "matches": matches, "truncated": True}
    return {"ok": True, "matches": matches, "truncated": False}


def _list(ep: Episode, actor: str, args: dict) -> dict:
    prefix = args.get("prefix", "")
    files = [
        {"path": p, "rev": ep.file_revs[p], "chars": len(ep.files[p])}
        for p in sorted(ep.files)
        if p.startswith(prefix) and ep.readable(actor, p)
    ]
    return {"ok": True, "files": files}


def schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
    }


WORKSPACE_TOOLS: ToolTable = {
    "episode_read": (
        schema("episode_read", "Read one workspace file. Returns its text and current revision.", {"path": PATH}, ["path"]),
        _read,
    ),
    "episode_write": (
        schema(
            "episode_write",
            "Create or replace a whole workspace file. 'rev' must equal the file's current revision "
            "(0 for a file that does not exist yet); a stale revision is refused, and so is a path that "
            "an existing file or directory is in the way of. Returns the new revision.",
            {"path": PATH, "text": {"type": "string", "description": "The complete new file text."}, "rev": REV},
            ["path", "text", "rev"],
        ),
        _write,
    ),
    "episode_edit": (
        schema(
            "episode_edit",
            "Replace one passage in an existing file. 'old' must occur exactly once in the file and "
            "'rev' must equal its current revision. Returns the new revision.",
            {
                "path": PATH,
                "old": {"type": "string", "minLength": 1, "description": "Exact text to replace; must occur exactly once."},
                "new": {"type": "string", "description": "Replacement text."},
                "rev": REV,
            },
            ["path", "old", "new", "rev"],
        ),
        _edit,
    ),
    "episode_grep": (
        schema(
            "episode_grep",
            f"Search workspace files line by line with a Python regular expression. Returns at most "
            f"{GREP_LIMIT} matches as path, line number and line text. A search that takes longer than "
            f"{GREP_SECONDS:g} seconds is refused.",
            {
                "pattern": {"type": "string", "minLength": 1, "description": "Python regular expression."},
                "prefix": {"type": "string", "description": "Only search paths starting with this prefix."},
            },
            ["pattern"],
        ),
        _grep,
    ),
    "episode_list": (
        schema(
            "episode_list",
            "List workspace files with their revision and length in characters.",
            {"prefix": {"type": "string", "description": "Only list paths starting with this prefix."}},
            [],
        ),
        _list,
    ),
}


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}
_ANNOTATIONS = {"description", "title"}
_CHECKED = {"type", "properties", "required", "additionalProperties", "items", "enum",
            "minLength", "maxLength", "minimum", "maximum"}


def validate(schema: dict, value: Any, where: str = "arguments") -> str | None:
    """The first violation of `schema` as a message, or None. Supports exactly the keywords in
    `_CHECKED`; a schema using any other keyword is a bug in its table and raises, because the
    model would be told a rule nobody enforces."""
    unknown = set(schema) - _CHECKED - _ANNOTATIONS
    if unknown:
        raise ValueError(f"schema keywords not enforced by validate(): {sorted(unknown)}")
    if "type" in schema:
        expected = _TYPES[schema["type"]]
        if not isinstance(value, expected) or (isinstance(value, bool) and schema["type"] in ("integer", "number")):
            return f"{where} must be of type {schema['type']}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where} must be one of {schema['enum']}"
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            return f"{where} must be at least {schema['minLength']} characters"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return f"{where} must be at most {schema['maxLength']} characters"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{where} must be >= {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{where} must be <= {schema['maximum']}"
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                return f"{where} is missing required '{key}'"
        for key, v in value.items():
            if key in props:
                problem = validate(props[key], v, f"'{key}'")
                if problem:
                    return problem
            elif schema.get("additionalProperties", True) is False:
                return f"{where} has unknown key '{key}'; allowed: {sorted(props)}"
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            problem = validate(schema["items"], v, f"{where}[{i}]")
            if problem:
                return problem
    return None


async def dispatch_tool(ep: Episode, actor: str, name: str, args: Any, tools: ToolTable) -> dict[str, Any]:
    """Run one tool call for `actor`. Refusals (a name outside `ep.actor_tools[actor]`, an
    unknown tool, arguments that fail the schema, a bad path, a stale revision) and any
    exception the tool raises come back as `{"ok": False, "error": ...}`; a `tool` event is
    logged either way, visible to the actor.
    `tools` is the suite's table; workspace tools cannot be shadowed by it."""
    entry = WORKSPACE_TOOLS.get(name) or tools.get(name)
    if name not in ep.actor_tools.get(actor, ()):
        result = _err(f"tool '{name}' is not available to you")
    elif entry is None:
        result = _err(f"unknown tool '{name}'")
    else:
        spec, fn = entry
        problem = validate(spec["parameters"], args)
        if problem:
            result = _err(f"invalid arguments to {name}: {problem}")
        else:
            try:
                result = fn(ep, actor, args)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as e:  # a tool bug must not lose the task; the actor sees an error
                log.exception("%s: tool %s raised for %s", ep.ep_id, name, actor)
                result = _err(f"tool error: {type(e).__name__}: {e}")
    ep.log(actor, "tool", {"name": name, "args": args, "result": result}, {actor: "full"})
    return result
