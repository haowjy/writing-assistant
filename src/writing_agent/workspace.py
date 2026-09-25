"""Bounded text tools for a private, per-task workspace.

This is a path-constrained filesystem interface, not an OS security sandbox.
Only the harness should have write access while a task runs.
"""

from pathlib import Path


class Workspace:
    def __init__(
        self,
        root: Path,
        max_bytes: int = 128_000,
        max_total_bytes: int = 1_000_000,
        *,
        strict_decode: bool = False,
    ):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_total_bytes = max_total_bytes
        self.strict_decode = strict_decode

    def resolve(self, path: str) -> Path:
        if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Use a relative workspace path without '..'")
        candidate = self.root / path
        for component in [candidate, *candidate.parents]:
            if component == self.root:
                break
            if component.is_symlink():
                raise ValueError("Symlinks are not allowed")
        candidate = candidate.resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("Path escapes workspace")
        return candidate

    def list_dir(self, path: str = ".") -> list[str]:
        directory = self.resolve(path)
        return sorted(p.name + ("/" if p.is_dir() else "") for p in directory.iterdir())[:500]

    def read_file(self, path: str) -> str:
        target = self.resolve(path)
        with target.open("rb") as handle:
            content = handle.read(self.max_bytes + 1)
        if len(content) > self.max_bytes:
            raise ValueError("File exceeds text-tool size limit")
        return content.decode("utf-8")

    def write_file(self, path: str, content: str) -> str:
        if len(content.encode("utf-8")) > self.max_bytes:
            raise ValueError("File exceeds text-tool size limit")
        target = self.resolve(path)
        current = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        replaced = target.stat().st_size if target.is_file() else 0
        if current - replaced + len(content.encode("utf-8")) > self.max_total_bytes:
            raise ValueError("Workspace storage budget exceeded")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return "written"

    def patch_file(self, path: str, old: str, new: str) -> str:
        content = self.read_file(path)
        if not old or content.count(old) != 1:
            raise ValueError("Patch must match exactly one nonempty span")
        return self.write_file(path, content.replace(old, new, 1))

    def search(self, query: str, path: str = ".") -> list[dict]:
        if not query:
            raise ValueError("Query cannot be empty")
        target = self.resolve(path)
        candidates = [target] if target.is_file() else sorted(target.rglob("*"))
        matches = []
        for file in candidates[:1000]:
            if not file.is_file() or file.is_symlink():
                continue
            relative = file.relative_to(self.root).as_posix()
            try:
                content = self.read_file(relative)
            except UnicodeError:
                if self.strict_decode:
                    raise
                continue
            except ValueError:
                continue
            for number, line in enumerate(content.splitlines(), 1):
                if query.casefold() in line.casefold():
                    matches.append({"path": relative, "line": number, "text": line[:1000]})
                    if len(matches) >= 100:
                        return matches
        return matches

    def snapshot(self) -> dict[str, str]:
        return {
            p.relative_to(self.root).as_posix(): self.read_file(p.relative_to(self.root).as_posix())
            for p in sorted(self.root.rglob("*"))
            if p.is_file()
        }


def _schema(name: str, description: str, required: list[str], **properties: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "string", "description": value}
                    for key, value in properties.items()
                },
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema("list_dir", "List a workspace directory (up to 500 entries).", [], path="Directory"),
    _schema("read_file", "Read a UTF-8 workspace file.", ["path"], path="File"),
    _schema(
        "search",
        "Literal case-insensitive search; up to 100 matches in 1000 files.",
        ["query"],
        query="Text",
        path="File or directory",
    ),
    _schema(
        "write_file",
        "Create or replace a file. Commit canon only when authorized.",
        ["path", "content"],
        path="File",
        content="Complete text",
    ),
    _schema(
        "patch_file",
        "Replace one exact text span; fails on ambiguous matches.",
        ["path", "old", "new"],
        path="File",
        old="Unique old text",
        new="Replacement",
    ),
]


def dispatch(workspace: Workspace, name: str, arguments: dict) -> dict:
    allowed = {schema["function"]["name"]: schema["function"] for schema in TOOL_SCHEMAS}
    try:
        if name not in allowed:
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(arguments, dict) or not all(
            isinstance(value, str) for value in arguments.values()
        ):
            raise ValueError("Tool arguments must be an object of strings")
        params = allowed[name]["parameters"]
        if (
            set(arguments) - params["properties"].keys()
            or set(params["required"]) - arguments.keys()
        ):
            raise ValueError("Tool arguments do not match schema")
    except (ValueError, TypeError) as exc:
        return {"ok": False, "valid": False, "error": str(exc)}
    try:
        return {"ok": True, "valid": True, "result": getattr(workspace, name)(**arguments)}
    except (OSError, ValueError, TypeError) as exc:
        return {"ok": False, "valid": True, "error": str(exc)}
