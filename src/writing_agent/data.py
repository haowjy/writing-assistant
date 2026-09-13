"""Versioned trajectory validation and explicit export of reviewed training records."""

import json
from pathlib import Path


def validate_records(records: list[dict]) -> None:
    if not records:
        raise ValueError("Dataset is empty")
    ids = set()
    groups: dict[tuple[str, str], str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object")
        if record.get("schema_version") != 1:
            raise ValueError("Unsupported schema_version")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("Each record needs a unique nonempty id")
        ids.add(identifier)
        split = record.get("split")
        if split not in ("train", "validation", "test"):
            raise ValueError(f"{identifier}: invalid split")
        provenance = record.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError(f"{identifier}: provenance must be an object")
        for key in ("author_id", "work_id", "source", "license"):
            if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                raise ValueError(f"{identifier}: missing provenance.{key}")
        for key in ("author_id", "work_id"):
            group = (key, provenance[key])
            if group in groups and groups[group] != split:
                raise ValueError(f"{identifier}: {key} crosses dataset splits")
            groups[group] = split
        for field in ("initial_files", "expected_files"):
            if not isinstance(record.get(field), dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in record[field].items()
            ):
                raise ValueError(f"{identifier}: {field} must map paths to text")
            for path in record[field]:
                if not path or Path(path).is_absolute() or ".." in Path(path).parts:
                    raise ValueError(f"{identifier}: unsafe file path")
        if record.get("review_status") not in ("pending", "accepted", "rejected"):
            raise ValueError(f"{identifier}: invalid review_status")
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{identifier}: messages required")
        tools = record.get("tools", [])
        if not isinstance(tools, list):
            raise ValueError(f"{identifier}: tools must be a list")
        tool_names = set()
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError(f"{identifier}: expected function tool schema")
            function = tool.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise ValueError(f"{identifier}: tool schema needs a name")
            if not function["name"] or function["name"] in tool_names:
                raise ValueError(f"{identifier}: empty or duplicate tool name")
            if not isinstance(function.get("parameters"), dict):
                raise ValueError(f"{identifier}: tool schema needs parameters")
            tool_names.add(function["name"])
        pending = set()
        seen_calls = set()
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"{identifier}: message must be an object")
            role = message.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                raise ValueError(f"{identifier}: invalid message role")
            if pending and role != "tool":
                raise ValueError(f"{identifier}: missing tool response")
            if role == "tool":
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str) or call_id not in pending:
                    raise ValueError(f"{identifier}: unmatched tool response")
                pending.remove(call_id)
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list):
                raise ValueError(f"{identifier}: tool_calls must be a list")
            if calls and role != "assistant":
                raise ValueError(f"{identifier}: only assistant can call tools")
            if not calls and not isinstance(message.get("content"), str):
                raise ValueError(f"{identifier}: text content required")
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError(f"{identifier}: expected function call")
                call_id = call.get("id")
                function = call.get("function", {})
                if not isinstance(call_id, str) or not call_id or call_id in seen_calls:
                    raise ValueError(f"{identifier}: invalid or duplicate tool call id")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    raise ValueError(f"{identifier}: tool name required")
                if function["name"] not in tool_names:
                    raise ValueError(f"{identifier}: unavailable tool")
                # TRL conversational tool datasets use object-valued arguments.
                if not isinstance(function.get("arguments"), dict):
                    raise ValueError(f"{identifier}: tool arguments must be an object")
                pending.add(call_id)
                seen_calls.add(call_id)
        if pending or messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls"):
            raise ValueError(f"{identifier}: trajectory must finish with an assistant answer")


def read_records(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                records.append(record)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    validate_records(records)
    return records


def export_sft(source: Path, destination: Path) -> int:
    records = read_records(source)
    selected = [r for r in records if r["split"] == "train" and r["review_status"] == "accepted"]
    if not selected:
        raise ValueError("No accepted training records to export")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        for record in selected:
            output.write(
                json.dumps(
                    {"messages": record["messages"], "tools": record.get("tools", [])},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(selected)
