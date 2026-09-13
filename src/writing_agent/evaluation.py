"""Small deterministic smoke scorers, deliberately separate from literary judging."""

import hashlib
import json
import platform
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from writing_agent import __version__
from writing_agent.agent import run_agent
from writing_agent.backends import ChatServerBackend, ScriptedBackend
from writing_agent.workspace import Workspace


def score(result: dict, before: dict, after: dict, expected: dict) -> dict:
    output = result["output"]
    checks = {"completed": result["status"] == "completed"}
    for text in expected.get("contains", []):
        checks[f"contains:{text}"] = text.casefold() in output.casefold()
    for text in expected.get("excludes", []):
        checks[f"excludes:{text}"] = text.casefold() not in output.casefold()
    if "max_words" in expected:
        checks["length"] = len(output.split()) <= expected["max_words"]
    if "max_tool_calls" in expected:
        checks["tool_budget"] = result["tool_calls"] <= expected["max_tool_calls"]
    for path, content in expected.get("files", {}).items():
        checks[f"file:{path}"] = after.get(path) == content
    allowed = set(expected.get("allowed_changes", []))
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    checks["edit_scope"] = changed <= allowed
    checks["valid_tools"] = result["tool_errors"] == 0
    return {
        "agent": {"checks": checks, "passed": all(checks.values())},
        "artifact": {"word_count": len(output.split()), "literary_quality": None},
    }


def evaluate(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    task_path = (base / config["tasks"]).resolve()
    tasks = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
    ids = [task["id"] for task in tasks]
    if (
        not tasks
        or len(ids) != len(set(ids))
        or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", task_id) for task_id in ids)
    ):
        raise ValueError("Tasks need unique, path-safe IDs and cannot be empty")
    backend_config = config["backend"]
    kind = backend_config["kind"]
    if kind not in ("scripted", "chat_server"):
        raise ValueError(f"Unknown backend: {kind}")
    limits = config.get("agent", {})
    if any(
        type(limits.get(key, 1)) is not int or limits.get(key, 1) < 1
        for key in ("max_steps", "max_tool_calls")
    ):
        raise ValueError("Agent budgets must be positive integers")
    scripts = {}
    if kind == "scripted":
        scripts = json.loads((base / backend_config["responses"]).read_text())
        if set(ids) - scripts.keys():
            raise ValueError("Missing scripted fixtures")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = (base / config["output_dir"] / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(source.name.encode() + source.read_bytes())
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "config": config,
        "package_version": __version__,
        "python": platform.python_version(),
        "source_sha256": source_hash.hexdigest(),
        "tasks_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        "is_model_evaluation": kind != "scripted",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_dir / "tasks.jsonl").write_bytes(task_path.read_bytes())
    if kind == "scripted":
        (run_dir / "scripted_responses.json").write_text(json.dumps(scripts, indent=2))
    results = []
    for task in tasks:
        task_dir = run_dir / task["id"]
        workspace = Workspace(task_dir / "workspace")
        for path, content in task["initial_files"].items():
            workspace.write_file(path, content)
        before = workspace.snapshot()
        backend = (
            ScriptedBackend(scripts[task["id"]])
            if kind == "scripted"
            else (ChatServerBackend(backend_config))
        )
        with (task_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:

            def emit(event: dict) -> None:
                trace.write(json.dumps(event, ensure_ascii=False) + "\n")
                trace.flush()

            result = run_agent(backend, workspace, task["messages"], emit=emit, **limits)
        scores = score(result, before, workspace.snapshot(), task["expected"])
        record = {"task_id": task["id"], **result, "scores": scores}
        (task_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
    summary = {
        "run_id": run_id,
        "is_model_evaluation": kind != "scripted",
        "tasks": len(results),
        "passed": sum(r["scores"]["agent"]["passed"] for r in results),
        "errors": sum(r["status"] == "error" for r in results),
        "literary_quality": None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
