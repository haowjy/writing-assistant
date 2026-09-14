"""Saved single-turn generations for external benchmarks, independent of their graders."""

import json
import time
from pathlib import Path

from writing_agent.catalog import save_json
from writing_agent.inference import TransformersBackend, load_checkpoint


def generate_tasks(tasks: list[dict], config: dict, destination: Path, *, on_result=None):
    """Generate public prompts once. Keep failures, thinking, and exact model traces."""
    manifest = {"tasks": tasks, "model": config}
    path = destination / "generation-manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Generation selection changed; use a separate destination")
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)) or any(i in {"", ".", ".."} or Path(i).name != i for i in ids):
        raise ValueError("Task IDs must be unique path components")
    save_json(path, manifest)
    summary = {"count": len(tasks), "items": [str(destination / "items" / i) for i in ids]}
    remaining = [
        t for t in tasks if not (destination / "items" / t["id"] / "generation.json").exists()
    ]
    if not remaining:
        return summary
    with load_checkpoint(config) as (model, tokenizer, record):
        for task in remaining:
            folder = destination / "items" / task["id"]
            events = []
            started = time.monotonic()
            save_json(folder / "started.json", {"task": task, "model": record})
            try:
                response = TransformersBackend(model, tokenizer, record).complete(
                    [{"role": "user", "content": task["prompt"]}], [], emit=events.append
                )
                result = {
                    "status": "completed",
                    "message": response.message,
                    "usage": response.usage,
                }
            except Exception as exc:
                result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            result.update(task=task, model=record, trace=events, seconds=time.monotonic() - started)
            save_json(folder / "generation.json", result)
            (folder / "response.md").write_text(result.get("message", {}).get("content", ""))
            (folder / "thinking.txt").write_text(result.get("message", {}).get("thinking", ""))
            if on_result:
                on_result(result)
    return summary
