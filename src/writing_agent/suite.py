"""Scenario compilation and serial, resumable research execution.

Only the visible package crosses into the tool loop. Labels remain evaluator data.
A failed or interrupted attempt is retained; an explicit retry creates a new attempt.
"""

import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from writing_agent.agent import run_agent
from writing_agent.backends import Backend
from writing_agent.catalog import (
    fingerprint,
    overlap_audit,
    safe_relative,
    save_json,
    validate_catalog,
)
from writing_agent.workspace import TOOL_SCHEMAS, Workspace

FAMILIES = {"F1", "F2", "F3", "F4", "F5"}
VISIBLE_FIELDS = {"brief", "initial_files", "followups", "tools", "budgets", "prose"}


def compile_scenarios(scenarios: list[dict], catalog: list[dict], destination: Path) -> dict:
    groups = validate_catalog(catalog)
    overlaps = overlap_audit(catalog)
    if any(pair["crosses_role"] for pair in overlaps):
        raise ValueError("Near-duplicate sources cross research roles; review and group them")
    sources = {r["id"]: r for r in catalog}
    ids = set()
    packages = []
    group_roles = {}
    for scenario in scenarios:
        key = scenario["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or key in ids:
            raise ValueError("Scenario IDs must be unique and path-safe")
        ids.add(key)
        if scenario["family"] not in FAMILIES or scenario["role"] not in {
            "train",
            "development",
            "final_eval",
        }:
            raise ValueError("Unknown scenario family or role")
        if not scenario["source_ids"] or set(scenario["source_ids"]) - sources.keys():
            raise ValueError("Missing source identity")
        source_groups = sorted({groups[key] for key in scenario["source_ids"]})
        for source in scenario["source_ids"]:
            if sources[source]["role"] != scenario["role"]:
                raise ValueError("Scenario/source role conflict")
        for group in source_groups:
            if group in group_roles and group_roles[group] != scenario["role"]:
                raise ValueError("Source group crosses scenario roles")
            group_roles[group] = scenario["role"]
        visible = copy.deepcopy(scenario["visible"])
        if set(visible) != VISIBLE_FIELDS:
            raise ValueError("Visible package has missing or evaluator-only fields")
        if not isinstance(visible["brief"], str) or not visible["brief"].strip():
            raise ValueError("Scenario needs a brief")
        if set(visible["tools"]) - {s["function"]["name"] for s in TOOL_SCHEMAS}:
            raise ValueError("Unknown tools")
        for path, text in visible["initial_files"].items():
            safe_relative(path)
            if not isinstance(text, str):
                raise ValueError("Initial files must contain text")
        if any(not isinstance(turn, str) for turn in visible["followups"]):
            raise ValueError("Follow-ups must be user text")
        for selector in visible["prose"]:
            if selector["kind"] not in {"reply", "file"}:
                raise ValueError("Unknown prose selector")
            if selector["kind"] == "file":
                safe_relative(selector["path"])
            if selector.get("selection", "whole") not in {"whole", "delimited", "span"}:
                raise ValueError("Unknown prose selection")
        budgets = visible["budgets"]
        for name in ("max_steps", "max_tool_calls", "max_read_tokens", "max_total_bytes"):
            if type(budgets.get(name)) is not int or budgets[name] < 0:
                raise ValueError("Budgets must be explicit nonnegative integers")
        if budgets["max_steps"] < 1 or budgets["max_total_bytes"] < 1:
            raise ValueError("Step/storage budgets must be positive")
        labels = copy.deepcopy(scenario["labels"])
        if not isinstance(labels.get("checks"), list):
            raise ValueError("Private checks must be explicit")
        metadata = {
            key: copy.deepcopy(value)
            for key, value in scenario.items()
            if key not in {"visible", "labels"}
        }
        metadata.update(
            schema_version=1,
            source_groups=source_groups,
            visible_hash=fingerprint(visible),
            labels_hash=fingerprint(labels),
        )
        packages.append((metadata, visible, labels))
    # Validate the complete release before writing any package.
    manifest = {
        "schema_version": 1,
        "catalog_hash": fingerprint(catalog),
        "scenarios": [p[0] for p in packages],
    }
    for metadata, visible, labels in packages:
        save_json(destination / "visible" / (metadata["id"] + ".json"), visible)
        save_json(destination / "private" / (metadata["id"] + ".json"), labels)
    save_json(destination / "catalog.json", catalog)
    save_json(destination / "overlap-audit.json", overlaps)
    save_json(destination / "manifest.json", manifest)
    return manifest


def load_scenarios(directory: Path, ids: list[str] | None = None) -> list[dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    all_ids = {r["id"] for r in manifest["scenarios"]}
    if ids is not None and set(ids) - all_ids:
        raise ValueError("Unknown requested scenarios")
    selected = []
    for metadata in manifest["scenarios"]:
        if ids is not None and metadata["id"] not in ids:
            continue
        key = metadata["id"]
        visible = json.loads((directory / "visible" / (key + ".json")).read_text())
        labels = json.loads((directory / "private" / (key + ".json")).read_text())
        if (
            fingerprint(visible) != metadata["visible_hash"]
            or fingerprint(labels) != metadata["labels_hash"]
        ):
            raise ValueError("Compiled scenario hash mismatch; recompile the release")
        selected.append({**metadata, "visible": visible, "labels": labels})
    return selected


def run_selected(
    scenarios: list[dict],
    model: dict,
    backend_factory: Callable[[], Backend],
    destination: Path,
    *,
    execute: bool = False,
    retry_failed: bool = False,
) -> list[dict]:
    """Each scenario gets a new backend and workspace. No execution by default.

    The caller owns model loading and explicit authorization. Include exact model,
    revision, protocol, generation settings and quantization in the model record.
    """
    if not execute:
        return [{"scenario_id": s["id"], "status": "planned", "model": model} for s in scenarios]
    if not all(model.get(key) for key in ("id", "revision", "protocol", "kind")):
        raise ValueError("Model identity/revision/protocol/kind must be explicit")
    harness = fingerprint(
        {p.name: fingerprint(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    )
    results = []
    for scenario in scenarios:
        visible = scenario["visible"]
        observation = {
            "scenario_id": scenario["id"],
            "condition": scenario["condition"],
            "builder_identity": scenario.get("builder_identity"),
            "source_groups": scenario["source_groups"],
            "input_provenance": scenario["provenance"],
            "style": scenario.get("style", "unspecified"),
            "instruction_specificity": scenario.get("instruction_specificity", "unspecified"),
            "genre": scenario.get("genre", "unspecified"),
        }
        identity = fingerprint(
            {"visible": visible, "observation": observation, "model": model, "harness": harness}
        )
        task_dir = destination / identity
        task_dir.mkdir(parents=True, exist_ok=True)
        attempts = sorted(task_dir.glob("attempt-*"))
        if attempts:
            latest = _read_attempt(attempts[-1])
            if latest["status"] == "completed" or not retry_failed:
                results.append(latest)
                continue
        attempt = task_dir / f"attempt-{len(attempts) + 1:04d}-{uuid4().hex[:8]}"
        attempt.mkdir()
        save_json(
            attempt / "started.json",
            {
                "identity": identity,
                "model": model,
                **observation,
                "family": scenario["family"],
                "provenance": "synthetic",
                "visible_hash": fingerprint(visible),
                "before": visible["initial_files"],
                "path": str(attempt.resolve()),
                "is_model_evaluation": model["kind"] != "scripted",
            },
        )
        save_json(attempt / "visible.json", visible)
        workspace = Workspace(
            attempt / "workspace", max_total_bytes=visible["budgets"]["max_total_bytes"]
        )
        trace_events = []
        with (attempt / "trace.jsonl").open("w") as trace:

            def emit(event, trace_events=trace_events):
                trace.write(json.dumps(event, ensure_ascii=False) + "\n")
                trace.flush()
                trace_events.append(event)

            try:
                for path, text in visible["initial_files"].items():
                    workspace.write_file(path, text)
                before = workspace.snapshot()
                budgets = {k: v for k, v in visible["budgets"].items() if k != "max_total_bytes"}
                result = run_agent(
                    backend_factory(),
                    workspace,
                    [{"role": "user", "content": visible["brief"]}],
                    tools=visible["tools"],
                    followups=visible["followups"],
                    emit=emit,
                    **budgets,
                )
                after = workspace.snapshot()
            except Exception as exc:
                before = visible["initial_files"]
                after = workspace.snapshot()
                result = {
                    "status": "setup_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "output": "",
                    "turns": [],
                    "tool_calls": 0,
                    "tool_errors": 0,
                    "usage": {},
                    "latency_seconds": 0,
                }
        record = {
            **result,
            "schema_version": 1,
            "identity": identity,
            "model": model,
            **observation,
            "family": scenario["family"],
            "provenance": "synthetic",
            "harness_hash": harness,
            "visible_hash": fingerprint(visible),
            "before": before,
            "after": after,
            "trace": trace_events,
            "path": str(attempt.resolve()),
            "is_model_evaluation": model["kind"] != "scripted",
        }
        save_json(attempt / "result.json", record)
        results.append(record)
    return results


def kb_use_scenario(scenario: dict, builder_result: dict, *, paths: list[str]) -> dict:
    """Copy only named KB artifacts to a fresh F5 scenario, with parent identity."""
    derived = copy.deepcopy(scenario)
    if derived["family"] != "F5":
        raise ValueError("KB use requires an F5 scenario")
    derived["visible"]["initial_files"] = {
        safe_relative(path): builder_result["after"][path] for path in paths
    }
    derived["condition"] = "generated_kb"
    derived["builder_identity"] = builder_result["identity"]
    derived["source_groups"] = sorted(
        set(scenario["source_groups"]) | set(builder_result["source_groups"])
    )
    derived["visible_hash"] = fingerprint(derived["visible"])
    return derived


def render_kb(sections: dict[str, str], *, linked: bool) -> dict[str, str]:
    """Same headings and substantive content, either linked pages or one flat page."""
    if not sections or any(not re.fullmatch(r"[a-z0-9_-]+", key) for key in sections):
        raise ValueError("KB section IDs must be lowercase path-safe names")
    if linked:
        return {
            "kb/index.md": "# Index\n\n" + "\n".join(f"- [{key}]({key}.md)" for key in sections),
            **{f"kb/{key}.md": f"# {key}\n\n{text}" for key, text in sections.items()},
        }
    index = "# Index\n\n" + "\n".join(f"- [{key}](#{key})" for key in sections)
    return {
        "kb/index.md": index
        + "\n\n"
        + "\n\n".join(f"# {key}\n\n{text}" for key, text in sections.items())
    }


def run_navigation(
    builder_scenario: dict,
    builder_result: dict,
    reader: dict,
    backend_factory: Callable[[], Backend],
    destination: Path,
    *,
    execute=False,
) -> list[dict]:
    """Fresh, fixed-reader probes; their semantic answer checks remain private."""
    kb = {p: t for p, t in builder_result["after"].items() if p.startswith("kb/")}
    scenarios = []
    for probe in builder_scenario["labels"]["probes"]:
        scenarios.append(
            {
                "id": builder_scenario["id"] + "-probe-" + probe["id"],
                "family": "F4",
                "source_groups": builder_scenario["source_groups"],
                "condition": "navigation_probe",
                "provenance": "synthetic",
                "builder_identity": builder_result["identity"],
                "visible": {
                    "brief": "Read the wiki starting at kb/index.md. " + probe["question"],
                    "initial_files": kb,
                    "followups": [],
                    "prose": [],
                    "tools": ["list_dir", "read_file"],
                    "budgets": {
                        "max_steps": 8,
                        "max_tool_calls": 12,
                        "max_read_tokens": 2000,
                        "max_total_bytes": 24_000,
                    },
                },
                "labels": {
                    "checks": [
                        {
                            "id": probe["id"],
                            "metric": "Q10",
                            "kind": "semantic",
                            "method": "llm_judge",
                            "required": True,
                            "text": probe["expected"] + " Cite supporting page paths.",
                        }
                    ],
                    "rubrics": {},
                },
            }
        )
    results = run_selected(scenarios, reader, backend_factory, destination, execute=execute)
    return [{"scenario": s, "result": r} for s, r in zip(scenarios, results, strict=True)]


def _read_attempt(attempt: Path) -> dict:
    path = attempt / "result.json"
    if path.exists():
        return json.loads(path.read_text())
    started = json.loads((attempt / "started.json").read_text())
    events = []
    trace = attempt / "trace.jsonl"
    if trace.exists():
        for line in trace.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break  # Retain a possibly truncated final line in the raw trace.
    return {
        **started,
        "status": "interrupted",
        "output": "",
        "after": {},
        "turns": [],
        "usage": {},
        "latency_seconds": None,
        "trace": events,
        "error": "Attempt has no final result; partial workspace remains on disk",
    }


def saved_results(destination: Path) -> list[dict]:
    """Latest attempt per identity, including interruptions and prior failure metadata."""
    records = []
    for task in sorted(destination.glob("*")):
        attempts = sorted(task.glob("attempt-*/started.json"))
        if not attempts:
            continue
        history = [_read_attempt(p.parent) for p in attempts]
        latest = history[-1]
        latest["attempt_history"] = [{"path": r["path"], "status": r["status"]} for r in history]
        records.append(latest)
    return records
