"""Blinded grading packets, bounded Codex execution, and human review materials."""

import copy
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from writing_agent.catalog import fingerprint, save_json
from writing_agent.scoring import aggregate_checks, measurement

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["assessments", "checks"],
    "properties": {
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "metric",
                    "value",
                    "evidence",
                    "rationale",
                    "uncertainty",
                    "dimensions",
                ],
                "properties": {
                    "metric": {"type": "string"},
                    "value": {"type": "number"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                    "uncertainty": {"type": "string"},
                    "dimensions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "value", "reason"],
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": ["number", "null"]},
                                "reason": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "passed", "evidence", "rationale"],
                "properties": {
                    "id": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


def grading_packet(scenario: dict, result: dict, scorecard: dict) -> dict:
    prose = [
        {k: a[k] for k in ("id", "text", "spans", "source_hash")}
        for a in scorecard["artifacts"]
        if a["status"] == "ok"
    ]
    return {
        "version": 3,
        "execution_status": result.get("status"),
        "rubric_version": scenario["labels"].get("rubric_version", 1),
        "brief": scenario["visible"]["brief"],
        "followups": scenario["visible"]["followups"],
        "source_files": scenario["visible"]["initial_files"],
        "rubrics": {
            k: v
            for k, v in scenario["labels"].get("rubrics", {}).items()
            if scorecard["scores"][k]["status"] == "pending"
        },
        "checks": [c for c in scenario["labels"]["checks"] if c["method"] == "llm_judge"],
        "knowledge_labels": scenario["labels"].get("knowledge", []),
        "turn_outputs": [
            {"turn": index, "output": turn["output"], "snapshot": turn["snapshot"]}
            for index, turn in enumerate(result.get("turns", []))
        ],
        "output": result.get("output", ""),
        "after": result.get("after", {}),
        "tool_trace": [e for e in result.get("trace", []) if e["type"] == "tool"],
        "selected_prose": prose,
        "scope": "Judge prose quality only on selected_prose. KB and plans use their own rubrics.",
    }


def validate_judgment(packet: dict, judgment: dict) -> None:
    if set(judgment) != {"assessments", "checks"}:
        raise ValueError("Invalid judgment fields")
    assessments = judgment["assessments"]
    expected = set(packet["rubrics"])
    if {a["metric"] for a in assessments} != expected or len(assessments) != len(expected):
        raise ValueError("Missing, duplicate or unexpected metric judgments")
    for assessment in assessments:
        bounds = packet["rubrics"][assessment["metric"]].get("range", [1, 5])
        value = assessment["value"]
        if type(value) not in {int, float} or not bounds[0] <= value <= bounds[1]:
            raise ValueError("Judgment outside rubric range")
        dimensions = assessment.get("dimensions", [])
        names = [d["name"] for d in dimensions]
        expected_dimensions = packet["rubrics"][assessment["metric"]].get("dimensions", [])
        if (expected_dimensions and set(names) != set(expected_dimensions)) or len(names) != len(
            set(names)
        ):
            raise ValueError("Missing or unexpected rubric dimensions")
        for dimension in dimensions:
            if dimension["value"] is not None and not 1 <= dimension["value"] <= 5:
                raise ValueError("Invalid dimension rating")
            if not dimension.get("reason"):
                raise ValueError("Dimension needs supporting explanation or N/A reason")
        if not assessment.get("evidence") or not assessment.get("rationale"):
            raise ValueError("Judgments require evidence and rationale")
    checks = judgment["checks"]
    if {c["id"] for c in checks} != {c["id"] for c in packet["checks"]} or len(checks) != len(
        packet["checks"]
    ):
        raise ValueError("Missing or duplicate check judgments")
    if any(
        type(c["passed"]) is not bool or not c.get("evidence") or not c.get("rationale")
        for c in checks
    ):
        raise ValueError("Check judgments require boolean outcome and evidence")


class CodexGrader:
    """Research grader; locked persisted call budget includes failed attempts.

    No paid-API fallback. The Codex CLI uses existing subscription authentication.
    Cache identity includes packet, model, schema and instruction versions.
    """

    def __init__(self, destination: Path, *, model="gpt-6-astra", max_calls=3, timeout=180):
        self.destination = destination
        self.model = model
        self.max_calls = max_calls
        self.timeout = timeout

    def grade(self, packet: dict) -> dict:
        instruction = Path(__file__).with_name("grader_instructions.md").read_text()
        identity = fingerprint(
            {
                "packet": packet,
                "model": self.model,
                "schema": JUDGE_SCHEMA,
                "instruction_hash": fingerprint(instruction),
                "isolation_version": 2,
            }
        )
        directory = self.destination / identity
        cached = directory / "judgment.json"
        if cached.exists():
            record = json.loads(cached.read_text())
            validate_judgment(packet, record["judgment"])
            return record
        executable = shutil.which("codex")
        if not executable:
            return {"status": "unavailable", "reason": "Codex executable not found"}
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / "packet.json", packet)
        with (self.destination / "budget.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            consumed = len(list(self.destination.glob("*/call-*.json")))
            if consumed >= self.max_calls:
                return {"status": "budget_exhausted", "identity": identity}
            save_json(directory / ("call-" + uuid4().hex + ".json"), {"model": self.model})
        started = time.perf_counter()
        (directory / "instructions.md").write_text(instruction)
        with tempfile.TemporaryDirectory(prefix="writing-grader-") as temporary:
            root = Path(temporary)
            save_json(root / "schema.json", JUDGE_SCHEMA)
            output = root / "answer.json"
            instructions_path = root / "instructions.md"
            instructions_path.write_text(instruction)
            command = [
                executable,
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "-c",
                f"model_instructions_file={json.dumps(str(instructions_path))}",
                "-c",
                'developer_instructions=""',
                "-c",
                "project_doc_max_bytes=0",
                "-c",
                "skills.include_instructions=false",
                "-c",
                "features.skip_host_skill_discovery=true",
                "-c",
                "features.memory_tool=false",
                "-c",
                "features.apps=false",
                "-c",
                "features.apply_patch_freeform=false",
                "-c",
                'personality="none"',
                "-c",
                'model_reasoning_effort="high"',
                "--disable",
                "shell_tool",
                "--disable",
                "unified_exec",
                "--disable",
                "multi_agent",
                "-c",
                'web_search="disabled"',
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                self.model,
                "--color",
                "never",
                "--json",
                "--output-schema",
                str(root / "schema.json"),
                "--output-last-message",
                str(output),
                "-",
            ]
            environment = {
                k: v
                for k, v in os.environ.items()
                if k not in {"OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"}
            }
            try:
                process = subprocess.run(
                    command,
                    input=json.dumps(packet),
                    text=True,
                    capture_output=True,
                    cwd=root,
                    env=environment,
                    timeout=self.timeout,
                    check=False,
                )
                save_json(
                    directory / "launch.json",
                    {
                        "command": command,
                        "cwd": str(root),
                        "instruction_hash": fingerprint(instruction),
                        "packet_hash": fingerprint(packet),
                    },
                )
                (directory / "events.jsonl").write_text(process.stdout)
                (directory / "stderr.txt").write_text(process.stderr)
                if process.returncode:
                    raise RuntimeError(
                        f"Codex exited with code {process.returncode}; see stderr.txt"
                    )
                raw = output.read_text()
                (directory / "raw.json").write_text(raw)
                judgment = json.loads(raw)
                validate_judgment(packet, judgment)
                record = {
                    "status": "ok",
                    "identity": identity,
                    "judge": self.model,
                    "packet_hash": fingerprint(packet),
                    "judgment": judgment,
                    "human_review": "pending",
                    "latency_seconds": time.perf_counter() - started,
                }
                save_json(cached, record)
                return record
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                subprocess.TimeoutExpired,
            ) as exc:
                record = {
                    "status": "grader_error",
                    "identity": identity,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "latency_seconds": time.perf_counter() - started,
                }
                save_json(directory / "error.json", record)
                return record


def apply_judgment(scorecard: dict, packet: dict, record: dict) -> dict:
    """A failed grader leaves scores pending, never turns into a candidate zero."""
    card = copy.deepcopy(scorecard)
    card["judgment"] = record
    if record["status"] != "ok":
        return card
    if record.get("packet_hash") != fingerprint(packet):
        raise ValueError("Judgment belongs to a different packet")
    validate_judgment(packet, record["judgment"])
    for assessment in record["judgment"]["assessments"]:
        card["scores"][assessment["metric"]] = measurement(
            assessment["value"],
            method="llm_judge",
            evidence=assessment["evidence"],
            rationale=assessment["rationale"],
            uncertainty=assessment["uncertainty"],
            calibration="unvalidated",
            dimensions=assessment.get("dimensions", []),
        )
    semantic = {c["id"]: c for c in record["judgment"]["checks"]}
    for check in card["checks"]:
        if check["id"] in semantic:
            check.update(semantic[check["id"]], status="ok")
    card["scores"].update(
        aggregate_checks(
            card["checks"], card["status"], card["artifacts"], rubric_metrics=packet["rubrics"]
        )
    )
    return card


def write_review(scenarios: list[dict], destination: Path, *, judged: list[dict] | None = None):
    destination.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "scenario_id": s["id"],
            "family": s["family"],
            "instruction_specificity": s.get("instruction_specificity", "unspecified"),
            "genre": s.get("genre", "unspecified"),
            "visible": s["visible"],
            "labels": s["labels"],
            "source_ids": s["source_ids"],
            "review_status": "pending",
            "reviewer": None,
            "label_corrections": [],
            "comments": "",
        }
        for s in scenarios
    ]
    review_path = destination / "scenario-review.json"
    previous = (
        {r["scenario_id"]: r for r in json.loads(review_path.read_text())}
        if review_path.exists()
        else {}
    )
    for record in records:
        old = previous.get(record["scenario_id"])
        if old is None:
            continue
        if old["visible"] == record["visible"] and old["labels"] == record["labels"]:
            for field in (
                "review_status",
                "reviewer",
                "label_corrections",
                "comments",
                "previous_reviews",
            ):
                if field in old:
                    record[field] = old[field]
        else:
            record["previous_reviews"] = [
                *old.get("previous_reviews", []),
                {k: v for k, v in old.items() if k != "previous_reviews"},
            ]
    save_json(review_path, records)
    if judged is not None or not (destination / "judgment-review.json").exists():
        save_json(destination / "judgment-review.json", judged or [])
    lines = [
        "# Scenario review",
        "",
        "All labels are model-authored drafts. Review importance, interpretation, task",
        "feasibility, scoring scope, and source cutoff before accepting a case.",
        "",
        "Edit scenario-review.json with reviewer identity, status, and corrections.",
        "Judgment review retains scores, evidence, explanations, and uncertainty.",
        "",
    ]
    for scenario in scenarios:
        lines.extend(
            [
                f"## {scenario['id']} — {scenario['family']}",
                "",
                "Genre: " + scenario.get("genre", "unspecified"),
                "",
                "Instruction specificity: "
                + scenario.get("instruction_specificity", "unspecified"),
                "",
                scenario["visible"]["brief"],
                "",
                "**Private review labels**",
                "",
                "```json",
                json.dumps(scenario["labels"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    (destination / "review.md").write_text("\n".join(lines))
    return records
