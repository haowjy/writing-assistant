"""Mechanical scorecards and reports over saved artifacts; no candidate calls."""

import fnmatch
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from writing_agent.artifacts import extract_prose, markdown_graph
from writing_agent.catalog import fingerprint, save_json

QUALITY = {
    "Q1": "Instruction adherence",
    "Q2": "Prose quality",
    "Q3": "Task completion",
    "Q4": "Tool correctness",
    "Q5": "Planning usefulness",
    "Q6": "Alternative diversity",
    "Q7": "KB faithfulness",
    "Q8": "Important-information coverage",
    "Q9": "Knowledge interpretation",
    "Q10": "Navigability",
    "Q11": "Update correctness",
    "Q12": "Retrieval success",
    "Q13": "Continuity",
}


def measurement(value=None, *, status="ok", method="deterministic", reason=None, **extra):
    return {"value": value, "status": status, "method": method, "reason": reason, **extra}


def aggregate_checks(
    checks: list[dict], execution_status: str, artifacts: list[dict], *, rubric_metrics=()
) -> dict:
    """One aggregation policy for mechanical and judge-supplied check outcomes.

    Q3 is always derived from execution, delivery, and every required check;
    it is never the average of checks that happen to carry the Q3 label.
    """
    scores = {}
    for metric in {c["metric"] for c in checks} - set(rubric_metrics) - {"Q3"}:
        applicable = [c for c in checks if c["metric"] == metric]
        scored = [c for c in applicable if c["status"] == "ok"]
        total = sum(c.get("weight", 1) for c in scored)
        numerator = sum(c.get("weight", 1) * c["passed"] for c in scored)
        method = (
            "llm_assisted"
            if any(c["method"] == "llm_judge" for c in applicable)
            else (
                "reference_match"
                if any(c["method"] == "reference_match" for c in applicable)
                else "deterministic"
            )
        )
        scores[metric] = measurement(
            numerator / total if total else None,
            status="ok" if len(scored) == len(applicable) else "pending",
            method=method,
            numerator=numerator,
            denominator=total,
            pending=len(applicable) - len(scored),
        )
    required = [c for c in checks if c.get("required", False)]
    if execution_status != "completed" or any(a["status"] == "missing_prose" for a in artifacts):
        scores["Q3"] = measurement(0, denominator=1)
    elif any(c["passed"] is False for c in required):
        scores["Q3"] = measurement(
            0,
            denominator=1,
            method="llm_assisted"
            if any(c["method"] == "llm_judge" and c["passed"] is False for c in required)
            else "deterministic",
        )
    elif any(c["status"] != "ok" for c in required) or any(
        a["status"] == "needs_review" for a in artifacts
    ):
        scores["Q3"] = measurement(status="pending", reason="Required checks or extraction pending")
    else:
        scores["Q3"] = measurement(
            1,
            denominator=1,
            method="llm_assisted"
            if any(c["method"] == "llm_judge" for c in required)
            else "deterministic",
        )
    return scores


def mechanical_score(scenario: dict, result: dict) -> dict:
    labels = scenario["labels"]
    artifacts = extract_prose(result, scenario["visible"]["prose"])
    checks = []
    changed = {
        p
        for p in result.get("before", {}).keys() | result.get("after", {}).keys()
        if result.get("before", {}).get(p) != result.get("after", {}).get(p)
    }
    for check in labels["checks"]:
        entry = {**check, "status": "pending", "passed": None}
        if check["method"] not in {"deterministic", "reference_match"}:
            checks.append(entry)
            continue
        kind = check["kind"]
        text = result.get("output", "")
        if "path" in check:
            text = result.get("after", {}).get(check["path"], "")
        if "artifact" in check:
            matches = [a for a in artifacts if a["id"] == check["artifact"] and a["status"] == "ok"]
            if not matches:
                entry.update(status="unscored", reason="No valid prose extraction")
                checks.append(entry)
                continue
            text = matches[0]["text"]
        if kind == "kb_word_budget":
            passed = (
                sum(
                    len(t.split())
                    for p, t in result.get("after", {}).items()
                    if p.startswith("kb/")
                )
                <= check["max"]
            )
        elif kind == "wiki_links":
            wiki = {p: t for p, t in result.get("after", {}).items() if p.startswith("kb/")}
            graph = markdown_graph(wiki, labels.get("entrypoints", ["kb/index.md"]))
            passed = (
                graph["pages"] >= 2
                and graph["valid_fraction"] == 1
                and graph["reachability"] == 1
                and not graph["unsupported_reference_links"]
            )
        elif kind == "nonempty":
            passed = bool(text.strip())
        elif kind == "contains":
            passed = check["text"].casefold() in text.casefold()
        elif kind == "excludes":
            passed = check["text"].casefold() not in text.casefold()
        elif kind == "word_range":
            passed = check["min"] <= len(text.split()) <= check["max"]
        elif kind == "exact":
            passed = text == check["text"]
        elif kind == "protected":
            before = result.get("before", {}).get(check["path"])
            passed = (
                text == before
                if "text" not in check
                else (check["text"] in text and before is not None and check["text"] in before)
            )
        elif kind == "allowed_changes":
            passed = all(
                any(fnmatch.fnmatchcase(p, pattern) for pattern in check["paths"]) for p in changed
            )
        elif kind == "evidence_exposed":
            observations = [
                json.dumps(e["observation"].get("result", ""), ensure_ascii=False)
                for e in result.get("trace", [])
                if e["type"] == "tool"
                and e["observation"]["ok"]
                and e["call"]["function"]["name"] in {"read_file", "search"}
            ]
            passed = any(check["text"].casefold() in obs.casefold() for obs in observations)
        elif kind == "alias_answer":
            normalized = " ".join(text.casefold().split()).strip(".!?")
            passed = normalized in {
                " ".join(a.casefold().split()).strip(".!?") for a in check["aliases"]
            }
        else:
            raise ValueError(f"Unknown mechanical check: {kind}")
        entry.update(status="ok", passed=passed)
        checks.append(entry)
    scores = {
        key: measurement(status="not_applicable", reason="No applicable checks") for key in QUALITY
    }
    calls = [e for e in result.get("trace", []) if e["type"] == "tool"]
    attempted = result.get("attempted_tool_calls", len(calls))
    if attempted:
        scores["Q4"] = measurement(
            sum(e["observation"].get("valid", False) for e in calls) / attempted,
            denominator=attempted,
            execution_errors=sum(not e["observation"]["ok"] for e in calls),
        )
    if "entrypoints" in labels:
        kb = {p: t for p, t in result.get("after", {}).items() if p.startswith("kb/")}
        graph = markdown_graph(kb, labels["entrypoints"])
    else:
        graph = None
    for key in labels.get("rubrics", {}):
        scores[key] = measurement(
            status="pending", method="llm_judge", reason="Awaiting evidence-backed judgment"
        )
    if not any(a["status"] == "ok" for a in artifacts):
        scores["Q2"] = measurement(status="not_applicable", reason="No valid selected prose")
    if labels.get("probes"):
        scores["Q10"] = measurement(status="pending", reason="Fresh-reader probes not executed")
    scores["R1"] = measurement(result.get("latency_seconds"), method="deterministic")
    scores["R2"] = measurement(
        result.get("usage") or None,
        status="ok" if result.get("usage") else "unavailable",
        reason=None if result.get("usage") else "Backend usage not reported",
    )
    scores.update(
        aggregate_checks(
            checks, result["status"], artifacts, rubric_metrics=labels.get("rubrics", {})
        )
    )
    return {
        "version": 1,
        "result_hash": fingerprint(result),
        "labels_hash": fingerprint(labels),
        "scenario_id": scenario["id"],
        "scores": scores,
        "checks": checks,
        "artifacts": artifacts,
        "wiki_graph": graph,
        "status": result["status"],
        "family": scenario["family"],
        "condition": scenario["condition"],
        "provenance": result.get("provenance", "unknown"),
        "input_provenance": scenario["provenance"],
        "model": result.get("model", {}),
        "source_groups": scenario.get("source_groups", []),
        "style": scenario.get("style", "unspecified"),
        "instruction_specificity": scenario.get("instruction_specificity", "unspecified"),
        "genre": scenario.get("genre", "unspecified"),
    }


def build_report(scorecards: list[dict], destination: Path | None = None) -> dict:
    """Macro means retain conditional denominators and pending/failure counts."""
    groups = defaultdict(list)
    for card in scorecards:
        key = (
            card.get("model", {}).get("id", "unknown"),
            card["family"],
            card["condition"],
            card["provenance"],
            card["input_provenance"],
            card["style"],
            fingerprint(card.get("model", {})),
            card.get("instruction_specificity", "unspecified"),
            card.get("genre", "unspecified"),
        )
        groups[key].append(card)
    summaries = []
    for key, cards in sorted(groups.items()):
        metrics = {}
        for metric in [*QUALITY, "R1"]:
            values = [
                c["scores"][metric]["value"]
                for c in cards
                if c["scores"][metric]["status"] == "ok"
                and isinstance(c["scores"][metric]["value"], (float, int))
            ]
            metrics[metric] = {
                "mean": mean(values) if values else None,
                "median": median(values) if values else None,
                "scored": len(values),
                "unscored": len(cards) - len(values),
            }
        latencies = sorted(
            c["scores"]["R1"]["value"] for c in cards if c["scores"]["R1"]["value"] is not None
        )
        metrics["R1"].update(
            median=median(latencies) if latencies else None,
            p95=latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
        )
        summaries.append(
            {
                "model": key[0],
                "family": key[1],
                "condition": key[2],
                "provenance": key[3],
                "input_provenance": key[4],
                "style": key[5],
                "model_configuration": cards[0].get("model", {}),
                "model_configuration_hash": key[6],
                "instruction_specificity": key[7],
                "genre": key[8],
                "attempts": len(cards),
                "failed_execution": sum(c["status"] != "completed" for c in cards),
                "source_groups": sorted({g for c in cards for g in c["source_groups"]}),
                "metrics": metrics,
            }
        )
    report = {
        "version": 1,
        "attempts": len(scorecards),
        "groups": summaries,
        "ranking": None,
        "calibration": "unvalidated",
        "uncertainty": "No confidence intervals: development labels await review",
    }
    if destination is not None:
        save_json(destination / "report.json", report)
        lines = [
            "# Development evaluation report",
            "",
            f"{len(scorecards)} saved attempts. Semantic calibration is unvalidated.",
            "",
            "| Model | Family | Condition | Genre | Specificity | Attempts | Execution failures |",
            "|---|---|---|---|---|---:|---:|",
        ]
        for group in summaries:
            lines.append(
                f"| {group['model']} | {group['family']} | {group['condition']} | "
                f"{group['genre']} | {group['instruction_specificity']} | {group['attempts']} | "
                f"{group['failed_execution']} |"
            )
        (destination / "report.md").write_text("\n".join(lines) + "\n")
    return report
