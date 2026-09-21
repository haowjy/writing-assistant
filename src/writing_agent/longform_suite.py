"""Compile the held-out long-form benchmark into the shared scenario format.

The tracked specification names a public-domain work, the sections to supply, and the
files to write. This module turns that into a catalog and executable scenarios, and it
refuses to build when a case cites a fact its own supplied text does not contain.

Supplied manuscripts are deterministic slices, so a rebuild reproduces the same bytes.
The benchmark is reserved for final evaluation: no training source, no development case
and no checkpoint-selection decision may use it.
"""

import json
import re
from pathlib import Path

from writing_agent.catalog import fingerprint
from writing_agent.development import PROSE_RUBRIC, WRITE_TOOLS

BENCHMARK = "longform-v1"
MANUSCRIPT = "manuscript"


def section_slug(heading: str) -> str:
    """A stable file stem for one supplied section heading."""
    return re.sub(r"[^a-z0-9]+", "-", heading.strip().rstrip(".").lower()).strip("-")


def split_sections(text: str, pattern: str) -> list[tuple[str, str]]:
    """Split a work on its own heading pattern, keeping each heading with its body."""
    marks = list(re.finditer(pattern, text, re.M))
    if not marks:
        raise ValueError(f"No sections matched the heading pattern: {pattern}")
    return [
        (
            mark.group(0).strip(),
            text[
                mark.start() : (marks[index + 1].start() if index + 1 < len(marks) else len(text))
            ],
        )
        for index, mark in enumerate(marks)
    ]


def _work_text(raw: Path, work: dict) -> str:
    path = raw / "gutenberg" / f"{work['gutenberg']}.txt"
    if not path.exists():
        raise ValueError(f"Missing acquired work: {path}")
    return path.read_text()


def supplied_files(spec: dict, work: dict, sections: list[tuple[str, str]], case: dict) -> dict:
    """Materialize the supplied manuscript, one file per declared section."""
    names = case.get("names", {})
    files = {}
    for index in case["supplied"]:
        heading, body = sections[index]
        stem = names.get(str(index), section_slug(heading))
        files[f"{MANUSCRIPT}/{stem}.md"] = body.strip() + "\n"
    return files


def verify_anchors(case: dict, files: dict) -> None:
    """Every anchor must occur in the text this case actually hands the candidate."""
    text = "\n".join(files.values()).casefold()
    missing = [anchor for anchor in case.get("anchors", []) if anchor.casefold() not in text]
    if missing:
        raise ValueError(f"{case['id']} cites facts its supplied text lacks: {missing}")


def _budgets(files: dict, case: dict) -> dict:
    """Budgets sized to the supplied manuscript plus its declared deliverables."""
    supplied_bytes = sum(len(text.encode()) for text in files.values())
    touched = len(case["supplied"]) + len(case.get("write", []))
    return {
        "max_steps": 12 + 10 * max(touched, 1),
        "max_tool_calls": 4 * max(touched, 1) + 24,
        "max_read_tokens": max(40_000, 12 * len(json.dumps(files)) // 4),
        "max_total_bytes": max(250_000, 6 * supplied_bytes),
    }


def _prose(case: dict) -> list[dict]:
    if case.get("replies"):
        return [{"id": "reply", "kind": "reply", "selection": "whole"}]
    selectors = [
        {"id": stem, "kind": "file", "path": f"{MANUSCRIPT}/{stem}.md", "selection": "whole"}
        for stem in case.get("write", [])
    ]
    if case["axis"] == "kb_bootstrapped_novel":
        selectors.append(
            {
                "id": "new-case",
                "kind": "file",
                "path": f"{MANUSCRIPT}/new-case.md",
                "selection": "whole",
            }
        )
    return selectors


def build_release(spec_path: Path, raw: Path) -> tuple[list[dict], list[dict]]:
    """Return ``(catalog, scenarios)`` for the tracked specification."""
    spec = json.loads(spec_path.read_text())
    catalog, scenarios = [], []
    cache = {}
    for work_id, work in spec["works"].items():
        text = _work_text(raw, work)
        cache[work_id] = split_sections(text, work["headings"])
        catalog.append(
            {
                "id": work_id,
                "work_id": work_id,
                "role": spec["role"],
                "provenance": "human",
                "text": text,
                "sha256": fingerprint(text),
                "parents": [],
                "revision": fingerprint(work),
                "author_id": work["author"],
                "upstream_split": "unspecified",
                "review_status": "frozen",
                "transformations": [],
                "terms": {
                    "license": "Project Gutenberg license in raw file; US public domain",
                    "evaluation": "reserved: final benchmark",
                    "training": "excluded: final evaluation benchmark",
                    "redistribution": "review_terms",
                },
                "title": work["title"],
            }
        )
    for case in spec["cases"]:
        manuscript = supplied_files(spec, spec["works"][case["work"]], cache[case["work"]], case)
        # Anchors must be grounded in the supplied manuscript itself, not in a file we wrote.
        verify_anchors(case, manuscript)
        files = {**manuscript, **case.get("extra_files", {})}
        scenarios.append(
            {
                "id": case["id"],
                "family": case["family"],
                "role": spec["role"],
                "condition": case["axis"],
                "style": case["style"],
                "instruction_specificity": case["specificity"],
                "genre": case["genre"],
                "provenance": "human",
                "review_status": "frozen",
                "source_ids": [case["work"]],
                "benchmark": spec["benchmark"],
                "longform_axis": case["axis"],
                "title": case["title"],
                "context_target_tokens": case["context_target_tokens"],
                "visible": {
                    "brief": case["brief"],
                    "initial_files": files,
                    "followups": list(case.get("followups", [])),
                    "tools": [] if case.get("replies") else list(WRITE_TOOLS),
                    "budgets": _budgets(files, case),
                    "prose": _prose(case),
                },
                "labels": {
                    "rubric_version": 1,
                    "benchmark": spec["benchmark"],
                    "axis": case["axis"],
                    "anchors": list(case.get("anchors", [])),
                    "entrypoints": list(case.get("entrypoints", [])),
                    "checks": [dict(check) for check in case["checks"]],
                    "rubrics": {"Q2": json.loads(json.dumps(PROSE_RUBRIC))},
                },
            }
        )
    return catalog, scenarios


def documented_sampling(spec_path: Path) -> dict:
    """The specimen's declared attempts per case, and what that sample count supports.

    A suite that names a sample count but not its consequences invites a metric to fail
    quietly at scoring time, months after the GPU time was spent. This resolves the
    declared count against the sample floors so the gap is visible at build time.
    """
    from writing_agent.prose import sampling_plan

    spec = json.loads(spec_path.read_text())
    sampling = spec.get("sampling") or {}
    samples = sampling.get("samples_per_case", 1)
    if type(samples) is not int or samples < 1:
        raise ValueError("sampling.samples_per_case must be a positive integer")
    return {
        "samples_per_case": samples,
        "plan": sampling_plan(samples),
        "distributional_component": sampling.get("distributional_component", "undeclared"),
    }


def holdout_audit(catalog: list[dict], *, claimed: dict[str, set[str]]) -> dict:
    """Prove the benchmark shares no text with anything already spoken for.

    Checked by content hash rather than by filename, because the same book can be
    imported twice under different names. ``claimed`` maps a label to the hashes that
    are genuinely in use, not merely present in some catalog: a work imported as an
    available source but never referenced is still free to reserve for final evaluation.
    """
    hashes = {record["sha256"] for record in catalog}
    disputed = {
        label: sorted(hashes & set(values))
        for label, values in claimed.items()
        if hashes & set(values)
    }
    if disputed:
        raise ValueError(f"Benchmark sources are not held out: {disputed}")
    return {
        "status": "held_out",
        "benchmark_hashes": len(hashes),
        "claimed": {label: len(values) for label, values in claimed.items()},
    }


def claimed_hashes(catalog_path: Path, *, ids=None) -> set[str]:
    """Hashes that are actually in use: a whole catalog, or a referenced subset."""
    if not catalog_path.exists():
        return set()
    records = json.loads(catalog_path.read_text())
    records = records if isinstance(records, list) else records.get("scenarios", [])
    if ids is None:
        return {record["sha256"] for record in records if record.get("sha256")}
    by_id = {record["id"]: record for record in records}
    return {by_id[key]["sha256"] for key in ids if key in by_id and by_id[key].get("sha256")}


def freeze(manifest: dict, destination: Path) -> dict:
    """Write the benchmark's identity next to the compiled release."""
    frozen = {
        "benchmark": manifest.get("benchmark", BENCHMARK),
        "catalog_hash": manifest["catalog_hash"],
        "cases": len(manifest["scenarios"]),
        "case_hashes": {
            row["id"]: {"visible": row["visible_hash"], "labels": row["labels_hash"]}
            for row in manifest["scenarios"]
        },
    }
    frozen["freeze_hash"] = fingerprint(frozen)
    return frozen
