"""Grade saved custom50 artifacts in fresh, independently instructed Astra sessions."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.pilot_e2b import ROOT
from writing_agent.artifacts import extract_prose
from writing_agent.catalog import save_json
from writing_agent.grading import CodexGrader, apply_judgment, grading_packet
from writing_agent.scoring import build_report

SOURCE = ROOT / "runs/custom50-e2b-it-2026-09-14"
OUTPUT = SOURCE / "astra-graded"


def inputs():
    selection = json.loads((SOURCE / "selection.json").read_text())
    results = {
        json.loads(p.read_text())["scenario_id"]: p for p in SOURCE.glob("attempts/*/*/result.json")
    }
    rows = []
    for scenario in selection["scenarios"]:
        path = results[scenario["id"]]
        result = json.loads(path.read_text())
        card = json.loads((SOURCE / "rescored" / (scenario["id"] + ".json")).read_text())
        card["artifacts"] = extract_prose(result, scenario["visible"]["prose"])
        rows.append((scenario, result, card, path.parent))
    assert len(rows) == 50
    return rows


def run(*, subset=None, workers=3):
    grader = CodexGrader(OUTPUT / "judgments", max_calls=50, timeout=600)
    rows = inputs()

    def grade(row):
        scenario, result, card, _ = row
        packet = grading_packet(scenario, result, card)
        record = grader.grade(packet)
        scored = apply_judgment(card, packet, record)
        save_json(OUTPUT / scenario["id"] / "scorecard.json", scored)
        return scenario["id"], record["status"]

    selected = [r for r in rows if subset is None or r[0]["id"] in subset]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(grade, row) for row in selected]):
            print(*future.result(), flush=True)
            report()
    return report()


def report():
    cards = []
    lines = [
        "# Astra grading — custom50",
        "",
        "Independent, blinded gpt-6-astra judgments. Human calibration/review remains pending.",
        "Numerical metrics are preserved. Astra grades semantic rubrics and checks.",
        "",
        "| Case | Grading | Task completion | Evidence and scores |",
        "|---|---|---|---|",
    ]
    for scenario, _, original, _attempt in inputs():
        path = OUTPUT / scenario["id"] / "scorecard.json"
        card = json.loads(path.read_text()) if path.exists() else original
        cards.append(card)
        status = card.get("judgment", {}).get("status", "pending")
        task = card["scores"]["Q3"]
        value = task["value"] if task["status"] == "ok" else task["status"]
        link = f"[Scorecard]({scenario['id']}/scorecard.json)" if path.exists() else "Pending"
        lines.append(f"| {scenario['id']} | {status} | {value} | {link} |")
        if status == "ok":
            details = [
                f"# {scenario['id']} — Astra assessment",
                "",
                f"Task completion: {value}.",
                "",
            ]
            for assessment in card["judgment"]["judgment"]["assessments"]:
                details += [
                    f"## {assessment['metric']}: {assessment['value']}",
                    "",
                    assessment["rationale"],
                    "",
                    f"Uncertainty: {assessment['uncertainty']}",
                    "",
                ]
                details += ["- " + evidence for evidence in assessment["evidence"]]
                details.append("")
            for check in card["judgment"]["judgment"]["checks"]:
                details += [
                    f"## {check['id']}: {'pass' if check['passed'] else 'fail'}",
                    "",
                    check["rationale"],
                    "",
                ]
                details += ["- " + evidence for evidence in check["evidence"]]
                details.append("")
            (path.parent / "review.md").write_text("\n".join(details))
            lines[-1] = lines[-1].replace(
                link, f"[Read assessment]({scenario['id']}/review.md) · " + link
            )
    completed = sum(c.get("judgment", {}).get("status") == "ok" for c in cards)
    status = {
        "graded": completed,
        "expected": 50,
        "task_completion_pending": sum(c["scores"]["Q3"]["status"] == "pending" for c in cards),
    }
    save_json(OUTPUT / "status.json", status)
    build_report(cards, OUTPUT / "reports")
    lines[2:2] = [f"{completed}/50 cases graded.", ""]
    (OUTPUT / "review.md").write_text("\n".join(lines) + "\n")
    return status


def publish():
    """Update the original review entry points after all judgments validate; archive old views."""
    import os
    import re

    rows = inputs()
    cards = {}
    for scenario, result, original, _ in rows:
        path = OUTPUT / scenario["id"] / "scorecard.json"
        previous = json.loads(path.read_text())
        packet = grading_packet(scenario, result, original)
        cards[scenario["id"]] = apply_judgment(original, packet, previous["judgment"])
        save_json(path, cards[scenario["id"]])
    report()
    if any(c.get("judgment", {}).get("status") != "ok" for c in cards.values()):
        raise ValueError("All 50 judgments must validate before publishing the complete review")
    lines = [
        "# Gemma E2B-IT — 50-case development run",
        "",
        "All 50 attempts have Astra semantic judgments. Human calibration/review remains pending.",
        "Task completion checks required conditions; prose quality is scored separately.",
        "",
        "[Astra assessments](astra-graded/review.md) · "
        "[Aggregate scores](astra-graded/reports/report.md)",
        "",
        "[Exact inputs](selection.json) · [Pre-Astra review](review.pre-astra.md) · "
        "[Numerical profiles before Astra grading](rescored/review.md)",
        "",
        "| Case | Genre | Request | Execution | Task completion | Failed checks | Review |",
        "|---|---|---|---|---|---|---|",
    ]
    for scenario, result, _, attempt in rows:
        card = cards[scenario["id"]]
        for name in ("scorecard.json", "review.md"):
            source = attempt / name
            backup = source.with_name(source.stem + ".pre-astra" + source.suffix)
            if not backup.exists():
                backup.write_bytes(source.read_bytes())
        save_json(attempt / "scorecard.json", card)
        q3 = card["scores"]["Q3"]
        value = q3["value"] if q3["status"] == "ok" else q3["status"]
        failed = ", ".join(c["id"] for c in card["checks"] if c.get("passed") is False) or "none"
        text = (attempt / "review.pre-astra.md").read_text()
        text = re.sub(
            r"^Task completion: .*?$",
            f"Task completion: {value}. Failed checks: {failed}.",
            text,
            flags=re.M,
        )
        relative = os.path.relpath(OUTPUT / scenario["id"] / "review.md", attempt)
        text += f"\n## Astra semantic assessment\n\n[Read evidence and judgments]({relative}). "
        text += "Human review and calibration remain pending.\n"
        (attempt / "review.md").write_text(text)
        link = (attempt / "review.md").relative_to(SOURCE).as_posix()
        lines.append(
            f"| {scenario['id']} | {scenario['genre']} | {scenario['instruction_specificity']} | "
            f"{result['status']} | {value} | {failed} | [Read case]({link}) |"
        )
    main = SOURCE / "review.md"
    backup = SOURCE / "review.pre-astra.md"
    if not backup.exists():
        backup.write_bytes(main.read_bytes())
    main.write_text("\n".join(lines) + "\n")
