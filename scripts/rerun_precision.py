"""Rerun a saved selection at a different weight precision.

The 50-case E2B baseline was generated in nf4. Every cross-model comparison built on it
therefore carries an unquantified confound: nobody can say how much of a difference is
capability and how much is 4-bit weight approximation. E2B fits in bf16 on this device, so
the confound is measurable rather than permanent.

This copies the source run's model record and flips exactly one field. Scenarios, seeds,
generation settings, context limit and max tokens are carried over verbatim, because a
comparison that also changes the context window measures two things and attributes both to
precision.

Only deterministic metrics are compared, so both arms are scored identically and nothing
here costs money. Semantic rubrics stay out of it; they would need a second paid grading
pass.

Inspect with `uv run python scripts/rerun_precision.py`; pass `--execute` to run.
"""

import argparse
import json
from pathlib import Path

from writing_agent.catalog import save_json
from writing_agent.inference import evaluate_checkpoint
from writing_agent.scoring import mechanical_score

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/custom50-e2b-it-2026-09-14"
DESTINATION = ROOT / "runs/custom50-e2b-it-2026-09-21-bf16"
QUANTIZATION = "none"  # bf16 weights; the source run used "nf4"
COMPARABLE = ("Q1", "Q3", "Q4", "Q13", "R1")


def model_config() -> dict:
    """The source record with only the precision changed."""
    model = json.loads((SOURCE / "selection.json").read_text())["model"]
    return {**model, "quantization": QUANTIZATION}


def generate(*, execute: bool = False, allow_download: bool = False) -> None:
    selection = json.loads((SOURCE / "selection.json").read_text())
    scenarios = selection["scenarios"]
    model = model_config()
    save_json(
        DESTINATION / "selection.json",
        {
            "model": model,
            "scenarios": scenarios,
            "source_run": str(SOURCE.relative_to(ROOT)),
            "controlled_variable": "quantization",
            "limitation": (
                "Single-variable rerun of a saved selection; identical except weight precision."
            ),
        },
    )
    return evaluate_checkpoint(
        scenarios,
        model,
        DESTINATION / "attempts",
        execute=execute,
        retry_failed=True,
        allow_download=allow_download,
    )


def scores(root: Path) -> dict[str, dict]:
    """Deterministic metrics per scenario, computed the same way for both arms."""
    collected = {}
    for path in sorted(root.glob("attempts/*/attempt-*/result.json")):
        result = json.loads(path.read_text())
        scenario = json.loads((path.parent / "visible.json").read_text())
        card = mechanical_score(scenario, result)
        if card["status"] != "completed":
            continue
        collected[result["scenario_id"]] = card["scores"]
        save_json(path.parent / "card.json", card)
    return collected


def compare() -> dict:
    arms = {"nf4": scores(SOURCE), "bf16": scores(DESTINATION)}
    shared = sorted(set(arms["nf4"]) & set(arms["bf16"]))
    table = {}
    for key in COMPARABLE:
        row = {}
        for arm, cards in arms.items():
            values = [
                cards[case][key]["value"]
                for case in shared
                if cards[case].get(key, {}).get("status") == "ok"
            ]
            row[arm] = sum(values) / len(values) if values else None
            row[f"{arm}_n"] = len(values)
        if row["nf4"] is not None and row["bf16"] is not None:
            row["delta"] = row["bf16"] - row["nf4"]
        table[key] = row
    result = {
        "cases_nf4": len(arms["nf4"]),
        "cases_bf16": len(arms["bf16"]),
        "overlap": len(shared),
        "metrics": table,
        "limitation": (
            "Deterministic metrics only. Sampling is not reproducible across precisions, "
            "so per-case deltas carry generation noise; only the aggregate is meaningful."
        ),
    }
    save_json(DESTINATION / "precision-comparison.json", result)
    write_report(result)
    return result


def write_report(result: dict) -> None:
    lines = [
        "# E2B precision comparison: nf4 vs bf16",
        "",
        f"{result['overlap']} cases completed in both arms "
        f"({result['cases_nf4']} nf4, {result['cases_bf16']} bf16).",
        "",
        "| Metric | nf4 | bf16 | Δ | n |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, row in result["metrics"].items():

        def fmt(value):
            return "—" if value is None else f"{value:.4f}"

        lines.append(
            f"| {key} | {fmt(row['nf4'])} | {fmt(row['bf16'])} | "
            f"{fmt(row.get('delta'))} | {row['bf16_n']} |"
        )
    lines += ["", result["limitation"], "", "[Full comparison](precision-comparison.json)", ""]
    (DESTINATION / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Generate; requires the GPU")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare saved arms only")
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare()["metrics"], indent=1))
        return
    planned = generate(execute=args.execute, allow_download=args.allow_download)
    if not args.execute:
        model = model_config()
        print(
            f"{len(planned)} scenario(s) planned for {DESTINATION.name} "
            f"({model['id']} @ {model['quantization']}, context {model['context_tokens']})"
        )
        return
    compare()


if __name__ == "__main__":
    main()
