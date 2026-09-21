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
from writing_agent.suite import load_scenarios

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/processed/custom-eval"
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


def scores(root: Path, scenarios: dict[str, dict]) -> dict[str, dict]:
    """Deterministic metrics per scenario, computed the same way for both arms."""
    collected = {}
    for path in sorted(root.glob("attempts/*/attempt-*/result.json")):
        result = json.loads(path.read_text())
        scenario = scenarios.get(result.get("scenario_id"))
        if scenario is None:
            continue
        card = mechanical_score(scenario, result)
        if card["status"] != "completed":
            continue
        collected[result["scenario_id"]] = card["scores"]
        save_json(path.parent / "card.json", card)
    return collected


def compare() -> dict:
    selection = json.loads((SOURCE / "selection.json").read_text())
    ids = [scenario["id"] for scenario in selection["scenarios"]]
    scenarios = {scenario["id"]: scenario for scenario in load_scenarios(RELEASE, ids)}
    arms = {"nf4": scores(SOURCE, scenarios), "bf16": scores(DESTINATION, scenarios)}
    shared = sorted(set(arms["nf4"]) & set(arms["bf16"]))
    table = {}
    for key in COMPARABLE:
        # Paired on the cases where both arms actually produced the metric. Averaging each
        # arm over whatever cases happened to be scored compares different case subsets,
        # and a case drops out precisely when its checks did not all resolve.
        pairs = [
            (arms["nf4"][case][key]["value"], arms["bf16"][case][key]["value"])
            for case in shared
            if arms["nf4"][case].get(key, {}).get("status") == "ok"
            and arms["bf16"][case].get(key, {}).get("status") == "ok"
            and arms["nf4"][case][key]["value"] is not None
            and arms["bf16"][case][key]["value"] is not None
        ]
        table[key] = {
            "paired": len(pairs),
            "nf4": sum(a for a, _ in pairs) / len(pairs) if pairs else None,
            "bf16": sum(b for _, b in pairs) / len(pairs) if pairs else None,
            "delta": sum(b - a for a, b in pairs) / len(pairs) if pairs else None,
            "nf4_scored": sum(
                1 for case in shared if arms["nf4"][case].get(key, {}).get("status") == "ok"
            ),
            "bf16_scored": sum(
                1 for case in shared if arms["bf16"][case].get(key, {}).get("status") == "ok"
            ),
        }
    result = {
        "cases_nf4": len(arms["nf4"]),
        "cases_bf16": len(arms["bf16"]),
        "overlap": len(shared),
        "metrics": table,
        "limitation": (
            "Deterministic metrics only, paired on cases where both arms scored. Most "
            "development checks are semantic and stay pending until a judge runs, so the "
            "paired counts are small and cover only the all-deterministic cases. Sampling "
            "is not reproducible across precisions, so per-case deltas carry generation "
            "noise. This measures the precision confound on a subset, not the baseline."
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
        "Paired on cases where both arms produced the metric, so the two columns are the",
        "same case set. *Scored* shows how many cases each arm resolved on its own.",
        "",
        "| Metric | Paired | nf4 | bf16 | Δ | nf4 scored | bf16 scored |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, row in result["metrics"].items():

        def fmt(value):
            return "—" if value is None else f"{value:.4f}"

        lines.append(
            f"| {key} | {row['paired']} | {fmt(row['nf4'])} | {fmt(row['bf16'])} | "
            f"{fmt(row['delta'])} | {row['nf4_scored']} | {row['bf16_scored']} |"
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
