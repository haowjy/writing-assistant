"""Sample one prompt many times so the across-output measures have something to measure.

The rest of the suite runs one attempt per case, which is why D2 (MMD), D4 (self-BLEU)
and D6 (dispersion) report insufficient_samples everywhere. This run is deliberately the
opposite shape: one short prompt, sampled enough times to clear the floors, so the
baseline gets a real distribution number instead of a withheld one.

Precision is an explicit axis, not a setting. A 4-bit measurement and a bf16 measurement
are not the same quantity, so each arm writes to its own directory and the seeds are
shared between them: run `--quantization nf4` and `--quantization bf16` with the same
repeat count and every difference between the two is quantization rather than sampling.

Inspect with `uv run python scripts/baseline_distribution.py`; pass `--execute` to run.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

from writing_agent.catalog import fingerprint, save_json
from writing_agent.inference import (
    HARNESS_CONTEXT_TOKENS,
    PROTOCOL,
    TransformersBackend,
    load_checkpoint,
)
from writing_agent.prose import ProseFeatures, bandwidth, sample_distribution, sampling_plan
from writing_agent.scoring import mechanical_score
from writing_agent.suite import load_scenarios, run_selected

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/processed/custom-eval"
STAMP = "2026-09-21"
QUANTIZATIONS = ("bf16", "nf4")
SCENARIO_IDS = ["F1-01"]
REPEATS = 25
BASE_SEED = 4200
MODEL = {
    "id": "google/gemma-4-E2B-it",
    "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "variant": "it",
    "kind": "transformers",
    "protocol": PROTOCOL,
    "prompt_format": "chat",
    "enable_thinking": True,
    "loader": "causal_lm",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "attention": "sdpa",
    "context_tokens": HARNESS_CONTEXT_TOKENS,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": BASE_SEED,
    "max_tokens": 2048,
    "purpose": "local repeated sampling to make the across-output prose measures computable",
}


def output_dir(quantization: str) -> Path:
    return ROOT / f"runs/distribution-e2b-it-{STAMP}-{quantization}"


def model_config(quantization: str) -> dict:
    """bf16 is the reference rendering; nf4 is the training-time approximation."""
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"Choose one of {QUANTIZATIONS}")
    return {**MODEL, "quantization": "none" if quantization == "bf16" else "nf4"}


def references(catalog_path: Path) -> list[dict]:
    """The same exploratory human reference policy the pilot rescoring used.

    Unmatched by genre or length, and recorded as such: this measures distance from a
    human prose distribution, not from the right human prose distribution.
    """
    catalog = json.loads(catalog_path.read_text())
    records = []
    for book in catalog:
        if not book["id"].startswith("gutenberg-"):
            continue
        eligible = [
            p.strip() for p in re.split(r"\n\s*\n", book["text"]) if 100 <= len(p.split()) <= 300
        ]
        for index, text in enumerate(eligible[:3]):
            records.append(
                {
                    "id": f"{book['id']}-paragraph-{index}",
                    "text": text,
                    "parent_id": book["id"],
                    "parent_hash": book["sha256"],
                    "provenance": "human",
                    "role": "development",
                    "review_status": "unreviewed",
                    "sha256": fingerprint(text),
                }
            )
    return records


def generate(
    destination: Path,
    quantization: str,
    *,
    execute: bool,
    repeats: int,
    allow_download: bool = False,
):
    """One model load, `repeats` generations, each on its own seed."""
    scenario = load_scenarios(RELEASE, SCENARIO_IDS)[0]
    config = {
        **model_config(quantization),
        "code_checkpoint": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    save_json(destination / "selection.json", {"model": config, "scenarios": [scenario]})
    if not execute:
        return run_selected([scenario], config, lambda: None, destination / "attempts")
    with load_checkpoint(config, allow_download=allow_download) as (model, tokenizer, record):
        for repeat in range(repeats):
            variant = {**record, "seed": BASE_SEED + repeat}
            run_selected(
                [scenario],
                variant,
                lambda variant=variant: TransformersBackend(model, tokenizer, variant),
                destination / "attempts",
                execute=True,
            )
    return collect(destination, scenario)


def collect(destination: Path, scenario: dict) -> list[dict]:
    """Score every saved attempt without a candidate or judge call."""
    cards = []
    for path in sorted((destination / "attempts").glob("*/attempt-*/result.json")):
        result = json.loads(path.read_text())
        if result.get("scenario_id") != scenario["id"]:
            continue
        card = mechanical_score(scenario, result)
        card["seed"] = json.loads((path.parent / "started.json").read_text())["model"]["seed"]
        cards.append(card)
        save_json(path.parent / "card.json", card)
    return cards


def review(destination: Path, cards: list[dict], *, allow_download: bool = False) -> dict:
    extractor = ProseFeatures(destination / "features")
    records = references(RELEASE / "catalog.json")
    save_json(
        destination / "reference-selection.json",
        {
            "policy": "First three 100-300 word paragraphs per Gutenberg book (whitespace count)",
            "limitation": "Unreviewed; not genre or length matched; paragraphs share works",
            "references": records,
        },
    )
    features = [
        extractor.extract(r["text"], tokens=True, embeddings=True, allow_download=allow_download)
        for r in records
    ]
    vectors = [f["embedding"] for f in features if "embedding" in f]
    sigma = bandwidth(vectors) if len(vectors) == len(features) and len(vectors) >= 2 else None
    result = sample_distribution(
        cards,
        extractor,
        references=features,
        sigma=sigma,
        allow_download=allow_download,
    )
    result["model"] = MODEL["id"]
    result["model_revision"] = MODEL["revision"]
    result["instruction_specificity"] = "explicit"
    result["limitation"] = (
        "One development case sampled repeatedly; references are exploratory and unmatched. "
        "Measures the spread of one prompt's outputs, not performance."
    )
    save_json(destination / "distribution.json", result)
    write_report(destination, result)
    return result


def _format(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_format(v)}" for k, v in value.items())
    if isinstance(value, list):
        return f"{len(value)} entries"
    return str(value)


def write_report(destination: Path, result: dict) -> None:
    rows = []
    for name in ("D1", "D2", "D4", "D6", "D11"):
        entry = result["metrics"][name]
        value = entry.get("value")
        detail = _format(value) if value is not None else (entry.get("reason") or "")
        rows.append(
            f"| {name} | {entry['status']} | {entry.get('power', '')} | "
            f"{entry.get('samples', '')} | {detail} |"
        )
    duplicates = result["metrics"]["D11"]["value"].get("duplicate_output_rate")
    summary = (
        f"Duplicate output rate: {duplicates:.3f}."
        if duplicates is not None
        else "Duplicate output rate needs at least two attempts."
    )
    text = "\n".join(
        [
            "# E2B-IT repeated-sample baseline",
            "",
            f"Model `{result['model']}` at `{result['model_revision']}`, "
            f"{result['attempts']} attempts on `{result['scenario_id']}`.",
            "",
            "| Measure | Status | Power | Samples | Value |",
            "| --- | --- | --- | --- | --- |",
            *rows,
            "",
            summary,
            "",
            result["limitation"],
            "",
            "[Full measurements](distribution.json) · "
            "[Reference selection](reference-selection.json)",
            "",
        ]
    )
    (destination / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Generate; requires the GPU")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--quantization", choices=QUANTIZATIONS, default="bf16")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--collect", action="store_true", help="Review what is already saved")
    args = parser.parse_args()
    destination = output_dir(args.quantization)
    scenario = load_scenarios(RELEASE, SCENARIO_IDS)[0]
    if args.collect:
        print(json.dumps(review(destination, collect(destination, scenario))["metrics"], indent=1))
        return
    planned = generate(
        destination,
        args.quantization,
        execute=args.execute,
        repeats=args.repeats,
        allow_download=args.allow_download,
    )
    if not args.execute:
        print(f"{len(planned)} attempt(s) planned for {SCENARIO_IDS[0]} ({args.quantization})")
        print(f"sampling plan for {args.repeats} repeats: {sampling_plan(args.repeats)}")
        return
    review(destination, planned, allow_download=args.allow_download)


if __name__ == "__main__":
    main()
