"""Editable research experiment. Run with `uv run python scripts/evaluate.py`.

Default: inspect the proposed work, with no downloads or model calls.
Use the stage functions from another Python script, or edit STAGES below.
"""

import json
from dataclasses import asdict
from pathlib import Path

from writing_agent.acquisition import acquire_sources, compile_downloads
from writing_agent.catalog import save_json
from writing_agent.development import author_development
from writing_agent.grading import CodexGrader, apply_judgment, grading_packet, write_review
from writing_agent.inference import PROTOCOL, evaluate_checkpoint
from writing_agent.prose import (
    FeatureConfig,
    ProseFeatures,
    compare_groups,
    paired_similarity,
    prose_profile,
)
from writing_agent.scoring import build_report, mechanical_score
from writing_agent.suite import compile_scenarios, load_scenarios, saved_results

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RELEASE = DATA / "processed/custom-eval"
OUTPUT = ROOT / "runs/custom-eval"
SOURCES = ("tell_me_a_story", "hanna", "ifeval", "gutenberg")
PROVENANCE = {"human", "synthetic", "half_synthetic", "synthetic_fanfic", "unknown"}
SCENARIO_IDS = None  # Example: ['F1-01']; None selects the 50 development cases.
STAGES = ("inspect",)
APPROVED_CANDIDATE_RUN = False
FEATURES = FeatureConfig()
REFERENCE_IDS = []  # Select reviewed human reference records from the shared catalog.
MMD_BANDWIDTH = None  # Freeze from development reference embeddings before comparison.
COMPUTE_MODEL_FEATURES = False  # Optional locally cached tokenizer and embedding weights.
MODELS = [
    {
        "id": "google/gemma-4-12B",
        "revision": "023679ed352de9bb66cc873c9009ce3482585c08",
        "variant": "base",
    },
    {
        "id": "google/gemma-4-12B-it",
        "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "variant": "it",
    },
    {
        "id": "google/gemma-4-E4B",
        "revision": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "variant": "base",
    },
    {
        "id": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "variant": "it",
    },
]
for model in MODELS:
    model.update(
        kind="transformers",
        protocol=PROTOCOL,
        prompt_format="chat" if model["variant"] == "it" else "transcript",
        loader="multimodal_lm" if "12B" in model["id"] else "causal_lm",
        device="cuda:0",
        dtype="bfloat16",
        attention="sdpa",
        context_tokens=8192,
        top_p=0.95,
        runtime_verified=False,
        quantization="nf4",
        temperature=0.7,
        seed=42,
        max_tokens=2048,
    )


def inspect_experiment():
    _, scenarios = author_development(DATA / "scenarios/worlds.json")
    if SCENARIO_IDS is not None:
        scenarios = [s for s in scenarios if s["id"] in SCENARIO_IDS]
    scenarios = [s for s in scenarios if s["provenance"] in PROVENANCE]
    attempts = len(scenarios) * len(MODELS)
    return {
        "schema_version": 1,
        "status": "proposed_not_executed",
        "models": MODELS,
        "scenarios": len(scenarios),
        "core_attempts": attempts,
        "source_selections": SOURCES,
        "provenance_filters": sorted(PROVENANCE),
        "features": asdict(FEATURES),
        "variability": {
            "scenario_ids": ["F1-01", "F1-04", "F1-07", "F1-10"],
            "samples_per_prompt_per_model": 3,
            "additional_attempts": 48,
            "approved": False,
        },
        "navigation": {
            "builder_cases": 10,
            "probes_per_builder": 2,
            "reader_attempts": 80,
            "reader_model": "google/gemma-4-12B-it",
            "read_budget_tokens": 2000,
            "approved": False,
        },
        "external": [
            {
                "name": "IFEval",
                "cases": 541,
                "attempts": 2164,
                "approved": False,
                "protocol": "official strict/loose instruction/prompt scores",
            },
            {
                "name": "HumanEval+",
                "cases": 32,
                "attempts": 128,
                "approved": False,
                "subset": "HumanEval/0 through HumanEval/31",
                "protocol": "diagnostic subset; isolated execution required",
            },
        ],
        "cost_estimate": {
            "measured": False,
            "generated_tokens_per_attempt": [3000, 8000],
            "hypothetical_tokens_per_second": [20, 50],
            "decode_hours_at_3000_tokens": [
                attempts * 3000 / 50 / 3600,
                attempts * 3000 / 20 / 3600,
            ],
            "decode_hours_at_8000_tokens": [
                attempts * 8000 / 50 / 3600,
                attempts * 8000 / 20 / 3600,
            ],
            "excluded": [
                "prefill",
                "model loading",
                "reader probes",
                "repeated generations",
                "grading",
                "external benchmarks",
            ],
            "dollars": None,
            "grading": "subscription allowance; quota unmeasured",
        },
        "approval": "Candidate generation and full benchmarks require explicit approval",
    }


def prepare(*, download=False):
    acquisition = acquire_sources(DATA / "raw/research", selections=SOURCES) if download else None
    imported = compile_downloads(DATA / "raw/research", DATA / "processed/research")
    sources, scenarios = author_development(DATA / "scenarios/worlds.json")
    catalog = sources + imported["catalog"]
    manifest = compile_scenarios(scenarios, catalog, RELEASE)
    save_json(DATA / "scenarios/development.json", scenarios)
    save_json(DATA / "scenarios/sources.json", sources)
    write_review(scenarios, ROOT / "work/custom-eval-suite/review")
    save_json(ROOT / "work/custom-eval-suite/experiment.json", inspect_experiment())
    return {
        "manifest": manifest,
        "acquisition": acquisition,
        "inspection": imported["inspections"],
        "failures": imported["failures"],
    }


def generate(model: dict, *, approved=False, scenario_ids=None):
    if not approved:
        raise ValueError("Candidate generation requires explicit approval")
    if not model["runtime_verified"]:
        raise ValueError(
            "Verify checkpoint loading, prompt formatting and native tool protocol before running"
        )
    scenarios = [s for s in load_scenarios(RELEASE, scenario_ids) if s["provenance"] in PROVENANCE]
    return evaluate_checkpoint(
        scenarios,
        model,
        OUTPUT / "attempts",
        execute=True,
    )


def score_saved(*, judge=False, model_features=False):
    scenarios = {
        s["id"]: s for s in load_scenarios(RELEASE, SCENARIO_IDS) if s["provenance"] in PROVENANCE
    }
    cards, reviewed = [], []
    features = ProseFeatures(OUTPUT / "features", FEATURES)
    grader = CodexGrader(OUTPUT / "judgments", max_calls=3) if judge else None
    for result in saved_results(OUTPUT / "attempts"):
        path = Path(result["path"]) / "result.json"
        if result["scenario_id"] not in scenarios:
            continue
        scenario = scenarios[result["scenario_id"]]
        if result["visible_hash"] != scenario["visible_hash"]:
            raise ValueError("Saved result uses a different scenario package")
        card = mechanical_score(scenario, result)
        prose = [a["text"] for a in card["artifacts"] if a["status"] == "ok"]
        extracted = [
            features.extract(t, tokens=model_features, embeddings=model_features) for t in prose
        ]
        card["prose_profile"] = prose_profile(prose, extracted)
        paired = scenario["labels"].get("paired_reference")
        if paired is not None and len(prose) == 1:
            card["prose_profile"]["metrics"].update(paired_similarity(prose[0], paired))
        if grader:
            packet = grading_packet(scenario, result, card)
            judgment = grader.grade(packet)
            card = apply_judgment(card, packet, judgment)
            reviewed.append({"scenario_id": scenario["id"], "packet": packet, "judgment": judgment})
        save_json(path.parent / "scorecard.json", card)
        cards.append(card)
    if judge:
        save_json(OUTPUT / "judgment-review.json", reviewed)
    catalog = json.loads((RELEASE / "catalog.json").read_text())
    references = [r for r in catalog if r["id"] in REFERENCE_IDS]
    if len(references) != len(REFERENCE_IDS):
        raise ValueError("Unknown reference IDs")
    save_json(
        OUTPUT / "reports/prose-groups.json",
        compare_groups(
            cards,
            references,
            features,
            group_by=("model", "family", "provenance", "instruction_specificity", "genre"),
            sigma=MMD_BANDWIDTH,
            model_features=model_features,
        ),
    )
    return cards


def report():
    paths = [Path(r["path"]) / "scorecard.json" for r in saved_results(OUTPUT / "attempts")]
    cards = [json.loads(p.read_text()) for p in paths if p.exists()]
    return build_report(cards, OUTPUT / "reports")


def main():
    results = {}
    for stage in STAGES:
        if stage == "inspect":
            results[stage] = inspect_experiment()
        elif stage == "prepare":
            results[stage] = prepare(download=True)
        elif stage == "generate":
            # Evaluate one verified checkpoint at a time on the 3090.
            results[stage] = generate(
                MODELS[0], approved=APPROVED_CANDIDATE_RUN, scenario_ids=SCENARIO_IDS
            )
        elif stage == "score":
            results[stage] = score_saved(model_features=COMPUTE_MODEL_FEATURES)
        elif stage == "grade":
            results[stage] = score_saved(judge=True, model_features=COMPUTE_MODEL_FEATURES)
        elif stage == "report":
            results[stage] = report()
        else:
            raise ValueError(f"Unknown stage: {stage}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
