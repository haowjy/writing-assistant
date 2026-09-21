"""Python-driven Creative Writing v3: local generation, paid rubric grading, reports."""

import json
import re
from statistics import mean

from scripts.pilot_e2b import MODEL, ROOT
from writing_agent.anthropic_grading import from_env
from writing_agent.catalog import fingerprint, save_json
from writing_agent.inference import (
    HARNESS_CONTEXT_TOKENS,
    TransformersBackend,
    load_checkpoint,
)

UPSTREAM = ROOT / "data/raw/research/creative-writing-v3"
OUTPUT = ROOT / "runs/creative-writing-v3-32-e2b-it-2026-09-14"
PAID_BUDGET_USD = 2
CONFIG = {
    **MODEL,
    "purpose": "Creative Writing v3 rubric benchmark",
    "max_tokens": 12000,
    "context_tokens": HARNESS_CONTEXT_TOKENS,
    "temperature": 0.7,
    "top_p": 1.0,
    "min_p": 0.1,
    "top_k": 0,
    "enable_thinking": True,
}


def tasks():
    prompts = json.loads((UPSTREAM / "data/creative_writing_prompts_v3.json").read_text())
    return [
        {
            "id": f"{key}-{iteration + 1}",
            "prompt_id": key,
            "iteration": iteration + 1,
            "base_prompt": item["writing_prompt"],
            "prompt": item["writing_prompt"].replace("<SEED>", item["seed_modifiers"][iteration]),
            "category": item["category"],
        }
        for key, item in prompts.items()
        for iteration in range(3)
    ]


def prepare():
    selected = [task for task in tasks() if task["iteration"] == 1]
    assert len(selected) == 32
    manifest = {
        "tasks": selected,
        "model": CONFIG,
        "judge": "claude-sonnet-4-6",
        "paid_budget_usd": PAID_BUDGET_USD,
        "scope": "32-output development subset; first variant of each prompt",
        "selection_hash": fingerprint(selected),
        "elo": False,
        "upstream": json.loads((UPSTREAM / "revision.json").read_text()),
        "deviations": [
            "NF4 local inference",
            "No regeneration retries",
            "Only named rubric criteria are parsed; omitted criteria remain unscored",
        ],
    }
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Experiment changed; use a new output directory")
    save_json(path, manifest)
    return manifest


def generate(*, subset=None):
    manifest = prepare()
    with load_checkpoint(CONFIG) as (model, tokenizer, record):
        for i, task in enumerate(manifest["tasks"]):
            if subset is not None and task["id"] not in subset:
                continue
            directory = OUTPUT / "items" / task["id"]
            path = directory / "generation.json"
            if path.exists():
                continue
            directory.mkdir(parents=True, exist_ok=True)
            config = {**record, "seed": CONFIG["seed"] + i}
            save_json(directory / "started.json", {"task": task, "model": config})
            events = []
            try:
                answer = TransformersBackend(model, tokenizer, config).complete(
                    [{"role": "user", "content": task["prompt"]}], [], emit=events.append
                )
                result = {"status": "completed", "message": answer.message, "usage": answer.usage}
                if len(answer.message.get("content", "").strip()) < 500:
                    result.update(
                        status="failed", error="Output below upstream 500-character minimum"
                    )
            except Exception as exc:
                result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            result.update(task=task, model=config, trace=events)
            save_json(path, result)
            (directory / "response.md").write_text(result.get("message", {}).get("content", ""))
            (directory / "thinking.txt").write_text(result.get("message", {}).get("thinking", ""))
            print(task["id"], result["status"], flush=True)
    return report()


def parse_scores(text, criteria):
    section = text.split("[Scores]")[-1]
    scores = {}
    for name in criteria:
        pattern = rf"^\s*(?:\*\*)?{re.escape(name)}(?:\*\*)?\s*:\s*\[?(?:Score\s+)?(\d+(?:\.\d+)?)"
        match = re.search(pattern, section, flags=re.M | re.I)
        if not match:
            continue
        if not 0 <= float(match[1]) <= 20:
            raise ValueError(f"Missing or invalid rubric score: {name}")
        scores[name] = float(match[1])
    if not scores:
        raise ValueError("No valid rubric scores")
    return scores


def grade(*, subset=None):
    manifest = prepare()
    grader = from_env(OUTPUT / "judgments", ROOT / ".env", budget_usd=manifest["paid_budget_usd"])
    template = (UPSTREAM / "data/creative_writing_judging_prompt.txt").read_text()
    criteria = (UPSTREAM / "data/creative_writing_criteria.txt").read_text().splitlines()
    negatives = (UPSTREAM / "data/negative_criteria.txt").read_text().splitlines()
    for task in manifest["tasks"]:
        if subset is not None and task["id"] not in subset:
            continue
        directory = OUTPUT / "items" / task["id"]
        path = directory / "generation.json"
        if not path.exists() or (directory / "judgment.json").exists():
            continue
        generated = json.loads(path.read_text())
        if generated["status"] != "completed":
            continue
        prose = generated["message"].get("content", "")
        if not prose.strip():
            continue
        # Upstream rubric uses base prompt, with its SEED marker, not the substituted variant.
        prompt = template.format(
            writing_prompt=task["base_prompt"],
            test_model_response=prose,
            creative_writing_criteria="\n".join("- " + c for c in criteria),
            lower_is_better_criteria=", ".join(negatives),
        )
        response = grader.grade(prompt)
        judgment = {
            "response_hash": fingerprint(response),
            "cost_micro_usd": response["cost_micro_usd"],
        }
        try:
            if response["response"]["stop_reason"] != "end_turn":
                raise ValueError("Judge response did not finish")
            scores = parse_scores(response["text"], criteria)
            judgment.update(
                status="completed",
                scores=scores,
                unscored_criteria=[name for name in criteria if name not in scores],
                score_0_100=5 * mean(20 - v if k in negatives else v for k, v in scores.items()),
            )
        except ValueError as exc:
            judgment.update(status="failed", error=str(exc))
        save_json(directory / "judgment.json", judgment)
        print(task["id"], "judged", judgment["status"], flush=True)
    return report()


def report():
    rows = []
    manifest = prepare()
    expected = len(manifest["tasks"])
    for task in manifest["tasks"]:
        directory = OUTPUT / "items" / task["id"]
        g = (
            json.loads((directory / "generation.json").read_text())
            if (directory / "generation.json").exists()
            else {}
        )
        j = (
            json.loads((directory / "judgment.json").read_text())
            if (directory / "judgment.json").exists()
            else {}
        )
        rows.append(
            {
                "id": task["id"],
                "generation": g.get("status", "pending"),
                "judgment": j.get("status", "pending"),
                "score": j.get("score_0_100"),
            }
        )
    scores = [r["score"] for r in rows if r["score"] is not None]
    ledger_path = OUTPUT / "judgments/ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    result = {
        "expected": expected,
        "scope": manifest["scope"],
        "selection_hash": manifest["selection_hash"],
        "generated": sum(r["generation"] == "completed" for r in rows),
        "graded": len(scores),
        "score_0_100": None,  # Reserved for the full upstream 96-output benchmark.
        "selected_score_0_100": mean(scores) if len(scores) == expected else None,
        "observed_subset_mean": mean(scores) if scores else None,
        "charged_or_reserved_usd": sum(r["charged_or_reserved_micro_usd"] for r in ledger.values())
        / 1e6,
        "rows": rows,
    }
    save_json(OUTPUT / "report.json", result)
    lines = [
        "# Creative Writing v3 — E2B-IT",
        "",
        f"Generated {result['generated']}/{expected}; graded {len(scores)}/{expected}. "
        f"Charged/reserved: ${result['charged_or_reserved_usd']:.4f}.",
        "",
        f"32-output subset score: {result['selected_score_0_100']}. "
        "This is not the full 96-output benchmark. No Elo run.",
        "",
        "| Item | Generation | Grading | Score /100 | Artifacts |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['id']} | {r['generation']} | {r['judgment']} | {r['score']} | "
        + (
            f"[Text](items/{r['id']}/response.md) · "
            f"[Thinking](items/{r['id']}/thinking.txt) · "
            f"[Record](items/{r['id']}/generation.json) |"
            if r["generation"] != "pending"
            else "Pending |"
        )
        for r in rows
    ]
    (OUTPUT / "review.md").write_text("\n".join(lines) + "\n")
    return result
