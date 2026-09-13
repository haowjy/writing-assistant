"""Inspect with python -m scripts.pilot_e2b; call run(execute=True) for this pilot."""

import json
import os
import subprocess
from pathlib import Path

from writing_agent.catalog import save_json
from writing_agent.inference import PROTOCOL, evaluate_checkpoint
from writing_agent.prose import ProseFeatures, prose_profile
from writing_agent.scoring import build_report, mechanical_score
from writing_agent.suite import load_scenarios

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/pilot-e2b-it-2026-09-12"
SCENARIO_IDS = ["F1-01", "F2-06", "F3-03", "F4-08", "F5-05"]
MODEL = {
    "id": "google/gemma-4-E2B-it",
    "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "variant": "it",
    "kind": "transformers",
    "protocol": PROTOCOL,
    "prompt_format": "chat",
    "loader": "causal_lm",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "attention": "sdpa",
    "context_tokens": 8192,
    "quantization": "nf4",
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 42,
    "max_tokens": 2048,
    "purpose": "user-authorized five-case development pilot",
}


def _fenced(text):
    # Generated content may contain Markdown fences; preserve it as literal text.
    fence = "```"
    while fence in text:
        fence += "`"
    return fence + "\n" + text + "\n" + fence + "\n"


def run(*, execute=False, allow_download=False):
    scenarios = load_scenarios(ROOT / "data/processed/custom-eval", SCENARIO_IDS)
    config = {
        **MODEL,
        "code_checkpoint": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    if not execute:
        return evaluate_checkpoint(scenarios, config, OUTPUT / "attempts")
    save_json(OUTPUT / "selection.json", {"model": config, "scenarios": scenarios})
    results = evaluate_checkpoint(
        scenarios,
        config,
        OUTPUT / "attempts",
        execute=True,
        allow_download=allow_download,
    )
    return review_results(scenarios, results, config)


def review_results(scenarios, results, config):
    """Build review artifacts from saved results without model execution."""
    features = ProseFeatures(OUTPUT / "features")
    cards = []
    index = [
        "# Gemma E2B-IT five-case pilot",
        "",
        f"Code checkpoint: `{config['code_checkpoint']}`.",
        "",
        "Mechanical scores and basic prose profiles only; semantic judgments are pending.",
        "No reference/model-feature comparisons or Astra calls were run.",
        "",
        "[Exact inputs and private draft labels](selection.json)",
        "",
        "Execution completion means the loop ended, not that the task succeeded.",
        "",
        "| Case | Genre | Request | Execution | Task completion | Failed checks | Review |",
        "|---|---|---|---|---|---|---|",
    ]
    for scenario, result in zip(scenarios, results, strict=True):
        card = mechanical_score(scenario, result)
        prose = [a["text"] for a in card["artifacts"] if a["status"] == "ok"]
        extracted = [features.extract(t, tokens=False, embeddings=False) for t in prose]
        card["prose_profile"] = prose_profile(prose, extracted)
        path = Path(result["path"])
        save_json(path / "scorecard.json", card)
        cards.append(card)
        completion = card["scores"]["Q3"]
        completion_label = (
            str(completion["value"]) if completion["status"] == "ok" else completion["status"]
        )
        failed = ", ".join(c["id"] for c in card["checks"] if c.get("passed") is False) or "none"
        review = [
            f"# {scenario['id']} — {scenario['genre']}",
            "",
            f"Execution: **{result['status']}**. "
            f"Instruction specificity: {scenario['instruction_specificity']}.",
            "",
            f"Task completion: {completion_label}. Failed checks: {failed}.",
            "",
            "[Result](result.json) · [Raw trace](trace.jsonl) · [Scorecard](scorecard.json)",
            "",
            "## Tool context",
            "",
            "The trace records these available tools. The local backend appends their schemas",
            "and tool-call instructions before applying the model chat template. The conversation",
            "below is the harness history before that rendering, not the full model prompt.",
            "",
            _fenced(
                json.dumps(
                    next((e["tools"] for e in result.get("trace", []) if e["type"] == "input"), []),
                    indent=2,
                )
            ),
            "## Conversation",
            "",
        ]
        for message in result.get("messages", []):
            review += [
                f"### {message['role']}",
                "",
                _fenced(
                    json.dumps(message, indent=2, ensure_ascii=False)
                    if message.get("tool_calls")
                    else str(message.get("content", ""))
                ),
            ]
        if result.get("error"):
            review += ["## Execution error", "", _fenced(result["error"])]
        review += ["## Final files", ""]
        for name, text in result["after"].items():
            review += [f"### [{name}](workspace/{name})", "", _fenced(text)]
        review += ["## Extracted prose", "", _fenced(json.dumps(card["artifacts"], indent=2))]
        (path / "review.md").write_text("\n".join(review))
        relative = Path(os.path.relpath(path / "review.md", OUTPUT)).as_posix()
        index.append(
            f"| {scenario['id']} | {scenario['genre']} | "
            f"{scenario['instruction_specificity']} | {result['status']} | "
            f"{completion_label} | {failed} | "
            f"[Read case]({relative}) |"
        )
    report = build_report(cards, OUTPUT / "reports")
    index += ["", "[Aggregate report](reports/report.md)", ""]
    (OUTPUT / "review.md").write_text("\n".join(index))
    return {
        "review": str(OUTPUT / "review.md"),
        "attempts": report["attempts"],
        "statuses": {r["scenario_id"]: r["status"] for r in results},
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
