"""Collect the three tools' raw outputs under /work into guardrails.json on stdout.

Runs inside the guardrails image (`python3`, standard library only). Exits non-zero when an
output is missing or does not cover the whole suite, so guardrails.sh writes no file.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

work = Path("/work")
parser = argparse.ArgumentParser()
for flag in ("--endpoint", "--model", "--base-url", "--image", "--commit"):
    parser.add_argument(flag, required=True)
parser.add_argument("--seconds", action="append", default=[])
args = parser.parse_args()
seconds = dict(s.split("=") for s in args.seconds)
versions = json.loads(Path("/opt/versions.json").read_text())


def fail(message: str):
    sys.exit(f"summarize: {message}")


def one(pattern: str) -> Path:
    found = sorted(work.glob(pattern))
    if len(found) != 1:
        fail(f"expected one {pattern} under {work}, found {len(found)}")
    return found[0]


# IFEval: 541 prompts, each with one or more verifiable instructions.
ifeval_run = json.loads(one("ifeval/*/results_*.json").read_text())
ifeval_metrics = ifeval_run["results"]["ifeval"]
ifeval_n = ifeval_run["n-samples"]["ifeval"]["effective"]
if ifeval_n != 541:
    fail(f"IFEval scored {ifeval_n} of 541 prompts")
ifeval_metrics = {
    m: round(100 * ifeval_metrics[f"{m},none"], 2)
    for m in ("prompt_level_strict_acc", "inst_level_strict_acc",
              "prompt_level_loose_acc", "inst_level_loose_acc")
}

# HumanEval+: 164 problems, one greedy sample each; a pass needs base and plus tests.
humaneval_eval = json.loads(one("humaneval/exec/samples_eval_results.json").read_text())["eval"]
if len(humaneval_eval) != 164 or any(len(r) != 1 for r in humaneval_eval.values()):
    fail(f"HumanEval+ results cover {len(humaneval_eval)} of 164 problems")
base_pass = sum(r[0]["base_status"] == "pass" for r in humaneval_eval.values())
plus_pass = sum(r[0]["base_status"] == r[0]["plus_status"] == "pass"
                for r in humaneval_eval.values())

# BFCL multi-turn: four categories of 200; the overall score is their unweighted mean, as on
# the leaderboard's multi-turn column.
bfcl = json.loads(one("bfcl/scores.json").read_text())
bfcl_categories = ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param",
                   "multi_turn_long_context")
for c in bfcl_categories:
    if bfcl["categories"].get(c, {}).get("total_count") != 200:
        fail(f"BFCL {c} not scored on all 200 entries: {bfcl['categories'].get(c)}")
bfcl_accuracy = {c: round(100 * bfcl["categories"][c]["accuracy"], 2) for c in bfcl_categories}

result = {
    "endpoint": args.endpoint,
    "model": args.model,
    "base_url": args.base_url,
    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "unit": "percent (0-100); compare drops in points against the base checkpoint",
    "ifeval": round(ifeval_metrics["prompt_level_strict_acc"], 2),
    "humaneval_plus": round(100 * plus_pass / 164, 2),
    "bfcl": round(sum(bfcl_accuracy.values()) / len(bfcl_accuracy), 2),
    "detail": {
        "ifeval": {
            "tool": "lm-eval", "version": versions["lm-eval"],
            "scoring": "task ifeval (v4.0), 0-shot, 541 prompts, chat template applied by the "
                       "endpoint (/chat/completions), greedy (temperature 0), max_gen_toks 1280; "
                       "score = prompt_level_strict_acc",
            "dataset": f"google/IFEval@{versions['ifeval_dataset_revision']}",
            "metrics": ifeval_metrics,
            "seconds": int(seconds["ifeval"]),
        },
        "humaneval_plus": {
            "tool": "evalplus", "version": versions["evalplus"],
            "scoring": "HumanEval+ instruct prompt via /chat/completions, greedy (temperature 0, "
                       "n=1), max_new_tokens 768, code sanitized by evalplus; score = pass@1 on "
                       "base + plus tests; executed with no network in the container",
            "dataset": f"HumanEvalPlus {versions['humaneval_plus_dataset']}",
            "humaneval_base": round(100 * base_pass / 164, 2),
            "passed": plus_pass, "total": 164,
            "seconds": int(seconds["humaneval_plus"]),
        },
        "bfcl": {
            "tool": "bfcl-eval", "version": versions["bfcl-eval"],
            "scoring": "BFCL v4 multi_turn (base, miss_func, miss_param, long_context; 200 each, "
                       "all 800 entries), native function calling (FC) via /chat/completions "
                       "tools, temperature 0.001; score = unweighted mean of the four category "
                       "accuracies",
            "categories": bfcl_accuracy,
            "model_errors": bfcl["model_errors"],
            "tokens": bfcl["tokens"],
            "seconds": int(seconds["bfcl"]),
        },
    },
    "versions": versions,
    "image": args.image,
    "script_commit": args.commit,
}
print(json.dumps(result, indent=2))
