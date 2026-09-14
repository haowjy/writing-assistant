"""Serial execution: custom50, the 32-output prose baseline, then automatic checks.

Call run() from Python after starting the authorized custom50 run. This script
never starts automatically when imported and never loads or prints credentials.
"""

import json
import time

from scripts import creative_writing_v3 as creative
from scripts import external_checks
from writing_agent.catalog import save_json

CUSTOM = creative.ROOT / "runs/custom50-e2b-it-2026-09-14"


def _run_creative():
    state = creative.OUTPUT / "pipeline.json"
    save_json(state, {"stage": "waiting_for_custom50"})
    while not (CUSTOM / "review.md").exists():
        time.sleep(10)
    attempts = list(CUSTOM.glob("attempts/*/*/result.json"))
    if len(attempts) != 50:
        raise ValueError("Custom run has not produced all 50 attempt records")
    save_json(state, {"stage": "creative_writing_integration"})
    creative.generate(subset={"1-1"})
    first = json.loads((creative.OUTPUT / "items/1-1/generation.json").read_text())
    if first["status"] != "completed":
        raise RuntimeError(
            "First external generation failed; inspect before starting remaining cases"
        )
    creative.grade(subset={"1-1"})
    first_judge = json.loads((creative.OUTPUT / "items/1-1/judgment.json").read_text())
    if first_judge["status"] != "completed":
        raise RuntimeError("First paid judgment failed; inspect before starting remaining cases")
    save_json(state, {"stage": "creative_writing_generation"})
    creative.generate()
    save_json(state, {"stage": "creative_writing_grading"})
    creative.grade()
    report = creative.report()
    save_json(state, {"stage": "finished", "report": report})
    return report


def run():
    """Keep automatic checks runnable even if paid grading stops at its budget."""
    try:
        result = _run_creative()
    except Exception as exc:
        result = {
            "stage": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "report": creative.report(),
        }
        save_json(creative.OUTPUT / "pipeline.json", result)
    external_checks.run()
    return {"creative_writing": result, "automatic_checks": "finished"}
