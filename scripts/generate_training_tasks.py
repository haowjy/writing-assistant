"""Inspect or explicitly generate the approved first task collection from Python."""

import json
from pathlib import Path

from scripts.prepare_training_tasks import CATALOG, DESTINATION, INSTRUCTIONS, MANIFEST
from writing_agent.catalog import fingerprint
from writing_agent.openrouter import from_env
from writing_agent.task_authoring import author_tasks

ROOT = Path(__file__).resolve().parents[1]
BUDGET_USD = 10
OUTPUT = DESTINATION / "generated"
CALLS = DESTINATION / "paid-calls"


def generate(*, execute=False):
    manifest = json.loads(MANIFEST.read_text())
    batch = json.loads((DESTINATION / "requests.json").read_text())
    context = json.loads((DESTINATION / "generation-context.json").read_text())
    catalog = json.loads(CATALOG.read_text())
    expected = {
        "requests_hash": fingerprint(batch["requests"]),
        "catalog_hash": fingerprint(catalog),
        "generation_context_hash": fingerprint(context),
        "generator_instructions_hash": fingerprint(INSTRUCTIONS.read_text()),
    }
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError("Prepared inputs changed; prepare and inspect the batch again")
    if context["instructions"] != INSTRUCTIONS.read_text():
        raise ValueError("Generator instructions differ from prepared context")
    if not execute:
        return {
            "status": "planned",
            "tasks": len(batch["requests"]),
            "maximum_calls": 2 * len(batch["requests"]),
            "budget_usd": BUDGET_USD,
            "output": str(OUTPUT),
            "sft_trajectories": 0,
        }
    client = from_env(CALLS, ROOT / ".env", budget_usd=BUDGET_USD)
    return author_tasks(
        batch["requests"],
        catalog,
        context["instructions"],
        context["tool_schemas"],
        OUTPUT,
        client=client,
    )


if __name__ == "__main__":
    result = generate()  # Explicitly call generate(execute=True) to spend the approved budget.
    print(f"Status: {result['status']}; tasks: {result['tasks']}; cap: ${BUDGET_USD}.")
    print(f"Artifacts: {OUTPUT}")
