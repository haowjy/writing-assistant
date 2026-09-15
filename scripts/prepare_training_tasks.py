"""Prepare the first 100 task-authoring requests. No API or model calls."""

import json
from pathlib import Path

from scripts.train_sft import evaluation_source_groups
from writing_agent.catalog import fingerprint, save_json
from writing_agent.task_generation import prepare_task_requests
from writing_agent.workspace import TOOL_SCHEMAS

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/processed/sft-starter-v1/catalog.json"
DESTINATION = ROOT / "data/processed/training-tasks-v1"
MANIFEST = ROOT / "data/training/task-generation-v1.json"
INSTRUCTIONS = ROOT / "work/sft/task-generator-instructions.md"
VARIATIONS = ROOT / "data/training/variation-catalog-v1.json"
SOURCE_IDS = [
    "gutenberg-289-opening",
    "tmas-train-example_104-story",
    "tmas-train-example_100-story",
    "tmas-train-example_033-story",
    "tmas-train-example_046-story",
]


def prepare():
    catalog = json.loads(CATALOG.read_text())
    selection = json.loads((ROOT / "data/training/starter-selection.json").read_text())
    frozen = {s["id"]: s["sha256"] for s in selection["source_inventory"]}
    for source in catalog:
        if source["id"] in SOURCE_IDS and frozen.get(source["id"]) != source["sha256"]:
            raise ValueError("Source differs from the frozen starter inventory")
    batch = prepare_task_requests(
        catalog,
        SOURCE_IDS,
        excluded_source_groups=evaluation_source_groups(),
        variation_catalog=json.loads(VARIATIONS.read_text()),
        count=100,
    )
    instruction_text = INSTRUCTIONS.read_text()
    context = {"instructions": instruction_text, "tool_schemas": TOOL_SCHEMAS}
    save_json(DESTINATION / "requests.json", batch)
    save_json(DESTINATION / "generation-context.json", context)
    manifest = {key: value for key, value in batch.items() if key != "requests"}
    manifest.update(
        count=len(batch["requests"]),
        source_ids=SOURCE_IDS,
        requests_hash=fingerprint(batch["requests"]),
        generator_instructions_hash=fingerprint(instruction_text),
        generation_context_hash=fingerprint(context),
        generator={"transport": "openrouter", "provider": "reka", "model": "z-ai/glm-5.3"},
        training_use_permission="settled_by_user",
        generation_executed=False,
        limitations=[
            "Five source works in two conservative lineage groups, not 100 independent works.",
            "No generated task, KB, branch contract, or target answer is accepted yet.",
            "Genre blends need grounding; assignments are not verified stories.",
        ],
    )
    save_json(MANIFEST, manifest)
    lines = [
        "# First training-task batch",
        "",
        "100 requests; no generated or accepted tasks. Five works in two lineage groups.",
        "",
        "| Request | Source | Stages | Specificity | Transformation | Genres | Trope | Situation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for request in batch["requests"]:
        a = request["assignment"]
        lines.append(
            f"| {request['id']} | {request['packet']['source']['id']} | "
            f"{' → '.join(a['stage_families'])} | {a['instruction_specificity']} | "
            f"{a['transformation']} | {' + '.join(a['genre_blend']) or a['genre']} | "
            f"{a['trope']} | {a['situation']} |"
        )
    (DESTINATION / "review.md").write_text("\n".join(lines) + "\n")
    return manifest


if __name__ == "__main__":
    result = prepare()
    print(f"Prepared {result['count']} requests; generated tasks: {result['generated_tasks']}.")
    print(f"Review: {DESTINATION / 'review.md'}")
