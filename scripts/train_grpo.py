"""Editable, inspect-first Gemma GRPO connection. No tasks or graders selected by default."""

import argparse
import json
from pathlib import Path

from writing_agent.catalog import fingerprint
from writing_agent.grpo import GRPOSettings, inspect_grpo, train_grpo
from writing_agent.reward import Reward, mechanics_score
from writing_agent.scoring import mechanical_score
from writing_agent.suite import load_scenarios

# Select a compiled, accepted train-role release explicitly; no task generation here.
RELEASE = None
TASK_IDS = []
EXCLUDED_SOURCE_GROUPS = set()  # Explicit additional held-out source/work/author/series IDs.
OUTPUT = Path("runs/grpo-probe")
SETTINGS = GRPOSettings(revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7")
REWARD_SPEC = {
    "id": "mechanical-only-smoke-v1",
    "mode": "mechanical-only-smoke",
    "config": {},
}


def mechanical_smoke_reward(task, result):
    """Engineering-only mechanical scalar; supplies no semantic judgments."""
    if result["status"] != "completed":
        return Reward("ok", 0.0, components={"execution": 0.0}, reason="Candidate incomplete")
    checks = mechanical_score(task, result)["checks"]
    if any(c["method"] not in {"deterministic", "reference_match"} for c in checks):
        return Reward(
            "unavailable", reason="Semantic checks need an explicit mixed reward provider"
        )
    value = mechanics_score([{"id": c["id"], "passed": bool(c["passed"])} for c in checks])
    return Reward("ok", value, components={"mechanics": value})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if RELEASE is None or not TASK_IDS:
        if args.execute:
            parser.error("Select RELEASE, TASK_IDS and frozen reward settings before execution")
        print(
            json.dumps(
                {
                    "status": "inspect",
                    "ready": False,
                    "reason": "No training tasks selected",
                    "settings": SETTINGS.__dict__,
                },
                indent=2,
            )
        )
        return
    release = Path(RELEASE)
    catalog = json.loads((release / "catalog.json").read_text())
    manifest = json.loads((release / "manifest.json").read_text())
    if fingerprint(catalog) != manifest["catalog_hash"]:
        raise ValueError("Release catalog hash mismatch")
    admission = {
        "mode": "production",
        "catalog": catalog,
        "catalog_hash": manifest["catalog_hash"],
        "release_manifest": manifest,
        "excluded_source_groups": sorted(EXCLUDED_SOURCE_GROUPS),
    }
    tasks = load_scenarios(release, TASK_IDS)
    call = train_grpo if args.execute else inspect_grpo
    kwargs = dict(settings=SETTINGS, reward_spec=REWARD_SPEC, admission=admission)
    if args.execute:
        kwargs.update(
            execute=True,
            reward_callback=mechanical_smoke_reward,
            resume_from_checkpoint=args.resume,
        )
    print(json.dumps(call(tasks, OUTPUT, **kwargs), indent=2))


if __name__ == "__main__":
    main()
