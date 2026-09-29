"""Inspect intact full48 or validate mechanical fixtures offline; never load weights."""

import argparse
import json
from pathlib import Path

from writing_agent.grpo_full48 import load_full48_release
from writing_agent.grpo_full48_fixtures import measure_full48, validate_full48


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "validate", "measure"])
    parser.add_argument("release", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    if args.command == "measure":
        if args.evidence is None or args.tokenizer is None:
            parser.error(
                "measure requires --evidence from validate and --tokenizer local directory"
            )
        report = measure_full48(args.release, args.evidence, args.tokenizer)
        print(json.dumps({"tasks": len(report["tasks"]), "warning": report["warning"]}))
    elif args.command == "validate":
        if args.evidence is None:
            parser.error("validate requires --evidence (an absent directory)")
        report = validate_full48(args.release, args.evidence)
        print(
            json.dumps(
                {
                    "cases": len(report["cases"]),
                    "failures": report["failures"],
                    "evidence": str(args.evidence),
                }
            )
        )
    else:
        release = load_full48_release(args.release)
        print(
            json.dumps(
                {
                    "tasks": [t["id"] for t in release["tasks"]],
                    "reward_spec": release["reward_spec"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
