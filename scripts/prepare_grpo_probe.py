"""Build/inspect/validate the frozen CPU-only engineering probe (no model loading)."""

import argparse
import json
from pathlib import Path

from writing_agent.grpo_probe_data import (
    build_probe_release,
    inspect_probe_release,
    validate_probe_fixtures,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "inspect", "validate"])
    parser.add_argument("release", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build_probe_release(args.release)
    if args.command == "validate":
        if args.evidence is None:
            parser.error("validate requires --evidence pointing to an empty directory")
        result = validate_probe_fixtures(args.release, args.evidence)
        print(json.dumps({"tasks": len(result["tasks"]), "evidence": str(args.evidence)}))
    else:
        print(json.dumps(inspect_probe_release(args.release), indent=2))


if __name__ == "__main__":
    main()
