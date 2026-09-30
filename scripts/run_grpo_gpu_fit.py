"""Inspect-first, one-attempt production Gemma controlled GPU memory gate."""

import argparse
import json
from pathlib import Path

from writing_agent.grpo_gpu_fit import execute_fit, inspect_fit, preflight_fit, prepare_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--phase", choices=("inspect", "prepare", "preflight", "fit"), default="inspect"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and args.phase != "fit":
        parser.error("--execute requires --phase fit")
    if args.phase == "prepare":
        result = prepare_fit(args.run_dir)
    elif args.phase == "preflight":
        result = preflight_fit(args.run_dir)
    elif args.phase == "fit" and args.execute:
        result = execute_fit(args.run_dir)
    else:
        result = {"execute": False, "plan": inspect_fit()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
