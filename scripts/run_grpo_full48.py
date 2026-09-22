"""Inspect/prepare/preflight or explicitly supervise the intact two-pass full48 run."""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from writing_agent.grpo_full48 import load_full48_release
from writing_agent.grpo_full48_runner import (
    SETTINGS,
    coverage,
    execute_training,
    frozen_plan,
    preflight,
    prepare,
)
from writing_agent.grpo_full48_supervisor import supervise, verify_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--phase",
        choices=("inspect", "prepare", "preflight", "coverage", "train", "resume"),
        default="inspect",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--_lease-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase in {"inspect", "prepare", "preflight", "coverage"} and args.execute:
        parser.error("--execute applies only to train/resume")
    if args._lease_fd is not None:
        if not args.execute or args.phase not in {"train", "resume"}:
            parser.error("Lease descriptor requires an execution worker")
        verify_lease(args.run_dir, args._lease_fd)

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt("Full48 worker interrupted")

        signal.signal(signal.SIGTERM, interrupted)
        result = execute_training(
            args.release, args.run_dir, lease_fd=args._lease_fd, resume=args.phase == "resume"
        )
    elif args.phase == "prepare":
        result = prepare(args.release, args.run_dir)
    elif args.phase == "preflight":
        # Read-only; infer which admission to inspect, never launch or load weights.
        data = load_full48_release(args.release)
        report = coverage(args.run_dir / "trainer", data["tasks"], SETTINGS)
        result = preflight(
            args.release, args.run_dir, resume=bool(report["completed_optimizer_boundaries"])
        )
    elif args.phase == "coverage":
        data = load_full48_release(args.release)
        result = coverage(args.run_dir / "trainer", data["tasks"], SETTINGS)
    elif args.phase == "inspect" or not args.execute:
        result = {
            "execute": False,
            "phase": args.phase,
            "plan": frozen_plan(args.release, args.run_dir),
        }
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(args.release.resolve()),
            str(args.run_dir.resolve()),
            "--phase",
            args.phase,
            "--execute",
        ]
        result = supervise(
            args.run_dir,
            command,
            admit=lambda: preflight(args.release, args.run_dir, resume=args.phase == "resume"),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
