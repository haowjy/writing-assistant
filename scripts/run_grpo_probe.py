"""Inspect or execute one frozen, local-only Gemma GRPO probe phase."""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from writing_agent.grpo_probe import (
    DEFAULT_TIMEOUT_SECONDS,
    PHASES,
    admit_phase,
    execute_phase,
    frozen_plan,
    inspect_probe,
    supervise,
    worker_record,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--phase", choices=PHASES, default="inspect")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _call(args):
    return execute_phase(
        args.run_dir,
        args.phase,
        resume=args.resume,
        adapter=args.adapter,
        stop_after_step=args.stop_after_step,
    )


def main():
    args = _parser().parse_args()
    if args.phase == "inspect":
        if args.execute:
            _parser().error("inspect is always read-only; omit --execute")
        print(json.dumps(inspect_probe(args.run_dir), indent=2))
        return
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "execute": False,
                    "phase": args.phase,
                    "plan": frozen_plan(args.run_dir),
                },
                indent=2,
            )
        )
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args._worker:
        admit_phase(
            args.run_dir,
            args.phase,
            resume=args.resume,
            adapter=args.adapter,
            stop_after_step=args.stop_after_step,
        )

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt("Supervisor requested graceful termination")

        signal.signal(signal.SIGTERM, interrupted)
        result = worker_record(args.run_dir, args.phase, lambda: _call(args))
        print(json.dumps(result, indent=2))
        return
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.run_dir),
        "--phase",
        args.phase,
        "--execute",
        "--_worker",
    ]
    if args.resume is not None:
        command.extend(("--resume", str(args.resume)))
    if args.adapter is not None:
        command.extend(("--adapter", str(args.adapter)))
    if args.stop_after_step is not None:
        command.extend(("--stop-after-step", str(args.stop_after_step)))
    result = supervise(
        args.run_dir,
        args.phase,
        command,
        admission=admit_phase(
            args.run_dir,
            args.phase,
            resume=args.resume,
            adapter=args.adapter,
            stop_after_step=args.stop_after_step,
        ),
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
