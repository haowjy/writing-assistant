"""Prepare, preflight, or explicitly launch the checkpoint-31 fork.

Launch is always supervised by the inherited writer lease.  Inspection phases
never import the model stack; an execution worker is the only path that admits
headless NVML ownership and imports the trainer.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from writing_agent.grpo import train_grpo
from writing_agent.grpo_checkpoint31_fork import (
    fork_preflight,
    fork_resume_options,
    fork_trainer_options,
    prepare_fork,
)
from writing_agent.grpo_full48 import load_full48_release, mechanical_full48_reward
from writing_agent.grpo_full48_runner import SETTINGS
from writing_agent.grpo_full48_supervisor import supervise, verify_lease
from writing_agent.grpo_gpu import HEADLESS_POLICY, admit_gpu, configure_cuda_allocator
from writing_agent.grpo_runtime import STREAMING, verify_runtime


def _ownership_path(output: Path) -> Path:
    """Return a durable per-admission record path (safe across resume)."""
    return output / "ownership" / uuid4().hex


def _execute(args, *, resume, lease_fd):
    verify_lease(args.output, lease_fd)
    fork_preflight(
        args.output,
        release=args.release,
        source_checkpoint=args.source_checkpoint,
        source_group=args.source_group,
        resume=resume,
    )
    verify_runtime(STREAMING)
    configure_cuda_allocator()
    # Every admission gets a durable unique record; resuming must never reuse
    # the first launch's ownership directory.
    admit_gpu(_ownership_path(args.output), policy=HEADLESS_POLICY)
    data = load_full48_release(args.release)
    options = (
        fork_resume_options(args.output)
        if resume
        else fork_trainer_options(args.output, wandb_run_id=args.wandb_run_id)
    )
    if options.get("resume_checkpoint_identity") is None:
        options.pop("resume_checkpoint_identity", None)
    train_grpo(
        data["tasks"],
        args.output / "trainer",
        settings=SETTINGS,
        reward_spec=data["reward_spec"],
        admission=data["admission"],
        reward_callback=mechanical_full48_reward,
        execute=True,
        implementation=STREAMING,
        **options,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("source_group", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wandb-run-id")
    parser.add_argument(
        "--phase", choices=("prepare", "preflight", "launch", "resume"), default="preflight"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--_lease-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.phase == "prepare":
        if args.execute or args._lease_fd is not None:
            parser.error("prepare is read-only and does not accept --execute")
        result = prepare_fork(
            args.source_checkpoint,
            args.source_group,
            args.output,
            wandb_run_id=args.wandb_run_id,
            release=args.release,
        )
    elif args.phase == "preflight":
        if args.execute or args._lease_fd is not None:
            parser.error("preflight is read-only and does not accept --execute")
        result = fork_preflight(
            args.output,
            release=args.release,
            source_checkpoint=args.source_checkpoint,
            source_group=args.source_group,
        )
    elif not args.execute:
        result = fork_preflight(
            args.output,
            release=args.release,
            source_checkpoint=args.source_checkpoint,
            source_group=args.source_group,
            resume=args.phase == "resume",
        )
    elif args._lease_fd is not None:
        if args.phase not in {"launch", "resume"}:
            parser.error("worker lease requires launch or resume")
        _execute(args, resume=args.phase == "resume", lease_fd=args._lease_fd)
        result = {"status": "completed", "phase": args.phase}
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        command = [sys.executable, str(Path(__file__).resolve()), *map(str, [
            args.release.resolve(), args.source_checkpoint.resolve(), args.source_group.resolve(),
            args.output.resolve(), "--phase", args.phase, "--execute",
        ])]
        if args.wandb_run_id:
            command.extend(["--wandb-run-id", args.wandb_run_id])
        result = supervise(
            args.output,
            command,
            admit=lambda: fork_preflight(
                args.output,
                release=args.release,
                source_checkpoint=args.source_checkpoint,
                source_group=args.source_group,
                resume=args.phase == "resume",
            ),
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
