"""Finite intact-task schedule, evidence accounting and inspect-first lifecycle.

No model imports, GPU inspection or filesystem mutation at import/inspection time.
The public trainer owns all optimization and full-state checkpoint serialization.
"""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from writing_agent.catalog import fingerprint, save_json
from writing_agent.grpo import (
    GRPOSettings,
    file_hashes,
    inspect_grpo,
    train_grpo,
    verify_checkpoint,
)
from writing_agent.grpo_full48 import load_full48_release, mechanical_full48_reward
from writing_agent.grpo_full48_supervisor import verify_lease, writer_lease
from writing_agent.grpo_rollout import verify_tokens
from writing_agent.grpo_runtime import STREAMING

SETTINGS = GRPOSettings(
    revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    runtime_profile="intact-full48-v1",
    context_tokens=131072,
    max_tokens=8192,
    max_generated_tokens=65536,
    max_steps=96,
    max_invocations=8,
    group_size=4,
    microbatch_size=1,
    loss_type="dapo",
    tie_policy="continue",
)
BUDGET_RATIONALE = {
    "original_envelope_maxima": {
        "max_steps": 48,
        "max_tool_calls": 64,
        "max_read_tokens": 12000,
        "max_total_bytes": 262144,
        "followups": 9,
    },
    "constructed_native_paths": {
        "initial_tokens_max": 850,
        "trajectory_tokens_max": 4066,
        "action_tokens_max": 3014,
        "per_decision_tokens_max": 951,
        "source_sha256": "d224e4a05399e03a04c8edc230fc541c313234773fc1fdccbbacb46799c08c43",
        "source": "primary-full48-fixtures/native-token-evidence.json",
        "qualification": "compressible constructions, not sampled or worst-case bounds",
    },
    "per_decision": "8192: 1200-word deliverable at 4 tokens/word + 3392 reasoning/framing",
    "sampled_total": (
        "65536: final 3600-word multi-file delivery at 4 tokens/word (14400), "
        "48 decisions with 512 reasoning/framing tokens (24576), and 26560 revision headroom"
    ),
    "context": (
        "131072: 65536 actions + 12000 whitespace read units at 4 tokens/unit (48000) "
        "+ 850 initial + 222 followup tokens leaves 16464 for tool framing and other observations"
    ),
    "limits": (
        "Engineering allocation, not a token-per-word theorem or GPU fit proof. "
        "Allows all unchanged tasks and final-pass full delivery; does not promise arbitrary "
        "maximal rewriting, long reasoning, or adversarial tokenization fits. "
        "Context overflow halts; candidate exhaustion retains failure evidence without retry. "
        "Original per-task step/tool/read/storage budgets apply."
    ),
}


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or corrupt evidence: {path}") from exc


def schedule(tasks, settings):
    """Zero-based group/pass/task/slot identities, disjoint decision seed intervals."""
    settings.validate()
    if settings.runtime_profile != "intact-full48-v1":
        raise ValueError("Ordered schedule requires the intact-full48-v1 profile")
    if not tasks or settings.max_steps != 2 * len(tasks):
        raise ValueError("Schedule requires exactly two complete passes")
    if max(t["visible"]["budgets"]["max_steps"] for t in tasks) > settings.decision_seed_stride:
        raise ValueError("Decision seed stride cannot cover original task envelope")
    return [
        {
            "group": step,
            "pass_index": step // len(tasks),
            "task_index": step % len(tasks),
            "task": tasks[step % len(tasks)]["id"],
            "task_hash": fingerprint(tasks[step % len(tasks)]),
            "slots": [
                {
                    "slot": slot,
                    "seed": settings.seed
                    + (step * settings.group_size + slot) * settings.decision_seed_stride,
                    "decisions": tasks[step % len(tasks)]["visible"]["budgets"]["max_steps"],
                }
                for slot in range(settings.group_size)
            ],
        }
        for step in range(settings.max_steps)
    ]


def frozen_plan(release, run_dir):
    data = load_full48_release(Path(release))
    run_dir = Path(run_dir).resolve()
    plan = inspect_grpo(
        data["tasks"],
        run_dir / "trainer",
        settings=SETTINGS,
        reward_spec=data["reward_spec"],
        admission=data["admission"],
        implementation=STREAMING,
    )
    root = Path(__file__).resolve().parents[2]
    return {
        "recipe": "intact-full48-v1",
        "release": str(Path(release).resolve()),
        "trainer_plan": plan,
        "schedule": schedule(data["tasks"], SETTINGS),
        "first_stop": 48,
        "total_updates": 96,
        "total_attempts": 384,
        "elapsed_cutoff": None,
        "budget_rationale": BUDGET_RATIONALE,
        "source_hashes": {
            str(p.relative_to(root)): fingerprint(p.read_bytes())
            for p in [
                *sorted((root / "src/writing_agent").glob("*.py")),
                root / "scripts/run_grpo_full48.py",
            ]
        },
        "launch_prerequisites": [
            "Reviewed source-pinned trainer compatibility in the execution environment",
            "Native Gemma BF16 long-trajectory training fit at these frozen limits",
        ],
    }


def prepare(release, run_dir):
    """Create one immutable recipe in a new directory; never edits the release."""
    plan = frozen_plan(release, run_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "prepared.json", {"identity": fingerprint(plan), "plan": plan})
    return {"status": "prepared", "identity": fingerprint(plan), "plan": plan}


def coverage(output, tasks, settings):
    """Report partial coverage; damaged evidence is an explicit error, never a zero reward.

    Only a verified full-state checkpoint proves completed optimizer boundaries.
    Scored groups beyond it are uncommitted and cannot be sampled again.
    """
    output = Path(output)
    expected = schedule(tasks, settings)
    slots = Counter()
    groups = Counter()
    errors, rows, invocations = [], [], {}
    checkpoint_step, checkpoint_path, sealed_files = 0, None, {}
    identity = None
    if (output / "experiment.json").exists():
        try:
            experiment = read_json(output / "experiment.json")
            identity = experiment["identity"]
            saved_plan = experiment["manifest"]["plan"]
            if saved_plan["settings"] != asdict(settings) or saved_plan["tasks"] != tasks:
                raise ValueError("Trainer experiment differs from requested schedule/settings")
            if fingerprint(experiment["manifest"]) != identity:
                raise ValueError("Experiment identity hash mismatch")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(str(exc))
    for path in sorted((output / "invocations").glob("*")):
        try:
            start = read_json(path / "started.json")
            if start["identity"] != identity:
                raise ValueError("Invocation identity mismatch")
            invocations[path.name] = {"id": path.name, **start, "started_hash": fingerprint(start)}
            for terminal in ("complete", "stopped"):
                if (path / f"{terminal}.json").exists():
                    invocations[path.name][terminal] = read_json(path / f"{terminal}.json")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(f"{path.name}: {exc}")
    for path in sorted(output.glob("checkpoint-*")):
        # Missing final marker means interrupted save, preserved by trainer quarantine.
        if not (path / "complete.json").exists():
            continue
        try:
            step = verify_checkpoint(path, identity)
            if path.name != f"checkpoint-{step}" or not 1 <= step <= settings.max_steps:
                raise ValueError("Checkpoint step outside schedule or wrong directory")
            if step > checkpoint_step:
                checkpoint_step, checkpoint_path = step, str(path)
                sealed_files = read_json(path / "complete.json")["group_files"]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(f"{path.name}: {exc}")
    committed_files = {}
    for path in sorted((output / "groups").glob("*")):
        row = {
            "path": path.name,
            "attempts": 0,
            "scored": False,
            "attempt_directories": len(list(path.glob("attempt-*"))),
        }
        rows.append(row)
        try:
            started = read_json(path / "started.json")
            step = started["group"]
            if type(step) is not int or not 0 <= step < len(expected):
                raise ValueError("Unexpected group identity")
            groups[step] += 1
            want = expected[step]
            if any(
                started.get(k) != want[k]
                for k in ("group", "pass_index", "task_index", "task", "task_hash")
            ) or not path.name.startswith(f"step-{step:06d}-"):
                raise ValueError("Group/task/pass identity mismatch")
            invocation = invocations.get(started.get("invocation"))
            if (
                invocation is None
                or invocation["resumed_step"] > step
                or started.get("invocation_hash") != invocation["started_hash"]
            ):
                raise ValueError("Missing or invalid invocation lineage")
            row.update({k: want[k] for k in ("group", "pass_index", "task_index", "task")})
            row["invocation"] = started["invocation"]
            row["optimizer_boundary_completed"] = step < checkpoint_step
            attempt_errors_before = len(errors)
            for attempt in sorted(path.glob("attempt-*")):
                try:
                    a = read_json(attempt / "started.json")
                    slot = a["slot"]
                    if type(slot) is not int or not 0 <= slot < settings.group_size:
                        raise ValueError("Unexpected slot")
                    slots[step, slot] += 1
                    row["attempts"] += 1
                    if attempt.name != f"attempt-{slot:03d}" or any(
                        a.get(k) != value
                        for k, value in {
                            "seed": want["slots"][slot]["seed"],
                            "task": want["task"],
                            "group": step,
                            "pass_index": want["pass_index"],
                            "decision_seed_stride": settings.decision_seed_stride,
                        }.items()
                    ):
                        raise ValueError("Attempt seed/slot/pass/task identity mismatch")
                    result = read_json(attempt / "result.json")
                    reward = read_json(attempt / "reward.json")
                    tokens = read_json(attempt / "tokens.json")
                    verify_tokens(tokens)
                    if (
                        result.get("failure_class") == "infrastructure"
                        or result.get("seed") != a["seed"]
                        or reward.get("status") != "ok"
                        or not any(tokens["env_mask"])
                    ):
                        raise ValueError("Unavailable or corrupt attempt evidence")
                    if (
                        len(tokens["boundaries"]) > want["slots"][slot]["decisions"]
                        or any(
                            len(b.get("output_ids", [])) > settings.max_tokens
                            for b in tokens["boundaries"]
                        )
                        or sum(tokens["env_mask"]) > settings.max_generated_tokens
                        or len(tokens["prompt_ids"]) + len(tokens["completion_ids"])
                        > settings.context_tokens
                    ):
                        raise ValueError("Attempt exceeds frozen envelope")
                except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
                    errors.append(f"{path.name}/{attempt.name}: {exc}")
            if len(errors) != attempt_errors_before:
                raise ValueError("Group contains invalid attempt evidence")
            stats = read_json(path / "group.json")
            complete = read_json(path / "complete.json")
            if (
                stats["status"] != "ok"
                or stats["tie_policy"] != "continue"
                or complete != {"status": "scored", "attempts": settings.group_size}
                or row["attempts"] != settings.group_size
                or (path / "stopped.json").exists()
            ):
                raise ValueError("Incomplete or unavailable group")
            values = [read_json(p / "reward.json")["value"] for p in sorted(path.glob("attempt-*"))]
            tied = len(set(values)) == 1
            if stats["zero_variance"] != tied:
                raise ValueError("Tie evidence mismatch")
            row.update(scored=True, tied=tied, signal=not tied)
            if step < checkpoint_step:
                committed_files.update(
                    {f"{path.name}/{k}": v for k, v in file_hashes(path).items()}
                )
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            errors.append(f"{path.name}: {exc}")
    missing = [
        [step, slot]
        for step in range(settings.max_steps)
        for slot in range(settings.group_size)
        if slots[step, slot] == 0
    ]
    duplicates = [[*key, count] for key, count in slots.items() if count > 1]
    if duplicates or any(n != 1 for n in groups.values()):
        errors.append("Duplicate scheduled groups or slots")
    if any(groups[step] != 1 for step in range(checkpoint_step)):
        errors.append("Checkpoint lacks complete ordered group coverage")
    if committed_files != sealed_files:
        errors.append("Checkpoint seal does not cover exact committed group evidence")
    uncommitted = sorted(step for step in groups if step >= checkpoint_step)
    total_complete = checkpoint_step == settings.max_steps and not errors and not missing
    return {
        "status": "complete" if total_complete else "partial",
        "scheduled_groups": settings.max_steps,
        "scheduled_attempts": settings.max_steps * settings.group_size,
        "groups_started": len(rows),
        "attempts_started": sum(r["attempt_directories"] for r in rows),
        "slots_with_identity": sum(slots.values()),
        "attempts_with_results": len(list((output / "groups").glob("*/attempt-*/result.json"))),
        "groups_scored": sum(r["scored"] for r in rows),
        "tied_groups": sum(r.get("tied", False) for r in rows),
        "signal_groups": sum(r.get("signal", False) for r in rows),
        "completed_optimizer_boundaries": checkpoint_step,
        "latest_complete_checkpoint": checkpoint_path,
        "missing_slots": missing,
        "duplicate_slots": duplicates,
        "uncommitted_groups": uncommitted,
        "errors": errors,
        "invocations": list(invocations.values()),
        "groups": rows,
        "semantic_status": "unjudged",
    }


def preflight(release, run_dir, *, resume=False):
    run_dir = Path(run_dir).resolve()
    plan = frozen_plan(release, run_dir)
    if read_json(run_dir / "prepared.json") != {"identity": fingerprint(plan), "plan": plan}:
        raise ValueError("Prepared recipe changed; use a new run directory")
    data = load_full48_release(Path(release))
    report = coverage(run_dir / "trainer", data["tasks"], SETTINGS)
    admit_coverage(report, len(data["tasks"]), resume=resume)
    return {"status": "preflight-passed", "coverage": report, "plan": plan}


def admit_coverage(report, first_stop, *, resume):
    """Boundary recovery may advance to pass one, then pass two; never repeat a slot."""
    if report["errors"] or report["uncommitted_groups"]:
        raise ValueError(
            "Evidence cannot recover without resampling: "
            + str(report["errors"] or report["uncommitted_groups"])
        )
    step = report["completed_optimizer_boundaries"]
    if resume:
        if not report["latest_complete_checkpoint"] or step >= report["scheduled_groups"]:
            raise ValueError("Resume requires latest complete, unexhausted checkpoint")
    elif step or report["groups_started"] or report["invocations"]:
        raise ValueError("Training already started; explicit resume required")
    return first_stop if step < first_stop else report["scheduled_groups"]


def execute_training(release, run_dir, *, lease_fd, resume=False):
    """Caller must hold the supervisor lease; CLI workers inherit its locked descriptor."""
    verify_lease(run_dir, lease_fd)
    preflight(release, run_dir, resume=resume)
    data = load_full48_release(Path(release))
    return run_training(
        Path(run_dir) / "trainer",
        data["tasks"],
        SETTINGS,
        admission=data["admission"],
        reward_spec=data["reward_spec"],
        reward_callback=mechanical_full48_reward,
        implementation=STREAMING,
        lease_fd=lease_fd,
        resume=resume,
    )


def run_training(output, tasks, settings, *, resume=False, lease_fd=None, **trainer_kwargs):
    """Hold one writer lease across admission, trainer execution and final accounting."""
    root = Path(output).parent
    if lease_fd is not None:
        verify_lease(root, lease_fd)
        return _run_training(output, tasks, settings, resume=resume, **trainer_kwargs)
    with writer_lease(root):
        return _run_training(output, tasks, settings, resume=resume, **trainer_kwargs)


def _run_training(output, tasks, settings, *, resume=False, **trainer_kwargs):
    """Shared finite lifecycle; also exercised with a caller-owned tiny CPU model."""
    before = coverage(output, tasks, settings)
    boundary = admit_coverage(before, len(tasks), resume=resume)
    try:
        result = train_grpo(
            tasks,
            output,
            settings=settings,
            execute=True,
            resume_from_checkpoint=before["latest_complete_checkpoint"] if resume else None,
            stop_after_steps=boundary,
            **trainer_kwargs,
        )
        after = coverage(output, tasks, settings)
        if (
            after["errors"]
            or after["uncommitted_groups"]
            or (after["completed_optimizer_boundaries"] != boundary)
        ):
            raise ValueError("Trainer did not finish at a verified schedule boundary")
        return {"training": result, "coverage": after}
    finally:
        save_json(Path(output).parent / "coverage.json", coverage(output, tasks, settings))
