"""Fail-closed checkpoint-31 fork and native W&B configuration.

The fork is deliberately separate from the stopped full48 run.  Source files are
copied only into the fork after their sealed hashes are recorded; the source tree
is never used as an output directory and is never resealed.  Saved sampled actions
are immutable.  The one incomplete slot is resumed through the native continuation
boundary, so no earlier action is decoded, rendered, or sampled again.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from writing_agent.agent import run_agent
from writing_agent.catalog import fingerprint, save_json
from writing_agent.grpo import file_hashes, verify_checkpoint
from writing_agent.grpo_rollout import (
    NativeRolloutBackend,
    ProtocolError,
    RolloutGroups,
    verify_tokens,
)
from writing_agent.reward import Reward, group_advantages
from writing_agent.workspace import Workspace


FORK_ID = "checkpoint31-dapo-native-fork-v1"
GROUP_STEP = 31
GROUP_NAME = "step-000031-checkpoint31-fork"
WandbConfig = dict


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Missing or corrupt evidence: {path}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity(checkpoint: Path) -> str:
    experiment = checkpoint.parent / "experiment.json"
    data = _read_json(experiment)
    if data.get("identity") != fingerprint(data.get("manifest")):
        raise ValueError("Original experiment identity hash mismatch")
    return data["identity"]


def _source_experiment(checkpoint: Path) -> dict:
    data = _read_json(checkpoint.parent / "experiment.json")
    if data.get("identity") != fingerprint(data.get("manifest")):
        raise ValueError("Original experiment identity hash mismatch")
    return data


def _check_source_checkpoint(checkpoint: Path) -> dict:
    checkpoint = checkpoint.resolve()
    if checkpoint.name != "checkpoint-31":
        raise ValueError("Fork source must be the complete checkpoint-31")
    experiment = _source_experiment(checkpoint)
    identity = experiment["identity"]
    step = verify_checkpoint(checkpoint, identity)
    if step != GROUP_STEP:
        raise ValueError("Fork source checkpoint is not optimizer boundary 31")
    marker = _read_json(checkpoint / "complete.json")
    actual_group_files = file_hashes(checkpoint.parent / "groups")
    sealed_actual = {
        relative: actual_group_files.get(relative) for relative in marker["group_files"]
    }
    if sealed_actual != marker["group_files"]:
        raise ValueError("Original committed group files differ from checkpoint seal")
    return {
        "path": str(checkpoint),
        "identity": identity,
        "experiment_manifest": experiment["manifest"],
        "complete_sha256": _sha256(checkpoint / "complete.json"),
        "files": marker["files"],
        "group_files": marker["group_files"],
    }


def _check_source_group(
    group: Path, *, source_experiment: dict | None = None, source_root: Path | None = None
) -> dict:
    group = group.resolve()
    if not group.is_dir() or not group.name.startswith("step-000031-"):
        raise ValueError("Fork source must be the stopped checkpoint-31 group")
    started = _read_json(group / "started.json")
    if started.get("group") != GROUP_STEP or started.get("task") != "wave1-train-032":
        raise ValueError("Unexpected checkpoint-31 group identity")
    if source_experiment is not None:
        tasks = source_experiment["manifest"]["plan"]["tasks"]
        task = next((item for item in tasks if item["id"] == started["task"]), None)
        if task is None or fingerprint(task) != started.get("task_hash"):
            raise ValueError("Stopped group task hash is outside the source experiment")
        if source_root is not None:
            invocation = source_root / "invocations" / started["invocation"] / "started.json"
            saved_invocation = _read_json(invocation)
            if (
                saved_invocation.get("identity") != source_experiment.get("identity")
                or fingerprint(saved_invocation) != started.get("invocation_hash")
            ):
                raise ValueError("Stopped group invocation is outside the source experiment")
    if (group / "complete.json").exists():
        raise ValueError("Source group must remain unresolved before the fork")
    attempts = {}
    for slot in range(4):
        attempt = group / f"attempt-{slot:03d}"
        if not attempt.is_dir():
            raise ValueError(f"Missing source attempt {slot}")
        evidence = _read_json(attempt / "tokens.json")
        verify_tokens(evidence)
        result = _read_json(attempt / "result.json")
        reward = _read_json(attempt / "reward.json")
        saved = _read_json(attempt / "started.json")
        if saved.get("slot") != slot or saved.get("group") != GROUP_STEP:
            raise ValueError(f"Source attempt {slot} identity mismatch")
        expected_seed = 42 + (GROUP_STEP * 4 + slot) * 48
        if saved.get("seed") != expected_seed or saved.get("task") != "wave1-train-032":
            raise ValueError(f"Source attempt {slot} seed/task identity mismatch")
        if slot == 2:
            if result.get("failure_class") != "infrastructure":
                raise ValueError("Slot002 is no longer the recorded infrastructure stop")
            if reward.get("status") != "unavailable":
                raise ValueError("Slot002 historical unavailable reward changed")
            if not (group / "stopped.json").exists():
                raise ValueError("Slot002 stopped evidence is missing")
            messages = result.get("messages")
            if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
                raise ValueError("Slot002 continuation messages are missing")
            if messages[-1].get("role") != "user":
                raise ValueError("Slot002 does not end at the saved follow-up boundary")
            if len(evidence["boundaries"]) != 5:
                raise ValueError("Slot002 must contain exactly five saved boundaries")
            _verify_trace(attempt, result, evidence)
        elif reward.get("status") != "ok":
            raise ValueError(f"Saved slot {slot} does not have a usable reward")
        attempts[str(slot)] = {
            "files": file_hashes(attempt),
            "result_sha256": _sha256(attempt / "result.json"),
            "tokens_sha256": _sha256(attempt / "tokens.json"),
            "reward_sha256": _sha256(attempt / "reward.json"),
        }
    return {
        "path": str(group),
        "started": started,
        "files": file_hashes(group),
        "attempts": attempts,
    }


def _verify_trace(attempt: Path, result: dict, evidence: dict) -> int:
    """Check every saved model input/output event against the immutable ledger."""
    trace = result.get("trace")
    if not isinstance(trace, list):
        raise ValueError("Saved trace is missing")
    try:
        on_disk = [json.loads(line) for line in (attempt / "trace.jsonl").read_text().splitlines()]
    except (OSError, ValueError) as exc:
        raise ValueError("Saved trace is corrupt") from exc
    # The embedded trace is captured by reference while the run mutates message
    # history; the flushed JSONL stream is the immutable event record.
    trace = on_disk
    inputs = [event for event in trace if event.get("type") == "model_input"]
    outputs = [event for event in trace if event.get("type") == "model_output"]
    generations = [event for event in trace if event.get("type") == "generation"]
    boundaries = evidence["boundaries"]
    if not (len(inputs) == len(outputs) == len(generations) == len(boundaries)):
        raise ValueError("Trace generation count differs from saved boundaries")
    if [event.get("step") for event in generations] != list(range(len(boundaries))):
        raise ValueError("Saved generation steps are not contiguous")
    for event, boundary in zip(inputs, boundaries, strict=True):
        if event.get("input_ids") != boundary["input_ids"]:
            raise ValueError("Saved model input differs from token ledger")
    for event, boundary in zip(outputs, boundaries, strict=True):
        if event.get("output_ids") != boundary.get("output_ids"):
            raise ValueError("Saved model output differs from token ledger")
    return len(generations)


def native_wandb_config(
    *, run_id: str, project: str = "gemma-writing", entity: str = "immpanda"
) -> WandbConfig:
    """Return a scalar-only native Trainer W&B binding.

    The environment variables bind a caller-created run explicitly.  No W&B
    import or network call occurs here, and Trainer is not asked to watch the
    model, upload checkpoints, or persist task/prose data.
    """
    if not run_id or any(c.isspace() for c in run_id):
        raise ValueError("A concrete W&B run id is required for an explicit binding")
    if not project or not entity:
        raise ValueError("W&B project and entity are required")
    return {
        "report_to": "wandb",
        "run_name": run_id,
        "project": project,
        "entity": entity,
        "env": {
            "WANDB_ENTITY": entity,
            "WANDB_PROJECT": project,
            "WANDB_RUN_ID": run_id,
            "WANDB_RESUME": "allow",
            "WANDB_MODE": "online",
            "WANDB_LOG_MODEL": "false",
            "WANDB_WATCH": "false",
            "WANDB_DISABLE_CODE": "true",
        },
        "privacy": {
            "allow": ["trainer scalar metrics", "system metrics", "package metadata"],
            "deny": ["task text", "prose", "traces", "model weights", "checkpoints"],
        },
    }


def apply_native_wandb_binding(config: WandbConfig) -> None:
    """Bind the approved run in the current trainer process, without importing W&B."""
    if config.get("report_to") != "wandb" or not isinstance(config.get("env"), dict):
        raise ValueError("Not a native W&B configuration")
    import sys

    active = sys.modules.get("wandb")
    run = getattr(active, "run", None) if active else None
    if run is not None and (
        getattr(run, "id", None) != config.get("env", {}).get("WANDB_RUN_ID")
        or getattr(run, "entity", config["env"].get("WANDB_ENTITY"))
        != config["env"].get("WANDB_ENTITY")
        or getattr(run, "project_name", config["env"].get("WANDB_PROJECT"))
        != config["env"].get("WANDB_PROJECT")
    ):
        raise ValueError("A different W&B run is already active")
    for key, value in config["env"].items():
        existing = os.environ.get(key)
        if existing is not None and existing.lower() != str(value).lower():
            raise ValueError(f"Existing {key} conflicts with frozen W&B binding")
    for key, value in config["env"].items():
        os.environ[key] = str(value)


def _copy_attempt(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(source, target)


def prepare_fork(source_checkpoint, source_group, output, *, wandb_run_id=None) -> dict:
    """Create a new fork directory while pinning immutable source evidence."""
    checkpoint = Path(source_checkpoint)
    group = Path(source_group)
    source = _check_source_checkpoint(checkpoint)
    if group.resolve().parent != checkpoint.parent.resolve() / "groups":
        raise ValueError("Stopped group is not from the original checkpoint experiment")
    group_info = _check_source_group(
        group,
        source_experiment={
            "identity": source["identity"],
            "manifest": source["experiment_manifest"],
        },
        source_root=checkpoint.parent,
    )
    output = Path(output).resolve()
    source_root = checkpoint.parent.resolve()
    if output == source_root or output.is_relative_to(source_root):
        raise ValueError("Fork output must be outside immutable source tree")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fork output must be a new, empty directory")
    trainer = output / "trainer"
    fork_group = trainer / "groups" / GROUP_NAME
    fork_group.mkdir(parents=True)
    for slot in (0, 1, 3):
        _copy_attempt(group / f"attempt-{slot:03d}", fork_group / f"attempt-{slot:03d}")
    # Keep the stopped attempt byte-for-byte under source/.  The live attempt
    # receives only newly generated suffix evidence and never overwrites this copy.
    (fork_group / "attempt-002").mkdir()
    _copy_attempt(group / "attempt-002", fork_group / "attempt-002" / "source")
    shutil.copy2(
        group / "attempt-002" / "started.json", fork_group / "attempt-002" / "started.json"
    )
    save_json(
        fork_group / "started.json",
        {
            **group_info["started"],
            "fork_id": FORK_ID,
            "source_group": str(group.resolve()),
            "imported_slots": [0, 1, 3],
            "continuation_slot": 2,
        },
    )
    manifest = {
        "fork_id": FORK_ID,
        "source_checkpoint": source,
        "source_group": group_info,
        "lineage": {
            "original_committed_prefix": {"through_checkpoint": GROUP_STEP, "inclusive": True},
            "fork_local_tail": {"starts_at_group": GROUP_STEP, "first_new_checkpoint": 32},
        },
        "identity": None,
        "wandb": native_wandb_config(run_id=wandb_run_id) if wandb_run_id else None,
    }
    manifest["identity"] = fingerprint({k: v for k, v in manifest.items() if k != "identity"})
    save_json(output / "fork.json", manifest)
    save_json(
        fork_group / "import.json",
        {
            "fork_id": FORK_ID,
            "source_group": str(group.resolve()),
            "attempts": {slot: group_info["attempts"][str(slot)] for slot in (0, 1, 3)},
            "slot002_prefix": group_info["attempts"]["2"],
        },
    )
    return manifest


def verify_fork_manifest(output: Path) -> dict:
    """Verify source pins and imported bytes without mutating either tree."""
    output = Path(output).resolve()
    manifest = _read_json(output / "fork.json")
    if manifest.get("fork_id") != FORK_ID or manifest.get("identity") != fingerprint(
        {k: v for k, v in manifest.items() if k != "identity"}
    ):
        raise ValueError("Fork identity mismatch")
    source_root = Path(manifest["source_checkpoint"]["path"]).resolve().parent
    if output == source_root or output.is_relative_to(source_root):
        raise ValueError("Fork output is inside immutable source tree")
    source = _check_source_checkpoint(Path(manifest["source_checkpoint"]["path"]))
    if source != manifest["source_checkpoint"]:
        raise ValueError("Original checkpoint seal or path changed")
    group = _check_source_group(
        Path(manifest["source_group"]["path"]),
        source_experiment={
            "identity": source["identity"],
            "manifest": source["experiment_manifest"],
        },
        source_root=Path(source["path"]).parent,
    )
    if group != manifest["source_group"]:
        raise ValueError("Original group evidence changed")
    fork_group = output / "trainer" / "groups" / GROUP_NAME
    for slot in (0, 1, 3):
        if file_hashes(fork_group / f"attempt-{slot:03d}") != group["attempts"][str(slot)]["files"]:
            raise ValueError(f"Imported attempt {slot} differs from source")
    if file_hashes(fork_group / "attempt-002" / "source") != group["attempts"]["2"]["files"]:
        raise ValueError("Imported slot002 prefix differs from source")
    return manifest


def fork_preflight(output: Path) -> dict:
    """Validate source/import admission before any checkpoint-32 evidence exists."""
    output = Path(output).resolve()
    manifest = verify_fork_manifest(output)
    trainer = output / "trainer"
    if (trainer / "experiment.json").exists():
        experiment = _read_json(trainer / "experiment.json")
        if experiment.get("manifest", {}).get("fork_manifest_identity") != manifest["identity"]:
            raise ValueError("Trainer experiment is not bound to this fork manifest")
    return {
        "status": "preflight-passed",
        "fork_id": FORK_ID,
        "source_identity": manifest["source_checkpoint"]["identity"],
        "first_new_checkpoint": 32,
        "first_stop": 48,
    }


def fork_coverage(output: Path) -> dict:
    """Report verified original-prefix plus fork-local tail coverage.

    The original 31 optimizer boundaries are evidence from v2, not work done by
    this fork.  Only a fork checkpoint-32 seal can claim the first new update.
    """
    output = Path(output).resolve()
    errors = []
    try:
        manifest = verify_fork_manifest(output)
        trainer = output / "trainer"
        experiment = _read_json(trainer / "experiment.json")
        fork_identity = experiment["identity"]
        if fork_identity == manifest["source_checkpoint"]["identity"]:
            raise ValueError("Fork trainer identity must differ from v2")
        group = trainer / "groups" / GROUP_NAME
        if _read_json(group / "complete.json") != {"status": "scored", "attempts": 4}:
            raise ValueError("Checkpoint-31 fork group is not scored")
        results = group / "fork-results"
        for slot in range(4):
            reward_path = results / f"attempt-{slot:03d}" / "reward.json"
            if not reward_path.exists():
                raise ValueError(f"Fork reward missing for slot {slot}")
            reward = _read_json(reward_path)
            if reward.get("status") != "ok":
                raise ValueError(f"Fork reward unavailable for slot {slot}")
        checkpoints = []
        for candidate in sorted(trainer.glob("checkpoint-*")):
            try:
                step = verify_checkpoint(candidate, fork_identity)
            except ValueError:
                continue
            checkpoints.append((step, candidate))
        if not checkpoints:
            raise ValueError("No complete fork checkpoint exists")
        checkpoint_step, checkpoint = max(checkpoints)
        if checkpoint_step < 32:
            raise ValueError("Fork checkpoint does not seal update 32")
        marker = _read_json(checkpoint / "complete.json")
        actual_group_files = file_hashes(trainer / "groups")
        if any(
            actual_group_files.get(path) != digest
            for path, digest in marker["group_files"].items()
        ):
            raise ValueError("Latest checkpoint group seal differs from committed fork files")
        committed_groups = {path.split("/", 1)[0] for path in marker["group_files"]}
        pending_groups = sorted(
            path.name
            for path in (trainer / "groups").iterdir()
            if path.is_dir() and path.name not in committed_groups
        )
        started = _read_json(group / "started.json")
        if started.get("group") != GROUP_STEP or started.get("task") != "wave1-train-032":
            raise ValueError("Fork group task identity mismatch")
        for slot in range(4):
            saved = _read_json(group / f"attempt-{slot:03d}/started.json")
            expected_seed = 42 + (GROUP_STEP * 4 + slot) * 48
            if (
                saved.get("slot") != slot
                or saved.get("group") != GROUP_STEP
                or saved.get("task") != "wave1-train-032"
                or saved.get("seed") != expected_seed
            ):
                raise ValueError(f"Fork slot {slot} seed/task identity mismatch")
    except (OSError, ValueError, KeyError, TypeError, ProtocolError) as exc:
        errors.append(str(exc))
        manifest = None
    return {
        "status": (
            "ready"
            if not errors and not pending_groups
            else ("partial" if not errors else "blocked")
        ),
        "fork_id": FORK_ID,
        "original_prefix": {
            "source_identity": manifest["source_checkpoint"]["identity"] if manifest else None,
            "through_checkpoint": GROUP_STEP,
            "verified": manifest is not None,
        },
        "fork_tail": {
            "group": GROUP_STEP,
            "first_new_checkpoint": 32,
            "latest_checkpoint": checkpoint_step if not errors else None,
            "pending_groups": pending_groups if not errors else [],
            "verified": not errors,
        },
        "errors": errors,
    }


class Checkpoint31RolloutGroups(RolloutGroups):
    """TRL callback that imports three actions and resumes slot002 once."""

    def __init__(self, *args, fork_output, task, **kwargs):
        super().__init__(*args, **kwargs)
        self.fork_output = Path(fork_output)
        self.fork_root = self.fork_output.parent
        self.fork_group = self.fork_output / "groups" / GROUP_NAME
        self.results = self.fork_group / "fork-results"
        self.task = copy.deepcopy(task)

    def __call__(self, prompts, trainer):
        if trainer.state.global_step != GROUP_STEP:
            return super().__call__(prompts, trainer)
        if len(prompts) != self.settings.group_size or len(set(prompts)) != 1:
            raise ProtocolError("Expected one checkpoint-31 group")
        if prompts[0] != self.task["id"]:
            raise ProtocolError("Checkpoint-31 prompt differs from imported task")
        verify_fork_manifest(self.fork_root)
        evidence, rewards = [], []
        for slot in (0, 1, 3):
            attempt = self.fork_group / f"attempt-{slot:03d}"
            tokens = _read_json(attempt / "tokens.json")
            result = _read_json(attempt / "result.json")
            verify_tokens(tokens)
            reward = self.reward_callback(copy.deepcopy(self.task), copy.deepcopy(result))
            if not isinstance(reward, Reward) or reward.status != "ok":
                raise ValueError(f"Imported slot {slot} reward unavailable")
            save_json(self.results / f"attempt-{slot:03d}" / "reward.json", asdict(reward))
            evidence.append(tokens)
            rewards.append(reward)
        slot2 = self._continue_slot002(trainer)
        evidence.append(slot2["tokens"])
        rewards.append(slot2["reward"])
        # Reorder by slot; slot002 was appended after imported 003 for convenient
        # immutable reads above.
        evidence[2], evidence[3] = evidence[3], evidence[2]
        rewards[2], rewards[3] = rewards[3], rewards[2]
        stats = group_advantages(rewards)
        stats["tie_policy"] = self.settings.tie_policy
        if stats["status"] != "ok":
            raise ValueError("Fork group reward is unavailable; refusing optimizer update")
        save_json(self.fork_group / "group.json", stats)
        save_json(self.fork_group / "complete.json", {"status": "scored", "attempts": 4})
        return {
            "prompt_ids": [e["prompt_ids"] for e in evidence],
            "completion_ids": [e["completion_ids"] for e in evidence],
            "env_mask": [e["env_mask"] for e in evidence],
            "logprobs": None,
            "rollout_rewards": [r.value for r in rewards],
        }

    def _continue_slot002(self, trainer):
        source = self.fork_group / "attempt-002" / "source"
        target = self.fork_group / "attempt-002"
        suffix_path = target / "continuation-trace.jsonl"
        if suffix_path.exists():
            raise ProtocolError(
                "Slot002 continuation evidence already exists; inspect instead of resampling"
            )
        result = _read_json(source / "result.json")
        tokens = _read_json(source / "tokens.json")
        verify_tokens(tokens)
        saved_generations = _verify_trace(source, result, tokens)
        if saved_generations != len(tokens["boundaries"]):
            raise ProtocolError("Saved generation count differs from token boundaries")
        visible = self.task["visible"]
        workspace = Workspace(
            target / "workspace", max_total_bytes=visible["budgets"]["max_total_bytes"]
        )
        for path, content in result["after"].items():
            workspace.write_file(path, content)
        backend = self.backend_factory(trainer.model, trainer.processing_class, result["seed"])
        if not isinstance(backend, NativeRolloutBackend):
            raise ProtocolError("Checkpoint-31 continuation requires the native backend")
        backend.restore_answer_followup(result["messages"], tokens)
        trace = []
        with (target / "continuation-trace.jsonl").open("w") as stream:
            def emit(event):
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
                trace.append(event)

            resumed = run_agent(
                backend,
                workspace,
                [{"role": "user", "content": visible["brief"]}],
                tools=visible["tools"],
                followups=visible["followups"],
                emit=emit,
                system_prompt=self.system_prompt,
                resume={
                    **{k: result[k] for k in (
                        "messages", "turns", "tool_calls", "attempted_tool_calls",
                        "tool_errors", "read_tokens", "read_tokenizer", "usage",
                    )},
                    "step": saved_generations,
                },
                **{k: v for k, v in visible["budgets"].items() if k != "max_total_bytes"},
            )
        # The scorer needs the complete ordered ledger. Keep the immutable source
        # JSONL under source/ and retain the newly generated suffix separately.
        source_trace = [
            json.loads(line) for line in (source / "trace.jsonl").read_text().splitlines()
        ]
        resumed["trace"] = source_trace + trace
        resumed["before"] = visible["initial_files"]
        resumed["seed"] = result["seed"]
        resumed["after"] = workspace.snapshot()
        resumed["failure_class"] = backend.failure or (
            None if resumed["status"] == "completed" else "candidate_invalid"
        )
        final_tokens = backend.evidence()
        verify_tokens(final_tokens)
        save_json(target / "result.json", resumed)
        with (target / "trace.jsonl").open("w") as stream:
            for event in resumed["trace"]:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        save_json(target / "tokens.json", final_tokens)
        reward = self.reward_callback(copy.deepcopy(self.task), copy.deepcopy(resumed))
        if not isinstance(reward, Reward) or reward.status != "ok":
            raise ValueError("Resumed slot002 reward unavailable; refusing optimizer update")
        save_json(self.results / "attempt-002" / "reward.json", asdict(reward))
        return {"tokens": final_tokens, "reward": reward}


def fork_rollout_factory(_fork_root: Path):
    """Return the factory passed to :func:`train_grpo` for a checkpoint-31 fork."""
    # The closure names the fork root for call-site readability.  The trainer
    # passes its actual ``output`` directory below, avoiding accidental writes
    # beside the imported evidence.
    def factory(
        tasks,
        settings,
        output,
        reward_callback,
        backend_factory,
        system_prompt,
        *,
        invocation_id,
    ):
        task = next(t for t in tasks if t["id"] == "wave1-train-032")
        return Checkpoint31RolloutGroups(
            tasks,
            settings,
            output,
            reward_callback,
            backend_factory,
            system_prompt,
            invocation_id=invocation_id,
            fork_output=Path(output),
            task=task,
        )

    return factory


def fork_trainer_options(fork_root, *, wandb_run_id=None) -> dict:
    """Build the guarded public-Trainer options for the first pass.

    This is configuration only.  The caller must still pass ``execute=True`` to
    ``train_grpo`` explicitly.  The first invocation stops at checkpoint 48;
    continuing to 96 requires a separate inspected invocation.
    """
    fork_root = Path(fork_root).resolve()
    manifest = verify_fork_manifest(fork_root)
    frozen_logging = manifest.get("wandb")
    if frozen_logging is not None:
        frozen_id = frozen_logging["env"]["WANDB_RUN_ID"]
        if wandb_run_id is not None and wandb_run_id != frozen_id:
            raise ValueError("Requested W&B run differs from frozen fork manifest")
        logging = frozen_logging
    elif wandb_run_id:
        logging = native_wandb_config(run_id=wandb_run_id)
    else:
        logging = {"report_to": "none", "run_name": None, "env": {}, "privacy": {}}
    return {
        "resume_from_checkpoint": manifest["source_checkpoint"]["path"],
        "resume_checkpoint_identity": manifest["source_checkpoint"]["identity"],
        "rollout_factory": fork_rollout_factory(fork_root),
        "stop_after_steps": 48,
        "report_to": logging["report_to"],
        "wandb_run_name": logging["run_name"],
        "wandb_environment": logging["env"],
        "fork_manifest_identity": manifest["identity"],
    }


def fork_resume_options(fork_root, *, checkpoint=48) -> dict:
    """Explicit second invocation: resume a fork-local checkpoint to step 96."""
    fork_root = Path(fork_root).resolve()
    manifest = verify_fork_manifest(fork_root)
    trainer = fork_root / "trainer"
    experiment = _read_json(trainer / "experiment.json")
    if experiment.get("manifest", {}).get("fork_manifest_identity") != manifest["identity"]:
        raise ValueError("Trainer experiment is not bound to this fork manifest")
    checkpoint_path = trainer / f"checkpoint-{checkpoint}"
    if not checkpoint_path.exists():
        raise ValueError("Fork-local checkpoint is missing; inspect before resuming")
    return {
        "resume_from_checkpoint": str(checkpoint_path),
        "resume_checkpoint_identity": None,
        "stop_after_steps": 96,
        "fork_manifest_identity": manifest["identity"],
        "report_to": experiment["manifest"].get("logging", {}).get("report_to", "none"),
        "wandb_run_name": experiment["manifest"].get("logging", {}).get("wandb_run_name"),
        "wandb_environment": experiment["manifest"].get("logging", {}).get(
            "wandb_environment", {}
        ),
        "rollout_factory": fork_rollout_factory(fork_root),
    }
