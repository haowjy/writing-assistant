"""Gated single-device TRL GRPO and hash-bound adapter/resume artifacts.

No imports here load optional model libraries. Attempts are never pruned; the
full48 profile retains every trainer checkpoint, while probes retain the latest
two. A run has a fixed total step budget, including resumption. Only complete,
hash-verified trainer checkpoints may resume it.
"""

import copy
import hashlib
import inspect
import json
import os
import re
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from writing_agent.agent import SYSTEM_PROMPT
from writing_agent.catalog import fingerprint
from writing_agent.grpo_checkpoint import (
    file_hashes as file_hashes,
)
from writing_agent.grpo_checkpoint import (
    seal_directory as seal_directory,
)
from writing_agent.grpo_checkpoint import (
    verify_checkpoint as verify_checkpoint,
)
from writing_agent.grpo_identity import admission_identity, base_tensor_identity
from writing_agent.grpo_rollout import NativeRolloutBackend, RolloutGroups
from writing_agent.grpo_runtime import (
    LEGACY,
    implementation_plan,
    validate_streaming_model,
    verify_runtime,
)
from writing_agent.inference import checkpoint_identity
from writing_agent.workspace import TOOL_SCHEMAS


def canonical_json_value(value):
    """Normalize JSON-equivalent mappings (notably integer config keys)."""
    if isinstance(value, dict):
        return {
            str(key): canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(canonical_json_value(item) for item in value)
    return value


@dataclass(frozen=True)
class GRPOSettings:
    model_id: str = "google/gemma-4-E2B-it"
    revision: str = ""  # Caller must select an immutable checkpoint explicitly.
    context_tokens: int = 4096
    max_tokens: int = 256
    max_generated_tokens: int = 1024
    max_steps: int = 2
    max_invocations: int = 3
    group_size: int = 2
    microbatch_size: int | None = None  # None retains full-group training.
    loss_type: str = "grpo"  # Public TRL objective, frozen in experiment identity.
    tie_policy: str = "halt"  # "continue" passes tied rewards to ordinary TRL/Adam.
    learning_rate: float = 1e-5
    lora_rank: int = 8
    seed: int = 42
    enable_thinking: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    scale_rewards: str = "group"  # TRL reward normalization, frozen in experiment identity.

    runtime_profile: str = "probe"  # Explicit admission/seed policy; not a TRL backend.

    @property
    def decision_seed_stride(self):
        return 48 if self.runtime_profile == "intact-full48-v1" else 32

    @property
    def gradient_accumulation_steps(self):
        return self.group_size // (self.microbatch_size or self.group_size)

    def validate(self):
        if self.runtime_profile not in ("probe", "intact-full48-v1"):
            raise ValueError("Unknown runtime admission profile")
        full48 = self.runtime_profile == "intact-full48-v1"
        if self.loss_type not in ("grpo", "dapo"):
            raise ValueError("Loss type must be grpo or dapo")
        if self.tie_policy not in ("halt", "continue"):
            raise ValueError("Tie policy must be halt or continue")
        if self.scale_rewards not in ("group", "batch", "none"):
            raise ValueError("Reward scaling must be group, batch or none")
        checkpoint_identity(self.model_id, self.revision)
        if not (2 <= self.group_size <= 8 and 1 <= self.max_steps <= (96 if full48 else 20)):
            raise ValueError("Serial probe requires group size 2..8 and optimizer steps 1..20")
        if self.microbatch_size is not None and (
            type(self.microbatch_size) is not int
            or not 1 <= self.microbatch_size <= self.group_size
            or self.group_size % self.microbatch_size != 0
        ):
            raise ValueError("Training microbatch must be a positive integer dividing group size")
        ceiling = 131072 if full48 else 4096
        if not (1 <= self.max_tokens <= self.max_generated_tokens < self.context_tokens <= ceiling):
            raise ValueError(f"Require bounded generated tokens < context <= {ceiling}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 - (
            self.max_steps * self.group_size * self.decision_seed_stride
        ):
            raise ValueError("Seed ranges must fit unsigned 32-bit RNG space")
        if not 1 <= self.max_invocations <= 8:
            raise ValueError("Probe invocations must be bounded to 1..8")
        if self.lora_rank < 1 or not 0 < self.learning_rate <= 0.01:
            raise ValueError("Invalid optimizer/LoRA settings")
        if not isinstance(self.gradient_checkpointing, bool) or not isinstance(
            self.gradient_checkpointing_use_reentrant, bool
        ):
            raise ValueError("Activation-checkpointing settings must be explicit booleans")


def inspect_grpo(
    tasks,
    output,
    *,
    settings,
    reward_spec,
    admission,
    system_prompt=SYSTEM_PROMPT,
    implementation=LEGACY,
):
    selected_implementation = implementation_plan(implementation)
    settings.validate()
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Select explicit unique training tasks")
    admitted = admission_identity(tasks, admission)
    for task in tasks:
        if task["role"] != "train" or not task.get("source_groups"):
            raise ValueError("Only train-role tasks with explicit source groups are allowed")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task["id"]):
            raise ValueError("Task IDs must be path-safe")
        visible = task["visible"]
        if set(visible["tools"]) - {s["function"]["name"] for s in TOOL_SCHEMAS}:
            raise ValueError("Only the five workspace tools are supported")
        b = visible["budgets"]
        full48 = settings.runtime_profile == "intact-full48-v1"
        if not (
            1 <= b["max_steps"] <= (48 if full48 else 16)
            and 0 <= b["max_tool_calls"] <= (64 if full48 else 32)
            and 0 <= b["max_read_tokens"] <= (12000 if full48 else 8192)
            and 0 < b["max_total_bytes"] <= 1_000_000
        ):
            raise ValueError("Task exceeds bounded probe budgets")
        if len(visible["followups"]) >= b["max_steps"]:
            raise ValueError("Follow-up stages exceed step budget")
        if not isinstance(task["labels"]["checks"], list):
            raise ValueError("Private scoring checks must be frozen before sampling")
    if not all(key in reward_spec for key in ("id", "config", "mode")):
        raise ValueError("Freeze reward id, config and mode before sampling")
    if reward_spec["mode"] not in {"mixed", "mechanical-only-smoke"}:
        raise ValueError("Reward mode must be mixed or explicitly mechanical-only-smoke")
    return {
        **({"implementation": selected_implementation} if selected_implementation else {}),
        "schema_version": 2,
        "admission": admitted,
        "settings": asdict(settings),
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "tasks": copy.deepcopy(tasks),
        "reward": copy.deepcopy(reward_spec),
        "system_prompt": system_prompt,
        "output": str(Path(output).resolve()),
        "beta": 0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed_policy": (
            "run_seed + (global_step * group_size + slot) * "
            f"{settings.decision_seed_stride} + decision"
        ),
        "dropout": 0,
        "weight_decay": 0,
        "retention": {
            "trainer_checkpoints": settings.max_steps if full48 else 2,
            "inference_adapters": settings.max_invocations,
            "attempts": settings.max_steps
            * settings.group_size
            * (1 if settings.runtime_profile == "intact-full48-v1" else settings.max_invocations),
            "policy": "Keep every attempt; refuse after max_invocations",
        },
        "evaluation": "none",
        "protocol": "append-only-gemma-native-v1",
    }


def train_grpo(
    tasks,
    output,
    *,
    settings,
    reward_spec,
    admission,
    reward_callback=None,
    execute=False,
    model=None,
    tokenizer=None,
    runtime_identity=None,
    backend_factory=None,
    rollout_factory=None,
    resume_from_checkpoint=None,
    resume_checkpoint_identity=None,
    report_to="none",
    wandb_run_name=None,
    wandb_environment=None,
    fork_manifest_identity=None,
    stop_after_steps=None,
    system_prompt=SYSTEM_PROMPT,
    implementation=LEGACY,
):
    """Explicit execution over caller-owned weights, or a local-only Gemma LoRA load.

    Custom backends implement complete/evidence/failure and declare their identity in
    runtime_identity. They must sample the live trainer model and return exact IDs.
    stop_after_steps interrupts at a saved optimizer boundary without changing the
    immutable total schedule, useful for scheduler/RNG resume verification.
    """
    tasks = copy.deepcopy(tasks)
    plan = inspect_grpo(
        tasks,
        output,
        settings=settings,
        reward_spec=reward_spec,
        admission=admission,
        system_prompt=system_prompt,
        implementation=implementation,
    )
    if not execute:
        return plan
    if reward_callback is None:
        raise ValueError("Execution requires an explicit frozen reward callback")
    if not inspect.isfunction(reward_callback) or reward_callback.__closure__:
        raise ValueError("Reward callbacks must be plain functions without opaque closures")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Only single-device serial execution is supported")
    output = Path(output)
    fork_manifest = None
    if fork_manifest_identity is not None:
        try:
            fork_manifest = json.loads((output.parent / "fork.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Missing/truncated fork manifest") from exc
        if fork_manifest.get("identity") != fork_manifest_identity:
            raise ValueError("Fork manifest identity changed")
        source_manifest = fork_manifest.get("source_checkpoint", {}).get("experiment_manifest")
        if not isinstance(source_manifest, dict):
            raise ValueError("Fork manifest lacks original experiment contract")
        source_plan = source_manifest.get("plan", {})
        for key in ("settings", "tasks", "reward", "admission"):
            if source_plan.get(key) != plan.get(key):
                raise ValueError(f"Fork/source {key} contract differs")
    if resume_from_checkpoint is None and output.exists() and any(output.iterdir()):
        raise ValueError("New output must be empty; use an explicit complete checkpoint to resume")
    if len(list((output / "invocations").glob("*"))) >= settings.max_invocations:
        raise ValueError(
            "Experiment invocation budget exhausted; preserve artifacts in a new experiment"
        )
    if (model is None) != (tokenizer is None):
        raise ValueError("Supply both caller-owned model and tokenizer")
    if model is not None and not runtime_identity:
        raise ValueError("Caller-owned weights/tokenizer/backend need an explicit identity")
    if backend_factory is not None and not runtime_identity:
        raise ValueError("Custom backend requires an explicit identity")
    wandb_reporting = report_to == "wandb" or report_to == ["wandb"]
    if wandb_reporting and not wandb_run_name:
        raise ValueError("W&B reporting requires an explicit bound run name")
    if report_to not in ("none", "wandb") and report_to != ["wandb"]:
        raise ValueError("Only disabled reporting or native W&B reporting is admitted")
    if wandb_reporting:
        if not isinstance(wandb_environment, dict):
            raise ValueError("W&B reporting requires frozen environment bindings")
        required_env = {
            "WANDB_RUN_ID": wandb_run_name,
            "WANDB_ENTITY": wandb_environment.get("WANDB_ENTITY"),
            "WANDB_PROJECT": wandb_environment.get("WANDB_PROJECT"),
            "WANDB_MODE": "online",
            "WANDB_LOG_MODEL": "false",
            "WANDB_WATCH": "false",
            "WANDB_DISABLE_CODE": "true",
        }
        if not required_env["WANDB_ENTITY"] or not required_env["WANDB_PROJECT"]:
            raise ValueError("W&B reporting requires explicit entity and project bindings")
        if any(
            str(wandb_environment.get(k, "")).lower() != v.lower() for k, v in required_env.items()
        ):
            raise ValueError("W&B environment does not satisfy the frozen privacy/run binding")
        for key, expected in required_env.items():
            existing = os.environ.get(key)
            if existing is not None and existing.lower() != expected.lower():
                raise ValueError(f"Existing {key} conflicts with frozen W&B binding")
        import sys

        active_wandb = sys.modules.get("wandb")
        active_run = getattr(active_wandb, "run", None) if active_wandb else None
        if active_run is not None and (
            getattr(active_run, "id", None) != wandb_run_name
            or getattr(active_run, "entity", None) != wandb_environment.get("WANDB_ENTITY")
            or getattr(active_run, "project", None) != wandb_environment.get("WANDB_PROJECT")
        ):
            raise ValueError("A different W&B run is already active")
    verified_runtime = verify_runtime(implementation)
    if wandb_reporting:
        for key, value in required_env.items():
            os.environ[key] = str(value)
    from writing_agent.grpo_trainer import load_trainer_api, run_trainer

    api = load_trainer_api()

    caller_owned = model is not None
    if isinstance(model, api.PeftModel) or getattr(model, "peft_config", None):
        raise ValueError("Supply a fresh caller base model, not a PEFT model")
    if model is None:
        tokenizer = api.AutoTokenizer.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = api.AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            local_files_only=True,
            trust_remote_code=False,
            device_map={"": "cuda:0"},
            dtype=api.torch.bfloat16,
            attn_implementation="sdpa",
        )
        runtime_identity = {
            "model": checkpoint_identity(settings.model_id, settings.revision),
            "backend": "NativeRolloutBackend-v1",
        }
    if verified_runtime:
        validate_streaming_model(model.config)
    devices = {str(p.device) for p in model.parameters()}
    if len(devices) != 1:
        raise ValueError("Model must reside on one device")
    base_identity = (
        base_tensor_identity(model)
        if caller_owned
        else {
            "trust": "immutable-local-HF-revision",
            **checkpoint_identity(settings.model_id, settings.revision),
        }
    )
    lora_config = api.LoraConfig(
        task_type="CAUSAL_LM",
        r=settings.lora_rank,
        lora_alpha=2 * settings.lora_rank,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
    )
    manifest = {
        "plan": plan,
        **({"implementation": verified_runtime} if verified_runtime else {}),
        "runtime_identity": runtime_identity,
        "base_identity": base_identity,
        "reward_implementation": fingerprint(inspect.getsource(reward_callback)),
        "backend_implementation": fingerprint(
            inspect.getsource(backend_factory or NativeRolloutBackend)
        ),
        "rollout_implementation": fingerprint(inspect.getsource(rollout_factory or RolloutGroups)),
        "logging": {
            "report_to": report_to,
            "wandb_run_name": wandb_run_name,
            "wandb_environment": wandb_environment,
        },
        "fork_manifest_identity": fork_manifest_identity,
        "tokenizer": fingerprint(tokenizer.backend_tokenizer.to_str()),
        "chat_template": fingerprint(tokenizer.chat_template),
        "response_template": fingerprint(getattr(tokenizer, "response_template", None)),
        "model_config": model.config.to_dict(),
        "generation_config": model.generation_config.to_dict(),
        "proposed_peft": lora_config.to_dict(),
        "packages": {p: version(p) for p in ("torch", "transformers", "trl", "peft", "accelerate")},
        "code": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    if fork_manifest_identity is not None:
        source_manifest = fork_manifest["source_checkpoint"]["experiment_manifest"]
        compatibility = (
            ("base_identity", base_identity),
            ("tokenizer", manifest["tokenizer"]),
            ("chat_template", manifest["chat_template"]),
            ("response_template", manifest["response_template"]),
            ("proposed_peft", manifest["proposed_peft"]),
            ("model_config", manifest["model_config"]),
            ("packages", manifest["packages"]),
        )
        for key, current in compatibility:
            if canonical_json_value(source_manifest.get(key)) != canonical_json_value(current):
                raise ValueError(f"Fork/source {key} compatibility differs")
    # PEFT configs include sets; normalize once for portable JSON equality.
    manifest = json.loads(
        json.dumps(manifest, default=lambda x: sorted(x) if isinstance(x, set) else str(x))
    )
    identity = fingerprint(manifest)
    trainer_config_values = trainer_config(
        settings,
        output,
        use_cpu=next(model.parameters()).device.type == "cpu",
        bf16=next(model.parameters()).dtype == api.torch.bfloat16,
        implementation_config=verified_runtime["config"] if verified_runtime else None,
        report_to=report_to,
        run_name=wandb_run_name,
        scale_rewards=settings.scale_rewards,
    )
    return run_trainer(
        api=api,
        tasks=tasks,
        output=output,
        settings=settings,
        plan=plan,
        identity=identity,
        manifest=manifest,
        model=model,
        tokenizer=tokenizer,
        lora_config=lora_config,
        trainer_config_values=trainer_config_values,
        reward_callback=reward_callback,
        backend_factory=backend_factory,
        rollout_factory=rollout_factory,
        system_prompt=system_prompt,
        resume_from_checkpoint=resume_from_checkpoint,
        resume_checkpoint_identity=resume_checkpoint_identity,
        stop_after_steps=stop_after_steps,
    )


def trainer_config(
    settings,
    output,
    *,
    use_cpu,
    bf16,
    implementation_config=None,
    report_to="none",
    run_name=None,
    scale_rewards="group",
):
    """Public TRL configuration shared by native training and controlled memory sizing."""
    return dict(
        **(implementation_config or {}),
        output_dir=str(output),
        max_steps=settings.max_steps,
        per_device_train_batch_size=settings.microbatch_size or settings.group_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        generation_batch_size=settings.group_size,
        num_generations=settings.group_size,
        learning_rate=settings.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        seed=settings.seed,
        data_seed=settings.seed,
        use_cpu=use_cpu,
        bf16=bf16,
        gradient_checkpointing=settings.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": settings.gradient_checkpointing_use_reentrant
        },
        optim="adamw_torch",
        beta=0.0,
        num_iterations=1,
        disable_dropout=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_completion_length=settings.max_generated_tokens,
        scale_rewards=scale_rewards,
        loss_type=settings.loss_type,
        mask_truncated_completions=False,
        shuffle_dataset=False,
        report_to=report_to,
        run_name=run_name,
        logging_steps=1,
        save_steps=1,
        save_total_limit=None if settings.runtime_profile == "intact-full48-v1" else 2,
        eval_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
