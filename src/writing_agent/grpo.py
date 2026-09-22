"""Gated single-device TRL GRPO and hash-bound adapter/resume artifacts.

No imports here load optional model libraries. Attempts are never pruned; trainer
checkpoints retain the latest two. A run has a fixed total step budget, including
resumption. Only complete, hash-verified trainer checkpoints may resume it.
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
from uuid import uuid4

from writing_agent.agent import SYSTEM_PROMPT
from writing_agent.catalog import fingerprint, save_json
from writing_agent.grpo_identity import admission_identity, base_tensor_identity
from writing_agent.grpo_rollout import NativeRolloutBackend, RolloutGroups, saved_rewards
from writing_agent.grpo_runtime import (
    LEGACY,
    implementation_plan,
    validate_streaming_model,
    verify_runtime,
)
from writing_agent.inference import PROTOCOL, checkpoint_identity
from writing_agent.workspace import TOOL_SCHEMAS


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

    @property
    def gradient_accumulation_steps(self):
        return self.group_size // (self.microbatch_size or self.group_size)

    def validate(self):
        if self.loss_type not in ("grpo", "dapo"):
            raise ValueError("Loss type must be grpo or dapo")
        if self.tie_policy not in ("halt", "continue"):
            raise ValueError("Tie policy must be halt or continue")
        checkpoint_identity(self.model_id, self.revision)
        if not (2 <= self.group_size <= 8 and 1 <= self.max_steps <= 20):
            raise ValueError("Serial probe requires group size 2..8 and optimizer steps 1..20")
        if self.microbatch_size is not None and (
            type(self.microbatch_size) is not int
            or not 1 <= self.microbatch_size <= self.group_size
            or self.group_size % self.microbatch_size != 0
        ):
            raise ValueError("Training microbatch must be a positive integer dividing group size")
        if not (1 <= self.max_tokens <= self.max_generated_tokens < self.context_tokens <= 4096):
            raise ValueError("Require bounded generated tokens < context <= 4096")
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
        if not (
            1 <= b["max_steps"] <= 16
            and 0 <= b["max_tool_calls"] <= 32
            and 0 <= b["max_read_tokens"] <= 8192
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
        "seed_policy": "run_seed + (global_step * group_size + slot) * 32 + decision",
        "dropout": 0,
        "weight_decay": 0,
        "retention": {
            "trainer_checkpoints": 2,
            "inference_adapters": settings.max_invocations,
            "attempts": settings.max_steps * settings.group_size * settings.max_invocations,
            "policy": "Keep every attempt; refuse after max_invocations",
        },
        "evaluation": "none",
        "protocol": "append-only-gemma-native-v1",
    }


def file_hashes(path, *, exclude=()):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name not in exclude
    }


def seal_directory(path, identity, kind, *, group_files=None):
    files = file_hashes(path, exclude=("complete.json",))
    marker = {"identity": identity, "kind": kind, "files": files}
    if kind == "trainer":
        marker["group_files"] = group_files or {}
    save_json(path / "complete.json", marker)


def verify_checkpoint(path, identity):
    try:
        marker = json.loads((path / "complete.json").read_text())
        required = {
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "adapter_config.json",
            "adapter_model.safetensors",
        }
        if (
            marker["identity"] != identity
            or marker["kind"] != "trainer"
            or not required <= marker["files"].keys()
            or marker["files"] != file_hashes(path, exclude=("complete.json",))
        ):
            raise ValueError("Incomplete or changed trainer checkpoint")
        for relative, digest in marker["group_files"].items():
            item = path.parent / "groups" / relative
            if not item.resolve().is_relative_to((path.parent / "groups").resolve()):
                raise ValueError("Invalid checkpoint group path")
            if hashlib.sha256(item.read_bytes()).hexdigest() != digest:
                raise ValueError("Checkpoint rollout evidence changed")
        step = json.loads((path / "trainer_state.json").read_text())["global_step"]
        if type(step) is not int or step < 1:
            raise ValueError("Invalid checkpoint global step")
        return step
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("Missing/truncated complete trainer checkpoint") from exc


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
    resume_from_checkpoint=None,
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
    verified_runtime = verify_runtime(implementation)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    caller_owned = model is not None
    if isinstance(model, PeftModel) or getattr(model, "peft_config", None):
        raise ValueError("Supply a fresh caller base model, not a PEFT model")
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            local_files_only=True,
            trust_remote_code=False,
            device_map={"": "cuda:0"},
            dtype=torch.bfloat16,
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
    lora_config = LoraConfig(
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
    # PEFT configs include sets; normalize once for portable JSON equality.
    manifest = json.loads(
        json.dumps(manifest, default=lambda x: sorted(x) if isinstance(x, set) else str(x))
    )
    identity = fingerprint(manifest)
    resumed_step = 0
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = Path(resume_from_checkpoint)
        try:
            saved = json.loads((output / "experiment.json").read_text())
            if saved != {"identity": identity, "manifest": manifest}:
                raise ValueError("Resume experiment identity changed")
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Missing/truncated experiment manifest") from exc
        if resume_from_checkpoint.resolve().parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to this output directory")
        resumed_step = verify_checkpoint(resume_from_checkpoint, identity)
        if resumed_step >= settings.max_steps:
            raise ValueError("Checkpoint has already exhausted this experiment's step budget")
        partial_saves = []
        for later in sorted(output.glob("checkpoint-*")):
            suffix = later.name.split("-")[-1]
            if not suffix.isdigit() or int(suffix) <= resumed_step:
                continue
            try:
                verify_checkpoint(later, identity)
            except ValueError:
                partial_saves.append(later)
            else:
                raise ValueError("Resume must use the latest complete checkpoint")
    else:
        partial_saves = []
    if stop_after_steps is not None and not resumed_step < stop_after_steps <= settings.max_steps:
        raise ValueError("Stop boundary must advance within the frozen step budget")
    # All identity/checkpoint checks precede mutations to caller model, tokenizer and RNG.
    quarantined = []
    for partial in partial_saves:
        quarantine = output / "quarantine" / f"{partial.name}-{uuid4().hex}"
        quarantine.parent.mkdir(exist_ok=True)
        partial.rename(quarantine)
        quarantined.append(str(quarantine))
    if resume_from_checkpoint is None:
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / "experiment.json", {"identity": identity, "manifest": manifest})
    set_seed(settings.seed)
    model = get_peft_model(model, lora_config)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    tokenizer.padding_side = "left"

    def trainable_evidence():
        tensors = {
            name: value.detach().float().cpu()
            for name, value in model.named_parameters()
            if value.requires_grad
        }
        digest = hashlib.sha256()
        for name, value in sorted(tensors.items()):
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
        lora_b = [value for name, value in tensors.items() if "lora_B" in name]
        return {
            "sha256": digest.hexdigest(),
            "tensor_count": len(tensors),
            "lora_B_tensor_count": len(lora_b),
            "lora_B_l1": sum(value.abs().sum().item() for value in lora_b),
        }

    trainable_before = None
    invocation = output / "invocations" / uuid4().hex
    invocation.mkdir(parents=True)
    save_json(
        invocation / "started.json",
        {
            "identity": identity,
            "resumed_from": str(resume_from_checkpoint) if resume_from_checkpoint else None,
            "resumed_step": resumed_step,
            "quarantined": quarantined,
        },
    )
    if backend_factory is None:

        def backend_factory(live_model, live_tokenizer, seed):
            return NativeRolloutBackend(
                live_model,
                live_tokenizer,
                {
                    "protocol": PROTOCOL,
                    "prompt_format": "chat",
                    "seed": seed,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "enable_thinking": settings.enable_thinking,
                    "context_tokens": settings.context_tokens,
                    "max_tokens": settings.max_tokens,
                    "max_generated_tokens": settings.max_generated_tokens,
                },
            )

    rollouts = RolloutGroups(
        tasks, settings, output, reward_callback, backend_factory, system_prompt
    )

    class CheckpointLifecycle(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            nonlocal trainable_before
            # Trainer restores resumed adapter weights before this callback.
            trainable_before = trainable_evidence()
            return control

        def on_save(self, args, state, control, **kwargs):
            checkpoint = output / f"checkpoint-{state.global_step}"
            seal_directory(
                checkpoint, identity, "trainer", group_files=file_hashes(output / "groups")
            )
            verify_checkpoint(checkpoint, identity)
            if stop_after_steps is not None and state.global_step >= stop_after_steps:
                control.should_training_stop = True
            return control

    args = GRPOConfig(
        **(verified_runtime["config"] if verified_runtime else {}),
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
        use_cpu=next(model.parameters()).device.type == "cpu",
        bf16=next(model.parameters()).dtype == torch.bfloat16,
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
        scale_rewards="group",
        loss_type=settings.loss_type,
        mask_truncated_completions=False,
        shuffle_dataset=False,
        report_to="none",
        logging_steps=1,
        save_steps=1,
        save_total_limit=2,
        eval_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    try:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=Dataset.from_list([{"prompt": task["id"]} for task in tasks]),
            reward_funcs=saved_rewards,
            rollout_func=rollouts,
            callbacks=[CheckpointLifecycle()],
        )
        if trainer.accelerator.num_processes != 1:
            raise ValueError("Only a single accelerator process is supported")
        result = trainer.train(
            resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
        )
        if trainable_before is None:
            raise RuntimeError("Trainer did not capture the pre-update adapter state")
        adapter = invocation / "adapter"
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(adapter)
        seal_directory(adapter, identity, "inference-adapter")
        checkpoint = output / f"checkpoint-{trainer.state.global_step}"
        verify_checkpoint(checkpoint, identity)
        metadata = {
            "identity": identity,
            "global_step": trainer.state.global_step,
            "resumed_step": resumed_step,
            "quarantined": quarantined,
            "checkpoint": str(checkpoint),
            "adapter": str(adapter),
            "metrics": result.metrics,
            "trainable_before": trainable_before,
            "trainable_after": trainable_evidence(),
            "retention": plan["retention"],
        }
        metadata["trainable_changed"] = (
            metadata["trainable_before"]["sha256"] != metadata["trainable_after"]["sha256"]
        )
        save_json(invocation / "complete.json", metadata)
        return metadata
    except BaseException as exc:
        save_json(invocation / "stopped.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
