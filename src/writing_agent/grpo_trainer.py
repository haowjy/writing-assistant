"""Identity-agnostic GRPO trainer lifecycle shared by experiment profiles."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from writing_agent.catalog import save_json
from writing_agent.grpo_checkpoint import file_hashes, seal_directory, verify_checkpoint
from writing_agent.grpo_rollout import NativeRolloutBackend, RolloutGroups, saved_rewards
from writing_agent.inference import PROTOCOL


@dataclass(frozen=True)
class TrainerAPI:
    """Optional training packages imported only after runtime admission."""

    torch: object
    Dataset: object
    LoraConfig: object
    PeftModel: object
    get_peft_model: object
    AutoModelForCausalLM: object
    AutoTokenizer: object
    TrainerCallback: object
    set_seed: object
    GRPOConfig: object
    GRPOTrainer: object


def load_trainer_api():
    """Import model libraries at the admitted execution boundary."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    return TrainerAPI(
        torch=torch,
        Dataset=Dataset,
        LoraConfig=LoraConfig,
        PeftModel=PeftModel,
        get_peft_model=get_peft_model,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        TrainerCallback=TrainerCallback,
        set_seed=set_seed,
        GRPOConfig=GRPOConfig,
        GRPOTrainer=GRPOTrainer,
    )


def trainable_evidence(model):
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


def run_trainer(
    *,
    api,
    tasks,
    output,
    settings,
    plan,
    identity,
    manifest,
    model,
    tokenizer,
    lora_config,
    trainer_config_values,
    reward_callback,
    backend_factory,
    rollout_factory,
    system_prompt,
    resume_from_checkpoint,
    resume_checkpoint_identity,
    stop_after_steps,
):
    """Resume safely, construct one trainer, and persist its adapter and evidence."""
    output = Path(output)
    resumed_step = 0
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = Path(resume_from_checkpoint)
        external_resume = resume_from_checkpoint.resolve().parent != output.resolve()
        try:
            saved = json.loads((output / "experiment.json").read_text())
        except (OSError, json.JSONDecodeError):
            if not external_resume:
                raise ValueError("Missing/truncated experiment manifest") from None
            # Imported fork evidence can exist before model admission; bind the
            # new output identity without touching the external source.
            save_json(output / "experiment.json", {"identity": identity, "manifest": manifest})
        else:
            if saved != {"identity": identity, "manifest": manifest}:
                raise ValueError("Resume experiment identity changed")
        if external_resume and resume_checkpoint_identity is None:
            raise ValueError("External resume requires the source checkpoint identity")
        if not external_resume and resume_checkpoint_identity is not None:
            raise ValueError("Source checkpoint identity is only valid for an external resume")
        resumed_step = verify_checkpoint(
            resume_from_checkpoint,
            resume_checkpoint_identity if external_resume else identity,
        )
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
    api.set_seed(settings.seed)
    model = api.get_peft_model(model, lora_config)
    for module in model.modules():
        if isinstance(module, api.torch.nn.Dropout):
            module.p = 0.0
    tokenizer.padding_side = "left"

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

    rollouts = (
        rollout_factory(
            tasks,
            settings,
            output,
            reward_callback,
            backend_factory,
            system_prompt,
            invocation_id=invocation.name,
        )
        if rollout_factory is not None
        else RolloutGroups(
            tasks,
            settings,
            output,
            reward_callback,
            backend_factory,
            system_prompt,
            invocation_id=invocation.name,
        )
    )

    class CheckpointLifecycle(api.TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            nonlocal trainable_before
            # Trainer restores resumed adapter weights before this callback.
            trainable_before = trainable_evidence(model)
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

    args = api.GRPOConfig(**trainer_config_values)
    try:
        trainer = api.GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=args,
            train_dataset=api.Dataset.from_list([{"prompt": task["id"]} for task in tasks]),
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
            "trainable_after": trainable_evidence(model),
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
