"""One immutable controlled-memory profile; no sampled task success is inferred."""

import gc
import hashlib
import json
import math
import multiprocessing
import os
import resource
import sys
import time
import traceback
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from writing_agent.catalog import fingerprint, save_json
from writing_agent.grpo import file_hashes, seal_directory, trainer_config, verify_checkpoint
from writing_agent.grpo_full48_runner import SETTINGS
from writing_agent.grpo_gpu import (
    CUDA_ALLOCATOR_CONF,
    HEADLESS_POLICY,
    admit_gpu,
    capture_inventory,
    configure_cuda_allocator,
)
from writing_agent.grpo_runtime import (
    STREAMING,
    implementation_plan,
    validate_streaming_model,
    verify_runtime,
)

FIT_SETTINGS = replace(SETTINGS, max_steps=1, max_invocations=1)
FIT_CONTEXT_TOKENS = FIT_SETTINGS.context_tokens
FIT_ACTIVE_TOKENS = FIT_SETTINGS.max_tokens
FIT_PREFIX_TOKENS = FIT_CONTEXT_TOKENS - FIT_ACTIVE_TOKENS
OBSERVATION = "The archive records a quiet room, an open window, and a letter on the desk. "
ACTION = "She reads the letter carefully and writes the next page of her story. "
REWARDS = [0.0, 0.25, 0.75, 1.0]


def inspect_fit():
    root = Path(__file__).resolve().parents[2]
    return {
        "profile": "gemma-full48-controlled-fit-v6",
        "scope": "controlled memory sizing; not sampled success or native rollout semantics",
        "settings": asdict(FIT_SETTINGS),
        "implementation": implementation_plan(STREAMING),
        "ownership_policy": HEADLESS_POLICY,
        "allocator": CUDA_ALLOCATOR_CONF,
        "dtype": "bfloat16 base; ordinary FP32 PEFT adapter storage",
        "attention": "sdpa",
        "lora": {
            "rank": 8,
            "alpha": 16,
            "target_modules": "all-linear",
            "dropout": 0,
            "bias": "none",
        },
        "training": {
            "attempts": 4,
            "total_tokens_each": FIT_CONTEXT_TOKENS,
            "active_tokens_each": FIT_ACTIVE_TOKENS,
            "observation_tokens_each": (f"{FIT_PREFIX_TOKENS} minus native initial prompt length"),
            "mask": (f"zero for observation prefix, one for final {FIT_ACTIVE_TOKENS} actions"),
            "observation_text": OBSERVATION,
            "action_text": ACTION,
            "rewards": REWARDS,
            "dapo_group_active_tokens": len(REWARDS) * FIT_ACTIVE_TOKENS,
            "optimizer_steps": 1,
            "beta": 0,
        },
        "generation": {
            "input_tokens": FIT_CONTEXT_TOKENS - 1,
            "new_tokens": 1,
            "total_tokens": FIT_CONTEXT_TOKENS,
            "native_chat": True,
            "use_cache": True,
            "separate_base_load": True,
        },
        "elapsed_cutoff": None,
        "attempt_limit": 1,
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *sorted((root / "src/writing_agent").glob("*.py")),
                root / "scripts/run_grpo_gpu_fit.py",
            ]
        },
    }


def prepare_fit(directory):
    plan = inspect_fit()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    record = {"identity": fingerprint(plan), "plan": plan}
    save_json(directory / "prepared.json", record)
    return record


def preflight_fit(directory):
    plan = inspect_fit()
    record = {"identity": fingerprint(plan), "plan": plan}
    if json.loads((Path(directory) / "prepared.json").read_text()) != record:
        raise ValueError("Fit source/profile identity changed; prepare fresh evidence")
    return record


def repeat_tokens(tokens, count):
    if not tokens or count < 0:
        raise ValueError("Empty token pool or negative ledger length")
    return (tokens * ((count + len(tokens) - 1) // len(tokens)))[:count]


def token_ledgers(tokenizer):
    prompt = tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Continue the archive story."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        ),
        add_special_tokens=False,
    )
    observation = tokenizer.encode(OBSERVATION, add_special_tokens=False)
    action = tokenizer.encode(ACTION, add_special_tokens=False)
    if set(observation + action) & set(tokenizer.all_special_ids):
        raise ValueError("Sizing text unexpectedly contains special tokens")
    masked = FIT_PREFIX_TOKENS - len(prompt)
    rows = []
    for slot in range(4):
        # Rotate actual text-token pools to avoid identical reward/action rows.
        actions = action[slot:] + action[:slot]
        rows.append(
            {
                "slot": slot,
                "prompt_ids": prompt,
                "completion_ids": repeat_tokens(observation, masked)
                + repeat_tokens(actions, FIT_ACTIVE_TOKENS),
                "env_mask": [0] * masked + [1] * FIT_ACTIVE_TOKENS,
                "reward": REWARDS[slot],
            }
        )
    return rows


def generation_prefix(tokenizer):
    # Preserve native chat framing around an exact-length user body. This is a
    # token-level sizing construction, not a task or a sampled conversation.
    sentinel = "GPU_FIT_BODY"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if rendered.count(sentinel) != 1:
        raise ValueError("Native template did not preserve sizing body boundary")
    before, after = rendered.split(sentinel)
    prefix = tokenizer.encode(before, add_special_tokens=False)
    suffix = tokenizer.encode(after, add_special_tokens=False)
    pool = tokenizer.encode(OBSERVATION, add_special_tokens=False)
    return prefix + repeat_tokens(pool, FIT_CONTEXT_TOKENS - 1 - len(prefix) - len(suffix)) + suffix


def finite_state(value):
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def load_base():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"revision": SETTINGS.revision, "local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(SETTINGS.model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        SETTINGS.model_id,
        **kwargs,
        device_map={"": "cuda:0"},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    validate_streaming_model(model.config)
    if model.config.get_text_config().max_position_embeddings < FIT_CONTEXT_TOKENS:
        raise ValueError("Cached base cannot admit the enforced context")
    return model, tokenizer


def generation_check(directory):
    import torch

    model, tokenizer = load_base()
    model.eval()
    ids = generation_prefix(tokenizer)
    save_json(directory / "generation-input.json", {"input_ids": ids, "input_tokens": len(ids)})
    inputs = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        generated = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_logits=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    if (
        generated.sequences.shape[1] != FIT_CONTEXT_TOKENS
        or not torch.isfinite(generated.logits[0]).all()
    ):
        raise ValueError("Generation length/nonfinite score failure")
    cache = generated.past_key_values
    if cache is None or int(cache.get_seq_length()) != FIT_CONTEXT_TOKENS - 1:
        raise ValueError("Generation did not retain the full prefill cache")
    result = {
        "input_tokens": len(ids),
        "output_ids": generated.sequences[0, len(ids) :].tolist(),
        "cache_seq_length": int(cache.get_seq_length()),
        "finite_scores": True,
        "model_config": model.config.to_dict(),
    }
    save_json(directory / "generation.json", result)
    del generated, inputs, cache, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def training_check(directory, identity, runtime):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    set_seed(SETTINGS.seed)
    model, tokenizer = load_base()
    rows = token_ledgers(tokenizer)
    save_json(directory / "tokens.json", rows)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            target_modules="all-linear",
            bias="none",
        ),
    )
    tokenizer.padding_side = "left"
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    if not finite_state(before):
        raise ValueError("Nonfinite initial adapter")
    calls = 0
    gradient_checks = {"count": 0}

    def gradient_check(gradient):
        if not finite_state(gradient):
            raise ValueError("Nonfinite adapter gradient")
        gradient_checks["count"] += 1
        return gradient

    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(gradient_check)

    def rollout(prompts, trainer):
        nonlocal calls
        if calls or len(prompts) != 4 or trainer.state.global_step != 0:
            raise ValueError("Fit allows exactly one four-attempt ledger group")
        calls += 1
        return {
            "prompt_ids": [r["prompt_ids"] for r in rows],
            "completion_ids": [r["completion_ids"] for r in rows],
            "env_mask": [r["env_mask"] for r in rows],
            "logprobs": None,
            "rollout_rewards": REWARDS,
        }

    def reward(prompts, completions, rollout_rewards, **kwargs):
        return rollout_rewards

    class Evidence(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if not finite_state(
                {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
            ):
                raise ValueError("Nonfinite accumulated gradients")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not finite_state(logs):
                raise ValueError("Nonfinite trainer logs")
            save_json(directory / "training-logs.json", state.log_history)

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            seal_directory(checkpoint, identity, "trainer")
            if verify_checkpoint(checkpoint, identity) != 1:
                raise ValueError("Unexpected fit checkpoint boundary")

    output = directory / "trainer"
    config = trainer_config(
        FIT_SETTINGS, output, use_cpu=False, bf16=True, implementation_config=runtime["config"]
    )
    save_json(directory / "trainer-config.json", config)
    save_json(directory / "model-config.json", model.config.to_dict())
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=GRPOConfig(**config),
        train_dataset=Dataset.from_list([{"prompt": "controlled-memory-ledger"}]),
        reward_funcs=reward,
        rollout_func=rollout,
        callbacks=[Evidence()],
    )
    observations = []
    expected = {fingerprint(r["completion_ids"]): r for r in rows}
    expected_normalizer = sum(sum(r["env_mask"]) for r in rows)
    loss_code = GRPOTrainer._compute_loss.__code__

    def observe(frame, event, result):
        if event != "return" or frame.f_code is not loss_code or result is None:
            return
        local = frame.f_locals
        inputs, mask = local["inputs"], local["mask"]
        if not finite_state(result):
            raise ValueError("Nonfinite actual TRL microbatch loss")
        completion = inputs["completion_ids"].detach().cpu().tolist()
        if len(completion) != 1:
            raise ValueError("Fit must consume exactly one ledger per microbatch")
        row = expected.pop(fingerprint(completion[0]))
        if (
            inputs["prompt_ids"].detach().cpu().tolist() != [row["prompt_ids"]]
            or mask.detach().cpu().tolist() != [row["env_mask"]]
            or not bool(local["attention_mask"].all())
            or int(local["attention_mask"].shape[1]) != FIT_CONTEXT_TOKENS
            or float(local["normalizer"]) != expected_normalizer
        ):
            raise ValueError("TRL consumed different token/mask/denominator geometry")
        observation = {
            "slot": row["slot"],
            "loss": float(result.detach()),
            "total_tokens": FIT_CONTEXT_TOKENS,
            "active_tokens": int(mask.sum()),
            "normalizer": float(local["normalizer"]),
            "masked_gradient_zero": False,
        }
        observations.append(observation)

        def logprob_gradient(gradient):
            if not finite_state(gradient) or torch.count_nonzero(gradient[mask == 0]):
                raise ValueError("Nonfinite logprob gradient or active observation loss")
            observation["masked_gradient_zero"] = True
            save_json(directory / "loss-observations.json", observations)
            return gradient

        local["per_token_logps"].register_hook(logprob_gradient)
        save_json(directory / "loss-observations.json", observations)

    previous_profile = sys.getprofile()
    try:
        sys.setprofile(observe)
        result = trainer.train()
    finally:
        sys.setprofile(previous_profile)
    if (
        expected
        or len(observations) != 4
        or not all(o["masked_gradient_zero"] for o in observations)
    ):
        raise ValueError("Incomplete actual loss/mask evidence")
    after = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    checkpoint = output / "checkpoint-1"
    if verify_checkpoint(checkpoint, identity) != 1 or trainer.state.global_step != 1 or calls != 1:
        raise ValueError("Incomplete optimizer/checkpoint evidence")
    state = {
        "adapter": load_file(str(checkpoint / "adapter_model.safetensors")),
        "optimizer": torch.load(
            checkpoint / "optimizer.pt", map_location="cpu", weights_only=False
        ),
        "scheduler": torch.load(
            checkpoint / "scheduler.pt", map_location="cpu", weights_only=False
        ),
        "rng": torch.load(checkpoint / "rng_state.pth", map_location="cpu", weights_only=False),
    }
    changed = any(not torch.equal(before[n], after[n]) for n in before)
    optimizer_steps = [float(v["step"]) for v in state["optimizer"]["state"].values()]
    if (
        not changed
        or not finite_state((after, state, result.metrics))
        or not gradient_checks["count"]
        or not optimizer_steps
        or set(optimizer_steps) != {1.0}
    ):
        raise ValueError("Missing or nonfinite optimizer update")
    from writing_agent.grpo_probe import verify_loaded_adapter

    report = {
        "adapter_reload": verify_loaded_adapter(model, checkpoint),
        "loss_observations": observations,
        "global_step": 1,
        "adapter_changed": changed,
        "finite": True,
        "gradient_checks": gradient_checks,
        "optimizer_steps": sorted(set(optimizer_steps)),
        "checkpoint": str(checkpoint),
        "checkpoint_hashes": file_hashes(checkpoint),
        "token_ledger_sha256": hashlib.sha256((directory / "tokens.json").read_bytes()).hexdigest(),
        "metrics": result.metrics,
    }
    save_json(directory / "training.json", report)
    return report


def _stage_worker(directory, stage, identity):
    """A fresh process releases CUDA context/KV before the next ownership check."""
    directory = Path(directory)
    started = time.monotonic()
    result = {"stage": stage, "status": "failed", "failures": []}
    torch = None
    try:
        record = preflight_fit(directory)
        if record["identity"] != identity:
            raise ValueError("Worker profile identity changed")
        runtime = verify_runtime(STREAMING)
        configure_cuda_allocator()
        admit_gpu(directory / f"ownership-{stage}", policy=HEADLESS_POLICY)
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Fit requires exactly one visible CUDA device")
        torch.cuda.reset_peak_memory_stats()
        if stage == "generation":
            generation_check(directory)
        elif stage == "training":
            training_check(directory, identity, runtime)
        else:
            raise ValueError("Unknown fit stage")
        torch.cuda.synchronize()
        result["status"] = "passed"
    except BaseException as exc:
        result["failures"].append(
            {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if torch is not None and torch.cuda.is_initialized():
            result["torch_allocated_peak_bytes"] = torch.cuda.max_memory_allocated()
            result["torch_reserved_peak_bytes"] = torch.cuda.max_memory_reserved()
        try:
            capture_inventory(directory / f"nvml-{stage}-after.xml")
        except BaseException as exc:
            result["status"] = "failed"
            result["failures"].append({"error": f"{type(exc).__name__}: {exc}"})
        save_json(directory / f"{stage}-result.json", result)
    if result["status"] != "passed":
        raise RuntimeError("Stage evidence capture failed")


def execute_fit(directory):
    directory = Path(directory)
    # Inspection identity must match before source admission, ownership, or ML imports.
    record = preflight_fit(directory)
    with (directory / "attempt.json").open("x") as stream:
        json.dump({"identity": record["identity"], "attempt": 1}, stream)
    started = time.monotonic()
    report = {"identity": record["identity"], "status": "failed", "stages": {}, "failures": []}
    stage = "source-admission"
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        runtime = verify_runtime(STREAMING)
        allocator = configure_cuda_allocator()
        save_json(
            directory / "runtime.json",
            {
                "implementation": runtime,
                "allocator": allocator,
                "packages": {
                    p: version(p)
                    for p in ("torch", "transformers", "trl", "peft", "accelerate", "liger-kernel")
                },
            },
        )
        stage = "ownership"
        report["ownership"] = admit_gpu(directory / "ownership-before", policy=HEADLESS_POLICY)
        for stage in ("generation", "training"):
            process = multiprocessing.get_context("spawn").Process(
                target=_stage_worker, args=(str(directory), stage, record["identity"])
            )
            process.start()
            process.join()  # Finite work, no elapsed cutoff or retry.
            result_path = directory / f"{stage}-result.json"
            if result_path.is_file():
                report["stages"][stage] = json.loads(result_path.read_text())
            if process.exitcode != 0 or report["stages"].get(stage, {}).get("status") != "passed":
                raise RuntimeError(
                    f"Fit {stage} failed (exit {process.exitcode}); preserve evidence"
                )
        report["status"] = "passed"
    except BaseException as exc:
        report["failures"].append(
            {
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        try:
            capture_inventory(directory / "nvml-after.xml")
        except BaseException as exc:
            report["status"] = "failed"
            report["failures"].append(
                {"stage": "nvml-after", "error": f"{type(exc).__name__}: {exc}"}
            )
        save_json(directory / "result.json", report)
    if report["status"] != "passed":
        raise RuntimeError("Fit evidence capture failed")
    return report
