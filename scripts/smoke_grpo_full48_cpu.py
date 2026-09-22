"""Live tiny CPU proof of the full48 schedule/recovery seam; no production weights."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from scripts.smoke_grpo_cpu import VariableToyBackend, same_state, tiny_model
from writing_agent.catalog import save_json
from writing_agent.grpo import train_grpo
from writing_agent.grpo_full48_runner import SETTINGS, coverage, run_training, schedule
from writing_agent.reward import Reward


def slot_reward(task, result):
    return Reward(
        "ok", 1.0 if task["id"] != "cpu-full48-1" else ((result["seed"] - 42) // 48 % 4) / 3
    )


def smoke(output):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    tasks = [
        {
            "id": f"cpu-full48-{i}",
            "role": "train",
            "source_groups": ["engineered-only"],
            "labels": {"checks": []},
            "visible": {
                "brief": "word3 word4",
                "initial_files": {},
                "followups": ["word5"],
                "tools": [],
                "prose": [],
                "budgets": {
                    "max_steps": 2,
                    "max_tool_calls": 0,
                    "max_read_tokens": 0,
                    "max_total_bytes": 1024,
                },
            },
        }
        for i in range(3)
    ]
    settings = replace(
        SETTINGS,
        model_id="caller-owned/tiny-random-llama",
        revision="a" * 40,
        max_steps=6,
        context_tokens=128,
        max_tokens=4,
        max_generated_tokens=8,
        lora_rank=2,
        learning_rate=0.001,
    )
    common = {
        "admission": {"mode": "engineered-fixture", "label": "Full48 scheduler CPU only"},
        "reward_spec": {"id": "cpu-full48-slots", "mode": "mechanical-only-smoke", "config": {}},
        "reward_callback": slot_reward,
        "backend_factory": VariableToyBackend,
        "runtime_identity": {
            "model": "tiny-random-llama-seed123-v1",
            "purpose": "schedule verification, not production reward",
        },
    }
    model, tokenizer = tiny_model()
    full = train_grpo(
        tasks,
        output / "full" / "trainer",
        settings=settings,
        execute=True,
        model=model,
        tokenizer=tokenizer,
        **common,
    )
    (output / "resumed").mkdir()
    model, tokenizer = tiny_model()
    first = run_training(
        output / "resumed" / "trainer", tasks, settings, model=model, tokenizer=tokenizer, **common
    )
    assert first["coverage"]["completed_optimizer_boundaries"] == 3
    assert first["coverage"]["attempts_started"] == 12
    assert first["coverage"]["status"] == "partial"
    save_json(output / "pass-one.json", first)
    model, tokenizer = tiny_model()
    last = run_training(
        output / "resumed" / "trainer",
        tasks,
        settings,
        resume=True,
        model=model,
        tokenizer=tokenizer,
        **common,
    )
    assert last["training"]["trainable_before"] == first["training"]["trainable_after"]
    for root in (output / "full" / "trainer", output / "resumed" / "trainer"):
        report = coverage(root, tasks, settings)
        assert report["status"] == "complete", report
        assert report["attempts_started"] == 24
        assert report["tied_groups"] == 4 and report["signal_groups"] == 2
        assert len(list(root.glob("checkpoint-*"))) == 2
        assert [g["task"] for g in report["groups"]] == [t["id"] for t in tasks] * 2
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        a, b = [
            torch.load(Path(r["checkpoint"]) / filename, weights_only=False)
            for r in (full, last["training"])
        ]
        assert same_state(a, b), filename
    assert same_state(
        *[
            load_file(str(Path(r["checkpoint"]) / "adapter_model.safetensors"))
            for r in (full, last["training"])
        ]
    )
    ledgers = [
        [json.loads(p.read_text()) for p in sorted(root.glob("groups/*/attempt-*/tokens.json"))]
        for root in (output / "full" / "trainer", output / "resumed" / "trainer")
    ]
    assert ledgers[0] == ledgers[1]
    seeds = [
        s["seed"] + decision
        for group in schedule(tasks, settings)
        for s in group["slots"]
        for decision in range(s["decisions"])
    ]
    assert len(seeds) == len(set(seeds))
    result = {
        "status": "passed",
        "fixture_only": True,
        "production_model_loaded": False,
        "exact_adapter_optimizer_scheduler_rng_and_tokens": True,
        "pass_one_updates": 3,
        "total_updates": 6,
        "coverage": last["coverage"],
    }
    save_json(output / "smoke.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Inspect-only: add --execute for a six-update tiny random CPU fixture")
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    result = smoke(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "coverage"}, indent=2))


if __name__ == "__main__":
    main()
