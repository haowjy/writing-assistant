"""Offline CPU proof of ordinary TRL/Adam continuation through tied reward groups.

Run from the checkout with CUDA_VISIBLE_DEVICES='' and PYTHONPATH=src:
python -m scripts.smoke_grpo_ties_cpu --execute --output /absolute/new-directory.
With CUDA hidden this uses no downloads or GPU. Manufactured
slot rewards prove accounting and optimizer behavior, not useful learning.
"""

import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from scripts.smoke_grpo_cpu import LossObserver, VariableToyBackend, same_state, tiny_model
from writing_agent.catalog import save_json
from writing_agent.grpo import GRPOSettings
from writing_agent.reward import Reward


def diagnostic_reward(task, result):
    """Predeclared fixture: ties or [0, 1/3, 2/3, 1], independent of policy quality."""
    value = (
        1.0 if task["labels"]["reward_case"] == "tied" else ((result["seed"] - 42) // 32 % 4) / 3
    )
    return Reward("ok", value, components={"diagnostic_only": value})


def snapshot(trainer):
    return {
        "adapter": {
            name: parameter.detach().cpu().clone()
            for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        },
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": copy.deepcopy(trainer.lr_scheduler.state_dict()),
    }


class StateObserver(LossObserver):
    """Read-only snapshots before each update and after training; no optimizer hooks."""

    def __init__(self):
        super().__init__()
        self.states = {}
        self.trainer = None

    def __call__(self, frame, event, result):
        if event == "return" and frame.f_code is self.code and result is not None:
            self.trainer = frame.f_locals["self"]
            step = self.trainer.state.global_step
            current = snapshot(self.trainer)
            if step in self.states:
                assert same_state(self.states[step], current), (
                    "State changed within group or restore"
                )
            else:
                self.states[step] = current
        super().__call__(frame, event, result)

    def train(self, **kwargs):
        result = super().train(**kwargs)
        self.states[self.trainer.state.global_step] = snapshot(self.trainer)
        return result


def smoke(output):
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    tasks = [
        {
            "id": f"cpu-ties-{index}",
            "role": "train",
            "source_groups": ["engineered-only"],
            "labels": {"checks": [], "reward_case": case},
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
        for index, case in enumerate(("tied", "signal", "tied"))
    ]
    settings = GRPOSettings(
        model_id="caller-owned/tiny-random-llama",
        revision="a" * 40,
        loss_type="dapo",
        tie_policy="continue",
        group_size=4,
        microbatch_size=1,
        max_steps=6,
        context_tokens=128,
        max_tokens=4,
        max_generated_tokens=8,
        lora_rank=2,
        learning_rate=0.001,
    )
    common = dict(
        tasks=tasks,
        settings=settings,
        admission={"mode": "engineered-fixture", "label": "CPU tie continuation only"},
        reward_spec={"id": "cpu-slots-v1", "config": {}, "mode": "mechanical-only-smoke"},
        reward_callback=diagnostic_reward,
        backend_factory=VariableToyBackend,
        runtime_identity={
            "model": "tiny-random-llama-seed123-v1",
            "probe_sources": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("smoke_grpo_cpu.py", "smoke_grpo_ties_cpu.py")
            },
        },
        execute=True,
    )
    observers = {name: StateObserver() for name in ("full", "dense", "resumed", "all-tied")}
    results = {}
    for name in ("full", "dense", "resumed"):
        model, tokenizer = tiny_model()
        config = replace(settings, microbatch_size=None) if name == "dense" else settings
        result = observers[name].train(
            output=output / name,
            model=model,
            tokenizer=tokenizer,
            stop_after_steps=2 if name == "resumed" else None,
            **{**common, "settings": config},
        )
        if name == "resumed":
            partial = result
            model, tokenizer = tiny_model()
            result = observers[name].train(
                output=output / name,
                model=model,
                tokenizer=tokenizer,
                resume_from_checkpoint=Path(partial["checkpoint"]),
                **common,
            )
            assert result["trainable_before"] == partial["trainable_after"]
        results[name] = result
        observer = observers[name]
        accounting = observer.verify(output / name, config)
        save_json(output / f"loss-{name}.json", {"batches": observer.batches, "tokens": accounting})
        groups = sorted((output / name / "groups").iterdir())
        assert [json.loads((g / "started.json").read_text())["task"] for g in groups] == [
            t["id"] for t in tasks
        ] * 2
        stats = [json.loads((g / "group.json").read_text()) for g in groups]
        assert [s["zero_variance"] for s in stats] == [True, False, True] * 2
        assert all(s["tie_policy"] == "continue" for s in stats)
        assert all((g / "complete.json").is_file() for g in groups)
        seeds = [
            json.loads(p.read_text())["seed"]
            for g in groups
            for p in sorted(g.glob("attempt-*/started.json"))
        ]
        assert seeds == [42 + i * 32 for i in range(24)], "Resampling or missing attempts"
        assert result["global_step"] == 6

    def ledgers(name):
        return [
            json.loads(p.read_text())
            for p in sorted((output / name).glob("groups/*/attempt-*/tokens.json"))
        ]

    assert ledgers("full") == ledgers("resumed") == ledgers("dense")
    assert same_state(observers["full"].states, observers["resumed"].states)
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        left, right = [
            torch.load(Path(results[name]["checkpoint"]) / filename, weights_only=False)
            for name in ("full", "resumed")
        ]
        assert same_state(left, right), filename
    full, resumed, dense = [
        load_file(str(Path(results[name]["adapter"]) / "adapter_model.safetensors"))
        for name in ("full", "resumed", "dense")
    ]
    assert same_state(full, resumed)
    torch.testing.assert_close(full, dense, rtol=0, atol=1e-6)
    torch.testing.assert_close(
        observers["full"].states[6]["optimizer"],
        observers["dense"].states[6]["optimizer"],
        rtol=1e-5,
        atol=1e-8,
    )

    states = observers["full"].states
    assert same_state(states[0]["adapter"], states[1]["adapter"]), "Leading tie moved fresh weights"
    tied_changes = {}
    for step in (2, 3, 5):
        before, after = states[step], states[step + 1]
        delta = max(
            (after["adapter"][k] - before["adapter"][k]).abs().max().item()
            for k in before["adapter"]
        )
        assert delta > 0, "Expected ordinary Adam momentum movement, not a skipped update"
        tied_changes[str(step)] = delta
        assert after["scheduler"]["last_epoch"] == before["scheduler"]["last_epoch"] + 1
        for key, state in before["optimizer"]["state"].items():
            changed = after["optimizer"]["state"][key]
            assert changed["step"].item() == state["step"].item() + 1
            for moment, beta in (("exp_avg", 0.9), ("exp_avg_sq", 0.999)):
                torch.testing.assert_close(
                    changed[moment], state[moment] * beta, rtol=1e-5, atol=1e-9
                )
    for batch in observers["full"].batches:
        if batch["step"] in (0, 2, 3, 5):
            assert batch["loss"] == 0 and all(a == 0 for a in batch["advantages"])

    # All-tied coverage finishes normally without fabricating parameter change.
    model, tokenizer = tiny_model()
    all_tied = observers["all-tied"].train(
        output=output / "all-tied",
        model=model,
        tokenizer=tokenizer,
        **{**common, "tasks": tasks[:1], "settings": replace(settings, max_steps=2)},
    )
    assert all_tied["global_step"] == 2 and not all_tied["trainable_changed"]
    observer = observers["all-tied"]
    observer.verify(output / "all-tied", replace(settings, max_steps=2))
    for state in observer.states.values():
        assert same_state(state["adapter"], observer.states[0]["adapter"])
    for state in observer.states[2]["optimizer"]["state"].values():
        assert state["step"].item() == 2
        assert not state["exp_avg"].count_nonzero() and not state["exp_avg_sq"].count_nonzero()
    for name, observer in observers.items():
        torch.save(observer.states, output / f"states-{name}.pt")
    report = {
        "kind": "engineered CPU optimizer-policy proof, not quality evidence",
        "tie_policy": "continue",
        "passes": 2,
        "visits": 6,
        "attempts": 24,
        "optimizer_steps": 6,
        "tied_groups": 4,
        "nonzero_advantage_groups": 2,
        "resampled_groups": 0,
        "leading_tie_parameters_unchanged": True,
        "later_tie_parameter_max_changes": tied_changes,
        "tied_advantages_and_losses_zero": True,
        "tied_adam_moments_decay_and_counters_advance": True,
        "resume_states_and_rng_exact": True,
        "all_token_ledgers_match": True,
        "dense_adapter_max_difference": max((full[k] - dense[k]).abs().max().item() for k in full),
        "all_tied": {"visits": 2, "optimizer_steps": 2, "parameter_change": False},
        "results": results,
    }
    save_json(output / "smoke.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/grpo-ties-cpu-smoke"))
    args = parser.parse_args()
    print(json.dumps(smoke(args.output) if args.execute else {"status": "inspect"}, indent=2))


if __name__ == "__main__":
    main()
