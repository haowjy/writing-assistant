"""Real offline CPU optimizer/save/reload/resume probe, using engineered toy tasks.

Run explicitly with --execute --output /tmp/a-new-directory. No weights downloaded;
this samples a random tiny Llama and checks LoRA/optimizer resume, not Gemma quality.
"""

import argparse
import json
from pathlib import Path

from writing_agent.backends import Completion
from writing_agent.catalog import save_json
from writing_agent.grpo import GRPOSettings, train_grpo
from writing_agent.grpo_rollout import verify_tokens
from writing_agent.reward import Reward


class ToyBackend:
    """Caller-owned fixed-length text protocol, sampling the live trainer model."""

    def __init__(self, model, tokenizer, seed):
        self.model, self.tokenizer, self.seed = model, tokenizer, seed
        self.failure = None
        self.tokens = {"prompt_ids": [], "completion_ids": [], "env_mask": [], "boundaries": []}
        self.calls = 0

    def complete(self, messages, tools, *, emit):
        import torch

        assert not tools
        # Every toy task has this explicit visible brief. Private checks never enter it.
        supplied = self.tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
        if not self.calls:
            self.tokens["prompt_ids"] = supplied
        else:
            self.tokens["completion_ids"].extend(supplied)
            self.tokens["env_mask"].extend([0] * len(supplied))
        prompt = self.tokens["prompt_ids"] + self.tokens["completion_ids"]
        self.tokens["boundaries"].append(
            {"input_ids": prompt.copy(), "completion_offset": len(self.tokens["completion_ids"])}
        )
        ids = torch.tensor([prompt], device=self.model.device)
        emit({"type": "model_input", "input_ids": prompt, "prompt": messages[-1]["content"]})
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.random.fork_rng(), torch.inference_mode():
                torch.manual_seed(self.seed + self.calls)
                output = self.model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=4,
                    min_new_tokens=0,
                    do_sample=True,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    eos_token_id=None,
                    pad_token_id=0,
                )[0][len(prompt) :].tolist()
        finally:
            for module, mode in modes:
                module.training = mode
        text = self.tokenizer.decode(output, skip_special_tokens=False)
        emit({"type": "model_output", "output_ids": output, "text": text})
        self.tokens["boundaries"][-1]["output_ids"] = output
        self.tokens["completion_ids"].extend(output)
        self.tokens["env_mask"].extend([1] * len(output))
        self.calls += 1
        return Completion({"role": "assistant", "content": text}, {})

    def evidence(self):
        verify_tokens(self.tokens)
        return self.tokens


def toy_reward(task, result):
    """Explicit artificial arithmetic over surface words; no semantic score."""
    words = result["output"].split()
    values = {f"word{i}": i / 15 for i in range(16)}
    value = sum(values.get(w, 0.0) for w in words) / max(len(words), 1)
    return Reward("ok", value, components={"engineered_word_index": value})


def tiny_model():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(123)
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2, **{f"word{i}": i + 3 for i in range(16)}}
    native = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    native.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=native, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>"
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            eos_token_id=None,
            pad_token_id=0,
            attention_dropout=0.0,
        )
    )
    return model, tokenizer


def smoke(output):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    task = {
        "id": "engineered-cpu-smoke",
        "role": "train",
        "source_groups": ["toy-only"],
        "labels": {"checks": [{"id": "word-index", "kind": "engineered"}]},
        "visible": {
            "brief": "word3 word4",
            "initial_files": {"notes.txt": "fixture"},
            "tools": [],
            "followups": ["word5"],
            "prose": [],
            "budgets": {
                "max_steps": 2,
                "max_tool_calls": 0,
                "max_read_tokens": 0,
                "max_total_bytes": 1024,
            },
        },
    }
    settings = GRPOSettings(
        model_id="caller-owned/tiny-random-llama",
        revision="a" * 40,
        max_steps=3,
        group_size=4,
        context_tokens=128,
        max_tokens=4,
        max_generated_tokens=8,
        lora_rank=2,
        learning_rate=0.001,
    )
    common = dict(
        tasks=[task],
        admission={
            "mode": "engineered-fixture",
            "label": "CPU toy feasibility, no useful-training evidence",
        },
        settings=settings,
        reward_spec={"id": "toy-word-index-v1", "mode": "mechanical-only-smoke", "config": {}},
        reward_callback=toy_reward,
        execute=True,
        backend_factory=ToyBackend,
        runtime_identity={
            "model": "tiny-random-llama-seed123-v1",
            "backend": "toy-fixed-length-v1",
        },
    )
    full_model, tokenizer = tiny_model()
    base_before = {k: p.detach().clone() for k, p in full_model.named_parameters()}
    full = train_grpo(output=output / "full", model=full_model, tokenizer=tokenizer, **common)
    model, tokenizer = tiny_model()
    partial = train_grpo(
        output=output / "resumed", model=model, tokenizer=tokenizer, stop_after_steps=1, **common
    )
    saved = load_file(str(Path(partial["adapter"]) / "adapter_model.safetensors"))
    delta = sum(t.abs().sum().item() for k, t in saved.items() if "lora_B" in k)
    assert delta > 0, "No nonzero LoRA update"
    reload_base, tokenizer = tiny_model()
    reloaded = PeftModel.from_pretrained(reload_base, partial["adapter"], local_files_only=True)
    reloaded.save_pretrained(output / "reloaded-adapter")
    roundtrip = load_file(str(output / "reloaded-adapter" / "adapter_model.safetensors"))
    assert all(torch.equal(saved[k], roundtrip[k]) for k in saved)
    # Simulate a crashed save after complete checkpoint 1; retain every partial byte.
    interrupted = output / "resumed" / "checkpoint-2"
    interrupted.mkdir()
    (interrupted / "optimizer.pt").write_bytes(b"interrupted optimizer save")
    (interrupted / "complete.json").write_text("{")
    model, tokenizer = tiny_model()
    resumed = train_grpo(
        output=output / "resumed",
        model=model,
        tokenizer=tokenizer,
        resume_from_checkpoint=Path(partial["checkpoint"]),
        **common,
    )
    assert resumed["global_step"] == 3 and resumed["resumed_step"] == 1
    assert not Path(partial["checkpoint"]).exists(), "Step 1 should be pruned by retention=2"
    assert resumed["trainable_before"] == partial["trainable_after"], (
        "Before-state was not restored"
    )
    assert resumed["trainable_changed"], "Resume only loaded weights without updating them"
    quarantine = Path(resumed["quarantined"][0])
    assert (quarantine / "optimizer.pt").read_bytes() == b"interrupted optimizer save"
    assert (quarantine / "complete.json").read_text() == "{"
    full_weights = load_file(str(Path(full["adapter"]) / "adapter_model.safetensors"))
    resumed_weights = load_file(str(Path(resumed["adapter"]) / "adapter_model.safetensors"))
    max_diff = max((full_weights[k] - resumed_weights[k]).abs().max().item() for k in full_weights)
    assert max_diff == 0, f"Resume diverged: {max_diff}"
    for key, before in base_before.items():
        current_key = key.replace(".weight", ".base_layer.weight")
        params = dict(full_model.named_parameters())
        after = params.get(key, params.get(current_key))
        assert after is not None and torch.equal(before, after), f"Base changed: {key}"
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        left = torch.load(Path(full["checkpoint"]) / name, weights_only=False)
        right = torch.load(Path(resumed["checkpoint"]) / name, weights_only=False)

        def equal(a, b):
            import numpy as np

            if isinstance(a, np.ndarray):
                return np.array_equal(a, b)
            if isinstance(a, torch.Tensor):
                return torch.equal(a, b)
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)):
                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
            return a == b

        assert equal(left, right), f"Resume state diverged: {name}"

    def step_tokens(root):
        return [
            json.loads(p.read_text())
            for p in sorted(root.glob("groups/step-00000[12]-*/attempt-*/tokens.json"))
        ]

    assert step_tokens(output / "full") == step_tokens(output / "resumed")
    assert all(e["env_mask"] == [1] * 4 + [0] + [1] * 4 for e in step_tokens(output / "full"))
    report = {
        "kind": "engineered CPU feasibility only",
        "lora_B_l1_after_step1": delta,
        "adapter_reload_exact": True,
        "partial_checkpoint_quarantined_exact": True,
        "resume_steps": [1, 3],
        "old_checkpoint_pruned": True,
        "restored_before_state_exact": True,
        "sampled_resume_tokens_exact": True,
        "environment_tokens_masked": True,
        "resume_max_abs_difference": max_diff,
        "optimizer_scheduler_rng_resume_exact": True,
        "base_weights_unchanged": True,
        "full": full,
        "partial": partial,
        "resumed": resumed,
    }
    save_json(output / "smoke.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/grpo-cpu-smoke"))
    args = parser.parse_args()
    print(
        json.dumps(
            smoke(args.output) if args.execute else {"status": "inspect", "execute": False},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
