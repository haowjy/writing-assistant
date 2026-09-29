"""Offline public Gemma4 train/resume qualification for the pinned streaming path.

Requires the separately installed pinned environment. Run with CUDA hidden,
HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 and PYTHONPATH=src, using --execute
--output /absolute/new-directory. No production weights or tokenizer are loaded.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

from scripts.smoke_grpo_cpu import LossObserver, VariableToyBackend, same_state
from scripts.smoke_grpo_ties_cpu import diagnostic_reward
from writing_agent.catalog import save_json
from writing_agent.grpo import GRPOSettings, file_hashes
from writing_agent.grpo_runtime import STREAMING, verify_runtime


def tiny_gemma():
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        Gemma4Config,
        Gemma4ForConditionalGeneration,
        Gemma4TextConfig,
        PreTrainedTokenizerFast,
    )

    torch.manual_seed(123)
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    vocab.update({f"word{i}": i for i in range(3, 8209)})
    native = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    native.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=native, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>"
    )
    text = Gemma4TextConfig(
        vocab_size=8209,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        num_global_key_value_heads=1,
        max_position_embeddings=4096,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=16,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=8209,
        num_kv_shared_layers=2,
        attention_k_eq_v=True,
        enable_moe_block=False,
        final_logit_softcapping=30.0,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        eos_token_id=None,
        pad_token_id=0,
    )
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=text, tie_word_embeddings=False, attn_implementation="eager")
    ).to(dtype=torch.bfloat16, device="cpu")
    model.generation_config.eos_token_id = None
    return model, tokenizer


def common_settings():
    tasks = [
        {
            "id": f"streaming-cpu-{index}",
            "role": "train",
            "source_groups": ["engineered"],
            "labels": {"checks": [], **({"tied_value": 1.0} if index != 1 else {})},
            "visible": {
                "brief": "word3 word4",
                "initial_files": {},
                "followups": [" ".join(["word5"] * (2050 if index == 1 else 1))],
                "tools": [],
                "prose": [],
                "budgets": {
                    "max_steps": 2,
                    "max_tool_calls": 0,
                    "max_read_tokens": 0,
                    "max_total_bytes": 20000,
                },
            },
        }
        for index in range(3)
    ]
    return dict(
        tasks=tasks,
        settings=GRPOSettings(
            model_id="caller-owned/tiny-random-gemma4",
            revision="a" * 40,
            loss_type="dapo",
            tie_policy="continue",
            group_size=4,
            microbatch_size=1,
            max_steps=6,
            context_tokens=4096,
            max_tokens=4,
            max_generated_tokens=8,
            lora_rank=8,
            learning_rate=0.001,
        ),
        implementation=STREAMING,
        admission={"mode": "engineered-fixture", "label": "CPU streaming qualification only"},
        reward_spec={"id": "cpu-slots-v1", "config": {}, "mode": "mechanical-only-smoke"},
        reward_callback=diagnostic_reward,
        backend_factory=VariableToyBackend,
        runtime_identity={
            "model": "random-gemma4-bf16-seed123-ple-sharedkv-v1",
            "sources": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "smoke_grpo_cpu.py",
                    "smoke_grpo_ties_cpu.py",
                    "smoke_grpo_streaming_cpu.py",
                )
            },
        },
        execute=True,
    )


def methods(model):
    import torch.nn.functional as functional
    from transformers.models.gemma4 import modeling_gemma4

    result = {
        name: (type(module), getattr(module.forward, "__func__", module.forward))
        for name, module in model.named_modules()
    }
    result["cross_entropy"] = functional.cross_entropy
    for name, cls in vars(modeling_gemma4).items():
        if name.startswith("Gemma4") and isinstance(cls, type):
            result[f"class:{name}"] = (cls, getattr(cls, "forward", None))
    return result


def describe(mapping):
    def item(value):
        return {
            "id": id(value),
            "module": getattr(value, "__module__", None),
            "name": getattr(value, "__qualname__", None),
        }

    return {
        name: [item(v) for v in value] if isinstance(value, tuple) else item(value)
        for name, value in mapping.items()
    }


class StreamingObserver(LossObserver):
    """Observe upstream calls and gradient hooks without replacing any implementation."""

    def __init__(self):
        from transformers.integrations.liger import apply_liger_kernel
        from trl import GRPOTrainer
        from trl.trainer.utils import _ChunkedLogProbFunction

        super().__init__()
        self.init_code = GRPOTrainer.__init__.__code__
        self.dispatch_code = apply_liger_kernel.__code__
        self.chunk_code = _ChunkedLogProbFunction.forward.__code__
        self.entries, self.chunks, self.gradients = [], [], []
        self.before = {}
        self.trainer = None
        self.conditioning_inputs = None
        self.logps = None

    def __call__(self, frame, event, result):
        code, local = frame.f_code, frame.f_locals
        if code in (self.init_code, self.dispatch_code):
            model = local.get("model")
            if model is not None and event == "call":
                self.before[code] = methods(model)
            if model is not None and event == "return":
                before, after = self.before.pop(code), methods(model)
                assert before == after, "Unrelated model methods changed"
                self.entries.append(
                    {
                        "phase": "trainer-init" if code is self.init_code else "train-entry",
                        "unchanged": True,
                        "before": describe(before),
                        "after": describe(after),
                    }
                )
        if code is self.chunk_code and event == "return":
            self.chunks.append(
                {
                    key: local[key]
                    for key in (
                        "N",
                        "vocab",
                        "chunk_size",
                        "max_token_chunk",
                        "token_start",
                        "start",
                        "C",
                        "final_logit_softcapping",
                    )
                }
            )
        if code is self.code and event == "return" and result is not None:
            self.trainer = local["self"]
            inputs = local["inputs"]
            self.logps = local["per_token_logps"].detach().clone()
            if self.conditioning_inputs is None and inputs["completion_ids"].shape[1] > 2048:
                self.conditioning_inputs = copy.deepcopy(inputs)
            mask = local["mask"].detach().clone()
            attention = local["attention_mask"]
            import torch

            assert torch.equal(
                attention, torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
            )

            def record(gradient):
                assert torch.isfinite(gradient).all()
                assert torch.count_nonzero(gradient[mask == 0]) == 0
                self.gradients.append(
                    {"masked_zero": True, "finite": True, "actions": int(mask.sum())}
                )

            if local["per_token_logps"].requires_grad:
                local["per_token_logps"].register_hook(record)
        super().__call__(frame, event, result)

    def conditioning(self):
        import torch

        original = self.conditioning_inputs
        assert original is not None
        altered = copy.deepcopy(original)
        mask = original["tool_mask"][0].bool()
        position = int(torch.where(~mask)[0][-1])
        altered["completion_ids"][0, position] = 6  # Change only the last external observation.
        assert not mask[position] and mask[position + 1 :].any()
        self.trainer.model.eval()
        previous = sys.getprofile()
        try:
            sys.setprofile(self)
            with torch.no_grad():
                self.trainer.compute_loss(self.trainer.model, original)
                baseline = self.logps.clone()
                self.trainer.compute_loss(self.trainer.model, altered)
                delta = (self.logps - baseline).abs()[0]
        finally:
            sys.setprofile(previous)
        change = delta[position + 1 :][mask[position + 1 :]].max().item()
        assert change > 0, "External observations stopped conditioning subsequent actions"
        return {"changed_masked_position": position, "later_action_logprob_max_change": change}


def smoke(output):
    import torch
    from safetensors.torch import load_file

    for key, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        assert os.environ.get(key) == value, (key, value)
    runtime = verify_runtime(STREAMING)
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    common = common_settings()
    results, observers = {}, {}
    for name in ("full", "resumed"):
        observer = observers[name] = StreamingObserver()
        model, tokenizer = tiny_gemma()
        result = observer.train(
            output=output / name,
            model=model,
            tokenizer=tokenizer,
            stop_after_steps=2 if name == "resumed" else None,
            **common,
        )
        if name == "resumed":
            partial = result
            model, tokenizer = tiny_gemma()
            result = observer.train(
                output=output / name,
                model=model,
                tokenizer=tokenizer,
                resume_from_checkpoint=partial["checkpoint"],
                **common,
            )
            assert result["trainable_before"] == partial["trainable_after"]
        results[name] = result
        accounting = observer.verify(output / name, common["settings"])
        assert len(observer.gradients) == 24
        assert sum(entry["phase"] == "train-entry" for entry in observer.entries) == (
            2 if name == "resumed" else 1
        )
        assert any(
            c["token_start"] == 2048 and c["start"] == 8192 and c["C"] == 17
            for c in observer.chunks
        )
        assert all(c["final_logit_softcapping"] == 30 for c in observer.chunks)
        assert result["global_step"] == 6
        save_json(
            output / f"observations-{name}.json",
            {
                "batches": observer.batches,
                "accounting": accounting,
                "methods": observer.entries,
                "streaming_chunks": observer.chunks,
                "direct_gradients": observer.gradients,
            },
        )
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        a, b = [
            torch.load(Path(results[n]["checkpoint"]) / filename, weights_only=False)
            for n in ("full", "resumed")
        ]
        assert same_state(a, b), filename
    a, b = [
        load_file(str(Path(results[n]["adapter"]) / "adapter_model.safetensors"))
        for n in ("full", "resumed")
    ]
    assert same_state(a, b)
    ledgers = [
        [
            json.loads(p.read_text())
            for p in sorted((output / n).glob("groups/*/attempt-*/tokens.json"))
        ]
        for n in ("full", "resumed")
    ]
    assert ledgers[0] == ledgers[1] and len(ledgers[0]) == 24
    groups = [
        json.loads(p.read_text()) for p in sorted((output / "full").glob("groups/*/group.json"))
    ]
    assert [g["zero_variance"] for g in groups] == [True, False, True] * 2
    tied_batches = [b for b in observers["full"].batches if b["step"] in (0, 2, 3, 5)]
    assert all(b["loss"] == 0 and all(a == 0 for a in b["advantages"]) for b in tied_batches)
    optimizer = torch.load(Path(results["full"]["checkpoint"]) / "optimizer.pt", weights_only=False)
    assert all(s["step"].item() == 6 for s in optimizer["state"].values())
    scheduler = torch.load(Path(results["full"]["checkpoint"]) / "scheduler.pt", weights_only=False)
    assert scheduler["last_epoch"] == 6
    conditioning = observers["full"].conditioning()
    report = {
        "runtime": runtime,
        "dtype": "BF16 backbone, ordinary FP32 PEFT adapters",
        "native_bf16_parity_claimed": False,
        "group_size": 4,
        "microbatch_size": 1,
        "accumulation": 4,
        "optimizer_steps": 6,
        "tied_groups": 4,
        "heterogeneous_groups": 2,
        "attempts": 24,
        "resampling": False,
        "exact_resume": ["adapter", "optimizer", "scheduler", "RNG", "token ledgers"],
        "unrelated_model_replacements": False,
        "masked_direct_gradients_zero": True,
        "chunk_boundaries": {"tokens": 2048, "vocabulary": 8192, "vocabulary_tail": 17},
        "changed_observation": conditioning,
        "results": results,
    }
    save_json(output / "summary.json", report)
    save_json(output / "artifact-hashes.json", file_hashes(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(smoke(args.output) if args.execute else {"status": "inspect"}, indent=2))


if __name__ == "__main__":
    main()
