"""CPU end-to-end task-graph DAPO resume check using tiny Gemma4 and public fixtures.

Run offline with the Phase 8 overlay and CUDA hidden. The scripted native sampler emits
bounded fixture actions, but computes each recorded logprob from the live tiny Gemma4
model; the production route defaults to NativeGemmaSampleBackend.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

from writing_agent.catalog import save_json
from writing_agent.grpo import file_hashes
from writing_agent.grpo_checkpoint import verify_checkpoint
from writing_agent.grpo_task_graph import TaskGraphLossObserver
from writing_agent.grpo_task_graph_probe_experiment import (
    TOKENIZER_PATH,
    fixture_plans,
    make_task_graph_run,
    settings,
    tiny_gemma,
)
from writing_agent.native_gemma import (
    NativeGemmaRenderer,
    NativeGemmaSampleBackend,
)
from writing_agent.native_protocol import parse_native_response
from writing_agent.task_graph_ports import BinaryLogprobEvidence, SampleResultV2
from writing_agent.task_graph_probe_tasks import load_probe_tasks


class ScriptedNativeBackend:
    """Fixture action source with logprobs from the same live model used by TRL."""

    def __init__(self, model, tokenizer, *, manifest_descriptors, adapter_name, model_ref, plans):
        template = NativeGemmaSampleBackend(
            None,
            tokenizer,
            manifest_descriptors=manifest_descriptors,
            adapter_name=adapter_name,
            model_ref=model_ref,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.manifest_descriptors = manifest_descriptors
        self.descriptor = template.descriptor
        self.plans = plans
        self.renderer = NativeGemmaRenderer(tokenizer, manifest_descriptors[0])

    def sample(self, prepared):
        import torch

        plan = self.plans[prepared.writer_seed]
        raw = plan[prepared.decision_ordinal]
        generated = tuple(self.tokenizer.encode(raw, add_special_tokens=False))
        if self.tokenizer.decode(generated, skip_special_tokens=False) != raw:
            raise ValueError("fixture sample is not tokenization round-trip stable")
        messages = json.loads(prepared.messages_json)
        tools = json.loads(prepared.tools_json)
        if prepared.native_history is None:
            _prompt, input_ids = self.renderer.render_initial(messages, tools)
        else:
            suffix = self.renderer.external_suffix(prepared.native_history, messages)
            input_ids = (
                *prepared.native_history.input_token_ids,
                *prepared.native_history.generated_token_ids,
                *suffix,
            )
        model_ids = input_ids + generated[:-1]
        input_tensor = torch.tensor([model_ids], dtype=torch.long)
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.inference_mode():
                logits = self.model(
                    input_ids=input_tensor,
                    attention_mask=torch.ones_like(input_tensor),
                ).logits[0]
            start = len(input_ids) - 1
            selected_logits = logits[start : start + len(generated)].float()
            targets = torch.tensor(generated, dtype=torch.long)
            logprobs = torch.log_softmax(selected_logits, dim=-1).gather(1, targets[:, None])[:, 0]
        finally:
            for module, training in modes:
                module.training = training
        values = tuple(float(value) for value in logprobs.tolist())
        stop_id = generated[-1]
        termination = {"kind": "native_stop", "stop_token_id": stop_id, "limit": None}
        parsed = parse_native_response(
            self.tokenizer,
            raw,
            prefix=self.tokenizer.decode(input_ids, skip_special_tokens=False),
            action_id=prepared.action_id,
            termination=termination,
        )
        return SampleResultV2(
            message=parsed.message,
            input_token_ids=tuple(input_ids),
            generated_token_ids=generated,
            usage={
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(generated),
                "total_tokens": len(input_ids) + len(generated),
                "prefill_tokens": len(input_ids),
                "cached_input_tokens": 0,
            },
            logprobs=BinaryLogprobEvidence(
                struct.pack(f"<{len(values)}f", *values), "f32-le", (len(values),)
            ),
            termination=termination,
            sampling_pins={
                "manifest_ref": prepared.adapter_ref,
                "behavior_policy_ref": prepared.behavior_policy_ref,
                "decoding_ref": prepared.decoding_ref,
                "renderer_ref": self.manifest_descriptors[0].identity(),
                "seed": prepared.writer_seed,
            },
            raw_output=raw,
            trace={"native_parse_failed": True} if parsed.failed else None,
        )


def _make_run(
    root: Path,
    *,
    resume: Path | None = None,
    stop_after_steps: int | None = None,
    model_factory=None,
    sample_backend_factory=None,
    trainer_callback_factory=None,
    runtime_identity=None,
    all_tie: bool = False,
):
    recipe = settings()
    configs = load_probe_tasks()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH), local_files_only=True, trust_remote_code=False
    )
    if model_factory is None:

        def model_factory():
            return tiny_gemma(tokenizer.vocab_size)

    if sample_backend_factory is None:
        plans = fixture_plans(configs, recipe, all_tie=all_tie)

        def sample_backend_factory(*args, **kwargs):
            return ScriptedNativeBackend(*args, plans=plans, **kwargs)

    return make_task_graph_run(
        root,
        resume=resume,
        stop_after_steps=stop_after_steps,
        model_factory=model_factory,
        sample_backend_factory=sample_backend_factory,
        trainer_callback_factory=trainer_callback_factory,
        runtime_identity=runtime_identity,
        tokenizer=tokenizer,
        recipe=recipe,
    )


def _read_saved_run(root: Path):
    completes = sorted((root / "invocations").glob("*/complete.json"))
    if len(completes) != 1:
        raise ValueError("uninterrupted run must have exactly one completed invocation")
    result = json.loads(completes[0].read_text())
    observer = TaskGraphLossObserver(root)
    for batch_path in sorted((root / "batches").glob("step-*.json")):
        batch = json.loads(batch_path.read_text())
        observer.register_batch(batch["step"], batch["members"])
    for observer_path in sorted((root / "observer").glob("step-*.json")):
        observer.records.extend(json.loads(observer_path.read_text())["microbatches"])
    result["task_graph_observer"] = observer.verify()
    return result


def _same_state(left, right):
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.equal(right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def smoke(output: Path, *, uninterrupted_from: Path | None = None):
    import torch

    for key, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        if os.environ.get(key) != value:
            raise ValueError(f"required offline CPU environment: {key}={value!r}")
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    uninterrupted = (
        _read_saved_run(uninterrupted_from)
        if uninterrupted_from is not None
        else _make_run(output / "uninterrupted")
    )
    partial = _make_run(output / "resumed", stop_after_steps=2)
    resumed = _make_run(
        output / "resumed",
        resume=Path(partial["checkpoint"]),
    )
    if uninterrupted["global_step"] != 3 or resumed["global_step"] != 3:
        raise AssertionError("task-graph DAPO run did not reach global step 3")
    if partial["global_step"] != 2:
        raise AssertionError("stop boundary did not preserve checkpoint 2")
    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        left = torch.load(Path(uninterrupted["checkpoint"]) / filename, weights_only=False)
        right = torch.load(Path(resumed["checkpoint"]) / filename, weights_only=False)
        if not _same_state(left, right):
            raise AssertionError(f"resume changed {filename}")
    from safetensors.torch import load_file

    left_adapter = load_file(str(Path(uninterrupted["adapter"]) / "adapter_model.safetensors"))
    right_adapter = load_file(str(Path(resumed["adapter"]) / "adapter_model.safetensors"))
    if left_adapter.keys() != right_adapter.keys() or any(
        not left_adapter[key].equal(right_adapter[key]) for key in left_adapter
    ):
        raise AssertionError("resume changed the final PEFT adapter")
    if uninterrupted["task_graph_observer"]["steps"] != [0, 1, 2]:
        raise AssertionError("uninterrupted observer missed a training step")
    if resumed["task_graph_observer"]["steps"] != [2]:
        raise AssertionError("resumed observer did not capture the post-checkpoint step")
    report = {
        "dtype": "FP32 tiny Gemma4 on CPU",
        "group_size": 4,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 4,
        "optimizer_steps": 3,
        "stop_resume_boundary": 2,
        "exact_resume": ["adapter", "optimizer", "scheduler", "RNG"],
        "observer": uninterrupted["task_graph_observer"],
        "task_graph_checkpoints": {
            key: [
                verify_checkpoint(
                    Path(result["checkpoint"]).parent / f"checkpoint-{step}",
                    json.loads((Path(result["checkpoint"]).parent / "experiment.json").read_text())[
                        "identity"
                    ],
                )
                for step in range(1, 4)
            ]
            for key, result in (("uninterrupted", uninterrupted), ("resumed", resumed))
        },
    }
    save_json(output / "summary.json", report)
    save_json(output / "artifact-hashes.json", file_hashes(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--uninterrupted-from",
        type=Path,
        help="reuse a completed uninterrupted run while exercising fresh stop/resume runs",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            smoke(args.output, uninterrupted_from=args.uninterrupted_from)
            if args.execute
            else {"status": "inspect"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
