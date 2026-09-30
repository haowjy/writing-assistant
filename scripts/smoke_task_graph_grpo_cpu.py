"""CPU end-to-end task-graph DAPO resume check using tiny Gemma4 and public fixtures.

Run offline with the Phase 8 overlay and CUDA hidden. The scripted native sampler emits
bounded fixture actions, but computes each recorded logprob from the live tiny Gemma4
model; the production route defaults to NativeGemmaSampleBackend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from writing_agent.catalog import save_json
from writing_agent.grpo import GRPOSettings, file_hashes
from writing_agent.grpo_checkpoint import verify_checkpoint
from writing_agent.grpo_task_graph import (
    TaskGraphLossObserver,
    TaskGraphTaskV1,
    train_task_graph,
)
from writing_agent.inference import parse_response
from writing_agent.native_gemma import (
    NativeGemmaRenderer,
    NativeGemmaSampleBackend,
    make_native_manifest_descriptors,
)
from writing_agent.native_protocol import bind_native_tool_call_ids
from writing_agent.task_graph_derive_entry import derive_entry
from writing_agent.task_graph_environment import RolloutEnvironment
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_group_contract import derive_group_seed
from writing_agent.task_graph_ports import BinaryLogprobEvidence, SampleResultV2
from writing_agent.task_graph_store import TaskGraphStore

TOKENIZER_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
TOKENIZER_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
    / TOKENIZER_REVISION
)
BUILDER_PATH = Path(__file__).resolve().parents[1] / "configs/phase8/probe-tasks/build.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_BUILDER_SPEC = importlib.util.spec_from_file_location("p8_probe_task_builder", BUILDER_PATH)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_BUILDER_SPEC)
_BUILDER_SPEC.loader.exec_module(_BUILDER)


def settings() -> GRPOSettings:
    return GRPOSettings(
        model_id="google/gemma-4-E2B-it",
        revision=TOKENIZER_REVISION,
        runtime_profile="task-graph-v1",
        loss_type="dapo",
        scale_rewards="none",
        enable_thinking=False,
        group_size=4,
        microbatch_size=1,
        max_steps=3,
        context_tokens=4096,
        max_tokens=512,
        max_generated_tokens=1536,
        seed=123,
        gradient_checkpointing=True,
        gradient_checkpointing_use_reentrant=False,
    )


def tiny_gemma(vocab_size: int):
    import torch
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

    torch.manual_seed(123)
    text_config = Gemma4TextConfig(
        vocab_size=vocab_size,
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
        vocab_size_per_layer_input=vocab_size,
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
        Gemma4Config(
            text_config=text_config, tie_word_embeddings=False, attn_implementation="eager"
        )
    ).to(dtype=torch.float32, device="cpu")
    model.eval()
    return model


def _persist_entry(store: TaskGraphStore, entry) -> str:
    for body in entry.reader.public.values():
        store.put_artifact(body)
    for body in entry.reader.private.values():
        store.put_artifact(body, private=True)
    store.persist(entry.graph.instance)
    for artifact in entry.artifacts:
        store.persist_artifact(artifact)
    return store.save_checkpoint(entry.state)


def _bind_native_tokenizer(fixture, tokenizer_descriptor):
    tokenizer_ref = fixture.reader.add(tokenizer_descriptor.to_wire())
    if tokenizer_ref != tokenizer_descriptor.identity():
        raise ValueError("fixture tokenizer descriptor ref differs from its native identity")
    params = replace(
        fixture.params,
        rendering={**fixture.params.rendering, "tokenizer_ref": tokenizer_ref},
    )
    derived = derive_entry(fixture.graph, fixture.node_id, params, fixture.reader)
    return replace(fixture, params=params, state=derived.state, artifacts=derived.artifacts)


def task_entries(output: Path, store: TaskGraphStore, gate: LineageGate, tokenizer_descriptor):
    entries = []
    for config in _BUILDER.load_probe_tasks():
        fixture = _bind_native_tokenizer(
            _BUILDER.build_admitted_entry(config), tokenizer_descriptor
        )
        entry_checkpoint = _persist_entry(store, fixture)
        environment = RolloutEnvironment(store, fixture.graph, None, gate, fixture.graph.policy)
        entries.append(TaskGraphTaskV1(config["id"], environment, entry_checkpoint))
    return entries


def _quote(value: str) -> str:
    return f'<|"|>{value}<|"|>'


def _tool_call(name: str, arguments: dict[str, Any]) -> str:
    fields = []
    for key in sorted(arguments):
        value = arguments[key]
        if isinstance(value, str):
            rendered = _quote(value)
        elif isinstance(value, list):
            rendered = "[" + ",".join(_quote(item) for item in value) + "]"
        else:
            raise TypeError("fixture tool arguments use only strings and string arrays")
        fields.append(f"{key}:{rendered}")
    return f"<|tool_call>call:{name}{{{','.join(fields)}}}<tool_call|>"


def _fixture_actions(config: dict[str, Any], lineage: str) -> tuple[str, ...]:
    public = config["public"]
    example = config["scripted_lineages"][lineage]
    first = _tool_call(
        "write_file", {"path": "scene.txt", "content": public["initial_files"]["scene.txt"]}
    )
    decision = public["decision"]
    ask = _tool_call(
        "ask_author",
        {
            "question": decision["question"],
            "decision_ids": [decision["id"]],
            "proposals": [],
            "option_refs": [],
        },
    )
    writes = [_tool_call("write_file", {"path": "scene.txt", "content": example["scene"]})]
    if example["notes"] is not None:
        writes.append(
            _tool_call("write_file", {"path": "field-notes.txt", "content": example["notes"]})
        )
    return (
        first + "<|tool_response>",
        ask + "<|tool_response>",
        "".join(writes) + "<|tool_response>",
        "The scene is ready for review.<turn|>",
        "I applied the feedback revision.<turn|>",
    )


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
        return SampleResultV2(
            message=bind_native_tool_call_ids(
                parse_response(
                    self.tokenizer,
                    raw,
                    prefix=self.tokenizer.decode(input_ids, skip_special_tokens=False),
                ),
                prepared.action_id,
            ),
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
            termination={"kind": "native_stop", "stop_token_id": stop_id, "limit": None},
            sampling_pins={
                "manifest_ref": prepared.adapter_ref,
                "behavior_policy_ref": prepared.behavior_policy_ref,
                "decoding_ref": prepared.decoding_ref,
                "renderer_ref": self.manifest_descriptors[0].identity(),
                "seed": prepared.writer_seed,
            },
            raw_output=raw,
        )


def _plans(configs, settings_value, *, all_tie=False):
    variants = (
        ("nonempty_only",) * 4,
        ("nonempty_only", "decision_phrase", "phrase_and_detail", "all_optional"),
        ("all_optional",) * 4,
    )
    result = {}
    for step in range(settings_value.max_steps):
        config = configs[step % len(configs)]
        choices = (
            ("nonempty_only",) * settings_value.group_size
            if all_tie
            else variants[step % len(variants)]
        )
        group_seed = derive_group_seed(settings_value.seed, "task-graph-step", step)
        for ordinal, lineage in enumerate(choices):
            writer_seed = derive_group_seed(group_seed, "writer", ordinal)
            result[writer_seed] = _fixture_actions(config, lineage)
    return result


def _make_run(
    root: Path,
    *,
    resume: Path | None = None,
    stop_after_steps=None,
    model_factory=None,
    sample_backend_factory=None,
    runtime_identity=None,
    all_tie=False,
):
    import torch
    from transformers import AutoTokenizer

    from writing_agent.native_gemma import NATIVE_STOP_TOKEN_IDS
    from writing_agent.native_protocol import NATIVE_STOP_TOKENS

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH), local_files_only=True, trust_remote_code=False
    )
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.parent.chmod(0o700)
    configs = _BUILDER.load_probe_tasks()
    seed_fixture = _BUILDER.build_admitted_entry(configs[0])
    seed_derived = derive_entry(
        seed_fixture.graph, seed_fixture.node_id, seed_fixture.params, seed_fixture.reader
    )
    names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    tokenizer_hashes = {
        name: hashlib.sha256((TOKENIZER_PATH / name).read_bytes()).hexdigest() for name in names
    }
    recipe = settings()
    descriptors = make_native_manifest_descriptors(
        tokenizer,
        model_id=recipe.model_id,
        revision=recipe.revision,
        tokenizer_files_sha256=tokenizer_hashes,
        template_ref=seed_derived.view.context.rendering["template_ref"],
        tool_schema_ref=seed_derived.view.context.rendering["tool_schema_ref"],
        max_tokens_per_decision=recipe.max_tokens,
    )
    gate = LineageGate()
    store = TaskGraphStore(root, verifier=gate)
    entries = task_entries(root, store, gate, descriptors[1])
    if tuple(tokenizer.convert_tokens_to_ids(token) for token in NATIVE_STOP_TOKENS) != (
        NATIVE_STOP_TOKEN_IDS
    ):
        raise ValueError("cached Gemma tokenizer does not match native stop tokens")
    if runtime_identity is None:
        runtime_identity = {
            "model": "gemma4-e2b-tiny-random-fp32-cpu-v1",
            "tokenizer_revision": TOKENIZER_REVISION,
            "fixture_tasks": [config["id"] for config in configs],
        }
    if model_factory is None:

        def model_factory():
            return tiny_gemma(tokenizer.vocab_size)

    if sample_backend_factory is None:
        plans = _plans(configs, recipe, all_tie=all_tie)

        def sample_backend_factory(*args, **kwargs):
            return ScriptedNativeBackend(*args, plans=plans, **kwargs)

    return train_task_graph(
        entries,
        root,
        settings=recipe,
        model=None,
        model_factory=model_factory,
        tokenizer=tokenizer,
        manifest_descriptors=descriptors,
        runtime_identity=runtime_identity,
        resume_from_checkpoint=resume,
        stop_after_steps=stop_after_steps,
        sample_backend_factory=sample_backend_factory,
        tokenizer_root=TOKENIZER_PATH,
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
