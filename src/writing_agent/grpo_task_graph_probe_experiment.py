"""Trainer composition for the Phase 8 probe experiment."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from writing_agent.grpo import GRPOSettings
from writing_agent.grpo_task_graph import TaskGraphTaskV1, train_task_graph
from writing_agent.task_graph_derive_entry import EntryV1, derive_entry
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_group_contract import derive_group_seed
from writing_agent.task_graph_probe_tasks import (
    CONFIG_DIR,
    EntryFixture,
    build_admitted_entry,
    load_probe_tasks,
    task_entries,
)
from writing_agent.task_graph_store import TaskGraphStore

TOKENIZER_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
TOKENIZER_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
    / TOKENIZER_REVISION
)


def bind_native_tokenizer(entry: EntryFixture, descriptor) -> EntryFixture:
    """Pin the tokenizer descriptor to an admitted task entry."""
    entry.reader.public[descriptor.identity()] = descriptor.to_wire()
    params = replace(
        entry.params,
        rendering={**entry.params.rendering, "tokenizer_ref": descriptor.identity()},
    )
    derived = derive_entry(entry.graph, entry.node_id, params, entry.reader)
    return replace(entry, params=params, state=derived.state, artifacts=derived.artifacts)


def settings() -> GRPOSettings:
    """Return the identity-bound probe recipe."""
    return GRPOSettings(
        model_id="google/gemma-4-E2B-it",
        revision=TOKENIZER_REVISION,
        runtime_profile="task-graph-v1",
        loss_type="dapo",
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
    """Create the small CPU Gemma used by the committed sampler composition check."""
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
    writes = [_tool_call("write_file", {"path": "scene.txt", "content": example["scene"]})]
    if example["notes"] is not None:
        writes.append(
            _tool_call("write_file", {"path": "field-notes.txt", "content": example["notes"]})
        )
    return (
        first + "<|tool_response>",
        "".join(writes) + "<|tool_response>",
        "The scene is ready for review.<turn|>",
    )


def fixture_plans(configs, recipe: GRPOSettings | None = None, *, all_tie: bool = False):
    """Return deterministic writer-seed-indexed messages used by CPU sampling."""
    recipe = settings() if recipe is None else recipe
    variants = (
        ("nonempty_only",) * 4,
        ("nonempty_only", "stated_detail", "phrase_and_detail", "all_optional"),
        ("all_optional",) * 4,
    )
    result = {}
    for step in range(recipe.max_steps):
        config = configs[step % len(configs)]
        choices = (
            ("nonempty_only",) * recipe.group_size if all_tie else variants[step % len(variants)]
        )
        group_seed = derive_group_seed(recipe.seed, "task-graph-step", step)
        for ordinal, lineage in enumerate(choices):
            writer_seed = derive_group_seed(group_seed, "writer", ordinal)
            result[writer_seed] = _fixture_actions(config, lineage)
    return result


def make_task_graph_run(
    root: Path,
    *,
    model_factory,
    sample_backend_factory,
    resume: Path | None = None,
    stop_after_steps: int | None = None,
    trainer_callback_factory=None,
    runtime_identity=None,
    tokenizer=None,
    tokenizer_root: Path = TOKENIZER_PATH,
    config_dir: Path = CONFIG_DIR,
    recipe: GRPOSettings | None = None,
):
    """Build, admit and train the shared P1 task group with injected model seams."""
    import torch
    from transformers import AutoTokenizer

    from writing_agent.native_gemma import NATIVE_STOP_TOKEN_IDS, make_native_manifest_descriptors
    from writing_agent.native_protocol import NATIVE_STOP_TOKENS

    torch.set_num_threads(2)
    tokenizer_root = tokenizer_root.expanduser().resolve()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_root), local_files_only=True, trust_remote_code=False
        )
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.parent.chmod(0o700)
    configs = load_probe_tasks(config_dir)
    seed_fixture = build_admitted_entry(configs[0])
    seed_derived: EntryV1 = derive_entry(
        seed_fixture.graph, seed_fixture.node_id, seed_fixture.params, seed_fixture.reader
    )
    recipe = settings() if recipe is None else recipe
    tokenizer_hashes = {
        name: hashlib.sha256((tokenizer_root / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    }
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
    task_entries_core = task_entries(
        store,
        gate,
        lambda entry: bind_native_tokenizer(entry, descriptors[1]),
        config_dir=config_dir,
    )
    entries = tuple(
        TaskGraphTaskV1(item.task_id, item.environment, item.entry_checkpoint_id)
        for item in task_entries_core
    )
    if (
        tuple(tokenizer.convert_tokens_to_ids(token) for token in NATIVE_STOP_TOKENS)
        != NATIVE_STOP_TOKEN_IDS
    ):
        raise ValueError("cached Gemma tokenizer does not match native stop tokens")
    if runtime_identity is None:
        runtime_identity = {
            "model": "gemma4-e2b-tiny-random-fp32-cpu-v1",
            "tokenizer_revision": TOKENIZER_REVISION,
            "fixture_tasks": [config["id"] for config in configs],
        }
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
        trainer_callback_factory=trainer_callback_factory,
        tokenizer_root=tokenizer_root,
    )


__all__ = [
    "TOKENIZER_PATH",
    "TOKENIZER_REVISION",
    "bind_native_tokenizer",
    "fixture_plans",
    "make_task_graph_run",
    "settings",
    "tiny_gemma",
]
