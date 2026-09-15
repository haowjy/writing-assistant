"""Explicit SFT preparation and QLoRA execution; imports never load model weights."""

import copy
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from writing_agent.catalog import fingerprint, save_json
from writing_agent.data import validate_records
from writing_agent.inference import checkpoint_identity, render_messages


@dataclass(frozen=True)
class SFTSettings:
    model_id: str = "google/gemma-4-E2B-it"
    revision: str = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    max_length: int = 2048
    max_steps: int = 20
    batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    lora_rank: int = 16
    seed: int = 42


def training_template(template: str) -> str:
    """Annotate Gemma native outputs without supervising embedded tool observations."""
    call_start = "{%- for tool_call in message.get('tool_calls') -%}"
    call_end = "{%- set ns.prev_message_type = 'tool_call' -%}"
    turn_end = "{{- '<turn|>\\n' -}}"
    ending = "{%- elif not (ns_tr_out.flag and not has_content and not next_nt.found) -%}"
    ending += "\n            "
    content = "{{- captured_content -}}"

    def assistant_span(text):
        return (
            "{% if role == 'model' %}{% generation %}"
            + text
            + "{% endgeneration %}{% else %}"
            + text
            + "{% endif %}"
        )

    replacements = {
        call_start: "{% generation %}" + call_start,
        call_end: "{% endgeneration %}" + call_end,
        content: assistant_span(content),
        ending + turn_end: ending + assistant_span(turn_end),
    }
    for original, replacement in replacements.items():
        if template.count(original) != 1:
            raise ValueError("Unsupported chat template; review native loss-mask annotations")
        template = template.replace(original, replacement)
    return template


def encode_trajectory(record: dict, tokenizer, *, max_length: int) -> dict:
    """Return audited native tokens/labels. Reject overflow instead of cutting targets."""
    validate_records([record])
    for message in record["messages"]:
        if any(key in message for key in ("thinking", "reasoning", "reasoning_content")):
            raise ValueError("Reasoning supervision requires a separately reviewed policy")
    # Reject literal protocol tokens in source material before template rendering.
    source = json.dumps({"messages": record["messages"], "tools": record.get("tools", [])})
    if any(token in source for token in tokenizer.all_special_tokens):
        raise ValueError("Trajectory contains literal special tokens")
    messages = copy.deepcopy(record["messages"])
    for message in messages:
        if message["role"] == "tool":
            try:
                json.loads(message["content"])
            except json.JSONDecodeError:
                message["content"] = json.dumps(message["content"])
    messages = render_messages(messages)
    kwargs = dict(tools=record.get("tools", []), add_generation_prompt=False, enable_thinking=True)
    original = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    template = training_template(tokenizer.chat_template)
    annotated = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False, **kwargs
    )
    if original != annotated:
        raise ValueError("Loss annotations changed the native conversation")
    encoded = tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    ids, mask = encoded["input_ids"], encoded["assistant_masks"]
    if len(ids) > max_length:
        raise ValueError(f"{record['id']}: {len(ids)} tokens exceed {max_length}; no truncation")
    if len(ids) != len(mask) or not any(mask):
        raise ValueError("Missing or invalid assistant loss mask")
    return {
        "id": record["id"],
        "split": record["split"],
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [token if selected else -100 for token, selected in zip(ids, mask, strict=True)],
        "rendered": original,
        "supervised_tokens": sum(mask),
        "supervised_text": tokenizer.decode(
            [t for t, selected in zip(ids, mask, strict=True) if selected]
        ),
        "record_hash": fingerprint(record),
        "template_hash": fingerprint(template),
    }


def prepare_sft(
    records: list[dict],
    tokenizer,
    destination: Path,
    *,
    settings: SFTSettings,
    excluded_source_groups: set[str],
) -> dict:
    """Prepare accepted train/validation records with explicit source-group exclusions."""
    validate_records(records)
    selected = [
        r
        for r in records
        if r["review_status"] == "accepted" and r["split"] in {"train", "validation"}
    ]
    if not any(r["split"] == "train" for r in selected):
        raise ValueError("No accepted training records; fixtures are not production training data")
    groups = {}
    rows = []
    for record in selected:
        sources = record.get("source_groups")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(isinstance(g, str) and g for g in sources)
        ):
            raise ValueError(f"{record['id']}: explicit source_groups required")
        if set(sources) & excluded_source_groups:
            raise ValueError(f"{record['id']}: evaluation source group in training collection")
        for group in sources:
            if group in groups and groups[group] != record["split"]:
                raise ValueError(f"{record['id']}: related source crosses train/validation")
            groups[group] = record["split"]
        rows.append(encode_trajectory(record, tokenizer, max_length=settings.max_length))
    identity = {
        "settings": asdict(settings),
        "records": [r["record_hash"] for r in rows],
        "excluded_source_groups": sorted(excluded_source_groups),
        "template_hash": rows[0]["template_hash"],
        "tokenizer_hash": fingerprint(tokenizer.backend_tokenizer.to_str()),
    }
    manifest = {
        **identity,
        "preparation_hash": fingerprint(identity),
        "examples": len(rows),
        "tokens": sum(len(r["input_ids"]) for r in rows),
        "supervised_tokens": sum(r["supervised_tokens"] for r in rows),
        "rows_hash": fingerprint(rows),
    }
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "manifest.json", manifest)
    save_json(destination / "trajectories.json", selected)
    save_json(destination / "encoded.json", rows)
    tokenizer.save_pretrained(destination / "tokenizer")
    return manifest


def train_sft(
    prepared: Path,
    output: Path,
    *,
    settings: SFTSettings,
    execute: bool = False,
    resume_from_checkpoint: Path | None = None,
) -> dict:
    """Train only after explicit execution; resume binds to the same prepared experiment."""
    manifest = json.loads((prepared / "manifest.json").read_text())
    rows = json.loads((prepared / "encoded.json").read_text())
    if manifest["settings"] != asdict(settings) or manifest["rows_hash"] != fingerprint(rows):
        raise ValueError("Prepared data or training configuration changed")
    checkpoint_identity(settings.model_id, settings.revision)
    if (
        min(
            settings.max_steps,
            settings.batch_size,
            settings.gradient_accumulation,
            settings.lora_rank,
            settings.max_length,
        )
        < 1
        or settings.learning_rate <= 0
    ):
        raise ValueError("Training settings must be positive")
    plan = {
        "settings": asdict(settings),
        "preparation_hash": manifest["preparation_hash"],
        "output": str(output.resolve()),
        "packing": False,
        "loss": "assistant content, native tool calls, assistant turn endings",
        "evaluation": "separate explicit saved-checkpoint evaluation; no automatic benchmarks",
    }
    if not execute:
        return plan
    if resume_from_checkpoint is None:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "experiment.json", plan)
    elif (
        json.loads((output / "experiment.json").read_text()) != plan
        or not resume_from_checkpoint.resolve().is_relative_to(output.resolve())
        or not (resume_from_checkpoint / "trainer_state.json").is_file()
    ):
        raise ValueError("Resume checkpoint must belong to this training experiment")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA experiment")
    set_seed(settings.seed)
    tokenizer = AutoTokenizer.from_pretrained(prepared / "tokenizer", local_files_only=True)
    if fingerprint(tokenizer.backend_tokenizer.to_str()) != manifest["tokenizer_hash"]:
        raise ValueError("Prepared tokenizer changed")
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        local_files_only=True,
        trust_remote_code=False,
        device_map={"": "cuda:0"},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=settings.lora_rank,
            lora_alpha=2 * settings.lora_rank,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
        ),
    )
    columns = ("input_ids", "attention_mask", "labels")
    datasets = {
        split: Dataset.from_list([{k: r[k] for k in columns} for r in rows if r["split"] == split])
        for split in ("train", "validation")
    }
    args = SFTConfig(
        output_dir=str(output),
        max_steps=settings.max_steps,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings.gradient_accumulation,
        learning_rate=settings.learning_rate,
        seed=settings.seed,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        packing=False,
        padding_free=False,
        max_length=settings.max_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        assistant_only_loss=False,  # Prepared labels already carry the assistant mask.
        completion_only_loss=False,
        loss_type="chunked_nll",
        report_to="none",
        logging_steps=1,
        save_steps=10,
        save_total_limit=3,
        eval_strategy="no",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"] if len(datasets["validation"]) else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
    )
    save_json(
        output / "runtime.json",
        {
            "packages": {
                p: version(p) for p in ("torch", "transformers", "trl", "peft", "bitsandbytes")
            },
            "gpu": torch.cuda.get_device_name(0),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
    )
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
    )
    trainer.save_model(str(output / "adapter"))
    trainer.save_state()
    tokenizer.save_pretrained(output / "adapter")
    metrics = {
        **result.metrics,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "validation": trainer.evaluate() if len(datasets["validation"]) else None,
    }
    save_json(output / "metrics.json", metrics)
    return metrics
