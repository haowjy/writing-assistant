"""Model, trace, and batch-measurement helpers for the S11 runner.

All model-stack imports are lazy so the CLI's argument and directory checks run in the
CI environment without torch or Transformers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import resource
import struct
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
MAX_TOKENS_PER_DECISION = 512
MAX_GENERATED_TOKENS = 1536
MAX_CONTEXT_TOKENS = 4096
TOKENIZER_ROOT = (
    Path.home() / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots" / MODEL_REVISION
)


def write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def offline_cpu_environment() -> None:
    # This script is a CPU-only, offline gate. Set these before importing torch/Transformers.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_builder():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "phase8" / "probe-tasks" / "build.py"
    spec = importlib.util.spec_from_file_location("phase8_probe_task_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load probe task builder at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def with_context_cap_for_local_model(entry, cap: int):
    """Give only the local-model test harness the declared cap while S9b is being ported.

    The normal E2B run refuses an entry that does not carry the configured cap; it never
    silently repairs production task admission at runtime.
    """
    from dataclasses import replace

    from tests.task_graph_fixtures import EntryFixture
    from writing_agent.task_graph_admission import MappingArtifactResolver, admit_graph
    from writing_agent.task_graph_derive_entry import derive_entry

    node = entry.graph.node(entry.node_id)
    budget = node.contract.budget_contract
    if budget.max_context_tokens == cap:
        return entry
    if budget.max_context_tokens is not None:
        raise ValueError("tiny fixture context cap differs from the probe settings")
    contract = replace(
        node.contract,
        budgets=replace(budget, max_context_tokens=cap),
    )
    entry.reader.public[contract.identity()] = contract.to_dict()
    node_spec = replace(node.spec, entry_contract=contract.identity())
    instance = replace(entry.graph.instance, nodes=(node_spec,))
    graph = admit_graph(
        instance,
        MappingArtifactResolver(entry.reader.public, entry.reader.private),
        policy=entry.graph.policy,
    )
    derived = derive_entry(graph, entry.node_id, entry.params, entry.reader)
    return EntryFixture(
        graph,
        entry.node_id,
        entry.params,
        entry.reader,
        derived.state,
        derived.artifacts,
    )


def with_native_tokenizer(entry, descriptor):
    """Bind the native tokenizer descriptor into a freshly admitted task entry."""
    from dataclasses import replace

    from tests.task_graph_fixtures import EntryFixture
    from writing_agent.task_graph_derive_entry import derive_entry

    entry.reader.public[descriptor.identity()] = descriptor.to_wire()
    rendering = {**entry.params.rendering, "tokenizer_ref": descriptor.identity()}
    params = replace(entry.params, rendering=rendering)
    derived = derive_entry(entry.graph, entry.node_id, params, entry.reader)
    return EntryFixture(
        entry.graph,
        entry.node_id,
        params,
        entry.reader,
        derived.state,
        derived.artifacts,
    )


def tokenizer_files(tokenizer_root: Path) -> dict[str, str]:
    names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    hashes = {}
    for name in names:
        data = (tokenizer_root / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def load_model_and_tokenizer(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tokenizer_root = args.tokenizer_root.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_root), local_files_only=True)
    if args.model_path is not None:
        model = Gemma4ForConditionalGeneration.from_pretrained(
            str(args.model_path.expanduser().resolve()),
            local_files_only=True,
            attn_implementation="eager",
        )
    else:
        model = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    model = model.to(device="cpu")
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
        ),
    )
    model.eval()

    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError("trace check refuses all CUDA devices; CPU only")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError("trace-check model contains a non-CPU parameter")
    return torch, model, tokenizer, None


def adapter_tensor_hash(model, torch) -> str:
    digest = hashlib.sha256()
    parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if "lora_" in name
    ]
    if not parameters:
        raise RuntimeError("native trace check requires the initialized PEFT LoRA adapter")
    for name, parameter in sorted(parameters, key=lambda item: item[0]):
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(struct.pack("<I", value.ndim))
        for dimension in value.shape:
            digest.update(struct.pack("<Q", int(dimension)))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class SampleTrace:
    def __init__(self) -> None:
        self.member_ordinal: int | None = None
        self.current: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []

    def begin_member(self, ordinal: int) -> None:
        self.member_ordinal = ordinal

    def backend(self, delegate):
        trace = self

        class TimedNativeBackend:
            descriptor = delegate.descriptor
            manifest_descriptors = delegate.manifest_descriptors

            def sample(self, prepared):
                event = {
                    "member_ordinal": trace.member_ordinal,
                    "decision_ordinal": prepared.decision_ordinal,
                    "action_id": prepared.action_id,
                    "input_tokens": None,
                    "prefill_tokens": None,
                    "generated_tokens": None,
                    "termination": None,
                    "generate_seconds": None,
                    "sample_seconds": None,
                    "context_delta_tokens": None,
                    "sample_error": None,
                    "tool_calls": [],
                }
                history = prepared.native_history
                if history is not None:
                    event["prefix_tokens"] = len(history.input_token_ids) + len(
                        history.generated_token_ids
                    )
                try:
                    started = time.perf_counter()
                    trace.current = event
                    result = delegate.sample(prepared)
                    event["input_tokens"] = len(result.input_token_ids)
                    event["prefill_tokens"] = int(result.usage.get("prefill_tokens", 0))
                    event["generated_tokens"] = len(result.generated_token_ids)
                    event["termination"] = dict(result.termination)
                    event["tool_calls"] = [
                        {
                            "id": call.get("id"),
                            "name": call.get("function", {}).get("name")
                            if isinstance(call.get("function"), Mapping)
                            else None,
                            "result": "missing",
                        }
                        for call in result.message.get("tool_calls", ())
                        if isinstance(call, Mapping)
                    ]
                    prefix = event.get("prefix_tokens", 0)
                    event["context_delta_tokens"] = event["input_tokens"] - prefix
                    return result
                except Exception as exc:
                    event["sample_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    raise
                finally:
                    event["sample_seconds"] = time.perf_counter() - started
                    trace.current = None
                    trace.events.append(event)

        return TimedNativeBackend()


def instrument_generation_time(model, trace: SampleTrace) -> None:
    original_generate = model.generate

    def timed_generate(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_generate(*args, **kwargs)
        finally:
            if trace.current is not None:
                trace.current["generate_seconds"] = time.perf_counter() - started

    model.generate = timed_generate


def trace_events_for_artifact(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert measurement floats to integer nanoseconds for canonical task-graph storage."""
    artifact_events = []
    for event in events:
        artifact_event = dict(event)
        for field in ("generate_seconds", "sample_seconds"):
            seconds = artifact_event.pop(field, None)
            artifact_event[field.removesuffix("_seconds") + "_nanoseconds"] = (
                None if seconds is None else round(seconds * 1_000_000_000)
            )
        artifact_events.append(artifact_event)
    return artifact_events


def decision_summaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep each sample's tool calls and resolved results in the human report."""
    return [
        {
            "member_ordinal": event["member_ordinal"],
            "decision_ordinal": event["decision_ordinal"],
            "generated_tokens": event["generated_tokens"],
            "termination": event["termination"],
            "generate_seconds": event["generate_seconds"],
            "tool_calls": event["tool_calls"],
        }
        for event in events
    ]


def native_policy(
    store, rendering, manifest, model_ref: str, adapter_hash: str, experiment_ref: str
):
    from writing_agent.task_graph_record_contracts import POLICY_FIELDS, ContextPolicyV1

    policy = {
        field: store.put_artifact({"pin": field})
        for field in POLICY_FIELDS
        if field != "rng_derivation_version"
    }
    policy.update(
        model_ref=model_ref,
        behavior_policy_ref=store.put_artifact(
            {
                "kind": "trace-check-behavior-policy-v1",
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "adapter_tensor_sha256": adapter_hash,
                "global_step": 0,
                "experiment_ref": experiment_ref,
            }
        ),
        tokenizer_ref=manifest.tokenizer.identity(),
        template_ref=rendering["template_ref"],
        adapter_ref=manifest.identity(),
        decoding_ref=manifest.decoding.identity(),
        context_policy_ref=store.put_artifact(
            ContextPolicyV1(
                "compact", summarizer_version="visible-text-v1", max_summary_chars=20
            ).to_wire()
        ),
        rng_derivation_version="sha256-domain-v1",
    )
    return policy


def start_receipt_ordinal(coordinator, spec, ordinal: int) -> str:
    return coordinator._start_receipt(spec, ordinal)["start_checkpoint_id"]


def member_summaries(
    store, coordinator, spec, decision, trace, admission=None, *, initial_files=None
):
    from writing_agent.task_graph_gate import LineageGate
    from writing_agent.task_graph_group_records import GroupMemberResultV1

    admission_by_id = {
        item["member_id"]: item["status"] for item in (admission.members if admission else ())
    }
    members = []
    for ordinal, member in enumerate(spec.members):
        result_ref = (
            decision.member_result_refs[ordinal]
            if ordinal < len(decision.member_result_refs)
            else None
        )
        result = (
            GroupMemberResultV1.from_dict(store.get_artifact(result_ref))
            if result_ref is not None
            else None
        )
        summary = {
            "ordinal": ordinal,
            "member_id": member.member_id,
            "execution_status": result.execution_status if result else "pending",
            "task_status": None,
            "stop_reason": None,
            "termination_classes": [
                event["termination"]["kind"]
                for event in trace.events
                if event["member_ordinal"] == ordinal and event["termination"] is not None
            ],
            "terminations": [
                event["termination"]
                for event in trace.events
                if event["member_ordinal"] == ordinal and event["termination"] is not None
            ],
            "reward": None,
            "eligibility": None,
            "admission": admission_by_id.get(member.member_id),
            "final_files_differ_from_initial": None,
            "changed_paths": [],
        }
        if result is not None and result.final_checkpoint_id is not None:
            view = LineageGate().view(store, result.final_checkpoint_id)
            initial = dict(initial_files or {})
            final = dict(view.state.files)
            changed_paths = sorted(
                path
                for path in initial.keys() | final.keys()
                if initial.get(path) != final.get(path)
            )
            summary["final_files_differ_from_initial"] = bool(changed_paths)
            summary["changed_paths"] = changed_paths
            tool_results = _tool_results_by_id(view.context.messages)
            for event in trace.events:
                if event["member_ordinal"] != ordinal:
                    continue
                for call in event["tool_calls"]:
                    call["result"] = tool_results.get(call["id"], "missing")
            summary["task_status"] = view.outcome.task_status
            summary["stop_reason"] = view.outcome.stop_reason
            summary["eligibility"] = view.outcome.training_eligibility
            reward = coordinator.reward_of(result, view=view)
            if reward is not None:
                summary["reward"] = {
                    "numerator": reward.numerator,
                    "denominator": reward.denominator,
                    "value": float(reward),
                }
        members.append(summary)
    return members


def _tool_results_by_id(messages) -> dict[str, str]:
    results = {}
    for message in messages:
        value = message.to_dict() if callable(getattr(message, "to_dict", None)) else message
        if not isinstance(value, Mapping) or value.get("role") != "tool":
            continue
        for part in value.get("content", ()):
            if not isinstance(part, Mapping) or part.get("type") != "tool_result":
                continue
            response = part.get("content")
            if isinstance(response, Mapping) and response.get("ok") is True:
                results[part["call_id"]] = "ok"
            elif isinstance(response, Mapping) and response.get("ok") is False:
                error = response.get("error")
                results[part["call_id"]] = error if isinstance(error, str) else "tool call failed"
            else:
                results[part["call_id"]] = "malformed tool result"
    return results


def tool_result_protocol_errors(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return protocol-shaped tool-result failures without mistaking model errors for them."""
    markers = (
        "duplicate tool call id",
        "unknown tool call",
        "unknown call",
        "no matching writer-call source",
        "tool result does not name the queued call",
        "tool result id mismatch",
        "tool result pairing",
        "call/result pairing",
        "pairing error",
        "task-graph tool result ids do not match",
        "malformed tool result",
    )
    failures = []
    for event in events:
        for call in event.get("tool_calls", ()):
            result = call.get("result")
            if isinstance(result, str) and (
                result == "missing" or any(marker in result.lower() for marker in markers)
            ):
                failures.append(
                    {
                        "member_ordinal": event.get("member_ordinal"),
                        "decision_ordinal": event.get("decision_ordinal"),
                        "name": call.get("name"),
                        "result": result,
                    }
                )
    return failures


def trace_completion_outcome(
    tool_errors: list[dict[str, Any]], *, inspector_identical: bool, incomplete_reason: str | None
):
    """Apply the terminal S11 verdict policy, with tool protocol errors taking precedence."""
    if tool_errors:
        return "halt", {
            "type": "ProtocolToolResultError",
            "message": "trace contains a protocol-shaped tool result error",
            "protocol_shape": "tool_result",
            "tool_result_errors": tool_errors,
        }
    if not inspector_identical:
        return "fail", {
            "type": "InspectorNonDeterminism",
            "message": "the two offline inspector outputs differ byte for byte",
            "protocol_shape": "non-protocol-infrastructure",
        }
    if incomplete_reason is not None:
        return "fail", {
            "type": "FixtureIncompleteSignal",
            "message": (
                f"both members ended incomplete for {incomplete_reason}; revise S10 before R3"
            ),
            "protocol_shape": "termination",
        }
    return "pass", None


def same_incomplete_reason(members: list[dict[str, Any]]) -> str | None:
    if len(members) != 2 or any(member["task_status"] != "incomplete" for member in members):
        return None
    reason = members[0]["stop_reason"]
    if reason is not None and members[1]["stop_reason"] == reason:
        return reason
    return None


def classify_protocol_shape(error: dict[str, Any]) -> str:
    values = [error.get(key, "") for key in ("type", "message", "failed_check", "reason_code")]
    failed_checks = error.get("failed_checks", ())
    if isinstance(failed_checks, (list, tuple, set)):
        values.extend(failed_checks)
    text = " ".join(map(str, values)).lower()
    if any(
        marker in text
        for marker in (
            "duplicate tool call id",
            "unknown tool call",
            "unknown call",
            "no matching writer-call source",
            "tool result does not name the queued call",
            "tool result id mismatch",
            "tool result pairing",
            "call/result pairing",
            "pairing error",
            "tool result ids do not match",
        )
    ):
        return "tool_result"
    if any(word in text for word in ("external_suffix", "suffix", "delta", "tool result")):
        return "delta"
    if any(word in text for word in ("termination", "stop token", "token_limit", "context_limit")):
        return "termination"
    if any(word in text for word in ("parse", "response grammar", "raw_output_and_message")):
        return "parse"
    return "non-protocol-infrastructure"


def prefill_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    prefill_sum = sum(int(event["prefill_tokens"] or 0) for event in events)
    unique_sum = 0
    for ordinal in sorted({event["member_ordinal"] for event in events}):
        member_events = [
            event
            for event in events
            if event["member_ordinal"] == ordinal and event["input_tokens"] is not None
        ]
        if not member_events:
            continue
        unique = int(member_events[0]["input_tokens"])
        previous_input = None
        previous_generated = 0
        for event in member_events:
            input_tokens = int(event["input_tokens"])
            if previous_input is not None:
                suffix_tokens = input_tokens - previous_input - previous_generated
                if suffix_tokens < 0:
                    raise ValueError("native sample input is shorter than its exact token prefix")
                unique += suffix_tokens
            generated = int(event["generated_tokens"] or 0)
            unique += generated
            previous_input = input_tokens
            previous_generated = generated
        unique_sum += unique
    return {
        "prefill_tokens": prefill_sum,
        "unique_ledger_tokens": unique_sum,
        "re_prefill_ratio": (prefill_sum / unique_sum) if unique_sum else None,
    }


def decode_u32(data: bytes) -> tuple[int, ...]:
    if len(data) % 4:
        raise ValueError("training token artifact is not u32-le aligned")
    return struct.unpack(f"<{len(data) // 4}I", data) if data else ()


def recompute_member_logprobs(
    model, torch, prompt: tuple[int, ...], completion: tuple[int, ...], mask: bytes
):
    """Recompute causal logprobs with bounded output chunks and a local-only KV cache."""
    if len(completion) != len(mask):
        raise ValueError("completion and environment mask lengths differ")
    if not prompt:
        raise ValueError("training prompt cannot be empty")
    device = torch.device("cpu")
    prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            use_cache=True,
        )
        cache = output.past_key_values
        next_logits = output.logits[0, -1].float()
        recomputed: list[float] = []
        chunk_size = 16
        for start in range(0, len(completion), chunk_size):
            chunk = completion[start : start + chunk_size]
            targets = torch.tensor(chunk, dtype=torch.long, device=device)
            first_logprobs = torch.log_softmax(next_logits, dim=-1).gather(0, targets[:1])
            scored = [float(first_logprobs[0].cpu())]
            if len(chunk) > 1:
                past_length = len(prompt) + start
                chunk_ids = targets.unsqueeze(0)
                attention = torch.ones(
                    (1, past_length + len(chunk)), dtype=torch.long, device=device
                )
                positions = torch.arange(
                    past_length, past_length + len(chunk), dtype=torch.long, device=device
                )
                output = model(
                    input_ids=chunk_ids,
                    attention_mask=attention,
                    past_key_values=cache,
                    cache_position=positions,
                    use_cache=True,
                )
                next_logits = output.logits[0, -1].float()
                cache = output.past_key_values
                logits_for_rest = output.logits[0, :-1].float()
                if logits_for_rest.shape[0] != len(chunk) - 1:
                    raise RuntimeError("model returned an incomplete chunk of recomputed logits")
                rest = torch.log_softmax(logits_for_rest, dim=-1).gather(
                    1, targets[1:].unsqueeze(1)
                )
                scored.extend(float(value) for value in rest.squeeze(1).cpu().tolist())
            else:
                # Feed the one sampled token to advance the cache and retain its next-token logit.
                past_length = len(prompt) + start
                chunk_ids = targets.unsqueeze(0)
                attention = torch.ones((1, past_length + 1), dtype=torch.long, device=device)
                positions = torch.tensor([past_length], dtype=torch.long, device=device)
                output = model(
                    input_ids=chunk_ids,
                    attention_mask=attention,
                    past_key_values=cache,
                    cache_position=positions,
                    use_cache=True,
                )
                next_logits = output.logits[0, -1].float()
                cache = output.past_key_values
            recomputed.extend(
                value
                for value, is_masked in zip(scored, mask[start : start + len(chunk)], strict=True)
                if is_masked
            )
    return recomputed


def on_policy_drift(store, batch, model, torch) -> dict[str, Any]:
    from writing_agent.task_graph_gate import StoreArtifactReader
    from writing_agent.task_graph_records import WriterTurnV2

    reader = StoreArtifactReader(store)
    differences: list[float] = []
    for member in batch.members:
        prompt = decode_u32(reader.bytes_artifact(member["prompt_ids_ref"]))
        completion = decode_u32(reader.bytes_artifact(member["completion_ids_ref"]))
        mask = reader.bytes_artifact(member["env_mask_ref"])
        sampled_at: dict[int, float] = {}
        for span in member["turn_spans"]:
            turn = WriterTurnV2.from_dict(reader.artifact(span["turn_ref"]))
            logprob_data = reader.bytes_artifact(turn.logprobs["ref"])
            count = turn.generated_token_count
            if len(logprob_data) != count * 4:
                raise ValueError("sampled logprob artifact shape differs from its turn")
            values = struct.unpack(f"<{count}f", logprob_data) if count else ()
            for offset, value in enumerate(values):
                sampled_at[span["completion_start"] + offset] = float(value)
        recomputed = recompute_member_logprobs(model, torch, prompt, completion, mask)
        sampled = [sampled_at[index] for index, is_masked in enumerate(mask) if is_masked]
        if len(sampled) != len(recomputed):
            raise ValueError("sampled and recomputed masked-token logprob counts differ")
        differences.extend(
            abs(left - right) for left, right in zip(sampled, recomputed, strict=True)
        )
    return {
        "masked_token_count": len(differences),
        "mean_abs_logprob_drift": sum(differences) / len(differences) if differences else None,
        "max_abs_logprob_drift": max(differences) if differences else None,
    }
