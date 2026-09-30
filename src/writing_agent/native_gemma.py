"""Native Gemma rendering and same-model V2 sampling for task-graph rollouts.

The module itself stays importable without the model stack. Torch and Transformers are
loaded only when a sample is requested.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from writing_agent.inference import generate_with_seed, render_messages
from writing_agent.native_protocol import (
    NATIVE_STOP_TOKENS,
    ProtocolError,
    native_suffix,
    parse_native_response,
)
from writing_agent.task_graph import canonical_json
from writing_agent.task_graph_errors import AdapterContractError
from writing_agent.task_graph_group_contract import derive_group_seed
from writing_agent.task_graph_native_contracts import (
    NATIVE_TRAINING_CAPABILITIES,
    NativeSamplingHistory,
)
from writing_agent.task_graph_ports import (
    BinaryLogprobEvidence,
    PortDescriptorV1,
    PreparedSamplingInput,
    SampleResultV2,
)
from writing_agent.task_graph_records import (
    DecodingDescriptorV1,
    RendererDescriptorV1,
    TokenizerDescriptorV1,
)

NATIVE_STOP_TOKEN_IDS = (1, 106, 50)
NATIVE_CAPABILITIES = tuple(sorted(NATIVE_TRAINING_CAPABILITIES))


class NativeGemmaRenderer:
    """Append-only renderer: initial context once, then template-owned suffixes."""

    def __init__(self, tokenizer: Any, descriptor: RendererDescriptorV1) -> None:
        self.tokenizer = tokenizer
        self.descriptor = descriptor
        actual_stops = tuple(tokenizer.convert_tokens_to_ids(token) for token in NATIVE_STOP_TOKENS)
        if actual_stops != descriptor.stop_token_ids or actual_stops != NATIVE_STOP_TOKEN_IDS:
            raise ProtocolError("Tokenizer does not expose the pinned Gemma stop-token IDs")
        if descriptor.enable_thinking:
            raise ProtocolError("Native task-graph rendering disables thinking")

    def render_initial(
        self, messages: Sequence[Any], tools: Sequence[Mapping[str, Any]]
    ) -> tuple[str, tuple[int, ...]]:
        """Render the entry context; sampled assistant messages must never be replayed."""
        messages = _message_dicts(messages)
        gemma_messages = _gemma_messages(messages)
        if any(
            message.get("role") == "assistant" and message.get("loss_eligible")
            for message in messages
        ):
            raise ProtocolError("Initial native rendering cannot include a sampled assistant turn")
        prompt = self.tokenizer.apply_chat_template(
            render_messages(gemma_messages),
            tools=list(tools) or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        ids = tuple(int(token) for token in encoded["input_ids"][0].tolist())
        if not ids:
            raise ProtocolError("Gemma template produced an empty model input")
        return prompt, ids

    def external_suffix(
        self,
        history: NativeSamplingHistory,
        messages: Sequence[Any],
    ) -> tuple[int, ...]:
        """Continue from committed bytes plus the exact external-message suffix."""
        messages = _message_dicts(messages)
        positions = [
            index
            for index, message in enumerate(messages)
            if message.get("origin") == history.turn.action_id
            and message.get("role") == "assistant"
        ]
        if len(positions) != 1:
            raise ProtocolError("Committed assistant turn is not unique in the active context")
        position = positions[0]
        prior_message = messages[position]
        if not prior_message.get("loss_eligible"):
            raise ProtocolError("Committed writer action is not the active sampled assistant turn")
        assistant = _sampled_assistant(history.turn)
        external = _gemma_messages(messages[position + 1 :])
        sampled_call_ids = [call["id"] for call in assistant["tool_calls"]]
        tool_messages = [message for message in external if message.get("role") == "tool"]
        if len(tool_messages) != len(sampled_call_ids):
            raise ProtocolError("Task-graph tool results differ from sampled tool calls")
        result_call_ids = [message.get("tool_call_id") for message in tool_messages]
        if (
            len(set(sampled_call_ids)) != len(sampled_call_ids)
            or len(set(result_call_ids)) != len(result_call_ids)
            or set(result_call_ids) != set(sampled_call_ids)
        ):
            raise ProtocolError("Task-graph tool result IDs do not match sampled tool calls")
        suffix = native_suffix(
            self.tokenizer,
            assistant,
            external,
            history.generated_token_ids,
            thinking=False,
        )
        return tuple(int(token) for token in suffix)


def make_native_manifest_descriptors(
    tokenizer: Any,
    *,
    model_id: str,
    revision: str,
    tokenizer_files_sha256: Mapping[str, str],
    template_ref: str,
    tool_schema_ref: str,
    max_tokens_per_decision: int,
) -> tuple[RendererDescriptorV1, TokenizerDescriptorV1, DecodingDescriptorV1]:
    """Build the three immutable V2 descriptors for one pinned Gemma tokenizer."""
    if not tokenizer_files_sha256:
        raise ValueError("native tokenizer descriptor needs hashed source files")
    tokenizer_descriptor = TokenizerDescriptorV1(
        model_id=model_id,
        revision=revision,
        files_sha256=dict(tokenizer_files_sha256),
    )
    stop_ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in NATIVE_STOP_TOKENS)
    if stop_ids != NATIVE_STOP_TOKEN_IDS:
        raise ProtocolError("Tokenizer does not match the pinned Gemma stop-token set")
    renderer = RendererDescriptorV1(
        implementation="gemma4-native-append-v1",
        template_ref=template_ref,
        tokenizer_ref=tokenizer_descriptor.identity(),
        tool_schema_ref=tool_schema_ref,
        stop_token_ids=stop_ids,
        enable_thinking=False,
        suffix_rules_version="native-suffix-v1",
    )
    decoding = DecodingDescriptorV1(
        temperature=1,
        top_p=1,
        top_k=0,
        processors=(),
        max_tokens_per_decision=max_tokens_per_decision,
        seed_rule="writer_seed ⊕ action ordinal (sha256-domain-v1)",
        logprob_convention="log_softmax(model logits after model softcap), fp32",
        trainer_ratio="recomputed, num_iterations=1",
    )
    return renderer, tokenizer_descriptor, decoding


class _ObservationalLogitsProcessor:
    """Observe the selected token without retaining or changing vocabulary scores."""

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self._tokens: list[int] = []
        self._values: list[float] = []
        self._pending_logprobs = None
        self._finished = False

    def __call__(self, input_ids, scores):
        if (
            scores.ndim != 2
            or scores.shape[0] != 1
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or input_ids.shape[1] == 0
        ):
            raise ProtocolError("Native Gemma sampling is serial and single-sequence")
        if self._finished:
            raise ProtocolError("Observational processor cannot run after finish")
        if self._pending_logprobs is not None:
            self._record(int(input_ids[0, -1]), self._pending_logprobs)
        self._pending_logprobs = self._torch.log_softmax(scores[0].float(), dim=-1)
        return scores

    def _record(self, token_id: int, logprobs) -> None:
        if token_id < 0 or token_id >= logprobs.shape[0]:
            raise ProtocolError("Generated token is outside the observed logit row")
        self._tokens.append(token_id)
        self._values.append(float(logprobs[token_id].detach().cpu()))

    def finish(self, generated_ids: tuple[int, ...]) -> tuple[float, ...]:
        if self._finished:
            raise ProtocolError("Observational processor finish may only run once")
        if not generated_ids:
            if self._pending_logprobs is not None or self._tokens:
                raise ProtocolError("Observational processor disagrees with native sampled tokens")
            self._finished = True
            return ()
        if (
            self._pending_logprobs is None
            or len(generated_ids) != len(self._tokens) + 1
            or generated_ids[:-1] != tuple(self._tokens)
        ):
            raise ProtocolError("Observational processor disagrees with native sampled tokens")
        self._record(generated_ids[-1], self._pending_logprobs)
        self._pending_logprobs = None
        self._finished = True
        return tuple(self._values)


def assert_active_adapter(model: Any, name: str) -> None:
    """Require exactly one enabled active PEFT adapter before native sampling."""
    if not isinstance(name, str) or not name:
        raise AdapterContractError("Native sampling requires a named PEFT adapter")
    active = getattr(model, "active_adapters", None)
    active = active() if callable(active) else active
    if active is None:
        active = getattr(model, "active_adapter", None)
        active = active() if callable(active) else active
    if isinstance(active, str):
        active = (active,)
    if not isinstance(active, (tuple, list)) or tuple(active) != (name,):
        raise AdapterContractError("Expected PEFT adapter is not active for native sampling")
    modules = model.modules() if callable(getattr(model, "modules", None)) else ()
    if any(getattr(module, "disable_adapters", False) is True for module in modules):
        raise AdapterContractError("Native sampling refuses a disabled PEFT adapter")


class NativeGemmaSampleBackend:
    """Task-graph sampler over live PEFT weights with no cross-call token/KV state."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        manifest_descriptors: tuple[
            RendererDescriptorV1, TokenizerDescriptorV1, DecodingDescriptorV1
        ],
        adapter_name: str = "default",
        model_ref: str | None = None,
    ) -> None:
        if (
            not isinstance(manifest_descriptors, tuple)
            or len(manifest_descriptors) != 3
            or not isinstance(manifest_descriptors[0], RendererDescriptorV1)
            or not isinstance(manifest_descriptors[1], TokenizerDescriptorV1)
            or not isinstance(manifest_descriptors[2], DecodingDescriptorV1)
        ):
            raise TypeError("native backend requires renderer, tokenizer, and decoding descriptors")
        renderer, tokenizer_descriptor, decoding = manifest_descriptors
        if decoding.processors:
            raise ProtocolError("Native Gemma requires neutral decoding processors")
        self.model = model
        self.tokenizer = tokenizer
        self.renderer = NativeGemmaRenderer(tokenizer, renderer)
        self.manifest_descriptors = manifest_descriptors
        self.adapter_name = adapter_name
        self.model_ref = model_ref
        configuration = {"adapter_name": adapter_name}
        if model_ref is not None:
            configuration["model_ref"] = model_ref
        self.descriptor = PortDescriptorV1(
            role="sampling",
            implementation="gemma4-native-sample-v1",
            version="1",
            configuration_json=canonical_json(configuration),
            capabilities=NATIVE_CAPABILITIES,
        )
        if tokenizer_descriptor.identity() != renderer.tokenizer_ref:
            raise ProtocolError("Native renderer and tokenizer descriptors disagree")

    def sample(self, prepared: PreparedSamplingInput) -> SampleResultV2:
        import torch
        from transformers import LogitsProcessorList

        assert_active_adapter(self.model, self.adapter_name)
        renderer, tokenizer_descriptor, decoding = self.manifest_descriptors
        for reference, expected, name in (
            (prepared.tokenizer_ref, tokenizer_descriptor.identity(), "tokenizer"),
            (prepared.template_ref, renderer.template_ref, "template"),
            (prepared.decoding_ref, decoding.identity(), "decoding"),
        ):
            if reference != expected:
                raise AdapterContractError(f"Native sampler {name} differs from the sealed pin")
        if prepared.adapter_ref is None or prepared.behavior_policy_ref is None:
            raise AdapterContractError("Native sampler requires sealed adapter and policy refs")
        if self.model_ref is not None and prepared.model_ref != self.model_ref:
            raise AdapterContractError("Native sampler model differs from the sealed pin")
        if prepared.writer_seed is None or prepared.decision_ordinal is None:
            raise AdapterContractError("Native sampler requires a writer seed and decision ordinal")
        if prepared.action_id is None:
            raise AdapterContractError("Native sampler requires the committed writer action ID")
        if prepared.native_sampling_budget is None:
            raise AdapterContractError(
                "Native sampler requires the derive-provided sampling budget"
            )

        messages = json.loads(prepared.messages_json)
        tools = json.loads(prepared.tools_json)
        if prepared.native_history is None:
            _prompt, input_ids = self.renderer.render_initial(messages, tools)
        else:
            input_ids = (
                *prepared.native_history.input_token_ids,
                *prepared.native_history.generated_token_ids,
                *self.renderer.external_suffix(prepared.native_history, messages),
            )
        if not input_ids:
            raise ProtocolError("Native Gemma input token ledger is empty")

        budget = prepared.native_sampling_budget
        context_remaining = (
            float("inf")
            if budget.max_context_tokens is None
            else budget.max_context_tokens - len(input_ids)
        )
        generated_remaining = (
            float("inf")
            if budget.remaining_generated_tokens is None
            else budget.remaining_generated_tokens
        )
        terms = (
            ("decision", decoding.max_tokens_per_decision),
            ("generated_budget", generated_remaining),
            ("context", context_remaining),
        )
        allowed = min(value for _, value in terms)
        limit = next(name for name, value in terms if value == allowed)

        if allowed <= 0:
            if limit != "context":
                raise AdapterContractError("Native sampler was invoked after its generated budget")
            return self._result(
                prepared,
                input_ids,
                (),
                b"",
                {"kind": "context_limit", "stop_token_id": None, "limit": "context"},
                {
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": 0,
                    "total_tokens": len(input_ids),
                    "prefill_tokens": len(input_ids),
                    "cached_input_tokens": 0,
                },
                {"role": "assistant", "content": "", "tool_calls": []},
            )

        self._require_neutral_generation_defaults()
        stop_ids = renderer.stop_token_ids
        observer = _ObservationalLogitsProcessor(torch)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        generation = {
            "max_new_tokens": allowed,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "eos_token_id": list(stop_ids),
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "cache_implementation": "dynamic",
            "num_return_sequences": 1,
            "num_beams": 1,
            "return_dict_in_generate": True,
            "output_scores": False,
            "logits_processor": LogitsProcessorList([observer]),
        }
        seed = derive_group_seed(prepared.writer_seed, "decision", prepared.decision_ordinal)
        generated = generate_with_seed(
            self.model,
            {"input_ids": input_tensor, "attention_mask": torch.ones_like(input_tensor)},
            generation,
            seed=seed,
        )
        output_ids = tuple(
            int(token) for token in generated.sequences[0, len(input_ids) :].tolist()
        )
        values = observer.finish(output_ids)
        logprobs = tuple_to_f32(values)
        stops = set(stop_ids)
        stop_positions = [index for index, token in enumerate(output_ids) if token in stops]
        if stop_positions:
            if stop_positions != [len(output_ids) - 1]:
                raise ProtocolError("Generation continued after a pinned Gemma stop token")
            termination = {
                "kind": "native_stop",
                "stop_token_id": output_ids[-1],
                "limit": None,
            }
        elif len(output_ids) == allowed:
            termination = {"kind": "token_limit", "stop_token_id": None, "limit": limit}
        else:
            raise ProtocolError("Generation ended before a stop token or committed token limit")

        raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        parsed = parse_native_response(
            self.tokenizer,
            raw_output,
            prefix=prompt,
            action_id=prepared.action_id,
            termination=termination,
        )
        usage = {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(output_ids),
            "total_tokens": len(input_ids) + len(output_ids),
            "prefill_tokens": len(input_ids),
            "cached_input_tokens": 0,
        }
        return self._result(
            prepared,
            input_ids,
            output_ids,
            logprobs,
            termination,
            usage,
            parsed.message,
            raw_output,
            trace={"native_parse_failed": True} if parsed.failed else None,
        )

    def _result(
        self,
        prepared: PreparedSamplingInput,
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        logprobs: bytes,
        termination: Mapping[str, Any],
        usage: Mapping[str, Any],
        message: Mapping[str, Any],
        raw_output: str | None = None,
        trace: Mapping[str, Any] | None = None,
    ) -> SampleResultV2:
        renderer, _tokenizer, decoding = self.manifest_descriptors
        return SampleResultV2(
            message=message,
            input_token_ids=tuple(input_ids),
            generated_token_ids=tuple(output_ids),
            usage=usage,
            logprobs=BinaryLogprobEvidence(logprobs, "f32-le", (len(output_ids),)),
            termination=termination,
            sampling_pins={
                "manifest_ref": prepared.adapter_ref,
                "behavior_policy_ref": prepared.behavior_policy_ref,
                "decoding_ref": decoding.identity(),
                "renderer_ref": renderer.identity(),
                "seed": prepared.writer_seed,
            },
            raw_output=raw_output,
            trace=trace,
        )

    def _require_neutral_generation_defaults(self) -> None:
        config = getattr(self.model, "generation_config", None)
        if config is None:
            return
        neutral_values = {
            # Transformers 5 represents unset generation controls with None; older releases
            # used the equivalent no-op values below.
            "min_length": (0, None),
            "min_new_tokens": (None,),
            "min_p": (None,),
            "typical_p": (1.0, None),
            "repetition_penalty": (1.0, None),
            "no_repeat_ngram_size": (0, None),
            "forced_bos_token_id": (None,),
            "forced_eos_token_id": (None,),
            "suppress_tokens": (None,),
            "begin_suppress_tokens": (None,),
            "bad_words_ids": (None,),
            "sequence_bias": (None,),
            "exponential_decay_length_penalty": (None,),
        }
        for name, allowed in neutral_values.items():
            actual = getattr(config, name, allowed[0])
            if actual not in allowed:
                raise ProtocolError(f"Unexpected non-neutral generation setting: {name}")


def tuple_to_f32(values: Sequence[float]) -> bytes:
    import struct

    return struct.pack(f"<{len(values)}f", *values) if values else b""


def _gemma_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ProtocolError("Native renderer requires a message sequence")
    rendered = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ProtocolError("Native renderer received a malformed task-graph message")
        role = message.get("role")
        parts = message.get("content", ())
        if not isinstance(parts, (tuple, list)):
            raise ProtocolError("Task-graph message content must be an array")
        text, calls, results = [], [], []
        for part in parts:
            if isinstance(part, str):
                text.append(part)
            elif isinstance(part, Mapping) and part.get("type") == "text":
                text.append(part["text"])
            elif isinstance(part, Mapping) and part.get("type") == "tool_call":
                calls.append(
                    {
                        "id": part["id"],
                        "type": "function",
                        "function": {"name": part["name"], "arguments": dict(part["arguments"])},
                    }
                )
            elif isinstance(part, Mapping) and part.get("type") == "tool_result":
                results.append(part)
            else:
                raise ProtocolError("Unsupported task-graph message part in native Gemma input")

        if role == "tool":
            if len(results) != 1 or calls or text:
                raise ProtocolError("Native tool messages require one typed tool result")
            part = results[0]
            rendered.append(
                {
                    "role": "tool",
                    "tool_call_id": part["call_id"],
                    "content": json.dumps(
                        part["content"], ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
        else:
            if results:
                raise ProtocolError("Tool results must use the task-graph tool role")
            item = {"role": role, "content": "".join(text)}
            if calls:
                if role != "assistant":
                    raise ProtocolError("Only assistant messages may contain tool calls")
                item["tool_calls"] = calls
            rendered.append(item)
    return rendered


def _message_dicts(messages: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ProtocolError("Native renderer requires a message sequence")
    result = []
    for message in messages:
        if hasattr(message, "to_dict"):
            message = message.to_dict()
        if not isinstance(message, Mapping):
            raise ProtocolError("Native renderer received a malformed task-graph message")
        result.append(message)
    return tuple(result)


def _sampled_assistant(turn) -> dict[str, Any]:
    from writing_agent.task_graph_wire import decode_canonical_value

    content = decode_canonical_value(turn.message.content)
    if content is not None and not isinstance(content, str):
        raise ProtocolError("Committed native assistant content is not text")
    calls = []
    if turn.message.tool_calls_was_list:
        if not isinstance(turn.message.calls, (tuple, list)):
            raise ProtocolError("Committed native tool calls are malformed")
        for item in turn.message.calls:
            if not isinstance(item, Mapping) or item.get("bounded") is not True:
                raise ProtocolError("Committed native tool call exceeds its evidence bound")
            call = decode_canonical_value(item["value"])
            if (
                not isinstance(call, Mapping)
                or not isinstance(call.get("function"), Mapping)
                or not isinstance(call["function"].get("name"), str)
            ):
                raise ProtocolError("Committed native tool call is malformed")
            calls.append(dict(call))
    return {"role": "assistant", "content": content or "", "tool_calls": calls}


__all__ = [
    "NATIVE_CAPABILITIES",
    "NATIVE_STOP_TOKEN_IDS",
    "NativeGemmaRenderer",
    "NativeGemmaSampleBackend",
    "make_native_manifest_descriptors",
]
