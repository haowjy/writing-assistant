"""Local, serial Transformers inference through the shared agent harness.

Optional dependencies are imported only when loading or executing a model.
"""

import copy
import hashlib
import json
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

from writing_agent.backends import Completion
from writing_agent.catalog import fingerprint
from writing_agent.suite import run_selected

PROTOCOL = "writing-tools-v2"
TOOL_INSTRUCTION = """Reply to the user in ordinary text, never in a JSON response wrapper.
Only when calling tools, return exactly one JSON object:
{"tool_calls": [{"name": "tool_name", "arguments": {"parameter": "value"}}]}
Do not wrap tool calls in Markdown or mix them with a conversational reply.
Tool results appear in subsequent messages. After using tools, reply in ordinary text.
Available tools: """


def checkpoint_identity(source: str, revision: str | None = None) -> dict:
    """Hash a completed local save, or require an immutable Hub commit."""
    path = Path(source)
    if path.is_dir():
        files = {}
        for item in sorted(path.rglob("*")):
            if item.is_file():
                with item.open("rb") as stream:
                    files[str(item.relative_to(path))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        if not files:
            raise ValueError("Checkpoint directory is empty")
        return {"id": str(path.resolve()), "revision": fingerprint(files)}
    if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Hub checkpoints require a full immutable commit hash")
    return {"id": source, "revision": revision}


def render_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Represent calls/results as ordinary text for the versioned JSON protocol."""
    rendered = []
    for message in messages:
        role = message["role"]
        if message.get("tool_calls"):
            content = json.dumps(
                {
                    "tool_calls": [
                        {
                            "name": c["function"]["name"],
                            "arguments": (
                                json.loads(c["function"]["arguments"])
                                if isinstance(c["function"]["arguments"], str)
                                else c["function"]["arguments"]
                            ),
                        }
                        for c in message["tool_calls"]
                    ]
                },
                ensure_ascii=False,
            )
        elif role == "tool":
            role = "user"
            content = "Tool result " + message.get("tool_call_id", "") + ": " + message["content"]
        else:
            content = message.get("content", "")
        if tools and message["role"] == "system":
            content += "\n\n" + TOOL_INSTRUCTION + json.dumps(tools, ensure_ascii=False)
        # Some model templates require strictly alternating user/assistant roles.
        if rendered and rendered[-1]["role"] == role:
            rendered[-1]["content"] += "\n\n" + content
        else:
            rendered.append({"role": role, "content": content})
    if tools and not any(m["role"] == "system" for m in rendered):
        rendered.insert(0, {"role": "system", "content": TOOL_INSTRUCTION + json.dumps(tools)})
    return rendered


def parse_response(text: str, *, tools: bool) -> dict:
    reply = {"role": "assistant", "content": text}
    if not tools or not text.lstrip().startswith("{"):
        return reply
    try:
        value = json.loads(text)
    except ValueError as exc:
        if '"tool_calls"' in text:
            raise ValueError(f"Invalid {PROTOCOL} tool call: {text}") from exc
        return reply
    if not isinstance(value, dict) or "tool_calls" not in value:
        return reply
    try:
        calls = value.get("tool_calls")
        if set(value) != {"tool_calls"} or not isinstance(calls, list) or not calls:
            raise ValueError("Expected nonempty tool_calls")
        for call in calls:
            if (
                not isinstance(call, dict)
                or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str)
                or not isinstance(call["arguments"], dict)
            ):
                raise ValueError("Each call requires a name and argument object")
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                }
                for i, call in enumerate(calls)
            ],
        }
    except (ValueError, TypeError) as exc:
        # Preserve malformed output in the attempt error for protocol diagnostics.
        raise ValueError(f"Invalid {PROTOCOL} response: {text}") from exc


class TransformersBackend:
    """A session over caller-owned weights; no conversation or KV cache persists."""

    def __init__(self, model, tokenizer, config: dict):
        if config["protocol"] != PROTOCOL:
            raise ValueError("Unsupported local protocol")
        if config["prompt_format"] not in {"chat", "transcript"}:
            raise ValueError("Choose chat or transcript formatting explicitly")
        if config["max_tokens"] < 1 or config["context_tokens"] < 1:
            raise ValueError("Token budgets must be positive")
        if config["temperature"] < 0 or not 0 < config["top_p"] <= 1:
            raise ValueError("Invalid sampling configuration")
        self.model, self.tokenizer = model, tokenizer
        self.config = copy.deepcopy(config)
        self.calls = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        import torch

        rendered = render_messages(messages, tools)
        if self.config["prompt_format"] == "chat":
            prompt = self.tokenizer.apply_chat_template(
                rendered, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        else:
            prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in rendered)
            inputs = self.tokenizer(prompt + "\n\nASSISTANT:\n", return_tensors="pt")
        inputs = inputs.to(self.model.device)
        input_tokens = inputs["input_ids"].shape[-1]
        limit = self.config["max_tokens"]
        if input_tokens + limit > self.config["context_tokens"]:
            raise ValueError("Context budget exceeded; history was not truncated")
        temperature = self.config["temperature"]
        generation = {
            "max_new_tokens": limit,
            "do_sample": temperature > 0,
            "use_cache": True,
            "cache_implementation": "dynamic",
            "return_dict_in_generate": False,
            "num_return_sequences": 1,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation.update(temperature=temperature, top_p=self.config["top_p"])
        device = self.model.device
        devices = [device.index or 0] if device.type == "cuda" else []
        # Keep a training caller's RNG and per-module train/eval modes intact.
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.random.default_generator.manual_seed(self.config["seed"] + self.calls)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(
                        self.config["seed"] + self.calls
                    )
                self.calls += 1
                output = self.model.generate(**inputs, **generation)[0][input_tokens:]
        finally:
            for module, training in modes:
                module.training = training
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        text = self.tokenizer.decode(output, skip_special_tokens=True)
        if len(output) >= limit and int(output[-1]) not in eos:
            raise ValueError(f"Generation token limit reached; incomplete output: {text}")
        return Completion(
            parse_response(text, tools=bool(tools)),
            {
                "prompt_tokens": input_tokens,
                "completion_tokens": len(output),
            },
        )


@contextmanager
def load_checkpoint(config: dict, *, allow_download: bool = False):
    """Load one full checkpoint plus an optional PEFT adapter for a serial batch."""
    import torch
    import transformers

    record = copy.deepcopy(config)
    record.update(checkpoint_identity(config["id"], config.get("revision")))
    record["kind"] = "transformers"
    record["runtime"] = {name: version(name) for name in ("torch", "transformers", "accelerate")}
    kwargs = {"local_files_only": not allow_download, "trust_remote_code": False}
    if not Path(record["id"]).is_dir():
        kwargs["revision"] = record["revision"]
    tokenizer_source = config.get("tokenizer", {"id": record["id"], "revision": record["revision"]})
    tokenizer_identity = checkpoint_identity(
        **{"source": tokenizer_source["id"], "revision": tokenizer_source.get("revision")}
    )
    record["tokenizer"] = tokenizer_identity
    tokenizer_kwargs = {"local_files_only": not allow_download, "trust_remote_code": False}
    if not Path(tokenizer_identity["id"]).is_dir():
        tokenizer_kwargs["revision"] = tokenizer_identity["revision"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_identity["id"], **tokenizer_kwargs
    )
    record["chat_template_hash"] = fingerprint(tokenizer.chat_template)
    device = config["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver before loading weights")
    dtype = getattr(torch, config["dtype"])
    quantization = config["quantization"]
    if quantization == "nf4":
        record["runtime"]["bitsandbytes"] = version("bitsandbytes")
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization != "none":
        raise ValueError("Choose nf4 or none quantization")
    loaders = {
        "causal_lm": transformers.AutoModelForCausalLM,
        "multimodal_lm": getattr(transformers, "AutoModelForMultimodalLM", None),
    }
    loader = loaders.get(config["loader"])
    if loader is None:
        raise ValueError("Model loader unavailable in the installed Transformers version")
    model = loader.from_pretrained(
        record["id"],
        **kwargs,
        dtype=dtype,
        device_map={"": device},
        attn_implementation=config["attention"],
    )
    try:
        if config.get("adapter"):
            from peft import PeftModel

            adapter = checkpoint_identity(
                config["adapter"]["id"], config["adapter"].get("revision")
            )
            record["adapter"] = adapter
            record["runtime"]["peft"] = version("peft")
            adapter_kwargs = {"local_files_only": not allow_download}
            if not Path(adapter["id"]).is_dir():
                adapter_kwargs["revision"] = adapter["revision"]
            model = PeftModel.from_pretrained(
                model, adapter["id"], is_trainable=False, **adapter_kwargs
            )
        model.eval()
        record["generation_defaults"] = model.generation_config.to_dict()
        if device.startswith("cuda"):
            record["runtime"]["gpu"] = torch.cuda.get_device_name(torch.device(device))
            record["runtime"]["cuda"] = torch.version.cuda
        yield model, tokenizer, record
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def evaluate_checkpoint(
    scenarios: list[dict],
    config: dict,
    destination: Path,
    *,
    execute: bool = False,
    allow_download: bool = False,
    retry_failed: bool = False,
) -> list[dict]:
    """Evaluate a saved checkpoint; inspection does not load optional dependencies."""
    if not execute:
        return run_selected(scenarios, config, lambda: None, destination)
    with load_checkpoint(config, allow_download=allow_download) as (model, tokenizer, record):
        return run_selected(
            scenarios,
            record,
            lambda: TransformersBackend(model, tokenizer, record),
            destination,
            execute=True,
            retry_failed=retry_failed,
        )
