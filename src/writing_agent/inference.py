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

PROTOCOL = "gemma-native-v1"


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


def render_messages(messages: list[dict]) -> list[dict]:
    """Convert harness history to Gemma's native assistant tool-response structure."""
    rendered = []
    for original in messages:
        message = copy.deepcopy(original)
        if "thinking" in message:
            message["reasoning"] = message.pop("thinking")
        if message["role"] == "tool":
            if not rendered or not rendered[-1].get("tool_calls"):
                raise ValueError("Tool response has no preceding calls")
            owner = rendered[-1]
            call = next(
                (c for c in owner["tool_calls"] if c["id"] == message["tool_call_id"]), None
            )
            if call is None:
                raise ValueError("Unknown tool response ID")
            owner.setdefault("tool_responses", []).append(
                {
                    "name": call["function"]["name"],
                    "response": json.loads(message["content"]),
                }
            )
            continue
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
        rendered.append(message)
    return rendered


def parse_response(tokenizer, text: str, *, prefix: str) -> dict:
    """Use the checkpoint's response grammar, preserving native delimiters until parsed."""
    message = tokenizer.parse_response(text, prefix=prefix)
    calls = message.get("tool_calls", [])
    if text.count("<|tool_call>") != len(calls):
        raise ValueError("Native tool-call output was not completely parsed")
    for i, call in enumerate(calls):
        call["id"] = f"call_{i}"
        if not isinstance(call["function"]["arguments"], dict):
            raise ValueError("Native tool arguments must be an object")
    return message


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
        thinking = config.get("enable_thinking", config["prompt_format"] == "chat")
        if not isinstance(thinking, bool):
            raise ValueError("enable_thinking must be a boolean")
        if thinking and config["prompt_format"] != "chat":
            raise ValueError("Thinking requires native chat formatting")
        self.model, self.tokenizer = model, tokenizer
        self.config = copy.deepcopy(config)
        self.config["enable_thinking"] = thinking
        self.calls = 0

    def complete(
        self, messages: list[dict], tools: list[dict], *, emit=lambda event: None
    ) -> Completion:
        import torch

        rendered = render_messages(messages)
        if self.config["prompt_format"] == "chat":
            prompt = self.tokenizer.apply_chat_template(
                rendered,
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config["enable_thinking"],
            )
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        else:
            if tools:
                raise ValueError(
                    "Native tools require a verified chat template; base tool runs are unsupported"
                )
            prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in rendered)
            prompt += "\n\nASSISTANT:\n"
            inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        input_tokens = inputs["input_ids"].shape[-1]
        emit(
            {
                "type": "model_input",
                "prompt": prompt,
                "input_ids": inputs["input_ids"][0].tolist(),
                "protocol": PROTOCOL,
            }
        )
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
        text = self.tokenizer.decode(output, skip_special_tokens=False)
        emit({"type": "model_output", "text": text, "output_ids": output.tolist()})
        if len(output) >= limit and int(output[-1]) not in eos:
            raise ValueError(f"Generation token limit reached; incomplete output: {text}")
        return Completion(
            (
                parse_response(self.tokenizer, text, prefix=prompt)
                if self.config["prompt_format"] == "chat"
                else {
                    "role": "assistant",
                    "content": self.tokenizer.decode(output, skip_special_tokens=True),
                }
            ),
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
    record.setdefault("enable_thinking", record["prompt_format"] == "chat")
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
    record["response_template_hash"] = fingerprint(tokenizer.response_template)
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
