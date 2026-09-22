"""Append-only native actions and serial file-workspace groups for TRL rollouts."""

import copy
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from writing_agent.agent import run_agent
from writing_agent.catalog import fingerprint, save_json
from writing_agent.inference import ContextBudgetExceeded, TransformersBackend, render_messages
from writing_agent.reward import Reward, group_advantages
from writing_agent.workspace import Workspace


class ProtocolError(RuntimeError):
    """Unsupported framing is infrastructure failure, never a bad-candidate reward."""


class GroupPending(RuntimeError):
    """No optimizer step is allowed for this group."""


def native_suffix(tokenizer, assistant, external, raw_ids, *, thinking):
    """Derive only external bytes using a minimal dummy chat, as TRL does.

    Never render the sampled action: argument sorting and old-thinking removal are
    noninvertible. Require the native sampled stopping boundary explicitly. Gemma's
    tool-response opener is itself sampled; its body is external. A final sampled
    turn-end omits the template's following newline, which belongs in the suffix.
    """
    if not raw_ids or not external:
        raise ProtocolError("Missing sampled boundary or environment suffix")
    calls = assistant.get("tool_calls") or []
    if calls:
        if (assistant.get("content") or "").strip():
            raise ProtocolError("Mixed tool-call/content suffix is unsupported")
        if any(m["role"] != "tool" for m in external) or len(external) != len(calls):
            raise ProtocolError("Tool suffix requires one response per call")
        dummy = {"role": "assistant", "content": "", "tool_calls": copy.deepcopy(calls)}
        for call in dummy["tool_calls"]:
            call["function"]["arguments"] = {}
        boundary = "<|tool_response>"
    else:
        if len(external) != 1 or external[0]["role"] != "user":
            raise ProtocolError("Only a single user follow-up is supported")
        dummy = {"role": "assistant", "content": "dummy reply"}
        boundary = "<turn|>"
    boundary_ids = tokenizer.encode(boundary, add_special_tokens=False)
    if len(boundary_ids) != 1 or raw_ids[-1:] != boundary_ids:
        raise ProtocolError(f"Unsupported raw stopping boundary; expected {boundary}")
    messages = [{"role": "user", "content": "dummy"}, dummy]

    def encode(history, generation):
        text = tokenizer.apply_chat_template(
            render_messages(history),
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=thinking,
        )
        return tokenizer.encode(text, add_special_tokens=False)

    prefix = encode(messages, False)
    positions = [i for i, token in enumerate(prefix) if token == boundary_ids[0]]
    if not positions:
        raise ProtocolError("Native template is missing the expected suffix boundary")
    prefix = prefix[: positions[-1] + 1]
    full = encode(messages + external, True)
    if full[: len(prefix)] != prefix:
        raise ProtocolError("Native environment suffix is not token-prefix stable")
    return full[len(prefix) :]


class NativeRolloutBackend(TransformersBackend):
    """One attempt's exact sampled prefix; ordinary evaluation stays stateless."""

    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)
        if config["temperature"] != 1 or config["top_p"] != 1 or config.get("top_k") != 0:
            raise ProtocolError("GRPO requires neutral sampling processors")
        defaults = model.generation_config
        for name in (
            "min_p",
            "typical_p",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "forced_bos_token_id",
            "forced_eos_token_id",
            "suppress_tokens",
            "begin_suppress_tokens",
            "bad_words_ids",
            "sequence_bias",
            "min_length",
            "min_new_tokens",
            "exponential_decay_length_penalty",
        ):
            expected = 1.0 if name in {"typical_p", "repetition_penalty"} else None
            if getattr(defaults, name, None) not in (expected, None, 0):
                raise ProtocolError(f"Unsupported sampling processor: {name}")
        expected_stops = [
            tokenizer.convert_tokens_to_ids(t) for t in ("<eos>", "<turn|>", "<|tool_response>")
        ]
        stops = defaults.eos_token_id
        if not isinstance(stops, list) or set(stops) != set(expected_stops):
            raise ProtocolError("Native GRPO requires the pinned Gemma stop set")
        self.prompt_ids = []
        self.completion_ids = []
        self.env_mask = []
        self.boundaries = []
        self.previous = None
        self.previous_history = None
        self.last_output = []
        self.failure = None
        self.generated_limit = config["max_generated_tokens"]
        self.per_call_limit = config["max_tokens"]

    def prepare_inputs(self, messages, tools):
        import torch
        from transformers import BatchEncoding

        if self.previous_history is None:
            prompt, inputs = super().prepare_inputs(messages, tools)
            self.prompt_ids = inputs["input_ids"][0].tolist()
        else:
            prefix = self.previous_history + [self.previous]
            if messages[: len(prefix)] != prefix:
                raise ProtocolError("Harness rewrote sampled history")
            suffix = native_suffix(
                self.tokenizer,
                self.previous,
                messages[len(prefix) :],
                self.last_output,
                thinking=self.config["enable_thinking"],
            )
            if (
                len(self.prompt_ids) + len(self.completion_ids) + len(suffix)
                > self.config["context_tokens"]
            ):
                raise ContextBudgetExceeded("Environment suffix exceeds context; no truncation")
            self.completion_ids.extend(suffix)
            self.env_mask.extend([0] * len(suffix))
        ids = self.prompt_ids + self.completion_ids
        self.boundaries.append(
            {"input_ids": ids.copy(), "completion_offset": len(self.completion_ids)}
        )
        return self.tokenizer.decode(ids, skip_special_tokens=False), BatchEncoding(
            {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        )

    def complete(self, messages, tools, *, emit=lambda event: None):
        remaining = self.generated_limit - sum(self.env_mask)
        self.config["max_tokens"] = min(self.per_call_limit, remaining)
        if remaining <= 0:
            self.failure = "candidate_invalid"
            raise ValueError("Total generated-token budget exhausted")

        emitted_output = False

        def record(event):
            nonlocal emitted_output
            if event["type"] == "model_output":
                emitted_output = True
                self.last_output = event["output_ids"].copy()
                self.boundaries[-1].update(
                    output_ids=self.last_output,
                    termination="native-stop"
                    if self.last_output
                    and self.last_output[-1] in self.model.generation_config.eos_token_id
                    else "token-limit-or-empty",
                )
                self.completion_ids.extend(self.last_output)
                self.env_mask.extend([1] * len(self.last_output))
            emit(event)

        try:
            response = super().complete(messages, tools, emit=record)
            self.previous_history = copy.deepcopy(messages)
            self.previous = copy.deepcopy(response.message)
            return response
        except ValueError as exc:
            self.failure = (
                "candidate_invalid"
                if emitted_output and not isinstance(exc, ContextBudgetExceeded)
                else "infrastructure"
            )
            raise
        except Exception:
            self.failure = "infrastructure"
            raise

    def evidence(self):
        evidence = {
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "env_mask": self.env_mask,
            "boundaries": self.boundaries,
        }
        verify_tokens(evidence)
        return evidence


def verify_tokens(evidence):
    prompt, completion, mask = (evidence[k] for k in ("prompt_ids", "completion_ids", "env_mask"))
    if not prompt or len(completion) != len(mask) or any(m not in (0, 1) for m in mask):
        raise ProtocolError("Invalid rollout token ledger")
    reconstructed_mask = [0] * len(completion)
    for boundary in evidence["boundaries"]:
        offset = boundary["completion_offset"]
        if (
            not 0 <= offset <= len(completion)
            or boundary["input_ids"] != prompt + completion[:offset]
        ):
            raise ProtocolError("Generation input differs from the training prefix")
        outputs = boundary.get("output_ids", [])
        if completion[offset : offset + len(outputs)] != outputs:
            raise ProtocolError("Sampled output differs from training actions")
        if any(reconstructed_mask[offset : offset + len(outputs)]):
            raise ProtocolError("Overlapping generation boundaries")
        reconstructed_mask[offset : offset + len(outputs)] = [1] * len(outputs)
    if mask != reconstructed_mask:
        raise ProtocolError("Loss mask differs from sampled action spans")


class RolloutGroups:
    """TRL callback: routing strings identify tasks; only visible data reaches models.

    reward_callback(task, result) returns Reward, including rollout_reward or
    session_reward results. Its declared identity/config is frozen in the manifest.
    A callback exception or unavailable reward pends the entire group before TRL can
    turn missing entries into zeros. Ties halt by default; explicit continuation
    passes raw tied rewards to ordinary TRL/Adam without resampling. Advantages
    are mathematically zero but float32 reductions can leave residuals. Those
    residuals or Adam momentum can move weights; continuation is not skipping.
    """

    def __init__(self, tasks, settings, output, reward_callback, backend_factory, system_prompt):
        self.tasks = {t["id"]: t for t in copy.deepcopy(tasks)}
        self.settings, self.output = settings, Path(output)
        self.reward_callback, self.backend_factory = reward_callback, backend_factory
        self.system_prompt = system_prompt

    def __call__(self, prompts, trainer):
        if len(prompts) != self.settings.group_size or len(set(prompts)) != 1:
            raise ProtocolError("Expected exactly one complete same-task group per serial rollout")
        task = self.tasks[prompts[0]]
        group = self.output / "groups" / f"step-{trainer.state.global_step:06d}-{uuid4().hex}"
        group.mkdir(parents=True)
        save_json(group / "started.json", {"task": task["id"], "task_hash": fingerprint(task)})
        rewards, evidence = [], []
        try:
            for index in range(self.settings.group_size):
                attempt = group / f"attempt-{index:03d}"
                attempt.mkdir()
                seed = (
                    self.settings.seed
                    + (trainer.state.global_step * self.settings.group_size + index) * 32
                )
                save_json(attempt / "started.json", {"seed": seed, "status": "started"})
                tokens, result = self._attempt(task, trainer, attempt, seed)
                if evidence and tokens["prompt_ids"] != evidence[0]["prompt_ids"]:
                    result["failure_class"] = "infrastructure"
                    result["error"] = "Group initial prompts differ"
                save_json(attempt / "result.json", result)
                evidence.append(tokens)
                if result["failure_class"] == "infrastructure" or not any(tokens["env_mask"]):
                    reward = Reward("unavailable", reason="Candidate infrastructure failed")
                else:
                    try:
                        reward = self.reward_callback(copy.deepcopy(task), copy.deepcopy(result))
                        if not isinstance(reward, Reward):
                            raise TypeError("Reward callback must return Reward")
                    except Exception as exc:
                        reward = Reward("unavailable", reason=f"Reward callback failed: {exc}")
                rewards.append(reward)
                save_json(attempt / "reward.json", asdict(reward))
            stats = group_advantages(rewards)
            stats["tie_policy"] = self.settings.tie_policy
            if stats["status"] == "ok":
                # Python formula estimate, not observed TRL tensors. Float32 reductions
                # can differ, including nonzero residuals for exactly tied rewards.
                sample_std = stats["std"] * (len(rewards) / (len(rewards) - 1)) ** 0.5
                stats["trl_advantages_estimate"] = [
                    (r.value - stats["mean"]) / (sample_std + 1e-4) for r in rewards
                ]
            save_json(group / "group.json", stats)
            if stats["status"] != "ok":
                raise GroupPending("Unavailable reward/infrastructure: whole group pending")
            if stats["zero_variance"] and self.settings.tie_policy == "halt":
                raise GroupPending(
                    "All-tie group: no learning signal; stopped before optimizer update"
                )
            save_json(group / "complete.json", {"status": "scored", "attempts": len(rewards)})
        except BaseException as exc:
            save_json(group / "stopped.json", {"reason": f"{type(exc).__name__}: {exc}"})
            raise
        return {
            "prompt_ids": [e["prompt_ids"] for e in evidence],
            "completion_ids": [e["completion_ids"] for e in evidence],
            "env_mask": [e["env_mask"] for e in evidence],
            "logprobs": None,
            "rollout_rewards": [r.value for r in rewards],
        }

    def _attempt(self, task, trainer, attempt, seed):
        visible = task["visible"]
        workspace = None
        backend = None
        tokens = {"prompt_ids": [], "completion_ids": [], "env_mask": [], "boundaries": []}
        result = {
            "status": "setup_error",
            "output": "",
            "turns": [],
            "failure_class": "infrastructure",
        }
        try:
            workspace = Workspace(
                attempt / "workspace", max_total_bytes=visible["budgets"]["max_total_bytes"]
            )
            for path, content in visible["initial_files"].items():
                workspace.write_file(path, content)
            backend = self.backend_factory(trainer.model, trainer.processing_class, seed)
            trace_events = []
            with (attempt / "trace.jsonl").open("w") as stream:

                def emit(event):
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                    stream.flush()
                    trace_events.append(event)

                result = run_agent(
                    backend,
                    workspace,
                    [{"role": "user", "content": visible["brief"]}],
                    tools=visible["tools"],
                    followups=visible["followups"],
                    emit=emit,
                    system_prompt=self.system_prompt,
                    **{k: v for k, v in visible["budgets"].items() if k != "max_total_bytes"},
                )
            result["trace"] = trace_events
            result["failure_class"] = backend.failure or (
                "candidate_invalid" if result["status"] != "completed" else None
            )
            tokens = backend.evidence()
            verify_tokens(tokens)
            if (
                len(tokens["prompt_ids"]) + len(tokens["completion_ids"])
                > self.settings.context_tokens
                or sum(tokens["env_mask"]) > self.settings.max_generated_tokens
            ):
                raise ProtocolError("Backend exceeded frozen token budgets")
        except Exception as exc:
            result.update(failure_class="infrastructure", error=f"{type(exc).__name__}: {exc}")
        except BaseException as exc:
            result.update(
                status="interrupted",
                failure_class="infrastructure",
                error=f"{type(exc).__name__}: {exc}",
            )
            if backend is not None:
                try:
                    tokens = backend.evidence()
                except Exception:
                    pass  # Exact events already flushed to trace.jsonl.
            raise
        finally:
            result.update(before=visible["initial_files"], seed=seed)
            # Persist even interrupted generation/setup; the group is never marked complete.
            result["after"] = workspace.snapshot() if workspace is not None else {}
            save_json(attempt / "result.json", result)
            save_json(attempt / "tokens.json", tokens)
        return tokens, result


def saved_rewards(prompts, completions, rollout_rewards, **kwargs):
    """Never grade TRL's textual decode of a multi-turn token sequence."""
    return rollout_rewards
