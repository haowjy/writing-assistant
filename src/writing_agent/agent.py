"""A bounded tool loop shared by all model variants."""

import copy
import json
import time
from collections.abc import Callable

from writing_agent.backends import Backend
from writing_agent.workspace import TOOL_SCHEMAS, Workspace, dispatch

SYSTEM_PROMPT = """You are a conversational creative-writing collaborator.
Use project files when needed. Follow the latest explicit user decision over stale notes.
Distinguish proposals, drafts, accepted prose, and committed canon. New draft prose does
not automatically authorize canon updates. Preserve character knowledge and deferred reveals.
Write only the requested amount. Make local edits without changing unrelated text.
Use relative workspace paths. Return a final answer when the requested work is complete.
"""


def run_agent(
    backend: Backend,
    workspace: Workspace,
    messages: list[dict],
    *,
    max_steps: int = 12,
    max_tool_calls: int = 32,
    emit: Callable[[dict], None] = lambda event: None,
    tools: list[str] | None = None,
    followups: list[str] | None = None,
    max_read_tokens: int = 20_000,
    count_tokens: Callable[[str], int] = lambda text: len(text.split()),
    read_tokenizer: str = "whitespace-v1",
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    if max_steps < 1 or max_tool_calls < 0 or max_read_tokens < 0:
        raise ValueError("Invalid agent budget")
    available = {s["function"]["name"] for s in TOOL_SCHEMAS}
    allowed = available if tools is None else set(tools)
    if allowed - available:
        raise ValueError("Unknown tool configuration")
    schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed]
    pending = iter(followups or [])
    turns = []
    read_tokens = 0
    history = [{"role": "system", "content": system_prompt}, *copy.deepcopy(messages)]
    calls = 0
    attempted_calls = 0
    errors = 0
    usage = {}
    started = time.perf_counter()
    emit({"type": "input", "messages": history, "tools": schemas})

    def finish(status: str, output: str = "", error: str | None = None) -> dict:
        return {
            "status": status,
            "output": output,
            "error": error,
            "tool_calls": calls,
            "attempted_tool_calls": attempted_calls,
            "tool_errors": errors,
            "usage": usage,
            "latency_seconds": time.perf_counter() - started,
            "messages": history,
            "turns": turns,
            "read_tokens": read_tokens,
            "read_tokenizer": read_tokenizer,
        }

    def merge_usage(target, current):
        for key, value in current.items():
            if isinstance(value, dict):
                merge_usage(target.setdefault(key, {}), value)
            elif isinstance(value, (int, float)):
                target[key] = target.get(key, 0) + value

    try:
        for step in range(max_steps):
            before = time.perf_counter()
            completion = backend.complete(history, schemas, emit=emit)
            message = completion.message
            emit(
                {
                    "type": "generation",
                    "step": step,
                    "message": message,
                    "usage": completion.usage,
                    "latency_seconds": time.perf_counter() - before,
                }
            )
            if message.get("role") != "assistant":
                raise ValueError("Backend must return an assistant message")
            merge_usage(usage, completion.usage)
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ValueError("tool_calls must be a list")
            attempted_calls += len(tool_calls)
            history.append(message)
            if not tool_calls:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Final response must contain text")
                turn = {
                    "output": content,
                    "snapshot": workspace.snapshot(),
                    "message_index": len(history) - 1,
                }
                turns.append(turn)
                emit({"type": "turn", "turn": turn})
                followup = next(pending, None)
                if followup is None:
                    return finish("completed", content)
                history.append({"role": "user", "content": followup})
                emit({"type": "followup", "content": followup})
                continue
            for call in tool_calls:
                if calls >= max_tool_calls:
                    return finish("tool_limit")
                if (
                    not isinstance(call, dict)
                    or not isinstance(call.get("id"), str)
                    or not call["id"]
                ):
                    emit(
                        {
                            "type": "tool",
                            "call": call,
                            "observation": {
                                "ok": False,
                                "valid": False,
                                "error": "Tool call needs an id",
                            },
                        }
                    )
                    raise ValueError("Tool call needs an id")
                calls += 1
                try:
                    function = call["function"]
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if function["name"] not in allowed:
                        raise ValueError("Tool is not available in this condition")
                    observation = dispatch(workspace, function["name"], arguments)
                    if observation["ok"] and function["name"] in {
                        "read_file",
                        "search",
                        "list_dir",
                    }:
                        cost = count_tokens(json.dumps(observation["result"], ensure_ascii=False))
                        if read_tokens + cost > max_read_tokens:
                            observation = {
                                "ok": False,
                                "valid": True,
                                "error": "Read-token budget exceeded",
                            }
                        else:
                            read_tokens += cost
                except (KeyError, TypeError, ValueError) as exc:
                    observation = {
                        "ok": False,
                        "valid": False,
                        "error": f"Invalid tool call: {exc}",
                    }
                errors += int(not observation["ok"])
                reply = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(observation, ensure_ascii=False),
                }
                history.append(reply)
                emit({"type": "tool", "call": call, "observation": observation})
        return finish("step_limit")
    except Exception as exc:
        # Preserve failed samples so a server error cannot silently shrink the denominator.
        emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        return finish("error", error=f"{type(exc).__name__}: {exc}")
