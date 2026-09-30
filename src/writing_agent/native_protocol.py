"""Gemma native-chat framing shared by legacy and task-graph sampling."""

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

NATIVE_STOP_TOKENS = ("<eos>", "<turn|>", "<|tool_response>")


class ProtocolError(RuntimeError):
    """Unsupported native framing is an infrastructure failure."""


@dataclass(frozen=True)
class NativeParseResult:
    """One native parse outcome, including the adapter's parse-failure claim."""

    message: Mapping[str, Any]
    failed: bool


def parse_native_response(
    tokenizer,
    raw_text: str,
    *,
    prefix: str,
    action_id: str,
    termination: Mapping[str, Any],
) -> NativeParseResult:
    """Parse and bind a sampled native response, retaining malformed output as text.

    A tokenizer parse failure is model output, not a sampler fault. Keep the decoded
    response as ordinary assistant content and let the committed turn claim the failure;
    the tokenizer-backed audit independently reproduces that claim.
    """
    if not isinstance(termination, Mapping) or termination.get("kind") not in {
        "native_stop",
        "token_limit",
    }:
        raise ProtocolError("Native response parsing requires a sampled termination")

    from writing_agent.inference import parse_response

    try:
        message = parse_response(tokenizer, raw_text, prefix=prefix)
    except (AttributeError, TypeError, ValueError):
        return NativeParseResult(
            {"role": "assistant", "content": raw_text, "tool_calls": []},
            True,
        )
    return NativeParseResult(bind_native_tool_call_ids(message, action_id), False)


def native_suffix(tokenizer, assistant, external, raw_ids, *, thinking):
    """Derive only template-owned suffix tokens after one sampled assistant turn.

    Sampled assistant tokens are never reconstructed from parsed messages. Tool
    results may optionally be followed by one ``ask_author`` user reply; a final
    answer may be followed by one user message or by an internal check-driven
    continuation with no new context. Every other context delta fails closed.
    """
    from writing_agent.inference import render_messages

    if not isinstance(assistant, Mapping):
        raise ProtocolError("Sampled assistant must be a message")
    if not raw_ids:
        raise ProtocolError("Missing sampled boundary")
    if not isinstance(external, Sequence) or isinstance(external, (str, bytes)):
        raise ProtocolError("External suffix must be a message sequence")

    calls = assistant.get("tool_calls") or []
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise ProtocolError("Sampled tool calls must be a sequence")
    if calls:
        if any(not isinstance(call, Mapping) for call in calls):
            raise ProtocolError("Malformed sampled tool call")
        tool_results = external[: len(calls)]
        if len(tool_results) != len(calls) or any(
            not isinstance(message, Mapping) or message.get("role") != "tool"
            for message in tool_results
        ):
            raise ProtocolError("Tool suffix requires one response per call")

        has_author_reply = len(external) == len(calls) + 1 and (
            isinstance(external[-1], Mapping) and external[-1].get("role") == "user"
        )
        if len(external) not in {len(calls), len(calls) + 1}:
            raise ProtocolError("Tool suffix has an unsupported context delta")
        if len(external) == len(calls) + 1 and not has_author_reply:
            raise ProtocolError("Tool suffix may only append an ask_author user reply")
        if has_author_reply and not any(
            isinstance(call.get("function"), Mapping)
            and call["function"].get("name") == "ask_author"
            for call in calls
        ):
            raise ProtocolError("User reply must follow an ask_author tool call")

        dummy = {"role": "assistant", "content": "", "tool_calls": copy.deepcopy(calls)}
        for call in dummy["tool_calls"]:
            try:
                if not isinstance(call.get("function"), Mapping):
                    raise TypeError
                call["function"]["arguments"] = {}
            except (KeyError, TypeError) as exc:
                raise ProtocolError("Malformed sampled tool call") from exc
        boundary = "<|tool_response>"
    else:
        if len(external) > 1 or (
            external and (not isinstance(external[0], Mapping) or external[0].get("role") != "user")
        ):
            raise ProtocolError("Only a single user follow-up is supported")
        dummy = {"role": "assistant", "content": "dummy reply"}
        boundary = "<turn|>"

    boundary_ids = tokenizer.encode(boundary, add_special_tokens=False)
    final_eos = not calls and raw_ids[-1] == tokenizer.convert_tokens_to_ids("<eos>")
    if len(boundary_ids) != 1 or (raw_ids[-1] != boundary_ids[0] and not final_eos):
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
    full = encode(messages + list(external), True)
    if full[: len(prefix)] != prefix:
        raise ProtocolError("Native environment suffix is not token-prefix stable")
    return full[len(prefix) :]


def bind_native_tool_call_ids(message, action_id):
    """Bind parsed calls to core IDs derived from the committed writer action.

    Task-graph tool results use ``<lineage>:call:<ordinal>:<index>``. Gemma's
    parser-local IDs restart for every response, so native sampling assigns that
    same identity before intake; the audit uses this function to reproduce it.
    """
    if not isinstance(message, Mapping):
        raise ProtocolError("Parsed native response must be a message")
    if not isinstance(action_id, str) or any(
        character.isspace() or ord(character) < 0x20 for character in action_id
    ):
        raise ProtocolError("Native tool calls require a logical writer action ID")
    lineage_id, separator, ordinal_text = action_id.rpartition(":action:")
    if (
        not separator
        or not lineage_id
        or not ordinal_text.isascii()
        or not ordinal_text.isdecimal()
        or (len(ordinal_text) > 1 and ordinal_text.startswith("0"))
    ):
        raise ProtocolError("Native tool calls require a canonical writer action ID")
    result = copy.deepcopy(message)
    calls = result.get("tool_calls", [])
    if not isinstance(calls, list) or any(
        not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping)
        for call in calls
    ):
        raise ProtocolError("Parsed native tool calls are malformed")
    for index, call in enumerate(calls):
        call["id"] = f"{lineage_id}:call:{ordinal_text}:{index}"
    return result


__all__ = [
    "NATIVE_STOP_TOKENS",
    "ProtocolError",
    "bind_native_tool_call_ids",
    "native_suffix",
]
