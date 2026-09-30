"""Tokenizer-backed contracts for supported task-graph native suffixes."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import unittest

from writing_agent.native_protocol import NATIVE_STOP_TOKENS, ProtocolError, native_suffix

TOKENIZER_ID = "google/gemma-4-E2B-it"
TOKENIZER_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"

B_SUFFIX_N1 = [
    6275,
    236787,
    891,
    236779,
    6386,
    236782,
    680,
    236787,
    3397,
    236764,
    3709,
    236787,
    52,
    3709,
    236771,
    52,
    236783,
    51,
    106,
    107,
    105,
    2364,
    107,
    27252,
    625,
    1932,
    236761,
    106,
    107,
    105,
    4368,
    107,
]
B_SUFFIX_N2 = [
    6275,
    236787,
    1399,
    236779,
    2164,
    236782,
    680,
    236787,
    3397,
    236764,
    3709,
    236787,
    52,
    3709,
    236771,
    52,
    236783,
    51,
    50,
    6275,
    236787,
    891,
    236779,
    6386,
    236782,
    680,
    236787,
    3397,
    236764,
    3709,
    236787,
    52,
    3709,
    236770,
    52,
    236783,
    51,
    106,
    107,
    105,
    2364,
    107,
    27252,
    625,
    1932,
    236761,
    106,
    107,
    105,
    4368,
    107,
]


def _call(name, arguments, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _tool_message(call, content):
    return {"role": "tool", "tool_call_id": call["id"], "content": content}


def _result(result, *, ok=True, valid=None):
    response = {"ok": ok}
    if valid is not None:
        response["valid"] = valid
    response["result" if ok else "error"] = result
    return json.dumps(response, separators=(",", ":"))


class NativeProtocolImportTests(unittest.TestCase):
    def test_native_protocol_import_does_not_load_model_libraries(self):
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers'}:
        raise AssertionError(f'eager model import: {name}')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import writing_agent.native_protocol
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            env=env,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(
    importlib.util.find_spec("transformers") is not None,
    "requires the optional transformers dependency for tokenizer-backed tests",
)
class NativeProtocolTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_ID,
            revision=TOKENIZER_REVISION,
            local_files_only=True,
        )

    def _token_id(self, token):
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        self.assertEqual(len(token_ids), 1, token)
        return token_ids[0]

    def _is_prefix_stable(self, assistant, external, suffix, boundary):
        from writing_agent.inference import render_messages

        dummy = copy.deepcopy(assistant)
        for call in dummy.get("tool_calls", ()):
            call["function"]["arguments"] = {}
        messages = [{"role": "user", "content": "dummy"}, dummy]
        boundary_id = self._token_id(boundary)

        def encode(history, generation):
            text = self.tokenizer.apply_chat_template(
                render_messages(history),
                tokenize=False,
                add_generation_prompt=generation,
                enable_thinking=False,
            )
            return self.tokenizer.encode(text, add_special_tokens=False)

        rendered_prefix = encode(messages, False)
        end = max(i for i, token in enumerate(rendered_prefix) if token == boundary_id) + 1
        rendered_prefix = rendered_prefix[:end]
        full = encode(messages + list(external), True)
        self.assertEqual(full[: len(rendered_prefix)], rendered_prefix)
        self.assertEqual(suffix, full[len(rendered_prefix) :])

    def test_stop_set_is_the_pinned_native_stop_set(self):
        stop_ids = [self._token_id(token) for token in NATIVE_STOP_TOKENS]
        self.assertEqual(NATIVE_STOP_TOKENS, ("<eos>", "<turn|>", "<|tool_response>"))
        self.assertEqual(len(stop_ids), len(set(stop_ids)))

    def test_a_plain_tool_results_include_errors_and_invalid_calls(self):
        for name, content in (
            ("read_file", _result({"text": "found"})),
            ("write_file", _result("bad path", ok=False)),
            ("invalid_call", _result("invalid call", ok=False, valid=False)),
        ):
            with self.subTest(name=name):
                call = _call(name, {"payload": "varies"}, "call_0")
                external = [_tool_message(call, content)]
                assistant = {"role": "assistant", "content": "", "tool_calls": [call]}
                suffix = native_suffix(
                    self.tokenizer,
                    assistant,
                    external,
                    [self._token_id("<|tool_response>")],
                    thinking=False,
                )
                self._is_prefix_stable(assistant, external, suffix, "<|tool_response>")
                changed = copy.deepcopy(assistant)
                changed["tool_calls"][0]["function"]["arguments"] = {"other": "value"}
                self.assertEqual(
                    native_suffix(
                        self.tokenizer,
                        changed,
                        external,
                        [self._token_id("<|tool_response>")],
                        thinking=False,
                    ),
                    suffix,
                )

    def test_b_ask_author_suffix_ids_match_the_pinned_probe(self):
        one = _call("ask_author", {"question": "Should the ending stay open?"}, "call_0")
        external_one = [
            _tool_message(one, _result("result0")),
            {"role": "user", "content": "Keep it open."},
        ]
        assistant_one = {"role": "assistant", "content": "", "tool_calls": [one]}
        suffix_one = native_suffix(
            self.tokenizer,
            assistant_one,
            external_one,
            [self._token_id("<|tool_response>")],
            thinking=False,
        )
        self.assertEqual(suffix_one, B_SUFFIX_N1)
        self._is_prefix_stable(assistant_one, external_one, suffix_one, "<|tool_response>")

        file_call = _call("read_file", {"path": "n.md"}, "call_0")
        ask_call = _call("ask_author", {"question": "Should the ending stay open?"}, "call_1")
        external_two = [
            _tool_message(file_call, _result("result0")),
            _tool_message(ask_call, _result("result1")),
            {"role": "user", "content": "Keep it open."},
        ]
        assistant_two = {"role": "assistant", "content": "", "tool_calls": [file_call, ask_call]}
        suffix_two = native_suffix(
            self.tokenizer,
            assistant_two,
            external_two,
            [self._token_id("<|tool_response>")],
            thinking=False,
        )
        self.assertEqual(suffix_two, B_SUFFIX_N2)
        self._is_prefix_stable(assistant_two, external_two, suffix_two, "<|tool_response>")

        changed = copy.deepcopy(assistant_one)
        changed["tool_calls"][0]["function"]["arguments"] = {
            "question": "Should the ending remain open after the storm?"
        }
        self.assertEqual(
            native_suffix(
                self.tokenizer,
                changed,
                external_one,
                [self._token_id("<|tool_response>")],
                thinking=False,
            ),
            suffix_one,
        )

    def test_c_user_feedback_suffix_accepts_turn_and_eos_without_inventing_turn(self):
        external = [{"role": "user", "content": "Make the final image more concrete."}]
        assistant = {"role": "assistant", "content": "The ending should stay open."}
        for sampled_boundary in ("<turn|>", "<eos>"):
            with self.subTest(sampled_boundary=sampled_boundary):
                suffix = native_suffix(
                    self.tokenizer,
                    assistant,
                    external,
                    [self._token_id(sampled_boundary)],
                    thinking=False,
                )
                self._is_prefix_stable(assistant, external, suffix, "<turn|>")
                if sampled_boundary == "<eos>":
                    decoded_suffix = self.tokenizer.decode(suffix, skip_special_tokens=False)
                    self.assertTrue(decoded_suffix.startswith("\n<|turn>user\n"))
                changed = {"role": "assistant", "content": "A different sampled answer."}
                self.assertEqual(
                    native_suffix(
                        self.tokenizer,
                        changed,
                        external,
                        [self._token_id(sampled_boundary)],
                        thinking=False,
                    ),
                    suffix,
                )

    def test_d_check_driven_continuation_supports_empty_context_delta(self):
        assistant = {"role": "assistant", "content": "Draft submitted for checks."}
        for sampled_boundary in ("<turn|>", "<eos>"):
            with self.subTest(sampled_boundary=sampled_boundary):
                suffix = native_suffix(
                    self.tokenizer,
                    assistant,
                    [],
                    [self._token_id(sampled_boundary)],
                    thinking=False,
                )
                self._is_prefix_stable(assistant, [], suffix, "<turn|>")
                self.assertEqual(
                    native_suffix(
                        self.tokenizer,
                        {"role": "assistant", "content": "Different checked draft."},
                        [],
                        [self._token_id(sampled_boundary)],
                        thinking=False,
                    ),
                    suffix,
                )

    def test_unsupported_context_shapes_fail_as_protocol_errors(self):
        call = _call("read_file", {"path": "x.md"}, "call_0")
        assistant_with_call = {"role": "assistant", "content": "", "tool_calls": [call]}
        assistant = {"role": "assistant", "content": "answer"}
        boundary = [self._token_id("<|tool_response>")]
        invalid_deltas = (
            (
                assistant_with_call,
                [_tool_message(call, _result("x")), {"role": "tool", "content": "extra"}],
            ),
            (
                assistant_with_call,
                [_tool_message(call, _result("x")), {"role": "user", "content": "not ask_author"}],
            ),
            (assistant_with_call, [{"role": "user", "content": "out of order"}]),
            (assistant, [{"role": "tool", "content": "no call"}]),
            (assistant, [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]),
        )
        for sampled, external in invalid_deltas:
            with self.subTest(external=external), self.assertRaises(ProtocolError):
                native_suffix(
                    self.tokenizer,
                    sampled,
                    external,
                    boundary,
                    thinking=False,
                )

        with self.assertRaises(ProtocolError):
            native_suffix(
                self.tokenizer,
                assistant_with_call,
                [_tool_message(call, _result("x"))],
                [self._token_id("<eos>")],
                thinking=False,
            )
        with self.assertRaises(ProtocolError):
            native_suffix(
                self.tokenizer,
                assistant,
                [{"role": "user", "content": "follow up"}],
                [self._token_id("<|tool_response>")],
                thinking=False,
            )
        with self.assertRaises(ProtocolError):
            native_suffix(self.tokenizer, assistant, [], [], thinking=False)


if __name__ == "__main__":
    unittest.main()
