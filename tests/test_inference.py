"""Contracts for local inference without downloading candidate checkpoints."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from writing_agent.agent import run_agent
from writing_agent.inference import (
    PROTOCOL,
    TransformersBackend,
    checkpoint_identity,
    evaluate_checkpoint,
    parse_response,
    render_messages,
)
from writing_agent.workspace import Workspace

CONFIG = {
    "id": "fixture",
    "revision": "a" * 40,
    "kind": "transformers",
    "protocol": PROTOCOL,
    "prompt_format": "chat",
    "max_tokens": 8,
    "context_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 42,
}


class ProtocolTests(unittest.TestCase):
    def test_native_history_keeps_tool_results_and_conversation_distinct(self):
        history = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "a",
                        "function": {"name": "read_file", "arguments": '{"path":"notes.md"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": '{"ok":true,"result":"Notes"}'},
            {"role": "assistant", "content": "Here is the scene."},
        ]
        rendered = render_messages(history)
        self.assertEqual(
            rendered[0]["tool_calls"][0]["function"]["arguments"], {"path": "notes.md"}
        )
        self.assertEqual(
            rendered[0]["tool_responses"],
            [{"name": "read_file", "response": {"ok": True, "result": "Notes"}}],
        )
        self.assertEqual(rendered[1], history[2])
        self.assertIsInstance(history[0]["tool_calls"][0]["function"]["arguments"], str)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "optional tokenizer dependency")
    def test_pinned_native_parser_and_template(self):
        from transformers import AutoTokenizer

        from writing_agent.workspace import TOOL_SCHEMAS

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "google/gemma-4-E2B-it",
                revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                local_files_only=True,
            )
        except OSError:
            self.skipTest("Pinned tokenizer is not cached; no downloads in tests")
        text = (
            '<|tool_call>call:write_file{path:<|"|>draft.md<|"|>,'
            'content:<|"|>Hello, {world}!\nA "quote" and café.<|"|>}<tool_call|><|tool_response>'
        )
        text = "<|channel>thought\nRead then write.\n<channel|>" + text
        call = parse_response(tokenizer, text, prefix="")
        self.assertEqual(call["thinking"], "Read then write.")
        self.assertNotIn("Read then write.", call.get("content", ""))
        self.assertEqual(
            call["tool_calls"][0]["function"]["arguments"]["content"],
            'Hello, {world}!\nA "quote" and café.',
        )
        history = [
            {"role": "user", "content": "Write a file."},
            call,
            {"role": "tool", "tool_call_id": "call_0", "content": '{"ok":true}'},
        ]
        prompt = tokenizer.apply_chat_template(
            render_messages(history), tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True
        )
        self.assertIn("<|channel>thought\nRead then write.\n<channel|>", prompt)
        self.assertIn("<|tool>declaration:write_file", prompt)
        self.assertIn("<|tool_response>response:write_file{ok:true}", prompt)
        self.assertEqual(
            parse_response(tokenizer, "Saved.<turn|>", prefix=prompt)["content"], "Saved."
        )
        with self.assertRaises(ValueError):
            parse_response(tokenizer, "<|tool_call>call:write_file{path:", prefix="")

    def test_local_checkpoint_contents_change_identity(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.safetensors"
            path.write_bytes(b"first checkpoint")
            before = checkpoint_identity(root)
            path.write_bytes(b"second checkpoint")
            self.assertNotEqual(before, checkpoint_identity(root))
        with self.assertRaises(ValueError):
            checkpoint_identity("organization/model", "main")

    def test_inspect_does_not_load_model_or_create_runs(self):
        with patch("writing_agent.inference.load_checkpoint") as load:
            result = evaluate_checkpoint([{"id": "example"}], CONFIG, Path("unused"))
        load.assert_not_called()
        self.assertEqual(result[0]["status"], "planned")


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional inference dependencies")
class GenerationTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            pad_token_id = 0
            chat_template = "fixture"

            def __call__(self, text, **kwargs):
                return Inputs(input_ids=torch.tensor([[3, 4]]))

            def apply_chat_template(self, messages, **kwargs):
                return str(messages)

            def decode(self, output, **kwargs):
                self.current = self.replies.pop(0)
                return "<|tool_call>" if isinstance(self.current, dict) else self.current

            def parse_response(self, text, *, prefix):
                return (
                    self.current
                    if isinstance(self.current, dict)
                    else {"role": "assistant", "content": self.current}
                )

        class Model(torch.nn.Module):
            device = torch.device("cpu")
            generation_config = type("Generation", (), {"eos_token_id": 1})()

            def generate(self, input_ids, **kwargs):
                if self.training or torch.is_grad_enabled():
                    raise AssertionError("Generation must disable training and gradients")
                torch.rand(1)
                return torch.cat([input_ids, torch.tensor([[7, 1]])], dim=-1)

        self.model = Model()
        self.tokenizer = Tokenizer()

    def test_tool_loop_writes_actual_file_and_restores_training_rng(self):
        self.tokenizer.replies = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "draft.md", "content": "The rain stopped."},
                        }
                    }
                ],
            },
            "Saved.",
        ]
        before = self.torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(Path(root))
            result = run_agent(
                TransformersBackend(self.model, self.tokenizer, CONFIG),
                workspace,
                [{"role": "user", "content": "Write a story to draft.md"}],
                tools=["write_file"],
            )
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual((Path(root) / "draft.md").read_text(), "The rain stopped.")
            self.assertEqual(result["usage"]["completion_tokens"], 4)
        self.assertTrue(self.model.training)
        self.assertTrue(self.torch.equal(before, self.torch.random.get_rng_state()))

    def test_context_overflow_is_not_silently_truncated(self):
        backend = TransformersBackend(self.model, self.tokenizer, {**CONFIG, "context_tokens": 1})
        with self.assertRaisesRegex(ValueError, "Context budget"):
            backend.complete([{"role": "user", "content": "hello"}], [])

    def test_generation_failure_restores_mode_and_rng(self):
        before = self.torch.random.get_rng_state().clone()
        with patch.object(self.model, "generate", side_effect=RuntimeError("out of memory")):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                TransformersBackend(self.model, self.tokenizer, CONFIG).complete(
                    [{"role": "user", "content": "hello"}], []
                )
        self.assertTrue(self.model.training)
        self.assertTrue(self.torch.equal(before, self.torch.random.get_rng_state()))
