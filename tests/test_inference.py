"""Contracts for local inference without downloading candidate checkpoints."""

import importlib.util
import json
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
    "prompt_format": "transcript",
    "max_tokens": 8,
    "context_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 42,
}


class ProtocolTests(unittest.TestCase):
    def test_protocol_keeps_prose_and_tool_arguments_separate(self):
        prose = '{"a story": "with braces"}'
        self.assertEqual(parse_response(prose, tools=False)["content"], prose)
        call = parse_response(
            json.dumps(
                {
                    "tool_calls": [
                        {"name": "write_file", "arguments": {"path": "draft.md", "content": prose}}
                    ]
                }
            ),
            tools=True,
        )
        transcript = render_messages(
            [
                {"role": "system", "content": "Write."},
                call,
                {"role": "tool", "tool_call_id": "call_0", "content": "saved"},
                {"role": "tool", "tool_call_id": "call_1", "content": "second result"},
            ],
            [{"name": "write_file"}],
        )
        self.assertIn("Available tools", transcript[0]["content"])
        self.assertIn("second result", transcript[-1]["content"])
        self.assertEqual(
            json.loads(transcript[1]["content"])["tool_calls"][0]["arguments"]["content"], prose
        )
        self.assertEqual(
            parse_response("Saved the draft.", tools=True)["content"], "Saved the draft."
        )
        self.assertEqual(parse_response(prose, tools=True)["content"], prose)
        reply = {"role": "assistant", "content": "Saved the draft."}
        self.assertEqual(render_messages([reply], [{"name": "write_file"}])[-1], reply)
        for invalid in (
            '{"tool_calls": []}',
            '{"tool_calls":',
            '{"tool_calls": [{"name": "write_file", "arguments": "bad"}]}',
        ):
            with self.assertRaisesRegex(ValueError, "Invalid writing-tools"):
                parse_response(invalid, tools=True)

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
                return self.replies.pop(0)

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
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "name": "write_file",
                            "arguments": {"path": "draft.md", "content": "The rain stopped."},
                        }
                    ]
                }
            ),
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
