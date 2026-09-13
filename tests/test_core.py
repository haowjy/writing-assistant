import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from writing_agent.agent import run_agent
from writing_agent.backends import ChatServerBackend, ScriptedBackend
from writing_agent.data import export_sft, read_records, validate_records
from writing_agent.evaluation import evaluate, score
from writing_agent.workspace import Workspace, dispatch

ROOT = Path(__file__).resolve().parents[1]


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Workspace(Path(self.tmp.name) / "story")

    def test_traversal_absolute_and_symlinks(self):
        for path in ("../secret", "/tmp/secret", "a/../../secret"):
            with self.assertRaises(ValueError):
                self.workspace.write_file(path, "bad")
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        (self.workspace.root / "link").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.workspace.write_file("link/secret", "bad")
        self.assertFalse((outside / "secret").exists())

    def test_local_patch_and_ambiguous_patch(self):
        self.workspace.write_file("draft.md", "red door\nquiet sea\n")
        self.workspace.patch_file("draft.md", "red", "blue")
        self.assertEqual(self.workspace.read_file("draft.md"), "blue door\nquiet sea\n")
        self.workspace.write_file("duplicate.md", "red red")
        with self.assertRaises(ValueError):
            self.workspace.patch_file("duplicate.md", "red", "blue")
        self.assertEqual(self.workspace.read_file("duplicate.md"), "red red")

    def test_tools_search_and_limits(self):
        self.workspace.write_file("notes/facts.md", "Mara keeps the light.")
        self.assertEqual(self.workspace.list_dir(), ["notes/"])
        self.assertEqual(self.workspace.search("MARA")[0]["path"], "notes/facts.md")
        self.assertFalse(dispatch(self.workspace, "resolve", {"path": "."})["ok"])
        self.assertFalse(dispatch(self.workspace, "read_file", {"path": 42})["ok"])
        with self.assertRaises(ValueError):
            self.workspace.write_file("huge", "x" * 128_001)


class AgentTests(unittest.TestCase):
    def test_invalid_arguments_are_observed_and_recoverable(self):
        responses = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {"name": "read_file", "arguments": "{bad"},
                    }
                ],
            },
            {"role": "assistant", "content": "Please specify a file."},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(ScriptedBackend(responses), Workspace(Path(tmp)), [])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_errors"], 1)
        self.assertEqual(result["messages"][-2]["role"], "tool")

    def test_step_limit_and_backend_error(self):
        tool_message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {"name": "list_dir", "arguments": "{}"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(
                ScriptedBackend([tool_message]), Workspace(Path(tmp)), [], max_steps=1
            )
            failed = run_agent(ScriptedBackend([]), Workspace(Path(tmp)), [])
        self.assertEqual(result["status"], "step_limit")
        self.assertEqual(failed["status"], "error")

    def test_tool_budget_stops_mutations(self):
        calls = [
            {
                "id": str(i),
                "function": {
                    "name": "write_file",
                    "arguments": {"path": f"{i}.md", "content": "draft"},
                },
            }
            for i in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            result = run_agent(
                ScriptedBackend([{"role": "assistant", "tool_calls": calls}]),
                workspace,
                [],
                max_tool_calls=1,
            )
            self.assertEqual(workspace.snapshot(), {"0.md": "draft"})
        self.assertEqual(result["status"], "tool_limit")

    def test_chat_transport_and_usage(self):
        import io

        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello", "extra": "ignore"},
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
        with patch(
            "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())
        ):
            completion = ChatServerBackend({"model": "test", "base_url": "http://localhost/v1"})
            result = completion.complete([], [])
        self.assertEqual(result.usage["prompt_tokens"], 5)
        self.assertNotIn("extra", result.message)

    def test_truncated_generation_is_not_success(self):
        import io

        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "unfinished"},
                }
            ]
        }
        with patch(
            "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())
        ):
            with self.assertRaisesRegex(ValueError, "Incomplete generation"):
                ChatServerBackend({"model": "test", "base_url": "http://localhost/v1"}).complete(
                    [], []
                )


class DataTests(unittest.TestCase):
    def setUp(self):
        self.record = read_records(ROOT / "data/fixtures/trajectories.jsonl")[0]

    def test_split_leakage_rejected(self):
        other = copy.deepcopy(self.record)
        other.update(id="other", split="test")
        with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
            validate_records([self.record, other])

    def test_malformed_objects_rejected(self):
        for field in ("provenance", "messages", "tools"):
            record = copy.deepcopy(self.record)
            record[field] = [None]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_records([record])

    def test_valid_tool_trajectory(self):
        from writing_agent.workspace import TOOL_SCHEMAS

        self.record["tools"] = TOOL_SCHEMAS
        self.record["messages"][1:1] = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "one",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "one", "content": "[]"},
        ]
        validate_records([self.record])
        self.record["tools"] = []
        with self.assertRaisesRegex(ValueError, "unavailable tool"):
            validate_records([self.record])

    def test_unmatched_tool_response_rejected(self):
        self.record["messages"].insert(
            1, {"role": "tool", "tool_call_id": "missing", "content": "result"}
        )
        with self.assertRaisesRegex(ValueError, "unmatched tool response"):
            validate_records([self.record])

    def test_export_only_reviewed_train_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            destination = Path(tmp) / "sft.jsonl"
            source.write_text(json.dumps(self.record) + "\n")
            with self.assertRaisesRegex(ValueError, "No accepted"):
                export_sft(source, destination)
            self.record["review_status"] = "accepted"
            source.write_text(json.dumps(self.record) + "\n")
            self.assertEqual(export_sft(source, destination), 1)
            self.assertEqual(set(json.loads(destination.read_text())), {"messages", "tools"})
            with self.assertRaises(FileExistsError):
                export_sft(source, destination)


class EvaluationTests(unittest.TestCase):
    def test_unrequested_file_change_fails(self):
        result = {"output": "Done", "status": "completed", "tool_calls": 1, "tool_errors": 0}
        scores = score(result, {"canon": "old"}, {"canon": "new"}, {})
        self.assertFalse(scores["agent"]["passed"])
        self.assertIsNone(scores["artifact"]["literary_quality"])

    def test_smoke_end_to_end_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "smoke.toml"
            config.write_text(
                f'tasks = "{ROOT}/data/fixtures/tasks.jsonl"\noutput_dir = "runs"\n'
                f'[backend]\nkind = "scripted"\n'
                f'responses = "{ROOT}/data/fixtures/responses.json"\n'
            )
            first = evaluate(config)
            second = evaluate(config)
            self.assertNotEqual(first, second)
            summary = json.loads((first / "summary.json").read_text())
            self.assertEqual(summary["passed"], 5)
            self.assertFalse(summary["is_model_evaluation"])
            self.assertTrue((first / "retrieve/trace.jsonl").exists())
            self.assertEqual(
                (first / "local_edit/workspace/drafts/scene.md").read_text(),
                "Mara opened the blue door.\nThe sea was quiet.\n",
            )


if __name__ == "__main__":
    unittest.main()
