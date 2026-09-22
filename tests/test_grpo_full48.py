"""Bound identity and trace boundaries; FULL48_RELEASE supplies read-only originals."""

import copy
import inspect
import os
import tempfile
import unittest
from pathlib import Path

from writing_agent.grpo_full48 import (
    CONTRACT_HASH,
    TASK_IDS,
    _binding,
    load_full48_release,
    mechanical_full48_reward,
    reward_spec,
)
from writing_agent.grpo_full48_fixtures import fixture_messages, run_fixture


class Full48BindingTests(unittest.TestCase):
    def test_contract_and_callback_interface(self):
        self.assertEqual(tuple(_binding()["tasks"]), TASK_IDS)
        self.assertEqual(reward_spec()["config"]["contract_hash"], CONTRACT_HASH)
        self.assertTrue(inspect.isfunction(mechanical_full48_reward))
        self.assertIsNone(mechanical_full48_reward.__closure__)
        with self.assertRaisesRegex(ValueError, "intact bound"):
            mechanical_full48_reward({"id": "development-001"}, {})


@unittest.skipUnless(os.environ.get("FULL48_RELEASE"), "Set FULL48_RELEASE to original release")
class Full48EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_full48_release(Path(os.environ["FULL48_RELEASE"]))["tasks"]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fixture(self, number, messages=None):
        task = self.tasks[number - 1]
        result = run_fixture(task, messages or fixture_messages(task), self.root / str(number))
        self.assertEqual(mechanical_full48_reward(task, result).value, 1)
        return task, result

    def test_mutated_original_and_rehashed_derivative_rejected(self):
        task = copy.deepcopy(self.tasks[0])
        task["visible"]["budgets"]["max_steps"] = 12
        from writing_agent.catalog import fingerprint

        task["visible_hash"] = fingerprint(task["visible"])
        with self.assertRaisesRegex(ValueError, "intact bound"):
            mechanical_full48_reward(task, {})

    def test_sequence_is_evidence_not_intermediate_prose_judgment(self):
        task, result = self.fixture(1)
        self.assertEqual(result["turns"][0]["output"], "Done.")
        # Losing a followup while claiming completion must fail.
        result["trace"] = [e for e in result["trace"] if e["type"] != "followup"]
        self.assertEqual(mechanical_full48_reward(task, result).value, 0)

    def test_file_delivery_does_not_need_verbal_acknowledgment(self):
        task, result = self.fixture(23)
        # run_agent requires text to close turns; scorer adds no redundant acknowledgment gate.
        result["output"] = ""
        result["turns"][-1]["output"] = ""
        result["messages"][result["turns"][-1]["message_index"]]["content"] = ""
        self.assertEqual(mechanical_full48_reward(task, result).value, 1)

    def test_patch_delivery_is_supported_and_missing_write_evidence_fails(self):
        task = self.tasks[22]
        messages = fixture_messages(task)
        write = next(
            m["tool_calls"][0]["function"]
            for m in messages
            if m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "write_file"
        )
        path, content = write["arguments"]["path"], write["arguments"]["content"]
        write.update(
            name="patch_file",
            arguments={"path": path, "old": task["visible"]["initial_files"][path], "new": content},
        )
        task, result = self.fixture(23, messages)
        result["trace"] = [e for e in result["trace"] if e["type"] != "tool"]
        self.assertEqual(mechanical_full48_reward(task, result).value, 0)

    def test_upper_length_failure_retains_signal_after_delivery(self):
        task = self.tasks[0]
        messages = fixture_messages(task)
        messages[-1]["content"] = " ".join(["fixture"] * 151)
        result = run_fixture(task, messages, self.root / "overlong")
        reward = mechanical_full48_reward(task, result)
        self.assertGreater(reward.value, 0)
        self.assertLess(reward.value, 1)
        self.assertEqual(reward.components["semantic_status"], "unjudged")


if __name__ == "__main__":
    unittest.main()
