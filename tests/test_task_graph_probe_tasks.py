"""Admission and scripted-lineage evidence for the public Phase 8 probe tasks."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from tests.task_graph_rollout_fixtures import (
    bind_usage_reporting_session,
    build_rollout_fixture,
    run_slice,
)
from writing_agent.task_graph_ports import SampleResult

BUILDER_PATH = Path(__file__).parents[1] / "configs/phase8/probe-tasks/build.py"
_BUILDER_SPEC = importlib.util.spec_from_file_location("phase8_probe_task_builder", BUILDER_PATH)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_BUILDER_SPEC)
_BUILDER_SPEC.loader.exec_module(_BUILDER)


def _tool_call(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _sample(message: dict) -> SampleResult:
    return SampleResult(message, usage={"prompt_tokens": 32, "completion_tokens": 8})


def _samples(config: dict, example: dict) -> tuple[SampleResult, ...]:
    decision = config["public"]["decision"]
    calls = [
        _tool_call(
            "write_file",
            {
                "path": "scene.txt",
                "content": config["public"]["initial_files"]["scene.txt"],
            },
            f"{config['id']}-starter",
        )
    ]
    samples = [_sample({"role": "assistant", "content": "", "tool_calls": calls})]
    samples.append(
        _sample(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "ask_author",
                        {
                            "question": decision["question"],
                            "decision_ids": [decision["id"]],
                            "proposals": [],
                            "option_refs": [],
                        },
                        f"{config['id']}-ask",
                    )
                ],
            }
        )
    )
    writes = [
        _tool_call(
            "write_file",
            {"path": "scene.txt", "content": example["scene"]},
            f"{config['id']}-scene",
        )
    ]
    if example["notes"] is not None:
        writes.append(
            _tool_call(
                "write_file",
                {"path": "field-notes.txt", "content": example["notes"]},
                f"{config['id']}-notes",
            )
        )
    samples.append(_sample({"role": "assistant", "content": "", "tool_calls": writes}))
    samples.extend(
        (
            _sample(
                {
                    "role": "assistant",
                    "content": "The scene is ready for review.",
                    "tool_calls": [],
                }
            ),
            _sample(
                {
                    "role": "assistant",
                    "content": "I applied the feedback revision.",
                    "tool_calls": [],
                }
            ),
        )
    )
    return tuple(samples)


def _run(config: dict, samples: tuple[SampleResult, ...]):
    with tempfile.TemporaryDirectory() as temporary:
        fixture = build_rollout_fixture(
            Path(temporary) / "rollout",
            mode="feedback",
            sample_results=samples,
            entry_fixture=_BUILDER.build_admitted_entry(config),
        )
        # Probe graphs are token-limited, so they run under a usage-reporting session.
        bind_usage_reporting_session(fixture)
        runtime = run_slice(fixture)
        view = fixture.env.verify(runtime)
        reward = fixture.store.get_artifact(view.outcome.reward_ref)
        return fixture.counter.counts, view.outcome, reward


class ProbeTaskGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = _BUILDER.load_probe_tasks()

    def test_three_configs_admit_with_one_deterministic_family_and_probe_budgets(self):
        self.assertEqual(
            [config["id"] for config in self.configs],
            [
                "t1-lighthouse",
                "t2-winter-garden",
                "t3-coastal-post",
            ],
        )
        for config in self.configs:
            with self.subTest(task=config["id"]):
                settings = config["probe_settings"]
                self.assertEqual(settings["max_context_tokens"], 4096)
                self.assertEqual(settings["max_tokens_per_decision"], 512)
                self.assertEqual(settings["training_mode"], "native")
                entry = _BUILDER.build_admitted_entry(config)
                self.assertEqual(len(entry.graph.instance.nodes), 1)
                node = entry.graph.node(entry.node_id)
                self.assertEqual(node.spec.kind, "writer")
                self.assertEqual(node.contract.interaction_contract.mode, "scripted_author")
                self.assertEqual(len(node.contract.interaction_contract.mandatory_feedback), 1)
                self.assertEqual(
                    {check.evaluator_version for check in node.checks.values()},
                    {"deterministic-v1"},
                )
                self.assertEqual(
                    [check.id for check in node.checks.values() if check.required],
                    ["scene_nonempty"],
                )
                self.assertEqual(len(node.checks) - 1, 4)
                decision = config["public"]["decision"]
                self.assertEqual(
                    node.author_packet.preferences[decision["binding"]],
                    config["public"]["author_packet"]["preferences"][decision["binding"]],
                )
                budget = node.contract.budget_contract
                self.assertEqual(budget.max_generated_tokens, 1536)
                self.assertEqual(budget.max_total_tokens, 26112)
                self.assertEqual(budget.max_steps, 6)
                self.assertEqual(budget.max_tool_calls, 8)
                self.assertEqual(budget.max_author_calls, 2)

    def test_scripted_lineages_reach_five_reward_levels_through_author_and_feedback(self):
        cases = (
            ("t1-lighthouse", "nonempty_only", 1000),
            ("t1-lighthouse", "decision_phrase", 3500),
            ("t2-winter-garden", "phrase_and_detail", 5500),
            ("t3-coastal-post", "word_range", 8000),
            ("t3-coastal-post", "all_optional", 10000),
        )
        by_id = {config["id"]: config for config in self.configs}
        observed = {}
        for task_id, lineage_id, expected_score in cases:
            config = by_id[task_id]
            counts, outcome, reward = _run(
                config, _samples(config, config["scripted_lineages"][lineage_id])
            )
            with self.subTest(task=task_id, lineage=lineage_id):
                self.assertEqual(outcome.task_status, "complete")
                self.assertEqual(counts["author"], 2, "ask_author answer plus mandatory feedback")
                self.assertEqual(reward["numerator"], expected_score)
                observed[lineage_id] = reward["numerator"]
        self.assertEqual(
            observed,
            {
                "nonempty_only": 1000,
                "decision_phrase": 3500,
                "phrase_and_detail": 5500,
                "word_range": 8000,
                "all_optional": 10000,
            },
        )
        self.assertGreaterEqual(len(set(observed.values())), 5)

    def test_writer_turn_exhaustion_is_an_incomplete_lineage(self):
        config = self.configs[-1]
        samples = tuple(
            _sample(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tool_call(
                            "write_file",
                            {"path": "scene.txt", "content": f"A brief draft, pass {index}."},
                            f"incomplete-{index}",
                        )
                    ],
                }
            )
            for index in range(6)
        )
        _counts, outcome, reward = _run(config, samples)
        self.assertEqual(outcome.task_status, "incomplete")
        self.assertEqual(outcome.execution_status, "valid")
        self.assertEqual(reward["numerator"], 0)


if __name__ == "__main__":
    unittest.main()
