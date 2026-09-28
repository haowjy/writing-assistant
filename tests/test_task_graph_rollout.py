"""End-to-end driver contracts, offline replay, and adapter boundaries."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.task_graph_rollout_fixtures import (
    CANARIES,
    PortCallCounter,
    build_rollout_fixture,
    run_slice,
)
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_errors import AdapterContractError, DriverBudgetError
from writing_agent.task_graph_evaluation import EvaluationEvidenceV1
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_gatherers import CheckRunner, SamplingRunner
from writing_agent.task_graph_local import ScriptedSampleBackend
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_records import ContextOperationInputV1
from writing_agent.task_graph_rollout import RolloutDriver
from writing_agent.task_graph_rollout_env import CheckInput, RolloutEnvironment
from writing_agent.task_graph_store import TaskGraphStore


class RolloutDriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_phase5_scripted_slice_runs_to_reward_with_checkpoint_order(self):
        fixture = build_rollout_fixture(self.root / "slice")
        final, checkpoints = run_slice(
            fixture,
            max_steps=40,
        )
        self.assertEqual(len(checkpoints), 14)
        self.assertEqual(len(set(checkpoints)), len(checkpoints))
        self.assertEqual(checkpoints[-1], final.checkpoint_id)
        events = []
        commit = fixture.store.read_head("rollout-fixture")
        while commit is not None:
            record = fixture.store.load_commit(commit)
            events.extend(record.events)
            commit = record.parent_commit
        event_kinds = [fixture.store.load_event(identity).kind for identity in reversed(events)]
        self.assertEqual(
            event_kinds,
            [
                "writer_action",
                "tool_result",
                "writer_action",
                "external_requested",
                "author_turn",
                "writer_action",
                "tool_result",
                "writer_action",
                "external_requested",
                "check_recorded",
                "transition_committed",
                "termination_recorded",
                "reward_recorded",
            ],
        )
        availability = fixture.store.get_artifact(final.state.outcome_ref)
        self.assertEqual(availability["reward_status"], "available")
        self.assertEqual(
            fixture.counter.counts,
            {"sampling": 4, "tools": 2, "author": 1, "evaluator": 1},
        )

    def test_sampler_inputs_exclude_all_private_canaries(self):
        fixture = build_rollout_fixture(self.root / "privacy")
        sampler_inputs = []
        driver = fixture.driver(alternatives=lambda directive, port: sampler_inputs.append(port))
        final = driver.run(fixture.runtime, max_steps=40)
        self.assertEqual(final.state.position["phase"], "terminal")
        captured = repr(sampler_inputs) + repr(fixture.gatherers.sampler.backend.prepared_inputs)
        for canary in CANARIES.values():
            self.assertNotIn(canary, captured)

    def test_context_operation_runs_only_when_returned_by_the_alternatives_callback(self):
        fixture = build_rollout_fixture(self.root / "context-alternative")
        policy_ref = fixture.store.put_artifact(ContextPolicyV1("drop").to_dict())
        selected = False

        def alternatives(directive, _port):
            nonlocal selected
            if directive.kind == "sample_writer" and not selected:
                selected = True
                return ContextOperationInputV1(policy_ref)
            return None

        final = fixture.driver(alternatives=alternatives).run(fixture.runtime, max_steps=40)
        self.assertTrue(selected)
        self.assertEqual(
            fixture.store.load_event(final.state.history["head"]).kind, "reward_recorded"
        )
        self.assertEqual(fixture.counter.counts["sampling"], 4)

    def test_open_and_gate_replay_every_checkpoint_with_raising_ports(self):
        fixture = build_rollout_fixture(self.root / "source")
        final, checkpoints = run_slice(fixture)
        copied_store = self.root / "fresh-store"
        shutil.copytree(fixture.store.root, copied_store)
        gate = LineageGate()
        store = TaskGraphStore(copied_store, verifier=gate)
        env = RolloutEnvironment(store, fixture.entry.graph, None, gate, fixture.entry.graph.policy)
        replay_runtime = env.open(final.checkpoint_id)
        self.assertEqual(replay_runtime.checkpoint_id, final.checkpoint_id)
        for checkpoint_id in checkpoints:
            self.assertEqual(gate.view(store, checkpoint_id).checkpoint_id, checkpoint_id)
        counter = PortCallCounter(raising=True)
        gatherers = fixture.recovery_gatherers(
            replay_runtime, store=store, raising=True, counter=counter
        )
        RolloutDriver(env, gatherers).run(replay_runtime, max_steps=0)
        self.assertEqual(counter.total, 0)

    def test_crash_after_each_commit_resumes_to_identical_final_checkpoint(self):
        baseline = build_rollout_fixture(self.root / "baseline")
        expected_final, expected_ids = run_slice(baseline)
        lineage = baseline.runtime.state.position["lineage_id"]
        for stop_after in range(1, len(expected_ids)):
            with self.subTest(commit=stop_after):
                fixture = build_rollout_fixture(self.root / f"crash-{stop_after}")
                original_commit = fixture.env.commit
                captured = {}
                count = 0

                def crash_after_commit(
                    runtime,
                    input_record,
                    _commit=original_commit,
                    _captured=captured,
                    _stop_after=stop_after,
                ):
                    nonlocal count
                    result = _commit(runtime, input_record)
                    count += 1
                    _captured["runtime"] = result.runtime
                    if count == _stop_after:
                        raise RuntimeError("simulated process death")
                    return result

                fixture.env.commit = crash_after_commit
                with self.assertRaisesRegex(RuntimeError, "simulated process death"):
                    fixture.driver().run(fixture.runtime, max_steps=40)
                fixture.env.commit = original_commit
                stopped = captured["runtime"]
                head = fixture.store.read_head(lineage)
                self.assertEqual(fixture.store.load_commit(head).checkpoint, stopped.checkpoint_id)

                fresh_gate = LineageGate()
                fresh_store = TaskGraphStore(fixture.store.root, verifier=fresh_gate)
                fresh_env = RolloutEnvironment(
                    fresh_store,
                    fixture.entry.graph,
                    None,
                    fresh_gate,
                    fixture.entry.graph.policy,
                )
                resumed = fresh_env.open(stopped.checkpoint_id)
                gatherers = fixture.recovery_gatherers(resumed, store=fresh_store)
                actual = RolloutDriver(fresh_env, gatherers).run(resumed, max_steps=40)
                self.assertEqual(actual.checkpoint_id, expected_final.checkpoint_id)

    def test_none_mode_writer_only_node_reaches_reward(self):
        fixture = build_rollout_fixture(self.root / "none", mode="none")
        final, _ = run_slice(fixture)
        self.assertEqual(final.state.position["phase"], "terminal")
        self.assertEqual(
            fixture.store.get_artifact(final.state.outcome_ref)["reward_status"], "available"
        )
        self.assertEqual(fixture.counter.counts["author"], 0)
        self.assertEqual(fixture.counter.counts["tools"], 0)

    def test_max_steps_is_an_operational_guard(self):
        fixture = build_rollout_fixture(self.root / "budget")
        with self.assertRaises(DriverBudgetError) as caught:
            fixture.driver().run(fixture.runtime, max_steps=0)
        self.assertEqual(caught.exception.max_steps, 0)
        self.assertIsNone(fixture.store.read_head("rollout-fixture"))


class GathererContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = build_rollout_fixture(Path(self.temporary.name) / "fixture")
        self.view = self.fixture.env.verify(self.fixture.runtime)
        self.port = self.fixture.env.port_input(self.view, next_step(self.view))

    def test_sampling_rejects_invalid_role_content_and_missing_budget_usage(self):
        self.assertEqual(self.port.__class__.__name__, "SamplerInput")
        for result, port in (
            (
                SampleResult({"role": "user", "content": "bad", "tool_calls": []}),
                self.port,
            ),
            (
                SampleResult({"role": "assistant", "content": [], "tool_calls": []}),
                self.port,
            ),
            (
                SampleResult({"role": "assistant", "content": "ok", "tool_calls": []}),
                replace(self.port, usage_requirements=frozenset({"completion_tokens"})),
            ),
        ):
            with self.subTest(port=port), self.assertRaises(AdapterContractError):
                SamplingRunner(
                    self.fixture.store,
                    ScriptedSampleBackend((result,)),
                ).turn(port)
        self.assertIsNone(self.fixture.store.read_head("rollout-fixture"))

    def test_evaluator_family_mismatch_is_rejected_before_dispatch(self):
        check = next(
            iter(self.fixture.entry.graph.node(self.fixture.entry.node_id).checks.values())
        )

        class WrongFamilyEvaluator:
            family = "fixture-file-count-v1"
            calls = 0

            def evaluate(self, _request):
                self.calls += 1
                return EvaluationEvidenceV1(self.family, "pass", {})

        evaluator = WrongFamilyEvaluator()
        runner = CheckRunner(self.fixture.store, evaluator, {check.identity(): check})
        request = {
            "check_contract_hash": check.identity(),
            "check_id": check.id,
            "evaluator_packet_ref": "not-dispatched",
        }
        port = CheckInput("request-ref", request, {}, {"packet": "private"})
        with self.assertRaises(AdapterContractError):
            runner.result(port)
        self.assertEqual(evaluator.calls, 0)


if __name__ == "__main__":
    unittest.main()
