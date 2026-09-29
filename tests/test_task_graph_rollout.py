"""End-to-end driver contracts, offline replay, and adapter boundaries."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from tests.task_graph_rollout_fixtures import (
    AUTHOR_PACKET_CANARY,
    CANARIES,
    EVALUATOR_PACKET_CANARY,
    LEDGER_CANARY,
    PortCallCounter,
    build_rollout_fixture,
    make_gatherers,
    ports_disabled,
    run_slice,
)
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_environment import (
    CheckInput,
    RolloutEnvironment,
)
from writing_agent.task_graph_environment import (
    derive_input as producer_derive_input,
)
from writing_agent.task_graph_errors import AdapterContractError, DriverBudgetError, ProjectionError
from writing_agent.task_graph_evaluation import (
    FAMILIES,
    DecodedEvidence,
    EvaluationEvidenceV1,
    EvaluationRequestV1,
    EvidenceFamily,
    verify_evaluation_evidence,
)
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_gate import derive_input as gate_derive_input
from writing_agent.task_graph_gatherers import CheckRunner
from writing_agent.task_graph_local import LocalTextToolProvider
from writing_agent.task_graph_ports import EnvironmentResult, EnvironmentSnapshot, SampleResult
from writing_agent.task_graph_records import ContextOperationInputV1, SampledMessageV1, WriterTurnV1
from writing_agent.task_graph_rollout import RolloutDriver
from writing_agent.task_graph_store import TaskGraphStore


def _artifact_count(store: TaskGraphStore) -> int:
    return sum(path.is_file() for path in store.root.rglob("*"))


def _lineage_events(fixture):
    commit_id = fixture.store.read_head(fixture.lineage_id)
    events = []
    while commit_id is not None:
        commit = fixture.store.load_commit(commit_id)
        events.extend(fixture.store.load_event(ref) for ref in commit.events)
        commit_id = commit.parent_commit
    return list(reversed(events))


class RolloutDriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_phase5_scripted_slice_runs_to_reward_with_checkpoint_order(self):
        fixture = build_rollout_fixture(self.root / "slice")
        final = run_slice(fixture)
        checkpoints = fixture.checkpoint_ids
        self.assertEqual(len(checkpoints), 14)
        self.assertEqual(len(set(checkpoints)), len(checkpoints))
        self.assertEqual(checkpoints[-1], final.checkpoint_id)
        event_kinds = [event.kind for event in _lineage_events(fixture)]
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

    def test_sampler_inputs_exclude_private_canaries_and_are_all_captured(self):
        fixture = build_rollout_fixture(self.root / "privacy")
        final = run_slice(fixture)
        self.assertEqual(final.state.position["phase"], "terminal")
        self.assertEqual(len(fixture.sampler_inputs), 4)
        captured = repr(fixture.sampler_inputs) + repr(
            fixture.gatherers.sampler.backend.prepared_inputs
        )
        for canary in CANARIES.values():
            self.assertNotIn(canary, captured)
        author_packet = fixture.entry.graph.node(fixture.entry.node_id).author_packet
        self.assertNotIn(LEDGER_CANARY, repr(author_packet))
        self.assertNotIn(
            EVALUATOR_PACKET_CANARY,
            repr(fixture.entry.graph.node(fixture.entry.node_id).contract.completion_contract),
        )
        for body in fixture.entry.reader.public.values():
            self.assertNotIn(EVALUATOR_PACKET_CANARY, repr(body))
        self.assertIn(AUTHOR_PACKET_CANARY, repr(author_packet))

    def test_backend_echoed_content_hash_is_the_derive_bound_context_hash(self):
        class EchoBackend:
            def sample(self, prepared):
                self.prepared = prepared
                return SampleResult(
                    {"role": "assistant", "content": "echo accepted", "tool_calls": []},
                    trace={
                        "context_content_hash": prepared.context_content_hash,
                        "context_revision_ref": prepared.context_revision_ref,
                    },
                )

        fixture = build_rollout_fixture(self.root / "context-hash-echo", mode="none")
        view = fixture.env.verify(fixture.runtime)
        backend = EchoBackend()
        gatherers = make_gatherers(fixture, sampler=backend)
        try:
            result = RolloutDriver(fixture.env, gatherers).run(fixture.runtime, max_steps=1)
            runtime = result.runtime
        except DriverBudgetError as exc:
            runtime = exc.runtime

        self.assertEqual(backend.prepared.context_content_hash, view.context.content_ref)
        event = fixture.store.load_event(runtime.state.history["head"])
        turn = WriterTurnV1.from_dict(fixture.store.get_artifact(event.payload_ref))
        self.assertEqual(turn.adapter_trace["context_content_hash"], view.context.content_ref)

    def test_driver_combines_dispatch_view_and_scope_per_step(self):
        fixture = build_rollout_fixture(self.root / "step-counts", mode="none")
        scopes = 0
        depth = 0
        view_calls = 0
        old_operation = fixture.store.operation
        original_view = fixture.gate.view

        @contextmanager
        def count_scope():
            nonlocal scopes, depth
            if depth == 0:
                scopes += 1
            depth += 1
            try:
                with old_operation():
                    yield
            finally:
                depth -= 1

        def count_view(*args, **kwargs):
            nonlocal view_calls
            view_calls += 1
            return original_view(*args, **kwargs)

        fixture.store.operation = count_scope
        fixture.gate.view = count_view
        committed = []

        def stop_after_commit(result):
            committed.append(result)
            raise RuntimeError("stop after one committed step")

        fixture.env.commit_observer = stop_after_commit
        with (
            patch(
                "writing_agent.task_graph_environment.derive_input",
                wraps=producer_derive_input,
            ) as producer_derive,
            patch(
                "writing_agent.task_graph_gate.derive_input",
                wraps=gate_derive_input,
            ) as verifier_derive,
            self.assertRaisesRegex(RuntimeError, "stop after one committed step"),
        ):
            fixture.driver().run(fixture.runtime, max_steps=40)

        self.assertEqual(len(committed), 1)
        # step_input and commit each open one environment scope.
        self.assertEqual(scopes, 2)
        self.assertEqual(view_calls, 3)
        self.assertEqual(producer_derive.call_count, 1)
        self.assertEqual(verifier_derive.call_count, 1)

    def test_context_operation_runs_only_when_offered_and_callback_is_not_overoffered(self):
        fixture = build_rollout_fixture(self.root / "context-alternative")
        policy_ref = fixture.store.put_artifact(ContextPolicyV1("drop").to_dict())
        selected = False
        offered = []

        def alternatives(directive, _port):
            nonlocal selected
            offered.append(directive.kind)
            if directive.kind == "sample_writer" and not selected:
                selected = True
                return ContextOperationInputV1(policy_ref)
            return None

        result = fixture.driver(alternatives=alternatives).run(fixture.runtime, max_steps=40)
        self.assertTrue(selected)
        self.assertEqual(
            [kind for kind in offered],
            [
                next_step(fixture.gate.view(fixture.store, checkpoint)).kind
                for checkpoint in fixture.checkpoint_ids[:-1]
                if next_step(fixture.gate.view(fixture.store, checkpoint)).alternatives
            ],
        )
        self.assertEqual(result.directive.kind, "done")
        self.assertEqual(
            fixture.store.load_event(result.runtime.state.history["head"]).kind,
            "reward_recorded",
        )
        operations = [
            event for event in _lineage_events(fixture) if event.kind == "context_changed"
        ]
        self.assertEqual(len(operations), 1)
        payload = fixture.store.get_artifact(operations[0].payload_ref)
        self.assertEqual(payload["record_type"], "ContextOperationInputV1")
        self.assertEqual(fixture.counter.counts["sampling"], 4)

    def test_open_and_gate_replay_every_checkpoint_with_all_ports_disabled(self):
        fixture = build_rollout_fixture(self.root / "source")
        final = run_slice(fixture)
        copied_store = self.root / "fresh-store"
        shutil.copytree(fixture.store.root, copied_store)
        counter = PortCallCounter(raising=True)
        with ports_disabled():
            gate = LineageGate()
            store = TaskGraphStore(copied_store, verifier=gate)
            env = RolloutEnvironment(
                store, fixture.entry.graph, None, gate, fixture.entry.graph.policy
            )
            replay_runtime = env.open_head(fixture.lineage_id)
            self.assertEqual(replay_runtime.checkpoint_id, final.checkpoint_id)
            for checkpoint_id in fixture.checkpoint_ids:
                self.assertEqual(gate.view(store, checkpoint_id).checkpoint_id, checkpoint_id)
            gatherers = fixture.recovery_gatherers(
                replay_runtime, store=store, raising=True, counter=counter
            )
            self.assertIsNotNone(gatherers)
        self.assertEqual(counter.total, 0)

    def test_crash_after_each_commit_resumes_from_open_head_on_fresh_store(self):
        baseline = build_rollout_fixture(self.root / "baseline")
        expected = run_slice(baseline)
        for stop_after in range(1, len(baseline.checkpoint_ids)):
            with self.subTest(commit=stop_after):
                fixture = build_rollout_fixture(self.root / f"crash-{stop_after}")
                original_commit = fixture.env.commit
                count = 0

                def crash_after_commit(
                    runtime, input_record, original_commit=original_commit, stop_after=stop_after
                ):
                    nonlocal count
                    result = original_commit(runtime, input_record)
                    count += 1
                    if count == stop_after:
                        raise RuntimeError("simulated process death")
                    return result

                fixture.env.commit = crash_after_commit
                with self.assertRaisesRegex(RuntimeError, "simulated process death"):
                    fixture.driver().run(fixture.runtime, max_steps=40)
                fixture.env.commit = original_commit
                head = fixture.store.read_head(fixture.lineage_id)
                self.assertEqual(
                    fixture.store.load_commit(head).checkpoint, fixture.checkpoint_ids[-1]
                )

                fresh_gate = LineageGate()
                fresh_store = TaskGraphStore(fixture.store.root, verifier=fresh_gate)
                fresh_env = RolloutEnvironment(
                    fresh_store,
                    fixture.entry.graph,
                    None,
                    fresh_gate,
                    fixture.entry.graph.policy,
                )
                resumed = fresh_env.open_head(fixture.lineage_id)
                gatherers = fixture.recovery_gatherers(resumed, store=fresh_store)
                actual = RolloutDriver(fresh_env, gatherers).run(resumed, max_steps=40)
                self.assertEqual(actual.runtime.checkpoint_id, expected.checkpoint_id)

    def test_crash_inside_commit_recovers_every_stage_and_selected_commits(self):
        class SimulatedCrash(RuntimeError):
            pass

        expected_fixture = build_rollout_fixture(self.root / "crash-baseline")
        expected = run_slice(expected_fixture)
        stages = (
            "before_immutable_writes",
            "after_immutable_writes",
            "before_head_publication",
            "after_head_publication",
            "record_published",
        )
        for commit_number in (1, 3, 5, 11):
            for stage in stages:
                with self.subTest(commit=commit_number, stage=stage):
                    fixture = build_rollout_fixture(self.root / f"inside-{commit_number}-{stage}")
                    publish = fixture.store.publish
                    record_published = fixture.gate.record_published
                    state = {"publications": 0}

                    def crash_publish(
                        *args,
                        _commit_number=commit_number,
                        _stage=stage,
                        _publish=publish,
                        _state=state,
                        **kwargs,
                    ):
                        _state["publications"] += 1
                        if (
                            _state["publications"] == _commit_number
                            and _stage != "record_published"
                        ):

                            def fail_at(fault_stage, _stage=_stage):
                                if fault_stage == _stage:
                                    raise SimulatedCrash(_stage)

                            kwargs["fault"] = fail_at
                        return _publish(*args, **kwargs)

                    def crash_record(
                        store,
                        commit_id,
                        _commit_number=commit_number,
                        _stage=stage,
                        _record_published=record_published,
                        _state=state,
                    ):
                        if (
                            _state["publications"] == _commit_number
                            and _stage == "record_published"
                        ):
                            raise SimulatedCrash(_stage)
                        return _record_published(store, commit_id)

                    fixture.store.publish = crash_publish
                    fixture.gate.record_published = crash_record
                    with self.assertRaises(SimulatedCrash):
                        fixture.driver().run(fixture.runtime, max_steps=40)
                    fixture.store.publish = publish
                    fixture.gate.record_published = record_published

                    gate = LineageGate()
                    store = TaskGraphStore(fixture.store.root, verifier=gate)
                    env = RolloutEnvironment(
                        store, fixture.entry.graph, None, gate, fixture.entry.graph.policy
                    )
                    resumed = (
                        env.open_head(fixture.lineage_id)
                        if store.read_head(fixture.lineage_id) is not None
                        else env.open(fixture.checkpoint_ids[0])
                    )
                    result = RolloutDriver(
                        env, fixture.recovery_gatherers(resumed, store=store)
                    ).run(resumed, max_steps=40)
                    self.assertEqual(result.runtime.checkpoint_id, expected.checkpoint_id)
                    self.assertEqual(
                        store.read_head(fixture.lineage_id),
                        fixture.store.read_head(fixture.lineage_id),
                    )

    def test_feedback_fixture_runs_mandatory_reply_and_requirement_update(self):
        fixture = build_rollout_fixture(self.root / "feedback", mode="feedback")
        result = run_slice(fixture)
        events = _lineage_events(fixture)
        requests = []
        for event in events:
            if event.kind == "author_turn":
                reply = fixture.store.get_artifact(event.payload_ref)
                requests.append(fixture.store.get_artifact(reply["request_ref"], private=True))
        self.assertIn("mandatory_feedback", [request["source"] for request in requests])
        ledger = fixture.store.get_artifact(result.state.requirements_ref, private=True)
        self.assertEqual(ledger["active"], {"revised": "Revise for " + LEDGER_CANARY})
        self.assertEqual(ledger["superseded"], {"baseline": LEDGER_CANARY})
        self.assertEqual(fixture.counter.counts["author"], 2)

    def test_none_mode_writer_only_node_reaches_reward(self):
        fixture = build_rollout_fixture(self.root / "none", mode="none")
        final = run_slice(fixture)
        self.assertEqual(final.state.position["phase"], "terminal")
        self.assertEqual(
            fixture.store.get_artifact(final.state.outcome_ref)["reward_status"], "available"
        )
        self.assertEqual(fixture.counter.counts["author"], 0)
        self.assertEqual(fixture.counter.counts["tools"], 0)

    def test_run_result_distinguishes_halt_from_done(self):
        done_fixture = build_rollout_fixture(self.root / "done")
        done = done_fixture.driver().run(done_fixture.runtime, max_steps=40)
        self.assertEqual(done.directive.kind, "done")

        halt_fixture = build_rollout_fixture(self.root / "halt", mode="halt")
        halted = halt_fixture.driver().run(halt_fixture.runtime, max_steps=40)
        self.assertEqual(halted.directive.kind, "halt")
        self.assertNotEqual(halted.runtime.checkpoint_id, done.runtime.checkpoint_id)

    def test_max_steps_is_operational_and_writes_nothing(self):
        fixture = build_rollout_fixture(self.root / "budget")
        before = _artifact_count(fixture.store)
        head = fixture.store.read_head(fixture.lineage_id)
        with self.assertRaises(DriverBudgetError) as caught:
            fixture.driver().run(fixture.runtime, max_steps=0)
        self.assertEqual(caught.exception.max_steps, 0)
        self.assertEqual(caught.exception.runtime, fixture.runtime)
        self.assertEqual(fixture.counter.total, 0)
        self.assertEqual(_artifact_count(fixture.store), before)
        self.assertEqual(fixture.store.read_head(fixture.lineage_id), head)

    def test_total_token_limit_is_seeded_and_charges_overruns_to_writer_stop(self):
        fixture = build_rollout_fixture(
            self.root / "total-token-budget",
            mode="total_token_limited",
            sample_results=(
                SampleResult(
                    {"role": "assistant", "content": "Over the total-token budget."},
                    usage={"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
                ),
            ),
        )
        entry_budget = fixture.store.get_artifact(fixture.runtime.state.budgets_ref)
        self.assertEqual(entry_budget["limits"]["total_tokens"], 5)
        self.assertNotIn("total_tokens", entry_budget["consumed"])

        with self.assertRaises(DriverBudgetError) as stopped:
            fixture.driver().run(fixture.runtime, max_steps=1)
        runtime = stopped.exception.runtime
        view = fixture.env.verify(runtime)
        event = fixture.store.load_event(runtime.state.history["head"])
        budget = fixture.store.get_artifact(runtime.state.budgets_ref)
        self.assertEqual(event.kind, "budget_charged")
        self.assertEqual(view.outcome.stop_reason, "total_tokens_budget")
        self.assertEqual(budget["consumed"]["total_tokens"], 6)

    def test_total_token_limit_requires_usage_evidence_before_publication(self):
        fixture = build_rollout_fixture(
            self.root / "missing-total-token-usage",
            mode="total_token_limited",
            sample_results=(SampleResult({"role": "assistant", "content": "No usage."}),),
        )
        head = fixture.store.read_head(fixture.lineage_id)

        with self.assertRaises(AdapterContractError):
            fixture.driver().run(fixture.runtime, max_steps=1)

        self.assertEqual(fixture.store.read_head(fixture.lineage_id), head)


class GathererContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_adapter_contract_cases_through_driver_do_not_move_head(self):
        for result, mode in (
            (SampleResult({"role": "user", "content": "bad", "tool_calls": []}), "none"),
            (SampleResult({"role": "assistant", "content": [], "tool_calls": []}), "none"),
            (
                SampleResult({"role": "assistant", "content": "ok", "tool_calls": []}),
                "token_limited",
            ),
        ):
            with self.subTest(result=result, mode=mode):
                fixture = build_rollout_fixture(
                    self.root / f"adapter-{mode}-{len(list(self.root.iterdir()))}",
                    mode=mode,
                    sample_results=(result,),
                )
                old_head = fixture.store.read_head(fixture.lineage_id)
                with self.assertRaises(AdapterContractError):
                    fixture.driver().run(fixture.runtime, max_steps=1)
                self.assertEqual(fixture.store.read_head(fixture.lineage_id), old_head)

    def test_tool_effect_rejection_remains_adapter_error_after_gatherer_check_is_removed(self):
        fixture = build_rollout_fixture(self.root / "tool-effect", mode="slice")
        try:
            fixture.driver().run(fixture.runtime, max_steps=1)
        except DriverBudgetError as exc:
            runtime = exc.runtime
        port = fixture.env.step_input(runtime)[2]
        self.assertIsNotNone(port)

        class OversizeToolProvider:
            def execute(self, spec, _snapshot, _action):
                value = "x" * (spec.max_file_bytes + 1)
                return EnvironmentResult(
                    {"ok": True, "valid": True, "result": "written"},
                    EnvironmentSnapshot.from_files({"draft.txt": value}),
                )

        gatherers = make_gatherers(fixture, tools=OversizeToolProvider())
        old_head = fixture.store.read_head(fixture.lineage_id)
        with self.assertRaises(AdapterContractError):
            RolloutDriver(fixture.env, gatherers).run(runtime, max_steps=1)
        self.assertEqual(fixture.store.read_head(fixture.lineage_id), old_head)

    def test_evaluator_evidence_rejection_remains_adapter_error_at_derive(self):
        fixture = build_rollout_fixture(self.root / "bad-evidence", mode="slice")
        runtime = run_slice(fixture, until=lambda directive: directive.kind == "await_check_result")
        port = fixture.env.step_input(runtime)[2]

        class InvalidEvidenceEvaluator:
            family = "deterministic-file-v1"

            def evaluate(self, _request):
                return EvaluationEvidenceV1(self.family, "pass", {})

        gatherers = make_gatherers(fixture, evaluator=InvalidEvidenceEvaluator())
        old_head = fixture.store.read_head(fixture.lineage_id)
        with self.assertRaises(AdapterContractError):
            fixture.env.commit(runtime, gatherers.evaluator.result(port))
        self.assertEqual(fixture.store.read_head(fixture.lineage_id), old_head)

    def test_non_text_content_as_gatherer_vs_alternative_is_asymmetric(self):
        bad_sample = SampleResult({"role": "assistant", "content": [], "tool_calls": []})
        fixture = build_rollout_fixture(
            self.root / "asymmetry-gatherer", mode="none", sample_results=(bad_sample,)
        )
        old_head = fixture.store.read_head(fixture.lineage_id)
        with self.assertRaises(AdapterContractError):
            fixture.driver().run(fixture.runtime, max_steps=1)
        self.assertEqual(fixture.store.read_head(fixture.lineage_id), old_head)

        fixture = build_rollout_fixture(self.root / "asymmetry-alternative", mode="none")
        port = fixture.env.step_input(fixture.runtime)[2]
        turn = WriterTurnV1(
            action_id=port.action_id,
            context_revision_ref=port.context_revision_ref,
            raw_output_ref=None,
            usage={},
            adapter_trace=None,
            message=SampledMessageV1(content=[], tool_calls_was_list=True, calls=[]),
        )
        old_head = fixture.store.read_head(fixture.lineage_id)
        driver = fixture.driver(
            alternatives=lambda directive, _port: (
                turn if directive.kind == "sample_writer" else None
            )
        )
        with self.assertRaises(ProjectionError):
            driver.run(fixture.runtime, max_steps=1)
        self.assertEqual(fixture.store.read_head(fixture.lineage_id), old_head)

    def test_noncanonical_sampled_call_is_invalid_model_behavior_through_driver(self):
        malformed_call = {
            "id": "noncanonical-envelope",
            "type": "other",
            "function": {
                "name": "write_file",
                "arguments": {"path": "bad.txt", "content": b"not canonical"},
            },
        }
        fixture = build_rollout_fixture(self.root / "noncanonical-call", mode="none")

        class Backend:
            def sample(self, _prepared):
                return SampleResult(
                    {"role": "assistant", "content": "", "tool_calls": [malformed_call]}
                )

        fixture.gatherers = make_gatherers(fixture, sampler=Backend())

        with self.assertRaises(DriverBudgetError) as caught:
            fixture.driver().run(fixture.runtime, max_steps=2)

        runtime = caught.exception.runtime
        view = fixture.env.verify(runtime)
        self.assertEqual(runtime.state.files, fixture.runtime.state.files)
        self.assertEqual(runtime.state.continuation["next_call"], 1)
        self.assertEqual(runtime.state.continuation["tool_queue"][0]["name"], "invalid_call")
        assistant = next(
            message for message in view.context.messages if message.role == "assistant"
        )
        self.assertEqual(assistant.content[0]["raw"], {"$noncanonical": "no-sampled-content"})
        budget = fixture.store.get_artifact(runtime.state.budgets_ref)
        self.assertEqual(budget["consumed"]["attempted_tool_calls"], 1)

    def test_malformed_sample_envelope_is_adapter_error_before_lineage_recording(self):
        fixture = build_rollout_fixture(self.root / "malformed-sample-envelope", mode="none")

        class Backend:
            def sample(self, _prepared):
                return SampleResult(
                    {"role": "assistant", "content": "bad envelope", "tool_calls": None}
                )

        fixture.gatherers = make_gatherers(fixture, sampler=Backend())
        head = fixture.store.read_head(fixture.lineage_id)
        checkpoints = tuple(fixture.checkpoint_ids)

        with self.assertRaises(AdapterContractError):
            fixture.driver().run(fixture.runtime, max_steps=1)

        self.assertEqual(fixture.store.read_head(fixture.lineage_id), head)
        self.assertEqual(tuple(fixture.checkpoint_ids), checkpoints)

    def test_noncanonical_sampling_metadata_is_adapter_error_before_lineage_recording(self):
        fixture = build_rollout_fixture(self.root / "noncanonical-sampling-metadata", mode="none")

        class Backend:
            def sample(self, _prepared):
                return SampleResult(
                    {"role": "assistant", "content": "draft", "tool_calls": []},
                    trace={"opaque": b"not canonical"},
                )

        fixture.gatherers = make_gatherers(fixture, sampler=Backend())
        head = fixture.store.read_head(fixture.lineage_id)
        checkpoints = tuple(fixture.checkpoint_ids)

        with self.assertRaises(AdapterContractError):
            fixture.driver().run(fixture.runtime, max_steps=1)

        self.assertEqual(fixture.store.read_head(fixture.lineage_id), head)
        self.assertEqual(tuple(fixture.checkpoint_ids), checkpoints)

    def test_frozen_search_result_is_read_accounted_through_driver(self):
        sample = SampleResult(
            {
                "role": "assistant",
                "content": "Search notes.",
                "tool_calls": [
                    {
                        "id": "search-notes",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": {"query": "moon", "path": "notes"},
                        },
                    }
                ],
            }
        )
        fixture = build_rollout_fixture(self.root / "frozen-search", mode="none")
        local = LocalTextToolProvider()

        class FrozenSearchProvider:
            def execute(self, spec, snapshot, action):
                result = local.execute(spec, snapshot, action)
                observation = dict(result.observation)
                observation["result"] = tuple(
                    MappingProxyType(dict(row)) for row in observation["result"]
                )
                return EnvironmentResult(MappingProxyType(observation), result.snapshot)

        class Backend:
            def sample(self, _prepared):
                return sample

        fixture.gatherers = make_gatherers(fixture, sampler=Backend(), tools=FrozenSearchProvider())

        with self.assertRaises(DriverBudgetError) as caught:
            fixture.driver().run(fixture.runtime, max_steps=2)

        runtime = caught.exception.runtime
        budget = fixture.store.get_artifact(runtime.state.budgets_ref)
        self.assertGreater(budget["consumed"]["read_tokens"], 0)
        action = fixture.store.load_event(runtime.state.history["head"])
        self.assertEqual(action.kind, "tool_result")

    def test_frozen_budget_mapping_does_not_skip_writer_context_storage_charge(self):
        fixture = build_rollout_fixture(self.root / "frozen-writer-budget", mode="none")
        policy_ref = fixture.store.put_artifact(ContextPolicyV1("carry").to_wire())
        selected = False

        def alternatives(directive, _port):
            nonlocal selected
            if directive.kind == "sample_writer" and not selected:
                selected = True
                return ContextOperationInputV1(policy_ref)
            return None

        driver = fixture.driver(alternatives=alternatives)
        with self.assertRaises(DriverBudgetError) as caught:
            driver.run(fixture.runtime, max_steps=2)

        runtime = caught.exception.runtime
        budget = fixture.store.get_artifact(runtime.state.budgets_ref)
        self.assertGreater(budget["consumed"]["context_storage_bytes"], 0)
        self.assertGreater(budget["consumed"]["context_bytes"], 0)

    def test_evaluator_family_mismatch_is_rejected_before_dispatch(self):
        fixture = build_rollout_fixture(self.root / "evaluator")
        check = next(iter(fixture.entry.graph.node(fixture.entry.node_id).checks.values()))

        class WrongFamilyEvaluator:
            family = "fixture-file-count-v1"
            calls = 0

            def evaluate(self, _request):
                self.calls += 1
                return EvaluationEvidenceV1(self.family, "pass", {})

        evaluator = WrongFamilyEvaluator()
        packet = fixture.entry.graph.node(fixture.entry.node_id).evaluator_packet
        packet_ref = packet.identity()
        runner = CheckRunner(
            fixture.store,
            evaluator,
            {check.identity(): check},
        )
        request = {
            "check_contract_hash": check.identity(),
            "check_id": check.id,
            "evaluator_packet_ref": packet_ref,
            "target_checkpoint": fixture.runtime.checkpoint_id,
        }
        port = CheckInput("request-ref", request, {}, packet.to_dict())
        old_head = fixture.store.read_head(fixture.lineage_id)
        with self.assertRaises(AdapterContractError):
            runner.result(port)
        self.assertEqual(evaluator.calls, 0)
        self.assertEqual(fixture.store.read_head(fixture.lineage_id), old_head)

    def test_evaluator_family_claim_is_bound_by_verifier(self):
        fixture = build_rollout_fixture(self.root / "evaluator-family-claim")
        check = next(iter(fixture.entry.graph.node(fixture.entry.node_id).checks.values()))
        request = EvaluationRequestV1.create(
            "deterministic-file-v1",
            "target-checkpoint",
            replace(check, evaluator_version="fixture-file-count-v1"),
            "packet-ref",
            {},
        )

        def decode(status, body):
            return DecodedEvidence(status, body)

        def accept(_request, _evidence, _resolver):
            return None

        other_family = EvidenceFamily(
            "other-family-v1",
            "OtherEvidenceV1",
            "fixture-file-count-v1",
            None,
            None,
            decode,
            accept,
        )
        wire = {
            "record_type": other_family.record_type,
            "schema": 1,
            "target_checkpoint": request.target_checkpoint,
            "check_contract_hash": request.check.identity(),
            "evaluator_packet_ref": request.evaluator_packet_ref,
            "evidence": {},
            "status": "pass",
        }
        with (
            patch(
                "writing_agent.task_graph_evaluation.FAMILIES",
                MappingProxyType({**FAMILIES, other_family.name: other_family}),
            ),
            self.assertRaisesRegex(ProjectionError, "different admitted family"),
        ):
            verify_evaluation_evidence(request, wire)


if __name__ == "__main__":
    unittest.main()
