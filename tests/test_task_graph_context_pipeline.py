"""Context operations through the verified rollout pipeline."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.task_graph_rollout_fixtures import CANARIES, build_rollout_fixture, run_slice
from writing_agent.task_graph import EventV1
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_environment import derive_input
from writing_agent.task_graph_errors import DriverBudgetError, ProjectionError
from writing_agent.task_graph_gate import StoreArtifactReader
from writing_agent.task_graph_group import POLICY_FIELDS, GroupCoordinatorV1
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_record_contracts import ContextPolicyV1 as SealedContextPolicyV1
from writing_agent.task_graph_records import ContextOperationInputV1


def _write_sample(text: str, call_id: str, contents: str) -> SampleResult:
    return SampleResult(
        {
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": {"path": "draft.txt", "content": contents},
                    },
                }
            ],
        }
    )


def _read_sample(call_id: str) -> SampleResult:
    return SampleResult(
        {
            "role": "assistant",
            "content": f"Read draft {call_id}.",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "draft.txt"}},
                }
            ],
        }
    )


def _events(fixture) -> tuple[EventV1, ...]:
    commit_id = fixture.store.read_head(fixture.lineage_id)
    events = []
    while commit_id is not None:
        commit = fixture.store.load_commit(commit_id)
        events.extend(fixture.store.load_event(identity) for identity in commit.events)
        commit_id = commit.parent_commit
    return tuple(reversed(events))


class _Rollout:
    def __init__(self, fixture) -> None:
        self.fixture = fixture
        self.runtime = fixture.runtime
        self.calls = 0
        self.policy_refs: dict[int, str] = {}
        self.selected: list[int] = []
        self.bases = {}
        self.driver = fixture.driver(alternatives=self._alternative)

    def schedule(self, call: int, policy: ContextPolicyV1) -> str:
        policy_ref = self.fixture.store.put_artifact(policy.to_wire())
        self.policy_refs[call] = policy_ref
        return policy_ref

    def _alternative(self, directive, _port):
        if directive.kind != "sample_writer":
            return None
        self.calls += 1
        policy_ref = self.policy_refs.get(self.calls)
        if policy_ref is None:
            return None
        self.selected.append(self.calls)
        self.bases[self.calls] = {
            "runtime": self.runtime,
            "view": self.fixture.env.verify(self.runtime),
            "budget": self.fixture.store.get_artifact(self.runtime.state.budgets_ref),
            "head": self.fixture.store.read_head(self.fixture.lineage_id),
            "events": _events(self.fixture),
        }
        return ContextOperationInputV1(policy_ref)

    def step(self) -> None:
        try:
            self.runtime = self.driver.run(self.runtime, max_steps=1).runtime
        except DriverBudgetError as exc:
            self.runtime = exc.runtime

    def commit_at(self, call: int):
        for _ in range(40):
            self.step()
            if call in self.selected:
                return self.runtime
        raise AssertionError(f"rollout did not reach context alternative {call}")


class ContextPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _fixture(self, name="rollout", **kwargs):
        fixture = build_rollout_fixture(self.root / name, **kwargs)
        self.assertIsInstance(fixture.env.reader, StoreArtifactReader)
        return fixture

    def _group_member(self, name):
        fixture = self._fixture(name)
        store = fixture.store
        rendering = fixture.runtime.context.rendering
        policy = {
            field: store.put_artifact({"pin": field})
            for field in POLICY_FIELDS
            if field != "rng_derivation_version"
        }
        policy["model_ref"] = store.put_artifact({"model_id": "context-policy-test-v1"})
        policy["context_policy_ref"] = store.put_artifact(SealedContextPolicyV1("carry").to_wire())
        policy.update(
            tokenizer_ref=rendering["tokenizer_ref"],
            template_ref=rendering["template_ref"],
            rng_derivation_version="sha256-domain-v1",
        )
        coordinator = GroupCoordinatorV1(fixture.env, self.root / f"{name}-workers")
        spec = coordinator.seal(
            fixture.runtime.checkpoint_id,
            policy=policy,
            group_seed=771,
            group_sequence=0,
            member_count=2,
        )
        runtime = coordinator.start(spec, 0, policy=policy)
        return fixture, runtime, policy["context_policy_ref"]

    def _assert_context_commit(self, flow: _Rollout, call: int):
        fixture = flow.fixture
        before = flow.bases[call]
        after = fixture.env.verify(flow.runtime)
        event = fixture.store.load_event(flow.runtime.state.history["head"])
        self.assertEqual(event.kind, "context_changed")
        self.assertEqual(event.previous, before["runtime"].state.history["head"])
        self.assertEqual(event.seq, before["runtime"].state.history["seq"] + 1)
        payload = fixture.store.get_artifact(event.payload_ref)
        self.assertEqual(payload["record_type"], "ContextOperationInputV1")
        self.assertEqual(payload["policy_ref"], flow.policy_refs[call])
        self.assertEqual(flow.runtime.state.context_ref, after.context.revision_ref)
        self.assertEqual(flow.runtime.context.messages, after.context.messages)
        budget = fixture.store.get_artifact(flow.runtime.state.budgets_ref)
        old_consumed = before["budget"]["consumed"]
        self.assertEqual(
            budget["consumed"]["context_operations"],
            old_consumed.get("context_operations", 0) + 1,
        )
        self.assertGreater(
            budget["consumed"]["context_storage_bytes"],
            old_consumed.get("context_storage_bytes", 0),
        )
        commit = fixture.store.load_commit(fixture.store.read_head(fixture.lineage_id))
        self.assertEqual(commit.checkpoint, flow.runtime.checkpoint_id)
        self.assertEqual(commit.parent_commit, before["head"])
        self.assertEqual(_events(fixture)[:-1], before["events"])
        return before["view"], after, event, budget

    def test_source_event_context_operation_commits_through_real_reader(self):
        fixture = self._fixture("source-events")
        flow = _Rollout(fixture)
        policy_ref = flow.schedule(2, ContextPolicyV1("carry"))

        flow.commit_at(2)

        before, after, event, budget = self._assert_context_commit(flow, 2)
        self.assertEqual(fixture.store.get_artifact(event.payload_ref)["policy_ref"], policy_ref)
        source_ids = tuple(source for source in before.context.sources if source is not None)
        self.assertTrue(source_ids)
        self.assertTrue(
            all(isinstance(fixture.store.load_event(identity), EventV1) for identity in source_ids)
        )
        self.assertEqual(after.context.messages, before.context.messages)
        self.assertEqual(budget["consumed"]["context_operations"], 1)

    def test_foreign_context_policy_from_alternatives_is_projection_error_at_commit(self):
        fixture, runtime, _sealed_policy_ref = self._group_member("fresh-foreign-policy")
        foreign_policy_ref = fixture.store.put_artifact(ContextPolicyV1("drop").to_wire())
        member_id = runtime.state.position["lineage_id"]
        before_head = fixture.store.read_head(member_id)

        driver = fixture.driver(
            alternatives=lambda directive, _port: (
                ContextOperationInputV1(foreign_policy_ref)
                if directive.kind == "sample_writer"
                else None
            )
        )
        with self.assertRaises(ProjectionError) as caught:
            driver.run(runtime, max_steps=1)

        self.assertIn("input.policy_ref", str(caught.exception))
        self.assertEqual(fixture.store.read_head(member_id), before_head)

    def test_published_foreign_context_policy_has_same_projection_path(self):
        fixture, runtime, sealed_policy_ref = self._group_member("persisted-foreign-policy")
        view = fixture.env.verify(runtime)
        honest = derive_input(
            view,
            ContextOperationInputV1(sealed_policy_ref),
            fixture.env.reader,
        )
        for artifact in honest.artifacts:
            fixture.store.persist_artifact(artifact)
        foreign_policy_ref = fixture.store.put_artifact(ContextPolicyV1("drop").to_wire())
        payload_ref = fixture.store.put_artifact(
            ContextOperationInputV1(foreign_policy_ref).to_wire()
        )
        event = replace(honest.event, payload_ref=payload_ref, id=None)
        state = replace(honest.state, history={**honest.state.history, "head": event.id})
        member_id = runtime.state.position["lineage_id"]
        before_head = fixture.store.read_head(member_id)

        with self.assertRaises(ProjectionError) as caught:
            fixture.store.publish(member_id, before_head, (event,), state)

        self.assertIn("input.policy_ref", str(caught.exception))
        self.assertEqual(fixture.store.read_head(member_id), before_head)

    def test_context_policies_and_successive_compactions_commit_on_real_reader(self):
        fixture = self._fixture("policies")
        flow = _Rollout(fixture)
        flow.schedule(2, ContextPolicyV1("carry"))
        flow.schedule(
            3,
            ContextPolicyV1(
                "compact",
                retained_exchanges=1,
                summarizer_version="visible-text-v1",
                max_summary_chars=64,
            ),
        )
        flow.schedule(
            4,
            ContextPolicyV1(
                "compact",
                summarizer_version="visible-text-v1",
                max_summary_chars=5,
            ),
        )
        flow.schedule(5, ContextPolicyV1("drop"))

        flow.commit_at(2)
        original_context = flow.bases[2]["view"].context
        _, carry_after, _, _ = self._assert_context_commit(flow, 2)
        self.assertEqual(carry_after.context.messages, original_context.messages)

        flow.commit_at(3)
        _, first_compact, _, _ = self._assert_context_commit(flow, 3)
        self.assertEqual(first_compact.context.messages[2].role, "user")
        self.assertEqual(first_compact.context.messages[2].content[0]["text"], "")
        self.assertEqual(len(first_compact.context.messages), len(original_context.messages) + 1)

        flow.commit_at(4)
        before_second, second_compact, _, _ = self._assert_context_commit(flow, 4)
        self.assertTrue(any(source is not None for source in before_second.context.sources))
        self.assertEqual(len(second_compact.context.messages), 3)
        self.assertIn("user:", second_compact.context.messages[2].content[0]["text"])

        flow.commit_at(5)
        _, dropped, _, _ = self._assert_context_commit(flow, 5)
        self.assertEqual(dropped.context.messages, original_context.messages[:2])

        seed_ref = flow.schedule(
            6,
            ContextPolicyV1(
                "seed",
                seed_name="completed-exchange",
                seed_checkpoint_ref=flow.bases[2]["runtime"].checkpoint_id,
            ),
        )
        flow.commit_at(6)
        _, seeded, _, _ = self._assert_context_commit(flow, 6)
        self.assertEqual(
            fixture.store.get_artifact(seed_ref)["seed_checkpoint_ref"],
            flow.bases[2]["runtime"].checkpoint_id,
        )
        self.assertEqual(len(seeded.context.messages), len(original_context.messages))
        self.assertTrue(all(not message.loss_eligible for message in seeded.context.messages[2:]))
        self.assertTrue(any(source is not None for source in seeded.context.sources))

    def test_queued_tool_prevents_context_operation_without_publication(self):
        fixture = self._fixture(
            "queued-tool",
            sample_results=(_read_sample("read-queued"),),
        )
        runtime = run_slice(fixture, until=lambda directive: directive.kind == "execute_tool")
        self.assertEqual(next_step(fixture.env.verify(runtime)).kind, "execute_tool")

        policy_ref = fixture.store.put_artifact(ContextPolicyV1("drop").to_wire())
        before_head = fixture.store.read_head(fixture.lineage_id)
        before_events = _events(fixture)
        with self.assertRaisesRegex(ProjectionError, "not an accepted directive alternative"):
            fixture.env.commit(runtime, ContextOperationInputV1(policy_ref))

        self.assertEqual(fixture.store.read_head(fixture.lineage_id), before_head)
        self.assertEqual(_events(fixture), before_events)

    def test_compaction_summary_excludes_private_and_writer_evidence(self):
        read = _read_sample("read-private-evidence")
        sample = SampleResult(
            read.message,
            raw_output="RAW_OUTPUT_OUTSIDE_WRITER_CONTEXT",
            trace={"model": "MODEL_TRACE_OUTSIDE_WRITER_CONTEXT", "seed": 7},
        )
        fixture = self._fixture("private-compaction", mode="feedback", sample_results=(sample,))
        for canary in CANARIES.values():
            self.assertIn(canary, repr(fixture.entry.reader.private))

        flow = _Rollout(fixture)
        flow.schedule(
            2,
            ContextPolicyV1(
                "compact",
                summarizer_version="visible-text-v1",
                max_summary_chars=4000,
            ),
        )
        flow.commit_at(2)
        _before, after, event, _budget = self._assert_context_commit(flow, 2)
        self.assertEqual(event.kind, "context_changed")
        self.assertTrue(any(":context:" in message.origin for message in after.context.messages))
        summary = repr(after.context.messages)
        for canary in CANARIES.values():
            self.assertNotIn(canary, summary)
        for evidence in (
            "RAW_OUTPUT_OUTSIDE_WRITER_CONTEXT",
            "MODEL_TRACE_OUTSIDE_WRITER_CONTEXT",
        ):
            self.assertNotIn(evidence, summary)

    def test_unicode_summary_preserves_a_complete_retained_tail(self):
        fixture = self._fixture(
            "unicode-tail",
            sample_results=(
                _write_sample("雪と月", "write-snow", "moon"),
                _write_sample("éclair", "write-tail", "sun"),
            ),
        )
        flow = _Rollout(fixture)
        flow.schedule(
            3,
            ContextPolicyV1(
                "compact",
                retained_exchanges=1,
                summarizer_version="visible-text-v1",
                max_summary_chars=500,
            ),
        )

        flow.commit_at(3)

        before, after, _, _ = self._assert_context_commit(flow, 3)
        summary = after.context.messages[2].content[0]["text"]
        self.assertIn("雪と月", summary)
        self.assertNotIn("éclair", summary)
        for canary in fixture.canaries.values():
            self.assertNotIn(canary, repr(after.context.messages))
        self.assertEqual(
            [message.role for message in after.context.messages[-2:]], ["assistant", "tool"]
        )
        self.assertIn("éclair", str(after.context.messages[-2].content))
        self.assertTrue(any(source is not None for source in before.context.sources))

    def test_capacity_rejection_preserves_head_context_budget_and_events(self):
        fixture = self._fixture("capacity")
        flow = _Rollout(fixture)
        flow.schedule(
            2,
            ContextPolicyV1(
                "compact",
                retained_exchanges=1,
                summarizer_version="visible-text-v1",
                max_summary_chars=8,
                max_context_bytes=1,
            ),
        )
        flow.step()
        flow.step()
        old_runtime = flow.runtime
        old_head = fixture.store.read_head(fixture.lineage_id)
        old_events = _events(fixture)
        old_budget = fixture.store.get_artifact(old_runtime.state.budgets_ref)
        old_context = old_runtime.context
        self.assertTrue(
            any(source is not None for source in fixture.env.verify(old_runtime).context.sources)
        )

        with self.assertRaises(ProjectionError):
            flow.step()

        self.assertEqual(flow.runtime, old_runtime)
        self.assertEqual(fixture.store.read_head(fixture.lineage_id), old_head)
        self.assertEqual(_events(fixture), old_events)
        view = fixture.env.verify(old_runtime)
        self.assertEqual(view.context.messages, old_context.messages)
        self.assertEqual(fixture.store.get_artifact(old_runtime.state.budgets_ref), old_budget)

    def test_author_and_check_phase_rejections_are_non_mutating(self):
        for name, phase in (("author", "await_author_reply"), ("check", "await_check_result")):
            with self.subTest(phase=name):
                fixture = self._fixture(name)
                runtime = run_slice(
                    fixture,
                    until=lambda directive, phase=phase: directive.kind == phase,
                )
                policy_ref = fixture.store.put_artifact(ContextPolicyV1("drop").to_wire())
                before_view = fixture.env.verify(runtime)
                before_head = fixture.store.read_head(fixture.lineage_id)
                before_events = _events(fixture)
                before_budget = fixture.store.get_artifact(runtime.state.budgets_ref)
                before_checkpoints = tuple(fixture.checkpoint_ids)

                with self.assertRaises(ProjectionError):
                    fixture.env.commit(runtime, ContextOperationInputV1(policy_ref))

                after_view = fixture.env.verify(runtime)
                self.assertEqual(fixture.store.read_head(fixture.lineage_id), before_head)
                self.assertEqual(_events(fixture), before_events)
                self.assertEqual(tuple(fixture.checkpoint_ids), before_checkpoints)
                self.assertEqual(after_view.context.messages, before_view.context.messages)
                self.assertEqual(
                    fixture.store.get_artifact(runtime.state.budgets_ref), before_budget
                )

    def test_paid_context_storage_overrun_stops_after_the_tool_result(self):
        fixture = self._fixture(
            "paid-overrun",
            sample_results=(
                _read_sample("read-first"),
                _read_sample("read-paid"),
                SampleResult({"role": "assistant", "content": "Draft complete.", "tool_calls": []}),
            ),
        )
        flow = _Rollout(fixture)
        flow.schedule(
            2,
            ContextPolicyV1(
                "compact",
                summarizer_version="visible-text-v1",
                max_summary_chars=0,
                max_context_storage_bytes=10_000,
            ),
        )

        flow.commit_at(2)
        _, compacted, operation_event, _ = self._assert_context_commit(flow, 2)
        self.assertEqual(operation_event.kind, "context_changed")
        self.assertEqual(compacted.context.messages[2].content[0]["text"], "")
        for _ in range(20):
            if next_step(fixture.env.verify(flow.runtime)).kind == "done":
                break
            flow.step()
        else:
            self.fail("rollout did not reach its terminal directive")

        events = _events(fixture)
        kinds = [event.kind for event in events]
        operation_index = kinds.index("context_changed")
        paid = events[operation_index + 1 : operation_index + 3]
        self.assertEqual([event.kind for event in paid], ["writer_action", "tool_result"])
        self.assertEqual(kinds[operation_index + 3], "termination_recorded")
        budget = fixture.store.get_artifact(flow.runtime.state.budgets_ref)
        self.assertGreater(
            budget["consumed"]["context_storage_bytes"],
            budget["limits"]["context_storage_bytes"],
        )
        outcome = fixture.store.get_artifact(flow.runtime.state.outcome_ref)
        self.assertEqual(outcome["execution_status"], "valid")
        self.assertEqual(outcome["task_status"], "incomplete")
        self.assertEqual(outcome["reward_status"], "available")
        self.assertEqual(
            flow.runtime.state.history["action_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
