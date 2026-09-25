"""Context replacement contracts through the immutable store and real projection."""

import unittest
from copy import deepcopy
from unittest.mock import patch

from tests import test_task_graph_scripted as scripted_tests
from tests.test_task_graph_writer import WriterFixture
from writing_agent.task_graph import MessageV1
from writing_agent.task_graph_compaction import ContextPolicyV1, completed_exchanges
from writing_agent.task_graph_projection import project_writer_context


class ContextOperationsTest(WriterFixture):
    def _exchange(self, runtime, call_id="read-1", *, content=""):
        action = self.writer.submit_action(
            runtime,
            self.action(self.call("read_file", {"path": "draft.txt"}, call_id), content=content),
        )
        return action, self.writer.step_tool(action.runtime)

    def test_carry_compact_drop_and_named_seed_preserve_history(self):
        action, result = self._exchange(self.runtime)
        old_context = result.runtime.context
        old_action = self.store.get_artifact(action.record_ref)
        source_checkpoint = result.runtime.checkpoint_id

        carry = self.writer.change_context(result.runtime, ContextPolicyV1("carry"))
        self.assertEqual(carry.runtime.context.messages, old_context.messages)
        self.assertEqual(self.store.get_artifact(carry.record_ref)["dropped_messages"], [])

        compact = self.writer.change_context(
            carry.runtime,
            ContextPolicyV1("compact", summarizer_version="visible-text-v1", max_summary_chars=0),
        )
        record = self.store.get_artifact(compact.record_ref)
        self.assertEqual(record["summary_text"], "")
        self.assertEqual(
            self.store.get_artifact(record["summary_ref"], expected_domain="payload:bytes"), b""
        )
        self.assertEqual(len(record["source_event_ids"]), 2)
        self.assertEqual(len(compact.runtime.context.messages), 3)
        self.assertEqual(compact.runtime.context.messages[2].content[0]["text"], "")

        dropped = self.writer.change_context(compact.runtime, ContextPolicyV1("drop"))
        self.assertEqual(dropped.runtime.context.messages, old_context.messages[:2])
        seeded = self.writer.change_context(
            dropped.runtime,
            ContextPolicyV1(
                "seed", seed_name="before-compaction", seed_checkpoint_ref=source_checkpoint
            ),
        )
        seed_record = self.store.get_artifact(seeded.record_ref)
        self.assertEqual(seed_record["seed_name"], "before-compaction")
        self.assertEqual(seed_record["seed_checkpoint_ref"], source_checkpoint)
        self.assertEqual(len(seed_record["source_event_ids"]), 2)
        self.assertEqual(len(seeded.runtime.context.messages), len(old_context.messages))
        self.assertTrue(
            all(not message.loss_eligible for message in seeded.runtime.context.messages)
        )
        self.assertEqual(self.store.get_artifact(action.record_ref), old_action)
        self.assertEqual(self.store.load_context(old_context.identity()), old_context)
        self.assertEqual(
            project_writer_context(self.store, self.start, seeded.runtime.checkpoint_id),
            seeded.runtime.context,
        )
        commits = [carry.commit_id, compact.commit_id, dropped.commit_id, seeded.commit_id]
        with patch(
            "writing_agent.task_graph_compaction.fixed_summary",
            side_effect=AssertionError("summarizer must not run during replay"),
        ):
            self.assertEqual(
                self.store.replay("rollout-1", source_checkpoint, commits),
                seeded.runtime.checkpoint_id,
            )

    def test_unicode_summary_and_complete_retained_tail(self):
        _, first = self._exchange(self.runtime, content="雪と月")
        _, second = self._exchange(first.runtime, "read-2", content="éclair")
        compact = self.writer.change_context(
            second.runtime,
            ContextPolicyV1(
                "compact",
                retained_exchanges=1,
                summarizer_version="visible-text-v1",
                max_summary_chars=500,
            ),
        )
        record = self.store.get_artifact(compact.record_ref)
        self.assertIn("雪と月", record["summary_text"])
        self.assertNotIn("éclair", record["summary_text"])
        self.assertEqual(len(record["retained_tail"]), 2)
        self.assertEqual(
            [message.role for message in compact.runtime.context.messages[-2:]],
            ["assistant", "tool"],
        )
        self.assertEqual(
            self.store.get_artifact(record["summary_ref"], expected_domain="payload:bytes"),
            record["summary_text"].encode("utf-8"),
        )

    def test_immediate_successive_compactions_use_prior_summary_event(self):
        _, result = self._exchange(self.runtime)
        first = self.writer.change_context(
            result.runtime,
            ContextPolicyV1("compact", summarizer_version="visible-text-v1", max_summary_chars=0),
        )
        second = self.writer.change_context(
            first.runtime,
            ContextPolicyV1("compact", summarizer_version="visible-text-v1", max_summary_chars=5),
        )
        record = self.store.get_artifact(second.record_ref)
        self.assertEqual(record["source_event_ids"], [first.event_id])
        self.assertEqual(record["summary_text"], "user:")
        with patch(
            "writing_agent.task_graph_compaction.fixed_summary",
            side_effect=AssertionError("offline replay reran summary generator"),
        ):
            self.assertEqual(
                self.store.replay(
                    "rollout-1",
                    result.runtime.checkpoint_id,
                    (first.commit_id, second.commit_id),
                ),
                second.runtime.checkpoint_id,
            )

    def test_queued_call_and_checking_are_not_quiescent(self):
        action = self.writer.submit_action(
            self.runtime, self.action(self.call("read_file", {"path": "draft.txt"}))
        )
        with self.assertRaisesRegex(ValueError, "quiescent"):
            self.writer.change_context(action.runtime, ContextPolicyV1("drop"))
        result = self.writer.step_tool(action.runtime)
        final = self.writer.submit_action(result.runtime, self.action(content="done"))
        with self.assertRaisesRegex(ValueError, "quiescent"):
            self.writer.change_context(final.runtime, ContextPolicyV1("drop"))

    def test_context_storage_exhaustion_preserves_paid_action_then_valid_stop(self):
        _, first = self._exchange(self.runtime)
        compact = self.writer.change_context(
            first.runtime,
            ContextPolicyV1(
                "compact",
                summarizer_version="visible-text-v1",
                max_summary_chars=0,
                max_context_storage_bytes=5000,
            ),
        )
        action, second = self._exchange(compact.runtime, "read-2")
        budget = self.store.get_artifact(second.runtime.state.budgets_ref)
        self.assertGreater(
            budget["consumed"]["context_storage_bytes"],
            budget["limits"]["context_storage_bytes"],
        )
        with self.assertRaisesRegex(ValueError, "context_storage_bytes budget exhausted"):
            self.writer.prepare_request(second.runtime, {"prompt": "would spend"})
        stopped = self.writer.stop_exhausted(second.runtime)
        self.assertEqual(
            self.store.get_artifact(stopped.record_ref)["stop_reason"],
            "context_storage_budget",
        )
        self.assertIn(
            self.store.get_artifact(action.record_ref)["action_id"],
            stopped.runtime.state.history["action_ids"],
        )
        with self.assertRaisesRegex(ValueError, "quiescent"):
            self.writer.change_context(stopped.runtime, ContextPolicyV1("drop"))

    def test_unmatched_observation_has_no_complete_exchange(self):
        unmatched = MessageV1(
            role="tool",
            origin="action:missing",
            call_id="call:missing",
            content=({"type": "tool_result", "call_id": "call:missing", "content": "x"},),
        )
        with self.assertRaisesRegex(ValueError, "unmatched"):
            completed_exchanges((*self.runtime.context.messages, unmatched))

    def test_operation_and_context_capacity_reject_before_publication(self):
        _, result = self._exchange(self.runtime)
        head = self.store.read_head("rollout-1")
        with self.assertRaisesRegex(ValueError, "declared budget"):
            self.writer.change_context(
                result.runtime,
                ContextPolicyV1(
                    "compact",
                    summarizer_version="visible-text-v1",
                    max_summary_chars=8,
                    max_context_bytes=1,
                ),
            )
        self.assertEqual(self.store.read_head("rollout-1"), head)
        once = self.writer.change_context(
            result.runtime,
            ContextPolicyV1(
                "compact",
                summarizer_version="visible-text-v1",
                max_summary_chars=8,
                max_operations=1,
            ),
        )
        with self.assertRaisesRegex(ValueError, "declared budget"):
            self.writer.change_context(once.runtime, ContextPolicyV1("drop", max_operations=1))
        self.assertEqual(self.store.read_head("rollout-1"), once.commit_id)

    def test_forged_summary_rejected_before_head_and_on_forced_recovery(self):
        _, result = self._exchange(self.runtime)
        captured = {}

        def capture(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            raise RuntimeError("capture")

        with patch.object(self.store, "publish", side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                self.writer.change_context(
                    result.runtime,
                    ContextPolicyV1(
                        "compact", summarizer_version="visible-text-v1", max_summary_chars=12
                    ),
                )
        lineage, head, events, final = captured["args"][:4]
        record_ref = self.store.get_artifact(final.external_inputs_ref)["entries"][-1]["record_ref"]
        forged = deepcopy(self.store.get_artifact(record_ref))
        forged["summary_text"] = "forged"
        forged_ref = self.store.put_artifact(forged)
        log = self.store.get_artifact(final.external_inputs_ref)
        log["entries"][-1]["record_ref"] = forged_ref
        log_ref = self.store.put_artifact(log)
        effect = self.writer._effect(
            result.runtime.state,
            changes={
                "context_ref": final.context_ref,
                "budgets_ref": final.budgets_ref,
                "external_inputs_ref": log_ref,
            },
        )
        effect_ref = self.store.put_artifact(effect)
        mutant = self.writer._event(
            result.runtime.state,
            "context_changed",
            effect_ref,
            actor="environment",
            audience=("controller", "trainer"),
        )
        mutant_state = self.writer._reduced(result.runtime.state, mutant, effect)
        self.assertEqual(self.store.read_head(lineage), head)
        with self.assertRaises(ValueError):
            self.store.publish(
                lineage,
                head,
                (mutant,),
                mutant_state,
                artifact_refs=(forged_ref, log_ref, effect_ref),
            )
        self.assertEqual(self.store.read_head(lineage), head)
        with patch("writing_agent.task_graph_projection.project_writer_context"):
            commit = self.store.publish(
                lineage,
                head,
                (mutant,),
                mutant_state,
                artifact_refs=(forged_ref, log_ref, effect_ref),
            )
        checkpoint = self.store.load_commit(commit).checkpoint
        with self.assertRaises(ValueError):
            self.store.restore(checkpoint, self.root / "forged-recovery")

    def test_forged_provenance_charge_actor_and_status_cannot_publish(self):
        _, result = self._exchange(self.runtime)
        captured = {}

        def capture(*args, **kwargs):
            captured["args"] = args
            raise RuntimeError("capture")

        with patch.object(self.store, "publish", side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                self.writer.change_context(
                    result.runtime,
                    ContextPolicyV1(
                        "compact", summarizer_version="visible-text-v1", max_summary_chars=16
                    ),
                )
        lineage, head, _, final = captured["args"][:4]
        log = self.store.get_artifact(final.external_inputs_ref)
        record = self.store.get_artifact(log["entries"][-1]["record_ref"])
        false_status = self.store.put_artifact(
            {
                "schema": 1,
                "task_status": "complete",
                "execution_status": "valid",
                "stop_reason": None,
                "reward_status": "available",
                "training_eligibility": "eligible",
            }
        )
        cases = (
            ("source_ids", {"source_event_ids": ["0" * 64]}, "environment", {}),
            (
                "source_range",
                {"source_range": {"first_seq": 99, "last_seq": 99}},
                "environment",
                {},
            ),
            ("old_ref", {"old_context_ref": "0" * 64}, "environment", {}),
            ("new_ref", {"new_context_ref": "0" * 64}, "environment", {}),
            ("summary_bytes", {"summary_text": "forged"}, "environment", {}),
            ("summary_config", {"summarizer_config": {"max_chars": 17}}, "environment", {}),
            ("tail", {"retained_tail": record["dropped_messages"]}, "environment", {}),
            ("dropped", {"dropped_messages": []}, "environment", {}),
            (
                "charge",
                {"charges": {**record["charges"], "context_operations": 2}},
                "environment",
                {},
            ),
            ("seed", {"seed_name": "forged"}, "environment", {}),
            ("actor", {}, "author", {}),
            ("effect", {}, "environment", {"outcome_ref": result.runtime.state.outcome_ref}),
            ("status", {}, "environment", {"outcome_ref": false_status}),
        )
        for name, claims, actor, effect_extra in cases:
            with self.subTest(name=name):
                forged = {**record, **claims}
                forged_ref = self.store.put_artifact(forged)
                forged_log = deepcopy(log)
                forged_log["entries"][-1]["record_ref"] = forged_ref
                log_ref = self.store.put_artifact(forged_log)
                effect = self.writer._effect(
                    result.runtime.state,
                    changes={
                        "context_ref": final.context_ref,
                        "budgets_ref": final.budgets_ref,
                        "external_inputs_ref": log_ref,
                        **effect_extra,
                    },
                )
                effect_ref = self.store.put_artifact(effect)
                event = self.writer._event(
                    result.runtime.state,
                    "context_changed",
                    effect_ref,
                    actor=actor,
                    audience=("controller", "trainer"),
                )
                state = self.writer._reduced(result.runtime.state, event, effect)
                with self.assertRaises(ValueError):
                    self.store.publish(
                        lineage,
                        head,
                        (event,),
                        state,
                        artifact_refs=(forged_ref, log_ref, effect_ref),
                    )
                self.assertEqual(self.store.read_head(lineage), head)
        with patch("writing_agent.task_graph_projection.project_writer_context"):
            forced_commit = self.store.publish(
                lineage,
                head,
                (event,),
                state,
                artifact_refs=(forged_ref, log_ref, effect_ref),
            )
        forced_checkpoint = self.store.load_commit(forced_commit).checkpoint
        with self.assertRaises(ValueError):
            self.store.restore(forced_checkpoint, self.root / "forged-status-recovery")

    def test_interruption_on_both_sides_of_head_publication(self):
        _, result = self._exchange(self.runtime)
        policy = ContextPolicyV1(
            "compact", summarizer_version="visible-text-v1", max_summary_chars=32
        )
        old_head = self.store.read_head("rollout-1")
        with patch.object(self.store, "_replace_head", side_effect=OSError("before replace")):
            with self.assertRaisesRegex(OSError, "before replace"):
                self.writer.change_context(result.runtime, policy)
        self.assertEqual(self.store.read_head("rollout-1"), old_head)
        self.assertEqual(
            self.store.restore(result.runtime.checkpoint_id, self.root / "before-recovery").context,
            result.runtime.context,
        )

        replace = self.store._replace_head

        def after_replace(lineage, head):
            replace(lineage, head)
            raise OSError("after replace")

        with patch.object(self.store, "_replace_head", side_effect=after_replace):
            with self.assertRaisesRegex(OSError, "after replace"):
                self.writer.change_context(result.runtime, policy)
        new_head = self.store.read_head("rollout-1")
        self.assertNotEqual(new_head, old_head)
        checkpoint = self.store.load_commit(new_head).checkpoint
        restored = self.store.restore(checkpoint, self.root / "after-recovery")
        self.assertEqual(len(restored.context.messages), 3)
        self.assertEqual(
            project_writer_context(self.store, self.start, checkpoint), restored.context
        )


class ScriptedContextBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.fixture = scripted_tests.ScriptedFixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def __getattr__(self, name):
        fixture = self.__dict__.get("fixture")
        if fixture is None:
            raise AttributeError(name)
        return getattr(fixture, name)

    def test_author_and_check_phases_reject_compaction(self):
        ask = self.call(
            "ask_author",
            {"question": "Which?", "decision_ids": ["door"], "proposals": [], "option_refs": []},
        )
        action = self.writer.submit_action(self.runtime, self.action(ask))
        requested = self.writer.step_tool(action.runtime)
        with self.assertRaisesRegex(ValueError, "quiescent"):
            self.writer.change_context(requested.runtime, ContextPolicyV1("drop"))
        replied = self.author.reply(requested.runtime)
        compact = self.writer.change_context(
            replied.runtime,
            ContextPolicyV1("compact", summarizer_version="visible-text-v1", max_summary_chars=64),
        )
        self.assertNotIn(
            "Never reveal this hidden requirement", str(compact.runtime.context.messages)
        )
        dropped = self.writer.change_context(compact.runtime, ContextPolicyV1("drop"))
        for field in ("files", "requirements_ref", "decisions_ref", "disclosures_ref"):
            self.assertEqual(
                getattr(dropped.runtime.state, field),
                getattr(replied.runtime.state, field),
            )
        final = self.writer.submit_action(dropped.runtime, self.action(content="done"))
        batch = self.checks.request_checks(final.runtime)
        with self.assertRaisesRegex(ValueError, "quiescent"):
            self.writer.change_context(batch.runtime, ContextPolicyV1("drop"))

    def test_private_packet_and_prepared_action_evidence_do_not_enter_summary(self):
        prepared = self.writer.prepare_request(self.runtime, {"prompt": "exact"})
        action = self.writer.submit_action(
            self.runtime,
            self.action(self.call("read_file", {"path": "draft.txt"})),
            prepared_request_ref=prepared,
            raw_output=b"read_file",
            trace={"model": "test-model", "seed": 7, "generated_token_ids": [1]},
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )
        result = self.writer.step_tool(action.runtime)
        old_action = self.store.get_artifact(action.record_ref)
        old_trace = self.store.get_artifact(old_action["trace_ref"])
        compact = self.writer.change_context(
            result.runtime,
            ContextPolicyV1(
                "compact", summarizer_version="visible-text-v1", max_summary_chars=1000
            ),
        )
        summary = self.store.get_artifact(compact.record_ref)["summary_text"]
        for private in ("Never reveal this hidden requirement", "the blue door", "reward=999"):
            self.assertNotIn(private, summary)
        self.assertEqual(self.store.get_artifact(action.record_ref), old_action)
        self.assertEqual(self.store.get_artifact(old_action["trace_ref"]), old_trace)
        self.assertEqual(old_action["prepared_request_ref"], prepared)
        self.assertEqual(
            project_writer_context(self.store, self.start, compact.runtime.checkpoint_id),
            compact.runtime.context,
        )

    def test_paid_phase5_context_overrun_has_incomplete_available_reward(self):
        first = self.writer.submit_action(
            self.runtime, self.action(self.call("read_file", {"path": "draft.txt"}))
        )
        observed = self.writer.step_tool(first.runtime)
        compact = self.writer.change_context(
            observed.runtime,
            ContextPolicyV1(
                "compact",
                summarizer_version="visible-text-v1",
                max_summary_chars=0,
                max_context_storage_bytes=10_000,
            ),
        )
        second = self.writer.submit_action(
            compact.runtime,
            self.action(self.call("read_file", {"path": "draft.txt"}, "read-2")),
        )
        paid_result = self.writer.step_tool(second.runtime)
        stopped = self.writer.stop_exhausted(paid_result.runtime)
        outcome = self.store.get_artifact(stopped.record_ref)
        self.assertEqual(outcome["stop_reason"], "context_storage_budget")
        self.assertEqual(outcome["execution_status"], "valid")
        self.assertEqual(outcome["task_status"], "incomplete")
        rewarded = self.terminal.reward(stopped.runtime)
        self.assertEqual(
            self.store.get_artifact(rewarded.runtime.state.outcome_ref)["reward_status"],
            "available",
        )
        self.assertIn(
            self.store.get_artifact(second.record_ref)["action_id"],
            rewarded.runtime.state.history["action_ids"],
        )
