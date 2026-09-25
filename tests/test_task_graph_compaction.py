"""Context replacement contracts through the immutable store and real projection."""

import unittest
from copy import deepcopy
from dataclasses import replace as replace_record
from unittest.mock import patch

from tests import test_task_graph_scripted as scripted_tests
from tests.task_graph_forgery import forged_effect, forged_event, forged_reduced
from tests.test_task_graph_writer import WriterFixture
from writing_agent.task_graph import ContextRevisionV1, MessageV1
from writing_agent.task_graph_compaction import ContextPolicyV1, completed_exchanges
from writing_agent.task_graph_projection import project_writer_context


def assert_envelope_bound(test, fixture):
    captured = {}

    def capture(*args, **kwargs):
        captured["args"] = args
        raise RuntimeError("capture")

    with patch.object(fixture.store, "publish", side_effect=capture):
        with test.assertRaisesRegex(RuntimeError, "capture"):
            fixture.writer.change_context(
                fixture.runtime,
                ContextPolicyV1(
                    "compact", summarizer_version="visible-text-v1", max_summary_chars=0
                ),
            )
    lineage, head, events, _ = captured["args"][:4]
    original = events[0]
    effect = fixture.store.get_artifact(original.payload_ref)
    cases = {
        "node_visit_id": "false-visit",
        "versions_ref": fixture.store.put_artifact({"false": "versions"}),
        "provenance_ref": fixture.store.put_artifact({"false": "provenance"}),
    }
    for name, value in cases.items():
        with test.subTest(lineage=type(fixture).__name__, field=name):
            event = replace_record(original, **{name: value}, id=None)
            state = forged_reduced(fixture.writer, fixture.runtime.state, event, effect)
            with test.assertRaises(ValueError):
                fixture.store.publish(
                    lineage, head, (event,), state, parent_checkpoint=fixture.start
                )
            with patch("writing_agent.task_graph_projection.project_writer_context"):
                commit = fixture.store.publish(
                    lineage, head, (event,), state, parent_checkpoint=fixture.start
                )
            with test.assertRaises(ValueError):
                fixture.store.restore(
                    fixture.store.load_commit(commit).checkpoint,
                    fixture.root / f"false-envelope-{name}",
                )
            fixture.store._replace_head(lineage, head)


def assert_runtime_log_sequences_are_exact(test, fixture):
    """A context producer must not publish or recover a retyped log ordinal."""
    policy = ContextPolicyV1("compact", summarizer_version="visible-text-v1", max_summary_chars=0)

    def captured_compaction(runtime):
        captured = {}

        def capture(*args, **kwargs):
            captured["args"] = args
            raise RuntimeError("capture")

        with patch.object(fixture.store, "publish", side_effect=capture):
            with test.assertRaisesRegex(RuntimeError, "capture"):
                fixture.writer.change_context(runtime, policy)
        return captured["args"][:4]

    def check_mutations(runtime, candidate, prefix_commits, cases):
        lineage, head, _, final = candidate
        for name, mutate in cases.items():
            with test.subTest(lineage=type(fixture).__name__, mutation=name):
                log = deepcopy(fixture.store.get_artifact(final.external_inputs_ref))
                mutate(log["entries"])
                if name.startswith("float-"):
                    # Float has no canonical task-graph representation to force-persist.
                    with test.assertRaisesRegex(TypeError, "float"):
                        fixture.store.put_artifact(log)
                    test.assertEqual(fixture.store.read_head(lineage), head)
                    continue
                log_ref = fixture.store.put_artifact(log)
                effect = forged_effect(
                    runtime.state,
                    changes={
                        "context_ref": final.context_ref,
                        "budgets_ref": final.budgets_ref,
                        "external_inputs_ref": log_ref,
                    },
                )
                event = forged_event(
                    fixture.writer,
                    runtime.state,
                    "context_changed",
                    fixture.store.put_artifact(effect),
                    actor="environment",
                    audience=("controller", "trainer"),
                )
                state = forged_reduced(fixture.writer, runtime.state, event, effect)
                kwargs = {"parent_checkpoint": fixture.start} if head is None else {}
                try:
                    with test.assertRaises(ValueError):
                        fixture.store.publish(lineage, head, (event,), state, **kwargs)
                    test.assertEqual(fixture.store.read_head(lineage), head)
                    with patch("writing_agent.task_graph_projection.project_writer_context"):
                        commit = fixture.store.publish(lineage, head, (event,), state, **kwargs)
                    checkpoint = fixture.store.load_commit(commit).checkpoint
                    with test.assertRaises(ValueError):
                        fixture.store.restore(checkpoint, fixture.root / f"log-seq-{name}")
                    with test.assertRaises(ValueError):
                        fixture.store.replay(lineage, fixture.start, (*prefix_commits, commit))
                finally:
                    fixture.store._replace_head(lineage, head)

    first_candidate = captured_compaction(fixture.runtime)
    check_mutations(
        fixture.runtime,
        first_candidate,
        (),
        {
            "bool-current": lambda entries: entries[0].update(seq=True),
            "float-current": lambda entries: entries[0].update(seq=1.0),
            "string-current": lambda entries: entries[0].update(seq="1"),
            "negative-current": lambda entries: entries[0].update(seq=-1),
            "gap-current": lambda entries: entries[0].update(seq=2),
            "zero-current": lambda entries: entries[0].update(seq=0),
        },
    )
    first = fixture.writer.change_context(fixture.runtime, policy)
    second = fixture.writer.change_context(first.runtime, policy)
    third_candidate = captured_compaction(second.runtime)
    check_mutations(
        second.runtime,
        third_candidate,
        (first.commit_id, second.commit_id),
        {
            "duplicate-current": lambda entries: entries[-1].update(seq=2),
            "bool-historical": lambda entries: entries[0].update(seq=True),
            "float-historical": lambda entries: entries[0].update(seq=1.0),
            "string-historical": lambda entries: entries[0].update(seq="1"),
            "negative-historical": lambda entries: entries[0].update(seq=-1),
            "gap-historical": lambda entries: entries[0].update(seq=4),
            "duplicate-historical": lambda entries: entries[0].update(seq=2),
            "reordered-historical": lambda entries: entries.__setitem__(
                slice(0, 2), entries[1::-1]
            ),
        },
    )


class ContextOperationsTest(WriterFixture):
    def test_runtime_log_sequences_are_exact(self):
        assert_runtime_log_sequences_are_exact(self, self)

    def test_text_tool_context_event_envelope_is_causal(self):
        assert_envelope_bound(self, self)

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
        effect = forged_effect(
            result.runtime.state,
            changes={
                "context_ref": final.context_ref,
                "budgets_ref": final.budgets_ref,
                "external_inputs_ref": log_ref,
            },
        )
        effect_ref = self.store.put_artifact(effect)
        mutant = forged_event(
            self.writer,
            result.runtime.state,
            "context_changed",
            effect_ref,
            actor="environment",
            audience=("controller", "trainer"),
        )
        mutant_state = forged_reduced(self.writer, result.runtime.state, mutant, effect)
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
                effect = forged_effect(
                    result.runtime.state,
                    changes={
                        "context_ref": final.context_ref,
                        "budgets_ref": final.budgets_ref,
                        "external_inputs_ref": log_ref,
                        **effect_extra,
                    },
                )
                effect_ref = self.store.put_artifact(effect)
                event = forged_event(
                    self.writer,
                    result.runtime.state,
                    "context_changed",
                    effect_ref,
                    actor=actor,
                    audience=("controller", "trainer"),
                )
                state = forged_reduced(self.writer, result.runtime.state, event, effect)
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

    def test_first_operation_untyped_log_cannot_publish_or_recover(self):
        captured = {}

        def capture(*args, **kwargs):
            captured["args"] = args
            raise RuntimeError("capture")

        with patch.object(self.store, "publish", side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                self.writer.change_context(
                    self.runtime,
                    ContextPolicyV1(
                        "compact", summarizer_version="visible-text-v1", max_summary_chars=0
                    ),
                )
        lineage, head, _, final = captured["args"][:4]
        old = self.store.load_context(final.context_ref)
        forged = ContextRevisionV1(
            messages=(
                *old.messages,
                MessageV1(role="user", content=("PRIVATE_CANARY",), origin="forged"),
            ),
            tools=old.tools,
            rendering=old.rendering,
            event_head=old.event_head,
            provenance_refs=old.provenance_refs,
        )
        self.store.persist(forged)
        untyped = self.store.put_artifact({"untyped_log": True})
        effect = forged_effect(
            self.runtime.state,
            changes={
                "context_ref": forged.identity(),
                "budgets_ref": final.budgets_ref,
                "external_inputs_ref": untyped,
            },
        )
        event = forged_event(
            self.writer,
            self.runtime.state,
            "context_changed",
            self.store.put_artifact(effect),
            actor="environment",
            audience=("controller", "trainer"),
        )
        state = forged_reduced(self.writer, self.runtime.state, event, effect)
        with self.assertRaises(ValueError):
            self.store.publish(lineage, head, (event,), state, parent_checkpoint=self.start)
        with patch("writing_agent.task_graph_projection.project_writer_context"):
            commit = self.store.publish(
                lineage, head, (event,), state, parent_checkpoint=self.start
            )
        with self.assertRaises(ValueError):
            self.store.restore(
                self.store.load_commit(commit).checkpoint, self.root / "first-forged"
            )

    def test_verified_future_messages_reject_stale_payload(self):
        stale = {"messages": [message.to_dict() for message in self.runtime.context.messages]}
        compact = self.writer.change_context(
            self.runtime,
            ContextPolicyV1("compact", summarizer_version="visible-text-v1", max_summary_chars=0),
        )
        with self.assertRaisesRegex(ValueError, "messages differ"):
            self.writer.prepare_verified_messages(compact.runtime, stale)
        forged_pin = self.store.put_artifact(
            {
                "record_type": "VerifiedWriterMessagesV1",
                "context_content_hash": compact.runtime.context.content_hash,
                "context_revision_ref": compact.runtime.context.identity(),
                "rendering": dict(compact.runtime.context.rendering),
                "payload_ref": self.store.put_artifact(stale),
            }
        )
        with self.assertRaisesRegex(ValueError, "verified request messages are stale"):
            self.writer.submit_action(
                compact.runtime,
                self.action(content="not sampled"),
                prepared_request_ref=forged_pin,
            )
        exact = {"messages": [message.to_dict() for message in compact.runtime.context.messages]}
        prepared = self.writer.prepare_verified_messages(compact.runtime, exact)
        action = self.writer.submit_action(
            compact.runtime, self.action(content="done"), prepared_request_ref=prepared
        )
        self.assertEqual(
            self.store.get_artifact(prepared)["record_type"], "VerifiedWriterMessagesV1"
        )
        record = self.store.get_artifact(action.record_ref)
        self.assertIs(
            self.store.get_artifact(record["trace_ref"])["native_on_policy_eligible"], False
        )
        self.store.restore(action.runtime.checkpoint_id, self.root / "verified-recovery")

    def test_retyped_log_and_multi_event_batch_do_not_skip_semantics(self):
        _, observed = self._exchange(self.runtime)
        captured = {}

        def capture(*args, **kwargs):
            captured["args"] = args
            raise RuntimeError("capture")

        with patch.object(self.store, "publish", side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                self.writer.change_context(
                    observed.runtime,
                    ContextPolicyV1(
                        "compact", summarizer_version="visible-text-v1", max_summary_chars=8
                    ),
                )
        lineage, head, _, final = captured["args"][:4]
        log = self.store.get_artifact(final.external_inputs_ref)
        record = self.store.get_artifact(log["entries"][-1]["record_ref"])
        false_record = self.store.put_artifact({**record, "summary_text": "forged"})
        for variant in ("retyped-log", "multi-event", "valid-first-invalid-second"):
            with self.subTest(variant=variant):
                changed_log = deepcopy(log)
                if variant != "valid-first-invalid-second":
                    changed_log["entries"][-1]["record_ref"] = false_record
                if variant == "retyped-log":
                    changed_log["entries"][-1]["kind"] = "writer_action"
                effect = forged_effect(
                    observed.runtime.state,
                    changes={
                        "context_ref": final.context_ref,
                        "budgets_ref": final.budgets_ref,
                        "external_inputs_ref": self.store.put_artifact(changed_log),
                    },
                )
                event = forged_event(
                    self.writer,
                    observed.runtime.state,
                    "context_changed",
                    self.store.put_artifact(effect),
                    actor="environment",
                    audience=("controller", "trainer"),
                )
                state = forged_reduced(self.writer, observed.runtime.state, event, effect)
                events = [event]
                if variant in {"multi-event", "valid-first-invalid-second"}:
                    next_effect = forged_effect(state, changes={})
                    extra = forged_event(
                        self.writer,
                        state,
                        "budget_charged",
                        self.store.put_artifact(next_effect),
                        actor="environment",
                        audience=("controller",),
                    )
                    state = forged_reduced(self.writer, state, extra, next_effect)
                    events.append(extra)
                with self.assertRaises(ValueError):
                    self.store.publish(lineage, head, events, state)
                with patch("writing_agent.task_graph_projection.project_writer_context"):
                    commit = self.store.publish(lineage, head, events, state)
                with self.assertRaises(ValueError):
                    self.store.restore(
                        self.store.load_commit(commit).checkpoint,
                        self.root / f"dispatch-{variant}",
                    )
                self.store._replace_head(lineage, head)

    def test_nested_numeric_and_envelope_mutations_reject_both_gates(self):
        _, first = self._exchange(self.runtime)
        _, observed = self._exchange(first.runtime, "read-2")
        captured = {}

        def capture(*args, **kwargs):
            captured["args"] = args
            raise RuntimeError("capture")

        with patch.object(self.store, "publish", side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                self.writer.change_context(
                    observed.runtime,
                    ContextPolicyV1(
                        "compact",
                        retained_exchanges=1,
                        summarizer_version="visible-text-v1",
                        max_summary_chars=0,
                    ),
                )
        lineage, head, _, final = captured["args"][:4]
        log = self.store.get_artifact(final.external_inputs_ref)
        record = self.store.get_artifact(log["entries"][-1]["record_ref"])
        mutations = []
        for field, value in record["charges"].items():
            mutations.append(
                (f"charge-{field}", {"charges": {**record["charges"], field: bool(value)}})
            )
        for field, value in record["source_range"].items():
            mutations.append(
                (f"range-{field}", {"source_range": {**record["source_range"], field: bool(value)}})
            )
        mutations.append(("config", {"summarizer_config": {"max_chars": False}}))
        for field in ("old_messages", "new_messages", "dropped_messages", "retained_tail"):
            for position, item in enumerate(record[field]):
                values = deepcopy(record[field])
                values[position]["index"] = bool(item["index"])
                mutations.append((f"{field}-{position}", {field: values}))
        for name, claims in mutations:
            with self.subTest(name=name):
                changed = {**record, **claims}
                changed_log = deepcopy(log)
                changed_log["entries"][-1]["record_ref"] = self.store.put_artifact(changed)
                effect = forged_effect(
                    observed.runtime.state,
                    changes={
                        "context_ref": final.context_ref,
                        "budgets_ref": final.budgets_ref,
                        "external_inputs_ref": self.store.put_artifact(changed_log),
                    },
                )
                event = forged_event(
                    self.writer,
                    observed.runtime.state,
                    "context_changed",
                    self.store.put_artifact(effect),
                    actor="environment",
                    audience=("controller", "trainer"),
                )
                state = forged_reduced(self.writer, observed.runtime.state, event, effect)
                with self.assertRaises(ValueError):
                    self.store.publish(lineage, head, (event,), state)
                with patch("writing_agent.task_graph_projection.project_writer_context"):
                    commit = self.store.publish(lineage, head, (event,), state)
                with self.assertRaises(ValueError):
                    self.store.restore(
                        self.store.load_commit(commit).checkpoint, self.root / f"numeric-{name}"
                    )
                self.store._replace_head(lineage, head)
        original = self.store.get_artifact(log["entries"][-1]["record_ref"])
        self.assertEqual(original, record)


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

    def test_runtime_log_sequences_are_exact(self):
        assert_runtime_log_sequences_are_exact(self, self.fixture)

    def test_scripted_context_event_envelope_is_causal(self):
        assert_envelope_bound(self, self.fixture)

    def test_private_canaries_never_enter_honest_context_operations(self):
        node = self.writer.graph.node("legacy-writer")
        canaries = {
            "requirement": "PRIVATE_REQUIREMENT_CANARY",
            "preference": "PRIVATE_PREFERENCE_CANARY",
            "binding": "PRIVATE_BINDING_CANARY",
            "script": "PRIVATE_SCRIPT_CANARY",
            "evaluator": "PRIVATE_EVALUATOR_CANARY",
            "check": "PRIVATE_CHECK_CANARY",
            "reward": "PRIVATE_REWARD_CANARY",
            "import": "PRIVATE_IMPORT_CANARY",
            "sibling": "PRIVATE_SIBLING_CANARY",
            "artifact": "PRIVATE_ARTIFACT_CANARY",
        }
        packet = replace_record(
            node.author_packet,
            requirements={"r1": canaries["requirement"]},
            preferences={canaries["binding"]: canaries["preference"]},
        )
        bindings = replace_record(node.decision_bindings, bindings={"door": canaries["binding"]})
        answer = dict(node.script.answers["door"])
        answer["value"] = canaries["preference"]
        answer["utterance"] = f"{canaries['script']} {canaries['preference']}"
        script = replace_record(node.script, answers={"door": answer})
        check = replace_record(
            node.checks["nonempty"],
            id=canaries["reward"],
            spec={
                **node.checks["nonempty"].spec,
                "id": canaries["reward"],
                "path": canaries["check"],
            },
        )
        optional = replace_record(
            node.checks["nonempty"],
            id=canaries["evaluator"],
            required=False,
            spec={
                **node.checks["nonempty"].spec,
                "id": canaries["evaluator"],
                "required": False,
            },
        )
        reward = scripted_tests.RewardContractV1(components={canaries["reward"]: 10000})
        evaluation = scripted_tests.EvaluatorPacketV1(
            reward_contract_ref=reward.identity(),
            check_ids=(canaries["reward"], canaries["evaluator"]),
        )
        for item in (packet, bindings, script, check, optional, reward, evaluation):
            self.store.put_artifact(item.to_dict(), private=True)
        contract = replace_record(
            node.contract,
            interaction=replace_record(
                node.contract.interaction_contract,
                author_packet_ref=packet.identity(),
                script_ref=script.identity(),
                decision_bindings_ref=bindings.identity(),
            ),
            mandatory_checks=(check.identity(),),
            optional_checks=(optional.identity(),),
            completion=replace_record(
                node.contract.completion_contract,
                required_check_ids=(canaries["reward"],),
                evaluation_packet_ref=evaluation.identity(),
            ),
        )
        self.store.put_artifact(contract.to_dict())
        instance = replace_record(
            self.writer.graph.instance,
            nodes=(replace_record(node.spec, entry_contract=contract.identity()),),
        )
        self.store.persist(instance)
        graph = scripted_tests.admit_graph(
            instance, scripted_tests.StoreArtifactResolver(self.store)
        )
        sibling = self.runtime.context
        sibling = ContextRevisionV1(
            messages=(
                *sibling.messages,
                MessageV1(role="user", content=(canaries["sibling"],), origin="sibling"),
            ),
            tools=sibling.tools,
            rendering=sibling.rendering,
        )
        self.store.persist(sibling)
        sibling_state = self.runtime.state.to_dict()
        sibling_state["position"]["lineage_id"] = "sibling"
        sibling_state["context_ref"] = sibling.identity()
        sibling_checkpoint = self.store.save_checkpoint(
            scripted_tests.EnvironmentStateV1.from_dict(sibling_state)
        )
        imported = self.store.put_artifact({"private_marker": canaries["import"]})
        unrelated = self.store.put_artifact(
            {"private_marker": canaries["artifact"], "scope": "private"}, private=True
        )
        state = self.runtime.state.to_dict()
        state["instance_ref"] = instance.identity()
        state["position"]["entry_contract"] = contract.identity()
        state["author_packet_ref"] = packet.identity()
        state["requirements_ref"] = self.store.put_artifact(
            {
                "record_type": "RequirementLedgerV1",
                "schema": 1,
                "active": {"r1": canaries["requirement"]},
                "superseded": {},
            }
        )
        state["history"]["imported_refs"] = [sibling_checkpoint, imported]
        state["provenance_ref"] = self.store.put_artifact({"private_marker": canaries["artifact"]})
        self.start = self.store.save_checkpoint(scripted_tests.EnvironmentStateV1.from_dict(state))
        self.runtime = self.store.restore(self.start, self.root / "canary-entry")
        self.writer = scripted_tests.TransactionalWriterV1(
            self.store, graph, "rollout-1", self.start
        )
        self.assertEqual(
            self.store.get_artifact(unrelated, private=True)["private_marker"],
            canaries["artifact"],
        )
        action = self.writer.submit_action(
            self.runtime, self.action(self.call("read_file", {"path": "draft.txt"}))
        )
        observed = self.writer.step_tool(action.runtime)
        compact = self.writer.change_context(
            observed.runtime,
            ContextPolicyV1("compact", summarizer_version="visible-text-v1", max_summary_chars=500),
        )
        stale = {"messages": [message.to_dict() for message in self.runtime.context.messages]}
        with self.assertRaisesRegex(ValueError, "messages differ"):
            self.writer.prepare_verified_messages(compact.runtime, stale)
        carry = self.writer.change_context(compact.runtime, ContextPolicyV1("carry"))
        dropped = self.writer.change_context(carry.runtime, ContextPolicyV1("drop"))
        seeded = self.writer.change_context(
            dropped.runtime,
            ContextPolicyV1(
                "seed",
                seed_name="visible-ancestor",
                seed_checkpoint_ref=observed.runtime.checkpoint_id,
            ),
        )
        request = {"messages": [message.to_dict() for message in seeded.runtime.context.messages]}
        prepared = self.writer.prepare_verified_messages(seeded.runtime, request)
        evidence = str(
            (
                compact.runtime.context.to_dict(),
                carry.runtime.context.to_dict(),
                dropped.runtime.context.to_dict(),
                seeded.runtime.context.to_dict(),
                self.store.get_artifact(compact.record_ref)["summary_text"],
                self.store.get_artifact(self.store.get_artifact(prepared)["payload_ref"]),
            )
        )
        for name, canary in canaries.items():
            with self.subTest(surface=name):
                self.assertNotIn(canary, evidence)

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
