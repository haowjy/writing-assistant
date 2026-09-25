"""High-risk group admission, isolation, exact rewards and action-credit evidence."""

from __future__ import annotations

import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

from tests import test_task_graph_scripted as scripted
from writing_agent.task_graph import EnvironmentStateV1, domain_hash
from writing_agent.task_graph_checks import DeterministicChecksV1
from writing_agent.task_graph_compaction import ContextPolicyV1
from writing_agent.task_graph_group import (
    POLICY_FIELDS,
    GroupCoordinatorV1,
    GroupError,
    GroupMemberResultV1,
    GroupSpecV1,
)
from writing_agent.task_graph_scripted import ScriptedAuthorRuntimeV1
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_terminal import ScriptedTerminalV1
from writing_agent.task_graph_writer import TransactionalWriterV1


class GroupCoordinatorTests(unittest.TestCase):
    def setUp(self):
        fixture = scripted.ScriptedFixture()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.store = fixture.store
        self.start = fixture.start
        self.runtime = fixture.runtime
        self.writer = fixture.writer
        self.action = fixture.action
        self.call = fixture.call
        self.coordinator = GroupCoordinatorV1(self.store, self.root / "group-workers")
        rendering = self.runtime.context.rendering
        self.policy = {
            field: self.store.put_artifact({"pin": field})
            for field in POLICY_FIELDS
            if field != "rng_derivation_version"
        }
        self.policy.update(
            tokenizer_ref=rendering["tokenizer_ref"],
            template_ref=rendering["template_ref"],
            rng_derivation_version="sha256-domain-v1",
        )
        self.compaction_policy = ContextPolicyV1(
            "compact", summarizer_version="visible-text-v1", max_summary_chars=20
        )
        self.policy["context_policy_ref"] = self.store.put_artifact(
            self.compaction_policy.to_dict()
        )

    def group(self, sequence=0, mode="real"):
        return self.coordinator.seal(
            self.start,
            policy=self.policy,
            group_seed=771,
            group_sequence=sequence,
            member_count=2,
            runner_mode=mode,
        )

    def starts(self, spec):
        return tuple(
            self.coordinator.start(spec, ordinal, policy=self.policy) for ordinal in range(2)
        )

    def test_full_contract_drift_and_start_isolation(self):
        receipt = self.root / "atomic-receipt.json"
        with patch("writing_agent.task_graph_group.os.link", side_effect=OSError("fault")):
            with self.assertRaises(OSError):
                self.coordinator._receipt(receipt, {"schema": 1})
        self.assertFalse(receipt.exists())
        self.coordinator._receipt(receipt, {"schema": 1})
        self.assertEqual(receipt.read_bytes(), b'{"schema":1}')
        spec = self.group()
        parent_path = self.store.root / "checkpoints" / f"{self.start}.json"
        parent_bytes = parent_path.read_bytes()
        one, two = self.starts(spec)
        self.assertEqual(one.state.files, two.state.files)
        self.assertEqual(one.context.content_hash, two.context.content_hash)
        self.assertEqual(spec.members[0].environment_seed, spec.members[1].environment_seed)
        self.assertNotEqual(spec.members[0].writer_seed, spec.members[1].writer_seed)
        self.assertNotEqual(one.state.rng_ref, two.state.rng_ref)
        self.assertEqual(parent_bytes, parent_path.read_bytes())
        (one.workspace / "draft.txt").write_text("sibling canary", encoding="utf-8")
        self.assertNotIn("sibling canary", (two.workspace / "draft.txt").read_text())
        self.assertNotIn("sibling canary", str(two.context.messages))
        self.assertEqual(
            self.coordinator.restore_member(spec, 1, self.root / "resumed-member").state,
            two.state,
        )
        self.assertEqual(
            TaskGraphStore(self.store.root).replay(
                spec.members[0].member_id,
                self.start,
                (self.coordinator._start_receipt(spec, 0)["commit_id"],),
            ),
            one.checkpoint_id,
        )
        for field in POLICY_FIELDS:
            changed = dict(self.policy)
            changed[field] = (
                "sha256-domain-v2"
                if field == "rng_derivation_version"
                else self.store.put_artifact({"different": field})
            )
            with self.subTest(policy_field=field), self.assertRaises(GroupError):
                self.coordinator.assert_start_contract(spec, self.start, changed)
        for field in spec.environment:
            changed = dict(spec.environment)
            changed[field] = "different"
            with self.subTest(environment_field=field), self.assertRaises((GroupError, ValueError)):
                GroupSpecV1(
                    group_id=spec.group_id,
                    group_sequence=spec.group_sequence,
                    group_seed=spec.group_seed,
                    runner_mode=spec.runner_mode,
                    environment=changed,
                    policy=dict(spec.policy),
                    members=spec.members,
                )
        state = self.runtime.state.to_dict()
        state["requirements_ref"] = self.store.put_artifact({"drift": "requirements"})
        changed_entry = self.store.save_checkpoint(EnvironmentStateV1.from_dict(state))
        with self.assertRaises(GroupError):
            self.coordinator.assert_start_contract(spec, changed_entry, self.policy)
        self.assertNotEqual(self.group(sequence=1).group_id, spec.group_id)
        self.assertEqual(self.group().identity(), spec.identity())
        larger = self.coordinator.seal(
            self.start,
            policy=self.policy,
            group_seed=771,
            group_sequence=0,
            member_count=3,
        )
        self.assertNotEqual(larger.group_id, spec.group_id)
        reordered = self.group(sequence=2)
        second = self.coordinator.start(reordered, 1, policy=self.policy)
        first = self.coordinator.start(reordered, 0, policy=self.policy)
        self.assertEqual(first.state.files, second.state.files)
        self.assertEqual(
            self.store.get_artifact(first.state.rng_ref)["environment_seed"],
            self.store.get_artifact(second.state.rng_ref)["environment_seed"],
        )
        self.assertNotEqual(
            self.store.get_artifact(first.state.rng_ref)["writer_seed"],
            self.store.get_artifact(second.state.rng_ref)["writer_seed"],
        )
        with self.assertRaises(GroupError):
            self.coordinator.seal(
                self.start, policy=self.policy, group_seed=0, group_sequence=2, member_count=1
            )
        incomplete = dict(self.policy)
        incomplete.pop("adapter_ref")
        with self.assertRaises(GroupError):
            self.coordinator.seal(
                self.start, policy=incomplete, group_seed=0, group_sequence=2, member_count=2
            )

    def test_scripted_rewards_pending_invalid_tie_and_exact_nontie(self):
        spec = self.group(mode="fixture")
        self.starts(spec)
        self.coordinator.collect_scripted(spec, 1, reward=Fraction(3, 2))
        self.assertEqual(self.coordinator.finalize(spec).status, "pending")
        self.coordinator.collect_scripted(spec, 0, reward_status="pending")
        self.assertEqual(self.coordinator.finalize(spec).status, "pending")
        self.coordinator.collect_scripted(spec, 0, reward_status="unavailable")
        pending = self.coordinator.finalize(spec)
        self.assertEqual(pending.status, "pending")
        self.assertEqual(pending.advantage_refs, ())
        self.coordinator.collect_scripted(spec, 0, reward=Fraction(-1, 2))
        ready = self.coordinator.finalize(spec)
        self.assertEqual(ready.status, "ready")
        advantages = [self.store.get_artifact(ref) for ref in ready.advantage_refs]
        self.assertEqual(
            [a["centered"] for a in advantages],
            [
                {"numerator": -1, "denominator": 1},
                {"numerator": 1, "denominator": 1},
            ],
        )
        self.assertEqual(
            [a["variance"] for a in advantages],
            [
                {"numerator": 1, "denominator": 1},
                {"numerator": 1, "denominator": 1},
            ],
        )
        self.assertFalse(ready.segment_credit_refs)
        offline = GroupCoordinatorV1(TaskGraphStore(self.store.root), self.root / "offline-workers")
        self.assertEqual(
            offline.finalize(offline.resume(spec.group_id)).identity(), ready.identity()
        )
        with self.assertRaises(GroupError):
            self.coordinator.collect_scripted(spec, 0, reward=Fraction(99))

        tie = self.group(sequence=1, mode="fixture")
        self.starts(tie)
        self.coordinator.collect_scripted(tie, 0, reward=Fraction(-5))
        self.coordinator.collect_scripted(tie, 1, reward=Fraction(-5))
        decision = self.coordinator.finalize(tie)
        self.assertEqual(decision.status, "tie")
        self.assertTrue(
            all(
                self.store.get_artifact(ref)["centered"]
                == {
                    "numerator": 0,
                    "denominator": 1,
                }
                for ref in decision.advantage_refs
            )
        )
        self.assertTrue(
            all(
                self.store.get_artifact(ref)["advantage"] == {"numerator": 0, "denominator": 1}
                and self.store.get_artifact(ref)["expression"] == "zero"
                for ref in decision.advantage_refs
            )
        )
        invalid = self.group(sequence=2, mode="fixture")
        self.starts(invalid)
        self.coordinator.collect_scripted(invalid, 0, reward=Fraction(-10))
        self.coordinator.collect_scripted(invalid, 1, execution_status="infrastructure_invalid")
        decision = self.coordinator.finalize(invalid)
        self.assertEqual(decision.status, "invalid")
        self.assertFalse(decision.advantage_refs)
        with self.assertRaises((GroupError, TypeError)):
            self.coordinator.collect_scripted(spec, 0, reward=1.0)
        interrupted = self.group(sequence=3)
        interrupted_starts = self.starts(interrupted)
        self.coordinator.collect(
            interrupted,
            GroupMemberResultV1(
                group_id=interrupted.group_id,
                member_id=interrupted.members[0].member_id,
                start_checkpoint_id=interrupted_starts[0].checkpoint_id,
            ),
        )
        self.assertEqual(self.coordinator.finalize(interrupted).status, "pending")
        self.coordinator.collect_invalid(interrupted, 0, reason="worker_crash")
        self.assertEqual(self.coordinator.finalize(interrupted).status, "invalid")
        irrational = self.coordinator.seal(
            self.start,
            policy=self.policy,
            group_seed=771,
            group_sequence=4,
            member_count=3,
            runner_mode="fixture",
        )
        for ordinal in (2, 0, 1):
            self.coordinator.start(irrational, ordinal, policy=self.policy)
            self.coordinator.collect_scripted(irrational, ordinal, reward=Fraction(ordinal))
        decision = self.coordinator.finalize(irrational)
        self.assertEqual(decision.status, "ready")
        self.assertEqual(
            [self.store.get_artifact(ref)["variance"] for ref in decision.advantage_refs],
            [{"numerator": 2, "denominator": 3}] * 3,
        )
        self.assertTrue(
            all(
                self.store.get_artifact(ref)["expression"] == "centered / sqrt(population_variance)"
                for ref in decision.advantage_refs
            )
        )

    def test_corrupt_group_receipts_fail_closed(self):
        spec = self.group(sequence=9, mode="fixture")
        self.starts(spec)
        self.coordinator.collect_scripted(spec, 0, reward=Fraction(1))
        directory = self.store.root / "groups" / spec.group_id
        receipt = directory / "result-0.json"
        receipt.write_bytes(b'{"result_ref":"tampered"}')
        with self.assertRaises(ValueError):
            self.coordinator.finalize(spec)
        (directory / "spec.json").write_bytes(b'{"schema":1}')
        with self.assertRaises(ValueError):
            self.coordinator.resume(spec.group_id)

    def test_phase5_terminal_binding_and_writer_only_segment_credit(self):
        spec = self.group()
        starts = self.starts(spec)
        self.coordinator.collect(
            spec,
            GroupMemberResultV1(
                group_id=spec.group_id,
                member_id=spec.members[0].member_id,
                start_checkpoint_id=starts[0].checkpoint_id,
            ),
        )
        results = []
        for ordinal, runtime in enumerate(starts):
            member = spec.members[ordinal]
            writer = TransactionalWriterV1(
                self.store, self.writer.graph, member.member_id, runtime.checkpoint_id
            )
            checks = DeterministicChecksV1(writer)
            terminal = ScriptedTerminalV1(writer)
            if ordinal == 0:
                # This valid writer failure stays in the group denominator.
                action = writer.submit_action(
                    runtime,
                    self.action(
                        self.call("write_file", {"path": "draft.txt", "content": ""}, "empty")
                    ),
                )
                written = writer.step_tool(action.runtime)
                runtime = written.runtime
            else:
                ask = writer.submit_action(
                    runtime,
                    self.action(
                        self.call(
                            "ask_author",
                            {
                                "question": "Which door?",
                                "decision_ids": ["door"],
                                "proposals": [],
                                "option_refs": [],
                            },
                        )
                    ),
                )
                request = writer.step_tool(ask.runtime)
                reply = ScriptedAuthorRuntimeV1(writer).reply(request.runtime)
                compacted = writer.change_context(reply.runtime, self.compaction_policy)
                runtime = compacted.runtime
            final = writer.submit_action(runtime, self.action(content="Final revision."))
            batch = checks.request_checks(final.runtime)
            checked = checks.check_next(batch.runtime)
            if ordinal == 1:
                transition = terminal.transition(checked.runtime)
                outcome = terminal.terminal_outcome(transition.runtime)
            else:
                outcome = terminal.terminal_outcome(checked.runtime)
            rewarded = terminal.reward(outcome.runtime)
            result = GroupMemberResultV1(
                group_id=spec.group_id,
                member_id=member.member_id,
                start_checkpoint_id=starts[ordinal].checkpoint_id,
                final_checkpoint_id=rewarded.runtime.checkpoint_id,
                terminal_outcome_ref=outcome.runtime.state.outcome_ref,
                availability_ref=rewarded.runtime.state.outcome_ref,
                execution_status="valid",
            )
            results.append(result)
        # Completion order has no effect on member order or advantage identities.
        self.coordinator.collect(spec, results[1])
        self.coordinator.collect(spec, results[0])
        decision = self.coordinator.finalize(spec)
        self.assertEqual(decision.status, "ready")
        self.assertEqual(len(decision.segment_credit_refs), 8)
        credits = [self.store.get_artifact(ref) for ref in decision.segment_credit_refs]
        self.assertEqual(
            [c["member_id"] for c in credits],
            [
                spec.members[0].member_id,
                spec.members[0].member_id,
                spec.members[0].member_id,
                spec.members[0].member_id,
                spec.members[1].member_id,
                spec.members[1].member_id,
                spec.members[1].member_id,
                spec.members[1].member_id,
            ],
        )
        self.assertEqual(
            [c["segment_kind"] for c in credits],
            [
                "tool_syntax",
                "assistant_ending",
                "assistant_text",
                "assistant_ending",
                "tool_syntax",
                "assistant_ending",
                "assistant_text",
                "assistant_ending",
            ],
        )
        for credit in credits:
            if credit["part_index"] is None:
                self.assertIsNone(credit["segment_content_hash"])
            else:
                message = self.store.get_artifact(credit["message_ref"], expected_domain="message")
                self.assertEqual(
                    credit["segment_content_hash"],
                    domain_hash("payload", message["content"][credit["part_index"]]),
                )
        self.assertTrue(
            all(
                c["native_optimizer_eligible"] is False
                and c["token_mask_ref"] is None
                and c["logprob_ref"] is None
                for c in credits
            )
        )
        self.assertTrue(
            all(
                c["excluded_roles"]
                == [
                    "system",
                    "user",
                    "author",
                    "tool",
                    "seed",
                    "environment",
                    "summary",
                ]
                for c in credits
            )
        )
        self.assertTrue(
            all(
                c["original_context_ref"]
                == self.store.get_artifact(c["trace_ref"])["context_revision_ref"]
                for c in credits
            )
        )
        self.assertNotEqual(
            credits[4]["original_context_ref"],
            self.store.load_checkpoint(results[1].final_checkpoint_id).state.context_ref,
        )
        self.assertEqual(decision.identity(), self.coordinator.finalize(spec).identity())
        wrong_member = replace(results[0], member_id=spec.members[1].member_id)
        with self.assertRaises(GroupError):
            self.coordinator.collect(spec, wrong_member)
        forged = replace(results[0], terminal_outcome_ref=results[1].terminal_outcome_ref)
        with self.assertRaises(GroupError):
            self.coordinator.collect(spec, forged)
        bad_reward = self.store.put_artifact(
            {
                **self.store.get_artifact(results[0].availability_ref),
                "terminal_outcome_ref": results[1].terminal_outcome_ref,
            }
        )
        with self.assertRaises(GroupError):
            self.coordinator.collect(spec, replace(results[0], availability_ref=bad_reward))
