"""Sample-credit, collection and receipt evidence for task-graph groups."""

from __future__ import annotations

import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

from tests.task_graph_group_support import GroupCoordinatorFixtureMixin
from tests.task_graph_rollout_fixtures import (
    build_rollout_fixture,
    make_gatherers,
    run_slice,
)
from writing_agent.task_graph import (
    canonical_bytes,
)
from writing_agent.task_graph_calls import intake_message
from writing_agent.task_graph_derive_writer import derive_writer_turn
from writing_agent.task_graph_environment import RolloutEnvironment
from writing_agent.task_graph_errors import (
    AdapterContractError,
    ProjectionError,
)
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_group import (
    GroupCoordinatorV1,
    GroupError,
    GroupMemberResultV1,
)
from writing_agent.task_graph_local import (
    ScriptedSampleBackend,
)
from writing_agent.task_graph_ports import (
    USAGE_REPORTING_CAPABILITY,
    PortDescriptorV1,
    SampleResult,
)
from writing_agent.task_graph_records import (
    WriterTurnV1,
)
from writing_agent.task_graph_store import TaskGraphStore


class GroupEvidenceTests(GroupCoordinatorFixtureMixin, unittest.TestCase):
    def test_noncanonical_tool_call_values_do_not_receive_segment_credit(self):
        def nested(depth):
            value = "x"
            for _ in range(depth):
                value = [value]
            return value

        cases = (
            (
                "unbounded_list_call",
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "deep",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": {"p": nested(80)},
                            },
                        }
                    ],
                },
            ),
        )
        for sequence, (label, malformed_call) in enumerate(cases, start=90):
            with self.subTest(call_shape=label):
                self.fixture.sample_results = (
                    SampleResult(malformed_call),
                    SampleResult(
                        {"role": "assistant", "content": "The revised draft.", "tool_calls": []}
                    ),
                )
                spec = self.group(sequence=sequence)
                for ordinal in range(2):
                    _, result = self.run_member(spec, ordinal)
                    self.coordinator.collect(spec, result)
                decision = self.coordinator.finalize(spec)
                credits = [self.store.get_artifact(ref) for ref in decision.segment_credit_refs]
                self.assertFalse(any(credit["segment_kind"] == "tool_syntax" for credit in credits))

    def test_context_updates_do_not_rebind_sample_credit(self):
        spec = self.group(sequence=23)
        results = [self.run_member(spec, ordinal)[1] for ordinal in range(2)]
        for result in results:
            self.coordinator.collect(spec, result)

        decision = self.coordinator.finalize(spec)
        self.assertEqual(decision.status, "tie")
        self.assertEqual(len(decision.segment_credit_refs), 16)
        contexts = set()
        for ref in decision.segment_credit_refs:
            credit = self.store.get_artifact(ref)
            turn = self.store.get_artifact(credit["trace_ref"])
            self.assertEqual(credit["original_context_ref"], turn["context_revision_ref"])
            contexts.add(credit["original_context_ref"])
        self.assertGreater(len(contexts), 1)

    def test_sampled_budget_stop_remains_a_valid_group_result(self):
        fixture = build_rollout_fixture(self.root / "token-limited", mode="token_limited")
        overrun = SampleResult(
            {"role": "assistant", "content": "overrun", "tool_calls": []},
            usage={"completion_tokens": 101},
        )
        backend = ScriptedSampleBackend((overrun, overrun))
        backend.descriptor = PortDescriptorV1(
            "sampling",
            "overrun-sampler",
            "1",
            capabilities=(USAGE_REPORTING_CAPABILITY,),
        )
        session = self.runtime_session(fixture, backend=backend)
        fixture.env.session = session
        policy = self.policy_for(fixture, session=session)
        coordinator = GroupCoordinatorV1(fixture.env, session=session)
        spec = coordinator.seal(
            fixture.runtime.checkpoint_id,
            policy=policy,
            group_seed=771,
            group_sequence=24,
            member_count=2,
        )

        for ordinal, member in enumerate(spec.members):
            runtime = coordinator.start(spec, ordinal, policy=policy)
            fixture.gatherers = make_gatherers(fixture, sampler=backend)
            runtime = run_slice(fixture, runtime=runtime)
            view = fixture.env.verify(runtime)
            self.assertEqual(view.state.position["phase"], "terminal")
            self.assertEqual(view.outcome.execution_status, "valid")
            self.assertEqual(view.outcome.task_status, "incomplete")
            reward = fixture.store.get_artifact(view.outcome.reward_ref)
            result = GroupMemberResultV1(
                group_id=spec.group_id,
                member_id=member.member_id,
                start_checkpoint_id=coordinator.start_receipt(spec, ordinal)["start_checkpoint_id"],
                final_checkpoint_id=runtime.checkpoint_id,
                terminal_outcome_ref=reward["terminal_outcome_ref"],
                availability_ref=view.outcome.reward_ref,
                execution_status="valid",
            )
            coordinator.collect(spec, result)

        decision = coordinator.finalize(spec)
        self.assertEqual(decision.status, "tie")
        self.assertEqual(len(decision.advantage_refs), 2)
        self.assertFalse(decision.segment_credit_refs)

    def test_sampling_policy_drift_is_adapter_error_and_gate_rejects_forged_drift(self):
        spec = self.group()
        runtime = self.coordinator.start(spec, 0, policy=self.policy)
        view = self.env.verify(runtime)
        port = self.env.step_input(runtime)[2]
        self.assertIsNotNone(port)
        other_ref = self.store.put_artifact({"other": "policy"})
        drift_rows = (
            ("seed", spec.members[0].writer_seed + 1),
            ("model", "other-model"),
            ("behavior_policy_ref", other_ref),
            ("context_policy_ref", other_ref),
        )
        for field, value in drift_rows:
            trace = {field: value}
            turn = WriterTurnV1(
                action_id=port.action_id,
                context_revision_ref=port.context_revision_ref,
                raw_output_ref=None,
                usage={},
                adapter_trace=trace,
                message=intake_message({"role": "assistant", "content": "draft"}),
            )
            head = self.store.read_head(spec.members[0].member_id)
            with (
                self.subTest(layer="sampling", field=field),
                self.assertRaises(AdapterContractError),
            ):
                self.env.commit(runtime, turn)
            self.assertEqual(self.store.read_head(spec.members[0].member_id), head)

        for field, value in drift_rows:
            turn = WriterTurnV1(
                action_id=port.action_id,
                context_revision_ref=port.context_revision_ref,
                raw_output_ref=None,
                usage={},
                adapter_trace={field: value},
                message=intake_message({"role": "assistant", "content": "draft"}),
            )
            transition = derive_writer_turn(replace(view, group=None), turn, self.env.reader)
            self.env._persist_transition(transition)
            head = self.store.read_head(spec.members[0].member_id)
            with self.subTest(layer="gate", field=field), self.assertRaises(ProjectionError):
                self.store.publish(
                    spec.members[0].member_id,
                    head,
                    (transition.event,),
                    transition.state,
                )
            self.assertEqual(self.store.read_head(spec.members[0].member_id), head)

    def test_collect_gate_rejects_persisted_group_sampling_drift(self):
        other_ref = self.store.put_artifact({"other": "policy"})
        drift_rows = (
            ("seed", lambda spec: spec.members[0].writer_seed + 1),
            ("model", lambda _spec: "other-model"),
            ("behavior_policy_ref", lambda _spec: other_ref),
            ("context_policy_ref", lambda _spec: other_ref),
        )

        class BypassVerifier:
            def verify_commit(self, *_args):
                return None

        for sequence, (field, value_for) in enumerate(drift_rows, start=40):
            spec = self.group(sequence=sequence)
            runtime = self.coordinator.start(spec, 0, policy=self.policy)
            view = self.env.verify(runtime)
            port = self.env.step_input(runtime)[2]
            self.assertIsNotNone(port)
            turn = WriterTurnV1(
                action_id=port.action_id,
                context_revision_ref=port.context_revision_ref,
                raw_output_ref=None,
                usage={},
                adapter_trace={field: value_for(spec)},
                message=intake_message({"role": "assistant", "content": "draft"}),
            )
            transition = derive_writer_turn(replace(view, group=None), turn, self.env.reader)
            self.env._persist_transition(transition)
            start = self.coordinator.start_receipt(spec, 0)
            base_head = self.store.read_head(spec.members[0].member_id)
            with patch.object(self.store, "_verifier", BypassVerifier()):
                forged_head = self.store.publish(
                    spec.members[0].member_id,
                    base_head,
                    (transition.event,),
                    transition.state,
                )
            forged_checkpoint = self.store.load_commit(forged_head).checkpoint
            result = GroupMemberResultV1(
                group_id=spec.group_id,
                member_id=spec.members[0].member_id,
                start_checkpoint_id=start["start_checkpoint_id"],
                final_checkpoint_id=forged_checkpoint,
                terminal_outcome_ref=view.state.outcome_ref,
                execution_status="valid",
            )

            gate = LineageGate()
            offline_store = TaskGraphStore(self.store.root, verifier=gate)
            offline_env = RolloutEnvironment(
                offline_store,
                self.fixture.entry.graph,
                None,
                gate,
                self.fixture.entry.graph.policy,
            )
            offline = GroupCoordinatorV1(offline_env)
            published = self.store.read_head(spec.members[0].member_id)
            with self.subTest(field=field), self.assertRaises(ProjectionError):
                offline.collect(spec, result)
            self.assertEqual(self.store.read_head(spec.members[0].member_id), published)

    def test_collection_rejects_a_view_sealed_to_another_group(self):
        spec = self.group(sequence=30)
        other = self.group(sequence=31)
        runtime, result = self.run_member(spec, 0)
        actual_view = self.env.verify(runtime)
        foreign_view = replace(actual_view, group=other)
        receipt = self.coordinator.start_receipt(spec, 0)
        member_head = self.store.read_head(spec.members[0].member_id)
        with (
            patch.object(self.coordinator, "resume", return_value=spec),
            patch.object(self.coordinator, "start_receipt", return_value=receipt),
            patch.object(self.env, "verify", return_value=foreign_view),
        ):
            with self.assertRaises(GroupError):
                self.coordinator.collect(spec, result)
        self.assertEqual(self.store.read_head(spec.members[0].member_id), member_head)

    def test_collect_rejects_slot_misbind_and_corrupt_receipts(self):
        spec = self.group(mode="fixture")
        starts = self.start_members(spec)
        self.coordinator.collect_scripted(spec, 0, reward=Fraction(0))
        result_path = self.coordinator.groups_root / spec.group_id / "result-0.json"
        original = result_path.read_bytes()
        fixture_ref = self.store.put_artifact(
            {
                "record_type": "GroupScriptedTerminalV1",
                "schema": 1,
                "group_id": spec.group_id,
                "member_id": spec.members[1].member_id,
                "start_checkpoint_id": starts[1].checkpoint_id,
                "execution_status": "valid",
                "reward_status": "available",
                "reward": {"numerator": 4, "denominator": 1},
                "native_optimizer_eligible": False,
            }
        )
        misbound = GroupMemberResultV1(
            group_id=spec.group_id,
            member_id=spec.members[0].member_id,
            start_checkpoint_id=starts[0].checkpoint_id,
            fixture_ref=fixture_ref,
            execution_status="valid",
        )
        with self.assertRaises(GroupError):
            self.coordinator.collect(spec, misbound)
        result_path.write_bytes(b'{"member_id":"wrong"}')
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.coordinator.finalize(spec)
        result_path.write_bytes(original)
        self.assertEqual(self.coordinator.finalize(spec).status, "pending")
        self.assertNotEqual(starts[0].checkpoint_id, starts[1].checkpoint_id)

        real = self.group(sequence=22)
        real_start = self.coordinator.start(real, 0, policy=self.policy)
        fixture_ref = self.store.put_artifact(
            {
                "record_type": "GroupScriptedTerminalV1",
                "schema": 1,
                "group_id": real.group_id,
                "member_id": real.members[0].member_id,
                "start_checkpoint_id": real_start.checkpoint_id,
                "execution_status": "valid",
                "reward_status": "available",
                "reward": {"numerator": 0, "denominator": 1},
                "native_optimizer_eligible": False,
            }
        )
        forged = GroupMemberResultV1(
            group_id=real.group_id,
            member_id=real.members[0].member_id,
            start_checkpoint_id=real_start.checkpoint_id,
            fixture_ref=fixture_ref,
            execution_status="valid",
        )
        with self.assertRaises(GroupError):
            self.coordinator.collect(real, forged)
        result_ref = self.store.put_artifact(forged.to_dict())
        forged_path = self.coordinator.groups_root / real.group_id / "result-0.json"
        forged_path.write_bytes(canonical_bytes({"result_ref": result_ref}))
        with self.assertRaises(GroupError):
            self.coordinator.finalize(real)

    def test_group_receipt_write_is_atomic_and_corrupt_receipts_fail_closed(self):
        spec = self.group()
        self.assertEqual(self.store.get_artifact(spec.identity()), spec.to_wire())
        spec_path = self.coordinator.groups_root / spec.group_id / "spec.json"
        spec_bytes = spec_path.read_bytes()
        spec_path.write_bytes(b'{"schema":1}')
        with self.assertRaises(ValueError):
            self.coordinator.resume(spec.group_id)
        spec_path.write_bytes(spec_bytes)
        receipt = self.root / "atomic-receipt.json"
        with patch("writing_agent.task_graph_group.os.link", side_effect=OSError("fault")):
            with self.assertRaises(OSError):
                self.coordinator._receipt(receipt, {"schema": 1})
        self.assertFalse(receipt.exists())
        self.coordinator._receipt(receipt, {"schema": 1})
        self.assertEqual(receipt.read_bytes(), b'{"schema":1}')


if __name__ == "__main__":
    unittest.main()
