"""Group coordination over verified task-graph views and sampled writer turns."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from tests import test_task_graph_scripted as legacy_scripted_tests
from tests.task_graph_rollout_fixtures import (
    build_rollout_fixture,
    make_gatherers,
    ports_disabled,
    run_slice,
)
from writing_agent.task_graph import canonical_bytes, domain_hash
from writing_agent.task_graph_calls import intake_message
from writing_agent.task_graph_compaction import ContextPolicyV1 as LegacyContextPolicyV1
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_derive_writer import derive_writer_turn
from writing_agent.task_graph_errors import AdapterContractError, ProjectionError
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_group import (
    POLICY_FIELDS,
    GroupCoordinatorV1,
    GroupError,
    GroupMemberResultV1,
    GroupSpecV1,
)
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_record_contracts import ContextPolicyV1
from writing_agent.task_graph_records import WriterTurnV1
from writing_agent.task_graph_rollout_env import RolloutEnvironment
from writing_agent.task_graph_store import TaskGraphStore


class TestGroupCoordinatorCore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = build_rollout_fixture(self.root / "rollout")
        self.store = self.fixture.store
        self.env = self.fixture.env
        self.entry_id = self.fixture.runtime.checkpoint_id
        self.coordinator = GroupCoordinatorV1(self.env, self.root / "workers")
        self.policy = self.policy_for(self.fixture)

    def policy_for(self, fixture):
        store = fixture.store
        rendering = fixture.runtime.context.rendering
        policy = {
            field: store.put_artifact({"pin": field})
            for field in POLICY_FIELDS
            if field != "rng_derivation_version"
        }
        policy["model_ref"] = store.put_artifact({"model_id": "group-model-v1"})
        context_policy = ContextPolicyV1(
            "compact", summarizer_version="visible-text-v1", max_summary_chars=20
        )
        policy["context_policy_ref"] = store.put_artifact(context_policy.to_dict())
        policy.update(
            tokenizer_ref=rendering["tokenizer_ref"],
            template_ref=rendering["template_ref"],
            rng_derivation_version="sha256-domain-v1",
        )
        return policy

    def group(self, sequence=0, mode="real"):
        return self.coordinator.seal(
            self.entry_id,
            policy=self.policy,
            group_seed=771,
            group_sequence=sequence,
            member_count=2,
            runner_mode=mode,
        )

    def start_members(self, spec):
        return tuple(
            self.coordinator.start(spec, ordinal, policy=self.policy) for ordinal in range(2)
        )

    def run_member(self, spec, ordinal):
        runtime = self.coordinator.start(spec, ordinal, policy=self.policy)
        self.fixture.gatherers = make_gatherers(self.fixture)
        runtime = run_slice(self.fixture, runtime=runtime)
        view = self.env.verify(runtime)
        outcome_ref = runtime.state.outcome_ref
        if view.outcome.reward_ref is not None:
            reward = self.store.get_artifact(view.outcome.reward_ref)
            outcome_ref = reward["terminal_outcome_ref"]
        return runtime, GroupMemberResultV1(
            group_id=spec.group_id,
            member_id=spec.members[ordinal].member_id,
            start_checkpoint_id=self.coordinator._start_receipt(spec, ordinal)[
                "start_checkpoint_id"
            ],
            final_checkpoint_id=runtime.checkpoint_id,
            terminal_outcome_ref=outcome_ref,
            availability_ref=view.outcome.reward_ref,
            execution_status="valid",
        )

    def test_full_contract_drift_and_start_isolation(self):
        spec = self.group()
        self.assertEqual(self.store.get_artifact(spec.identity()), spec.to_wire())

        first = self.coordinator.start(spec, 0, policy=self.policy)
        second = self.coordinator.start(spec, 1, policy=self.policy)
        first_view, second_view = self.env.verify(first), self.env.verify(second)
        self.assertEqual(first.state.files, self.fixture.runtime.state.files)
        self.assertEqual(first.context, self.fixture.runtime.context)
        self.assertEqual(first_view.context.revision_ref, second_view.context.revision_ref)
        self.assertEqual(first_view.state.position["lineage_id"], spec.members[0].member_id)
        self.assertEqual(second_view.state.position["lineage_id"], spec.members[1].member_id)
        self.assertEqual(first_view.group.identity(), spec.identity())
        self.assertNotEqual(first.checkpoint_id, second.checkpoint_id)

        first_seeds = self.store.get_artifact(first.state.rng_ref)
        second_seeds = self.store.get_artifact(second.state.rng_ref)
        self.assertEqual(first_seeds["writer_seed"], spec.members[0].writer_seed)
        self.assertEqual(second_seeds["writer_seed"], spec.members[1].writer_seed)
        self.assertEqual(first_seeds["environment_seed"], second_seeds["environment_seed"])
        self.assertNotEqual(first_seeds["writer_seed"], second_seeds["writer_seed"])
        self.assertEqual(first_seeds["parent_rng_ref"], self.fixture.runtime.state.rng_ref)
        self.assertEqual(
            self.coordinator.start(spec, 0, policy=self.policy).checkpoint_id,
            first.checkpoint_id,
        )
        self.assertEqual(self.group().identity(), spec.identity())
        self.assertNotEqual(self.group(sequence=1).group_id, spec.group_id)
        for field in POLICY_FIELDS:
            changed = dict(self.policy)
            changed[field] = (
                "sha256-domain-v2"
                if field == "rng_derivation_version"
                else self.store.put_artifact({"different": field})
            )
            with self.subTest(policy_field=field), self.assertRaises(GroupError):
                self.coordinator.assert_start_contract(spec, self.entry_id, changed)
        larger = self.coordinator.seal(
            self.entry_id,
            policy=self.policy,
            group_seed=771,
            group_sequence=0,
            member_count=3,
        )
        self.assertNotEqual(larger.group_id, spec.group_id)
        with self.assertRaises(GroupError):
            self.coordinator.seal(
                self.entry_id,
                policy=self.policy,
                group_seed=0,
                group_sequence=2,
                member_count=1,
            )
        incomplete = dict(self.policy)
        incomplete.pop("adapter_ref")
        with self.assertRaises(GroupError):
            self.coordinator.seal(
                self.entry_id,
                policy=incomplete,
                group_seed=0,
                group_sequence=2,
                member_count=2,
            )
        head = self.store.read_head(spec.members[0].member_id)
        self.coordinator.start(spec, 0, policy=self.policy)
        self.assertEqual(self.store.read_head(spec.members[0].member_id), head)

    def test_member_start_retry_recovers_every_publish_stage(self):
        stages = (
            "before_immutable_writes",
            "after_immutable_writes",
            "before_head_publication",
            "after_head_publication",
        )
        original_publish = self.store.publish
        for index, stage in enumerate(stages, start=1):
            spec = self.group(sequence=index)
            member_id = spec.members[0].member_id

            class InjectedCrash(RuntimeError):
                pass

            def publish_with_fault(*args, _stage=stage, **kwargs):
                def fault(current):
                    if current == _stage:
                        raise InjectedCrash(_stage)

                kwargs["fault"] = fault
                return original_publish(*args, **kwargs)

            with patch.object(self.store, "publish", side_effect=publish_with_fault):
                with self.assertRaises(InjectedCrash):
                    self.coordinator.start(spec, 0, policy=self.policy)

            resumed = self.coordinator.start(spec, 0, policy=self.policy)
            resumed_head = self.store.read_head(member_id)
            self.assertIsNotNone(resumed_head)
            self.assertEqual(
                self.env.open_head(member_id).checkpoint_id,
                resumed.checkpoint_id,
            )
            self.assertEqual(
                self.coordinator.start(spec, 0, policy=self.policy).checkpoint_id,
                resumed.checkpoint_id,
            )
            self.assertEqual(self.store.read_head(member_id), resumed_head)

    def test_scripted_pending_tie_invalid_and_exact_advantage_paths(self):
        spec = self.group(mode="fixture")
        self.start_members(spec)
        self.coordinator.collect_scripted(spec, 1, reward=Fraction(3, 2))
        self.assertEqual(self.coordinator.finalize(spec).status, "pending")
        self.coordinator.collect_scripted(spec, 0, reward_status="pending")
        self.coordinator.collect_scripted(spec, 0, reward_status="unavailable")
        pending = self.coordinator.finalize(spec)
        self.assertEqual((pending.status, pending.advantage_refs), ("pending", ()))

        self.coordinator.collect_scripted(spec, 0, reward=Fraction(-1, 2))
        ready = self.coordinator.finalize(spec)
        advantages = [self.store.get_artifact(ref) for ref in ready.advantage_refs]
        self.assertEqual(ready.status, "ready")
        self.assertEqual(
            [item["centered"] for item in advantages],
            [{"numerator": -1, "denominator": 1}, {"numerator": 1, "denominator": 1}],
        )
        self.assertFalse(ready.segment_credit_refs)

        tie = self.group(sequence=20, mode="fixture")
        self.start_members(tie)
        self.coordinator.collect_scripted(tie, 0, reward=Fraction(-5))
        self.coordinator.collect_scripted(tie, 1, reward=Fraction(-5))
        decision = self.coordinator.finalize(tie)
        self.assertEqual(decision.status, "tie")
        self.assertEqual(
            [self.store.get_artifact(ref)["advantage"] for ref in decision.advantage_refs],
            [{"numerator": 0, "denominator": 1}] * 2,
        )

        invalid = self.group(sequence=21)
        self.start_members(invalid)
        self.coordinator.collect_invalid(invalid, 0, reason="worker_crash")
        self.assertEqual(self.coordinator.finalize(invalid).status, "invalid")

    def test_real_collection_reads_outcome_reward_eligibility_and_samples(self):
        invalid_call = {
            "id": "malformed-call",
            "type": "function",
            "function": {"name": "not_admitted", "arguments": {"secret": "sampled"}},
        }
        self.fixture.sample_results = (
            SampleResult({"role": "assistant", "content": "", "tool_calls": [invalid_call]}),
            SampleResult({"role": "assistant", "content": "The revised draft.", "tool_calls": []}),
        )
        spec = self.group()
        results = []
        for ordinal in range(2):
            start = self.coordinator.start(spec, ordinal, policy=self.policy)
            self.coordinator.collect(
                spec,
                GroupMemberResultV1(
                    group_id=spec.group_id,
                    member_id=spec.members[ordinal].member_id,
                    start_checkpoint_id=start.checkpoint_id,
                ),
            )
            runtime, result = self.run_member(spec, ordinal)
            view = self.env.verify(runtime)
            self.assertEqual(view.outcome.reward_status, "available")
            self.assertEqual(view.outcome.training_eligibility, "ineligible")
            self.assertEqual(len(view.samples), 2)
            self.coordinator.collect(spec, result)
            results.append(result)

        wrong_outcome = replace(results[0], terminal_outcome_ref=results[1].terminal_outcome_ref)
        with self.assertRaises(GroupError):
            self.coordinator.collect(spec, wrong_outcome)
        first_reward = self.store.get_artifact(results[0].availability_ref)
        wrong_reward_ref = self.store.put_artifact(
            {
                **first_reward,
                "terminal_outcome_ref": results[1].terminal_outcome_ref,
            }
        )
        with self.assertRaises(GroupError):
            self.coordinator.collect(spec, replace(results[0], availability_ref=wrong_reward_ref))

        decision = self.coordinator.finalize(spec)
        self.assertEqual(decision.status, "tie")
        self.assertEqual(len(decision.segment_credit_refs), 8)
        credits = [self.store.get_artifact(ref) for ref in decision.segment_credit_refs]
        self.assertTrue(all(not credit["native_optimizer_eligible"] for credit in credits))
        self.assertTrue(all(credit["token_mask_ref"] is None for credit in credits))
        self.assertTrue(all(credit["logprob_ref"] is None for credit in credits))
        self.assertEqual(
            [credit["segment_kind"] for credit in credits],
            ["tool_syntax", "assistant_ending", "assistant_text", "assistant_ending"] * 2,
        )
        for credit in credits:
            message = self.store.get_artifact(credit["message_ref"], expected_domain="message")
            self.assertEqual(message["role"], "assistant")
            if credit["segment_kind"] == "tool_syntax":
                part = message["content"][credit["part_index"]]
                self.assertEqual(part["type"], "invalid_tool_call")
                self.assertEqual(part["raw"]["function"]["name"], "not_admitted")
                self.assertNotEqual(
                    credit["segment_content_hash"],
                    domain_hash("payload", {"name": "invalid_call", "arguments": {}}),
                )
        offline_gate = LineageGate()
        offline_store = TaskGraphStore(self.store.root, verifier=offline_gate)
        offline_env = RolloutEnvironment(
            offline_store,
            self.fixture.entry.graph,
            None,
            offline_gate,
            self.fixture.entry.graph.policy,
        )
        offline = GroupCoordinatorV1(offline_env, self.root / "offline-workers")
        with ports_disabled():
            replayed = offline.finalize(spec)
        self.assertEqual(replayed.identity(), decision.identity())

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
        policy = self.policy_for(fixture)
        coordinator = GroupCoordinatorV1(fixture.env, self.root / "token-limited-workers")
        spec = coordinator.seal(
            fixture.runtime.checkpoint_id,
            policy=policy,
            group_seed=771,
            group_sequence=24,
            member_count=2,
        )

        class OverrunSampler:
            def sample(self, _prepared):
                return SampleResult(
                    {"role": "assistant", "content": "overrun", "tool_calls": []},
                    usage={"completion_tokens": 101},
                )

        for ordinal, member in enumerate(spec.members):
            runtime = coordinator.start(spec, ordinal, policy=policy)
            fixture.gatherers = make_gatherers(fixture, sampler=OverrunSampler())
            runtime = run_slice(fixture, runtime=runtime)
            view = fixture.env.verify(runtime)
            self.assertEqual(view.state.position["phase"], "terminal")
            self.assertEqual(view.outcome.execution_status, "valid")
            self.assertEqual(view.outcome.task_status, "incomplete")
            reward = fixture.store.get_artifact(view.outcome.reward_ref)
            result = GroupMemberResultV1(
                group_id=spec.group_id,
                member_id=member.member_id,
                start_checkpoint_id=coordinator._start_receipt(spec, ordinal)[
                    "start_checkpoint_id"
                ],
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
        port = self.env.port_input(view, next_step(view))
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
                request_ref=None,
                prepared_request_ref=None,
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
                request_ref=None,
                prepared_request_ref=None,
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
            port = self.env.port_input(view, next_step(view))
            self.assertIsNotNone(port)
            turn = WriterTurnV1(
                action_id=port.action_id,
                context_revision_ref=port.context_revision_ref,
                request_ref=None,
                prepared_request_ref=None,
                raw_output_ref=None,
                usage={},
                adapter_trace={field: value_for(spec)},
                message=intake_message({"role": "assistant", "content": "draft"}),
            )
            transition = derive_writer_turn(replace(view, group=None), turn, self.env.reader)
            self.env._persist_transition(transition)
            start = self.coordinator._start_receipt(spec, 0)
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
            offline = GroupCoordinatorV1(offline_env, self.root / f"drift-{field}")
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
        receipt = self.coordinator._start_receipt(spec, 0)
        member_head = self.store.read_head(spec.members[0].member_id)
        with (
            patch.object(self.coordinator, "resume", return_value=spec),
            patch.object(self.coordinator, "_start_receipt", return_value=receipt),
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

    def test_group_receipt_write_is_atomic_and_pre_s1_specs_still_resume(self):
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

        # The legacy bridge is intentionally retained until the S7 caller switch.
        from tests import test_task_graph_scripted as scripted

        legacy = scripted.ScriptedFixture()
        legacy.setUp()
        self.addCleanup(legacy.doCleanups)
        old = GroupCoordinatorV1(legacy.store, self.root / "old-runtime")
        receipt_bytes = (Path(__file__).parent / "fixtures" / "pre_s1_group_spec.json").read_bytes()
        old_spec = GroupSpecV1.from_json(receipt_bytes)
        path = old.groups_root / old_spec.group_id / "spec.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(receipt_bytes)
        context_policy = ContextPolicyV1("drop").to_wire()

        def get_artifact(identity):
            return (
                context_policy
                if identity == old_spec.policy["context_policy_ref"]
                else {"pin": identity}
            )

        with (
            patch.object(old, "_entry_contract", return_value=(old_spec.environment, {})),
            patch.object(legacy.store, "get_artifact", side_effect=get_artifact),
        ):
            self.assertEqual(old.resume(old_spec.group_id).to_dict(), old_spec.to_dict())


class GroupCoordinatorTests:
    """Legacy entry fixture retained for derive tests that share its group inputs."""

    def __init__(self, _method_name=None):
        self.cleanups = []

    def addCleanup(self, function, *args, **kwargs):
        self.cleanups.append((function, args, kwargs))

    def doCleanups(self):
        while self.cleanups:
            function, args, kwargs = self.cleanups.pop()
            function(*args, **kwargs)

    def setUp(self):
        fixture = legacy_scripted_tests.ScriptedFixture()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.store = fixture.store
        self.start = fixture.start
        self.runtime = fixture.runtime
        self.writer = fixture.writer
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
        context_policy = LegacyContextPolicyV1(
            "compact", summarizer_version="visible-text-v1", max_summary_chars=20
        )
        self.policy["context_policy_ref"] = self.store.put_artifact(context_policy.to_dict())

    def group(self, sequence=0, mode="real"):
        return self.coordinator.seal(
            self.start,
            policy=self.policy,
            group_seed=771,
            group_sequence=sequence,
            member_count=2,
            runner_mode=mode,
        )


if __name__ == "__main__":
    unittest.main()
