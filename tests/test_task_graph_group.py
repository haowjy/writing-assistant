"""Group coordination over verified task-graph views and sampled writer turns."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from tests.task_graph_group_support import GroupCoordinatorFixtureMixin
from tests.task_graph_rollout_fixtures import (
    build_rollout_fixture,
    make_gatherers,
    ports_disabled,
    run_slice,
)
from writing_agent.task_graph import (
    CheckpointV1,
    canonical_bytes,
    domain_hash,
    load_canonical_json,
    thaw,
)
from writing_agent.task_graph_calls import intake_message
from writing_agent.task_graph_environment import RolloutEnvironment
from writing_agent.task_graph_errors import (
    AdapterContractError,
    ConcurrentUpdateError,
)
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_group import (
    POLICY_FIELDS,
    GroupCoordinatorV1,
    GroupError,
    GroupMemberResultV1,
    GroupScriptedTerminalV1,
)
from writing_agent.task_graph_group_records import GroupExecutionFailureV1
from writing_agent.task_graph_local import (
    ScriptedSampleBackend,
)
from writing_agent.task_graph_ports import (
    PortDescriptorV1,
    SampleResult,
)
from writing_agent.task_graph_record_contracts import ContextPolicyV1
from writing_agent.task_graph_records import (
    MemberStartV1,
    RuntimeManifestV1,
    RuntimePortDescriptorV1,
    WriterTurnV1,
)
from writing_agent.task_graph_store import TaskGraphStore


class TestGroupCoordinatorCore(GroupCoordinatorFixtureMixin, unittest.TestCase):
    def test_token_limited_group_seal_rejects_manifest_without_usage_reporting(self):
        fixture = build_rollout_fixture(self.root / "token-limited-group", mode="token_limited")
        session = self.runtime_session(fixture)
        policy = self.policy_for(fixture)
        policy["adapter_ref"] = session.manifest_ref
        coordinator = GroupCoordinatorV1(fixture.env, session=session)

        with self.assertRaises(AdapterContractError):
            coordinator.seal(
                fixture.runtime.checkpoint_id,
                policy=policy,
                group_seed=17,
                group_sequence=0,
                member_count=2,
            )

    def test_token_limited_session_binding_rejects_manifest_without_usage_reporting(self):
        fixture = build_rollout_fixture(self.root / "token-limited-bind", mode="token_limited")
        session = self.runtime_session(fixture)
        environment = RolloutEnvironment(
            fixture.store,
            fixture.entry.graph,
            session,
            fixture.gate,
            fixture.entry.graph.policy,
        )

        with self.assertRaises(AdapterContractError):
            environment.verify(fixture.runtime)

    def test_native_group_seal_refuses_scripted_and_other_v1_manifests(self):
        with self.assertRaises(AdapterContractError):
            self.seal_native(adapter_ref=self.session.manifest_ref, sequence=101)

        ports = tuple(
            RuntimePortDescriptorV1(
                schema=1,
                role=role,
                implementation=f"tests.{role.title()}",
                version="1",
                configuration={},
            )
            for role in ("sampling", "environment", "tools", "evaluator")
        )
        other_v1 = RuntimeManifestV1(schema=1, ports=ports)
        other_ref = self.store.put_artifact(other_v1.to_wire())
        with self.assertRaises(AdapterContractError):
            self.seal_native(adapter_ref=other_ref, sequence=102)

    def test_native_group_seal_refuses_each_missing_v2_capability(self):
        capabilities = {"usage_reporting", "native_token_ledger", "sampled_logprobs"}
        for index, missing in enumerate(sorted(capabilities), start=1):
            manifest, reference = self.native_manifest(capabilities - {missing})
            with self.subTest(missing=missing), self.assertRaises(AdapterContractError):
                self.seal_native(adapter_ref=reference, sequence=110 + index)
            self.assertEqual(manifest.identity(), reference)

    def test_native_group_seal_refuses_manifest_rendering_pin_drift(self):
        capabilities = {"usage_reporting", "native_token_ledger", "sampled_logprobs"}
        alternate_template = self.store.put_artifact({"template": "different"})
        _manifest, reference = self.native_manifest(capabilities, template_ref=alternate_template)
        with self.assertRaises(AdapterContractError):
            self.seal_native(adapter_ref=reference, sequence=120)

    def test_token_limited_run_requires_a_sealed_session_before_port_input(self):
        fixture = build_rollout_fixture(
            self.root / "token-limited-no-session", mode="token_limited"
        )
        self.assertIsNone(fixture.env.session)

        with self.assertRaises(AdapterContractError):
            fixture.env.step_input(fixture.runtime)

    def test_group_terminal_records_roundtrip_with_unchanged_payload_identity(self):
        fixture = GroupScriptedTerminalV1(
            schema=1,
            group_id="a" * 64,
            member_id="grp-example-00",
            start_checkpoint_id="b" * 64,
            execution_status="valid",
            reward_status="available",
            reward={"numerator": -3, "denominator": 2},
            native_optimizer_eligible=False,
        )
        failure = GroupExecutionFailureV1(
            schema=1,
            group_id="a" * 64,
            member_id="grp-example-00",
            start_checkpoint_id="b" * 64,
            reason="worker_crash",
            evidence_ref=None,
        )
        for record in (fixture, failure):
            with self.subTest(record=record.RECORD_TYPE):
                self.assertEqual(type(record).from_dict(record.to_wire()), record)
                self.assertEqual(record.identity(), self.store.put_artifact(record.to_wire()))

        with self.assertRaises(ValueError):
            GroupScriptedTerminalV1.from_dict({**fixture.to_wire(), "schema": 2})
        with self.assertRaises(ValueError):
            GroupScriptedTerminalV1.from_dict(
                {**fixture.to_wire(), "reward": {"numerator": -3, "denominator": 0}}
            )
        with self.assertRaises(ValueError):
            GroupExecutionFailureV1.from_dict({**failure.to_wire(), "reason": ""})

    def test_collect_completed_derives_terminal_and_availability_from_verified_view(self):
        spec = self.group()
        runtime, _manual_result = self.run_member(spec, 0)

        result_ref = self.coordinator.collect_completed(spec, 0, runtime)
        result = GroupMemberResultV1.from_dict(self.store.get_artifact(result_ref))
        view = self.env.verify(runtime)
        expected_terminal_ref = view.state.outcome_ref
        if view.outcome.reward_status == "available":
            expected_terminal_ref = self.store.get_artifact(view.outcome.reward_ref)[
                "terminal_outcome_ref"
            ]

        self.assertEqual(result.member_id, spec.members[0].member_id)
        self.assertEqual(
            result.start_checkpoint_id,
            self.coordinator.start_receipt(spec, 0)["start_checkpoint_id"],
        )
        self.assertEqual(result.final_checkpoint_id, runtime.checkpoint_id)
        self.assertEqual(result.terminal_outcome_ref, expected_terminal_ref)
        self.assertEqual(
            result.availability_ref,
            view.outcome.reward_ref if view.outcome.reward_status == "available" else None,
        )

    def test_group_sequence_index_returns_the_verified_sealed_spec(self):
        spec = self.group(sequence=89)
        (self.coordinator.groups_root / "step-000089").write_bytes(
            canonical_bytes(
                {
                    "schema": 1,
                    "step": 89,
                    "task_id": "task-1",
                    "group_id": spec.group_id,
                    "status": "sealed",
                }
            )
        )

        self.assertEqual(
            GroupCoordinatorV1.groups_by_sequence(self.coordinator.groups_root)[89], spec
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

    def test_real_group_seal_rejects_a_non_manifest_adapter_pin(self):
        unbound_environment = RolloutEnvironment(
            self.store,
            self.fixture.entry.graph,
            None,
            self.fixture.gate,
            self.fixture.entry.graph.policy,
        )
        unbound_coordinator = GroupCoordinatorV1(unbound_environment)

        with self.assertRaises(AdapterContractError):
            unbound_coordinator.seal(
                self.entry_id,
                policy=self.policy_for(self.fixture),
                group_seed=771,
                group_sequence=88,
                member_count=2,
                runner_mode="real",
            )

    def test_real_group_start_rejects_an_unbound_session_before_publishing_member(self):
        spec = self.group(sequence=88)
        other_backend = ScriptedSampleBackend(self.fixture.sample_results)
        other_backend.descriptor = PortDescriptorV1("sampling", "other-group-session", "1")
        other_session = self.runtime_session(self.fixture, backend=other_backend)
        self.env.session = other_session

        with self.assertRaises(AdapterContractError):
            self.coordinator.start(spec, 0, policy=self.policy)

        self.assertIsNone(self.store.read_head(spec.members[0].member_id))

    def test_real_group_member_rejects_sessionless_start_and_commit(self):
        spec = self.group(sequence=91)
        member = spec.members[0]
        sessionless = RolloutEnvironment(
            self.store,
            self.fixture.entry.graph,
            None,
            self.fixture.gate,
            self.fixture.entry.graph.policy,
        )

        with self.assertRaises(AdapterContractError):
            sessionless.start_member(self.entry_id, MemberStartV1(spec.identity(), member.ordinal))
        self.assertIsNone(self.store.read_head(member.member_id))

        runtime = self.coordinator.start(spec, member.ordinal, policy=self.policy)
        head = self.store.read_head(member.member_id)
        view, _directive, port = self.env.step_input(runtime)
        claims = {
            "context_revision_ref": view.context.revision_ref,
            "context_content_hash": view.context.content_ref,
            "rendering": thaw(view.context.rendering),
        }
        turn = WriterTurnV1(
            action_id=port.action_id,
            context_revision_ref=port.context_revision_ref,
            raw_output_ref=None,
            usage={},
            adapter_trace=claims,
            message=intake_message({"role": "assistant", "content": "draft"}),
        )

        with self.assertRaises(AdapterContractError):
            sessionless.commit(runtime, turn)
        with self.assertRaises(AdapterContractError):
            sessionless.step_input(runtime)
        self.assertEqual(self.store.read_head(member.member_id), head)
        self.assertFalse((self.coordinator.groups_root / spec.group_id / "result-0.json").exists())
        self.assertEqual(self.coordinator.finalize(spec).status, "pending")

    def test_orphaned_reward_checkpoint_cannot_be_collected_or_finalize_group(self):
        spec = self.group(sequence=89)
        member = spec.members[0]
        runtime = self.coordinator.start(spec, 0, policy=self.policy)
        runtime = run_slice(
            self.fixture,
            runtime=runtime,
            until=lambda directive: directive.kind == "publish_reward",
        )
        current_head = self.store.read_head(member.member_id)
        orphan = {}
        publish = self.store.publish

        class InjectedCrash(RuntimeError):
            pass

        def publish_with_fault(lineage_id, expected_head, events, next_state, **kwargs):
            if lineage_id == member.member_id and events[0].kind == "reward_recorded":
                parent_checkpoint = self.store.load_commit(expected_head).checkpoint
                orphan_checkpoint = CheckpointV1(
                    parents=(parent_checkpoint,),
                    state=next_state,
                    event_head=next_state.history["head"],
                )
                orphan["checkpoint_id"] = orphan_checkpoint.identity()

                def crash_at_head_publication(stage):
                    if stage == "before_head_publication":
                        raise InjectedCrash(stage)

                kwargs["fault"] = crash_at_head_publication
            return publish(lineage_id, expected_head, events, next_state, **kwargs)

        with patch.object(self.store, "publish", side_effect=publish_with_fault):
            with self.assertRaises(InjectedCrash):
                self.fixture.driver().run(runtime, max_steps=1)

        self.assertEqual(self.store.read_head(member.member_id), current_head)
        self.assertIsNotNone(self.store.load_checkpoint(orphan["checkpoint_id"]))
        published = self.env.open_head(member.member_id)
        published_view = self.env.verify(published)
        result = GroupMemberResultV1(
            group_id=spec.group_id,
            member_id=member.member_id,
            start_checkpoint_id=self.coordinator.start_receipt(spec, 0)["start_checkpoint_id"],
            final_checkpoint_id=orphan["checkpoint_id"],
            terminal_outcome_ref=published_view.state.outcome_ref,
            execution_status="valid",
        )

        with self.assertRaises(ConcurrentUpdateError):
            self.coordinator.collect(spec, result)

        self.assertEqual(self.coordinator.finalize(spec).status, "pending")

    def test_collect_invalid_rejects_a_member_with_a_valid_terminal_reward(self):
        spec = self.group(sequence=90)
        self.run_member(spec, 0)
        published = self.store.read_head(spec.members[0].member_id)

        with self.assertRaises(AdapterContractError):
            self.coordinator.collect_invalid(spec, 0, reason="claimed interruption")

        self.assertEqual(self.store.read_head(spec.members[0].member_id), published)
        self.assertEqual(self.coordinator.finalize(spec).status, "pending")

    def test_collect_invalid_binds_the_judged_head_before_a_later_valid_terminal(self):
        spec = self.group(sequence=92)
        member = spec.members[0]
        runtime = self.coordinator.start(spec, 0, policy=self.policy)
        runtime = run_slice(
            self.fixture,
            runtime=runtime,
            until=lambda directive: directive.kind == "execute_tool",
        )
        judged_checkpoint_id = runtime.checkpoint_id

        self.coordinator.collect_invalid(spec, 0, reason="worker_interrupted")
        failure_receipt = load_canonical_json(
            (self.coordinator.groups_root / spec.group_id / "result-0.json").read_bytes()
        )
        failure_result = GroupMemberResultV1.from_dict(
            self.store.get_artifact(failure_receipt["result_ref"])
        )

        finished = run_slice(self.fixture, runtime=self.env.open_head(member.member_id))
        self.assertNotEqual(finished.checkpoint_id, judged_checkpoint_id)
        self.assertEqual(self.env.verify(finished).outcome.execution_status, "valid")
        self.assertEqual(self.coordinator.finalize(spec).status, "invalid")
        failure = self.store.get_artifact(failure_result.failure_ref)
        self.assertEqual(failure["judged_checkpoint_id"], judged_checkpoint_id)

    def test_group_sampling_requires_all_active_context_claims(self):
        required = {
            "context_revision_ref",
            "context_content_hash",
            "rendering",
        }
        for index, missing in enumerate(sorted(required), start=91):
            with self.subTest(missing=missing):
                spec = self.group(sequence=index)
                runtime = self.coordinator.start(spec, 0, policy=self.policy)
                view, _directive, port = self.env.step_input(runtime)
                claims = {
                    "context_revision_ref": view.context.revision_ref,
                    "context_content_hash": view.context.content_ref,
                    "rendering": thaw(view.context.rendering),
                }
                claims.pop(missing)
                turn = WriterTurnV1(
                    action_id=port.action_id,
                    context_revision_ref=port.context_revision_ref,
                    raw_output_ref=None,
                    usage={},
                    adapter_trace=claims,
                    message=intake_message({"role": "assistant", "content": "draft"}),
                )
                head = self.store.read_head(spec.members[0].member_id)

                with self.assertRaises(AdapterContractError):
                    self.env.commit(runtime, turn)

                self.assertEqual(self.store.read_head(spec.members[0].member_id), head)

    def test_pending_reward_resolution_accepts_only_a_reward_recorded_suffix(self):
        spec = self.group(sequence=93)
        member0 = spec.members[0]
        runtime0 = self.coordinator.start(spec, 0, policy=self.policy)
        runtime0 = run_slice(
            self.fixture,
            runtime=runtime0,
            until=lambda directive: directive.kind == "publish_reward",
        )
        pending_view = self.env.verify(runtime0)
        pending_result = GroupMemberResultV1(
            group_id=spec.group_id,
            member_id=member0.member_id,
            start_checkpoint_id=self.coordinator.start_receipt(spec, 0)["start_checkpoint_id"],
            final_checkpoint_id=runtime0.checkpoint_id,
            terminal_outcome_ref=pending_view.state.outcome_ref,
            execution_status="valid",
        )
        self.coordinator.collect(spec, pending_result)
        _runtime1, result1 = self.run_member(spec, 1)
        self.coordinator.collect(spec, result1)

        runtime0 = run_slice(self.fixture, runtime=runtime0)
        resolved_view = self.env.verify(runtime0)
        self.assertEqual(resolved_view.outcome.reward_status, "available")
        with self.assertRaises(ConcurrentUpdateError):
            self.coordinator.collect(spec, pending_result)
        decision = self.coordinator.finalize(spec)
        self.assertIn(decision.status, {"ready", "tie"})
        self.assertEqual(len(decision.advantage_refs), 2)

        self.assertTrue(GroupCoordinatorV1._reward_only_suffix(("reward_recorded",)))
        self.assertFalse(
            GroupCoordinatorV1._reward_only_suffix(("reward_recorded", "writer_action"))
        )

    def test_start_retry_after_member_advanced_keeps_receipt_and_allows_collect(self):
        spec = self.group(sequence=77)
        member = spec.members[0]
        runtime = self.env.start_member(
            spec.environment["entry_checkpoint_id"], MemberStartV1(spec.identity(), 0)
        )
        start_checkpoint_id = runtime.checkpoint_id
        self.fixture.gatherers = make_gatherers(self.fixture)
        advanced = run_slice(
            self.fixture,
            runtime=runtime,
            until=lambda directive: directive.kind == "execute_tool",
        )
        self.assertNotEqual(advanced.checkpoint_id, start_checkpoint_id)

        resumed = self.coordinator.start(spec, 0, policy=self.policy)
        body = load_canonical_json(
            (self.coordinator.groups_root / spec.group_id / "start-0.json").read_bytes()
        )
        self.assertEqual(body["start_checkpoint_id"], start_checkpoint_id)
        self.assertEqual(resumed.checkpoint_id, advanced.checkpoint_id)
        result = GroupMemberResultV1(
            group_id=spec.group_id,
            member_id=member.member_id,
            start_checkpoint_id=start_checkpoint_id,
        )
        result_ref = self.coordinator.collect(spec, result)
        self.assertEqual(self.store.get_artifact(result_ref), result.to_dict())

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
        offline = GroupCoordinatorV1(offline_env)
        with ports_disabled():
            replayed = offline.finalize(spec)
        self.assertEqual(replayed.identity(), decision.identity())


class GroupCoordinatorTests:
    """New-core entry fixture retained for derive tests that share its group inputs."""

    def __init__(self, _method_name=None):
        self.cleanups = []

    def addCleanup(self, function, *args, **kwargs):
        self.cleanups.append((function, args, kwargs))

    def doCleanups(self):
        while self.cleanups:
            function, args, kwargs = self.cleanups.pop()
            function(*args, **kwargs)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = build_rollout_fixture(Path(temporary.name) / "fixture", mode="slice")
        self.fixture = fixture
        self.root = fixture.root
        self.store = fixture.store
        self.start = fixture.runtime.checkpoint_id
        self.runtime = fixture.runtime
        self.session = GroupCoordinatorFixtureMixin.runtime_session(fixture)
        fixture.env.session = self.session
        self.coordinator = GroupCoordinatorV1(fixture.env, session=self.session)
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
        context_policy = ContextPolicyV1(
            "compact", summarizer_version="visible-text-v1", max_summary_chars=20
        )
        self.policy["context_policy_ref"] = self.store.put_artifact(context_policy.to_wire())
        self.policy["adapter_ref"] = self.session.manifest_ref

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
