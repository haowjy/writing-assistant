"""Native batch export and fail-closed token-span contracts."""

from __future__ import annotations

import math
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from tests.task_graph_fixtures import MemoryArtifactReader
from tests.task_graph_rollout_fixtures import run_slice
from tests.test_task_graph_v2_writer import _bound_native_lineage, _native_view, _turn
from writing_agent.task_graph import CheckpointV1, EventV1, domain_hash
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_eligibility import decide_eligibility
from writing_agent.task_graph_gate import StoreArtifactReader
from writing_agent.task_graph_group import GroupCoordinatorV1, GroupInvariantError
from writing_agent.task_graph_group_contract import derive_group_seed
from writing_agent.task_graph_group_records import (
    GroupAdvantageV1,
    GroupDecisionV1,
    GroupError,
    GroupMemberResultV1,
    GroupSegmentCreditV1,
    fraction_wire,
)
from writing_agent.task_graph_record_contracts import (
    ContextPolicyV1,
    GroupMemberSpecV1,
    GroupSpecV1,
    group_identity,
)
from writing_agent.task_graph_records import (
    ContextOperationInputV1,
    EnvironmentStepV1,
    OutcomeV1,
    WriterTurnV1,
)
from writing_agent.task_graph_training_export import (
    TrainingExportError,
    export_training_batch,
    read_advantage_f64,
    training_turn_spans,
)
from writing_agent.task_graph_training_records import TrainingBatchV1
from writing_agent.task_graph_transition import SampleRef


def _token_bytes(ids):
    return b"".join(token.to_bytes(4, "little") for token in ids)


def _read_ids(reader, ref):
    data = reader.bytes_artifact(ref)
    return tuple(
        int.from_bytes(data[offset : offset + 4], "little") for offset in range(0, len(data), 4)
    )


def _with_context_cap(
    spec: GroupSpecV1,
    reader: MemoryArtifactReader,
    cap: int | None,
    *,
    member_count: int | None = None,
):
    budget = {"limits": {}, "consumed": {}}
    if cap is not None:
        budget["limits"]["context_tokens"] = cap
    budget_ref = reader.add(budget)
    environment = {
        **dict(spec.environment),
        "budget_ref": budget_ref,
        "budget_hash": domain_hash("payload", budget),
    }
    member_count = len(spec.members) if member_count is None else member_count
    group_id = group_identity(
        spec.group_sequence,
        environment,
        spec.policy,
        spec.group_seed,
        spec.runner_mode,
        member_count,
        spec.training_mode,
    )
    members = tuple(
        GroupMemberSpecV1(
            member_id=f"grp-{group_id[:24]}-{ordinal:02d}",
            ordinal=ordinal,
            writer_seed=derive_group_seed(spec.group_seed, "writer", ordinal),
            environment_seed=derive_group_seed(spec.group_seed, "environment"),
        )
        for ordinal in range(member_count)
    )
    return replace(spec, group_id=group_id, environment=environment, members=members)


def _advantage(spec, member_id, result_ref, ordinal, *, tie):
    if tie:
        reward = fraction_wire(Fraction(1, 2))
        return GroupAdvantageV1(
            group_id=spec.group_id,
            member_id=member_id,
            result_ref=result_ref,
            reward=reward,
            mean=reward,
            variance=fraction_wire(Fraction()),
            centered=fraction_wire(Fraction()),
            expression="zero",
            advantage=fraction_wire(Fraction()),
            zero_variance=True,
        )
    if len(spec.members) == 3:
        mean = Fraction(2, 3)
        variance = Fraction(2, 9)
        reward = Fraction(0) if ordinal == 0 else Fraction(1)
        centered = reward - mean
    else:
        mean = Fraction(1, 2)
        variance = Fraction(1, 4)
        reward = Fraction(ordinal)
        centered = reward - mean
    return GroupAdvantageV1(
        group_id=spec.group_id,
        member_id=member_id,
        result_ref=result_ref,
        reward=fraction_wire(reward),
        mean=fraction_wire(mean),
        variance=fraction_wire(variance),
        centered=fraction_wire(centered),
        expression="centered / sqrt(population_variance)",
        zero_variance=False,
    )


def _turn_specs(*, trailing_context_limit=False, empty=False):
    if empty:
        return [
            {
                "input_ids": (10, 11, 12),
                "generated_ids": (),
                "termination_kind": "context_limit",
                "stop_token_id": None,
                "limit": "context",
                "content": "",
            }
        ]
    turns = [
        {"input_ids": (10, 11), "generated_ids": (20, 21), "content": "first"},
        {
            "input_ids": (10, 11, 20, 21, 30),
            "generated_ids": (40, 1),
            "content": "second",
        },
    ]
    if trailing_context_limit:
        turns.append(
            {
                "input_ids": (10, 11, 20, 21, 30, 40, 1, 99),
                "generated_ids": (),
                "termination_kind": "context_limit",
                "stop_token_id": None,
                "limit": "context",
                "content": "",
            }
        )
    return turns


def _credit_spans(turns, reader):
    completion = []
    spans = {}
    previous_input = None
    previous_generated = None
    for turn in turns:
        generated = _read_ids(reader, turn.generated_token_ids_ref)
        input_ids = _read_ids(reader, turn.input_token_ids_ref)
        if not generated:
            continue
        if previous_input is not None:
            prefix = previous_input + previous_generated
            external = input_ids[len(prefix) :]
            completion.extend(external)
        start = len(completion)
        completion.extend(generated)
        spans[turn.identity()] = (start, len(completion))
        previous_input, previous_generated = input_ids, generated
    return spans


def _eligibility_view(reader, spec, turns, base_view, *, ordinal=0):
    member_id = spec.members[ordinal].member_id
    events = sorted(
        (
            EventV1.from_dict(value)
            for value in reader.events.values()
            if value["lineage_id"] == member_id
        ),
        key=lambda event: event.seq,
    )
    event_by_turn_ref = {
        event.payload_ref: event for event in events if event.kind == "writer_action"
    }
    samples = tuple(
        SampleRef(
            action_id=turn.action_id,
            event_id=event_by_turn_ref[turn.identity()].identity(),
            turn_ref=turn.identity(),
            outcome="action" if turn.generated_token_count else "budget_stop",
        )
        for turn in turns[ordinal]
    )
    state_data = base_view.state.to_dict()
    state_data["position"]["lineage_id"] = member_id
    return replace(
        base_view,
        state=type(base_view.state).from_dict(state_data),
        group=spec,
        samples=samples,
        head_event_id=events[-1].identity(),
        outcome=replace(base_view.outcome, execution_status="valid"),
    )


def _build_group(
    *,
    cap=7,
    turns_by_member=None,
    statuses=None,
    tie=False,
    reset_member=None,
    reset_operation="compact",
    member_count=None,
    budget_charged_member=None,
):
    fixture, base_view, _manifest = _native_view()
    reader = fixture.reader
    spec = _with_context_cap(base_view.group, reader, cap, member_count=member_count)
    statuses = statuses or ["structurally_eligible"] * len(spec.members)
    turns_by_member = turns_by_member or [
        _turn_specs(trailing_context_limit=(ordinal == 0)) for ordinal in range(len(spec.members))
    ]

    result_refs = []
    advantage_refs = []
    segment_credit_refs = []
    all_turn_records = []
    for ordinal, member in enumerate(spec.members):
        turns = []
        sample_events = []
        start = EventV1(
            seq=1,
            lineage_id=member.member_id,
            rollout_id=member.member_id,
            node_visit_id="visit-1",
            kind="rollout_started",
            actor="environment",
            audience=("controller",),
            payload_ref=spec.environment["entry_checkpoint_id"],
            versions_ref=spec.environment["versions_ref"],
            provenance_ref=spec.environment["provenance_ref"],
        )
        reader.events[start.identity()] = start.to_dict()
        previous_event = start.identity()
        event_sequence = 2
        for turn_ordinal, values in enumerate(turns_by_member[ordinal]):
            turn = _turn(
                base_view,
                reader,
                **values,
            )
            turn = replace(turn, action_id=f"{member.member_id}:action:{turn_ordinal}")
            turn_ref = reader.add(turn.to_wire())
            event = EventV1(
                previous=previous_event,
                seq=event_sequence,
                lineage_id=member.member_id,
                rollout_id=member.member_id,
                node_visit_id="visit-1",
                kind=(
                    "budget_charged"
                    if budget_charged_member == ordinal and turn_ordinal == 0
                    else "writer_action"
                ),
                actor="writer",
                audience=("trainer", "writer"),
                payload_ref=turn_ref,
                versions_ref=spec.environment["versions_ref"],
                provenance_ref=spec.environment["provenance_ref"],
            )
            reader.events[event.identity()] = event.to_dict()
            turns.append(turn)
            sample_events.append((turn, turn_ref, event))
            previous_event = event.identity()
            event_sequence += 1
            if reset_member == ordinal and turn_ordinal == 0:
                policy = (
                    ContextPolicyV1(
                        "compact", summarizer_version="visible-text-v1", max_summary_chars=20
                    )
                    if reset_operation == "compact"
                    else ContextPolicyV1(operation=reset_operation)
                )
                policy_ref = reader.add(policy.to_wire())
                operation = ContextOperationInputV1(policy_ref=policy_ref)
                reader.add(operation.to_wire())
                reset = EventV1(
                    previous=previous_event,
                    seq=event_sequence,
                    lineage_id=member.member_id,
                    rollout_id=member.member_id,
                    node_visit_id="visit-1",
                    kind="context_changed",
                    actor="environment",
                    audience=("controller", "trainer"),
                    payload_ref=operation.identity(),
                    versions_ref=spec.environment["versions_ref"],
                    provenance_ref=spec.environment["provenance_ref"],
                )
                reader.events[reset.identity()] = reset.to_dict()
                previous_event = reset.identity()
                event_sequence += 1

        all_turn_records.append(tuple(turns))

        terminal = OutcomeV1(
            schema=1,
            task_status="incomplete" if turns[-1].generated_token_count == 0 else "complete",
            execution_status="valid",
            stop_reason="context_tokens_budget" if turns[-1].generated_token_count == 0 else None,
            reward_status="pending",
            training_eligibility="pending",
            candidate_checkpoint=spec.environment["entry_checkpoint_id"],
            requirement_version=spec.environment["requirements_ref"],
            checks=(),
            transition_edge_id=None,
            failed_request_ref=None,
            reward_ref=None,
            eligibility_ref=None,
        )
        terminal_ref = reader.add(terminal.to_wire())
        status = statuses[ordinal]
        reason = (
            "native_evidence_structural"
            if status == "structurally_eligible"
            else "native_action_trace_unavailable"
        )
        eligibility = {
            "record_type": "TrainingEligibilityV1",
            "schema": 1,
            "terminal_outcome_ref": terminal_ref,
            "status": status,
            "reason": reason,
        }
        eligibility_ref = reader.add(eligibility)
        reward_numerator = 1 if tie else 0 if ordinal == 0 else max(1, len(spec.members) - 1)
        reward_normalization = 2 if tie else max(1, len(spec.members) - 1)
        reward = {
            "record_type": "RewardV1",
            "schema": 1,
            "terminal_outcome_ref": terminal_ref,
            "reward_contract_ref": spec.environment.get("reward_contract_hash") or "f" * 64,
            "candidate_checkpoint": spec.environment["entry_checkpoint_id"],
            "check_result_refs": [],
            "components": {},
            "numerator": reward_numerator,
            "normalization": reward_normalization,
            "availability": "available",
            "eligibility_ref": eligibility_ref,
        }
        reward_ref = reader.add(reward)
        outcome = replace(
            terminal,
            reward_status="available",
            training_eligibility=status,
            reward_ref=reward_ref,
            eligibility_ref=eligibility_ref,
        )
        reader.add(outcome.to_wire())
        state_data = base_view.state.to_dict()
        state_data["position"]["lineage_id"] = member.member_id
        state_data["position"]["phase"] = "terminal"
        state_data["history"]["head"] = previous_event
        state_data["history"]["seq"] = event_sequence - 1
        state_data["outcome_ref"] = outcome.identity()
        state = type(base_view.state).from_dict(state_data)
        checkpoint = CheckpointV1(
            parents=(spec.environment["entry_checkpoint_id"],),
            state=state,
            event_head=previous_event,
        )
        reader.checkpoints[checkpoint.identity()] = checkpoint

        result = GroupMemberResultV1(
            group_id=spec.group_id,
            member_id=member.member_id,
            start_checkpoint_id=spec.environment["entry_checkpoint_id"],
            final_checkpoint_id=checkpoint.identity(),
            terminal_outcome_ref=terminal_ref,
            availability_ref=reward_ref,
            execution_status="valid",
        )
        result_ref = reader.add(result.to_wire())
        advantage = _advantage(spec, member.member_id, result_ref, ordinal, tie=tie)
        advantage_ref = reader.add(advantage.to_wire())
        result_refs.append(result_ref)
        advantage_refs.append(advantage_ref)

        spans = _credit_spans(turns, reader)
        for turn, turn_ref, event in sample_events:
            span = spans.get(turn_ref)
            credit = GroupSegmentCreditV1(
                group_id=spec.group_id,
                member_id=member.member_id,
                action_id=turn.action_id,
                action_ref=event.identity(),
                message_ref="a" * 64,
                trace_ref=turn_ref,
                original_context_ref=turn.context_revision_ref,
                original_context_content_hash="b" * 64,
                advantage_ref=advantage_ref,
                segment_kind="assistant_ending",
                part_index=None,
                segment_content_hash=None,
                completion_start=None if span is None else span[0],
                completion_end=None if span is None else span[1],
            )
            segment_credit_refs.append(reader.add(credit.to_wire()))

    decision = GroupDecisionV1(
        group_id=spec.group_id,
        status="tie" if tie else "ready",
        reason="finalized",
        member_result_refs=tuple(result_refs),
        advantage_refs=tuple(advantage_refs),
        segment_credit_refs=tuple(segment_credit_refs),
    )
    return reader, spec, decision, turns_by_member, tuple(all_turn_records)


class TrainingBatchExportTests(unittest.TestCase):
    def test_real_coordinator_exports_tool_suffix_and_trailing_context_limit(self):
        tool_call = {
            "id": "write-1",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": {"path": "draft.txt", "content": "x"},
            },
        }
        with tempfile.TemporaryDirectory() as root:
            fixture, session, base_spec, _runtime = _bound_native_lineage(
                Path(root) / "group", mode="context_token_limited", max_tokens=4
            )
            coordinator = GroupCoordinatorV1(fixture.env, session=session)
            spec = coordinator.seal(
                fixture.runtime.checkpoint_id,
                policy=dict(base_spec.policy),
                group_seed=18,
                group_sequence=94,
                member_count=2,
                runner_mode="real",
                training_mode="native",
            )
            reader = StoreArtifactReader(fixture.store)

            for ordinal, member in enumerate(spec.members):
                runtime = coordinator.start(spec, ordinal, policy=spec.policy)
                view = fixture.env.verify(runtime)
                first_turn = _turn(
                    view,
                    reader,
                    input_ids=(10, 11),
                    generated_ids=(12, 50),
                    stop_token_id=50,
                    calls=(tool_call,),
                    content="",
                    sampling_pins={"seed": member.writer_seed},
                )
                runtime = fixture.env.commit(runtime, first_turn).runtime
                runtime = run_slice(
                    fixture,
                    runtime=runtime,
                    until=lambda directive: directive.kind == "sample_writer",
                )
                view = fixture.env.verify(runtime)
                if ordinal == 0:
                    final_turn = _turn(
                        view,
                        reader,
                        input_ids=(10, 11, 12, 50, 77, 78),
                        generated_ids=(20, 21, 22, 23),
                        termination_kind="token_limit",
                        stop_token_id=None,
                        limit="decision",
                        sampling_pins={"seed": member.writer_seed},
                    )
                else:
                    context_cap = view.budget["limits"]["context_tokens"]
                    final_turn = _turn(
                        view,
                        reader,
                        input_ids=(10, 11, 12, 50) + tuple(range(300, 300 + context_cap)),
                        generated_ids=(),
                        termination_kind="context_limit",
                        stop_token_id=None,
                        limit="context",
                        content="",
                        sampling_pins={"seed": member.writer_seed},
                    )
                terminal_runtime = fixture.env.commit(runtime, final_turn).runtime
                terminal = fixture.env.verify(terminal_runtime)
                self.assertEqual(next_step(terminal).kind, "publish_reward")
                final = fixture.env.verify(
                    fixture.env.commit(
                        terminal_runtime,
                        EnvironmentStepV1(directive={"kind": "publish_reward"}),
                    ).runtime
                )
                self.assertEqual(final.outcome.training_eligibility, "structurally_eligible")
                result = GroupMemberResultV1(
                    group_id=spec.group_id,
                    member_id=member.member_id,
                    start_checkpoint_id=coordinator.start_receipt(spec, ordinal)[
                        "start_checkpoint_id"
                    ],
                    final_checkpoint_id=final.checkpoint_id,
                    terminal_outcome_ref=terminal.state.outcome_ref,
                    availability_ref=final.outcome.reward_ref,
                    execution_status="valid",
                )
                coordinator.collect(spec, result)

            decision = coordinator.finalize(spec)
            exported = export_training_batch(spec, decision, reader)
            artifacts = {artifact.ref: artifact.value for artifact in exported.artifacts}
            members = exported.batch.members

            def token_ids(ref):
                data = artifacts[ref]
                return tuple(
                    int.from_bytes(data[offset : offset + 4], "little")
                    for offset in range(0, len(data), 4)
                )

            self.assertEqual(
                token_ids(members[0]["completion_ids_ref"]), (12, 50, 77, 78, 20, 21, 22, 23)
            )
            self.assertEqual(artifacts[members[0]["env_mask_ref"]], bytes((1, 1, 0, 0, 1, 1, 1, 1)))
            self.assertEqual(
                [
                    (span["ext_start"], span["completion_start"], span["completion_end"])
                    for span in members[0]["turn_spans"]
                ],
                [(0, 0, 2), (2, 4, 8)],
            )
            self.assertEqual(token_ids(members[1]["completion_ids_ref"]), (12, 50))
            self.assertEqual(artifacts[members[1]["env_mask_ref"]], bytes((1, 1)))
            self.assertEqual(
                [
                    (span["ext_start"], span["completion_start"], span["completion_end"])
                    for span in members[1]["turn_spans"]
                ],
                [(0, 0, 2)],
            )
            self.assertIn("trailing_context_limit_turn_ref", members[1])
            self.assertNotIn("trailing_context_limit_turn_ref", members[0])

            credits = [
                GroupSegmentCreditV1.from_dict(reader.artifact(ref))
                for ref in decision.segment_credit_refs
            ]
            self.assertEqual(
                [
                    (
                        credit.member_id,
                        credit.segment_kind,
                        credit.completion_start,
                        credit.completion_end,
                    )
                    for credit in credits
                ],
                [
                    (spec.members[0].member_id, "tool_syntax", 0, 2),
                    (spec.members[0].member_id, "assistant_ending", 0, 2),
                    (spec.members[0].member_id, "assistant_text", 4, 8),
                    (spec.members[0].member_id, "assistant_ending", 4, 8),
                    (spec.members[1].member_id, "tool_syntax", 0, 2),
                    (spec.members[1].member_id, "assistant_ending", 0, 2),
                    (spec.members[1].member_id, "assistant_ending", None, None),
                ],
            )

            with patch(
                "writing_agent.task_graph_group.training_turn_spans",
                side_effect=GroupError("eligible layout is inconsistent"),
            ):
                with self.assertRaises(GroupInvariantError):
                    coordinator.finalize(spec)

    def test_round_trip_reconstructs_last_generated_prefix_and_masks(self):
        reader, spec, decision, _turn_specs_by_member, turn_records = _build_group()

        exported = export_training_batch(spec, decision, reader)
        for artifact in exported.artifacts:
            reader.byte_values[artifact.ref] = artifact.value
        self.assertEqual(exported.batch.max_context_tokens, 7)

        for ordinal, member in enumerate(exported.batch.members):
            generated_turns = [turn for turn in turn_records[ordinal] if turn.generated_token_count]
            expected_turn = generated_turns[-1]
            prompt = _read_ids(reader, member["prompt_ids_ref"])
            completion = _read_ids(reader, member["completion_ids_ref"])
            mask = tuple(reader.bytes_artifact(member["env_mask_ref"]))
            expected_ledger = _read_ids(reader, expected_turn.input_token_ids_ref) + _read_ids(
                reader, expected_turn.generated_token_ids_ref
            )
            result = GroupMemberResultV1.from_dict(
                reader.artifact(decision.member_result_refs[ordinal])
            )
            with self.subTest(member=ordinal):
                self.assertEqual(prompt + completion, expected_ledger)
                self.assertEqual(
                    sum(mask),
                    sum(turn.generated_token_count for turn in generated_turns),
                )
                self.assertEqual(len(mask), len(completion))
                self.assertEqual(mask, (1, 1, 0, 1, 1))
                self.assertLessEqual(
                    len(prompt) + len(completion), exported.batch.max_context_tokens
                )
                self.assertEqual(
                    training_turn_spans(
                        result.final_checkpoint_id,
                        member["member_id"],
                        reader,
                        max_context_tokens=exported.batch.max_context_tokens,
                    ),
                    _credit_spans(turn_records[ordinal], reader),
                )

    def test_trailing_context_limit_is_audit_only_and_other_members_omit_it(self):
        reader, spec, decision, _turn_specs_by_member, _turn_records = _build_group()

        exported = export_training_batch(spec, decision, reader)
        for artifact in exported.artifacts:
            reader.byte_values[artifact.ref] = artifact.value
        member_with_audit = exported.batch.members[0]
        member_without_audit = exported.batch.members[1]
        self.assertIn("trailing_context_limit_turn_ref", member_with_audit)
        self.assertNotIn("trailing_context_limit_turn_ref", member_without_audit)
        self.assertEqual(len(member_with_audit["turn_spans"]), 2)
        self.assertNotIn(
            member_with_audit["trailing_context_limit_turn_ref"],
            {span["turn_ref"] for span in member_with_audit["turn_spans"]},
        )
        self.assertEqual(
            reader.bytes_artifact(member_with_audit["env_mask_ref"]),
            bytes((1, 1, 0, 1, 1)),
        )

    def test_tie_group_exports_exact_float64_zeros(self):
        reader, spec, decision, _turn_specs_by_member, _turn_records = _build_group(tie=True)

        exported = export_training_batch(spec, decision, reader)
        for artifact in exported.artifacts:
            reader.byte_values[artifact.ref] = artifact.value
        self.assertEqual(
            [read_advantage_f64(member, reader) for member in exported.batch.members],
            [0.0, 0.0],
        )

    def test_irrational_exact_advantage_is_computed_once_as_float64(self):
        reader, spec, decision, _turn_specs_by_member, _turn_records = _build_group(
            cap=10,
            member_count=3,
            turns_by_member=[_turn_specs() for _ in range(3)],
        )

        exported = export_training_batch(spec, decision, reader)
        for artifact in exported.artifacts:
            reader.byte_values[artifact.ref] = artifact.value
        advantages = [read_advantage_f64(member, reader) for member in exported.batch.members]
        self.assertEqual(advantages[0].hex(), "-0x1.6a09e667f3bcdp+0")
        self.assertAlmostEqual(advantages[0], -math.sqrt(2), places=15)
        self.assertAlmostEqual(advantages[1], 1 / math.sqrt(2), places=15)
        self.assertAlmostEqual(advantages[2], 1 / math.sqrt(2), places=15)
        self.assertEqual(
            exported.batch.members[0]["advantage"].variance,
            {"numerator": 2, "denominator": 9},
        )

    def test_pending_and_invalid_groups_refuse_export(self):
        reader, spec, _decision, _turn_specs_by_member, _turn_records = _build_group()
        for status in ("pending", "invalid"):
            decision = GroupDecisionV1(group_id=spec.group_id, status=status, reason=status)
            with self.subTest(status=status), self.assertRaises(TrainingExportError) as caught:
                export_training_batch(spec, decision, reader)
            self.assertEqual(caught.exception.reason_code, "group_not_exportable")

    def test_non_structurally_eligible_member_refuses_export(self):
        reader, spec, decision, _turn_specs_by_member, _turn_records = _build_group(
            statuses=["ineligible", "structurally_eligible"]
        )
        with self.assertRaises(TrainingExportError) as caught:
            export_training_batch(spec, decision, reader)
        self.assertEqual(caught.exception.reason_code, "member_not_eligible")

    def test_multiple_context_roots_refuse_export(self):
        reader, spec, decision, _turn_specs_by_member, _turn_records = _build_group(reset_member=0)
        with self.assertRaises(TrainingExportError) as caught:
            export_training_batch(spec, decision, reader)
        self.assertEqual(caught.exception.reason_code, "invalid_member_layout")

    def test_carry_context_change_is_ineligible_and_refused_by_export(self):
        reader, spec, decision, _, turns = _build_group(
            reset_member=0,
            reset_operation="carry",
        )
        _, base_view, _ = _native_view()
        view = _eligibility_view(reader, spec, turns, base_view)

        eligibility = decide_eligibility(view, reader)

        self.assertEqual(eligibility.status, "ineligible")
        self.assertEqual(eligibility.reason, "multi_segment_context")
        with self.assertRaises(TrainingExportError) as caught:
            export_training_batch(spec, decision, reader)
        self.assertEqual(caught.exception.reason_code, "invalid_member_layout")

    def test_empty_mask_refuses_export(self):
        turns = [_turn_specs(empty=True), _turn_specs()]
        reader, spec, decision, _, _turn_records = _build_group(turns_by_member=turns)
        with self.assertRaises(TrainingExportError) as caught:
            export_training_batch(spec, decision, reader)
        self.assertEqual(caught.exception.reason_code, "empty_generated_mask")

    def test_over_cap_member_refuses_export(self):
        turns = [[_turn_specs()[0]], [_turn_specs()[0]]]
        reader, spec, decision, _, _turn_records = _build_group(cap=2, turns_by_member=turns)
        with self.assertRaises(TrainingExportError) as caught:
            export_training_batch(spec, decision, reader)
        self.assertEqual(caught.exception.reason_code, "invalid_member_layout")

    def test_export_reports_machine_reasons_for_uncovered_layout_and_advantage_refusals(self):
        scenarios = (
            (
                "budget charged event",
                lambda: _build_group(budget_charged_member=0),
                "invalid_member_layout",
            ),
            (
                "zero generation before last action",
                lambda: _build_group(
                    turns_by_member=[
                        [
                            {
                                "input_ids": (10, 11, 12),
                                "generated_ids": (),
                                "termination_kind": "context_limit",
                                "stop_token_id": None,
                                "limit": "context",
                                "content": "",
                            },
                            _turn_specs()[0],
                        ],
                        _turn_specs(),
                    ]
                ),
                "invalid_member_layout",
            ),
            (
                "broken token prefix",
                lambda: _build_group(
                    turns_by_member=[
                        [
                            _turn_specs()[0],
                            {
                                **_turn_specs()[1],
                                "input_ids": (99, 100, 101, 102, 103),
                            },
                        ],
                        _turn_specs(),
                    ]
                ),
                "invalid_member_layout",
            ),
        )
        for label, build, expected_reason in scenarios:
            with self.subTest(scenario=label):
                reader, spec, decision, _, _ = build()
                with self.assertRaises(TrainingExportError) as caught:
                    export_training_batch(spec, decision, reader)
                self.assertEqual(caught.exception.reason_code, expected_reason)

    def test_export_refuses_v1_turns_and_ready_zero_variance_groups_by_code(self):
        reader, spec, decision, _, turns = _build_group()
        native_turn = turns[0][0]
        legacy_turn = WriterTurnV1(
            action_id=native_turn.action_id,
            context_revision_ref=native_turn.context_revision_ref,
            raw_output_ref=None,
            usage={},
            adapter_trace=None,
            message=native_turn.message,
        )
        reader.public[native_turn.identity()] = legacy_turn.to_wire()
        with self.assertRaises(TrainingExportError) as caught:
            export_training_batch(spec, decision, reader)
        self.assertEqual(caught.exception.reason_code, "invalid_member_layout")

        reader, spec, decision, _, _ = _build_group(tie=True)
        decision = replace(decision, status="ready", reason="group_relative")
        with self.assertRaises(TrainingExportError) as caught:
            export_training_batch(spec, decision, reader)
        self.assertEqual(caught.exception.reason_code, "advantage_status_mismatch")

    def test_export_refuses_segment_credit_span_mismatch_by_code(self):
        reader, spec, decision, _, _ = _build_group()
        credit = GroupSegmentCreditV1.from_dict(reader.artifact(decision.segment_credit_refs[0]))
        wrong_credit_ref = reader.add(replace(credit, completion_start=1).to_wire())
        decision = replace(
            decision,
            segment_credit_refs=(wrong_credit_ref, *decision.segment_credit_refs[1:]),
        )
        with self.assertRaises(TrainingExportError) as caught:
            export_training_batch(spec, decision, reader)
        self.assertEqual(caught.exception.reason_code, "segment_credit_mismatch")

    def test_training_batch_record_checks_reject_duplicate_members_bad_spans_and_audit_overlap(
        self,
    ):
        reader, spec, decision, _, _ = _build_group()
        batch = export_training_batch(spec, decision, reader).batch.to_wire()

        duplicate = deepcopy(batch)
        duplicate["members"][1]["member_id"] = duplicate["members"][0]["member_id"]
        with self.assertRaises(GroupError):
            TrainingBatchV1.from_dict(duplicate)

        bad_span = deepcopy(batch)
        bad_span["members"][0]["turn_spans"][0]["completion_end"] = 0
        with self.assertRaises(GroupError):
            TrainingBatchV1.from_dict(bad_span)

        audit_overlap = deepcopy(batch)
        audit_overlap["members"][0]["trailing_context_limit_turn_ref"] = audit_overlap["members"][
            0
        ]["turn_spans"][0]["turn_ref"]
        with self.assertRaises(GroupError):
            TrainingBatchV1.from_dict(audit_overlap)

    def test_group_segment_credit_omits_none_token_spans(self):
        legacy = GroupSegmentCreditV1(
            group_id="a" * 64,
            member_id="grp-member-00",
            action_id="writer:action:0",
            action_ref="b" * 64,
            message_ref="c" * 64,
            trace_ref="d" * 64,
            original_context_ref="e" * 64,
            original_context_content_hash="f" * 64,
            advantage_ref="0" * 64,
            segment_kind="assistant_ending",
        )
        self.assertNotIn("completion_start", legacy.to_wire())
        self.assertNotIn("completion_end", legacy.to_wire())
        with_spans = replace(legacy, completion_start=2, completion_end=5)
        self.assertEqual(with_spans.to_wire()["completion_start"], 2)
        self.assertEqual(with_spans.to_wire()["completion_end"], 5)
        with self.assertRaises(GroupError):
            replace(legacy, completion_start=5, completion_end=2)
        with self.assertRaises(GroupError):
            replace(legacy, completion_start=2, completion_end=2)


if __name__ == "__main__":
    unittest.main()
