"""Structural eligibility decisions for sampled native V2 turns."""

from __future__ import annotations

import unittest
from dataclasses import replace

from tests.test_task_graph_v2_writer import _native_view, _turn
from writing_agent.task_graph import EventV1
from writing_agent.task_graph_calls import intake_message
from writing_agent.task_graph_eligibility import EligibilityDecisionV1, decide_eligibility
from writing_agent.task_graph_record_contracts import ContextPolicyV1
from writing_agent.task_graph_records import (
    ContextOperationInputV1,
    RuntimeManifestV2,
    WriterTurnV1,
)
from writing_agent.task_graph_transition import SampleRef


class EligibilityDecisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture, self.view, self.manifest = _native_view()
        self.reader = self.fixture.reader

    def _sampled(self, turn=None, *, outcome="action"):
        turn = _turn(self.view, self.reader) if turn is None else turn
        turn_ref = turn.identity()
        self.reader.public[turn_ref] = turn.to_wire()
        sample = SampleRef(turn.action_id, "e" * 64, turn_ref, outcome)
        view = replace(
            self.view,
            samples=(sample,),
            outcome=replace(self.view.outcome, execution_status="valid"),
        )
        return view, turn

    def _assert_reason(self, view, reason):
        decision = decide_eligibility(view, self.reader)
        self.assertEqual(decision.status, "ineligible")
        self.assertEqual(decision.reason, reason)

    def test_v1_writer_turn_is_unavailable(self):
        turn = WriterTurnV1(
            action_id="v1-action",
            context_revision_ref=self.view.context.revision_ref,
            raw_output_ref=None,
            usage={},
            adapter_trace=None,
            message=intake_message({"content": "scripted output", "tool_calls": []}),
        )
        view, _ = self._sampled(turn)

        self._assert_reason(view, "native_action_trace_unavailable")

    def test_missing_manifest_capability_is_checked_before_group_mode(self):
        turn = _turn(self.view, self.reader)
        manifest = RuntimeManifestV2.from_dict(self.reader.artifact(self.manifest.identity()))
        ports = tuple(
            replace(
                port,
                configuration={
                    **dict(port.configuration),
                    "capabilities": tuple(
                        capability
                        for capability in port.configuration.get("capabilities", ())
                        if capability != "sampled_logprobs"
                    ),
                },
            )
            if port.role == "sampling"
            else port
            for port in manifest.ports
        )
        manifest = replace(manifest, ports=ports)
        self.reader.public[manifest.identity()] = manifest.to_wire()
        turn = replace(
            turn,
            sampling_pins={**turn.sampling_pins, "manifest_ref": manifest.identity()},
        )
        view, _ = self._sampled(turn)
        view = replace(view, group=None)

        self._assert_reason(view, "manifest_capability_missing")

    def test_missing_native_group_gets_group_reason(self):
        view, _ = self._sampled()
        view = replace(view, group=None)

        self._assert_reason(view, "group_training_mode_absent")

    def test_context_reset_gets_multi_segment_reason(self):
        view, _ = self._sampled()
        lineage_id = view.state.position["lineage_id"]
        start = EventV1(
            previous=None,
            seq=1,
            lineage_id=lineage_id,
            rollout_id=lineage_id,
            node_visit_id=view.state.position["visit_id"],
            kind="rollout_started",
            actor="environment",
            audience=("controller",),
            payload_ref="d" * 64,
            versions_ref=view.state.versions_ref,
            provenance_ref=view.state.provenance_ref,
        )
        policy = ContextPolicyV1(operation="drop")
        self.reader.public[policy.identity()] = policy.to_wire()
        operation = ContextOperationInputV1(policy_ref=policy.identity())
        self.reader.public[operation.identity()] = operation.to_wire()
        context_changed = EventV1(
            previous=start.identity(),
            seq=2,
            lineage_id=lineage_id,
            rollout_id=lineage_id,
            node_visit_id=view.state.position["visit_id"],
            kind="context_changed",
            actor="environment",
            audience=("controller",),
            payload_ref=operation.identity(),
            versions_ref=view.state.versions_ref,
            provenance_ref=view.state.provenance_ref,
        )
        self.reader.events[start.identity()] = start.to_dict()
        self.reader.events[context_changed.identity()] = context_changed.to_dict()
        view = replace(view, head_event_id=context_changed.identity())

        self._assert_reason(view, "multi_segment_context")

    def test_sampled_reasoning_channel_is_retained_and_rejected(self):
        message = intake_message(
            {"content": "answer", "reasoning_content": "private chain of thought"}
        )
        self.assertEqual(message.to_wire()["reasoning_content"], "private chain of thought")
        self.assertFalse(
            {"reasoning", "thinking", "reasoning_content"}
            & set(intake_message({"content": "answer"}).to_wire())
        )
        view, _ = self._sampled(replace(_turn(self.view, self.reader), message=message))

        self._assert_reason(view, "reasoning_content_present")

    def test_zero_generated_tokens_get_no_sampled_actions_reason(self):
        turn = _turn(
            self.view,
            self.reader,
            generated_ids=(),
            termination_kind="context_limit",
            stop_token_id=None,
            limit="context",
            content="",
        )
        view, _ = self._sampled(turn, outcome="budget_stop")

        self._assert_reason(view, "no_sampled_actions")

    def test_invalid_execution_gets_defensive_reason(self):
        view, _ = self._sampled()
        view = replace(view, outcome=replace(view.outcome, execution_status="simulator_error"))

        self._assert_reason(view, "execution_not_valid")

    def test_native_v2_group_records_structural_eligibility(self):
        view, _ = self._sampled()

        decision = decide_eligibility(view, self.reader)

        self.assertEqual(decision.status, "structurally_eligible")
        self.assertEqual(decision.reason, "native_evidence_structural")
        self.assertEqual(
            decision.training_wire("f" * 64),
            {
                "record_type": "TrainingEligibilityV1",
                "schema": 1,
                "terminal_outcome_ref": "f" * 64,
                "status": "structurally_eligible",
                "reason": "native_evidence_structural",
            },
        )

    def test_reserved_eligible_status_cannot_be_constructed(self):
        with self.assertRaises(ValueError):
            EligibilityDecisionV1("eligible", "native_evidence_structural")


if __name__ == "__main__":
    unittest.main()
