"""Direct regression tests for the fail-closed rules restored after S7.3."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.task_graph_rollout_fixtures import build_rollout_fixture
from writing_agent.task_graph_admission import (
    AdmissionError,
    MappingArtifactResolver,
    admit_graph,
)
from writing_agent.task_graph_compaction import charge_budget, require_quiescent
from writing_agent.task_graph_contracts import InteractionPolicyV1, RequirementUpdateV1
from writing_agent.task_graph_derive_entry import derive_entry
from writing_agent.task_graph_record_contracts import CompactionError, ContextPolicyV1
from writing_agent.task_graph_scripted import (
    ScriptCoverageError,
    resolve_script_reply,
    validate_ask_semantics,
)


class AdmissionGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        fixture = build_rollout_fixture(Path(self.temporary.name) / "feedback", mode="feedback")
        self.graph = fixture.entry.graph
        self.reader = fixture.entry.reader
        self.node = self.graph.node(fixture.entry.node_id)

    def _admit_with_contract(self, contract) -> None:
        contract_ref = contract.identity()
        self.reader.public[contract_ref] = contract.to_dict()
        spec = replace(self.node.spec, entry_contract=contract_ref)
        instance = replace(self.graph.instance, nodes=(spec,))
        admit_graph(
            instance,
            MappingArtifactResolver(self.reader.public, self.reader.private),
            policy=self.graph.policy,
        )

    def test_feedback_supersedes_only_an_initial_requirement(self) -> None:
        update = RequirementUpdateV1(
            id="not-admitted",
            supersedes="not-an-initial-requirement",
            replacement="An updated requirement.",
        )
        self.reader.private[update.identity()] = update.to_dict()
        feedback = tuple(
            {**dict(item), "requirement_update_ref": update.identity()}
            for item in self.node.script.feedback
        )
        script = replace(self.node.script, feedback=feedback)
        self.reader.private[script.identity()] = script.to_dict()
        interaction = replace(self.node.contract.interaction_contract, script_ref=script.identity())

        with self.assertRaises(AdmissionError) as rejected:
            self._admit_with_contract(replace(self.node.contract, interaction=interaction))

        self.assertEqual(rejected.exception.code, "requirement_update")

    def test_requirement_packet_and_version_disagreement_precedes_feedback_updates(self) -> None:
        packet = replace(self.node.author_packet, requirements={"baseline": "conflicting text"})
        self.reader.private[packet.identity()] = packet.to_dict()
        interaction = replace(
            self.node.contract.interaction_contract,
            author_packet_ref=packet.identity(),
        )

        with self.assertRaises(AdmissionError) as rejected:
            self._admit_with_contract(replace(self.node.contract, interaction=interaction))

        self.assertEqual(rejected.exception.code, "requirement_update")

    def test_evaluator_packet_is_not_an_author_packet(self) -> None:
        interaction = replace(
            self.node.contract.interaction_contract,
            author_packet_ref=self.node.evaluator_packet.identity(),
        )

        with self.assertRaises(AdmissionError):
            self._admit_with_contract(replace(self.node.contract, interaction=interaction))

    def test_reward_components_require_terminal_check_evidence(self) -> None:
        check = replace(self.node.checks["nonempty"], applicability="before_feedback:feedback-1")
        terminal_check = replace(
            self.node.checks["nonempty"],
            id="terminal",
            spec={**self.node.checks["nonempty"].spec, "id": "terminal"},
        )
        self.reader.private[check.identity()] = check.to_dict()
        self.reader.private[terminal_check.identity()] = terminal_check.to_dict()
        packet = replace(
            self.node.evaluator_packet,
            check_ids=(check.id, terminal_check.id),
        )
        self.reader.private[packet.identity()] = packet.to_dict()
        completion = replace(
            self.node.contract.completion_contract,
            required_check_ids=(check.id, terminal_check.id),
            evaluation_packet_ref=packet.identity(),
        )
        contract = replace(
            self.node.contract,
            completion=completion,
            mandatory_checks=(check.identity(), terminal_check.identity()),
        )

        with self.assertRaises(AdmissionError) as rejected:
            self._admit_with_contract(contract)

        self.assertEqual(rejected.exception.code, "reward_coverage")


class ScriptedPolicyGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        fixture = build_rollout_fixture(Path(self.temporary.name) / "scripted", mode="feedback")
        self.node = fixture.entry.graph.node(fixture.entry.node_id)

    def test_ask_proposals_and_decisions_obey_the_admitted_limits(self) -> None:
        too_many_chars_policy = InteractionPolicyV1(
            public_decisions=({"id": "door", "label": "door"},),
            max_question_chars=3,
        )
        too_many_chars_node = replace(self.node, interaction_policy=too_many_chars_policy)
        cases = (
            (
                "M4 prior proposal cannot be redefined",
                self.node,
                {
                    "question": "Pick.",
                    "decision_ids": ["door"],
                    "proposals": [{"id": "old", "text": "new wording"}],
                    "option_refs": ["old"],
                },
                {"proposals": {"old": {"text": "earlier wording"}}},
            ),
            (
                "M6 undeclared decision ID",
                self.node,
                {
                    "question": "Pick.",
                    "decision_ids": ["undeclared"],
                    "proposals": [],
                    "option_refs": [],
                },
                {},
            ),
            (
                "M6b undeclared option reference",
                self.node,
                {
                    "question": "Pick.",
                    "decision_ids": ["door"],
                    "proposals": [],
                    "option_refs": ["undeclared"],
                },
                {},
            ),
            (
                "M6c question length limit",
                too_many_chars_node,
                {
                    "question": "Too long.",
                    "decision_ids": ["door"],
                    "proposals": [],
                    "option_refs": [],
                },
                {},
            ),
        )
        for mutant, node, arguments, decisions in cases:
            with self.subTest(mutant=mutant), self.assertRaises(ValueError):
                validate_ask_semantics(arguments, node, decisions)

    def test_exhausted_positional_script_selector_is_coverage_failure(self) -> None:
        script = replace(
            self.node.script,
            answers={
                "door": {
                    "mode": "declared_option_position",
                    "utterance": "Choose the first option.",
                    "value": "first",
                    "selector": 0,
                    "prerequisite_check_ids": [],
                }
            },
        )
        request = {
            "request_ref": "a" * 64,
            "request_id": "request-1",
            "action_id": "action-1",
            "decision_ids": ["door"],
            "arguments": {"proposals": [], "option_refs": []},
            "prerequisite_results": {},
        }

        with self.assertRaises(ScriptCoverageError):
            resolve_script_reply(
                script,
                request,
                {"values": {}, "proposals": {}},
                {"decisions": []},
            )


class BudgetGuardTests(unittest.TestCase):
    def test_context_operation_budget_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_rollout_fixture(Path(temporary) / "entry", mode="feedback")
            state = fixture.entry.state
            context = fixture.store.materialize_context(state.context_ref)
            policy = ContextPolicyV1("carry", max_operations=1)
            budget = fixture.store.get_artifact(state.budgets_ref)
            budget["consumed"]["context_operations"] = policy.max_operations

            with self.assertRaises(CompactionError):
                charge_budget(budget, policy, context, context, None)

    def test_entry_author_call_limit_matches_the_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_rollout_fixture(Path(temporary) / "entry", mode="feedback")
            entry = fixture.entry
            contract_limit = entry.graph.node(
                entry.node_id
            ).contract.budget_contract.max_author_calls

            derived = derive_entry(entry.graph, entry.node_id, entry.params, entry.reader)
            budgets = fixture.store.get_artifact(derived.state.budgets_ref)

            self.assertEqual(budgets["limits"]["author_calls"], contract_limit)

    def test_quiescence_rejects_a_queued_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_rollout_fixture(Path(temporary) / "entry", mode="feedback")
            state = fixture.entry.state
            queued = replace(
                state,
                continuation={
                    **state.continuation,
                    "tool_queue": (
                        {
                            "call_id": "queued:tool:0",
                            "name": "read_file",
                            "arguments": {"path": "draft.txt"},
                            "rejection": None,
                        },
                    ),
                    "next_call": 0,
                },
            )

            with self.assertRaises(CompactionError):
                require_quiescent(queued)


if __name__ == "__main__":
    unittest.main()
