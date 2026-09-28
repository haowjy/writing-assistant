"""Entry derivation and legacy-fixture comparison evidence."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.task_graph_fixtures import make_entry_fixture
from writing_agent.agent import SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT
from writing_agent.task_graph import MessageV1, canonical_bytes, load_canonical_json
from writing_agent.task_graph_derive_entry import EntryParamsV1, derive_entry, params_of
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_gate import StoreArtifactReader as StoreReader
from writing_agent.task_graph_records import (
    AdmissionPolicyV1,
    ContextContentV1,
    ContextRevisionV1,
)
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_transition import LineageMode, first_difference


class DeriveEntryTests(unittest.TestCase):
    def test_entry_is_deterministic_and_params_of_midrun_is_not_a_fixed_point(self) -> None:
        fixture = make_entry_fixture()
        entry = derive_entry(fixture.graph, fixture.node_id, fixture.params, fixture.reader)
        self.assertEqual(canonical_bytes(entry.state), canonical_bytes(fixture.state))
        self.assertIsNone(first_difference(entry.state, fixture.state))

        params = params_of(fixture.state, fixture.reader)
        self.assertEqual(
            params.rendering,
            fixture.reader.context(fixture.state.context_ref).rendering,
        )
        regenerated = derive_entry(fixture.graph, fixture.node_id, params, fixture.reader)
        self.assertEqual(canonical_bytes(regenerated.state), canonical_bytes(fixture.state))

        root_revision = ContextRevisionV1.from_dict(
            fixture.reader.artifact(fixture.state.context_ref, domain="context_revision")
        )
        appended = ContextContentV1(
            parent_ref=root_revision.content_ref,
            messages=(MessageV1(role="assistant", content=("draft",), origin="action:1"),),
            tools=None,
            rendering=None,
        )
        appended_revision = ContextRevisionV1(
            content_ref=appended.identity(),
            event_head="a" * 64,
            provenance_refs=(),
        )
        fixture.reader.context_nodes[appended.identity()] = appended.to_wire()
        fixture.reader.context_revisions[appended_revision.identity()] = appended_revision.to_wire()
        midrun = replace(
            fixture.state,
            context_ref=appended_revision.identity(),
            position={**fixture.state.position, "phase": "checking"},
            history={**fixture.state.history, "head": "a" * 64, "seq": 1},
        )
        replayed_entry = derive_entry(
            fixture.graph, fixture.node_id, params_of(midrun, fixture.reader), fixture.reader
        )
        self.assertNotEqual(canonical_bytes(replayed_entry.state), canonical_bytes(midrun))
        self.assertEqual(first_difference(replayed_entry.state, midrun), "state.context_ref")

    def test_entry_artifacts_use_expected_domains_and_v1_records(self) -> None:
        fixture = make_entry_fixture()
        by_kind = {
            kind: {artifact.ref for artifact in fixture.artifacts if artifact.kind == kind}
            for kind in {artifact.kind for artifact in fixture.artifacts}
        }
        state = fixture.state
        self.assertIn(state.requirements_ref, by_kind["private"])
        self.assertTrue(
            {
                state.decisions_ref,
                state.disclosures_ref,
                state.budgets_ref,
                state.outcome_ref,
                state.external_inputs_ref,
            }
            <= by_kind["artifact"]
        )
        revision = next(item for item in fixture.artifacts if item.kind == "context_revision")
        content = next(item for item in fixture.artifacts if item.kind == "context_node")
        self.assertEqual(revision.ref, state.context_ref)
        self.assertEqual(revision.value.content_ref, content.ref)
        self.assertEqual(
            fixture.reader.artifact(state.requirements_ref, private=True)["record_type"],
            "RequirementLedgerV1",
        )
        versions = fixture.reader.artifact(state.versions_ref)
        self.assertEqual(versions["transition_semantics"], "task-graph-derive-v1")

    def test_initial_requirement_version_body_is_pinned(self) -> None:
        fixture = make_entry_fixture()
        source = {"requirements": {"canon": "Keep the moon hidden until chapter three."}}
        source_ref = fixture.reader.add(source, private=True)
        node = fixture.graph.node(fixture.node_id)
        entry_contract = replace(node.contract.entry_contract, requirement_version=source_ref)
        contract = replace(node.contract, entry=entry_contract)
        graph = replace(
            fixture.graph,
            nodes={
                **fixture.graph.nodes,
                fixture.node_id: replace(node, contract=contract),
            },
        )

        entry = derive_entry(graph, fixture.node_id, fixture.params, fixture.reader)

        self._assert_entry_payload_bodies(
            entry,
            {"canon": "Keep the moon hidden until chapter three."},
            external_source_ref="5b916b97b4b82b3ad735d65c81d26640a01ba8f2990686ff7db16f7d92149382",
        )

    def test_initial_requirements_reject_empty_keys_and_values(self) -> None:
        for requirements in ({"": "private text"}, {"canon": ""}):
            with self.subTest(requirements=requirements):
                fixture = make_entry_fixture()
                source_ref = fixture.reader.add({"requirements": requirements}, private=True)
                node = fixture.graph.node(fixture.node_id)
                contract = replace(
                    node.contract,
                    entry=replace(node.contract.entry_contract, requirement_version=source_ref),
                )
                graph = replace(
                    fixture.graph,
                    nodes={
                        **fixture.graph.nodes,
                        fixture.node_id: replace(node, contract=contract),
                    },
                )

                with self.assertRaisesRegex(
                    ValueError, "initial requirements must map nonempty strings to nonempty strings"
                ):
                    derive_entry(graph, fixture.node_id, fixture.params, fixture.reader)

    def test_entry_prompt_stays_byte_identical_to_legacy_prompt(self) -> None:
        from writing_agent.task_graph_derive_entry import SYSTEM_PROMPT

        self.assertEqual(SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT)

    def test_params_of_and_entry_fixed_point_use_real_store_context_reader(self) -> None:
        fixture = make_entry_fixture()
        with TemporaryDirectory() as directory:
            store = TaskGraphStore(Path(directory) / "store", verifier=LineageGate())
            for body in fixture.reader.public.values():
                store.put_artifact(body)
            for body in fixture.reader.private.values():
                store.put_artifact(body, private=True)
            store.persist(fixture.graph.instance)
            for artifact in fixture.artifacts:
                if artifact.kind in {"context_node", "context_revision"}:
                    store.persist(artifact.value)

            reader = StoreReader(store)
            bytes_ref = store.put_bytes_artifact(b"fixture-bytes")
            params = params_of(fixture.state, reader)
            entry = derive_entry(fixture.graph, fixture.node_id, params, reader)
            self.assertIsNone(first_difference(fixture.state, entry.state))
            self.assertEqual(reader.context(fixture.state.context_ref).rendering, params.rendering)
            self.assertEqual(reader.bytes_artifact(bytes_ref), b"fixture-bytes")
            self.assertEqual(
                reader.artifact(fixture.state.requirements_ref, private=True),
                {"record_type": "RequirementLedgerV1", "schema": 1, "active": {}, "superseded": {}},
            )
            checkpoint = store.save_checkpoint(fixture.state)
            self.assertEqual(
                reader.checkpoint(checkpoint).state.identity(), fixture.state.identity()
            )

    def test_old_hand_built_entries_expose_only_explicit_wire_differences(self) -> None:
        from tests.test_task_graph_writer import WriterFixture

        legacy = WriterFixture()
        legacy.setUp()
        self.addCleanup(legacy.doCleanups)
        old = legacy.runtime.state
        self._assert_legacy_difference_paths(
            legacy.bundle.admission(),
            old,
            old.position["node_id"],
            legacy,
            (
                "state.context_ref",
                "state.external_inputs_ref",
                "state.outcome_ref",
                "state.requirements_ref",
                "state.versions_ref",
            ),
        )

    def test_scripted_and_group_entries_have_only_their_explicit_deltas(self) -> None:
        from tests.test_task_graph_scripted import ScriptedFixture

        scripted = ScriptedFixture()
        scripted.setUp()
        self.addCleanup(scripted.doCleanups)
        scripted_node = scripted.writer.graph.node(scripted.runtime.state.position["node_id"])
        mode = LineageMode.for_node(scripted_node)
        self.assertEqual(mode.interaction, "scripted_author")
        self.assertTrue(mode.ask_semantics)
        self.assertTrue(mode.evaluation)
        self.assertEqual(mode.reward, scripted_node.reward_contract)
        generated_scripted = self._assert_legacy_difference_paths(
            scripted.writer.graph,
            scripted.runtime.state,
            scripted.runtime.state.position["node_id"],
            scripted,
            (
                "state.context_ref",
                "state.external_inputs_ref",
                "state.outcome_ref",
                "state.versions_ref",
            ),
        )
        scripted_budget = scripted.store.get_artifact(generated_scripted.budgets_ref)
        self.assertEqual(
            scripted_budget["limits"]["author_calls"],
            scripted_node.contract.budget_contract.max_author_calls,
        )
        plain = make_entry_fixture()
        plain_budget = plain.reader.artifact(plain.state.budgets_ref)
        self.assertNotIn("author_calls", plain_budget["limits"])
        self.assertEqual(
            generated_scripted.requirements_ref,
            scripted.runtime.state.requirements_ref,
        )
        self.assertIn(
            "public", scripted.store.artifact_visibilities(generated_scripted.requirements_ref)
        )
        derived_private = {
            artifact.ref: artifact.kind
            for artifact in derive_entry(
                scripted.writer.graph,
                scripted.runtime.state.position["node_id"],
                self._legacy_params(scripted),
                StoreReader(scripted.store),
            ).artifacts
        }
        self.assertEqual(derived_private[generated_scripted.requirements_ref], "private")

        from tests.test_task_graph_group import GroupCoordinatorTests

        group = GroupCoordinatorTests("test_full_contract_drift_and_start_isolation")
        group.setUp()
        self.addCleanup(group.doCleanups)
        self._assert_legacy_difference_paths(
            group.writer.graph,
            group.runtime.state,
            group.runtime.state.position["node_id"],
            group,
            (
                "state.context_ref",
                "state.external_inputs_ref",
                "state.outcome_ref",
                "state.versions_ref",
            ),
        )

    def _assert_legacy_difference_paths(self, graph, old, node_id, fixture, expected_paths):
        entry = derive_entry(
            graph,
            node_id,
            self._legacy_params(fixture),
            StoreReader(fixture.store),
        )
        generated = entry.state
        old_body, generated_body = old.to_dict(), generated.to_dict()
        for path in expected_paths:
            field = path.removeprefix("state.")
            self.assertNotEqual(old_body[field], generated_body[field], path)
            old_body[field] = generated_body[field] = None
        self.assertIsNone(first_difference(old_body, generated_body))
        self.assertEqual(first_difference(old, generated), expected_paths[0])
        node = graph.node(node_id)
        if node.author_packet is not None:
            expected_requirements = dict(node.author_packet.requirements)
        else:
            source_ref = node.contract.entry_contract.requirement_version
            expected_requirements = (
                {}
                if source_ref is None
                else StoreReader(fixture.store).artifact(source_ref, private=True)["requirements"]
            )
        self._assert_entry_payload_bodies(entry, expected_requirements)

        for artifact in entry.artifacts:
            if artifact.kind in {"context_node", "context_revision"}:
                fixture.store.persist(artifact.value)
        reader = StoreReader(fixture.store)
        derived_context = reader.context(generated.context_ref)
        legacy_context = fixture.store.load_context(old.context_ref)
        self.assertEqual(
            canonical_bytes(derived_context.messages), canonical_bytes(legacy_context.messages)
        )
        self.assertEqual(
            canonical_bytes(derived_context.tools), canonical_bytes(legacy_context.tools)
        )
        self.assertEqual(
            canonical_bytes(derived_context.rendering), canonical_bytes(legacy_context.rendering)
        )
        self.assertIn("state.versions_ref", expected_paths)
        self.assertEqual(
            fixture.store.get_artifact(generated.versions_ref)["transition_semantics"],
            "task-graph-derive-v1",
        )
        return generated

    def _assert_entry_payload_bodies(
        self,
        entry,
        active_requirements,
        *,
        external_source_ref="530b42f2360f4d0a62bcc31adc514272bb0700be7d7fcf92454939c12263b73e",
    ) -> None:
        bodies = {
            artifact.ref: artifact.value.to_wire()
            if hasattr(artifact.value, "to_wire")
            else load_canonical_json(artifact.value)
            for artifact in entry.artifacts
        }
        self.assertEqual(
            bodies[entry.state.outcome_ref],
            {
                "record_type": "OutcomeV1",
                "schema": 1,
                "task_status": "unknown",
                "execution_status": "running",
                "stop_reason": None,
                "reward_status": "pending",
                "training_eligibility": "pending",
                "candidate_checkpoint": None,
                "requirement_version": None,
                "checks": [],
                "transition_edge_id": None,
                "failed_request_ref": None,
                "reward_ref": None,
                "eligibility_ref": None,
            },
        )
        self.assertEqual(
            bodies[entry.state.requirements_ref],
            {
                "record_type": "RequirementLedgerV1",
                "schema": 1,
                "active": active_requirements,
                "superseded": {},
            },
        )
        self.assertIn(entry.state.external_inputs_ref, bodies)
        self.assertEqual(
            bodies[entry.state.external_inputs_ref],
            {
                "record_type": "ExternalInputsV1",
                "schema": 1,
                "source_refs": [external_source_ref],
            },
        )

    @staticmethod
    def _legacy_params(fixture):
        state = fixture.runtime.state
        graph = fixture.writer.graph
        admission_policy = AdmissionPolicyV1.from_admission_policy(graph.policy)
        admission_policy_ref = fixture.store.put_artifact(admission_policy.to_wire())
        versions = fixture.store.get_artifact(state.versions_ref)
        versions.update(
            transition_semantics="task-graph-derive-v1",
            admission_policy_ref=admission_policy_ref,
            tool_spec={"max_file_bytes": 128_000, "max_workspace_bytes": 4096},
            rendering=dict(fixture.runtime.context.rendering),
        )
        return EntryParamsV1(
            lineage_id=state.position["lineage_id"],
            visit_id=state.position["visit_id"],
            rendering=fixture.runtime.context.rendering,
            versions_ref=fixture.store.put_artifact(versions),
            provenance_ref=state.provenance_ref,
            rng_ref=state.rng_ref,
        )


if __name__ == "__main__":
    unittest.main()
