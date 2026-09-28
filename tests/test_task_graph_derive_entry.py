"""Entry derivation evidence."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.task_graph_fixtures import make_entry_fixture
from writing_agent.task_graph import (
    ContextContentV1,
    ContextRevisionV1,
    MessageV1,
    canonical_bytes,
    domain_hash,
    load_canonical_json,
)
from writing_agent.task_graph_derive_entry import SYSTEM_PROMPT, derive_entry, params_of
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_gate import StoreArtifactReader as StoreReader
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_transition import first_difference


class DeriveEntryTests(unittest.TestCase):
    def test_entry_system_prompt_identity_is_pinned(self) -> None:
        self.assertEqual(
            domain_hash("payload", SYSTEM_PROMPT),
            "8a826cfb9db10547562491f9709ff78dd5cc501967721b07c812e5ae5b93936d",
        )

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


if __name__ == "__main__":
    unittest.main()
