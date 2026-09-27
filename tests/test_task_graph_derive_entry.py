"""Entry derivation and legacy-fixture comparison evidence."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from dataclasses import replace

from tests.task_graph_fixtures import make_entry_fixture
from writing_agent.task_graph import MessageV1, canonical_bytes
from writing_agent.task_graph_derive_entry import (
    derive_entry,
    derive_entry_artifacts,
    params_of,
)
from writing_agent.task_graph_records import (
    AdmissionPolicyV1,
    ContextContentV1,
    ContextRevisionV1,
)
from writing_agent.task_graph_transition import LineageMode, first_difference


class StoreReader:
    def __init__(self, store) -> None:
        self.store = store

    def artifact(self, ref, *, domain="payload", private=False):
        return self.store.get_artifact(ref, expected_domain=domain, private=private)

    def bytes_artifact(self, ref):
        return self.store.get_bytes(ref)

    def checkpoint(self, ref):
        return self.store.load_checkpoint(ref)


def _difference_paths(left, right, path="state"):
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_difference_paths(left[key], right[key], child))
        return paths
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        paths = []
        for index in range(max(len(left), len(right))):
            child = f"{path}[{index}]"
            if index >= len(left) or index >= len(right):
                paths.append(child)
            else:
                paths.extend(_difference_paths(left[index], right[index], child))
        return paths
    return [] if type(left) is type(right) and left == right else [path]


class DeriveEntryTests(unittest.TestCase):
    def test_entry_is_deterministic_and_params_of_midrun_is_not_a_fixed_point(self) -> None:
        fixture = make_entry_fixture()
        derived = derive_entry(fixture.graph, fixture.node_id, fixture.params, fixture.reader)
        self.assertEqual(canonical_bytes(derived), canonical_bytes(fixture.state))
        self.assertIsNone(first_difference(derived, fixture.state))

        params = params_of(fixture.state)
        self.assertIsNone(params.rendering)
        self.assertEqual(
            canonical_bytes(derive_entry(fixture.graph, fixture.node_id, params, fixture.reader)),
            canonical_bytes(fixture.state),
        )

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
        fixture.reader.context_content[appended.identity()] = appended.to_wire()
        fixture.reader.context_revisions[appended_revision.identity()] = appended_revision.to_wire()
        midrun = replace(
            fixture.state,
            context_ref=appended_revision.identity(),
            position={**fixture.state.position, "phase": "checking"},
            history={**fixture.state.history, "head": "a" * 64, "seq": 1},
        )
        replayed_entry = derive_entry(
            fixture.graph, fixture.node_id, params_of(midrun), fixture.reader
        )
        self.assertNotEqual(canonical_bytes(replayed_entry), canonical_bytes(midrun))
        self.assertEqual(first_difference(replayed_entry, midrun), "state.context_ref")

    def test_entry_artifacts_use_the_expected_domains_and_v1_records(self) -> None:
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
        revision = next(item for item in fixture.artifacts if item.kind == "context")
        content = next(item for item in fixture.artifacts if item.kind == "context_content")
        self.assertEqual(revision.ref, state.context_ref)
        self.assertEqual(revision.value["content_ref"], content.ref)
        self.assertEqual(
            fixture.reader.artifact(state.requirements_ref, private=True)["record_type"],
            "RequirementLedgerV1",
        )
        versions = fixture.reader.artifact(state.versions_ref)
        self.assertEqual(versions["transition_semantics"], "task-graph-derive-v1")
        outcome = fixture.reader.artifact(state.outcome_ref)
        self.assertEqual(outcome["record_type"], "OutcomeV1")
        external = fixture.reader.artifact(state.external_inputs_ref)
        self.assertEqual(external["record_type"], "ExternalInputsV1")

    def test_old_hand_built_entries_expose_the_unrepresentable_wire_differences(self) -> None:
        # The legacy test fixtures use opaque placeholder ledger artifacts and the old
        # flattened context revision. S1.3 records those concrete incompatibilities rather
        # than accepting additional diffs as part of the wire-v1 allowance.
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
                "state.decisions_ref",
                "state.disclosures_ref",
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
        self.assertEqual(
            generated_scripted.requirements_ref,
            scripted.runtime.state.requirements_ref,
        )
        self.assertIn(
            "public", scripted.store.artifact_visibilities(generated_scripted.requirements_ref)
        )
        derived_private = {
            artifact.ref: artifact.kind
            for artifact in derive_entry_artifacts(
                scripted.writer.graph,
                scripted.runtime.state.position["node_id"],
                self._legacy_params(scripted),
                StoreReader(scripted.store),
            )
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

    def _assert_legacy_difference_paths(self, graph, old, node_id, fixture, expected_paths) -> None:
        generated = derive_entry(
            graph,
            node_id,
            self._legacy_params(fixture),
            StoreReader(fixture.store),
        )
        self.assertEqual(
            _difference_paths(old.to_dict(), generated.to_dict()), list(expected_paths)
        )
        self.assertEqual(first_difference(old, generated), expected_paths[0])
        self.assertIn("state.versions_ref", expected_paths)
        self.assertEqual(
            fixture.store.get_artifact(generated.versions_ref)["transition_semantics"],
            "task-graph-derive-v1",
        )
        return generated

    @staticmethod
    def _legacy_params(fixture):
        from writing_agent.task_graph_derive_entry import EntryParamsV1

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
