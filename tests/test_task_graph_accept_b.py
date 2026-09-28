"""Valid-hash disk forgeries and byte damage for transition-seam category b."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from tests.task_graph_rollout_fixtures import build_rollout_fixture, make_gatherers
from writing_agent.task_graph import (
    CheckpointV1,
    CommitV1,
    canonical_bytes,
    domain_hash,
)
from writing_agent.task_graph_environment import RolloutEnvironment
from writing_agent.task_graph_errors import CorruptRecordError, ProjectionError
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_records import WriterTurnV1
from writing_agent.task_graph_store import TaskGraphStore


def _writer_turn(fixture) -> WriterTurnV1:
    port = fixture.env.step_input(fixture.runtime)[2]
    return make_gatherers(fixture).sampler.turn(port)


def _copy_immutables(source: Path, destination: Path) -> None:
    for name in (
        "artifacts",
        "bytes",
        "checkpoints",
        "commits",
        "context_content",
        "context_revisions",
        "events",
        "instances",
        "private",
    ):
        source_dir = source / name
        if not source_dir.exists():
            continue
        target_dir = destination / name
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, target_dir / item.name)


def _set_budget_counter(target, state, counter: str) -> Any:
    budget = dict(target.store.get_artifact(state.budgets_ref))
    consumed = dict(budget["consumed"])
    consumed[counter] = consumed.get(counter, 0) + 1
    budget["consumed"] = consumed
    return replace(state, budgets_ref=target.store.put_artifact(budget))


def _forged_commit(target, parent_id, event, state, *, event_changes=None, lineage_id=None):
    if event_changes:
        event = replace(event, id=None, **event_changes)
        state = replace(state, history={**state.history, "head": event.id})
    checkpoint = CheckpointV1(
        parents=(parent_id,), state=state, event_head=event.id, artifact_refs=()
    )
    commit = CommitV1(parent_commit=None, events=(event.id,), checkpoint=checkpoint.identity())
    store = target.store
    # This is the attack surface: valid canonical records and identities are written
    # directly, and the mutable authority ref is redirected to the forged commit.
    store._record_path("events", event.identity()).write_bytes(canonical_bytes(event))
    store._record_path("checkpoints", checkpoint.identity()).write_bytes(
        canonical_bytes(checkpoint)
    )
    store._record_path("commits", commit.identity()).write_bytes(canonical_bytes(commit))
    store._ref_path(target.lineage_id if lineage_id is None else lineage_id).write_bytes(
        canonical_bytes({"head_commit": commit.identity()})
    )
    return commit.identity()


def _rewrite_payload(target, body) -> str:
    identity = domain_hash("payload", body)
    envelope = {"schema": 1, "domain": "payload", "encoding": "json", "body": body}
    target.store._artifact_path(identity, False).write_bytes(canonical_bytes(envelope))
    return identity


def _changed_outcome(fixture, state):
    outcome = dict(fixture.store.get_artifact(state.outcome_ref))
    outcome["task_status"] = next(
        value
        for value in ("complete", "accepted_partial", "incomplete", "unknown")
        if value != outcome["task_status"]
    )
    return replace(state, outcome_ref=fixture.store.put_artifact(outcome))


class PersistedForgeTests(unittest.TestCase):
    def _case(self, root: Path, name: str):
        donor = build_rollout_fixture(root / f"donor-{name}")
        target = build_rollout_fixture(root / f"target-{name}")
        parent_id = donor.runtime.checkpoint_id
        result = donor.env.commit(donor.runtime, _writer_turn(donor))
        event = donor.store.load_event(result.event_id)
        _copy_immutables(donor.store.root, target.store.root)
        state = result.runtime.state

        if name in {"writer_counter", "author_counter", "context_operations", "context_storage"}:
            counter = {
                "writer_counter": "writer_turns",
                "author_counter": "author_calls",
                "context_operations": "context_operations",
                "context_storage": "context_storage_bytes",
            }[name]
            state = _set_budget_counter(target, state, counter)
            path = "state.budgets_ref"
            event_changes = None
        elif name == "undeclared_decision":
            state = replace(state, decisions_ref=target.store.put_artifact({"forged": "decision"}))
            path = "state.decisions_ref"
            event_changes = None
        elif name in {"private_context_message", "private_context_message_from_ask"}:
            # A context revision is now derived from its source event, not included in
            # the input record. Substituting the earlier valid revision is the wire-level
            # forgery corresponding to the old private-message rewrite.
            state = replace(state, context_ref=target.runtime.state.context_ref)
            path = "state.context_ref"
            event_changes = None
        elif name in {"eligibility", "false_status"}:
            outcome = dict(target.store.get_artifact(state.outcome_ref))
            field = "training_eligibility" if name == "eligibility" else "task_status"
            options = (
                ("pending", "eligible", "ineligible")
                if field == "training_eligibility"
                else ("complete", "accepted_partial", "incomplete", "unknown")
            )
            outcome[field] = next(value for value in options if value != outcome[field])
            state = replace(state, outcome_ref=target.store.put_artifact(outcome))
            path = "state.outcome_ref"
            event_changes = None
        else:
            event_changes = {
                "unsupported_kind": {"kind": "context_changed"},
                "relabelled_kind": {"kind": "tool_result"},
                "check_as_budget": {"kind": "check_recorded"},
            }[name]
            path = "event.kind"

        forged_head = _forged_commit(target, parent_id, event, state, event_changes=event_changes)
        return target, forged_head, path

    def test_integrity_forgery_twelve_valid_hash_commits_reject_on_cold_open(self):
        cases = (
            ("writer_counter", "state.budgets_ref"),
            ("author_counter", "state.budgets_ref"),
            ("context_operations", "state.budgets_ref"),
            ("context_storage", "state.budgets_ref"),
            ("undeclared_decision", "state.decisions_ref"),
            ("private_context_message", "state.context_ref"),
            ("private_context_message_from_ask", "state.context_ref"),
            ("eligibility", "state.outcome_ref"),
            ("false_status", "state.outcome_ref"),
            ("unsupported_kind", "event.kind"),
            ("relabelled_kind", "event.kind"),
            ("check_as_budget", "event.kind"),
        )
        for name, expected_path in cases:
            with self.subTest(forgery=name), tempfile.TemporaryDirectory() as directory:
                target, forged_head, expected_path = self._case(Path(directory), name)
                before = target.store.read_head(target.lineage_id)
                self.assertEqual(before, forged_head)

                cold_gate = LineageGate()
                cold_store = TaskGraphStore(target.store.root, verifier=cold_gate)
                cold_env = RolloutEnvironment(
                    cold_store,
                    target.entry.graph,
                    None,
                    cold_gate,
                    target.entry.graph.policy,
                )
                with self.assertRaises(ProjectionError) as rejected:
                    cold_env.open_head(target.lineage_id)
                self.assertIn(expected_path, str(rejected.exception))
                self.assertEqual(target.store.read_head(target.lineage_id), before)

    def test_byte_tamper_after_warm_view_is_corrupt_and_does_not_move_head(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_rollout_fixture(Path(directory) / "store")
            result = fixture.env.commit(fixture.runtime, _writer_turn(fixture))
            checkpoint = result.runtime.checkpoint_id
            self.assertEqual(fixture.env.verify(result.runtime).checkpoint_id, checkpoint)
            head_path = fixture.store._ref_path(fixture.lineage_id)
            head_before = head_path.read_bytes()
            budget_path = fixture.store._artifact_path(result.runtime.state.budgets_ref, False)
            damaged = bytearray(budget_path.read_bytes())
            marker = b'"writer_turns":'
            index = damaged.index(marker) + len(marker)
            self.assertIn(damaged[index], b"0123456789")
            damaged[index] = ord("9") if damaged[index] != ord("9") else ord("8")
            budget_path.write_bytes(damaged)

            with self.assertRaises(CorruptRecordError):
                fixture.env.open_head(fixture.lineage_id)
            self.assertEqual(head_path.read_bytes(), head_before)

    def test_root_checkpoint_with_supplemental_refs_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_rollout_fixture(Path(directory) / "store")
            root = fixture.store.load_checkpoint(fixture.runtime.checkpoint_id)
            forged = CheckpointV1(
                state=root.state,
                event_head=root.event_head,
                artifact_refs=(root.state.versions_ref,),
            )
            fixture.store._record_path("checkpoints", forged.identity()).write_bytes(
                canonical_bytes(forged)
            )

            with self.assertRaises(ProjectionError) as rejected:
                fixture.env.open(forged.identity())
            self.assertIn("checkpoint.artifact_refs", str(rejected.exception))
            self.assertIsNone(fixture.store.read_head(fixture.lineage_id))

    def test_forged_parentless_entry_state_fails_the_root_fixed_point(self):
        mutations = (
            (
                "pre_phase",
                "state.position.phase",
                lambda fixture, state: replace(
                    state,
                    position={**state.position, "phase": "terminal"},
                ),
            ),
            (
                "pre_status",
                "state.outcome_ref",
                lambda fixture, state: _changed_outcome(fixture, state),
            ),
        )
        for case, path, mutate in mutations:
            with self.subTest(forgery=case), tempfile.TemporaryDirectory() as directory:
                fixture = build_rollout_fixture(Path(directory) / "store")
                state = mutate(fixture, fixture.runtime.state)
                checkpoint = CheckpointV1(
                    state=state,
                    event_head=state.history["head"],
                )
                fixture.store._record_path("checkpoints", checkpoint.identity()).write_bytes(
                    canonical_bytes(checkpoint)
                )
                head_path = fixture.store._ref_path(fixture.lineage_id)
                self.assertFalse(head_path.exists())
                with self.assertRaises(ProjectionError) as rejected:
                    fixture.env.open(checkpoint.identity())
                self.assertIn(path, str(rejected.exception))
                self.assertFalse(head_path.exists())

    def test_relineaged_writer_event_is_not_a_member_start(self):
        with tempfile.TemporaryDirectory() as directory:
            donor = build_rollout_fixture(Path(directory) / "donor")
            target = build_rollout_fixture(Path(directory) / "target")
            parent_id = donor.runtime.checkpoint_id
            result = donor.env.commit(donor.runtime, _writer_turn(donor))
            event = donor.store.load_event(result.event_id)
            _copy_immutables(donor.store.root, target.store.root)
            lineage_id = "forged-child"
            event = replace(event, id=None, lineage_id=lineage_id)
            state = replace(
                result.runtime.state,
                position={**result.runtime.state.position, "lineage_id": lineage_id},
                history={**result.runtime.state.history, "head": event.id},
            )
            forged_head = _forged_commit(target, parent_id, event, state, lineage_id=lineage_id)
            head_path = target.store._ref_path(lineage_id)
            head_before = head_path.read_bytes()
            cold_gate = LineageGate()
            cold_store = TaskGraphStore(target.store.root, verifier=cold_gate)
            cold_env = RolloutEnvironment(
                cold_store,
                target.entry.graph,
                None,
                cold_gate,
                target.entry.graph.policy,
            )
            self.assertEqual(cold_store.read_head(lineage_id), forged_head)
            with self.assertRaises(ProjectionError) as rejected:
                cold_env.open_head(lineage_id)
            self.assertIn("event.lineage_id", str(rejected.exception))
            self.assertEqual(head_path.read_bytes(), head_before)

    def _assert_hash_correct_payload_forgery_rejected(self, case, expected_path):
        with tempfile.TemporaryDirectory() as directory:
            donor = build_rollout_fixture(Path(directory) / "donor")
            target = build_rollout_fixture(Path(directory) / "target")
            parent_id = donor.runtime.checkpoint_id
            result = donor.env.commit(donor.runtime, _writer_turn(donor))
            event = donor.store.load_event(result.event_id)
            _copy_immutables(donor.store.root, target.store.root)
            payload = dict(donor.store.get_artifact(event.payload_ref))
            if case == "extra_key":
                payload["newly_forged_field"] = "valid-hash"
            else:
                payload["record_type"] = "WriterToolResultV2"
            forged_payload_ref = _rewrite_payload(target, payload)
            forged_event = replace(event, id=None, payload_ref=forged_payload_ref)
            forged_state = replace(
                result.runtime.state,
                history={**result.runtime.state.history, "head": forged_event.id},
            )
            _forged_commit(target, parent_id, forged_event, forged_state)
            head_path = target.store._ref_path(target.lineage_id)
            head_before = head_path.read_bytes()

            cold_gate = LineageGate()
            cold_store = TaskGraphStore(target.store.root, verifier=cold_gate)
            cold_env = RolloutEnvironment(
                cold_store,
                target.entry.graph,
                None,
                cold_gate,
                target.entry.graph.policy,
            )
            with self.assertRaises(ProjectionError) as rejected:
                cold_env.open_head(target.lineage_id)
            self.assertIs(type(rejected.exception), ProjectionError)
            self.assertIn(expected_path, str(rejected.exception))
            self.assertEqual(head_path.read_bytes(), head_before)

    def test_hash_correct_payload_with_extra_field_is_projection_rejection(self):
        self._assert_hash_correct_payload_forgery_rejected(
            "extra_key", "event.payload_ref.newly_forged_field"
        )

    def test_hash_correct_payload_with_unknown_record_type_is_projection_rejection(self):
        self._assert_hash_correct_payload_forgery_rejected(
            "unknown_record_type", "event.payload_ref.record_type"
        )


if __name__ == "__main__":
    unittest.main()
