"""No generic reducer or downgrade route exists on the admitted transition wire."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.task_graph_rollout_fixtures import build_rollout_fixture, make_gatherers
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_records import WriterTurnV1


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


def _valid_writer_candidate(root: Path):
    donor = build_rollout_fixture(root / "donor")
    target = build_rollout_fixture(root / "target")
    parent_id = donor.runtime.checkpoint_id
    view = donor.env.verify(donor.runtime)
    port = donor.env.port_input(view, next_step(view))
    turn: WriterTurnV1 = make_gatherers(donor).sampler.turn(port)
    result = donor.env.commit(donor.runtime, turn)
    event = donor.store.load_event(result.event_id)
    _copy_immutables(donor.store.root, target.store.root)
    return target, parent_id, event, result.runtime.state


class DowngradeForgeryTests(unittest.TestCase):
    def test_unsupported_generic_input_kind_has_no_commit_route(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_rollout_fixture(Path(directory) / "store")
            before = fixture.store.read_head(fixture.lineage_id)
            with self.assertRaises(ProjectionError) as rejected:
                fixture.env.commit(fixture.runtime, {"artifact_type": "Phase2RecordedEffectV1"})
            self.assertIn("input.record_type", str(rejected.exception))
            self.assertEqual(fixture.store.read_head(fixture.lineage_id), before)

    def test_retyped_admitted_event_rejects_at_event_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            target, parent_id, event, state = _valid_writer_candidate(Path(directory))
            # `seed_attached` is not in EventV1's wire vocabulary. `context_changed`
            # is a valid event header but still has no route for a WriterTurnV1 input.
            forged = replace(event, id=None, kind="context_changed")
            forged_state = replace(state, history={**state.history, "head": forged.id})
            with self.assertRaises(ProjectionError) as rejected:
                target.store.publish(
                    target.lineage_id,
                    None,
                    (forged,),
                    forged_state,
                    parent_checkpoint=parent_id,
                )
            self.assertIn("event.kind", str(rejected.exception))
            self.assertIsNone(target.store.read_head(target.lineage_id))

    def test_invalid_queue_patch_rejects_as_derived_state_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            target, parent_id, event, state = _valid_writer_candidate(Path(directory))
            continuation = state.to_dict()["continuation"]
            continuation["tool_queue"] = []
            forged_state = replace(state, continuation=continuation)
            with self.assertRaises(ProjectionError) as rejected:
                target.store.publish(
                    target.lineage_id,
                    None,
                    (event,),
                    forged_state,
                    parent_checkpoint=parent_id,
                )
            self.assertIn("state.continuation.tool_queue[0]", str(rejected.exception))
            self.assertIsNone(target.store.read_head(target.lineage_id))


if __name__ == "__main__":
    unittest.main()
