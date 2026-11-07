from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from writing_agent.task_graph import canonical_bytes
from writing_agent.task_graph_group_index import groups_by_sequence
from writing_agent.task_graph_record_contracts import GroupError


class GroupIndexTests(unittest.TestCase):
    def test_step_reservations_and_exact_lock_files_share_one_sequence_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".step-000003.lock").touch()
            (root / "step-000003").write_bytes(
                canonical_bytes({"schema": 1, "step": 3, "task_id": "task_3", "status": "sealing"})
            )

            self.assertEqual(groups_by_sequence(root), {3: None})

    def test_unknown_non_directory_entries_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stray.txt").write_text("evidence")

            with self.assertRaises(GroupError):
                groups_by_sequence(root)

    def test_malformed_reservation_and_directory_named_as_reservation_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "step-000002").write_text("{}")

            with self.assertRaises(GroupError):
                groups_by_sequence(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "step-000002").mkdir()

            with self.assertRaises(GroupError):
                groups_by_sequence(root)

    def test_symlinked_root_or_reservation_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "groups"
            link.symlink_to(target, target_is_directory=True)

            with self.assertRaises(GroupError):
                groups_by_sequence(link)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "reservation-target"
            target.write_bytes(
                canonical_bytes({"schema": 1, "step": 2, "task_id": "task_2", "status": "sealing"})
            )
            (root / "step-000002").symlink_to(target)

            with self.assertRaises(GroupError):
                groups_by_sequence(root)


if __name__ == "__main__":
    unittest.main()
