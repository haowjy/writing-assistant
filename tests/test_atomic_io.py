from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from writing_agent.atomic_io import atomic_write_bytes, atomic_write_json
from writing_agent.task_graph import canonical_bytes


class AtomicIoTests(unittest.TestCase):
    def test_canonical_and_readable_json_use_the_shared_atomic_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_path = root / "nested" / "canonical.json"
            atomic_write_json(
                canonical_path,
                {"z": 1, "a": "snow"},
                canonical=True,
                create_parent=True,
            )
            readable_path = root / "readable.json"
            atomic_write_json(readable_path, {"z": 1, "a": "snow"})

            self.assertEqual(canonical_path.read_bytes(), canonical_bytes({"z": 1, "a": "snow"}))
            self.assertEqual(json.loads(readable_path.read_text()), {"z": 1, "a": "snow"})
            self.assertTrue(readable_path.read_text().endswith("\n"))
            self.assertEqual(readable_path.stat().st_mode & 0o777, 0o600)

    def test_exclusive_atomic_bytes_refuse_to_replace_existing_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            self.assertTrue(atomic_write_bytes(path, b"first", replace=False))
            self.assertFalse(atomic_write_bytes(path, b"second", replace=False))

            self.assertEqual(path.read_bytes(), b"first")

    def test_nonfinite_pretty_json_is_refused_before_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_bytes(b"original")

            with self.assertRaises(ValueError):
                atomic_write_json(path, {"value": float("nan")})

            self.assertEqual(path.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
