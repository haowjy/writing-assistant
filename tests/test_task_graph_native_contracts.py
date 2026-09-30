"""Shared native ledger and context-root boundaries."""

from __future__ import annotations

import unittest

from writing_agent.task_graph_context_roots import context_root_changed_after
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_token_ledger import decode_u32_token_ids, encode_u32_token_ids


class EventReader:
    def __init__(self, events):
        self.events = events

    def artifact(self, ref, *, domain="payload"):
        if domain != "event":
            raise AssertionError("context ancestry reads events only")
        return self.events[ref]


class NativeContractHelpersTests(unittest.TestCase):
    def test_u32_ledger_round_trips_bounds_in_little_endian(self):
        tokens = (0, 1, 0x12345678, 0xFFFFFFFF)
        encoded = encode_u32_token_ids(tokens)

        self.assertEqual(
            encoded, b"\x00\x00\x00\x00\x01\x00\x00\x00\x78\x56\x34\x12\xff\xff\xff\xff"
        )
        self.assertEqual(decode_u32_token_ids(encoded, len(tokens)), tokens)
        self.assertEqual(decode_u32_token_ids(encoded), tokens)

    def test_u32_ledger_refuses_overflow_and_wrong_byte_count(self):
        for token in (-1, 0x1_0000_0000):
            with self.subTest(token=token), self.assertRaises(OverflowError):
                encode_u32_token_ids((token,))

        with self.assertRaises(ValueError):
            decode_u32_token_ids(b"\x01\x00\x00\x00", 2)
        with self.assertRaises(ValueError):
            decode_u32_token_ids(b"", -1)
        with self.assertRaises(ValueError):
            decode_u32_token_ids(b"\x01")

    def test_context_change_is_scoped_to_the_sample_ancestry(self):
        reader = EventReader(
            {
                "head": {"kind": "writer_turn", "previous": "context"},
                "context": {"kind": "context_changed", "previous": "sample"},
                "sample": {"kind": "writer_turn", "previous": None},
            }
        )

        self.assertTrue(context_root_changed_after(reader, "head", "sample"))

    def test_context_ancestry_requires_the_boundary_even_after_a_change(self):
        reader = EventReader(
            {
                "head": {"kind": "context_changed", "previous": None},
            }
        )

        with self.assertRaises(ProjectionError):
            context_root_changed_after(reader, "head", "outside")

    def test_rollout_scope_ends_at_start_and_cycles_fail_closed(self):
        reader = EventReader(
            {
                "head": {"kind": "context_changed", "previous": "start"},
                "start": {"kind": "rollout_started", "previous": "parent"},
                "parent": {"kind": "context_changed", "previous": None},
            }
        )
        self.assertTrue(context_root_changed_after(reader, "head", None))

        reader.events["head"] = {"kind": "writer_action", "previous": "start"}
        self.assertFalse(context_root_changed_after(reader, "head", None))

        reader.events["head"] = {"kind": "writer_turn", "previous": "head"}
        with self.assertRaises(ProjectionError):
            context_root_changed_after(reader, "head", None)


if __name__ == "__main__":
    unittest.main()
