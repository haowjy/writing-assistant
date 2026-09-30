"""Contract tests for the admitted Phase 8 probe task graphs."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from writing_agent.task_graph_probe_tasks import (
    AUTHOR_PACKET_CANARY,
    CONFIG_DIR,
    PRIVATE_STORE_DUMP_CANARY,
    build_admitted_entry,
    load_probe_tasks,
)


class ProbeTaskGraphIdentityTests(unittest.TestCase):
    def test_loader_accepts_an_explicit_config_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary)
            shutil.copyfile(CONFIG_DIR / "t1-lighthouse.json", task_dir / "t1-lighthouse.json")
            tasks = load_probe_tasks(task_dir)

        self.assertEqual([task["id"] for task in tasks], ["t1-lighthouse"])

    def test_graphs_match_the_planted_canary_task_identities(self):
        expected = {
            "t1-lighthouse": (
                "8a60bed1a896cf0c8c5447b9108dcf53fb33dc350118595336d138c275e6c006",
                "2b78a5898a560468e6590a8b74ac7a37ba8d9f7daeefc1c8ed2e8969d11cd584",
            ),
            "t2-winter-garden": (
                "81d03ad1dce41f062dcb8f0eb140a3b9576595961a6af90e14ac000fbeebe6d2",
                "6ac735ca750ab36a22ad5df7d155690a1fbe99267ed229b11f5067a95997062c",
            ),
            "t3-coastal-post": (
                "483c9faa358c7928e46087dd53b0468b1f2d326b885c5b466a4ec74aeb68cec9",
                "eec9745c12095a5e30f66517d373d8dc7639381707d22039b32844e771cbb28c",
            ),
        }
        actual = {}
        for config in load_probe_tasks():
            entry = build_admitted_entry(config)
            actual[config["id"]] = (entry.graph.instance.identity(), entry.state.identity())
            private_records = repr(entry.reader.private.values())
            public_records = repr(entry.reader.public.values())
            self.assertIn(AUTHOR_PACKET_CANARY, private_records)
            self.assertIn(PRIVATE_STORE_DUMP_CANARY, private_records)
            self.assertNotIn(AUTHOR_PACKET_CANARY, public_records)
            self.assertNotIn(PRIVATE_STORE_DUMP_CANARY, public_records)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
