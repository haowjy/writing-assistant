"""Contract tests for the admitted Phase 8 probe task graphs."""

from __future__ import annotations

import unittest

from writing_agent.task_graph_probe_experiment import build_admitted_entry, load_probe_tasks


class ProbeTaskGraphIdentityTests(unittest.TestCase):
    def test_graphs_match_the_pre_canary_task_identities(self):
        expected = {
            "t1-lighthouse": (
                "67bf74b4ffea0869a30454c925c126ecd8bbe8d3681a63e15ba68d21aa7f56e9",
                "305cd02c30b3cb714c4ca672a6c51d3cb742f6eba3899ceb1e41b5e43a3e6127",
            ),
            "t2-winter-garden": (
                "eb7bde92cda7630d5991e955137fe279381e65cf18c5193a63560dd06eb69026",
                "5b8ed2213d3e8fc3e2b072621b105c12e45bceb212a803b47c635cdd174f68df",
            ),
            "t3-coastal-post": (
                "214dfbe9918a89725f73e5630cacee835410af0db968a8804823f754df5c3549",
                "917c97a043f43b4dcde3edb4d7284feaba4999243b2b3ba23aad407f1e9ca4c7",
            ),
        }
        actual = {}
        for config in load_probe_tasks():
            entry = build_admitted_entry(config)
            actual[config["id"]] = (entry.graph.instance.identity(), entry.state.identity())
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
