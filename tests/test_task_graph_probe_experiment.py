"""Contract tests for the admitted Phase 8 probe task graphs."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from writing_agent.grpo_task_graph_probe_experiment import fixture_plans
from writing_agent.task_graph_contracts import writer_tool_schemas
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

    def test_experiment_tasks_render_stated_details_without_an_author_tool(self):
        details = {
            "t1-lighthouse": "amber lantern",
            "t2-winter-garden": "blue key",
            "t3-coastal-post": "copper bell",
        }
        for config in load_probe_tasks():
            with self.subTest(task=config["id"]):
                entry = build_admitted_entry(config)
                node = entry.graph.node(entry.node_id)
                self.assertIn(details[config["id"]].split()[0], config["public"]["brief"])
                self.assertEqual(node.checks["stated_detail"].spec["text"], details[config["id"]])
                self.assertNotIn("ask_author", node.contract.entry_contract.tool_allowlist)
                tool_manifest = writer_tool_schemas(
                    node.contract.entry_contract.tool_allowlist, node.interaction_policy
                )
                self.assertNotIn("ask_author", {tool["function"]["name"] for tool in tool_manifest})
                self.assertNotIn("ask_author", repr((config["public"]["brief"], tool_manifest)))

    def test_cpu_fixture_plans_never_request_author_interaction(self):
        plans = fixture_plans(load_probe_tasks())
        self.assertTrue(plans)
        self.assertTrue(all("ask_author" not in actions for actions in plans.values()))

    def test_graphs_match_the_planted_canaries_and_pinned_identities(self):
        expected = {
            "t1-lighthouse": (
                "92fd8b28155e40f61567561c5ef6fa43c3b6febe570b717878d484589f7b3555",
                "1a550a5e566ab4bdbc506a521c03519dd11c877ab9db22ec720f60e7397db0df",
            ),
            "t2-winter-garden": (
                "8a5b641472a7d5b25038038f33eee1c98e4ee47b61423dda1dfd300f1a4c7be8",
                "2ceba30998cee94b1b41ff82c7ac1c45de6146fd02d30bea41410c5dc189e0de",
            ),
            "t3-coastal-post": (
                "3a4545f8a78434c126edecc6b4fe14ba17572108d9263a684fceb0099aca8042",
                "00cb1821363d140a75f13cbd0c3b504d106415263ca38df52e6ae5cfe83d6325",
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
