"""High-risk freeze, admission and mechanical-completion boundaries (no model deps)."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from writing_agent.grpo_identity import admission_identity
from writing_agent.grpo_probe_data import (
    DEFAULT_PACKET,
    build_probe_release,
    inspect_probe_release,
    load_probe_release,
    mechanical_probe_reward,
    validate_probe_fixtures,
)


class ProbeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory()
        root = Path(cls.storage.name)
        release = build_probe_release(root / "release")
        evidence = validate_probe_fixtures(root / "release", root / "evidence")
        cls.frozen = root, release, evidence

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.tmp_path = Path(self.scratch.name)

    def test_rebuild_is_reproducible_and_production_admitted(self):
        frozen = self.frozen
        tmp_path = self.tmp_path
        root, release, _ = frozen
        rebuilt = build_probe_release(tmp_path / "release")
        assert rebuilt["manifest"] == release["manifest"]
        assert rebuilt["freeze"] == release["freeze"]
        info = inspect_probe_release(root / "release")
        assert info["admission_mode"] == "production"
        assert not info["overlap_audit"]
        train_groups = {g for t in release["train"] for g in t["source_groups"]}
        dev_groups = {g for t in release["development"] for g in t["source_groups"]}
        assert len(train_groups) == 1 and len(dev_groups) == 2
        assert not train_groups & dev_groups
        assert set(info["excluded_source_groups"]) == dev_groups
        for task in release["tasks"]:
            assert task["visible"]["budgets"] == dict(
                max_steps=8, max_tool_calls=12, max_read_tokens=2000, max_total_bytes=65536
            )
        with self.assertRaisesRegex(ValueError, "empty"):
            build_probe_release(root / "release")
        with self.assertRaisesRegex(ValueError, "metadata"):
            admission_identity(release["development"][:1], release["admission"])
        blocked = copy.deepcopy(release["admission"])
        blocked["excluded_source_groups"] += list(train_groups)
        with self.assertRaisesRegex(ValueError, "Held-out"):
            admission_identity(release["train"], blocked)

    def test_source_packet_keeps_exact_original_inputs_and_rejects_parent_tampering(self):
        tmp_path = self.tmp_path
        packet = json.loads(DEFAULT_PACKET.read_text())
        release = build_probe_release(tmp_path / "release")
        for item, task in zip(packet["scenarios"], release["tasks"], strict=True):
            assert task["visible"]["initial_files"] == item["original"]["visible"]["initial_files"]
            assert task["source_ids"] == item["original"]["source_ids"]
            assert task["source_groups"] == item["original"]["source_groups"]
            assert task["provenance"] == item["original"]["provenance"]
        packet["catalog"][0]["text"] += "tamper"
        bad = tmp_path / "packet.json"
        bad.write_text(json.dumps(packet))
        with self.assertRaisesRegex(ValueError, "parent source record hash"):
            build_probe_release(tmp_path / "bad", packet_path=bad)

    def test_release_hash_tampering_rejected(self):
        tmp_path = self.tmp_path
        release = build_probe_release(tmp_path / "release")
        path = tmp_path / "release/visible" / (release["train"][0]["id"] + ".json")
        visible = json.loads(path.read_text())
        visible["brief"] += " changed"
        path.write_text(json.dumps(visible))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_probe_release(tmp_path / "release")

    def test_nine_golden_and_invalid_outcomes(self):
        frozen = self.frozen
        _, _, evidence = frozen
        assert len(evidence["tasks"]) == 9
        for task in evidence["tasks"]:
            rows = {r["case"]: r for r in task["cases"]}
            assert rows["golden"]["reward"]["value"] == 1
            for case, row in rows.items():
                reward = row["reward"]
                assert reward["components"]["semantic_status"] == "unjudged"
                assert reward["components"]["mechanical_completion"] == (case == "golden")
                if case in {
                    "untouched",
                    "empty",
                    "failed",
                    "missing",
                    "partial",
                    "broken",
                    "partial-topic",
                }:
                    assert reward["value"] == 0
                elif case != "golden":
                    assert 0 < reward["value"] < 1
            assert task["tool_calls"] <= 12 and task["read_tokens"] <= 2000

    def test_partial_wiki_and_unchanged_draft_cannot_collect_reward(self):
        frozen = self.frozen
        root, release, _ = frozen
        for task in release["tasks"]:
            saved = json.loads((root / "evidence" / task["id"] / "golden.json").read_text())
            result = saved["result"]
            for path in task["labels"]["probe_contract"]["required_files"]:
                bad = copy.deepcopy(result)
                bad["after"][path] = task["visible"]["initial_files"].get(path, "")
                # Candidate's misleading before snapshot must not change edit-scope/delivery.
                bad["before"] = {}
                assert mechanical_probe_reward(task, bad).value == 0
            if task["labels"]["probe_contract"].get("wiki_topic_required"):
                bad = copy.deepcopy(result)
                bad["after"]["kb/state.md"] = "[Index](index.md)"
                assert mechanical_probe_reward(task, bad).value == 0
            if task["family"] == "F2":
                bad = copy.deepcopy(result)
                bad["after"]["drafts/scene.md"] = (
                    task["visible"]["initial_files"]["drafts/scene.md"] + "\n "
                )
                assert mechanical_probe_reward(task, bad).value == 0

    def test_scope_uses_changed_paths_and_retrieval_needs_successful_tools(self):
        frozen = self.frozen
        root, release, _ = frozen
        wiki = release["train"][1]
        result = json.loads((root / "evidence" / wiki["id"] / "golden.json").read_text())["result"]
        assert "notes/source.md" in result["after"]
        assert mechanical_probe_reward(wiki, result).value == 1
        result["after"]["notes/source.md"] += " unauthorized edit"
        assert mechanical_probe_reward(wiki, result).value < 1
        retrieval = release["train"][2]
        result = json.loads((root / "evidence" / retrieval["id"] / "golden.json").read_text())[
            "result"
        ]
        for event in result["trace"]:
            if event["type"] == "tool":
                event["observation"]["ok"] = False
        reward = mechanical_probe_reward(retrieval, result)
        assert reward.value < 1
        assert not next(
            c["passed"] for c in reward.components["checks"] if c["id"] == "instruction-exposed"
        )

    def test_literal_checks_do_not_claim_semantic_validation(self):
        frozen = self.frozen
        root, release, _ = frozen
        task = release["train"][0]
        result = json.loads((root / "evidence" / task["id"] / "golden.json").read_text())["result"]
        result["after"]["drafts/scene.md"] = result["after"]["drafts/scene.md"].replace(
            "He had not seen the contract", "He knew every clause in her contract"
        )
        reward = mechanical_probe_reward(task, result)
        # This is a documented blind spot, not a semantic-continuity pass.
        assert reward.value == 1
        assert reward.components["semantic_status"] == "unjudged"

    def test_import_does_not_import_optional_model_packages(self):
        code = (
            "import sys; import writing_agent.grpo_probe_data; "
            "assert not {'torch', 'transformers', 'peft', 'trl'} & sys.modules.keys()"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
