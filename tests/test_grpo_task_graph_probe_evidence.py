"""Privacy and sibling-lineage evidence contracts for P1 criterion 6."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from writing_agent.grpo_task_graph_probe_evidence import (
    _criterion_6,
    _privacy_scan,
    _sibling_input_scope,
)
from writing_agent.task_graph_probe_experiment import (
    AUTHOR_PACKET_CANARY,
    EVALUATOR_PACKET_CANARY,
)


class ProbePrivacyEvidenceTests(unittest.TestCase):
    def _criterion(self, root: Path):
        scan = _privacy_scan(root)
        sibling_scope = {
            "verified": True,
            "member_count": 4,
            "scope": "test double for the offline native audit's per-member reconstruction",
            "limits": "Only the stated member lineage scope is claimed.",
        }
        result = _criterion_6(
            inspections_byte_identical=True,
            privacy=scan,
            frozen_public_task_scope=True,
            network_disabled=True,
            sibling_input_scope=sibling_scope,
        )
        return scan, result

    def test_clean_run_passes_and_excludes_only_store_private_area(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("sampler-inputs", "batches", "logs", "training/private"):
                (root / directory).mkdir(parents=True)
            (root / "sampler-inputs" / "input.json").write_text('{"text":"clean"}')
            (root / "batches" / "step-1.json").write_text('{"ids":[]}')
            (root / "logs" / "stage.log").write_text("sampled")
            (root / "training" / "private" / "payload.json").write_text(
                AUTHOR_PACKET_CANARY + EVALUATOR_PACKET_CANARY
            )

            scan, criterion = self._criterion(root)

        self.assertEqual(scan["hits"], [])
        self.assertEqual(scan["checked_files"], 3)
        self.assertEqual(scan["excluded_private_store_area"], "training/private")
        self.assertEqual(
            scan["canaries_scanned"],
            [
                {
                    "canary_id": "unused_author_preference",
                    "sha256": hashlib.sha256(AUTHOR_PACKET_CANARY.encode()).hexdigest(),
                },
                {
                    "canary_id": "private_evaluator_check_spec",
                    "sha256": hashlib.sha256(EVALUATOR_PACKET_CANARY.encode()).hexdigest(),
                },
            ],
        )
        self.assertTrue(criterion["computed"])
        self.assertTrue(criterion["passed"])

    def test_canary_in_sampler_input_fails_criterion_6(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "sampler-inputs"
            inputs.mkdir()
            (inputs / "sample.json").write_text('{"messages":"' + AUTHOR_PACKET_CANARY + '"}')

            scan, criterion = self._criterion(root)

        self.assertEqual(
            scan["hits"],
            [
                {
                    "path": "sampler-inputs/sample.json",
                    "canary_id": "unused_author_preference",
                }
            ],
        )
        self.assertTrue(criterion["computed"])
        self.assertFalse(criterion["passed"])

    def test_sibling_scope_only_claims_audited_own_lineage_inputs(self):
        group = {
            "members": [{"sampler_inputs_bound_to_own_lineage": True} for _ in range(4)],
            "admission": SimpleNamespace(members=tuple({"status": "admitted"} for _ in range(4))),
        }
        reports = [
            {
                "status": "admitted",
                "member_count": 4,
                "admitted_count": 4,
                "mismatch_count": 0,
            }
            for _ in range(6)
        ]

        verified = _sibling_input_scope([group], reports, inspections_byte_identical=True)
        group["members"][0]["sampler_inputs_bound_to_own_lineage"] = False
        refused = _sibling_input_scope([group], reports, inspections_byte_identical=True)

        self.assertTrue(verified["verified"])
        self.assertIn("not absence of arbitrary shared public text", verified["limits"])
        self.assertFalse(refused["verified"])


if __name__ == "__main__":
    unittest.main()
