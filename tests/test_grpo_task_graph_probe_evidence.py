"""Privacy and sibling-lineage evidence contracts for P1 criterion 6."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from writing_agent.grpo_task_graph_probe_evidence import _ledger_metrics
from writing_agent.grpo_task_graph_probe_privacy import (
    criterion_6,
    scan_run_privacy,
    verify_member_input_scope,
)
from writing_agent.task_graph_probe_tasks import (
    AUTHOR_PACKET_CANARY,
    EVALUATOR_PACKET_CANARY,
)


class ProbePrivacyEvidenceTests(unittest.TestCase):
    def test_measurements_report_member_stop_reasons_and_parse_failures(self):
        groups = [
            {
                "rewards": [Fraction(0), Fraction(0)],
                "terminations": {"token_limit:decision": 2},
                "members": [
                    {
                        "member_id": "member-1",
                        "stop_reason": "unparsed_tool_call",
                        "turns": [
                            SimpleNamespace(
                                usage={"prefill_tokens": 10},
                                input_token_count=12,
                                generated_token_count=3,
                                native_parse_failed=True,
                            )
                        ],
                    },
                    {
                        "member_id": "member-2",
                        "stop_reason": "decision_token_limit",
                        "turns": [
                            SimpleNamespace(
                                usage={"prefill_tokens": 11},
                                input_token_count=13,
                                generated_token_count=4,
                                native_parse_failed=None,
                            )
                        ],
                    },
                ],
            }
        ]

        measurements = _ledger_metrics(groups)

        self.assertEqual(
            measurements["stop_reason_counts"],
            {
                "per_member": {
                    "member-1": {"unparsed_tool_call": 1},
                    "member-2": {"decision_token_limit": 1},
                },
                "totals": {"unparsed_tool_call": 1, "decision_token_limit": 1},
            },
        )
        self.assertEqual(measurements["native_parse_failed_count"], 1)

    def _criterion(self, root: Path):
        scan = scan_run_privacy(root)
        sibling_scope = {
            "verified": True,
            "member_count": 4,
            "scope": "test double for the offline native audit's per-member reconstruction",
            "limits": "Only the stated member lineage scope is claimed.",
        }
        result = criterion_6(
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

        verified = verify_member_input_scope([group], reports, inspections_byte_identical=True)
        group["members"][0]["sampler_inputs_bound_to_own_lineage"] = False
        refused = verify_member_input_scope([group], reports, inspections_byte_identical=True)

        self.assertTrue(verified["verified"])
        self.assertIn("not absence of arbitrary shared public text", verified["limits"])
        self.assertFalse(refused["verified"])


if __name__ == "__main__":
    unittest.main()
