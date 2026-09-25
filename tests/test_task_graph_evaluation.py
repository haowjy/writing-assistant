"""Replay admits only exact, independently verifiable evaluator families."""

import unittest

from writing_agent.task_graph_contracts import CheckContractV1
from writing_agent.task_graph_evaluation import (
    VERIFIERS,
    EvaluationRequestV1,
    verify_evaluation_evidence,
)
from writing_agent.task_graph_sampling import ProjectionError


class EvaluationFamilyTests(unittest.TestCase):
    def request(self, family):
        fixture = family == "fixture-file-count-v1"
        check = CheckContractV1(
            id="nonempty",
            evaluator_version="fixture-file-count-v1" if fixture else "deterministic-v1",
            spec={
                "id": "nonempty",
                "metric": "Q1",
                "kind": "fixture_file_count" if fixture else "nonempty",
                "method": "fixture" if fixture else "deterministic",
                "required": True,
                **({} if fixture else {"path": "draft.txt"}),
            },
        )
        return EvaluationRequestV1.create(family, "a" * 64, check, "b" * 64, {"draft.txt": "text"})

    def test_two_families_verify_without_evaluator(self):
        for family in ("deterministic-file-v1", "fixture-file-count-v1"):
            with self.subTest(family=family):
                request = self.request(family)
                evidence = VERIFIERS[family](request)
                wire = evidence.to_wire(request)
                self.assertEqual(verify_evaluation_evidence(request, wire), evidence)
                with self.assertRaises(ProjectionError):
                    verify_evaluation_evidence(request, {**wire, "status": "fail"})

    def test_unknown_family_rejects(self):
        request = self.request("unknown")
        with self.assertRaises(ProjectionError):
            verify_evaluation_evidence(request, {"record_type": "UnknownEvidenceV1"})

    def test_family_cannot_replace_a_different_admitted_check(self):
        deterministic = self.request("deterministic-file-v1")
        fixture = self.request("fixture-file-count-v1")
        wire = VERIFIERS["fixture-file-count-v1"](fixture).to_wire(fixture)
        with self.assertRaises(ProjectionError):
            verify_evaluation_evidence(deterministic, wire)


if __name__ == "__main__":
    unittest.main()
