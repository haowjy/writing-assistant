"""Replay admits only exact, independently verifiable evaluator families."""

import unittest

from writing_agent.task_graph import domain_hash, tree_hash
from writing_agent.task_graph_contracts import CheckContractV1
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_evaluation import (
    EvaluationRequestV1,
    produce_evaluation_evidence,
    verify_evaluation_evidence,
)


class EvaluationFamilyTests(unittest.TestCase):
    def test_transcript_family_verifies_frozen_inputs_not_subjective_truth(self):
        from writing_agent.task_graph_evaluation import EvaluationEvidenceV1

        packet = {"record_type": "EvaluatorPacketV1", "schema": 1, "instructions": "Judge."}
        packet_ref = domain_hash("payload", packet)
        check = CheckContractV1(
            id="review",
            evaluator_version="transcript-review-v1",
            spec={
                "id": "review",
                "metric": "Q1",
                "kind": "transcript_review",
                "method": "offline_transcript",
                "required": True,
            },
        )
        files = {"draft.txt": "Story."}
        request = EvaluationRequestV1.create(
            "transcript-review-v1",
            "a" * 64,
            check,
            packet_ref,
            files,
            evaluator_packet=packet,
        )

        class Resolver:
            def read_evaluator_packet(self, ref):
                if ref != packet_ref:
                    raise ValueError("unauthorized packet")
                return packet

        body = {
            "transcript": [
                {
                    "role": "request",
                    "payload": {
                        "target_checkpoint": request.target_checkpoint,
                        "check_contract_hash": check.identity(),
                        "evaluator_packet_ref": packet_ref,
                        "candidate_tree_hash": tree_hash(files),
                    },
                },
                {"role": "response", "status": "pass", "text": "subjective opinion"},
            ],
            "declared_status": "pass",
        }
        evidence = EvaluationEvidenceV1("transcript-review-v1", "pass", body)
        wire = evidence.to_wire(request)
        self.assertEqual(verify_evaluation_evidence(request, wire, Resolver()), evidence)
        with self.assertRaises(ProjectionError):
            verify_evaluation_evidence(request, wire)

        class WrongPacket:
            def read_evaluator_packet(self, ref):
                return {"record_type": "EvaluatorPacketV1", "instructions": "Other."}

        with self.assertRaises(ProjectionError):
            verify_evaluation_evidence(request, wire, WrongPacket())
        for bad in (
            {**wire, "schema": True},
            {**wire, "status": "fail"},
            {**wire, "evidence": {**body, "declared_status": "fail"}},
            {
                **wire,
                "evidence": {
                    **body,
                    "transcript": [
                        {
                            "role": "request",
                            "payload": {
                                **body["transcript"][0]["payload"],
                                "candidate_tree_hash": "b" * 64,
                            },
                        },
                        body["transcript"][1],
                    ],
                },
            },
        ):
            with self.assertRaises(ProjectionError):
                verify_evaluation_evidence(request, bad, Resolver())

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
                evidence = produce_evaluation_evidence(request)
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
        wire = produce_evaluation_evidence(fixture).to_wire(fixture)
        with self.assertRaises(ProjectionError):
            verify_evaluation_evidence(deterministic, wire)


if __name__ == "__main__":
    unittest.main()
