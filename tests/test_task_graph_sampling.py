import unittest

from writing_agent.task_graph import canonical_bytes
from writing_agent.task_graph_sampling import (
    AdapterEvidenceV1,
    PreparedRequestV1,
    ProjectionError,
    SamplingEvidenceV1,
)


class SamplingCodecTests(unittest.TestCase):
    def test_prepared_request_v1_round_trip_preserves_wire_identity(self):
        wire = {
            "record_type": "PreparedWriterRequestV1",
            "context_content_hash": "a" * 64,
            "context_revision_ref": "b" * 64,
            "rendering": {"template_ref": "c" * 64},
            "payload_ref": "d" * 64,
        }
        self.assertEqual(
            canonical_bytes(PreparedRequestV1.from_wire(wire).to_wire()), canonical_bytes(wire)
        )
        with self.assertRaises(ProjectionError):
            PreparedRequestV1.from_wire({**wire, "unowned": True})

    def test_sampling_evidence_v1_rejects_false_eligibility_claims(self):
        wire = SamplingEvidenceV1(
            action_id="run:action:0",
            context_content_hash="a" * 64,
            context_revision_ref="b" * 64,
            rendering_json='{"projection_version":"v1"}',
            exact_request_ref=None,
            prepared_request_ref=None,
            raw_output_ref=None,
            logprob_ref=None,
            usage_json='{"completion_tokens":1}',
            model="fake",
            seed=7,
            adapter=AdapterEvidenceV1.from_wire({"model": "fake", "seed": 7}),
        ).to_wire()
        self.assertEqual(
            canonical_bytes(SamplingEvidenceV1.from_wire(wire).to_wire()), canonical_bytes(wire)
        )
        for forged in (
            {**wire, "native_on_policy_eligible": True},
            {**wire, "token_evidence": "supplied"},
            {**wire, "reason": "native"},
        ):
            with self.subTest(forged=forged):
                with self.assertRaises(ProjectionError):
                    SamplingEvidenceV1.from_wire(forged)


if __name__ == "__main__":
    unittest.main()
