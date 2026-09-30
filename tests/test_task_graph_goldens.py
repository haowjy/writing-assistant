from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests.task_graph_golden_fixtures import (
    BYTE_GOLDEN,
    POST_GOLDEN_RECORD_TYPES,
    build_hash_golden,
    build_records_golden,
    build_records_v2_golden,
    build_rollout_golden,
)
from tests.task_graph_store_fixtures import PatchVerifier
from writing_agent.task_graph import (
    EnvironmentStateV1,
    canonical_bytes,
    domain_hash,
    domain_hash_bytes,
)
from writing_agent.task_graph_record_contracts import SEMANTICS_V1
from writing_agent.task_graph_records import (
    RECORD_TYPES,
    ContextContentV1,
    ContextRevisionV1,
    EnvironmentStepV1,
    record_reference_edges,
)
from writing_agent.task_graph_store import TaskGraphStore

FIXTURES = Path(__file__).parent / "fixtures"

# Deliberately separate from the regeneration path: changing a golden requires a
# semantics pin change and an explicit update to this digest table.
PINNED_GOLDEN_SHA256 = {
    "task-graph-derive-v1": {
        "task_graph_hashes.json": (
            "94b2de6a1b578e49e8ac3b7d466457bdb275343ba64c3f166826dd6a1886cfec"
        ),
        "task_graph_records_golden.json": (
            "4e7c69569b67f79750719fbeaaf46a0a33cf2b865bf227b4c9211edd9a9af70f"
        ),
        "task_graph_rollout_golden.json": (
            "4d7287b26cf2d038d6c623daea731ac7cdc2fb8d949183f38354854c463abe0f"
        ),
        "task_graph_records_v2_golden.json": (
            "66bc1d4a6e83a12788a216fd8863d4fd5a393dd90e977fe7819496b644268093"
        ),
    }
}


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _variants(record_type: str, value: dict):
    if record_type == "EnvironmentStepV1":
        return value["directives"].values()
    if record_type == "OutcomeV1":
        return value["lifecycle"].values()
    return (value,)


class TaskGraphGoldenTests(unittest.TestCase):
    def test_hash_golden_includes_the_literal_binary_artifact_envelope(self):
        golden = _fixture("task_graph_hashes.json")
        self.assertEqual(golden, build_hash_golden())
        binary = golden["bytes"]
        value = bytes.fromhex(binary["value_hex"])
        self.assertEqual(value, BYTE_GOLDEN)
        self.assertEqual(domain_hash_bytes("payload", value), binary["identity"])

        with tempfile.TemporaryDirectory() as temporary:
            store = TaskGraphStore(Path(temporary) / "store", verifier=PatchVerifier())
            identity = store.put_bytes_artifact(value)
            self.assertEqual(identity, binary["identity"])
            envelope_bytes = binary["canonical_envelope"].encode("utf-8")
            self.assertEqual(envelope_bytes, canonical_bytes(binary["envelope"]))
            self.assertEqual(
                store._artifact_path(identity, False).read_bytes(),
                envelope_bytes,
            )
            self.assertEqual(store.get_artifact(identity, expected_domain="payload:bytes"), value)

    def test_every_registered_wire_type_matches_its_canonical_body_and_identity(self):
        golden = _fixture("task_graph_records_golden.json")
        actual = build_records_golden()
        self.assertEqual(golden, actual)
        records = golden["records"]
        self.assertEqual(
            set(RECORD_TYPES) - POST_GOLDEN_RECORD_TYPES,
            set(records)
            - {
                "BudgetContractV1",
                "ContextContentV1",
                "ContextRevisionV1",
                "ExecutionVersionsV1",
                "EnvironmentStateV1",
            },
        )
        self.assertEqual(
            set(records["EnvironmentStepV1"]["directives"]),
            set(EnvironmentStepV1.FIELD_SPEC.required["directive"].variants),
        )
        self.assertEqual(
            set(records["OutcomeV1"]["lifecycle"]),
            {
                "entry",
                "checks_requested",
                "checks_completed",
                "transitioned",
                "sealed",
                "rewarded",
            },
        )

        for record_type in set(RECORD_TYPES) - POST_GOLDEN_RECORD_TYPES:
            for variant in _variants(record_type, records[record_type]):
                body = variant["body"]
                self.assertEqual(
                    variant["identity"],
                    domain_hash("payload", body),
                    msg=record_type,
                )
                self.assertEqual(
                    canonical_bytes(body),
                    canonical_bytes(json.loads(canonical_bytes(body))),
                    msg=record_type,
                )
                record_reference_edges(record_type, body)

        versions = records["ExecutionVersionsV1"]
        self.assertEqual(versions["body"]["transition_semantics"], SEMANTICS_V1)
        self.assertEqual(versions["identity"], domain_hash("payload", versions["body"]))
        record_reference_edges("ExecutionVersionsV1", versions["body"])

        context_records = records["ContextContentV1"]
        root = ContextContentV1.from_dict(context_records["root"]["body"])
        child = ContextContentV1.from_dict(context_records["child"]["body"])
        self.assertIsNone(root.parent_ref)
        self.assertEqual(child.parent_ref, root.identity())
        self.assertEqual(root.identity(), context_records["root"]["identity"])
        self.assertEqual(child.identity(), context_records["child"]["identity"])
        revision = ContextRevisionV1.from_dict(records["ContextRevisionV1"]["body"])
        self.assertEqual(revision.identity(), records["ContextRevisionV1"]["identity"])

        writer_turn = records["WriterTurnV1"]
        self.assertEqual(
            writer_turn["body"]["context_revision_ref"],
            writer_turn["body"]["adapter_trace"]["context_revision_ref"],
        )
        self.assertEqual(
            writer_turn["body"]["adapter_trace"]["context_content_hash"],
            context_records["child"]["identity"],
        )
        self.assertNotIn("request_ref", writer_turn["body"])
        self.assertNotIn("prepared_request_ref", writer_turn["body"])

        state = EnvironmentStateV1.from_dict(records["EnvironmentStateV1"]["body"])
        self.assertEqual(state.history["action_count"], 2)
        self.assertEqual(state.history["tool_result_count"], 2)
        self.assertNotIn("action_ids", state.history)
        self.assertNotIn("tool_result_ids", state.history)
        self.assertEqual(state.identity(), records["EnvironmentStateV1"]["identity"])

        without_limit = records["BudgetContractV1"]["without_max_generated_tokens"]["body"]
        with_limit = records["BudgetContractV1"]["with_max_generated_tokens"]["body"]
        self.assertNotIn("max_generated_tokens", without_limit)
        self.assertEqual(with_limit["max_generated_tokens"], 1024)

    def test_additive_v2_records_have_separate_identity_goldens(self):
        golden = _fixture("task_graph_records_v2_golden.json")
        self.assertEqual(golden, build_records_v2_golden())
        self.assertEqual(golden["transition_semantics"], SEMANTICS_V1)
        self.assertEqual(set(golden["records"]), POST_GOLDEN_RECORD_TYPES)
        for record_type, variant in golden["records"].items():
            body = variant["body"]
            with self.subTest(record_type=record_type):
                self.assertEqual(variant["identity"], domain_hash("payload", body))
                self.assertEqual(
                    canonical_bytes(body),
                    canonical_bytes(json.loads(canonical_bytes(body))),
                )
                record_reference_edges(record_type, body)
        optional_examples = golden["optional_field_examples"]
        self.assertEqual(
            set(optional_examples),
            {
                "GroupSegmentCreditV1_with_token_spans",
                "TrainingBatchV1_with_trailing_context_limit_turn_ref",
            },
        )
        for example_name, variant in optional_examples.items():
            body = variant["body"]
            record_type = body["record_type"]
            with self.subTest(optional_example=example_name):
                self.assertEqual(variant["identity"], domain_hash("payload", body))
                record = RECORD_TYPES[record_type].from_dict(body)
                self.assertEqual(record.to_wire(), body)
                self.assertEqual(record.identity(), variant["identity"])
                record_reference_edges(record_type, body)
        self.assertEqual(
            optional_examples["GroupSegmentCreditV1_with_token_spans"]["body"]["completion_start"],
            0,
        )
        self.assertIn(
            "trailing_context_limit_turn_ref",
            optional_examples["TrainingBatchV1_with_trailing_context_limit_turn_ref"]["body"][
                "members"
            ][0],
        )

    def test_scripted_rollout_event_checkpoint_commit_and_state_identities(self):
        golden = _fixture("task_graph_rollout_golden.json")
        with tempfile.TemporaryDirectory() as temporary:
            actual = build_rollout_golden(Path(temporary) / "rollout")
        self.assertEqual(golden, actual)
        self.assertEqual(
            golden["route"],
            [
                "entry",
                "write_file",
                "read_file",
                "ask_author",
                "reply",
                "final",
                "checks",
                "seal",
                "reward",
            ],
        )
        self.assertEqual(
            [event["kind"] for event in golden["events"]],
            [
                "writer_action",
                "tool_result",
                "writer_action",
                "tool_result",
                "writer_action",
                "external_requested",
                "author_turn",
                "writer_action",
                "external_requested",
                "check_recorded",
                "transition_committed",
                "termination_recorded",
                "reward_recorded",
            ],
        )
        self.assertEqual(len(golden["checkpoint_ids"]), len(golden["events"]) + 1)
        self.assertEqual(len(golden["commit_ids"]), len(golden["events"]))
        self.assertEqual(golden["final_directive"], "done")

    def test_golden_files_are_paired_to_the_semantics_pin(self):
        pinned = PINNED_GOLDEN_SHA256[SEMANTICS_V1]
        for name, expected_digest in pinned.items():
            path = FIXTURES / name
            with self.subTest(fixture=name):
                body = _fixture(name)
                self.assertEqual(body["transition_semantics"], SEMANTICS_V1)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_digest)


if __name__ == "__main__":
    unittest.main()
