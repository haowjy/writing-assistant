from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.task_graph_fixtures import make_entry_fixture
from tests.task_graph_record_fixtures import (
    _is_sha256,
    _replace_ref,
    _walk,
    record_examples,
)
from tests.task_graph_store_fixtures import PatchVerifier
from writing_agent.task_graph import (
    EnvironmentStateV1,
    EventV1,
    GraphInstanceV1,
    MessageV1,
    NodeSpecV1,
    canonical_bytes,
    domain_hash,
    tree_hash,
)
from writing_agent.task_graph_errors import (
    MissingReferenceError,
    ProjectionError,
    WrongRecordDomainError,
)
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_record_contracts import (
    ExecutionVersionsV1,
)
from writing_agent.task_graph_records import (
    RECORD_EDGES,
    RECORD_TYPES,
    AdmissionPolicyV1,
    AuthorReplyV1,
    ContextContentV1,
    ContextOperationInputV1,
    ContextRevisionV1,
    EnvironmentStepV1,
    EvaluatorResultV1,
    ExternalInputsV1,
    MemberStartV1,
    OutcomeV1,
    SampledMessageV1,
    WriterTurnV1,
    record_reference_edges,
)
from writing_agent.task_graph_store import TaskGraphStore

H = "a" * 64
P = "b" * 64
Q = "c" * 64
R = "d" * 64


class ChainedContextTests(unittest.TestCase):
    def test_merkle_identity_root_child_invariant_and_materialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            store = TaskGraphStore(root, verifier=PatchVerifier())
            refs = [
                store.put_artifact({"pin": name}) for name in ("template", "tokenizer", "tools")
            ]
            first = MessageV1(content=("Start the scene.",), origin="entry:request")
            second = MessageV1(content=("A door opens.",), origin="writer:turn")
            rendering = {
                "projection_version": "v1",
                "prefix_id": "root",
                "template_ref": refs[0],
                "tokenizer_ref": refs[1],
                "tool_schema_ref": refs[2],
            }
            node = ContextContentV1(None, (first,), (), rendering)
            child = ContextContentV1(node.identity(), (second,), None, None)
            event = EventV1(
                lineage_id="line",
                kind="context_changed",
                audience=("controller", "trainer"),
                payload_ref=store.put_artifact(
                    EnvironmentStepV1({"kind": "request_checks"}).to_wire()
                ),
                versions_ref=store.put_artifact({"fixture": "versions"}),
                provenance_ref=store.put_artifact({"fixture": "provenance"}),
            )
            store.persist(event)
            revision = ContextRevisionV1(child.identity(), event.identity(), (event.identity(),))
            self.assertEqual(node.identity(), domain_hash("context_content", node.to_wire()))
            self.assertEqual(revision.identity(), domain_hash("context", revision.to_wire()))
            self.assertEqual(store.persist(node), node.identity())
            self.assertEqual(store.persist(child), child.identity())
            self.assertEqual(store.persist(revision), revision.identity())

            reads = []
            original_load = store._load_record

            def count_records(identity, record_type, directory):
                if directory in {"context_content", "context_revisions", "events"}:
                    reads.append((directory, identity))
                return original_load(identity, record_type, directory)

            store._load_record = count_records
            with store.operation():
                materialized = store.materialize_context(revision.identity())
                again = store.materialize_context(revision.identity())
            self.assertEqual(materialized, again)
            self.assertEqual(
                tuple(message.content[0]["text"] for message in materialized.messages),
                ("Start the scene.", "A door opens."),
            )
            self.assertEqual(len(reads), 4)
            self.assertEqual(
                set(reads),
                {
                    ("context_content", node.identity()),
                    ("context_content", child.identity()),
                    ("context_revisions", revision.identity()),
                    ("events", event.identity()),
                },
            )
            self.assertEqual(materialized.tools, ())
            self.assertEqual(materialized.rendering, rendering)

            with self.assertRaises(ValueError):
                ContextContentV1(None, (first,), None, rendering)
            with self.assertRaises(ValueError):
                ContextContentV1(node.identity(), (second,), (), rendering)
            with self.assertRaises(ValueError):
                ContextContentV1(node.identity(), (second,), (), None)
            with self.assertRaises(ValueError):
                ContextContentV1(None, (first,), (), None)


class RecordClosureTests(unittest.TestCase):
    def _fixture(self, store):
        common = {
            key: store.put_artifact({"fixture": key})
            for key in (
                "entry",
                "template",
                "tokenizer",
                "tools",
                "decisions",
                "disclosures",
                "versions",
                "budgets",
                "rng",
                "external",
                "outcome",
                "provenance",
            )
        }
        common["requirements"] = store.put_artifact({"fixture": "requirements"}, private=True)
        common["versions"] = store.put_artifact(
            ExecutionVersionsV1(
                1,
                "task-graph-derive-v1",
                common["entry"],
                {"max_file_bytes": 4096, "max_workspace_bytes": 8192},
            ).to_wire()
        )
        private = store.put_artifact({"fixture": "private"}, private=True)
        instance = GraphInstanceV1(
            template_ref=common["template"],
            entry_node="write",
            nodes=(NodeSpecV1(id="write", entry_contract=common["entry"]),),
        )
        store.persist(instance)
        context_node = ContextContentV1(
            parent_ref=None,
            messages=(MessageV1(content=("Begin the story.",), origin="entry:request"),),
            tools=(),
            rendering={
                "projection_version": "v1",
                "prefix_id": "root",
                "template_ref": common["template"],
                "tokenizer_ref": common["tokenizer"],
                "tool_schema_ref": common["tools"],
            },
        )
        store.persist(context_node)
        context = ContextRevisionV1(context_node.identity(), None, ())
        store.persist(context)
        state = EnvironmentStateV1(
            instance_ref=instance.identity(),
            position={
                "node_id": "write",
                "visit_id": "visit-1",
                "phase": "ready_writer",
                "entry_contract": common["entry"],
                "start_checkpoint": None,
                "loop_counts": {},
                "lineage_id": "line",
            },
            files={"draft.txt": "A draft."},
            tree_hash=tree_hash({"draft.txt": "A draft."}),
            history={
                "head": None,
                "seq": 0,
                "branch_base": None,
                "imported_refs": (),
                "action_count": 0,
                "tool_result_count": 0,
            },
            context_ref=context.identity(),
            requirements_ref=common["requirements"],
            decisions_ref=common["decisions"],
            disclosures_ref=common["disclosures"],
            author_packet_ref=private,
            versions_ref=common["versions"],
            budgets_ref=common["budgets"],
            rng_ref=common["rng"],
            external_inputs_ref=common["external"],
            outcome_ref=common["outcome"],
            provenance_ref=common["provenance"],
            continuation={
                "tool_queue": (),
                "next_call": 0,
                "author_request": None,
                "check_requests": (),
                "external_requests": (),
                "applied_responses": (),
                "feedback_cursor": 0,
            },
            in_flight_effects=(),
        )
        checkpoint = store.save_checkpoint(state)
        return common, private, checkpoint, state

    def _new_context_revision(self, store, common):
        message = MessageV1(content=("Opening line.",), origin="entry:request")
        node = ContextContentV1(
            None,
            (message,),
            (),
            {
                "projection_version": "v1",
                "prefix_id": "root",
                "template_ref": common["template"],
                "tokenizer_ref": common["tokenizer"],
                "tool_schema_ref": common["tools"],
            },
        )
        store.persist(node)
        payload = store.put_artifact(EnvironmentStepV1({"kind": "request_checks"}).to_wire())
        event = EventV1(
            lineage_id="line",
            kind="context_changed",
            audience=("controller", "trainer"),
            payload_ref=payload,
            versions_ref=common["versions"],
            provenance_ref=common["provenance"],
        )
        store.persist(event)
        revision = ContextRevisionV1(node.identity(), event.identity(), (event.identity(),))
        store.persist(revision)
        return node, revision, event

    def test_registry_edges_are_followed_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TaskGraphStore(Path(temporary) / "store", verifier=PatchVerifier())
            common, private, checkpoint, legacy_state = self._fixture(store)
            binary = store.put_bytes_artifact(b"raw model bytes")
            _, revision, _ = self._new_context_revision(store, common)
            sample = SampledMessageV1("turn", True, [])
            records = {type(record).__name__: record for record in record_examples()}
            examples = {
                "WriterTurnV1": WriterTurnV1(
                    "line:action:0",
                    revision.identity(),
                    binary,
                    {"prompt_tokens": 1},
                    {
                        "per_token_logprobs_ref": binary,
                        "per_token_logprobs_codec": "f32-le",
                        "per_token_logprobs_shape": [1],
                        "model_ref": common["entry"],
                    },
                    sample,
                ),
                "ToolObservationV1": records["ToolObservationV1"],
                "AuthorReplyV1": AuthorReplyV1(private, "answered", "Keep it brief.", (), {}),
                "EvaluatorResultV1": EvaluatorResultV1(private, "pass", common["entry"]),
                "ContextOperationInputV1": ContextOperationInputV1(common["entry"]),
                "EnvironmentStepV1": EnvironmentStepV1({"kind": "request_checks"}),
                "MemberStartV1": MemberStartV1(common["entry"], 1),
                "ExternalInputsV1": ExternalInputsV1(1, (common["external"],)),
                "OutcomeV1": OutcomeV1(
                    1,
                    "unknown",
                    "running",
                    None,
                    "pending",
                    "pending",
                    checkpoint,
                    private,
                    ({"request_ref": private, "result_ref": common["entry"]},),
                    None,
                    private,
                    common["entry"],
                    common["entry"],
                ),
                "AdmissionPolicyV1": records["AdmissionPolicyV1"],
            }
            self.assertTrue(set(examples).issubset(RECORD_TYPES))

            for record_type, record in examples.items():
                with self.subTest(record_type=record_type):
                    body = record.to_wire()
                    artifact_ref = store.put_artifact(body)
                    self.assertEqual(
                        store.get_artifact(artifact_ref, expected_domain="payload"), body
                    )
                    edges = record_reference_edges(record_type, body)
                    declared_paths = tuple(RECORD_EDGES[record_type])
                    expected_edges = tuple(
                        (kind, value)
                        for path, kind in RECORD_EDGES[record_type].items()
                        for found_path, value in _walk(body)
                        if found_path == path and _is_sha256(value)
                    )
                    self.assertEqual(edges, expected_edges)
                    for path in declared_paths:
                        if not any(
                            path in leaf_path or path.replace("[]", "") in leaf_path
                            for leaf_path, value in _walk(body)
                            if _is_sha256(value)
                        ):
                            continue
                        bad = dict(body)
                        _replace_ref(bad, path, "f" * 64)
                        invalid_ref = store.put_artifact(bad)
                        with self.subTest(record_type=record_type, ref_path=path):
                            with self.assertRaises(MissingReferenceError):
                                store.get_artifact(invalid_ref, expected_domain="payload")

            admission = store.put_artifact(
                AdmissionPolicyV1(
                    1, None, ("read_file",), ("controller-v1",), ("check-v1",)
                ).to_wire()
            )
            new_semantics = store.put_artifact(
                ExecutionVersionsV1(
                    1,
                    "task-graph-derive-v1",
                    admission,
                    {"max_file_bytes": 4096, "max_workspace_bytes": 8192},
                ).to_wire()
            )
            chained_state = EnvironmentStateV1.from_dict(
                {
                    **legacy_state.to_dict(),
                    "context_ref": revision.identity(),
                    "requirements_ref": private,
                    "versions_ref": new_semantics,
                }
            )
            gated_store = TaskGraphStore(store.root, verifier=LineageGate())
            chained_checkpoint = gated_store.save_checkpoint(chained_state)
            self.assertEqual(
                store.load_checkpoint(chained_checkpoint).state.context_ref,
                revision.identity(),
            )

            missing_admission = store.put_artifact(
                ExecutionVersionsV1(
                    1,
                    "task-graph-derive-v1",
                    "f" * 64,
                    {"max_file_bytes": 4096, "max_workspace_bytes": 8192},
                ).to_wire()
            )
            missing_admission_state = EnvironmentStateV1.from_dict(
                {**chained_state.to_dict(), "versions_ref": missing_admission}
            )
            with self.assertRaises(MissingReferenceError):
                gated_store.save_checkpoint(missing_admission_state)

            public_requirement_request = store.put_artifact(
                {
                    "record_type": "CheckRequestV1",
                    "schema": 1,
                    "request_id": "rollout:check:public",
                    "target_checkpoint": chained_checkpoint,
                    "requirement_version": common["requirements"],
                    "check_contract_hash": private,
                    "evaluator_packet_ref": private,
                    "check_id": "continuity",
                    "purpose": "completion",
                },
                private=True,
            )
            with self.assertRaises(WrongRecordDomainError):
                store.get_artifact(public_requirement_request)

            unknown_versions = store.put_artifact(
                {
                    "schema": 1,
                    "transition_semantics": "task-graph-derive-v2",
                    "admission_policy_ref": admission,
                    "tool_spec": {"max_file_bytes": 4096, "max_workspace_bytes": 8192},
                }
            )
            unknown_state = EnvironmentStateV1.from_dict(
                {**chained_state.to_dict(), "versions_ref": unknown_versions}
            )
            with self.assertRaisesRegex(
                ProjectionError,
                "state.versions_ref.transition_semantics: runtime lineages require "
                "task-graph-derive-v1",
            ):
                store.save_checkpoint(unknown_state)

            unknown = store.put_artifact({"record_type": "UnregisteredV1", "value": 1})
            with self.assertRaises(ProjectionError):
                store.get_artifact(unknown, expected_domain="payload")
            public_reference = AuthorReplyV1(
                common["entry"], "answered", "This must not resolve publicly.", (), {}
            )
            public_record = store.put_artifact(public_reference.to_wire())
            with self.assertRaises(WrongRecordDomainError):
                store.get_artifact(public_record, expected_domain="payload")

            bad_binary_body = examples["WriterTurnV1"].to_wire()
            bad_binary_body["adapter_trace"]["per_token_logprobs_ref"] = common["entry"]
            bad_binary = store.put_artifact(bad_binary_body)
            with self.assertRaises(WrongRecordDomainError):
                store.get_artifact(bad_binary, expected_domain="payload")

    def test_forged_typed_artifact_reports_its_own_codec_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_entry_fixture()
            store = TaskGraphStore(Path(temporary) / "store", verifier=LineageGate())
            for value in fixture.reader.public.values():
                store.put_artifact(value)
            for value in fixture.reader.private.values():
                store.put_artifact(value, private=True)
            store.persist(fixture.graph.instance)
            for artifact in fixture.artifacts:
                store.persist_artifact(artifact)

            outcome = store.get_artifact(fixture.state.outcome_ref)
            outcome["forged_field"] = "not codec-owned"
            identity = domain_hash("payload", outcome)
            store._artifact_path(identity, False).write_bytes(
                canonical_bytes(
                    {"schema": 1, "domain": "payload", "encoding": "json", "body": outcome}
                )
            )

            with self.assertRaises(ProjectionError) as rejected:
                store.save_checkpoint(replace(fixture.state, outcome_ref=identity))

            self.assertIn("state.outcome_ref.forged_field", str(rejected.exception))


if __name__ == "__main__":
    unittest.main()
