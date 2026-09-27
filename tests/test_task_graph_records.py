from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from writing_agent.task_graph import (
    ContextRevisionV1 as LegacyContextRevisionV1,
)
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
from writing_agent.task_graph_errors import MissingReferenceError, WrongRecordDomainError
from writing_agent.task_graph_records import (
    ALL_RECORD_CODECS,
    LEGACY_PAYLOAD_RECORD_TYPES,
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
    ToolObservationV1,
    WriterRequestV1,
    WriterTurnV1,
    is_sha256_string,
    record_reference_edges,
)
from writing_agent.task_graph_store import TaskGraphStore

H = "a" * 64
P = "b" * 64
Q = "c" * 64
R = "d" * 64


def record_examples():
    message = MessageV1(content=("A draft is on the page.",), origin="entry:request")
    sampled = SampledMessageV1(
        content="Keep the dialogue terse.",
        tool_calls_was_list=True,
        calls=(
            {"bounded": True, "value": {"name": "read_file"}},
            {"bounded": False, "value": {"$noncanonical": "bounded-call"}},
        ),
    )
    rendering = {
        "projection_version": "v1",
        "prefix_id": "root",
        "template_ref": H,
        "tokenizer_ref": P,
        "tool_schema_ref": Q,
    }
    context_node = ContextContentV1(
        parent_ref=None,
        messages=(message,),
        tools=({"name": "read_file", "parameters": {"type": "object"}},),
        rendering=rendering,
    )
    context_revision = ContextRevisionV1(
        content_ref=context_node.identity(), event_head=H, provenance_refs=(P,)
    )
    return (
        sampled,
        WriterTurnV1(
            action_id="rollout-1:action:0",
            context_revision_ref=context_revision.identity(),
            request_ref=H,
            prepared_request_ref=P,
            raw_output_ref=Q,
            usage={"prompt_tokens": 4, "completion_tokens": 2, "backend_detail": {"x": 1}},
            adapter_trace={
                "model": "test-model",
                "seed": 23,
                "generated_token_ids": [4, 5],
                "per_token_logprobs_ref": R,
                "per_token_logprobs_codec": "f32-le",
                "per_token_logprobs_shape": [2],
                "model_ref": H,
                "behavior_policy_ref": P,
                "tokenizer_ref": Q,
                "template_ref": R,
                "adapter_ref": H,
                "decoding_ref": P,
                "context_policy_ref": Q,
                "policy_ref": R,
                "provider_detail": {"finish_reason": "stop"},
            },
            message=sampled,
        ),
        WriterRequestV1(context_revision.identity(), H, True),
        ToolObservationV1(
            call_id="call-1",
            dispatch={
                "spec": {"max_file_bytes": 20, "max_workspace_bytes": 40},
                "observation": {"ok": True, "valid": True, "result": {"text": "x"}},
                "effect": {"draft.txt": {"before": None, "after": "x"}},
            },
        ),
        AuthorReplyV1(
            request_ref=H,
            status="answered",
            utterance="Use the existing narrator.",
            decision_ids=("voice",),
            selected_proposals={"voice": ("close-third",)},
        ),
        EvaluatorResultV1(request_ref=H, status="pass", evidence_ref=P),
        ContextOperationInputV1(policy_ref=H),
        EnvironmentStepV1(directive={"kind": "request_author", "source": "writer_request"}),
        MemberStartV1(group_spec_ref=H, ordinal=0),
        ExternalInputsV1(schema=1, source_refs=(H, P)),
        OutcomeV1(
            schema=1,
            task_status="unknown",
            execution_status="running",
            stop_reason=None,
            reward_status="pending",
            training_eligibility="pending",
            candidate_checkpoint=H,
            requirement_version=P,
            checks=({"request_ref": Q, "result_ref": R},),
            transition_edge_id=None,
            failed_request_ref=Q,
            reward_ref=R,
            eligibility_ref=H,
        ),
        AdmissionPolicyV1(
            schema=1,
            writer_family=None,
            allowed_tools=("read_file", "write_file"),
            controller_versions=("deterministic-v1",),
            check_versions=("fixture-v1", "legacy-check-v1"),
        ),
        ContextContentV1(parent_ref=H, messages=(message,), tools=None, rendering=None),
        context_node,
        context_revision,
    )


def _walk(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            yield from _walk(item, child)
    elif isinstance(value, list):
        child = f"{path}[]"
        for item in value:
            yield from _walk(item, child)
    else:
        yield path, value


def _replace_ref(body, path, replacement):
    parts = path.split(".")

    def replace(value, index):
        segment = parts[index]
        expand = segment.endswith("[]")
        key = segment[:-2] if expand else segment
        if expand:
            for offset, item in enumerate(value[key]):
                if index == len(parts) - 1:
                    value[key][offset] = replacement
                else:
                    replace(item, index + 1)
        elif index == len(parts) - 1:
            value[key] = replacement
        else:
            replace(value[key], index + 1)

    replace(body, 0)


class RecordCodecTests(unittest.TestCase):
    def test_every_codec_has_a_canonical_exact_round_trip(self):
        examples = record_examples()
        self.assertEqual(len({type(item) for item in examples}), len(ALL_RECORD_CODECS))
        record_types = [item.RECORD_TYPE for item in examples if item.RECORD_TYPE is not None]
        self.assertEqual(len(record_types), len(set(record_types)))
        self.assertFalse(set(RECORD_TYPES) & LEGACY_PAYLOAD_RECORD_TYPES)
        for record in examples:
            with self.subTest(record=type(record).__name__):
                wire = record.to_wire()
                decoded = type(record).from_json(canonical_bytes(wire))
                self.assertEqual(canonical_bytes(decoded.to_wire()), canonical_bytes(wire))
                self.assertEqual(set(decoded.to_wire()), set(wire))

    def test_every_codec_rejects_floats_and_unknown_or_missing_keys(self):
        for record in record_examples():
            codec = type(record)
            wire = record.to_wire()
            with self.subTest(record=codec.__name__, bad="float"):
                bad = dict(wire)
                key = next(key for key in bad if key != "record_type")
                bad[key] = 1.25
                with self.assertRaises((TypeError, ValueError)):
                    codec.from_dict(bad)
            with self.subTest(record=codec.__name__, bad="unknown"):
                bad = {**wire, "unknown": "extra"}
                with self.assertRaises(ValueError):
                    codec.from_dict(bad)
            with self.subTest(record=codec.__name__, bad="missing"):
                bad = dict(wire)
                bad.pop(next(key for key in bad if key != "record_type"))
                with self.assertRaises(ValueError):
                    codec.from_dict(bad)

    def test_typed_bool_and_integer_fields_reject_python_coercions(self):
        for record in record_examples():
            wire = record.to_wire()
            codec = type(record)
            if isinstance(record, SampledMessageV1):
                wire["tool_calls_was_list"] = 1
                with self.subTest(record=codec.__name__, field="tool_calls_was_list"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)
            elif isinstance(record, WriterRequestV1):
                wire["verified_messages"] = 1
                with self.subTest(record=codec.__name__, field="verified_messages"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)
            elif isinstance(record, ToolObservationV1):
                wire["dispatch"]["observation"]["ok"] = 1
                with self.subTest(record=codec.__name__, field="observation.ok"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)
            elif isinstance(record, WriterTurnV1):
                wire["adapter_trace"]["seed"] = True
                with self.subTest(record=codec.__name__, field="seed"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)
            elif isinstance(record, MemberStartV1):
                wire["ordinal"] = True
                with self.subTest(record=codec.__name__, field="ordinal"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)
            elif isinstance(record, (ExternalInputsV1, AdmissionPolicyV1, OutcomeV1)):
                wire["schema"] = True
                with self.subTest(record=codec.__name__, field="schema"):
                    with self.assertRaises((TypeError, ValueError)):
                        codec.from_dict(wire)

    def test_refs_lint_covers_named_and_hash_shaped_values(self):
        for record in record_examples():
            refs = record.REFS
            paths = {path for path, _ in _walk(record.to_wire())}
            for path in paths:
                leaf = path.rsplit(".", 1)[-1].removesuffix("[]")
                if leaf.endswith(("_ref", "_refs")):
                    self.assertIn(path, refs, (type(record).__name__, path))
            for path, value in _walk(record.to_wire()):
                if is_sha256_string(value):
                    self.assertIn(path, refs, (type(record).__name__, path))

    def test_admission_policy_wire_is_sorted_and_unique(self):
        wire = AdmissionPolicyV1(
            schema=1,
            writer_family=None,
            allowed_tools=("read_file", "write_file"),
            controller_versions=("deterministic-v1",),
            check_versions=("fixture-v1",),
        ).to_wire()
        wire["allowed_tools"] = list(reversed(wire["allowed_tools"]))
        with self.assertRaises(ValueError):
            AdmissionPolicyV1.from_dict(wire)

    def test_environment_directive_union_uses_one_exact_shape_per_kind(self):
        directives = (
            {"kind": "request_author", "source": "mandatory_feedback"},
            {"kind": "request_checks"},
            {"kind": "commit_transition", "edge_id": "next"},
            {"kind": "seal_outcome", "task_status": "complete", "stop_reason": None},
            {"kind": "stop_exhausted", "stop_reason": "writer_budget"},
            {"kind": "publish_reward"},
        )
        for directive in directives:
            with self.subTest(kind=directive["kind"]):
                record = EnvironmentStepV1(directive)
                self.assertEqual(EnvironmentStepV1.from_json(record.to_json()), record)
                invalid = record.to_wire()
                invalid["directive"]["extra"] = "not on this directive"
                with self.assertRaises(ValueError):
                    EnvironmentStepV1.from_dict(invalid)

    def test_open_provider_maps_remain_ordinary_canonical_json(self):
        record = next(item for item in record_examples() if isinstance(item, WriterTurnV1))
        wire = record.to_wire()
        detail = {"$noncanonical": "provider metadata, not sampled intake"}
        wire["usage"]["vendor_detail"] = detail
        wire["adapter_trace"]["provider_detail"] = detail
        self.assertEqual(WriterTurnV1.from_dict(wire).to_wire(), wire)


class ChainedContextTests(unittest.TestCase):
    def test_merkle_identity_root_child_invariant_and_materialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "store"
            store = TaskGraphStore(root)
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
                "requirements",
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
        private = store.put_artifact({"fixture": "private"}, private=True)
        instance = GraphInstanceV1(
            template_ref=common["template"],
            entry_node="write",
            nodes=(NodeSpecV1(id="write", entry_contract=common["entry"]),),
        )
        store.persist(instance)
        context = LegacyContextRevisionV1(
            messages=(MessageV1(content=("Begin the story.",), origin="entry:request"),),
            rendering={
                "projection_version": "v1",
                "prefix_id": "root",
                "template_ref": common["template"],
                "tokenizer_ref": common["tokenizer"],
                "tool_schema_ref": common["tools"],
            },
        )
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
                "action_ids": (),
                "tool_result_ids": (),
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
            store = TaskGraphStore(Path(temporary) / "store")
            common, private, checkpoint, legacy_state = self._fixture(store)
            binary = store.put_bytes_artifact(b"raw model bytes")
            _, revision, _ = self._new_context_revision(store, common)
            sample = SampledMessageV1("turn", True, [])
            examples = {
                "WriterTurnV1": WriterTurnV1(
                    "line:action:0",
                    revision.identity(),
                    binary,
                    common["entry"],
                    common["entry"],
                    {"prompt_tokens": 1},
                    {
                        "per_token_logprobs_ref": binary,
                        "per_token_logprobs_codec": "f32-le",
                        "per_token_logprobs_shape": [1],
                        "model_ref": common["entry"],
                    },
                    sample,
                ),
                "WriterRequestV1": WriterRequestV1(revision.identity(), binary, True),
                "ToolObservationV1": record_examples()[3],
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
                "AdmissionPolicyV1": record_examples()[11],
            }
            self.assertEqual(set(examples), set(RECORD_TYPES))

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
                        if found_path == path and is_sha256_string(value)
                    )
                    self.assertEqual(edges, expected_edges)
                    for path in declared_paths:
                        if not any(
                            path in leaf_path or path.replace("[]", "") in leaf_path
                            for leaf_path, value in _walk(body)
                            if is_sha256_string(value)
                        ):
                            continue
                        bad = dict(body)
                        _replace_ref(bad, path, "f" * 64)
                        invalid_ref = store.put_artifact(bad)
                        with self.subTest(record_type=record_type, ref_path=path):
                            with self.assertRaises(MissingReferenceError):
                                store.get_artifact(invalid_ref, expected_domain="payload")

            new_semantics = store.put_artifact({"transition_semantics": "task-graph-derive-v1"})
            chained_state = EnvironmentStateV1.from_dict(
                {
                    **legacy_state.to_dict(),
                    "context_ref": revision.identity(),
                    "requirements_ref": private,
                    "versions_ref": new_semantics,
                }
            )
            chained_checkpoint = store.save_checkpoint(chained_state)
            self.assertEqual(
                store.load_checkpoint(chained_checkpoint).state.context_ref,
                revision.identity(),
            )

            legacy_context_with_new_semantics = EnvironmentStateV1.from_dict(
                {
                    **legacy_state.to_dict(),
                    "requirements_ref": private,
                    "versions_ref": new_semantics,
                }
            )
            with self.assertRaises(WrongRecordDomainError):
                store.save_checkpoint(legacy_context_with_new_semantics)

            chained_context_without_semantics = EnvironmentStateV1.from_dict(
                {**legacy_state.to_dict(), "context_ref": revision.identity()}
            )
            with self.assertRaises(WrongRecordDomainError):
                store.save_checkpoint(chained_context_without_semantics)

            unknown = store.put_artifact({"record_type": "UnregisteredV1", "value": 1})
            with self.assertRaises(WrongRecordDomainError):
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


if __name__ == "__main__":
    unittest.main()
