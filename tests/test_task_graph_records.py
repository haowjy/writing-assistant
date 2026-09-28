from __future__ import annotations

import hashlib
import re
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

from tests.task_graph_fixtures import make_entry_fixture
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
    validate_hash,
)
from writing_agent.task_graph_errors import (
    MaterializationError,
    MissingReferenceError,
    ProjectionError,
    WrongRecordDomainError,
)
from writing_agent.task_graph_gate import LineageGate
from writing_agent.task_graph_record_contracts import (
    CompactionError,
    ContextPolicyV1,
    ExecutionVersionsV1,
    GroupError,
    GroupMemberSpecV1,
    GroupSpecV1,
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
    ToolObservationV1,
    WriterRequestV1,
    WriterTurnV1,
    materialize_context_nodes,
    record_reference_edges,
)
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_wire import DictOf, Hash, KindUnion, ListOf, Obj, RecordOf, UnionOf

H = "a" * 64
P = "b" * 64
Q = "c" * 64
R = "d" * 64


def _is_sha256(value):
    try:
        validate_hash(value)
    except (TypeError, ValueError):
        return False
    return True


EXPECTED_REFS = {
    "AdmissionPolicyV1": (),
    "AuthorReplyV1": (("request_ref", "private"),),
    "AuthorRequestV1": (
        ("author_packet_ref", "private"),
        ("prerequisite_results{}.result_ref", "artifact"),
        ("requirement_version", "artifact|private"),
        ("script_ref", "private"),
    ),
    "CheckRequestV1": (
        ("check_contract_hash", "private"),
        ("evaluator_packet_ref", "private"),
        ("requirement_version", "private"),
        ("target_checkpoint", "checkpoint"),
    ),
    "ContextOperationInputV1": (("policy_ref", "artifact"),),
    "ContextPolicyV1": (("seed_checkpoint_ref", "checkpoint"),),
    "DecisionLedgerV1": (),
    "DisclosureLedgerV1": (),
    "EnvironmentStepV1": (),
    "EvaluatorResultV1": (("evidence_ref", "artifact"), ("request_ref", "private")),
    "DeterministicCheckEvidenceV1": (
        ("evaluator_packet_ref", "private"),
        ("target_checkpoint", "checkpoint"),
    ),
    "FixtureFileCountEvidenceV1": (
        ("evaluator_packet_ref", "private"),
        ("target_checkpoint", "checkpoint"),
    ),
    "TranscriptReviewEvidenceV1": (
        ("evaluator_packet_ref", "private"),
        ("target_checkpoint", "checkpoint"),
    ),
    "ExecutionVersionsV1": (("admission_policy_ref", "artifact"),),
    "ExternalInputsV1": (("source_refs[]", "artifact"),),
    "GroupMemberSeedsV1": (("parent_rng_ref", "artifact"),),
    "GroupScriptedTerminalV1": (("start_checkpoint_id", "checkpoint"),),
    "GroupExecutionFailureV1": (
        ("evidence_ref", "artifact"),
        ("start_checkpoint_id", "checkpoint"),
    ),
    "GroupMemberResultV1": (
        ("availability_ref", "artifact"),
        ("failure_ref", "artifact"),
        ("final_checkpoint_id", "checkpoint"),
        ("fixture_ref", "artifact"),
        ("start_checkpoint_id", "checkpoint"),
        ("terminal_outcome_ref", "artifact"),
    ),
    "GroupDecisionV1": (
        ("advantage_refs[]", "artifact"),
        ("member_result_refs[]", "artifact"),
        ("segment_credit_refs[]", "artifact"),
    ),
    "GroupAdvantageV1": (("result_ref", "artifact"),),
    "GroupSegmentCreditV1": (
        ("action_ref", "event"),
        ("advantage_ref", "artifact"),
        ("message_ref", "artifact"),
        ("original_context_ref", "context_revision"),
        ("trace_ref", "artifact"),
    ),
    "GroupMemberSpecV1": (),
    "GroupSpecV1": (
        ("environment.author_packet_ref", "private"),
        ("environment.budget_ref", "artifact"),
        ("environment.context_revision_ref", "context_revision"),
        ("environment.decisions_ref", "artifact"),
        ("environment.disclosures_ref", "artifact"),
        ("environment.entry_checkpoint_id", "checkpoint"),
        ("environment.external_inputs_ref", "artifact"),
        ("environment.outcome_ref", "artifact"),
        ("environment.provenance_ref", "artifact"),
        ("environment.requirements_ref", "private"),
        ("environment.rng_ref", "artifact"),
        ("environment.versions_ref", "artifact"),
        ("policy.adapter_ref", "artifact"),
        ("policy.behavior_policy_ref", "artifact"),
        ("policy.context_policy_ref", "artifact"),
        ("policy.controller_ref", "artifact"),
        ("policy.decoding_ref", "artifact"),
        ("policy.model_ref", "artifact"),
        ("policy.simulator_ref", "artifact"),
        ("policy.template_ref", "artifact"),
        ("policy.tokenizer_ref", "artifact"),
    ),
    "MemberStartV1": (("group_spec_ref", "artifact"),),
    "OutcomeV1": (
        ("candidate_checkpoint", "checkpoint"),
        ("checks[].request_ref", "private"),
        ("checks[].result_ref", "artifact"),
        ("eligibility_ref", "artifact"),
        ("failed_request_ref", "private"),
        ("requirement_version", "private"),
        ("reward_ref", "artifact"),
    ),
    "RequirementLedgerV1": (),
    "RuntimePortDescriptorV1": (),
    "RuntimeManifestV1": (),
    "RewardV1": (
        ("candidate_checkpoint", "checkpoint"),
        ("check_result_refs[]", "artifact"),
        ("eligibility_ref", "artifact"),
        ("reward_contract_ref", "private"),
        ("terminal_outcome_ref", "artifact"),
    ),
    "SampledMessageV1": (),
    "ToolObservationV1": (),
    "TrainingEligibilityV1": (("terminal_outcome_ref", "artifact"),),
    "WriterRequestV1": (
        ("context_revision_ref", "context_revision"),
        ("payload_ref", "artifact|bytes"),
    ),
    "WriterTurnV1": (
        ("adapter_trace.adapter_ref", "artifact"),
        ("adapter_trace.behavior_policy_ref", "artifact"),
        ("adapter_trace.context_policy_ref", "artifact"),
        ("adapter_trace.context_revision_ref", "context_revision"),
        ("adapter_trace.decoding_ref", "artifact"),
        ("adapter_trace.model_ref", "artifact"),
        ("adapter_trace.per_token_logprobs_ref", "bytes"),
        ("adapter_trace.policy_ref", "artifact"),
        ("adapter_trace.template_ref", "artifact"),
        ("adapter_trace.tokenizer_ref", "artifact"),
        ("context_revision_ref", "context_revision"),
        ("prepared_request_ref", "artifact"),
        ("raw_output_ref", "artifact|bytes"),
        ("request_ref", "artifact|bytes"),
    ),
    "context_node": (
        ("parent_ref", "context_node"),
        ("rendering.template_ref", "artifact"),
        ("rendering.tokenizer_ref", "artifact"),
        ("rendering.tool_schema_ref", "artifact"),
    ),
    "context_revision": (
        ("content_ref", "context_node"),
        ("event_head", "event"),
        ("provenance_refs[]", "event"),
    ),
}

EXPECTED_NON_EDGE_HASHES = {
    "GroupMemberSeedsV1": frozenset({"group_id"}),
    "WriterTurnV1": frozenset({"adapter_trace.context_content_hash"}),
    "DeterministicCheckEvidenceV1": frozenset({"check_contract_hash"}),
    "FixtureFileCountEvidenceV1": frozenset({"check_contract_hash"}),
    "TranscriptReviewEvidenceV1": frozenset({"check_contract_hash"}),
    "GroupScriptedTerminalV1": frozenset({"group_id"}),
    "GroupExecutionFailureV1": frozenset({"group_id"}),
    "GroupMemberResultV1": frozenset({"group_id"}),
    "GroupDecisionV1": frozenset({"group_id"}),
    "GroupAdvantageV1": frozenset({"group_id"}),
    "GroupSegmentCreditV1": frozenset(
        {"group_id", "original_context_content_hash", "segment_content_hash"}
    ),
    "GroupSpecV1": frozenset(
        """group_id environment.entry_state_hash environment.entry_tree_hash
        environment.instance_hash environment.graph_hash environment.node_contract_hash
        environment.controller_contract_hash environment.check_contracts_hash
        environment.source_refs_hash environment.request_refs_hash
        environment.visible_prefix_hash environment.context_messages_hash
        environment.rendering_hash environment.tool_schemas_hash environment.budget_hash
        environment.versions_hash environment.continuation_hash
        environment.reward_contract_hash environment.simulator_contract_hash""".split()
    ),
}


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
    environment = {
        "entry_checkpoint_id": H,
        "entry_state_hash": P,
        "entry_tree_hash": Q,
        "instance_hash": R,
        "graph_hash": H,
        "node_id": "write",
        "node_visit_id": "visit-1",
        "node_contract_hash": P,
        "controller_contract_hash": Q,
        "check_contracts_hash": R,
        "reward_contract_hash": H,
        "simulator_contract_hash": None,
        "source_refs_hash": P,
        "request_refs_hash": Q,
        "visible_prefix_hash": R,
        "context_revision_ref": H,
        "context_messages_hash": H,
        "rendering_hash": P,
        "tool_schemas_hash": Q,
        "budget_ref": R,
        "budget_hash": H,
        "versions_ref": P,
        "versions_hash": Q,
        "author_packet_ref": None,
        "requirements_ref": R,
        "decisions_ref": H,
        "disclosures_ref": P,
        "external_inputs_ref": Q,
        "rng_ref": R,
        "outcome_ref": H,
        "provenance_ref": P,
        "continuation_hash": Q,
        "horizon": "node_exit",
    }
    policy = {
        key: H
        for key in (
            "model_ref",
            "behavior_policy_ref",
            "tokenizer_ref",
            "template_ref",
            "adapter_ref",
            "decoding_ref",
            "simulator_ref",
            "context_policy_ref",
            "controller_ref",
        )
    }
    policy["rng_derivation_version"] = "sha256-domain-v1"
    group_seed = 7
    member_count = 2
    group_id = domain_hash(
        "payload", ["GroupIdV1", 0, environment, policy, group_seed, "fixture", member_count]
    )

    def seed(role, ordinal=None):
        material = canonical_bytes(["GroupSeedV1", group_seed, role, ordinal])
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    group_members = tuple(
        GroupMemberSpecV1(
            member_id=f"grp-{group_id[:24]}-{ordinal:02d}",
            ordinal=ordinal,
            writer_seed=seed("writer", ordinal),
            environment_seed=seed("environment"),
        )
        for ordinal in range(member_count)
    )
    group_spec = GroupSpecV1(
        group_id=group_id,
        group_sequence=0,
        group_seed=group_seed,
        runner_mode="fixture",
        environment=environment,
        policy=policy,
        members=group_members,
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
                "context_revision_ref": context_revision.identity(),
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
        GroupMemberSpecV1("grp-example-0", 0, 7, 11),
        group_spec,
        ContextPolicyV1("carry"),
        ExecutionVersionsV1(
            1,
            "task-graph-derive-v1",
            H,
            {"max_file_bytes": 1, "max_workspace_bytes": 1},
        ),
    )


def shared_payload_examples():
    return {
        "GroupScriptedTerminalV1": {
            "record_type": "GroupScriptedTerminalV1",
            "schema": 1,
            "group_id": H,
            "member_id": "group-member-0",
            "start_checkpoint_id": P,
            "execution_status": "valid",
            "reward_status": "available",
            "reward": {"numerator": 1, "denominator": 1},
            "native_optimizer_eligible": False,
        },
        "GroupExecutionFailureV1": {
            "record_type": "GroupExecutionFailureV1",
            "schema": 1,
            "group_id": H,
            "member_id": "group-member-0",
            "start_checkpoint_id": P,
            "reason": "provider unavailable",
            "evidence_ref": None,
        },
        "GroupMemberResultV1": {
            "record_type": "GroupMemberResultV1",
            "schema": 1,
            "group_id": H,
            "member_id": "group-member-0",
            "start_checkpoint_id": P,
            "final_checkpoint_id": None,
            "terminal_outcome_ref": None,
            "availability_ref": None,
            "fixture_ref": None,
            "failure_ref": None,
            "execution_status": "pending",
        },
        "GroupDecisionV1": {
            "record_type": "GroupDecisionV1",
            "schema": 1,
            "group_id": H,
            "status": "pending",
            "reason": "awaiting members",
            "member_result_refs": [None],
            "advantage_refs": [],
            "segment_credit_refs": [],
            "native_optimizer_eligible": False,
        },
        "GroupAdvantageV1": {
            "record_type": "GroupAdvantageV1",
            "schema": 1,
            "group_id": H,
            "member_id": "group-member-0",
            "result_ref": P,
            "reward": {"numerator": 1, "denominator": 1},
            "mean": {"numerator": 1, "denominator": 1},
            "variance": {"numerator": 0, "denominator": 1},
            "centered": {"numerator": 0, "denominator": 1},
            "expression": "zero",
            "advantage": {"numerator": 0, "denominator": 1},
            "zero_variance": True,
            "native_optimizer_eligible": False,
        },
        "GroupSegmentCreditV1": {
            "record_type": "GroupSegmentCreditV1",
            "schema": 1,
            "group_id": H,
            "member_id": "group-member-0",
            "action_id": "group-action-0",
            "action_ref": P,
            "message_ref": Q,
            "trace_ref": R,
            "original_context_ref": H,
            "original_context_content_hash": Q,
            "advantage_ref": R,
            "segment_kind": "assistant_ending",
            "part_index": None,
            "segment_content_hash": None,
            "excluded_roles": [
                "system",
                "user",
                "author",
                "tool",
                "seed",
                "environment",
                "summary",
            ],
            "native_optimizer_eligible": False,
            "token_mask_ref": None,
            "logprob_ref": None,
        },
        "RuntimePortDescriptorV1": {
            "record_type": "RuntimePortDescriptorV1",
            "schema": 1,
            "role": "sampling",
            "implementation": "tests.FakeSampler",
            "version": "1",
            "configuration": {},
        },
        "RuntimeManifestV1": {
            "record_type": "RuntimeManifestV1",
            "schema": 1,
            "ports": [
                {
                    "record_type": "RuntimePortDescriptorV1",
                    "schema": 1,
                    "role": role,
                    "implementation": f"tests.{role.title()}",
                    "version": "1",
                    "configuration": {},
                }
                for role in ("sampling", "environment", "tools", "evaluator")
            ],
        },
        **{
            record_type: {
                "record_type": record_type,
                "schema": 1,
                "target_checkpoint": H,
                "check_contract_hash": Q,
                "evaluator_packet_ref": R,
                "evidence": {},
                "status": "pass",
            }
            for record_type in (
                "DeterministicCheckEvidenceV1",
                "FixtureFileCountEvidenceV1",
                "TranscriptReviewEvidenceV1",
            )
        },
        "DecisionLedgerV1": {
            "record_type": "DecisionLedgerV1",
            "schema": 1,
            "values": {},
            "proposals": {},
        },
        "DisclosureLedgerV1": {
            "record_type": "DisclosureLedgerV1",
            "schema": 1,
            "decisions": [],
        },
        "AuthorRequestV1": {
            "record_type": "AuthorRequestV1",
            "schema": 1,
            "request_id": "rollout:author:0",
            "source": "writer_request",
            "action_id": "rollout:action:0",
            "call_id": "rollout:call:0",
            "feedback_id": None,
            "arguments": {"question": "What changes?"},
            "decision_ids": ["voice"],
            "prerequisite_results": {"check": {"result_ref": H, "status": "pass"}},
            "requirement_version": P,
            "script_ref": Q,
            "author_packet_ref": R,
        },
        "CheckRequestV1": {
            "record_type": "CheckRequestV1",
            "schema": 1,
            "request_id": "rollout:check:0:continuity",
            "target_checkpoint": H,
            "requirement_version": P,
            "check_contract_hash": Q,
            "evaluator_packet_ref": R,
            "check_id": "continuity",
            "purpose": "completion",
        },
        "RewardV1": {
            "record_type": "RewardV1",
            "schema": 1,
            "terminal_outcome_ref": H,
            "reward_contract_ref": P,
            "candidate_checkpoint": Q,
            "check_result_refs": [R],
            "components": {"continuity": {"weight": 10000, "earned": 10000, "status": "pass"}},
            "numerator": 10000,
            "normalization": 10000,
            "availability": "available",
            "eligibility_ref": H,
        },
        "TrainingEligibilityV1": {
            "record_type": "TrainingEligibilityV1",
            "schema": 1,
            "terminal_outcome_ref": H,
            "status": "ineligible",
            "reason": "native_action_trace_unavailable",
        },
        "GroupMemberSeedsV1": {
            "record_type": "GroupMemberSeedsV1",
            "group_id": H,
            "member_id": "grp-example-0",
            "derivation": "sha256-domain-v1",
            "writer_seed": 7,
            "environment_seed": 9,
            "parent_rng_ref": P,
        },
        "RequirementLedgerV1": {
            "record_type": "RequirementLedgerV1",
            "schema": 1,
            "active": {"req-1": "Keep the reveal deferred."},
            "superseded": {},
        },
    }


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


def _hash_specs(spec, path=""):
    if isinstance(spec, Hash):
        yield path, spec.edge
    elif isinstance(spec, ListOf):
        yield from _hash_specs(spec.item, f"{path}[]")
    elif isinstance(spec, DictOf):
        yield from _hash_specs(spec.value, f"{path}{{}}")
    elif isinstance(spec, Obj):
        for key, child in (*spec.required.items(), *spec.optional.items()):
            yield from _hash_specs(child, f"{path}.{key}" if path else key)
    elif isinstance(spec, UnionOf):
        for child in spec.options:
            yield from _hash_specs(child, path)
    elif isinstance(spec, KindUnion):
        for child in spec.variants.values():
            yield from _hash_specs(child, path)
    elif isinstance(spec, RecordOf):
        yield from _hash_specs(spec.record.FIELD_SPEC, path)


def edge_path_matches(template, path):
    parts = []
    for part in template.split("."):
        if part.endswith("{}"):
            parts.extend((re.escape(part[:-2]), r"[^.]+"))
        else:
            parts.append(re.escape(part))
    return re.fullmatch(r"\.".join(parts), path) is not None


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
    def test_every_registered_refs_mapping_matches_the_wire_contract(self):
        self.assertEqual(
            {name: tuple(sorted(refs.items())) for name, refs in RECORD_EDGES.items()},
            EXPECTED_REFS,
        )

    def test_pre_s1_group_spec_wire_and_identity_are_unchanged(self):
        literal = (Path(__file__).parent / "fixtures" / "pre_s1_group_spec.json").read_bytes()
        record = GroupSpecV1.from_json(literal)
        self.assertEqual(canonical_bytes(record.to_wire()), literal)
        self.assertEqual(
            record.identity(),
            "3e80cde8f3eba7ce6f376d5d5d1e858abdf3f21d88c2cc3e9df26a35ffa46c9e",
        )
        self.assertEqual(record.schema, 1)
        self.assertEqual({member.schema for member in record.members}, {1})

    def test_every_codec_has_a_canonical_exact_round_trip(self):
        examples = record_examples()
        class_names = {
            item.RECORD_TYPE or item.EDGE_TYPE or type(item).__name__ for item in examples
        }
        self.assertEqual(
            class_names,
            set(RECORD_EDGES)
            - set(shared_payload_examples())
            - {"context_node", "context_revision"},
        )
        record_types = [item.RECORD_TYPE for item in examples if item.RECORD_TYPE is not None]
        self.assertEqual(len(record_types), len(set(record_types)))
        self.assertEqual(
            {item.RECORD_TYPE for item in examples if item.RECORD_TYPE is not None}
            | set(shared_payload_examples()),
            set(RECORD_TYPES),
        )
        self.assertNotIn("ExecutionVersionsV1", RECORD_TYPES)
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

    def test_writer_usage_token_counts_are_nonnegative_integers(self):
        record = next(item for item in record_examples() if isinstance(item, WriterTurnV1))
        for value in (True, -1, 1.5):
            wire = record.to_wire()
            wire["usage"]["prompt_tokens"] = value
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                WriterTurnV1.from_dict(wire)

    def test_refs_lint_covers_named_and_hash_shaped_values(self):
        for record in record_examples():
            self.assertEqual(
                {field.name for field in fields(record)},
                set(record.FIELD_SPEC.required),
                type(record).__name__,
            )
            self.assertEqual(
                record.REFS,
                RECORD_EDGES.get(record.RECORD_TYPE, record.REFS),
                type(record).__name__,
            )
            for path, edge in _hash_specs(record.FIELD_SPEC):
                if edge is None:
                    self.assertNotIn(path, record.REFS)
                else:
                    self.assertEqual(record.REFS[path], edge)

    def test_non_edge_hashes_are_pinned_and_example_hashes_have_declared_hash_paths(self):
        non_edges = {}
        examples = []
        for record in record_examples():
            spec = record.FIELD_SPEC
            paths = frozenset(path for path, edge in _hash_specs(spec) if edge is None)
            if paths:
                non_edges[record.RECORD_TYPE or type(record).__name__] = paths
            examples.append((type(record).__name__, spec, record.to_wire()))
        for name, body in shared_payload_examples().items():
            codec = RECORD_TYPES[name]
            spec = codec.FIELD_SPEC if isinstance(codec, type) else codec.fields
            non_edges_for_codec = frozenset(
                path for path, edge in _hash_specs(spec) if edge is None
            )
            if non_edges_for_codec:
                non_edges[name] = non_edges_for_codec
            examples.append((name, spec, body))

        self.assertEqual(non_edges, EXPECTED_NON_EDGE_HASHES)
        for name, spec, body in examples:
            declared_paths = tuple(_hash_specs(spec))
            for path, value in _walk(body):
                if not _is_sha256(value):
                    continue
                self.assertTrue(
                    any(edge_path_matches(template, path) for template, _ in declared_paths),
                    f"{name}.{path} contains an undeclared SHA-256 value",
                )

    def test_outcome_wire_fields_remain_pinned(self):
        outcome = next(item for item in record_examples() if isinstance(item, OutcomeV1))
        self.assertEqual(
            set(outcome.to_wire()),
            set(
                """record_type schema task_status execution_status stop_reason reward_status
                training_eligibility candidate_checkpoint requirement_version checks
                transition_edge_id failed_request_ref reward_ref eligibility_ref""".split()
            ),
        )

    def test_group_and_context_policy_binding_rules_reject_drift(self):
        spec = next(item for item in record_examples() if isinstance(item, GroupSpecV1))
        first, second = spec.members
        group_mutations = (
            ("group ID binding", lambda: replace(spec, group_id="e" * 64)),
            ("runner mode participates in group ID", lambda: replace(spec, runner_mode="real")),
            (
                "member ordinals are canonical",
                lambda: replace(spec, members=(second, first)),
            ),
            (
                "writer seed derivation",
                lambda: replace(
                    spec,
                    members=(replace(first, writer_seed=first.writer_seed + 1), second),
                ),
            ),
            (
                "environment seed derivation",
                lambda: replace(
                    spec,
                    members=(replace(first, environment_seed=first.environment_seed + 1), second),
                ),
            ),
            (
                "member ID binds ordinal",
                lambda: replace(spec, members=(replace(first, member_id="grp-other-0"), second)),
            ),
        )
        for label, mutate in group_mutations:
            with self.subTest(rule=label), self.assertRaises(GroupError):
                mutate()
        with patch("writing_agent.task_graph_record_contracts._group_seed", return_value=7):
            collided_seeds = tuple(
                replace(member, writer_seed=7, environment_seed=7) for member in spec.members
            )
            with self.assertRaises(GroupError):
                replace(spec, members=collided_seeds)

        context_mutations = (
            (
                "compact cannot name a seed",
                lambda: ContextPolicyV1(
                    "compact",
                    seed_name="opening",
                    summarizer_version="visible-text-v1",
                    max_summary_chars=0,
                ),
                CompactionError,
            ),
            (
                "only compact retains a tail",
                lambda: ContextPolicyV1("carry", retained_exchanges=1),
                CompactionError,
            ),
            (
                "seed name contains no whitespace",
                lambda: ContextPolicyV1(
                    "seed", seed_name="opening scene", seed_checkpoint_ref="a" * 64
                ),
                ValueError,
            ),
            (
                "only seed names a prefix",
                lambda: ContextPolicyV1("carry", seed_name="opening", seed_checkpoint_ref="a" * 64),
                CompactionError,
            ),
        )
        for label, mutate, error in context_mutations:
            with self.subTest(rule=label), self.assertRaises(error):
                mutate()

    def test_seed_context_policy_requires_its_checkpoint_at_decode(self):
        body = ContextPolicyV1("carry").to_wire()
        body.update(operation="seed", seed_name="opening")
        with self.assertRaises(CompactionError):
            ContextPolicyV1.from_dict(body)

    def test_unknown_adapter_reference_claims_are_rejected(self):
        record = next(item for item in record_examples() if isinstance(item, WriterTurnV1))
        wire = record.to_wire()
        wire["adapter_trace"]["unregistered_policy_ref"] = H
        with self.assertRaises(ValueError):
            WriterTurnV1.from_dict(wire)

    def test_registered_shared_shapes_and_untagged_execution_versions_are_typed(self):
        expected = {
            "GroupScriptedTerminalV1": (("checkpoint", P),),
            "GroupExecutionFailureV1": (("checkpoint", P),),
            "GroupMemberResultV1": (("checkpoint", P),),
            "GroupDecisionV1": (),
            "GroupAdvantageV1": (("artifact", P),),
            "GroupSegmentCreditV1": (
                ("event", P),
                ("artifact", Q),
                ("artifact", R),
                ("context_revision", H),
                ("artifact", R),
            ),
            "RuntimePortDescriptorV1": (),
            "RuntimeManifestV1": (),
            "DeterministicCheckEvidenceV1": (("checkpoint", H), ("private", R)),
            "FixtureFileCountEvidenceV1": (("checkpoint", H), ("private", R)),
            "TranscriptReviewEvidenceV1": (("checkpoint", H), ("private", R)),
            "DecisionLedgerV1": (),
            "DisclosureLedgerV1": (),
            "AuthorRequestV1": (
                ("artifact", H),
                ("artifact|private", P),
                ("private", Q),
                ("private", R),
            ),
            "CheckRequestV1": (
                ("checkpoint", H),
                ("private", P),
                ("private", Q),
                ("private", R),
            ),
            "RewardV1": (
                ("artifact", H),
                ("private", P),
                ("checkpoint", Q),
                ("artifact", R),
                ("artifact", H),
            ),
            "TrainingEligibilityV1": (("artifact", H),),
            "GroupMemberSeedsV1": (("artifact", P),),
            "RequirementLedgerV1": (),
            "ExecutionVersionsV1": (("artifact", H),),
        }
        for record_type, body in shared_payload_examples().items():
            with self.subTest(record_type=record_type):
                self.assertEqual(record_reference_edges(record_type, body), expected[record_type])
                with self.assertRaises((TypeError, ValueError)):
                    record_reference_edges(record_type, {**body, "unknown": "field"})
                missing = dict(body)
                missing.pop(next(iter(missing)))
                with self.assertRaises((TypeError, ValueError)):
                    record_reference_edges(record_type, missing)
        self.assertIn(("private", Q), expected["CheckRequestV1"])
        versions = next(item for item in record_examples() if isinstance(item, ExecutionVersionsV1))
        self.assertEqual(
            record_reference_edges("ExecutionVersionsV1", versions.to_wire()),
            expected["ExecutionVersionsV1"],
        )
        self.assertNotIn("ExecutionVersionsV1", RECORD_TYPES)

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
        for directive in (
            {"kind": "seal_outcome", "task_status": "complete", "stop_reason": ""},
            {"kind": "stop_exhausted", "stop_reason": ""},
        ):
            with self.subTest(directive=directive), self.assertRaises(ValueError):
                EnvironmentStepV1(directive)

    def test_tool_observation_rejects_noop_effects_and_nonpositive_limits(self):
        base = {
            "spec": {"max_file_bytes": 10, "max_workspace_bytes": 20},
            "observation": {"ok": True, "valid": True},
            "effect": {"draft.txt": {"before": "same", "after": "same"}},
        }
        with self.assertRaises(ValueError):
            ToolObservationV1("call-1", base)
        base["effect"] = {}
        base["spec"]["max_file_bytes"] = 0
        with self.assertRaises(ValueError):
            ToolObservationV1("call-1", base)

    def test_materializer_classifies_a_missing_root_invariant(self):
        broken_root = object.__new__(ContextContentV1)
        for name, value in {
            "parent_ref": None,
            "messages": (),
            "tools": None,
            "rendering": None,
        }.items():
            object.__setattr__(broken_root, name, value)
        revision = ContextRevisionV1(H, None, ())
        with self.assertRaises(MaterializationError):
            materialize_context_nodes(revision, {H: broken_root})

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
            store = TaskGraphStore(Path(temporary) / "store", verifier=PatchVerifier())
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
