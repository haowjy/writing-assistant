from __future__ import annotations

import hashlib
import re

from writing_agent.task_graph import (
    MessageV1,
    canonical_bytes,
    domain_hash,
    validate_hash,
)
from writing_agent.task_graph_group_records import GroupAdvantageV1
from writing_agent.task_graph_record_contracts import (
    ContextPolicyV1,
    ExecutionVersionsV1,
    GroupMemberSpecV1,
    GroupSpecV1,
)
from writing_agent.task_graph_records import (
    AdmissionPolicyV1,
    AuthorReplyV1,
    ContextContentV1,
    ContextOperationInputV1,
    ContextRevisionV1,
    DecodingDescriptorV1,
    EnvironmentStepV1,
    EvaluatorResultV1,
    ExternalInputsV1,
    MemberStartV1,
    OutcomeV1,
    RendererDescriptorV1,
    RuntimeManifestV2,
    RuntimePortDescriptorV1,
    SampledMessageV1,
    TokenizerDescriptorV1,
    ToolObservationV1,
    TrainingAdmissionV1,
    TrainingBatchV1,
    WriterTurnV1,
    WriterTurnV2,
)
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
        ("judged_checkpoint_id", "checkpoint"),
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
    "RendererDescriptorV1": (
        ("template_ref", "artifact"),
        ("tokenizer_ref", "artifact"),
        ("tool_schema_ref", "artifact"),
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
    "RuntimeManifestV2": (
        ("renderer.template_ref", "artifact"),
        ("renderer.tokenizer_ref", "artifact"),
        ("renderer.tool_schema_ref", "artifact"),
    ),
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
    "TrainingAdmissionV1": (("batch_ref", "artifact"), ("decision_ref", "artifact")),
    "TrainingBatchV1": (
        ("decision_ref", "artifact"),
        ("members[].advantage.result_ref", "artifact"),
        ("members[].advantage_f64_ref", "bytes"),
        ("members[].advantage_ref", "artifact"),
        ("members[].completion_ids_ref", "bytes"),
        ("members[].env_mask_ref", "bytes"),
        ("members[].prompt_ids_ref", "bytes"),
        ("members[].result_ref", "artifact"),
        ("members[].trailing_context_limit_turn_ref", "artifact"),
        ("members[].turn_spans[].turn_ref", "artifact"),
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
        ("raw_output_ref", "artifact|bytes"),
    ),
    "WriterTurnV2": (
        ("adapter_trace.adapter_ref", "artifact"),
        ("adapter_trace.behavior_policy_ref", "artifact"),
        ("adapter_trace.context_policy_ref", "artifact"),
        ("adapter_trace.context_revision_ref", "context_revision"),
        ("adapter_trace.decoding_ref", "artifact"),
        ("adapter_trace.model_ref", "artifact"),
        ("adapter_trace.policy_ref", "artifact"),
        ("adapter_trace.template_ref", "artifact"),
        ("adapter_trace.tokenizer_ref", "artifact"),
        ("context_revision_ref", "context_revision"),
        ("generated_token_ids_ref", "bytes"),
        ("input_token_ids_ref", "bytes"),
        ("logprobs.ref", "bytes"),
        ("raw_output_ref", "artifact|bytes"),
        ("sampling_pins.behavior_policy_ref", "artifact"),
        ("sampling_pins.decoding_ref", "artifact"),
        ("sampling_pins.manifest_ref", "artifact"),
        ("sampling_pins.renderer_ref", "artifact"),
    ),
    "TokenizerDescriptorV1": (),
    "DecodingDescriptorV1": (),
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
    "TokenizerDescriptorV1": frozenset({"files_sha256{}"}),
    "RuntimeManifestV2": frozenset({"tokenizer.files_sha256{}"}),
    "WriterTurnV2": frozenset({"adapter_trace.context_content_hash"}),
    "TrainingAdmissionV1": frozenset(
        {
            "group_id",
            "renderer_ref",
            "tokenizer_descriptor_ref",
            "adapter_hash_before",
            "adapter_hash_after",
        }
    ),
    "TrainingBatchV1": frozenset(
        {"group_id", "members[].advantage.group_id", "members[].ledger_hash"}
    ),
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
    tokenizer = TokenizerDescriptorV1(
        model_id="tests/toy-tokenizer",
        revision="toy-r1",
        files_sha256={"tokenizer": P},
    )
    decoding = DecodingDescriptorV1(
        temperature=1,
        top_p=1,
        top_k=0,
        processors=(),
        max_tokens_per_decision=64,
        seed_rule="writer_seed ⊕ action ordinal (sha256-domain-v1)",
        logprob_convention="log_softmax(model logits after model softcap), fp32",
        trainer_ratio="recomputed, num_iterations=1",
    )
    renderer = RendererDescriptorV1(
        implementation="gemma4-native-append-v1",
        template_ref=H,
        tokenizer_ref=tokenizer.identity(),
        tool_schema_ref=Q,
        stop_token_ids=(1, 106, 50),
        enable_thinking=False,
        suffix_rules_version="native-suffix-v1",
    )
    native_capabilities = ("native_token_ledger", "sampled_logprobs", "usage_reporting")
    native_ports = tuple(
        RuntimePortDescriptorV1(
            schema=1,
            role=role,
            implementation=f"tests.{role.title()}",
            version="1",
            configuration={"capabilities": native_capabilities} if role == "sampling" else {},
        )
        for role in ("sampling", "environment", "tools", "evaluator")
    )
    native_manifest = RuntimeManifestV2(
        schema=2,
        ports=native_ports,
        capabilities=native_capabilities,
        renderer=renderer,
        tokenizer=tokenizer,
        decoding=decoding,
    )
    native_admission = TrainingAdmissionV1(
        group_id=H,
        decision_ref=P,
        batch_ref=Q,
        audit_version="training-audit-v1",
        renderer_ref=renderer.identity(),
        tokenizer_descriptor_ref=tokenizer.identity(),
        adapter_hash_before=R,
        adapter_hash_after=H,
        members=({"member_id": "grp-example-00", "status": "admitted", "failed_check": None},),
    )
    batch_advantages = tuple(
        GroupAdvantageV1(
            group_id=H,
            member_id=f"grp-batch-{ordinal:02d}",
            result_ref=P if ordinal == 0 else Q,
            reward={"numerator": 0, "denominator": 1},
            mean={"numerator": 0, "denominator": 1},
            variance={"numerator": 0, "denominator": 1},
            centered={"numerator": 0, "denominator": 1},
            expression="zero",
            advantage={"numerator": 0, "denominator": 1},
            zero_variance=True,
        )
        for ordinal in range(2)
    )
    training_batch = TrainingBatchV1(
        schema=1,
        group_id=H,
        decision_ref=Q,
        max_context_tokens=4096,
        members=tuple(
            {
                "member_id": advantage.member_id,
                "result_ref": advantage.result_ref,
                "advantage_ref": advantage.identity(),
                "advantage": advantage,
                "advantage_f64_ref": R,
                "prompt_ids_ref": H,
                "completion_ids_ref": P,
                "env_mask_ref": Q,
                "turn_spans": (
                    {
                        "action_id": f"batch:{advantage.member_id}:action:0",
                        "turn_ref": R,
                        "completion_start": 0,
                        "completion_end": 1,
                        "ext_start": 0,
                    },
                ),
                "ledger_hash": H,
            }
            for advantage in batch_advantages
        ),
    )
    return (
        sampled,
        WriterTurnV1(
            action_id="rollout-1:action:0",
            context_revision_ref=context_revision.identity(),
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
        WriterTurnV2(
            action_id="rollout-1:action:1",
            context_revision_ref=context_revision.identity(),
            raw_output_ref=Q,
            usage={
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "prefill_tokens": 3,
                "cached_input_tokens": 0,
            },
            adapter_trace={"model": "toy-native"},
            message=sampled,
            input_token_ids_ref=H,
            input_token_count=3,
            generated_token_ids_ref=P,
            generated_token_count=2,
            logprobs={"ref": R, "codec": "f32-le", "shape": (2,)},
            termination={"kind": "native_stop", "stop_token_id": 1, "limit": None},
            sampling_pins={
                "manifest_ref": H,
                "behavior_policy_ref": P,
                "decoding_ref": decoding.identity(),
                "renderer_ref": renderer.identity(),
                "seed": 27,
            },
        ),
        renderer,
        tokenizer,
        decoding,
        native_manifest,
        native_admission,
        training_batch,
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
