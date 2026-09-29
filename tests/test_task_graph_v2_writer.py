"""Native writer-turn binding, derived termination, and recovery contracts."""

from __future__ import annotations

import shutil
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.task_graph_fixtures import make_entry_fixture
from tests.task_graph_rollout_fixtures import build_rollout_fixture
from tests.test_task_graph_derive_writer import make_group, make_turn, make_view, with_budget
from writing_agent.task_graph import CheckpointV1, domain_hash_bytes, thaw
from writing_agent.task_graph_calls import intake_message
from writing_agent.task_graph_controller import next_step
from writing_agent.task_graph_derive_writer import derive_writer_turn_v2, writer_action_id
from writing_agent.task_graph_environment import RolloutEnvironment
from writing_agent.task_graph_errors import AdapterContractProjectionError, ProjectionError
from writing_agent.task_graph_gate import LineageGate, StoreArtifactReader
from writing_agent.task_graph_gatherers import SamplingRunner
from writing_agent.task_graph_group import GroupCoordinatorV1
from writing_agent.task_graph_ports import SampleResult
from writing_agent.task_graph_record_contracts import (
    POLICY_FIELDS,
    ContextPolicyV1,
    GroupSpecV1,
    group_identity,
)
from writing_agent.task_graph_records import (
    DecodingDescriptorV1,
    MemberStartV1,
    RendererDescriptorV1,
    RuntimeManifestV2,
    RuntimePortDescriptorV1,
    TokenizerDescriptorV1,
    WriterTurnV2,
)
from writing_agent.task_graph_sampling import NativeSamplingBudget
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_transition import CheckpointChain, SampleRef


def _native_view(max_tokens: int = 4):
    tokenizer = TokenizerDescriptorV1(
        model_id="tests/toy-tokenizer",
        revision="toy-r1",
        files_sha256={"tokenizer": "a" * 64},
    )
    fixture = make_entry_fixture(
        rendering_overrides={"tokenizer_ref": tokenizer.identity()},
        public_records=(tokenizer,),
    )
    _fixture, entry = make_view(fixture)
    old_spec, _member_view = make_group(entry, fixture.reader)
    renderer = RendererDescriptorV1(
        implementation="gemma4-native-append-v1",
        template_ref=entry.context.rendering["template_ref"],
        tokenizer_ref=tokenizer.identity(),
        tool_schema_ref=entry.context.rendering["tool_schema_ref"],
        stop_token_ids=(1, 106, 50),
        enable_thinking=False,
        suffix_rules_version="native-suffix-v1",
    )
    decoding = DecodingDescriptorV1(
        temperature=1,
        top_p=1,
        top_k=0,
        processors=(),
        max_tokens_per_decision=max_tokens,
        seed_rule="writer_seed ⊕ action ordinal (sha256-domain-v1)",
        logprob_convention="log_softmax(model logits after model softcap), fp32",
        trainer_ratio="recomputed, num_iterations=1",
    )
    native_capabilities = ("native_token_ledger", "sampled_logprobs", "usage_reporting")
    ports = tuple(
        RuntimePortDescriptorV1(
            schema=1,
            role=role,
            implementation=f"tests.{role.title()}",
            version="1",
            configuration={"capabilities": native_capabilities} if role == "sampling" else {},
        )
        for role in ("sampling", "environment", "tools", "evaluator")
    )
    manifest = RuntimeManifestV2(
        schema=2,
        ports=ports,
        capabilities=native_capabilities,
        renderer=renderer,
        tokenizer=tokenizer,
        decoding=decoding,
    )
    fixture.reader.public[manifest.identity()] = manifest.to_wire()
    fixture.reader.public[renderer.identity()] = renderer.to_wire()
    policy = dict(old_spec.policy)
    policy.update(
        adapter_ref=manifest.identity(),
        tokenizer_ref=tokenizer.identity(),
        decoding_ref=decoding.identity(),
    )
    group_id = group_identity(
        old_spec.group_sequence,
        old_spec.environment,
        policy,
        old_spec.group_seed,
        "fixture",
        len(old_spec.members),
        "native",
    )
    members = tuple(
        replace(member, member_id=f"grp-{group_id[:24]}-{member.ordinal:02d}")
        for member in old_spec.members
    )
    spec = GroupSpecV1(
        group_id=group_id,
        group_sequence=old_spec.group_sequence,
        group_seed=old_spec.group_seed,
        runner_mode="fixture",
        environment=old_spec.environment,
        policy=policy,
        members=members,
        training_mode="native",
    )
    state_data = entry.state.to_dict()
    state_data["position"]["lineage_id"] = members[0].member_id
    state = type(entry.state).from_dict(state_data)
    checkpoint_id = CheckpointV1(state=state, event_head=state.history["head"]).identity()
    view = replace(
        entry,
        root_checkpoint_id=checkpoint_id,
        checkpoint_id=checkpoint_id,
        state=state,
        group=spec,
        ancestry=CheckpointChain(checkpoint_id, entry.context),
    )
    return fixture, view, manifest


def _put_bytes(reader, data: bytes) -> str:
    ref = domain_hash_bytes("payload", data)
    if hasattr(reader, "byte_values"):
        reader.byte_values[ref] = data
    elif hasattr(reader, "store"):
        written = reader.store.put_bytes_artifact(data)
        if written != ref:
            raise AssertionError("test byte artifact was stored under another identity")
    else:
        written = reader.put_bytes_artifact(data)
        if written != ref:
            raise AssertionError("test byte artifact was stored under another identity")
    return ref


def _token_bytes(ids: tuple[int, ...]) -> bytes:
    return b"".join(token.to_bytes(4, "little") for token in ids)


def _turn(
    view,
    reader,
    *,
    input_ids=(10, 11),
    generated_ids=(1,),
    termination_kind="native_stop",
    stop_token_id=1,
    limit=None,
    usage=None,
    sampling_pins=None,
    adapter_trace=None,
    calls=(),
    content="draft",
):
    member = view.group.members[0]
    values = {
        "prompt_tokens": len(input_ids),
        "completion_tokens": len(generated_ids),
        "total_tokens": len(input_ids) + len(generated_ids),
        "prefill_tokens": len(input_ids),
        "cached_input_tokens": 0,
    }
    if usage is not None:
        values.update(usage)
    pins = {
        "manifest_ref": view.group.policy["adapter_ref"],
        "behavior_policy_ref": view.group.policy["behavior_policy_ref"],
        "decoding_ref": view.group.policy["decoding_ref"],
        "renderer_ref": view.group.policy["adapter_ref"],
        "seed": member.writer_seed,
    }
    # The renderer ref is a descriptor identity, not the enclosing manifest ref.
    manifest = RuntimeManifestV2.from_dict(reader.artifact(view.group.policy["adapter_ref"]))
    pins["renderer_ref"] = manifest.renderer.identity()
    if sampling_pins is not None:
        pins.update(sampling_pins)
    if adapter_trace is None:
        adapter_trace = {
            "context_revision_ref": view.context.revision_ref,
            "context_content_hash": view.context.content_ref,
            "rendering": thaw(view.context.rendering),
        }
    termination = {
        "kind": termination_kind,
        "stop_token_id": stop_token_id,
        "limit": limit,
    }
    return WriterTurnV2(
        action_id=writer_action_id(view),
        context_revision_ref=view.context.revision_ref,
        raw_output_ref=None,
        usage=values,
        adapter_trace=adapter_trace,
        message=intake_message({"content": content, "tool_calls": list(calls)}),
        input_token_ids_ref=_put_bytes(reader, _token_bytes(tuple(input_ids))),
        input_token_count=len(input_ids),
        generated_token_ids_ref=_put_bytes(reader, _token_bytes(tuple(generated_ids))),
        generated_token_count=len(generated_ids),
        logprobs={
            "ref": _put_bytes(
                reader, struct.pack(f"<{len(generated_ids)}f", *([-0.25] * len(generated_ids)))
            ),
            "codec": "f32-le",
            "shape": [len(generated_ids)],
        },
        termination=termination,
        sampling_pins=pins,
    )


def _assert_path(test, view, turn, reader, path: str):
    with test.assertRaises(AdapterContractProjectionError) as caught:
        derive_writer_turn_v2(view, turn, reader)
    test.assertTrue(str(caught.exception).startswith(path), str(caught.exception))


def _native_policy(store, rendering, manifest):
    context_policy = ContextPolicyV1(
        "compact", summarizer_version="visible-text-v1", max_summary_chars=20
    )
    policy = {
        field: store.put_artifact({"pin": field})
        for field in POLICY_FIELDS
        if field != "rng_derivation_version"
    }
    policy.update(
        model_ref=store.put_artifact({"model_id": "native-test-model"}),
        behavior_policy_ref=store.put_artifact({"policy": "native-test"}),
        tokenizer_ref=manifest.tokenizer.identity(),
        template_ref=rendering["template_ref"],
        adapter_ref=manifest.identity(),
        decoding_ref=manifest.decoding.identity(),
        context_policy_ref=store.put_artifact(context_policy.to_wire()),
        rng_derivation_version="sha256-domain-v1",
    )
    return policy


def _runtime_manifest_v2(store, rendering, *, max_tokens=4, tokenizer=None):
    tokenizer = tokenizer or TokenizerDescriptorV1(
        model_id="tests/toy-tokenizer",
        revision="toy-r1",
        files_sha256={"tokenizer": "b" * 64},
    )
    decoding = DecodingDescriptorV1(
        temperature=1,
        top_p=1,
        top_k=0,
        processors=(),
        max_tokens_per_decision=max_tokens,
        seed_rule="writer_seed ⊕ action ordinal (sha256-domain-v1)",
        logprob_convention="log_softmax(model logits after model softcap), fp32",
        trainer_ratio="recomputed, num_iterations=1",
    )
    renderer = RendererDescriptorV1(
        implementation="gemma4-native-append-v1",
        template_ref=rendering["template_ref"],
        tokenizer_ref=tokenizer.identity(),
        tool_schema_ref=rendering["tool_schema_ref"],
        stop_token_ids=(1, 106, 50),
        enable_thinking=False,
        suffix_rules_version="native-suffix-v1",
    )
    native_capabilities = ("native_token_ledger", "sampled_logprobs", "usage_reporting")
    ports = tuple(
        RuntimePortDescriptorV1(
            schema=1,
            role=role,
            implementation=f"tests.{role.title()}",
            version="1",
            configuration={"capabilities": native_capabilities} if role == "sampling" else {},
        )
        for role in ("sampling", "environment", "tools", "evaluator")
    )
    manifest = RuntimeManifestV2(
        schema=2,
        ports=ports,
        capabilities=native_capabilities,
        renderer=renderer,
        tokenizer=tokenizer,
        decoding=decoding,
    )
    store.put_artifact(tokenizer.to_wire())
    store.put_artifact(decoding.to_wire())
    store.put_artifact(renderer.to_wire())
    store.put_artifact(manifest.to_wire())
    return manifest


class WriterTurnV2Tests(unittest.TestCase):
    def setUp(self):
        self.fixture, self.view, self.manifest = _native_view()
        self.reader = self.fixture.reader

    def test_rule_1_chains_exact_prior_bytes_and_resets_at_context_root(self):
        first_turn = _turn(self.view, self.reader, input_ids=(10, 11), generated_ids=(1,))
        changed = _turn(self.view, self.reader, input_ids=(10, 12, 99), generated_ids=(1,))
        previous_ref = first_turn.identity()
        self.reader.public[previous_ref] = first_turn.to_wire()
        prior_event = "e" * 64
        sample = SampleRef("prior-action", prior_event, previous_ref, "action")

        from writing_agent.task_graph_sampling import decode_and_bind_sampling

        decode_and_bind_sampling(
            changed,
            self.view.context,
            self.view.group,
            (),
            None,
            self.view.budget,
            self.view.state.position["lineage_id"],
            self.reader,
        )
        chained = _turn(
            self.view,
            self.reader,
            input_ids=(10, 11, 1, 99),
            generated_ids=(1,),
        )
        decode_and_bind_sampling(
            chained,
            self.view.context,
            self.view.group,
            (sample,),
            prior_event,
            self.view.budget,
            self.view.state.position["lineage_id"],
            self.reader,
        )
        broken = replace(
            chained,
            input_token_ids_ref=_put_bytes(self.reader, _token_bytes((10, 12, 1, 99))),
        )
        with self.assertRaisesRegex(
            ProjectionError, "input.input_token_ids_ref: prior generated prefix"
        ):
            decode_and_bind_sampling(
                broken,
                self.view.context,
                self.view.group,
                (sample,),
                prior_event,
                self.view.budget,
                self.view.state.position["lineage_id"],
                self.reader,
            )

        context_event = "f" * 64
        self.reader.events[context_event] = {
            "kind": "context_changed",
            "previous": prior_event,
        }
        decode_and_bind_sampling(
            broken,
            self.view.context,
            self.view.group,
            (sample,),
            context_event,
            self.view.budget,
            self.view.state.position["lineage_id"],
            self.reader,
        )

    def test_rule_2_binds_counts_and_logprob_shape(self):
        bad_usage = _turn(
            self.view,
            self.reader,
            usage={"prompt_tokens": 99},
        )
        _assert_path(self, self.view, bad_usage, self.reader, "input.input_token_count:")

        bad_logprobs = replace(
            _turn(self.view, self.reader),
            logprobs={"ref": _put_bytes(self.reader, b""), "codec": "f32-le", "shape": [0]},
        )
        _assert_path(self, self.view, bad_logprobs, self.reader, "input.logprobs.shape:")

    def test_rule_3_rejects_false_token_limit_context_limit_and_nonfinal_stop(self):
        early_stop = _turn(
            self.view,
            self.reader,
            generated_ids=(10, 11),
            termination_kind="token_limit",
            stop_token_id=None,
            limit="decision",
        )
        _assert_path(self, self.view, early_stop, self.reader, "input.termination:")

        context_view = with_budget(
            self.view,
            self.reader,
            limits={"context_tokens": 20},
        )
        context_open = _turn(
            context_view,
            self.reader,
            input_ids=(10, 11),
            generated_ids=(),
            termination_kind="context_limit",
            stop_token_id=None,
            limit="context",
        )
        _assert_path(
            self,
            context_view,
            context_open,
            self.reader,
            "input.termination:",
        )

        nonfinal_stop = _turn(
            self.view,
            self.reader,
            generated_ids=(1, 12),
            termination_kind="native_stop",
            stop_token_id=12,
        )
        _assert_path(self, self.view, nonfinal_stop, self.reader, "input.termination:")
        misclaimed_stop = _turn(
            self.view,
            self.reader,
            generated_ids=(1,),
            termination_kind="native_stop",
            stop_token_id=106,
        )
        _assert_path(self, self.view, misclaimed_stop, self.reader, "input.termination:")

    def test_rule_3_uses_argmin_ties_and_maps_every_termination_class(self):
        tool_call = {
            "id": "write-1",
            "type": "function",
            "function": {"name": "write_file", "arguments": {"path": "draft.txt", "content": "x"}},
        }
        # A well-formed native stop on an ordinary final answer continues normally.
        final = derive_writer_turn_v2(
            self.view,
            _turn(self.view, self.reader, generated_ids=(1,), stop_token_id=1),
            self.reader,
        )
        self.assertEqual(final.view.outcome.stop_reason, self.view.outcome.stop_reason)
        self.assertEqual(final.state.position["phase"], "checking")

        # Tool-response EOS is correct only when the sampled message contains a call.
        well_terminated_call = derive_writer_turn_v2(
            self.view,
            _turn(
                self.view,
                self.reader,
                generated_ids=(50,),
                stop_token_id=50,
                calls=(tool_call,),
                content="",
            ),
            self.reader,
        )
        self.assertEqual(
            well_terminated_call.view.outcome.stop_reason, self.view.outcome.stop_reason
        )
        self.assertEqual(well_terminated_call.state.position["phase"], "ready_writer")

        unterminated_call = derive_writer_turn_v2(
            self.view,
            _turn(
                self.view,
                self.reader,
                generated_ids=(1,),
                stop_token_id=1,
                calls=(tool_call,),
                content="",
            ),
            self.reader,
        )
        self.assertEqual(unterminated_call.view.outcome.stop_reason, "unterminated_tool_call")
        self.assertEqual(unterminated_call.state.position["phase"], "ready_transition")

        unterminated_final = derive_writer_turn_v2(
            self.view,
            _turn(
                self.view,
                self.reader,
                generated_ids=(50,),
                stop_token_id=50,
                content="answer",
            ),
            self.reader,
        )
        self.assertEqual(unterminated_final.view.outcome.stop_reason, "unterminated_final_answer")

        decision_limit = derive_writer_turn_v2(
            self.view,
            _turn(
                self.view,
                self.reader,
                generated_ids=(10, 11, 12, 13),
                termination_kind="token_limit",
                stop_token_id=None,
                limit="decision",
            ),
            self.reader,
        )
        self.assertEqual(decision_limit.view.outcome.stop_reason, "decision_token_limit")

        generated_view = with_budget(
            self.view,
            self.reader,
            limits={"generated_tokens": 3},
        )
        generated_limit = derive_writer_turn_v2(
            generated_view,
            _turn(
                generated_view,
                self.reader,
                generated_ids=(10, 11, 12),
                termination_kind="token_limit",
                stop_token_id=None,
                limit="generated_budget",
            ),
            self.reader,
        )
        self.assertEqual(generated_limit.view.outcome.stop_reason, "generated_tokens_budget")

        context_view = with_budget(
            self.view,
            self.reader,
            limits={"context_tokens": 5},
        )
        context_limit = derive_writer_turn_v2(
            context_view,
            _turn(
                context_view,
                self.reader,
                input_ids=(10, 11),
                generated_ids=(12, 13, 14),
                termination_kind="token_limit",
                stop_token_id=None,
                limit="context",
            ),
            self.reader,
        )
        self.assertEqual(context_limit.view.outcome.stop_reason, "context_tokens_budget")
        self.assertEqual(context_limit.view.budget["consumed"]["context_tokens"], 5)

        zero_room = with_budget(
            self.view,
            self.reader,
            limits={"context_tokens": 2},
        )
        context_empty = derive_writer_turn_v2(
            zero_room,
            _turn(
                zero_room,
                self.reader,
                input_ids=(10, 11, 12),
                generated_ids=(),
                termination_kind="context_limit",
                stop_token_id=None,
                limit="context",
            ),
            self.reader,
        )
        self.assertEqual(context_empty.view.outcome.stop_reason, "context_tokens_budget")
        self.assertEqual(context_empty.view.budget["consumed"]["context_tokens"], 3)

        # Equal limits choose the first term: decision, then generated budget, then context.
        tied_view = with_budget(
            self.view,
            self.reader,
            limits={"generated_tokens": 4, "context_tokens": 6},
        )
        tied = _turn(
            tied_view,
            self.reader,
            input_ids=(10, 11),
            generated_ids=(12, 13, 14, 15),
            termination_kind="token_limit",
            stop_token_id=None,
            limit="decision",
        )
        self.assertEqual(
            derive_writer_turn_v2(tied_view, tied, self.reader).view.outcome.stop_reason,
            "decision_token_limit",
        )

    def test_rule_4_pins_manifest_behavior_decoding_renderer_and_seed(self):
        bad_pins = _turn(
            self.view,
            self.reader,
            sampling_pins={"behavior_policy_ref": "f" * 64},
        )
        _assert_path(
            self, self.view, bad_pins, self.reader, "input.sampling_pins.behavior_policy_ref:"
        )

    def test_rule_5_native_eligibility_claim_is_refused_by_v2_codec(self):
        body = _turn(self.view, self.reader).to_wire()
        body["adapter_trace"]["native_on_policy_eligible"] = True
        with self.assertRaisesRegex(ValueError, "forbidden claim"):
            WriterTurnV2.from_dict(body)

    def test_dispatch_refuses_cross_version_turn_manifest_pairs(self):
        from writing_agent.task_graph_sampling import decode_and_bind_sampling

        v1_fixture, v1_view = make_view()
        v1_group, v1_view = make_group(v1_view, v1_fixture.reader)
        v1_turn = make_turn(v1_view)
        with self.assertRaisesRegex(ProjectionError, "input.record_type:"):
            decode_and_bind_sampling(
                v1_turn,
                self.view.context,
                self.view.group,
                (),
                None,
                self.view.budget,
                self.view.state.position["lineage_id"],
                self.reader,
            )

        v2_turn = _turn(self.view, self.reader)
        with self.assertRaisesRegex(ProjectionError, "input.record_type:"):
            decode_and_bind_sampling(
                v2_turn,
                v1_view.context,
                v1_group,
                (),
                None,
                v1_view.budget,
                v1_view.state.position["lineage_id"],
                v1_fixture.reader,
            )

    def test_native_budget_is_public_and_reaches_prepared_sampler_input(self):
        view = with_budget(
            self.view,
            self.reader,
            limits={"generated_tokens": 12, "context_tokens": 100},
            consumed={"generated_tokens": 5, "context_tokens": 100},
        )
        environment = object.__new__(RolloutEnvironment)
        directive = next_step(view)
        self.assertEqual(directive.kind, "sample_writer")
        sampler_input = environment._build_port_input(view, directive)
        expected = NativeSamplingBudget(7, 100)
        self.assertEqual(sampler_input.native_sampling_budget, expected)

        class Artifacts:
            def put_artifact(self, _value, *, private=False):
                return "a" * 64

            def put_bytes_artifact(self, _value):
                return "b" * 64

        class Backend:
            prepared = None

            def sample(self, prepared):
                self.prepared = prepared
                return SampleResult({"role": "assistant", "content": "x", "tool_calls": []})

        backend = Backend()
        SamplingRunner(Artifacts(), backend).turn(sampler_input)
        self.assertEqual(backend.prepared.native_sampling_budget, expected)
        self.assertEqual(
            {
                field.name
                for field in backend.prepared.native_sampling_budget.__dataclass_fields__.values()
            },
            {"remaining_generated_tokens", "max_context_tokens"},
        )

    def test_crash_resume_rederives_identical_v2_commit_mid_lineage(self):
        with tempfile.TemporaryDirectory() as root:
            tokenizer = TokenizerDescriptorV1(
                model_id="tests/toy-tokenizer",
                revision="toy-r1",
                files_sha256={"tokenizer": "b" * 64},
            )
            fixture = build_rollout_fixture(
                Path(root) / "rollout",
                mode="context_token_limited",
                rendering_overrides={"tokenizer_ref": tokenizer.identity()},
                public_records=(tokenizer,),
            )
            rendering = fixture.runtime.context.rendering
            manifest = _runtime_manifest_v2(fixture.store, rendering, tokenizer=tokenizer)
            policy = _native_policy(fixture.store, rendering, manifest)

            class BoundSession:
                sealed_adapter_ref = manifest.identity()

                def require_seal(self, adapter_ref, **_kwargs):
                    if adapter_ref != self.sealed_adapter_ref:
                        raise ValueError("test session seal mismatch")

                def require_member_seal(self, view):
                    if view.group is None:
                        return
                    if view.group.policy["adapter_ref"] != self.sealed_adapter_ref:
                        raise ValueError("test member seal mismatch")

            session = BoundSession()
            fixture.env.session = session
            coordinator = GroupCoordinatorV1(fixture.env, session=session)
            spec = coordinator.seal(
                fixture.runtime.checkpoint_id,
                policy=policy,
                group_seed=17,
                group_sequence=93,
                member_count=2,
                runner_mode="fixture",
                training_mode="native",
            )
            runtime = fixture.env.start_member(
                fixture.runtime.checkpoint_id,
                MemberStartV1(spec.identity(), 0),
            )
            view = fixture.env.verify(runtime)
            turn = _turn(view, StoreArtifactReader(fixture.store))
            turn = replace(
                turn,
                action_id=writer_action_id(view),
                sampling_pins={
                    **turn.sampling_pins,
                    "manifest_ref": spec.policy["adapter_ref"],
                    "behavior_policy_ref": spec.policy["behavior_policy_ref"],
                    "decoding_ref": spec.policy["decoding_ref"],
                },
            )
            member_id = spec.members[0].member_id
            baseline_root = Path(root) / "baseline"
            shutil.copytree(fixture.store.root, baseline_root)

            baseline_gate = LineageGate()
            baseline_store = TaskGraphStore(baseline_root, verifier=baseline_gate)
            baseline_env = RolloutEnvironment(
                baseline_store,
                fixture.entry.graph,
                session,
                baseline_gate,
                fixture.entry.graph.policy,
            )
            baseline = baseline_env.commit(baseline_env.open_head(member_id), turn)

            old_head = fixture.store.read_head(member_id)
            publish = fixture.store.publish

            def crash_after_writes(*args, **kwargs):
                def fault(stage):
                    if stage == "after_immutable_writes":
                        raise RuntimeError("injected crash")

                kwargs["fault"] = fault
                return publish(*args, **kwargs)

            with patch.object(fixture.store, "publish", side_effect=crash_after_writes):
                with self.assertRaisesRegex(RuntimeError, "injected crash"):
                    fixture.env.commit(runtime, turn)
            self.assertEqual(fixture.store.read_head(member_id), old_head)

            retry_gate = LineageGate()
            retry_store = TaskGraphStore(fixture.store.root, verifier=retry_gate)
            retry_env = RolloutEnvironment(
                retry_store,
                fixture.entry.graph,
                session,
                retry_gate,
                fixture.entry.graph.policy,
            )
            retried = retry_env.commit(retry_env.open_head(member_id), turn)
            self.assertEqual(retried.commit_id, baseline.commit_id)


if __name__ == "__main__":
    unittest.main()
