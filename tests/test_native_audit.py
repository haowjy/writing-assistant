"""Admission audit checks exercised against sample-time adapter lies."""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.test_native_gemma import (
    TOKENIZER_PATH,
    TOKENIZER_REVISION,
    _descriptors,
    _tiny_peft_gemma,
    _tokenizer_file_hashes,
)

H = "a" * 64
P = "b" * 64


class LyingNativeBackend:
    """Mutate only the adapter's returned sample, never store artifacts."""

    def __init__(self, backend):
        self.backend = backend
        self.descriptor = backend.descriptor
        self.manifest_descriptors = backend.manifest_descriptors
        self.lies_by_seed: dict[int, str] = {}

    def sample(self, prepared):
        result = self.backend.sample(prepared)
        lie = self.lies_by_seed.get(prepared.writer_seed)
        if lie is None:
            return result

        if lie == "renderer_initial_context" and prepared.native_history is None:
            input_ids = list(result.input_token_ids)
            input_ids[0] += 1
            return self._replace_input(result, tuple(input_ids))

        if lie == "external_suffix_and_context_limit" and prepared.native_history is not None:
            prefix_length = len(prepared.native_history.input_token_ids) + len(
                prepared.native_history.generated_token_ids
            )
            input_ids = list(result.input_token_ids)
            if len(input_ids) <= prefix_length:
                raise AssertionError("fixture must render an external suffix after turn zero")
            input_ids[prefix_length] = (
                input_ids[prefix_length] + 1
            ) % self.backend.tokenizer.vocab_size
            return self._replace_input(result, tuple(input_ids))

        if lie == "raw_output_and_message" and prepared.native_history is None:
            message = dict(result.message)
            message["content"] = (message.get("content") or "") + "lying adapter"
            return replace(result, message=message)

        return result

    @staticmethod
    def _replace_input(result, input_ids):
        usage = dict(result.usage)
        usage.update(
            prompt_tokens=len(input_ids),
            total_tokens=len(input_ids) + len(result.generated_token_ids),
            prefill_tokens=len(input_ids),
        )
        return replace(result, input_token_ids=input_ids, usage=usage)


def _build_group(root, model, tokenizer, descriptors, *, member_count, lies_by_ordinal=None):
    from tests.task_graph_rollout_fixtures import (
        _entry_fixture,
        build_rollout_fixture,
        make_gatherers,
    )
    from tests.test_task_graph_v2_writer import _native_policy
    from writing_agent.native_gemma import NativeGemmaSampleBackend
    from writing_agent.task_graph_composition import RuntimeSession
    from writing_agent.task_graph_group import GroupCoordinatorV1
    from writing_agent.task_graph_group_records import GroupMemberResultV1
    from writing_agent.task_graph_local import (
        DeterministicEvaluator,
        LocalTextToolProvider,
        LocalWorkspaceEnvironment,
    )
    from writing_agent.task_graph_ports import RuntimeDependenciesV1
    from writing_agent.task_graph_rollout import RolloutDriver

    _renderer, tokenizer_descriptor, decoding = descriptors
    entry = _entry_fixture(
        "triple_feedback",
        "deterministic-file-v1",
        rendering_overrides={"tokenizer_ref": tokenizer_descriptor.identity()},
        public_records=(tokenizer_descriptor,),
    )
    from writing_agent.native_gemma import make_native_manifest_descriptors

    descriptors = make_native_manifest_descriptors(
        tokenizer,
        model_id=tokenizer_descriptor.model_id,
        revision=tokenizer_descriptor.revision,
        tokenizer_files_sha256=tokenizer_descriptor.files_sha256,
        template_ref=entry.params.rendering["template_ref"],
        tool_schema_ref=entry.params.rendering["tool_schema_ref"],
        max_tokens_per_decision=decoding.max_tokens_per_decision,
    )

    native = NativeGemmaSampleBackend(
        model,
        tokenizer,
        manifest_descriptors=descriptors,
    )
    backend = LyingNativeBackend(native)
    fixture_dir = Path(tempfile.mkdtemp(prefix="fixture-", dir=root))
    fixture = build_rollout_fixture(
        fixture_dir / "native-audit",
        mode="triple_feedback",
        entry_fixture=entry,
    )
    tools = LocalTextToolProvider()
    dependencies = RuntimeDependenciesV1(
        backend,
        LocalWorkspaceEnvironment(tools),
        tools,
        DeterministicEvaluator(),
    )
    unbound = RuntimeSession.create(fixture.store, dependencies)
    fixture.env.session = unbound.bind(fixture.store, unbound.manifest_ref)
    policy = _native_policy(
        fixture.store,
        fixture.runtime.context.rendering,
        dependencies.manifest(),
    )
    coordinator = GroupCoordinatorV1(fixture.env, session=fixture.env.session)
    spec = coordinator.seal(
        fixture.runtime.checkpoint_id,
        policy=policy,
        group_seed=117,
        group_sequence=940 + member_count,
        member_count=member_count,
        runner_mode="real",
        training_mode="native",
    )
    backend.lies_by_seed = {
        spec.members[ordinal].writer_seed: check
        for ordinal, check in (lies_by_ordinal or {}).items()
    }

    for ordinal, member in enumerate(spec.members):
        runtime = coordinator.start(spec, ordinal, policy=policy)
        result = RolloutDriver(
            fixture.env,
            make_gatherers(fixture, sampler=backend),
        ).run(runtime, max_steps=80)
        if result.directive.kind != "done":
            raise AssertionError(f"native fixture did not finish member {ordinal}")
        view = fixture.env.verify(result.runtime)
        outcome_ref = result.runtime.state.outcome_ref
        if view.outcome.reward_ref is not None:
            outcome_ref = fixture.store.get_artifact(view.outcome.reward_ref)[
                "terminal_outcome_ref"
            ]
        coordinator.collect(
            spec,
            GroupMemberResultV1(
                group_id=spec.group_id,
                member_id=member.member_id,
                start_checkpoint_id=coordinator._start_receipt(spec, ordinal)[
                    "start_checkpoint_id"
                ],
                final_checkpoint_id=result.runtime.checkpoint_id,
                terminal_outcome_ref=outcome_ref,
                availability_ref=view.outcome.reward_ref,
                execution_status="valid",
            ),
        )

    decision = coordinator.finalize(spec)
    if decision.status not in {"ready", "tie"}:
        raise AssertionError("native audit fixture must produce an exportable decision")
    return fixture, spec, decision, policy


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft")),
    "requires optional torch, transformers, and peft model dependencies",
)
class NativeAuditLyingAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoTokenizer

        cls.torch = torch
        cls.tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
        cls.model, cls.logit_hook = _tiny_peft_gemma(cls.tokenizer)
        cls.descriptors = _descriptors(cls.tokenizer)
        cls.temp = tempfile.TemporaryDirectory(prefix="native-audit-lies-")
        cls.root = Path(cls.temp.name)
        cls.fixture, cls.spec, cls.decision, cls.policy = _build_group(
            cls.root,
            cls.model,
            cls.tokenizer,
            cls.descriptors,
            member_count=4,
            lies_by_ordinal={
                0: "renderer_initial_context",
                1: "external_suffix_and_context_limit",
                2: "raw_output_and_message",
            },
        )
        cls.member_ids = tuple(member.member_id for member in cls.spec.members)

        cls.bad_tokenizer_root = cls.root / "lying-tokenizer"
        cls.bad_tokenizer_root.mkdir()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "config.json",
        ):
            (cls.bad_tokenizer_root / name).write_bytes((TOKENIZER_PATH / name).read_bytes())
        # Whitespace is semantically inert JSON, but makes the tokenizer file hash false.
        config = cls.bad_tokenizer_root / "tokenizer_config.json"
        config.write_bytes(config.read_bytes() + b"\n")
        cls.bad_tokenizer = AutoTokenizer.from_pretrained(
            str(cls.bad_tokenizer_root), local_files_only=True
        )
        from writing_agent.native_gemma import make_native_manifest_descriptors

        bad_descriptors = make_native_manifest_descriptors(
            cls.bad_tokenizer,
            model_id=str(cls.bad_tokenizer_root),
            revision=TOKENIZER_REVISION,
            tokenizer_files_sha256=_tokenizer_file_hashes(),
            template_ref=H,
            tool_schema_ref=P,
            max_tokens_per_decision=1,
        )
        (
            cls.bad_tokenizer_fixture,
            cls.bad_tokenizer_spec,
            cls.bad_tokenizer_decision,
            cls.bad_tokenizer_policy,
        ) = _build_group(
            cls.root,
            cls.model,
            cls.bad_tokenizer,
            bad_descriptors,
            member_count=2,
        )

    @classmethod
    def tearDownClass(cls):
        cls.logit_hook.remove()
        cls.temp.cleanup()

    def _clone_store(self, source, name):
        from writing_agent.task_graph_gate import LineageGate
        from writing_agent.task_graph_store import TaskGraphStore

        root = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.root))
        shutil.copytree(source.root, root, dirs_exist_ok=True)
        return TaskGraphStore(root, verifier=LineageGate())

    def _audit(self, store, spec, decision, tokenizer, *, tokenizer_root, after):
        from writing_agent.native_audit import audit_training_batch

        return audit_training_batch(
            store,
            spec,
            decision,
            tokenizer,
            adapter_hash_before=spec.policy["behavior_policy_ref"],
            adapter_hash_after=after,
            tokenizer_root=tokenizer_root,
        )

    def _assert_durable_refusal(self, store, admission, expected):
        from writing_agent.native_audit import TrainingAuditError, require_training_admission

        self.assertEqual(store.get_artifact(admission.identity()), admission.to_wire())
        by_member = {item["member_id"]: item for item in admission.members}
        for member_id, check in expected.items():
            self.assertEqual(by_member[member_id]["status"], "refused")
            self.assertEqual(by_member[member_id]["failed_check"], check)
        with self.assertRaises(TrainingAuditError) as caught:
            require_training_admission(admission)
        self.assertEqual(caught.exception.admission.to_wire(), admission.to_wire())

    def _assert_offline_refusal(self, store, group_id, admission):
        from writing_agent.native_audit import TrainingAuditError, inspect_group_offline

        with self.assertRaises(TrainingAuditError) as caught:
            inspect_group_offline(store.root, group_id)
        self.assertEqual(caught.exception.admission.to_wire(), admission.to_wire())

    def _audit_sample_lies(self):
        store = self._clone_store(self.fixture.store, "sample-lies")
        admission = self._audit(
            store,
            self.spec,
            self.decision,
            self.tokenizer,
            tokenizer_root=TOKENIZER_PATH,
            after=self.policy["behavior_policy_ref"],
        )
        return store, admission

    def _forced_native_group(self, raw_output, *, parse_failure_claim=None):
        from types import SimpleNamespace

        from writing_agent.native_gemma import (
            NativeGemmaSampleBackend,
            make_native_manifest_descriptors,
        )

        generated_ids = tuple(self.tokenizer.encode(raw_output, add_special_tokens=False))
        descriptors = make_native_manifest_descriptors(
            self.tokenizer,
            model_id="google/gemma-4-E2B-it",
            revision=TOKENIZER_REVISION,
            tokenizer_files_sha256=_tokenizer_file_hashes(),
            template_ref=H,
            tool_schema_ref=P,
            max_tokens_per_decision=len(generated_ids),
        )
        original_sample = NativeGemmaSampleBackend.sample

        def force_generation(model, inputs, generation, *, seed):
            prompt_ids = inputs["input_ids"]
            for index, token_id in enumerate(generated_ids):
                scores = self.torch.full(
                    (1, self.tokenizer.vocab_size), float("-inf"), dtype=self.torch.float32
                )
                scores[0, token_id] = 0.0
                prior = self.torch.tensor(
                    [(*prompt_ids[0].tolist(), *generated_ids[:index])], dtype=self.torch.long
                )
                generation["logits_processor"](prior, scores)
            generated = self.torch.tensor(
                [(*prompt_ids[0].tolist(), *generated_ids)], dtype=self.torch.long
            )
            return SimpleNamespace(sequences=generated)

        def sample(self, prepared):
            with patch("writing_agent.native_gemma.generate_with_seed", force_generation):
                result = original_sample(self, prepared)
            if parse_failure_claim is None:
                return result
            trace = dict(result.trace or {})
            if parse_failure_claim:
                trace["native_parse_failed"] = True
            else:
                trace.pop("native_parse_failed", None)
            return replace(result, trace=trace or None)

        with patch.object(NativeGemmaSampleBackend, "sample", sample):
            return _build_group(
                self.root,
                self.model,
                self.tokenizer,
                descriptors,
                member_count=2,
            )

    def _assert_parse_failure_group(self, raw_output, expected_stop_reason):
        from writing_agent.native_audit import require_training_admission
        from writing_agent.task_graph_gate import LineageGate, StoreArtifactReader
        from writing_agent.task_graph_group_records import GroupMemberResultV1
        from writing_agent.task_graph_records import WriterTurnV2

        fixture, spec, decision, policy = self._forced_native_group(raw_output)
        result = GroupMemberResultV1.from_dict(
            fixture.store.get_artifact(decision.member_result_refs[0])
        )
        final = LineageGate().view(fixture.store, result.final_checkpoint_id)
        self.assertEqual(final.outcome.task_status, "incomplete")
        self.assertEqual(final.outcome.execution_status, "valid")
        self.assertEqual(final.outcome.stop_reason, expected_stop_reason)
        self.assertEqual(final.outcome.training_eligibility, "structurally_eligible")
        turn = WriterTurnV2.from_dict(
            StoreArtifactReader(fixture.store).artifact(final.samples[0].turn_ref)
        )
        self.assertIs(turn.native_parse_failed, True)
        self.assertEqual(turn.message.content, raw_output)
        self.assertEqual(turn.message.calls, ())

        admission = self._audit(
            fixture.store,
            spec,
            decision,
            self.tokenizer,
            tokenizer_root=TOKENIZER_PATH,
            after=policy["behavior_policy_ref"],
        )
        require_training_admission(admission)
        self.assertEqual(admission.members[0]["status"], "admitted")

    def test_unparsed_truncated_and_malformed_calls_commit_and_audit_as_incomplete(self):
        self._assert_parse_failure_group(
            '<|tool_call>call:write_file{content:<|"|>A keeper waits by the',
            "decision_token_limit",
        )
        self._assert_parse_failure_group(
            "<|tool_call>call:write_file{content:A keeper,path:scene.txt"
            "<tool_call|><|tool_response>",
            "unparsed_tool_call",
        )

    def test_audit_refuses_both_directions_of_a_false_parse_failure_claim(self):
        cases = (
            ("A parseable answer.<turn|>", True),
            (
                "<|tool_call>call:write_file{content:A keeper,path:scene.txt"
                "<tool_call|><|tool_response>",
                False,
            ),
        )
        for raw_output, claim in cases:
            with self.subTest(claim=claim):
                fixture, spec, decision, policy = self._forced_native_group(
                    raw_output,
                    parse_failure_claim=claim,
                )
                admission = self._audit(
                    fixture.store,
                    spec,
                    decision,
                    self.tokenizer,
                    tokenizer_root=TOKENIZER_PATH,
                    after=policy["behavior_policy_ref"],
                )
                self.assertEqual(admission.members[0]["status"], "refused")
                self.assertEqual(admission.members[0]["failed_check"], "raw_output_and_message")

    def test_renderer_initial_context_rejects_only_its_lying_member(self):
        store, admission = self._audit_sample_lies()
        self._assert_durable_refusal(
            store, admission, {self.member_ids[0]: "renderer_initial_context"}
        )
        self.assertEqual(admission.members[3]["status"], "admitted")
        self.assertIsNone(admission.members[3]["failed_check"])
        self._assert_offline_refusal(store, self.spec.group_id, admission)

    def test_external_suffix_rejects_changed_suffix_with_prefix_intact(self):
        from writing_agent.native_audit import _member_turns
        from writing_agent.native_gemma import NativeGemmaRenderer
        from writing_agent.task_graph_gate import LineageGate, StoreArtifactReader
        from writing_agent.task_graph_group_records import GroupMemberResultV1
        from writing_agent.task_graph_native_contracts import NativeSamplingHistory

        store, admission = self._audit_sample_lies()
        self.assertEqual(admission.members[1]["failed_check"], "external_suffix_and_context_limit")
        self._assert_durable_refusal(
            store, admission, {self.member_ids[1]: "external_suffix_and_context_limit"}
        )
        self.assertEqual(admission.members[3]["status"], "admitted")
        reader = StoreArtifactReader(store)
        member_result = GroupMemberResultV1.from_dict(
            reader.artifact(self.decision.member_result_refs[1])
        )
        turns, _root_context = _member_turns(
            store,
            reader,
            LineageGate(),
            self.spec,
            member_result,
            self.member_ids[1],
        )
        previous, current = turns[:2]
        prefix = previous.input_ids + previous.generated_ids
        self.assertEqual(current.input_ids[: len(prefix)], prefix)
        current_context = reader.context(current.turn.context_revision_ref)
        expected_suffix = NativeGemmaRenderer(self.tokenizer, self.descriptors[0]).external_suffix(
            NativeSamplingHistory(previous.turn, previous.input_ids, previous.generated_ids),
            current_context.messages,
        )
        self.assertNotEqual(current.input_ids[len(prefix) :], expected_suffix)
        self._assert_offline_refusal(store, self.spec.group_id, admission)

    def test_generated_output_parses_to_the_committed_message(self):
        store, admission = self._audit_sample_lies()
        self.assertEqual(admission.members[2]["failed_check"], "raw_output_and_message")
        self._assert_durable_refusal(
            store, admission, {self.member_ids[2]: "raw_output_and_message"}
        )
        self.assertEqual(admission.members[3]["status"], "admitted")
        self._assert_offline_refusal(store, self.spec.group_id, admission)

    def test_adapter_hash_drift_is_refused_after_core_commits(self):
        store = self._clone_store(self.fixture.store, "adapter-hash-drift")
        changed_hash = "9" * 64
        self.assertNotEqual(changed_hash, self.policy["behavior_policy_ref"])
        admission = self._audit(
            store,
            self.spec,
            self.decision,
            self.tokenizer,
            tokenizer_root=TOKENIZER_PATH,
            after=changed_hash,
        )
        self._assert_durable_refusal(
            store, admission, {self.member_ids[3]: "policy_and_adapter_binding"}
        )
        self._assert_offline_refusal(store, self.spec.group_id, admission)

    def test_tokenizer_file_hash_mismatch_is_refused_offline_too(self):
        store = self._clone_store(self.bad_tokenizer_fixture.store, "tokenizer-file-lie")
        admission = self._audit(
            store,
            self.bad_tokenizer_spec,
            self.bad_tokenizer_decision,
            self.bad_tokenizer,
            tokenizer_root=self.bad_tokenizer_root,
            after=self.bad_tokenizer_policy["behavior_policy_ref"],
        )
        expected = {
            member.member_id: "tokenizer_files" for member in self.bad_tokenizer_spec.members
        }
        self._assert_durable_refusal(store, admission, expected)
        self._assert_offline_refusal(store, self.bad_tokenizer_spec.group_id, admission)

    def test_exported_token_layout_defense_is_a_core_unreachable_guard(self):
        from writing_agent.native_audit import _batch_member_matches, _member_turns
        from writing_agent.task_graph_gate import LineageGate, StoreArtifactReader
        from writing_agent.task_graph_group_records import GroupMemberResultV1
        from writing_agent.task_graph_training_export import export_training_batch

        reader = StoreArtifactReader(self.fixture.store)
        exported = export_training_batch(self.spec, self.decision, reader)
        for artifact in exported.artifacts:
            self.fixture.store.persist_artifact(artifact)
        member = exported.batch.members[3]
        result = GroupMemberResultV1.from_dict(reader.artifact(self.decision.member_result_refs[3]))
        turns, _root_context = _member_turns(
            self.fixture.store,
            reader,
            LineageGate(),
            self.spec,
            result,
            self.spec.members[3].member_id,
        )
        self.assertTrue(_batch_member_matches(reader, exported.batch, member, turns))
        mask_ref = member["env_mask_ref"]

        class CorruptMaskReader:
            def bytes_artifact(self, ref):
                value = reader.bytes_artifact(ref)
                if ref == mask_ref:
                    altered = bytearray(value)
                    first_generated = altered.index(1)
                    altered[first_generated] = 0
                    return bytes(altered)
                return value

        # Core export derives spans, IDs and masks together, so an invalid layout cannot
        # be requested through export_training_batch. Exercise the audit's defensive edge.
        self.assertFalse(_batch_member_matches(CorruptMaskReader(), exported.batch, member, turns))


if __name__ == "__main__":
    unittest.main()
