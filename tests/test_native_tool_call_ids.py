"""Core and admission regressions for native lineage-unique tool IDs."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_native_gemma import (
    TOKENIZER_PATH,
    TOKENIZER_REVISION,
    _tiny_peft_gemma,
    _tokenizer_file_hashes,
)
from writing_agent.native_gemma import NativeGemmaSampleBackend


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft")),
    "requires optional torch, transformers, and peft model dependencies",
)
class NativeToolCallIdIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoTokenizer

        cls.torch = torch
        cls.tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
        cls.model, cls.logit_hook = _tiny_peft_gemma(cls.tokenizer)

    @classmethod
    def tearDownClass(cls):
        cls.logit_hook.remove()

    def test_lineage_unique_call_ids_survive_core_derivation_and_audit(self):
        from collections.abc import Mapping
        from dataclasses import replace
        from types import SimpleNamespace

        import torch

        from scripts.task_graph_trace_check_support import with_native_tokenizer
        from tests.task_graph_rollout_fixtures import (
            _entry_fixture,
            build_rollout_fixture,
            make_gatherers,
        )
        from tests.test_task_graph_v2_writer import _native_policy
        from writing_agent.native_audit import audit_training_batch, require_training_admission
        from writing_agent.native_gemma import make_native_manifest_descriptors
        from writing_agent.task_graph_composition import RuntimeSession
        from writing_agent.task_graph_gate import StoreArtifactReader
        from writing_agent.task_graph_group import GroupCoordinatorV1
        from writing_agent.task_graph_group_records import GroupMemberResultV1
        from writing_agent.task_graph_local import (
            DeterministicEvaluator,
            LocalTextToolProvider,
            LocalWorkspaceEnvironment,
        )
        from writing_agent.task_graph_ports import RuntimeDependenciesV1
        from writing_agent.task_graph_rollout import RolloutDriver

        root = Path(__file__).resolve().parents[1]
        builder_spec = importlib.util.spec_from_file_location(
            "phase8_probe_task_builder", root / "configs/phase8/probe-tasks/build.py"
        )
        builder = importlib.util.module_from_spec(builder_spec)
        builder_spec.loader.exec_module(builder)
        probe = builder.load_probe_task(root / "configs/phase8/probe-tasks/t1-lighthouse.json")
        probe_entry = builder.build_admitted_entry(probe)
        revised_scene = (
            "At the lighthouse window, Mara watches the amber lantern settle as the boat enters "
            "the harbor. Low tide exposes silver stones. She checks the ropes, warms tea, and "
            "leaves the chart open for the returning crew."
        )
        outputs = {
            0: (
                '<|tool_call>call:ask_author{decision_ids:[<|"|>lantern_tone<|"|>],'
                'option_refs:[],proposals:[],question:<|"|>Which color should the lighthouse '
                'lantern be?<|"|>}<tool_call|><|tool_response>'
            ),
            1: (
                '<|tool_call>call:write_file{content:<|"|>A keeper waits for the boat.<|"|>,'
                'path:<|"|>scene.txt<|"|>}<tool_call|><|tool_response>'
            ),
            2: "The first scene is ready for review.<turn|>",
            3: (
                '<|tool_call>call:patch_file{new:<|"|>'
                + revised_scene
                + '<|"|>,old:<|"|>A keeper waits for the boat.<|"|>,'
                'path:<|"|>scene.txt<|"|>}<tool_call|><|tool_response>'
            ),
            4: "The revised scene is complete.<turn|>",
        }

        def run_group(
            parent: Path,
            entry,
            *,
            output_by_ordinal,
            mode,
            tamper_member=None,
            sequence=1,
        ):
            descriptors = make_native_manifest_descriptors(
                self.tokenizer,
                model_id="google/gemma-4-E2B-it",
                revision=TOKENIZER_REVISION,
                tokenizer_files_sha256=_tokenizer_file_hashes(),
                template_ref=entry.params.rendering["template_ref"],
                tool_schema_ref=entry.params.rendering["tool_schema_ref"],
                max_tokens_per_decision=512,
            )
            entry = with_native_tokenizer(entry, descriptors[1])
            fixture = build_rollout_fixture(
                parent / "store",
                mode=mode,
                entry_fixture=entry,
            )
            native = NativeGemmaSampleBackend(
                self.model,
                self.tokenizer,
                manifest_descriptors=descriptors,
            )

            class ScriptedBackend:
                descriptor = native.descriptor
                manifest_descriptors = native.manifest_descriptors
                member_ordinal = 0

                def sample(self, prepared):
                    if tamper_member == self.member_ordinal and prepared.native_history is not None:
                        from writing_agent.native_protocol import bind_native_tool_call_ids
                        from writing_agent.task_graph_calls import intake_message
                        from writing_agent.task_graph_native_contracts import NativeSamplingHistory
                        from writing_agent.task_graph_wire import decode_canonical_value

                        history = prepared.native_history
                        turn = history.turn
                        message = {
                            "role": "assistant",
                            "content": decode_canonical_value(turn.message.content),
                            "tool_calls": [
                                decode_canonical_value(item["value"]) for item in turn.message.calls
                            ],
                        }
                        message = bind_native_tool_call_ids(message, turn.action_id)
                        corrected_turn = replace(turn, message=intake_message(message))
                        prepared = replace(
                            prepared,
                            native_history=NativeSamplingHistory(
                                corrected_turn,
                                history.input_token_ids,
                                history.generated_token_ids,
                            ),
                        )
                    raw = output_by_ordinal[prepared.decision_ordinal]
                    generated_ids = tuple(self_tokenizer.encode(raw, add_special_tokens=False))

                    def generate(_model, inputs, generation, *, seed):
                        del seed
                        sequence = inputs["input_ids"]
                        processors = generation["logits_processor"]
                        for token_id in generated_ids:
                            scores = torch.full(
                                (1, self_tokenizer.vocab_size), -1000.0, dtype=torch.float32
                            )
                            scores[0, token_id] = 1000.0
                            processors(sequence, scores)
                            next_token = torch.tensor([[token_id]], dtype=torch.long)
                            sequence = torch.cat((sequence, next_token), dim=1)
                        return SimpleNamespace(sequences=sequence)

                    with patch("writing_agent.native_gemma.generate_with_seed", generate):
                        result = native.sample(prepared)
                    if (
                        tamper_member == self.member_ordinal
                        and prepared.decision_ordinal == 0
                        and result.message.get("tool_calls")
                    ):
                        message = dict(result.message)
                        message["tool_calls"] = [dict(call) for call in message["tool_calls"]]
                        message["tool_calls"][0]["id"] = "tampered-call-id"
                        result = replace(result, message=message)
                    return result

            self_tokenizer = self.tokenizer
            backend = ScriptedBackend()
            tools = LocalTextToolProvider()
            dependencies = RuntimeDependenciesV1(
                backend, LocalWorkspaceEnvironment(tools), tools, DeterministicEvaluator()
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
                group_sequence=sequence,
                member_count=2,
                runner_mode="real",
                training_mode="native",
            )
            for ordinal, member in enumerate(spec.members):
                backend.member_ordinal = ordinal
                runtime = coordinator.start(spec, ordinal, policy=policy)
                result = RolloutDriver(fixture.env, make_gatherers(fixture, sampler=backend)).run(
                    runtime, max_steps=80
                )
                self.assertEqual(result.directive.kind, "done")
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
            return fixture, spec, coordinator, coordinator.finalize(spec), policy

        with tempfile.TemporaryDirectory(prefix="native-call-ids-") as temporary:
            root = Path(temporary)
            good, good_spec, good_coordinator, good_decision, good_policy = run_group(
                root / "good",
                probe_entry,
                output_by_ordinal=outputs,
                mode="slice",
            )
            self.assertIn(good_decision.status, {"ready", "tie"})
            self.assertEqual(good_decision.status, "tie")
            initial = good.runtime.state.files
            reader = StoreArtifactReader(good.store)
            report_events = []
            for member_ordinal, ref in enumerate(good_decision.member_result_refs):
                result = GroupMemberResultV1.from_dict(reader.artifact(ref))
                view = good.env.gate.view(good.store, result.final_checkpoint_id)
                self.assertEqual(view.outcome.task_status, "complete")
                self.assertNotEqual(view.state.files["scene.txt"], initial["scene.txt"])
                self.assertEqual(view.state.files["scene.txt"], revised_scene)
                score = good_coordinator.reward_of(result, view=view)
                self.assertAlmostEqual(float(score), 0.8)
                calls = [
                    call
                    for message in view.context.messages
                    for call in message.content
                    if isinstance(call, Mapping) and call.get("type") == "tool_call"
                ]
                self.assertEqual(len(calls), 3)
                self.assertEqual(
                    [call["id"] for call in calls],
                    [f"{result.member_id}:call:{index}:0" for index in (0, 1, 3)],
                )
                for message in view.context.messages:
                    turns_calls = [
                        call
                        for call in message.content
                        if isinstance(call, Mapping) and call.get("type") == "tool_call"
                    ]
                    if turns_calls:
                        action_id = message.origin
                        report_events.append(
                            {
                                "member_ordinal": member_ordinal,
                                "decision_ordinal": int(action_id.rsplit(":action:", 1)[1]),
                                "action_id": action_id,
                                "termination": None,
                                "tool_calls": [
                                    {
                                        "id": call["id"],
                                        "name": call["name"],
                                        "result": "missing",
                                    }
                                    for call in turns_calls
                                ],
                            }
                        )
            from scripts.task_graph_trace_check_support import (
                member_summaries,
                tool_result_protocol_errors,
            )

            reported_members = member_summaries(
                good.store,
                good_coordinator,
                good_spec,
                good_decision,
                SimpleNamespace(events=report_events),
            )
            self.assertTrue(
                all(member["final_files_differ_from_initial"] for member in reported_members)
            )
            self.assertTrue(
                all("scene.txt" in member["changed_paths"] for member in reported_members)
            )
            self.assertEqual(
                [call["name"] for event in report_events for call in event["tool_calls"]],
                ["ask_author", "write_file", "patch_file"] * 2,
            )
            self.assertEqual(
                [call["result"] for event in report_events for call in event["tool_calls"]],
                ["ok"] * 6,
            )
            self.assertEqual(tool_result_protocol_errors(report_events), [])
            admission = audit_training_batch(
                good.store,
                good_spec,
                good_decision,
                self.tokenizer,
                adapter_hash_before=good_policy["behavior_policy_ref"],
                adapter_hash_after=good_policy["behavior_policy_ref"],
                tokenizer_root=TOKENIZER_PATH,
            )
            self.assertEqual([item["status"] for item in admission.members], ["admitted"] * 2)
            require_training_admission(admission)

            one_turn_entry = _entry_fixture("none", "deterministic-file-v1")
            single_call = {
                0: (
                    '<|tool_call>call:write_file{content:<|"|>A changed draft proves the tool ran.'
                    '<|"|>,path:<|"|>draft.txt<|"|>}<tool_call|><|tool_response>'
                ),
                1: "The changed draft is complete.<turn|>",
            }
            bad, bad_spec, _bad_coordinator, bad_decision, bad_policy = run_group(
                root / "tampered",
                one_turn_entry,
                output_by_ordinal=single_call,
                mode="none",
                tamper_member=1,
                sequence=2,
            )
            bad_admission = audit_training_batch(
                bad.store,
                bad_spec,
                bad_decision,
                self.tokenizer,
                adapter_hash_before=bad_policy["behavior_policy_ref"],
                adapter_hash_after=bad_policy["behavior_policy_ref"],
                tokenizer_root=TOKENIZER_PATH,
            )
            self.assertEqual(
                [(item["status"], item["failed_check"]) for item in bad_admission.members],
                [("admitted", None), ("refused", "external_suffix_and_context_limit")],
            )


if __name__ == "__main__":
    unittest.main()
