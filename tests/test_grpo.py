"""Token/mask and group availability contracts; no downloaded or GPU models."""

import copy
import errno
import json
import tempfile
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from writing_agent.agent import SYSTEM_PROMPT, run_agent
from writing_agent.catalog import fingerprint
from writing_agent.grpo import (
    GRPOSettings,
    inspect_grpo,
    seal_directory,
    train_grpo,
    verify_checkpoint,
)
from writing_agent.grpo_identity import admission_identity, base_tensor_identity
from writing_agent.grpo_rollout import (
    GroupPending,
    NativeRolloutBackend,
    ProtocolError,
    RolloutGroups,
    native_suffix,
    verify_tokens,
)
from writing_agent.reward import Reward, rollout_reward
from writing_agent.workspace import Workspace

REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


def task():
    return {
        "id": "fixture",
        "role": "train",
        "source_groups": ["engineered-fixtures"],
        "labels": {"checks": [{"id": "private-sentinel"}]},
        "visible": {
            "brief": "write then revise",
            "initial_files": {"note.txt": "before"},
            "followups": [],
            "tools": ["write_file", "read_file"],
            "prose": [],
            "budgets": {
                "max_steps": 4,
                "max_tool_calls": 4,
                "max_read_tokens": 256,
                "max_total_bytes": 4096,
            },
        },
    }


class IntegrityTests(unittest.TestCase):
    def test_objective_default_admission_and_identity(self):
        self.assertEqual(GRPOSettings().loss_type, "grpo")
        spec = {"id": "fixture-v1", "config": {}, "mode": "mechanical-only-smoke"}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "absent"
            plans = []
            for loss_type in ("grpo", "dapo"):
                plan = inspect_grpo(
                    [task()],
                    output,
                    settings=GRPOSettings(revision=REVISION, loss_type=loss_type),
                    reward_spec=spec,
                    admission={"mode": "engineered-fixture", "label": "test-only"},
                )
                self.assertEqual(plan["settings"]["loss_type"], loss_type)
                plans.append(fingerprint(plan))
            self.assertNotEqual(*plans)
            for loss_type in ("bnpo", "DAPO", "", None, True, [], {}):
                with (
                    self.subTest(loss_type=loss_type),
                    self.assertRaisesRegex(ValueError, "Loss type"),
                ):
                    train_grpo(
                        [task()],
                        output,
                        settings=GRPOSettings(revision=REVISION, loss_type=loss_type),
                        reward_spec=spec,
                        admission={"mode": "engineered-fixture", "label": "test-only"},
                        execute=True,
                    )
            self.assertFalse(output.exists())

    def test_tie_policy_is_explicit_and_identity_bound(self):
        settings = GRPOSettings(revision=REVISION)
        self.assertEqual(settings.tie_policy, "halt")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "absent"
            common = dict(
                tasks=[task()],
                output=output,
                reward_spec={"id": "fixture-v1", "config": {}, "mode": "mechanical-only-smoke"},
                admission={"mode": "engineered-fixture", "label": "test-only"},
            )
            halted = inspect_grpo(settings=settings, **common)
            continued = inspect_grpo(settings=replace(settings, tie_policy="continue"), **common)
            self.assertEqual(continued["settings"]["tie_policy"], "continue")
            self.assertNotEqual(fingerprint(halted), fingerprint(continued))
            for policy in ("skip", "CONTINUE", "", None, True, [], {}):
                with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "Tie policy"):
                    train_grpo(
                        settings=replace(settings, tie_policy=policy), execute=True, **common
                    )
            self.assertFalse(output.exists())

    def test_microbatch_preserves_one_group_per_optimizer_step(self):
        for size, accumulation in ((None, 1), (1, 4), (2, 2), (4, 1)):
            settings = GRPOSettings(revision=REVISION, group_size=4, microbatch_size=size)
            settings.validate()
            self.assertEqual(settings.gradient_accumulation_steps, accumulation)
        for size in (0, -1, 3, 5, True, 1.0):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "microbatch"):
                GRPOSettings(revision=REVISION, group_size=4, microbatch_size=size).validate()

    def test_inspect_rejects_evaluation_and_needs_immutable_revision_without_execution(self):
        settings = GRPOSettings(revision=REVISION)
        spec = {"id": "fixture-v1", "config": {}, "mode": "mechanical-only-smoke"}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "absent"
            plan = inspect_grpo(
                [task()],
                output,
                settings=settings,
                reward_spec=spec,
                admission={"mode": "engineered-fixture", "label": "test-only"},
            )
            self.assertFalse(output.exists())
            self.assertEqual(plan["beta"], 0)
            bad = task()
            bad["role"] = "development"
            with self.assertRaisesRegex(ValueError, "train-role"):
                inspect_grpo(
                    [bad],
                    output,
                    settings=settings,
                    reward_spec=spec,
                    admission={"mode": "engineered-fixture", "label": "test-only"},
                )
            with self.assertRaisesRegex(ValueError, "immutable"):
                inspect_grpo(
                    [task()],
                    output,
                    settings=GRPOSettings(),
                    reward_spec=spec,
                    admission={"mode": "engineered-fixture", "label": "test-only"},
                )

    def test_mask_and_generation_prefix_integrity(self):
        evidence = {
            "prompt_ids": [1],
            "completion_ids": [2, 3, 4],
            "env_mask": [1, 0, 1],
            "boundaries": [
                {"input_ids": [1], "completion_offset": 0, "output_ids": [2]},
                {"input_ids": [1, 2, 3], "completion_offset": 2, "output_ids": [4]},
            ],
        }
        verify_tokens(evidence)
        evidence["boundaries"][1]["input_ids"] = [1, 5, 3]
        with self.assertRaisesRegex(ProtocolError, "training prefix"):
            verify_tokens(evidence)
        evidence["env_mask"].pop()
        with self.assertRaisesRegex(ProtocolError, "ledger"):
            verify_tokens(evidence)

    def test_resume_requires_complete_hash_bound_optimizer_scheduler_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            names = [
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "adapter_config.json",
                "adapter_model.safetensors",
            ]
            for name in names:
                (checkpoint / name).write_text("fixture")
            (checkpoint / "trainer_state.json").write_text('{"global_step": 1}')
            seal_directory(checkpoint, "identity", "trainer")
            self.assertEqual(verify_checkpoint(checkpoint, "identity"), 1)
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_checkpoint(checkpoint, "other-identity")
            (checkpoint / "optimizer.pt").write_text("truncated")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_checkpoint(checkpoint, "identity")
            (checkpoint / "complete.json").write_text("{")
            with self.assertRaisesRegex(ValueError, "truncated"):
                verify_checkpoint(checkpoint, "identity")
            seal_directory(checkpoint, "identity", "inference-adapter")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_checkpoint(checkpoint, "identity")


class AdmissionTests(unittest.TestCase):
    def admission(self, selected):
        source = {
            "id": "source",
            "work_id": "work",
            "role": "train",
            "provenance": "human",
            "terms": "fixture permission",
            "text": "source passage",
            "sha256": fingerprint("source passage"),
        }
        selected.update(
            source_ids=["source"],
            source_groups=["source"],
            visible_hash=fingerprint(selected["visible"]),
            labels_hash=fingerprint(selected["labels"]),
        )
        return {
            "mode": "production",
            "catalog": [source],
            "catalog_hash": fingerprint([source]),
            "excluded_source_groups": [],
            "release_manifest": {
                "catalog_hash": fingerprint([source]),
                "scenarios": [
                    {k: v for k, v in selected.items() if k not in {"visible", "labels"}}
                ],
            },
        }

    def test_production_rejects_forged_roles_groups_missing_sources_and_catalog_changes(self):
        selected = task()
        admission = self.admission(selected)
        valid = admission_identity([selected], admission)
        self.assertEqual(valid["connected_groups"], {"source": "source"})
        self.assertEqual(valid["selected_task_ids"], ["fixture"])
        for variant in (
            "missing-id",
            "unknown-id",
            "group",
            "role",
            "hash",
            "excluded",
            "connected-holdout",
        ):
            with self.subTest(variant=variant):
                changed, data = copy.deepcopy(selected), copy.deepcopy(admission)
                if variant == "missing-id":
                    changed.pop("source_ids")
                elif variant == "unknown-id":
                    changed["source_ids"] = ["invented"]
                elif variant == "group":
                    changed["source_groups"] = ["invented"]
                elif variant == "role":
                    data["catalog"][0]["role"] = "final_eval"
                elif variant == "hash":
                    data["catalog_hash"] = "not-the-catalog-hash"
                elif variant == "excluded":
                    data["excluded_source_groups"] = ["work"]
                else:
                    holdout = {**data["catalog"][0], "id": "holdout", "role": "development"}
                    data["catalog"].append(holdout)
                if variant != "hash":
                    data["catalog_hash"] = fingerprint(data["catalog"])
                    data["release_manifest"]["catalog_hash"] = data["catalog_hash"]
                # Even a fabricated matching task manifest cannot override catalog lineage.
                data["release_manifest"]["scenarios"] = [
                    {k: v for k, v in changed.items() if k not in {"visible", "labels"}}
                ]
                with self.assertRaises(ValueError):
                    admission_identity([changed], data)
        with self.assertRaisesRegex(ValueError, "Explicit"):
            admission_identity([task()], None)
        with self.assertRaisesRegex(ValueError, "Malformed"):
            admission_identity([task()], {"mode": "production"})


class ResumePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch

            from scripts.smoke_grpo_cpu import ToyBackend, tiny_model, toy_reward
        except ImportError as exc:
            raise unittest.SkipTest("Optional CPU training dependencies") from exc
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name) / "run"
        selected = task()
        selected["visible"].update(brief="word3 word4", tools=[], followups=["word5"])
        cls.common = dict(
            tasks=[selected],
            settings=GRPOSettings(
                model_id="caller-owned/tiny-random-llama",
                revision="a" * 40,
                max_steps=2,
                group_size=4,
                context_tokens=128,
                max_tokens=4,
                max_generated_tokens=8,
                lora_rank=2,
                learning_rate=0.001,
            ),
            admission={"mode": "engineered-fixture", "label": "preflight tests only"},
            reward_spec={"id": "toy-word-index-v1", "mode": "mechanical-only-smoke", "config": {}},
            reward_callback=toy_reward,
            execute=True,
            backend_factory=ToyBackend,
            runtime_identity={"model": "fixed-tiny-base", "backend": "toy-v1"},
        )
        model, tokenizer = tiny_model()
        cls.partial = train_grpo(
            output=cls.output, model=model, tokenizer=tokenizer, stop_after_steps=1, **cls.common
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def snapshot(self, model, tokenizer):
        import pickle
        import random

        import numpy as np
        import torch

        return {
            "tensors": base_tensor_identity(model),
            "modules": [
                (n, type(m).__name__, m.training, getattr(m, "p", None))
                for n, m in model.named_modules()
            ],
            "grad_flags": [(n, p.requires_grad) for n, p in model.named_parameters()],
            "config": copy.deepcopy(model.config.to_dict()),
            "generation": copy.deepcopy(model.generation_config.to_dict()),
            "tokenizer": tokenizer.backend_tokenizer.to_str(),
            "padding": tokenizer.padding_side,
            "special": copy.deepcopy(tokenizer.special_tokens_map),
            "rng": pickle.dumps(
                (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes())
            ),
        }

    def test_rejected_resume_does_not_mutate_model_tokenizer_or_rng(self):
        import torch

        from scripts.smoke_grpo_cpu import tiny_model

        for variant in (
            "parameter",
            "buffer",
            "missing-manifest",
            "missing-checkpoint",
            "changed-checkpoint",
        ):
            with self.subTest(variant=variant):
                model, tokenizer = tiny_model()
                output, checkpoint = self.output, Path(self.partial["checkpoint"])
                if variant == "parameter":
                    with torch.no_grad():
                        next(model.parameters()).view(-1)[0].add_(1.0)
                elif variant == "buffer":
                    with torch.no_grad():
                        next(model.buffers()).view(-1)[0].add_(1.0)
                elif variant == "missing-manifest":
                    output = self.output.parent / "absent"
                    checkpoint = output / "checkpoint-1"
                elif variant == "missing-checkpoint":
                    checkpoint = output / "checkpoint-999"
                damaged = checkpoint / "optimizer.pt"
                original = damaged.read_bytes() if variant == "changed-checkpoint" else None
                if original is not None:
                    damaged.write_bytes(b"truncated")
                before = self.snapshot(model, tokenizer)
                try:
                    expected = (
                        "identity changed"
                        if variant in {"parameter", "buffer"}
                        else "manifest"
                        if variant == "missing-manifest"
                        else "checkpoint"
                    )
                    with self.assertRaisesRegex(ValueError, expected):
                        train_grpo(
                            output=output,
                            model=model,
                            tokenizer=tokenizer,
                            resume_from_checkpoint=checkpoint,
                            **self.common,
                        )
                finally:
                    if original is not None:
                        damaged.write_bytes(original)
                self.assertEqual(before, self.snapshot(model, tokenizer))
                self.assertFalse(any("lora" in n for n, _ in model.named_parameters()))

    def test_partial_later_checkpoint_is_quarantined_and_actual_resume_succeeds(self):
        from scripts.smoke_grpo_cpu import tiny_model

        partial = self.output / "checkpoint-2"
        partial.mkdir()
        (partial / "optimizer.pt").write_bytes(b"truncated optimizer evidence")
        (partial / "complete.json").write_text("{")
        model, tokenizer = tiny_model()
        resumed = train_grpo(
            output=self.output,
            model=model,
            tokenizer=tokenizer,
            resume_from_checkpoint=Path(self.partial["checkpoint"]),
            **self.common,
        )
        self.assertEqual(resumed["global_step"], 2)
        quarantine = Path(resumed["quarantined"][0])
        self.assertEqual(
            (quarantine / "optimizer.pt").read_bytes(), b"truncated optimizer evidence"
        )
        self.assertEqual((quarantine / "complete.json").read_text(), "{")
        self.assertEqual(verify_checkpoint(partial, resumed["identity"]), 2)
        model, tokenizer = tiny_model()
        before = self.snapshot(model, tokenizer)
        with self.assertRaisesRegex(ValueError, "latest complete"):
            train_grpo(
                output=self.output,
                model=model,
                tokenizer=tokenizer,
                resume_from_checkpoint=Path(self.partial["checkpoint"]),
                **self.common,
            )
        self.assertEqual(before, self.snapshot(model, tokenizer))


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise unittest.SkipTest("Optional inference dependencies") from exc
        cls.torch = torch
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                "google/gemma-4-E2B-it",
                revision=REVISION,
                local_files_only=True,
            )
        except OSError as exc:
            raise unittest.SkipTest("Pinned tokenizer unavailable") from exc

    def backend(self, outputs, *, seed=42, context=4096, total=512):
        from transformers import GenerationConfig

        torch, tokenizer = self.torch, self.tokenizer

        class ScriptedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.device = torch.device("cpu")
                self.generation_config = GenerationConfig(eos_token_id=[1, 106, 50])
                self.outputs = iter(outputs)
                self.inputs = []

            def generate(self, input_ids, **kwargs):
                self.inputs.append(input_ids[0].tolist())
                text = next(self.outputs)
                if isinstance(text, BaseException):
                    raise text
                ids = tokenizer.encode(text, add_special_tokens=False)
                return torch.tensor([input_ids[0].tolist() + ids])

        return NativeRolloutBackend(
            ScriptedModel(),
            tokenizer,
            {
                "protocol": "gemma-native-v1",
                "prompt_format": "chat",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "seed": seed,
                "enable_thinking": True,
                "max_tokens": 128,
                "max_generated_tokens": total,
                "context_tokens": context,
            },
        )

    def test_native_tool_multiple_responses_followup_preserves_actual_actions(self):
        first = (
            "<|channel>thought\nKEEP_OLD_REASONING\n<channel|>"
            '<|tool_call>call:write_file{path:<|"|>note.txt<|"|>,content:<|"|>after<|"|>}<tool_call|>'
            '<|tool_call>call:read_file{path:<|"|>note.txt<|"|>}<tool_call|><|tool_response>'
        )
        outputs = [
            first,
            "post-tool thought\n<channel|>First reply<turn|>",
            "Revised reply<turn|>",
            "Final reply<turn|>",
        ]
        backend = self.backend(outputs)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            workspace.write_file("note.txt", "before")
            result = run_agent(
                backend,
                workspace,
                [{"role": "user", "content": "write then revise"}],
                tools=["write_file", "read_file"],
                followups=["Revise.", "Again."],
                max_steps=4,
            )
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(len(result["turns"]), 3)
            self.assertEqual(result["turns"][0]["snapshot"]["note.txt"], "after")
        tokens = backend.evidence()
        selected = [
            t for t, mask in zip(tokens["completion_ids"], tokens["env_mask"], strict=True) if mask
        ]
        expected = sum([self.tokenizer.encode(s, add_special_tokens=False) for s in outputs], [])
        self.assertEqual(selected, expected)
        final_input = self.tokenizer.decode(tokens["boundaries"][-1]["input_ids"])
        self.assertIn(first, final_input)
        self.assertIn("KEEP_OLD_REASONING", final_input)
        self.assertIn('path:<|"|>note.txt<|"|>,content:', final_input)
        self.assertEqual(
            tokens["env_mask"][len(self.tokenizer.encode(first, add_special_tokens=False)) - 1], 1
        )
        environment = self.tokenizer.decode(
            [
                t
                for t, mask in zip(tokens["completion_ids"], tokens["env_mask"], strict=True)
                if not mask
            ]
        )
        self.assertIn("response:write_file", environment)
        self.assertIn("response:read_file", environment)
        self.assertIn("Revise.", environment)
        self.assertNotIn("KEEP_OLD_REASONING", environment)

    def test_tool_call_ending_eos_is_a_scored_candidate_failure(self):
        output = (
            '<|tool_call>call:write_file{path:<|"|>note.txt<|"|>,'
            'content:<|"|>after<|"|>}<tool_call|><eos>'
        )
        backend = self.backend([output])
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            workspace.write_file("note.txt", "before")
            result = run_agent(
                backend,
                workspace,
                [{"role": "user", "content": "write then revise"}],
                tools=["write_file"],
            )
            self.assertEqual(workspace.read_file("note.txt"), "before")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["failure_class"], "candidate_invalid")
        self.assertEqual(backend.failure, "candidate_invalid")
        self.assertEqual(result["tool_calls"], 0)
        tokens = backend.evidence()
        self.assertEqual(
            tokens["completion_ids"], self.tokenizer.encode(output, add_special_tokens=False)
        )
        self.assertTrue(all(tokens["env_mask"]))

    def test_final_eos_keeps_sampled_action_and_appends_masked_followup(self):
        first = "Finished.<eos>"
        second = "Revision<turn|>"
        backend = self.backend([first, second])
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(
                backend,
                Workspace(Path(tmp)),
                [{"role": "user", "content": "write then revise"}],
                tools=[],
                followups=["Revise."],
            )
        self.assertEqual(result["status"], "completed", result)
        self.assertIsNone(result["failure_class"])
        self.assertEqual([turn["output"] for turn in result["turns"]], ["Finished.", "Revision"])
        tokens = backend.evidence()
        verify_tokens(tokens)
        sampled = [
            token
            for token, mask in zip(tokens["completion_ids"], tokens["env_mask"], strict=True)
            if mask
        ]
        self.assertEqual(sampled, self.tokenizer.encode(first + second, add_special_tokens=False))
        self.assertEqual(len(tokens["boundaries"]), 2)
        next_input = backend.model.inputs[1]
        self.assertEqual(next_input, tokens["boundaries"][1]["input_ids"])
        self.assertIn(
            "<eos>\n<|turn>user\nRevise.<turn|>\n<|turn>model\n",
            self.tokenizer.decode(next_input, skip_special_tokens=False),
        )
        suffix = self.tokenizer.decode(
            [
                token
                for token, mask in zip(tokens["completion_ids"], tokens["env_mask"], strict=True)
                if not mask
            ],
            skip_special_tokens=False,
        )
        self.assertEqual(suffix, "\n<|turn>user\nRevise.<turn|>\n<|turn>model\n")

        # EOS also remains a valid last response when no follow-up is scheduled.
        standalone = self.backend([first])
        with tempfile.TemporaryDirectory() as tmp:
            completed = run_agent(
                standalone, Workspace(Path(tmp)), [{"role": "user", "content": "hi"}], tools=[]
            )
        self.assertEqual(completed["status"], "completed")

    def test_mixed_content_tool_call_preserves_raw_actions_and_external_suffix(self):
        first = (
            'A note before the edit. <|tool_call>call:write_file{path:<|"|>note.txt<|"|>,'
            'content:<|"|>after<|"|>}<tool_call|><|tool_response>'
        )
        backend = self.backend([first, "Done thinking<channel|>Done.<turn|>"])
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp))
            workspace.write_file("note.txt", "before")
            result = run_agent(
                backend,
                workspace,
                [{"role": "user", "content": "write then revise"}],
                tools=["write_file"],
                max_steps=2,
            )
            self.assertEqual(workspace.read_file("note.txt"), "after")
        self.assertEqual(result["status"], "completed", result)
        tokens = backend.evidence()
        self.assertEqual(
            [
                t
                for t, mask in zip(tokens["completion_ids"], tokens["env_mask"], strict=True)
                if mask
            ],
            self.tokenizer.encode(
                first + "Done thinking<channel|>Done.<turn|>", add_special_tokens=False
            ),
        )
        self.assertEqual(
            tokens["boundaries"][1]["input_ids"],
            tokens["prompt_ids"]
            + tokens["completion_ids"][: tokens["boundaries"][1]["completion_offset"]],
        )
        self.assertIn(
            "response:write_file",
            self.tokenizer.decode(
                [
                    t
                    for t, mask in zip(tokens["completion_ids"], tokens["env_mask"], strict=True)
                    if not mask
                ]
            ),
        )

    def test_unrecognized_final_continuation_and_context_are_explicit(self):
        with self.assertRaisesRegex(ProtocolError, "stopping boundary"):
            native_suffix(
                self.tokenizer,
                {"role": "assistant", "content": "done"},
                [{"role": "user", "content": "again"}],
                [self.tokenizer.convert_tokens_to_ids("<|tool_response>")],
                thinking=True,
            )
        with self.assertRaisesRegex(ProtocolError, "stopping boundary"):
            native_suffix(
                self.tokenizer,
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "read_file", "arguments": {}}}],
                },
                [{"role": "tool", "tool_call_id": "call_0", "content": "result"}],
                [self.tokenizer.convert_tokens_to_ids("<eos>")],
                thinking=True,
            )
        backend = self.backend(["Hi<turn|>"], context=16)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_agent(
                backend, Workspace(Path(tmp)), [{"role": "user", "content": "hi"}], tools=[]
            )
        self.assertEqual(result["status"], "error")
        self.assertIn("not truncated", result["error"])
        self.assertFalse(backend.completion_ids)

    def test_groups_keep_all_attempts_and_pending_is_not_zero(self):
        trainer = SimpleNamespace(
            model=None, processing_class=None, state=SimpleNamespace(global_step=0)
        )
        for policy, variant in product(
            ("halt", "continue"),
            ("mixed", "tie", "judge-error", "infrastructure", "candidate-invalid", "malformed"),
        ):
            settings = GRPOSettings(revision=REVISION, max_generated_tokens=512, tie_policy=policy)
            with self.subTest(policy=policy, variant=variant), tempfile.TemporaryDirectory() as tmp:

                def factory(model, tokenizer, seed, variant=variant):
                    output = (
                        RuntimeError("transport unavailable")
                        if variant == "infrastructure" and seed == 42
                        else ""
                        if variant == "candidate-invalid" and seed == 42
                        else "<|tool_call>broken<turn|>"
                        if variant == "malformed" and seed == 42
                        else "Hi<turn|>"
                    )
                    return self.backend([output], seed=seed)

                def reward(scenario, result, variant=variant):
                    if variant == "judge-error" and result["seed"] == 42:
                        return rollout_reward({"quality": {"status": "unavailable"}}, [])
                    value = (
                        0.0
                        if result["status"] != "completed"
                        else (1.0 if variant == "tie" else ((result["seed"] - 42) // 32) % 2)
                    )
                    return Reward("ok", value, components={"mechanical_fixture": value})

                groups = RolloutGroups(
                    [task()], settings, Path(tmp), reward, factory, SYSTEM_PROMPT
                )
                pending = variant in {"judge-error", "infrastructure", "candidate-invalid"} or (
                    variant == "tie" and policy == "halt"
                )
                if pending:
                    with self.assertRaises(GroupPending):
                        groups(["fixture"] * 2, trainer)
                else:
                    output = groups(["fixture"] * 2, trainer)
                    self.assertEqual(
                        output["rollout_rewards"], [1.0, 1.0] if variant == "tie" else [0.0, 1.0]
                    )
                self.assertEqual(bool(list(Path(tmp).glob("groups/*/complete.json"))), not pending)
                self.assertEqual(bool(list(Path(tmp).glob("groups/*/stopped.json"))), pending)
                attempts = sorted(Path(tmp).glob("groups/*/attempt-*"))
                self.assertEqual(len(attempts), 2)
                for attempt in attempts:
                    self.assertTrue((attempt / "result.json").is_file())
                    self.assertTrue((attempt / "tokens.json").is_file())
                    self.assertEqual((attempt / "workspace/note.txt").read_text(), "before")
                    events = (attempt / "trace.jsonl").read_text()
                    self.assertNotIn("private-sentinel", events)
                    self.assertTrue((attempt / "reward.json").is_file())
                stats = json.loads(next(Path(tmp).glob("groups/*/group.json")).read_text())
                self.assertEqual(stats["tie_policy"], policy)
                if variant == "tie":
                    self.assertTrue(stats["zero_variance"])
                    self.assertEqual(stats["trl_advantages_estimate"], [0.0, 0.0])
                    self.assertNotIn("trl_advantages", stats)
                self.assertEqual(
                    stats["status"],
                    "pending"
                    if variant in {"judge-error", "infrastructure", "candidate-invalid"}
                    else "ok",
                )

    def test_sampled_tool_protocol_shapes_score_without_resampling(self):
        trainer = SimpleNamespace(
            model=None, processing_class=None, state=SimpleNamespace(global_step=0)
        )
        tool_eos = (
            '<|tool_call>call:write_file{path:<|"|>note.txt<|"|>,'
            'content:<|"|>after<|"|>}<tool_call|><eos>'
        )
        mixed = (
            'Planning text <|tool_call>call:write_file{path:<|"|>note.txt<|"|>,'
            'content:<|"|>after<|"|>}<tool_call|><|tool_response>'
        )
        with tempfile.TemporaryDirectory() as tmp:
            groups = RolloutGroups(
                [task()],
                GRPOSettings(revision=REVISION, group_size=4, max_generated_tokens=512),
                Path(tmp),
                lambda scenario, result: Reward(
                    "ok", 0.0 if result["status"] != "completed" else 1.0
                ),
                lambda model, tokenizer, seed: self.backend(
                    [tool_eos]
                    if seed == 42
                    else [mixed, "Thinking<channel|>Done.<turn|>"]
                    if seed == 74
                    else ["Thinking<channel|>Done.<turn|>"],
                    seed=seed,
                ),
                SYSTEM_PROMPT,
            )
            output = groups(["fixture"] * 4, trainer)
            self.assertEqual(output["rollout_rewards"], [0.0, 1.0, 1.0, 1.0])
            attempts = sorted(Path(tmp).glob("groups/*/attempt-*"))
            self.assertEqual(len(attempts), 4)
            self.assertTrue(list(Path(tmp).glob("groups/*/complete.json")))
            for index, attempt in enumerate(attempts):
                result = json.loads((attempt / "result.json").read_text())
                reward = json.loads((attempt / "reward.json").read_text())
                tokens = json.loads((attempt / "tokens.json").read_text())
                verify_tokens(tokens)
                self.assertEqual(reward["status"], "ok")
                if index == 0:
                    self.assertEqual(result["failure_class"], "candidate_invalid")
                    self.assertEqual(reward["value"], 0.0)
                    self.assertEqual((attempt / "workspace/note.txt").read_text(), "before")
                if index == 1:
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual((attempt / "workspace/note.txt").read_text(), "after")

    def test_sampled_final_eos_followup_continues_whole_group_without_retry(self):
        trainer = SimpleNamespace(
            model=None, processing_class=None, state=SimpleNamespace(global_step=0)
        )
        scenario = task()
        scenario["visible"]["followups"] = ["Revise."]
        eos = "Finished<eos>"
        revision = "Revised<turn|>"
        with tempfile.TemporaryDirectory() as tmp:
            groups = RolloutGroups(
                [scenario],
                GRPOSettings(
                    revision=REVISION,
                    group_size=4,
                    max_generated_tokens=512,
                    tie_policy="continue",
                ),
                Path(tmp),
                lambda task, result: Reward("ok", 0.0 if result["status"] != "completed" else 1.0),
                lambda model, tokenizer, seed: self.backend(
                    [eos, revision] if seed == 42 else ["Draft<turn|>", revision], seed=seed
                ),
                SYSTEM_PROMPT,
            )
            values = groups(["fixture"] * 4, trainer)["rollout_rewards"]
            self.assertEqual(values, [1.0] * 4)
            self.assertTrue(list(Path(tmp).glob("groups/*/complete.json")))
            attempts = sorted(Path(tmp).glob("groups/*/attempt-*"))
            self.assertEqual(len(attempts), 4)
            first_result = json.loads((attempts[0] / "result.json").read_text())
            tokens = json.loads((attempts[0] / "tokens.json").read_text())
            self.assertEqual(first_result["status"], "completed")
            self.assertIsNone(first_result["failure_class"])
            self.assertEqual(len(tokens["boundaries"]), 2)
            verify_tokens(tokens)

    def test_workspace_host_failures_pend_group_instead_of_becoming_rewards(self):
        trainer = SimpleNamespace(
            model=None, processing_class=None, state=SimpleNamespace(global_step=0)
        )
        tool = '<|tool_call>call:read_file{path:<|"|>note.txt<|"|>}<tool_call|><|tool_response>'
        original = Workspace.read_file
        for failure in (
            RuntimeError("unexpected harness failure"),
            OSError(errno.EIO, "disk I/O failure"),
        ):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as tmp:
                remaining = [failure]
                called = []

                def fail_first_slot(workspace, path, failures=remaining):
                    if failures and "attempt-000" in str(workspace.root):
                        raise failures.pop()
                    return original(workspace, path)

                def reward(task, result, calls=called):
                    calls.append(result["seed"])
                    return Reward("ok", 1.0)

                groups = RolloutGroups(
                    [task()],
                    GRPOSettings(revision=REVISION, max_generated_tokens=512),
                    Path(tmp),
                    reward,
                    lambda model, tokenizer, seed: self.backend([tool, "Done<turn|>"], seed=seed),
                    SYSTEM_PROMPT,
                )
                with (
                    patch.object(Workspace, "read_file", fail_first_slot),
                    self.assertRaises(GroupPending),
                ):
                    groups(["fixture"] * 2, trainer)
                attempts = sorted(Path(tmp).glob("groups/*/attempt-*"))
                results = [json.loads((p / "result.json").read_text()) for p in attempts]
                rewards = [json.loads((p / "reward.json").read_text()) for p in attempts]
                self.assertEqual(results[0]["failure_class"], "infrastructure")
                self.assertEqual(rewards[0]["status"], "unavailable")
                self.assertEqual(rewards[1]["status"], "ok")
                self.assertEqual(called, [74])
                self.assertTrue(list(Path(tmp).glob("groups/*/stopped.json")))
                self.assertFalse(list(Path(tmp).glob("groups/*/complete.json")))

    def test_oversized_valid_tool_observation_pends_group_without_reward(self):
        selected = task()
        selected["visible"]["initial_files"]["note.txt"] = "observation " * 2500
        selected["visible"]["budgets"].update(max_total_bytes=100000, max_read_tokens=8192)
        settings = GRPOSettings(revision=REVISION, context_tokens=1024, max_generated_tokens=512)
        trainer = SimpleNamespace(
            model=None, processing_class=None, state=SimpleNamespace(global_step=0)
        )
        called = []

        def reward(task, result):
            called.append(result)
            return Reward("ok", 0.0)

        tool = '<|tool_call>call:read_file{path:<|"|>note.txt<|"|>}<tool_call|><|tool_response>'
        with tempfile.TemporaryDirectory() as tmp:
            groups = RolloutGroups(
                [selected],
                settings,
                Path(tmp),
                reward,
                lambda model, tokenizer, seed: self.backend([tool], seed=seed, context=1024),
                SYSTEM_PROMPT,
            )
            with self.assertRaises(GroupPending):
                groups(["fixture"] * 2, trainer)
            self.assertEqual(called, [])
            for attempt in Path(tmp).glob("groups/*/attempt-*"):
                result = json.loads((attempt / "result.json").read_text())
                self.assertEqual(result["failure_class"], "infrastructure")
                self.assertEqual(result["tool_errors"], 0)
                tokens = json.loads((attempt / "tokens.json").read_text())
                self.assertGreater(sum(tokens["env_mask"]), 0)
                self.assertIn("observation", (attempt / "trace.jsonl").read_text())
                self.assertEqual(
                    json.loads((attempt / "reward.json").read_text())["status"], "unavailable"
                )

    def test_interruption_preserves_started_attempt_and_stops_group(self):
        settings = GRPOSettings(revision=REVISION)
        trainer = SimpleNamespace(
            model=None, processing_class=None, state=SimpleNamespace(global_step=0)
        )
        with tempfile.TemporaryDirectory() as tmp:
            groups = RolloutGroups(
                [task()],
                settings,
                Path(tmp),
                lambda task, result: Reward("ok", 0.0),
                lambda model, tokenizer, seed: self.backend(
                    [KeyboardInterrupt("fixture interruption")]
                ),
                SYSTEM_PROMPT,
            )
            with self.assertRaises(KeyboardInterrupt):
                groups(["fixture"] * 2, trainer)
            result = json.loads(next(Path(tmp).glob("groups/*/attempt-*/result.json")).read_text())
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(result["after"]["note.txt"], "before")
            self.assertTrue(list(Path(tmp).glob("groups/*/stopped.json")))
            self.assertFalse(list(Path(tmp).glob("groups/*/complete.json")))


if __name__ == "__main__":
    unittest.main()
