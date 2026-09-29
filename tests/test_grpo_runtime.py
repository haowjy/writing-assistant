"""Runtime source admission and immutable legacy plan contracts, without downloads."""

import json
import os
import sys
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from writing_agent.grpo import GRPOSettings, inspect_grpo, train_grpo
from writing_agent.grpo_runtime import (
    LEGACY,
    STREAMING,
    implementation_plan,
    python_tree_hash,
    validate_streaming_model,
    verify_runtime,
)

try:
    INSTALLED_TRL = version("trl")
except PackageNotFoundError:
    INSTALLED_TRL = None


class RuntimeAdmissionTests(unittest.TestCase):
    def test_wandb_environment_is_unchanged_when_runtime_admission_fails(self):
        from test_grpo import REVISION, task

        from scripts.smoke_grpo_cpu import toy_reward

        bindings = {
            "WANDB_RUN_ID": "admission-failure",
            "WANDB_ENTITY": "fixture-entity",
            "WANDB_PROJECT": "fixture-project",
            "WANDB_MODE": "online",
            "WANDB_LOG_MODEL": "false",
            "WANDB_WATCH": "false",
            "WANDB_DISABLE_CODE": "true",
            "WANDB_API_KEY": "must-not-be-copied",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}):
            for key in bindings:
                os.environ.pop(key, None)
            before = dict(os.environ)
            with (
                patch.dict(sys.modules, {"wandb": None}),
                patch(
                    "writing_agent.grpo.verify_runtime",
                    side_effect=ValueError("runtime admission rejected"),
                ),
                self.assertRaisesRegex(ValueError, "runtime admission rejected"),
            ):
                train_grpo(
                    [task()],
                    Path(tmp) / "run",
                    settings=GRPOSettings(revision=REVISION),
                    reward_spec={
                        "id": "fixture",
                        "config": {},
                        "mode": "mechanical-only-smoke",
                    },
                    admission={"mode": "engineered-fixture", "label": "test-only"},
                    reward_callback=toy_reward,
                    execute=True,
                    report_to="wandb",
                    wandb_run_name=bindings["WANDB_RUN_ID"],
                    wandb_environment=bindings,
                )
            self.assertEqual(dict(os.environ), before)

    def test_legacy_plan_shape_and_explicit_streaming_identity(self):
        from test_grpo import REVISION, task

        common = dict(
            tasks=[task()],
            output="unused",
            settings=GRPOSettings(revision=REVISION),
            reward_spec={"id": "test", "config": {}, "mode": "mechanical-only-smoke"},
            admission={"mode": "engineered-fixture", "label": "test"},
        )
        legacy = inspect_grpo(**common)
        self.assertEqual(legacy, inspect_grpo(implementation=LEGACY, **common))
        self.assertNotIn("implementation", legacy)
        self.assertNotIn("implementation", legacy["settings"])
        streaming = inspect_grpo(implementation=STREAMING, **common)
        selected = streaming.pop("implementation")
        self.assertEqual(legacy, streaming)
        self.assertEqual(selected, implementation_plan(STREAMING))
        self.assertTrue(selected["config"]["use_liger_kernel"])
        self.assertFalse(any(selected["config"]["liger_kernel_config"].values()))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            inspect_grpo(implementation="trl-latest", **common)

    def test_source_digest_binds_content_paths_and_extra_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trl"
            root.mkdir()
            source = root / "__init__.py"
            source.write_text("original")
            original = python_tree_hash(root)
            source.write_text("changed")
            self.assertNotEqual(original, python_tree_hash(root))
            source.write_text("original")
            self.assertEqual(original, python_tree_hash(root))
            source.rename(root / "other.py")
            self.assertNotEqual(original, python_tree_hash(root))
            (root / "__init__.py").write_text("original")
            self.assertNotEqual(original, python_tree_hash(root))

    def test_installed_stack_and_wrong_opt_in(self):
        try:
            installed = version("trl")
        except PackageNotFoundError:
            self.skipTest("Optional TRL environment")
        if installed == "1.13.0":
            self.assertIsNone(verify_runtime(LEGACY))
            with self.assertRaisesRegex(ValueError, "requires trl"):
                verify_runtime(STREAMING)
        elif installed == "1.14.0.dev0":
            self.assertEqual(verify_runtime(STREAMING), implementation_plan(STREAMING))
            with self.assertRaisesRegex(ValueError, "explicit opt-in"):
                verify_runtime(LEGACY)
        else:
            self.fail(f"Unqualified test environment: {installed}")

    def test_same_version_changed_source_and_shadowed_import_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trl"
            root.mkdir()
            (root / "__init__.py").write_text("not the approved archive")
            dist = SimpleNamespace(version="1.14.0.dev0", locate_file=lambda module: root)
            with (
                patch("writing_agent.grpo_runtime.distribution", return_value=dist),
                patch(
                    "writing_agent.grpo_runtime.find_spec",
                    return_value=SimpleNamespace(origin=str(root / "__init__.py")),
                ),
                self.assertRaisesRegex(ValueError, "source mismatch"),
            ):
                verify_runtime(STREAMING)
            with (
                patch("writing_agent.grpo_runtime.distribution", return_value=dist),
                patch(
                    "writing_agent.grpo_runtime.find_spec",
                    return_value=SimpleNamespace(origin="/shadow/trl/__init__.py"),
                ),
                self.assertRaisesRegex(ValueError, "import/source mismatch"),
            ):
                verify_runtime(STREAMING)

    def test_model_family_and_moe_are_not_silent_dense_fallbacks(self):
        for family, moe in (("llama", False), ("gemma4", True)):
            config = SimpleNamespace(
                model_type=family,
                get_text_config=lambda moe=moe: SimpleNamespace(enable_moe_block=moe),
            )
            with self.assertRaises(ValueError):
                validate_streaming_model(config)

    @unittest.skipUnless(INSTALLED_TRL == "1.14.0.dev0", "Pinned environment only")
    def test_changed_implementation_resume_rejected_before_caller_mutation(self):
        import copy
        import pickle
        import random

        import numpy as np
        import torch

        from scripts.smoke_grpo_streaming_cpu import common_settings, tiny_gemma
        from writing_agent.grpo_identity import base_tensor_identity

        common = common_settings()
        model, tokenizer = tiny_gemma()

        def snapshot():
            return (
                base_tensor_identity(model),
                tokenizer.backend_tokenizer.to_str(),
                tokenizer.padding_side,
                copy.deepcopy(model.config.to_dict()),
                [(n, p.requires_grad) for n, p in model.named_parameters()],
                pickle.dumps(
                    (
                        random.getstate(),
                        np.random.get_state(),
                        torch.get_rng_state().numpy().tobytes(),
                    )
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            manifest = output / "experiment.json"
            # Incompatible implementation must fail at manifest comparison, before
            # checkpoint inspection or PEFT/tokenizer/RNG mutation.
            manifest.write_text(
                json.dumps({"identity": "legacy", "manifest": {"implementation": {"id": LEGACY}}})
            )
            before, files = snapshot(), manifest.read_bytes()
            with self.assertRaisesRegex(ValueError, "identity changed"):
                train_grpo(
                    output=output,
                    model=model,
                    tokenizer=tokenizer,
                    resume_from_checkpoint=output / "checkpoint-1",
                    **common,
                )
            self.assertEqual(before, snapshot())
            self.assertEqual(files, manifest.read_bytes())
            self.assertEqual(list(output.iterdir()), [manifest])
