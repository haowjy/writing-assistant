"""Pinned TRL/Liger source admission used by task-graph training."""

import tempfile
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from writing_agent.grpo_runtime import (
    SOURCE_PINS,
    STREAMING,
    RuntimeImportMismatch,
    RuntimeSourceMismatch,
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
    def test_only_the_qualified_streaming_route_has_a_plan(self):
        plan = implementation_plan(STREAMING)
        self.assertEqual(plan["id"], STREAMING)
        self.assertEqual(plan["trl_commit"], "6c5f1350488e9bba9a71242c47db45f2869796fa")
        self.assertTrue(plan["config"]["use_liger_kernel"])
        self.assertFalse(any(plan["config"]["liger_kernel_config"].values()))
        for retired in ("trl-1.13", "trl-latest"):
            with self.subTest(route=retired), self.assertRaises(ValueError):
                implementation_plan(retired)

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

    @unittest.skipUnless(INSTALLED_TRL == "1.14.0.dev0", "Pinned environment only")
    def test_installed_streaming_stack_matches_its_source_pins(self):
        self.assertEqual(verify_runtime(STREAMING), implementation_plan(STREAMING))

    def test_changed_source_and_shadowed_import_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "trl"
            root.mkdir()
            (root / "__init__.py").write_text("not the approved archive")
            dist = SimpleNamespace(version="1.14.0.dev0", locate_file=lambda module: root)
            with (
                patch(
                    "writing_agent.grpo_runtime.SOURCE_PINS",
                    {"trl": SOURCE_PINS["trl"]},
                ),
                patch("writing_agent.grpo_runtime.distribution", return_value=dist),
                patch(
                    "writing_agent.grpo_runtime.find_spec",
                    return_value=SimpleNamespace(origin=str(root / "__init__.py")),
                ),
                self.assertRaises(RuntimeSourceMismatch),
            ):
                verify_runtime(STREAMING)
            with (
                patch("writing_agent.grpo_runtime.distribution", return_value=dist),
                patch(
                    "writing_agent.grpo_runtime.find_spec",
                    return_value=SimpleNamespace(origin="/shadow/trl/__init__.py"),
                ),
                self.assertRaises(RuntimeImportMismatch),
            ):
                verify_runtime(STREAMING)

    def test_model_family_and_moe_are_not_silent_dense_fallbacks(self):
        for family, moe in (("llama", False), ("gemma4", True)):
            config = SimpleNamespace(
                model_type=family,
                get_text_config=lambda moe=moe: SimpleNamespace(enable_moe_block=moe),
            )
            with self.subTest(family=family, moe=moe), self.assertRaises(ValueError):
                validate_streaming_model(config)


if __name__ == "__main__":
    unittest.main()
