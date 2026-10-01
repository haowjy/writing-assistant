"""The task-graph trainer identity binds sources needed to reproduce a resume."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from writing_agent.grpo_task_graph import _task_graph_identity
from writing_agent.grpo_trainer import run_trainer


class TaskGraphIdentityTests(unittest.TestCase):
    def test_changed_native_parse_error_prefix_changes_experiment_identity(self):
        source = Path(__file__).parents[1] / "src" / "writing_agent" / "native_parse_errors.py"
        original = source.read_text()
        changed = original.replace(
            '"json parser could not parse region as JSON"',
            '"json parser could not parse malformed region"',
        )
        self.assertNotEqual(original, changed)

        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp)
            classifier = source_root / "native_parse_errors.py"
            classifier.write_text(original)

            experiment = {"schema": "task-graph-experiment-v1"}
            before, before_manifest = _task_graph_identity(experiment, source_root=source_root)
            classifier.write_text(changed)
            after, after_manifest = _task_graph_identity(experiment, source_root=source_root)

            self.assertNotEqual(before, after)
            self.assertEqual(set(before_manifest["code"]), {"native_parse_errors.py"})
            self.assertEqual(set(after_manifest["code"]), {"native_parse_errors.py"})

    def test_changed_native_source_refuses_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "src" / "writing_agent"
            source_root.mkdir(parents=True)
            for name in (
                "grpo_task_graph.py",
                "grpo_trainer.py",
                "native_gemma.py",
                "native_audit.py",
                "task_graph_derive_writer.py",
            ):
                (source_root / name).write_text(f"# {name} v1\n")

            experiment = {"schema": "task-graph-experiment-v1"}
            old_identity, old_manifest = _task_graph_identity(experiment, source_root=source_root)
            native_sampler = source_root / "native_gemma.py"
            native_sampler.write_text("# native_gemma.py v2\n")
            new_identity, new_manifest = _task_graph_identity(experiment, source_root=source_root)

            self.assertNotEqual(old_identity, new_identity)
            self.assertEqual(
                set(new_manifest["code"]),
                {
                    "grpo_task_graph.py",
                    "grpo_trainer.py",
                    "native_gemma.py",
                    "native_audit.py",
                    "task_graph_derive_writer.py",
                },
            )

            output = root / "training"
            output.mkdir()
            (output / "experiment.json").write_text(
                json.dumps({"identity": old_identity, "manifest": old_manifest})
            )
            checkpoint = output / "checkpoint-1"
            checkpoint.mkdir()
            with self.assertRaisesRegex(ValueError, "Resume experiment identity changed"):
                run_trainer(
                    api=None,
                    tasks=[],
                    output=output,
                    settings=SimpleNamespace(max_steps=2),
                    plan={},
                    identity=new_identity,
                    manifest=new_manifest,
                    model=None,
                    tokenizer=None,
                    lora_config=None,
                    trainer_config_values={},
                    make_rollouts=lambda _invocation_id: None,
                    resume_from_checkpoint=checkpoint,
                    stop_after_steps=None,
                )


if __name__ == "__main__":
    unittest.main()
