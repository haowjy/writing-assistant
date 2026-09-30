"""Contracts for the identity-agnostic trainer lifecycle."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from writing_agent.grpo_trainer import CheckpointLocationError, run_trainer


class RolloutFactoryTests(unittest.TestCase):
    def test_resume_from_another_output_is_refused_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "training"
            output.mkdir()
            (output / "experiment.json").write_text('{"identity":"identity","manifest":{}}')
            checkpoint = root / "previous-run" / "checkpoint-1"
            checkpoint.mkdir(parents=True)
            (checkpoint / "sentinel").write_text("unchanged")
            before = {
                path.relative_to(root): (
                    "directory" if path.is_dir() else "file",
                    path.read_bytes() if path.is_file() else None,
                )
                for path in sorted(root.rglob("*"))
            }
            tokenizer = SimpleNamespace(padding_side="right")
            api = SimpleNamespace(
                set_seed=lambda _seed: self.fail("trainer mutated before refusal")
            )

            with self.assertRaises(CheckpointLocationError):
                run_trainer(
                    api=api,
                    tasks=[],
                    output=output,
                    settings=SimpleNamespace(max_steps=2),
                    plan={},
                    identity="identity",
                    manifest={},
                    model=None,
                    tokenizer=tokenizer,
                    lora_config=None,
                    trainer_config_values={},
                    make_rollouts=lambda _invocation_id: None,
                    resume_from_checkpoint=checkpoint,
                    stop_after_steps=None,
                )

            after = {
                path.relative_to(root): (
                    "directory" if path.is_dir() else "file",
                    path.read_bytes() if path.is_file() else None,
                )
                for path in sorted(root.rglob("*"))
            }
            self.assertEqual(after, before)
            self.assertEqual(tokenizer.padding_side, "right")

    def test_factory_receives_the_new_invocation_id(self):
        class StopAtFactory(Exception):
            pass

        class Dropout:
            pass

        model = SimpleNamespace(modules=lambda: [])
        tokenizer = SimpleNamespace(padding_side="right")
        api = SimpleNamespace(
            set_seed=lambda seed: None,
            get_peft_model=lambda model, config: model,
            torch=SimpleNamespace(nn=SimpleNamespace(Dropout=Dropout)),
            TrainerCallback=object,
        )
        settings = SimpleNamespace(seed=7, max_steps=1)
        received = []

        def make_rollouts(invocation_id):
            received.append(invocation_id)
            raise StopAtFactory

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "training"
            with self.assertRaises(StopAtFactory):
                run_trainer(
                    api=api,
                    tasks=[],
                    output=output,
                    settings=settings,
                    plan={},
                    identity="identity",
                    manifest={},
                    model=model,
                    tokenizer=tokenizer,
                    lora_config=object(),
                    trainer_config_values={},
                    make_rollouts=make_rollouts,
                    resume_from_checkpoint=None,
                    stop_after_steps=None,
                )

            self.assertEqual(len(received), 1)
            self.assertEqual(len(received[0]), 32)
            self.assertTrue((output / "invocations" / received[0] / "started.json").is_file())

    def test_trainer_callback_factory_receives_live_peft_model(self):
        class StopAtCallback(Exception):
            pass

        class Dropout:
            pass

        model = SimpleNamespace(modules=lambda: [])
        tokenizer = SimpleNamespace(padding_side="right")
        api = SimpleNamespace(
            set_seed=lambda seed: None,
            get_peft_model=lambda model, config: model,
            torch=SimpleNamespace(nn=SimpleNamespace(Dropout=Dropout)),
            TrainerCallback=object,
            GRPOConfig=lambda **kwargs: object(),
        )
        settings = SimpleNamespace(seed=7, max_steps=1)
        received = []

        def make_callback(live_model):
            received.append(live_model)
            raise StopAtCallback

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StopAtCallback):
                run_trainer(
                    api=api,
                    tasks=[],
                    output=Path(tmp) / "training",
                    settings=settings,
                    plan={},
                    identity="identity",
                    manifest={},
                    model=model,
                    tokenizer=tokenizer,
                    lora_config=object(),
                    trainer_config_values={},
                    make_rollouts=lambda _invocation_id: object(),
                    resume_from_checkpoint=None,
                    stop_after_steps=None,
                    trainer_callback_factory=make_callback,
                )

        self.assertEqual(received, [model])


if __name__ == "__main__":
    unittest.main()
