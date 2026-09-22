"""CPU-only identity, exact-token envelope and complete-inventory admission contracts."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from writing_agent.grpo_gpu import (
    CUDA_ALLOCATOR_CONF,
    HEADLESS_POLICY,
    admit_gpu,
    configure_cuda_allocator,
    ownership_report,
)
from writing_agent.grpo_gpu_fit import (
    execute_fit,
    generation_prefix,
    inspect_fit,
    preflight_fit,
    prepare_fit,
    token_ledgers,
)


def xml(consumers=(), free=22000):
    rows = "".join(
        f"<process_info><pid>{pid}</pid><process_name>{name}</process_name>"
        f"<type>{kind}</type><used_memory>{memory} MiB</used_memory></process_info>"
        for pid, name, kind, memory in consumers
    )
    return (
        "<nvidia_smi_log><gpu><uuid>GPU-test</uuid><product_name>RTX 3090</product_name>"
        f"<fb_memory_usage><total>24576 MiB</total><free>{free} MiB</free>"
        f"<used>1000 MiB</used></fb_memory_usage><processes>{rows}</processes>"
        "</gpu></nvidia_smi_log>"
    )


class OwnershipTests(unittest.TestCase):
    def test_complete_graphics_and_compute_inventory_and_exact_limits(self):
        consumers = [(n, "chrome", "G", 256) for n in range(1, 4)]
        self.assertTrue(ownership_report(xml(consumers))["admitted"])
        self.assertFalse(ownership_report(xml(consumers, free=21999))["admitted"])
        self.assertFalse(ownership_report(xml(consumers + [(4, "cursor", "C", 1)]))["admitted"])
        self.assertFalse(ownership_report(xml([(1, "chrome", "G", 257)]))["admitted"])
        for kind in ("G", "C", "C+G"):
            for name in ("/usr/bin/Xwayland", "/usr/bin/ghostty"):
                with self.subTest(kind=kind, name=name):
                    result = ownership_report(xml([(1, name, kind, 1)]))
                    self.assertTrue(result["admitted"])
                    self.assertEqual(result["consumers"][0]["type"], kind)
        self.assertFalse(ownership_report(xml([(1, "python", "C", 0)]))["admitted"])

    def test_headless_policy_requires_empty_inventory_and_24000_mib_free(self):
        self.assertTrue(ownership_report(xml(free=24000), policy=HEADLESS_POLICY)["admitted"])
        self.assertFalse(ownership_report(xml(free=23999), policy=HEADLESS_POLICY)["admitted"])
        self.assertFalse(
            ownership_report(xml([(1, "cosmic-comp", "G", 1)], free=24000), policy=HEADLESS_POLICY)[
                "admitted"
            ]
        )

    def test_expandable_allocator_is_set_before_torch_import(self):
        env = {k: v for k, v in os.environ.items() if k not in CUDA_ALLOCATOR_CONF["environment"]}
        code = (
            "import os; "
            "from writing_agent.grpo_gpu import CUDA_ALLOCATOR_CONF, configure_cuda_allocator; "
            "assert configure_cuda_allocator() == CUDA_ALLOCATOR_CONF; "
            "assert all(os.environ[k] == CUDA_ALLOCATOR_CONF['value'] "
            "for k in CUDA_ALLOCATOR_CONF['environment'])"
        )
        subprocess.run([sys.executable, "-c", code], check=True, env=env)
        conflict = {**env, "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"}
        rejected = subprocess.run(
            [sys.executable, "-c", code],
            env=conflict,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.assertNotEqual(rejected.returncode, 0)
        with (
            patch.dict(sys.modules, {"torch": object()}),
            self.assertRaisesRegex(RuntimeError, "before importing torch"),
        ):
            configure_cuda_allocator()

    def test_incomplete_or_unknown_inventory_refuses_and_preserves(self):
        for data in (
            "",
            xml().replace("<processes></processes>", ""),
            xml().replace("<processes></processes>", "<processes>N/A</processes>"),
            xml().replace("22000 MiB", "N/A"),
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                ownership_report(data)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "ownership"
            with patch(
                "writing_agent.grpo_gpu.capture_inventory",
                return_value=xml([(1, "python", "C", 10)]),
            ):
                with self.assertRaisesRegex(RuntimeError, "ownership rejected"):
                    admit_gpu(output)
            self.assertFalse(json.loads((output / "ownership.json").read_text())["admitted"])
            self.assertTrue((output / "failure.json").is_file())


class Tokenizer:
    all_special_ids = [0]

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        return [1, 2, 3] if tokenize else "prefix" + messages[0]["content"] + "suffix"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class FitTests(unittest.TestCase):
    def test_inspection_imports_no_model_stack_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "absent"
            code = (
                "import sys; from writing_agent.grpo_gpu_fit import inspect_fit; inspect_fit(); "
                "assert not {'torch','transformers','trl','peft'} & set(sys.modules)"
            )
            subprocess.run([sys.executable, "-c", code], check=True)
            for flags in ([], ["--phase", "fit"]):
                subprocess.run(
                    [sys.executable, "scripts/run_grpo_gpu_fit.py", str(output), *flags],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                self.assertFalse(output.exists())

    def test_exact_ledger_masks_and_profile_identity(self):
        rows = token_ledgers(Tokenizer())
        self.assertEqual([r["reward"] for r in rows], [0, 0.25, 0.75, 1])
        for row in rows:
            self.assertEqual(len(row["prompt_ids"]) + len(row["completion_ids"]), 32768)
            self.assertEqual(row["env_mask"], [0] * (24576 - len(row["prompt_ids"])) + [1] * 8192)
            self.assertEqual(len(row["env_mask"]), len(row["completion_ids"]))
        self.assertEqual(len(generation_prefix(Tokenizer())), 32767)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fresh"
            record = prepare_fit(output)
            self.assertEqual(preflight_fit(output), record)
            self.assertEqual(inspect_fit()["profile"], "gemma-full48-controlled-fit-v3")
            self.assertEqual(inspect_fit()["allocator"], CUDA_ALLOCATOR_CONF)
            self.assertEqual(inspect_fit()["ownership_policy"], HEADLESS_POLICY)
            changed = inspect_fit()
            changed["training"]["active_tokens_each"] -= 1
            with patch("writing_agent.grpo_gpu_fit.inspect_fit", return_value=changed):
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    preflight_fit(output)

    def test_fit_source_and_ownership_reject_before_model_or_workers(self):
        for reject in ("source", "ownership"):
            with self.subTest(reject=reject), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "fit"
                prepare_fit(output)
                with (
                    patch(
                        "writing_agent.grpo_gpu_fit.verify_runtime",
                        return_value={},
                        side_effect=ValueError("source") if reject == "source" else None,
                    ),
                    patch(
                        "writing_agent.grpo_gpu_fit.configure_cuda_allocator",
                        return_value=CUDA_ALLOCATOR_CONF,
                    ) as configure_allocator,
                    patch(
                        "writing_agent.grpo_gpu_fit.admit_gpu",
                        side_effect=RuntimeError("ownership"),
                    ) as admit,
                    patch("writing_agent.grpo_gpu_fit.capture_inventory"),
                    patch("writing_agent.grpo_gpu_fit.version", return_value="fixture"),
                    patch("writing_agent.grpo_gpu_fit.multiprocessing.get_context") as spawn,
                ):
                    with self.assertRaisesRegex((ValueError, RuntimeError), reject):
                        execute_fit(output)
                    self.assertFalse(spawn.called)
                    self.assertEqual(configure_allocator.called, reject == "ownership")
                    self.assertEqual(admit.called, reject == "ownership")
                    with self.assertRaises(FileExistsError):
                        execute_fit(output)
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["stages"])

    def test_full48_order_for_fresh_and_resume(self):
        from writing_agent.grpo_full48_runner import execute_training

        for resume in (False, True):
            for reject in (
                "preflight",
                "verify_runtime",
                "configure_cuda_allocator",
                "admit_gpu",
            ):
                with self.subTest(resume=resume, reject=reject):
                    events = []

                    def boundary(name, events=events, reject=reject):
                        def check(*args, **kwargs):
                            events.append(name)
                            if name == reject:
                                raise ValueError(name)

                        return check

                    with (
                        patch("writing_agent.grpo_full48_runner.verify_lease"),
                        patch(
                            "writing_agent.grpo_full48_runner.preflight",
                            side_effect=boundary("preflight"),
                        ),
                        patch(
                            "writing_agent.grpo_full48_runner.verify_runtime",
                            side_effect=boundary("verify_runtime"),
                        ),
                        patch(
                            "writing_agent.grpo_full48_runner.configure_cuda_allocator",
                            side_effect=boundary("configure_cuda_allocator"),
                        ),
                        patch(
                            "writing_agent.grpo_full48_runner.admit_gpu",
                            side_effect=boundary("admit_gpu"),
                        ),
                        patch("writing_agent.grpo_full48_runner.run_training") as train,
                    ):
                        with self.assertRaisesRegex(ValueError, reject):
                            execute_training("release", "run", lease_fd=1, resume=resume)
                        self.assertFalse(train.called)
                    order = ["preflight", "verify_runtime", "configure_cuda_allocator", "admit_gpu"]
                    self.assertEqual(events, order[: order.index(reject) + 1])


if __name__ == "__main__":
    unittest.main()
