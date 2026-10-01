"""CPU-only tests for complete GPU inventory and ownership admission."""

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
    admit_gpu,
    configure_cuda_allocator,
    ownership_report,
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
    def test_gpu_admission_import_does_not_load_torch(self):
        code = "import sys; import writing_agent.grpo_gpu; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

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
