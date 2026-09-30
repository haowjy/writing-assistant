"""CPU-only contracts for terminal stage supervision and durable evidence."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from writing_agent.training_stages import (
    StageAttemptError,
    StageFailure,
    StageLimits,
    run_stage,
)


class TrainingStageTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)

    def test_supervisor_import_is_torch_and_legacy_probe_free(self):
        code = (
            "import sys; import writing_agent.training_stages; "
            "assert 'torch' not in sys.modules; "
            "assert 'writing_agent.grpo_probe' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_wall_time_kill_is_terminal_and_recorded(self):
        run = self.root / "timeout"
        with self.assertRaises(StageFailure) as raised:
            run_stage(
                run,
                "train",
                [sys.executable, "-c", "import time; time.sleep(30)"],
                limits=StageLimits(wall_time_seconds=0.1, kill_grace_seconds=0.05),
            )

        stage_dir = run / "stages" / "train"
        status = json.loads((stage_dir / "status.json").read_text())
        resources = json.loads((stage_dir / "resources.json").read_text())
        assert raised.exception.status == "timeout"
        assert status["status"] == "timeout"
        assert status["failure"].startswith("wall-time ceiling")
        assert resources["status"] == "timeout"
        assert resources["returncode"] < 0
        assert (run / "attempt.lock" / "terminal.json").exists()

    def test_a_second_stage_attempt_is_refused_by_persistent_marker(self):
        run = self.root / "single-attempt"
        command = [sys.executable, "-c", "pass"]
        run_stage(run, "train", command, limits=StageLimits(wall_time_seconds=5))

        with self.assertRaises(StageAttemptError):
            run_stage(run, "train", command, limits=StageLimits(wall_time_seconds=5))
        assert (run / "attempt.lock" / "train.started").exists()
        assert json.loads((run / "stages/train/status.json").read_text())["status"] == "completed"

    def test_memory_ceiling_breach_is_failed_and_recorded(self):
        run = self.root / "memory"
        worker = "from writing_agent.training_stages import worker_main; worker_main(lambda: None)"
        with self.assertRaises(StageFailure) as raised:
            run_stage(
                run,
                "train",
                [sys.executable, "-c", worker],
                limits=StageLimits(wall_time_seconds=5, peak_rss_bytes=1),
            )

        status = json.loads((run / "stages/train/status.json").read_text())
        resources = json.loads((run / "stages/train/resources.json").read_text())
        assert raised.exception.status == "failed"
        assert status["status"] == "failed"
        assert resources["peak_rss_bytes"] > 1
        assert "peak RSS ceiling exceeded" in resources["ceilings_exceeded"]
        with self.assertRaises(StageAttemptError):
            run_stage(
                run,
                "resume",
                [sys.executable, "-c", "pass"],
                limits=StageLimits(wall_time_seconds=5),
            )

    def test_disk_growth_ceiling_breach_is_failed_and_recorded(self):
        run = self.root / "disk"
        worker = (
            "from pathlib import Path; "
            "from writing_agent.training_stages import worker_main; "
            "worker_main(lambda: Path('payload.bin').write_bytes(b'x' * 128))"
        )
        with self.assertRaises(StageFailure) as raised:
            run_stage(
                run,
                "inspect",
                [sys.executable, "-c", worker],
                limits=StageLimits(wall_time_seconds=5, disk_growth_bytes=1),
            )

        status = json.loads((run / "stages/inspect/status.json").read_text())
        resources = json.loads((run / "stages/inspect/resources.json").read_text())
        assert raised.exception.status == "failed"
        assert status["status"] == "failed"
        assert resources["run_directory_growth_bytes"] > 1
        assert "run-directory disk-growth ceiling exceeded" in resources["ceilings_exceeded"]

    def test_gpu_hooks_are_stubbed_and_nvml_snapshots_are_recorded(self):
        run = self.root / "cpu-hook-fixture"
        inventory = {"name": "CPU test stub", "consumers": []}
        reads = []
        admissions = []

        def nvml_reader():
            reads.append(True)
            return inventory

        def admit():
            admissions.append(True)
            return {"admitted": True, "stubbed_for_cpu": True}

        worker = "from writing_agent.training_stages import worker_main; worker_main(lambda: None)"
        run_stage(
            run,
            "train",
            [sys.executable, "-c", worker],
            limits=StageLimits(wall_time_seconds=5),
            gpu_stage=True,
            admit=admit,
            nvml_reader=nvml_reader,
        )

        resources = json.loads((run / "stages/train/resources.json").read_text())
        assert len(reads) == 2 and len(admissions) == 1
        assert resources["nvml_before"] == inventory
        assert resources["nvml_after"] == inventory
        assert resources["admission"]["stubbed_for_cpu"]

    def test_gpu_ownership_refusal_does_not_launch_the_child(self):
        run = self.root / "refused"
        with self.assertRaises(StageFailure):
            run_stage(
                run,
                "train",
                [sys.executable, "-c", "raise SystemExit(91)"],
                limits=StageLimits(wall_time_seconds=5),
                gpu_stage=True,
                admit=lambda: {"admitted": False},
                nvml_reader=lambda: {"stubbed_for_cpu": True},
            )

        status = json.loads((run / "stages/train/status.json").read_text())
        resources = json.loads((run / "stages/train/resources.json").read_text())
        assert status["status"] == "failed"
        assert status["returncode"] is None
        assert "did not admit" in status["failure"]
        assert resources["admission"]["admitted"] is False

    def test_gpu_budget_is_shared_and_reserved_for_kill_grace(self):
        run = self.root / "aggregate-gpu-budget"
        limits = StageLimits(
            wall_time_seconds=5,
            kill_grace_seconds=0.1,
            gpu_budget_seconds=1.0,
        )

        def admit():
            return {"admitted": True, "stubbed_for_cpu": True}

        def nvml_reader():
            return {"stubbed_for_cpu": True}

        quick_worker = (
            "import time; from writing_agent.training_stages import worker_main; "
            "worker_main(lambda: time.sleep(0.2))"
        )
        slow_worker = (
            "import time; from writing_agent.training_stages import worker_main; "
            "worker_main(lambda: time.sleep(2))"
        )
        run_stage(
            run,
            "train",
            [sys.executable, "-c", quick_worker],
            limits=limits,
            gpu_stage=True,
            admit=admit,
            nvml_reader=nvml_reader,
        )
        with self.assertRaises(StageFailure) as raised:
            run_stage(
                run,
                "resume",
                [sys.executable, "-c", slow_worker],
                limits=limits,
                gpu_stage=True,
                admit=admit,
                nvml_reader=nvml_reader,
            )

        resources = json.loads((run / "stages/resume/resources.json").read_text())
        assert raised.exception.status == "timeout"
        assert resources["gpu_seconds_before"] > 0
        assert resources["wall_time_seconds_applied"] < limits.wall_time_seconds
        assert resources["gpu_seconds_after"] <= limits.gpu_budget_seconds

    def test_status_and_resource_records_are_atomic_json(self):
        run = self.root / "atomic"
        worker = "from writing_agent.training_stages import worker_main; worker_main(lambda: None)"
        run_stage(
            run,
            "inspect",
            [sys.executable, "-c", worker],
            limits=StageLimits(wall_time_seconds=5),
        )

        for filename in ("status.json", "resources.json"):
            path = run / "stages/inspect" / filename
            record = json.loads(path.read_text())
            assert record["stage"] == "inspect"
            assert record["status"] == "completed"
        assert not list(run.rglob("*.tmp"))
        assert not (run / "stages/inspect/.worker-resources.json").exists()


if __name__ == "__main__":
    unittest.main()
