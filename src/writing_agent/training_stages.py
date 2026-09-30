"""One-shot subprocess supervision and resource evidence for training stages."""

from __future__ import annotations

import fcntl
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from writing_agent.catalog import save_json

_WORKER_METRICS_ENV = "TASK_GRAPH_STAGE_WORKER_METRICS"
_STAGE_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


@dataclass(frozen=True)
class StageLimits:
    """Per-stage wall-time and resource ceilings, all byte limits inclusive."""

    wall_time_seconds: float
    kill_grace_seconds: float = 10.0
    peak_rss_bytes: int | None = None
    torch_peak_reserved_bytes: int | None = None
    disk_growth_bytes: int | None = None
    gpu_budget_seconds: float | None = None

    def validate(self) -> None:
        if (
            isinstance(self.wall_time_seconds, bool)
            or not isinstance(self.wall_time_seconds, (int, float))
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds <= 0
        ):
            raise ValueError("wall_time_seconds must be finite and positive")
        if (
            isinstance(self.kill_grace_seconds, bool)
            or not isinstance(self.kill_grace_seconds, (int, float))
            or not math.isfinite(self.kill_grace_seconds)
            or self.kill_grace_seconds < 0
        ):
            raise ValueError("kill_grace_seconds must be finite and nonnegative")
        for name in ("peak_rss_bytes", "torch_peak_reserved_bytes", "disk_growth_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if self.gpu_budget_seconds is not None and (
            isinstance(self.gpu_budget_seconds, bool)
            or not isinstance(self.gpu_budget_seconds, (int, float))
            or not math.isfinite(self.gpu_budget_seconds)
            or self.gpu_budget_seconds <= 0
        ):
            raise ValueError("gpu_budget_seconds must be finite and positive")


class StageFailure(RuntimeError):
    """A stage failed, timed out, or exceeded a configured resource ceiling."""

    def __init__(self, stage: str, status: str, status_path: Path):
        self.stage = stage
        self.status = status
        self.status_path = status_path
        super().__init__(f"Stage {stage!r} ended as {status}; inspect {status_path}")


class StageAttemptError(RuntimeError):
    """A stage or probe attempt was already started or left unreconciled."""


def read_nvml_inventory() -> dict[str, Any]:
    """Capture the complete inventory using grpo_gpu's strict NVML XML parser."""
    from writing_agent.grpo_gpu import inventory

    result = subprocess.run(["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True)
    return inventory(result.stdout)


def _worker_metrics() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(usage if sys.platform == "darwin" else usage * 1024)
    metrics: dict[str, Any] = {
        "peak_rss_bytes": rss_bytes,
        "torch_peak_allocated_bytes": None,
        "torch_peak_reserved_bytes": None,
        "torch_metrics_reason": "Torch was not loaded in the stage worker",
    }

    torch = sys.modules.get("torch")
    if torch is None:
        return metrics
    try:
        if not torch.cuda.is_initialized():
            metrics["torch_metrics_reason"] = "CUDA was not initialized in the stage worker"
            return metrics
        metrics["torch_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        metrics["torch_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        metrics["torch_metrics_reason"] = None
    except Exception as exc:  # Metrics must not conceal the stage's own result.
        metrics["torch_metrics_reason"] = f"{type(exc).__name__}: {exc}"
    return metrics


def worker_main(call: Callable[[], Any]) -> Any:
    """Run a stage body and report child RSS/Torch peaks without importing Torch here.

    A stage command should call this in its child process. The supervisor communicates
    the private report path through ``TASK_GRAPH_STAGE_WORKER_METRICS``; the parent
    merges it into the final ``resources.json`` and removes the temporary report.
    """
    try:
        return call()
    finally:
        metrics_path = os.environ.get(_WORKER_METRICS_ENV)
        if metrics_path:
            save_json(Path(metrics_path), _worker_metrics())


def _validate_stage(stage: str, command: Sequence[str]) -> None:
    if (
        not isinstance(stage, str)
        or not stage
        or any(char not in _STAGE_NAME_CHARS for char in stage)
    ):
        raise ValueError("stage must contain only ASCII letters, digits, '_' or '-'")
    if (
        isinstance(command, (str, bytes))
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ValueError("command must be a nonempty sequence of nonempty strings")


def _disk_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(current) / name).is_symlink()]
        for name in files:
            try:
                total += (Path(current) / name).lstat().st_size
            except FileNotFoundError:
                continue
    return total


def _gpu_process_seconds(run_dir: Path) -> float:
    total = 0.0
    for path in (run_dir / "stages").glob("*/resources.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise StageAttemptError(f"Stage resource record is unreadable: {path}") from exc
        if record.get("gpu_stage"):
            seconds = record.get("gpu_process_seconds")
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or seconds < 0
            ):
                raise StageAttemptError(f"GPU time is unavailable in {path}")
            total += seconds
    return total


def _peak_rss_from_proc(pid: int) -> int | None:
    try:
        for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _stop_process_group(process: subprocess.Popen, grace_seconds: float) -> None:
    """Stop only this stage's new process group, escalating after a bounded grace."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # Kill descendants too, even if the group leader exited on SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _claim_attempt(run_dir: Path, stage: str):
    attempt_dir = run_dir / "attempt.lock"
    attempt_dir.mkdir(exist_ok=True)
    if not attempt_dir.is_dir():
        raise StageAttemptError("Attempt marker is not a directory")

    if (attempt_dir / "terminal.json").exists():
        raise StageAttemptError("Probe attempt is terminal; retries require a new run directory")

    guard = (attempt_dir / ".guard").open("a+")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        guard.close()
        raise StageAttemptError("Another stage owns this probe attempt") from exc

    # An orphan marker or a still-running status means the owning supervisor did not
    # publish a terminal result. Refuse to advance rather than infer it succeeded.
    for marker in attempt_dir.glob("*.started"):
        prior_stage = marker.stem
        prior_status = run_dir / "stages" / prior_stage / "status.json"
        if not prior_status.exists():
            guard.close()
            raise StageAttemptError(f"Stage {prior_stage!r} has no terminal status record")
        try:
            prior = json.loads(prior_status.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            guard.close()
            raise StageAttemptError(f"Stage {prior_stage!r} status is unreadable") from exc
        if prior.get("status") != "completed":
            guard.close()
            raise StageAttemptError(f"Stage {prior_stage!r} did not complete")
        prior_resources = run_dir / "stages" / prior_stage / "resources.json"
        try:
            resource_record = json.loads(prior_resources.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            guard.close()
            raise StageAttemptError(f"Stage {prior_stage!r} resources are unreadable") from exc
        if resource_record.get("status") != "completed":
            guard.close()
            raise StageAttemptError(f"Stage {prior_stage!r} resources are not completed")

    marker = attempt_dir / f"{stage}.started"
    try:
        with marker.open("x") as stream:
            stream.write(f"{time.time():.6f}\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        guard.close()
        raise StageAttemptError(
            f"Stage {stage!r} was already attempted; retries require a new run directory"
        ) from exc
    return guard, attempt_dir


def _terminalize(attempt_dir: Path, stage: str, status: str, failure: str) -> None:
    terminal = attempt_dir / "terminal.json"
    if not terminal.exists():
        save_json(terminal, {"stage": stage, "status": status, "failure": failure})


def run_stage(
    run_dir: Path,
    stage: str,
    command: Sequence[str],
    *,
    limits: StageLimits,
    cwd: Path | None = None,
    gpu_stage: bool = False,
    admit: Callable[[], Any] | None = None,
    nvml_reader: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run one stage once, enforcing its limits and atomically saving its evidence.

    GPU work requires an ``admit`` callback. The probe runner supplies the callback
    which calls ``grpo_gpu``'s ownership gate; this module never imports the DAPO
    probe and never imports Torch in the supervisor process.
    """
    _validate_stage(stage, command)
    limits.validate()
    if gpu_stage and admit is None:
        raise ValueError("GPU stages require an ownership-admission callback")
    if not gpu_stage and admit is not None:
        raise ValueError("Ownership admission is only valid for GPU stages")
    if not gpu_stage and limits.gpu_budget_seconds is not None:
        raise ValueError("Aggregate GPU time is only valid for GPU stages")

    run_dir = Path(run_dir).resolve()
    child_cwd = Path(cwd).resolve() if cwd is not None else run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    disk_before = _disk_bytes(run_dir)
    guard, attempt_dir = _claim_attempt(run_dir, stage)
    stage_dir = run_dir / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    status_path = stage_dir / "status.json"
    resources_path = stage_dir / "resources.json"
    worker_metrics_path = stage_dir / ".worker-resources.json"
    stdout_path = stage_dir / "stdout.log"
    started_at = time.time()
    started = time.monotonic()
    status = "starting"
    failure: str | None = None
    returncode: int | None = None
    process: subprocess.Popen | None = None
    admission_result: Any = None
    nvml_before: Any = None
    nvml_after: Any = None
    nvml_reason: str | None = "not a GPU stage"
    peak_rss_bytes: int | None = None
    gpu_process_seconds: float | None = None
    gpu_seconds_before = 0.0
    effective_wall_time_seconds = limits.wall_time_seconds
    process_started: float | None = None
    raised: BaseException | None = None
    status_record = {
        "stage": stage,
        "status": status,
        "started_at": started_at,
        "command": list(command),
        "working_directory": str(child_cwd),
        "limits": {
            "wall_time_seconds": limits.wall_time_seconds,
            "peak_rss_bytes": limits.peak_rss_bytes,
            "torch_peak_reserved_bytes": limits.torch_peak_reserved_bytes,
            "disk_growth_bytes": limits.disk_growth_bytes,
            "gpu_budget_seconds": limits.gpu_budget_seconds,
        },
    }
    save_json(status_path, status_record)

    def capture_nvml() -> Any:
        nonlocal nvml_reason
        try:
            reader = nvml_reader or read_nvml_inventory
            value = reader()
            nvml_reason = None
            return value
        except Exception as exc:  # Record loss of evidence, then fail closed below.
            nvml_reason = f"{type(exc).__name__}: {exc}"
            return None

    try:
        if gpu_stage and limits.gpu_budget_seconds is not None:
            gpu_seconds_before = _gpu_process_seconds(run_dir)
            remaining = limits.gpu_budget_seconds - gpu_seconds_before
            effective_wall_time_seconds = min(
                limits.wall_time_seconds, remaining - limits.kill_grace_seconds
            )
            if effective_wall_time_seconds <= 0:
                raise RuntimeError("Aggregate GPU-stage budget cannot cover shutdown grace")
        if gpu_stage:
            nvml_before = capture_nvml()
            if nvml_before is None:
                raise RuntimeError(f"NVML before-stage inventory unavailable: {nvml_reason}")
            admission_result = admit()
            if (
                not isinstance(admission_result, dict)
                or admission_result.get("admitted") is not True
            ):
                raise RuntimeError("GPU ownership callback did not admit this stage")
        child_env = os.environ.copy()
        if "PYTHONPATH" in child_env:
            child_env["PYTHONPATH"] = os.pathsep.join(
                str(Path(part or os.curdir).resolve())
                for part in child_env["PYTHONPATH"].split(os.pathsep)
            )
        child_env[_WORKER_METRICS_ENV] = str(worker_metrics_path)
        with stdout_path.open("wb") as log:
            process_started = time.monotonic()
            process = subprocess.Popen(
                list(command),
                cwd=child_cwd,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status = "running"
            status_record.update(status=status, child_pid=process.pid)
            save_json(status_path, status_record)
            deadline = process_started + effective_wall_time_seconds
            while True:
                sampled_rss = _peak_rss_from_proc(process.pid)
                if sampled_rss is not None:
                    peak_rss_bytes = max(peak_rss_bytes or 0, sampled_rss)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    failure = f"wall-time ceiling ({limits.wall_time_seconds}s) exceeded"
                    _stop_process_group(process, limits.kill_grace_seconds)
                    returncode = process.returncode
                    break
                try:
                    returncode = process.wait(timeout=min(0.1, remaining))
                    status = "completed" if returncode == 0 else "failed"
                    if returncode != 0:
                        failure = f"child exited with return code {returncode}"
                    break
                except subprocess.TimeoutExpired:
                    continue
            gpu_process_seconds = time.monotonic() - process_started if gpu_stage else None
        if process is not None:
            if gpu_stage and gpu_process_seconds is None:
                gpu_process_seconds = (
                    time.monotonic() - process_started if process_started is not None else 0.0
                )
            final_rss = _peak_rss_from_proc(process.pid)
            if final_rss is not None:
                peak_rss_bytes = max(peak_rss_bytes or 0, final_rss)
    except BaseException as exc:
        raised = exc
        if process is not None and process.poll() is None:
            try:
                _stop_process_group(process, limits.kill_grace_seconds)
            except Exception as stop_error:
                failure = f"{type(exc).__name__}: {exc}; child cleanup failed: {stop_error}"
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        failure = failure or f"{type(exc).__name__}: {exc}"
        returncode = process.returncode if process is not None else None
    finally:
        if gpu_stage:
            nvml_after = capture_nvml()
        worker_metrics: dict[str, Any] = {}
        if worker_metrics_path.exists():
            try:
                worker_metrics = json.loads(worker_metrics_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                failure = failure or f"Worker resource report is unreadable: {exc}"
                status = "failed"
            worker_metrics_path.unlink(missing_ok=True)
        reported_rss = worker_metrics.get("peak_rss_bytes")
        if type(reported_rss) is int:
            peak_rss_bytes = max(peak_rss_bytes or 0, reported_rss)

        disk_after = _disk_bytes(run_dir)
        disk_growth = max(0, disk_after - disk_before)
        torch_reserved = worker_metrics.get("torch_peak_reserved_bytes")
        torch_allocated = worker_metrics.get("torch_peak_allocated_bytes")
        ceilings = []
        if limits.peak_rss_bytes is not None:
            if peak_rss_bytes is None:
                ceilings.append("peak RSS unavailable")
            elif peak_rss_bytes > limits.peak_rss_bytes:
                ceilings.append("peak RSS ceiling exceeded")
        if limits.torch_peak_reserved_bytes is not None:
            if type(torch_reserved) is not int:
                ceilings.append("Torch peak reserved memory unavailable")
            elif torch_reserved > limits.torch_peak_reserved_bytes:
                ceilings.append("Torch peak reserved memory ceiling exceeded")
        if limits.disk_growth_bytes is not None and disk_growth > limits.disk_growth_bytes:
            ceilings.append("run-directory disk-growth ceiling exceeded")
        if gpu_stage and nvml_after is None:
            ceilings.append("NVML after-stage inventory unavailable")
        if ceilings:
            failure = "; ".join(filter(None, [failure, *ceilings]))
            if status == "completed":
                status = "failed"
        ended_at = time.time()
        resources = {
            "stage": stage,
            "status": status,
            "returncode": returncode,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss_bytes,
            "rss_reason": "child worker RUSAGE_SELF and supervisor /proc VmHWM sampling",
            "torch_peak_allocated_bytes": torch_allocated,
            "torch_peak_reserved_bytes": torch_reserved,
            "torch_metrics_reason": worker_metrics.get("torch_metrics_reason", "No child report"),
            "nvml_before": nvml_before,
            "nvml_after": nvml_after,
            "nvml_reason": nvml_reason,
            "admission": admission_result,
            "run_directory_bytes_before": disk_before,
            "run_directory_bytes_after": disk_after,
            "run_directory_growth_bytes": disk_growth,
            "gpu_stage": gpu_stage,
            "gpu_budget_seconds": limits.gpu_budget_seconds,
            "gpu_seconds_before": gpu_seconds_before,
            "gpu_process_seconds": gpu_process_seconds,
            "gpu_seconds_after": gpu_seconds_before + (gpu_process_seconds or 0.0),
            "wall_time_seconds_applied": effective_wall_time_seconds,
            "ceilings_exceeded": ceilings,
            "failure": failure,
            "started_at": started_at,
            "ended_at": ended_at,
        }
        save_json(resources_path, resources)
        status_record.update(
            status=status,
            ended_at=ended_at,
            elapsed_seconds=resources["elapsed_seconds"],
            returncode=returncode,
            failure=failure,
            resources="resources.json",
        )
        save_json(status_path, status_record)
        if status != "completed":
            _terminalize(attempt_dir, stage, status, failure or "stage did not complete")
        fcntl.flock(guard, fcntl.LOCK_UN)
        guard.close()

    if raised is not None:
        if isinstance(raised, (KeyboardInterrupt, SystemExit)):
            raise raised
        raise StageFailure(stage, status, status_path) from raised
    if status != "completed":
        raise StageFailure(stage, status, status_path)
    return status_record
