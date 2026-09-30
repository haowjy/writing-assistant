"""One-shot runner for the Phase 8 task-graph integration probe.

The runner owns frozen preflight data and phase ordering. Training and offline
inspection run in supervised subprocesses; importing this module loads no model stack.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from writing_agent.catalog import save_json
from writing_agent.grpo_runtime import STREAMING, verify_runtime
from writing_agent.grpo_task_graph_probe_evidence import _select_verdict
from writing_agent.training_stages import (
    StageAttemptError,
    StageFailure,
    StageLimits,
    _disk_bytes,
    run_stage,
    worker_main,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_task_graph_probe.py"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
TOKENIZER_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
GIB = 1024**3
CEILINGS = {
    "aggregate_gpu_seconds": 45 * 60,
    "train_seconds": 25 * 60,
    "resume_seconds": 20 * 60,
    "peak_reserved_gpu_bytes": int(21.0 * GIB),
    "peak_rss_bytes": 24 * GIB,
    "run_directory_growth_bytes": 3 * GIB,
}
N3_POLICY = {
    "mode": "desktop",
    "names": [
        "cosmic-comp",
        "cosmic-panel",
        "cosmic-bg",
        "cosmic-app-library",
        "cosmic-edit",
        "cosmic-settings",
        "cosmic-files",
        "xdg-desktop-portal-cosmic",
        "xwayland",
        "ghostty",
        "chrome",
        "cursor",
    ],
    "per_process_mib": 256,
    "total_mib": 768,
    "minimum_free_mib": 22000,
}


class ProbeError(RuntimeError):
    """The prepared probe cannot safely advance."""


def _prepare_digest(record: dict[str, Any]) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _work_dir(run_dir: Path) -> Path:
    configured = os.environ.get("MERIDIAN_ACTIVE_WORK_DIR")
    if configured:
        return Path(configured).resolve()
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent.resolve()
    for parent in run_dir.parents:
        if (parent / "env-phase8" / "environment.json").is_file():
            return parent
    raise ProbeError("cannot locate the task-graph-training work item")


def _smoke_helpers():
    """Load the existing tiny-Gemma setup without executing its command-line entrypoint."""
    module_name = "_task_graph_probe_cpu_smoke"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = PROJECT_ROOT / "scripts" / "smoke_task_graph_grpo_cpu.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ProbeError("CPU smoke helpers are unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _source_identity() -> dict[str, str]:
    dirty = _git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ProbeError("source checkout is dirty; prepare requires a committed tree")
    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    return {"commit": commit, "tree": tree}


def _task_graph_hashes() -> list[dict[str, str]]:
    smoke = _smoke_helpers()
    root = PROJECT_ROOT / "configs" / "phase8" / "probe-tasks"
    result = []
    for path in sorted(root.glob("t*.json")):
        config = smoke._BUILDER.load_probe_task(path)
        entry = smoke._BUILDER.build_admitted_entry(config)
        result.append(
            {
                "task_id": config["id"],
                "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "graph_instance_hash": entry.graph.instance.identity(),
                "entry_state_hash": entry.state.identity(),
            }
        )
    if tuple(item["task_id"] for item in result) != (
        "t1-lighthouse",
        "t2-winter-garden",
        "t3-coastal-post",
    ):
        raise ProbeError("the frozen Phase 8 task order is unavailable")
    return result


def _environment_digests(work_dir: Path) -> dict[str, Any]:
    environment_path = work_dir / "env-phase8" / "environment.json"
    try:
        environment = json.loads(environment_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError("qualified env-phase8 environment.json is unavailable") from exc
    runtime = verify_runtime(STREAMING)
    package_digests = environment.get("source_digests", {}).get("packages", {})
    for package, pin in runtime["sources"].items():
        if package_digests.get(package, {}).get("python_tree_sha256") != pin["python_tree_sha256"]:
            raise ProbeError(f"env-phase8 digest record differs for {package}")
    expected_versions = environment.get("versions", {})
    actual_versions = {}
    from importlib.metadata import version

    for package, metadata_name in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("liger-kernel", "liger-kernel"),
    ):
        actual_versions[package] = version(metadata_name)
        if expected_versions.get(package) != actual_versions[package]:
            raise ProbeError(f"env-phase8 version record differs for {package}")
    return {
        "environment_json_sha256": hashlib.sha256(environment_path.read_bytes()).hexdigest(),
        "runtime_admission": runtime,
        "versions": actual_versions,
    }


def _offline_requirements(*, cpu: bool | None) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE"):
        if os.environ.get(name) != "1":
            raise ProbeError(f"offline probe requires {name}=1")
    if cpu is not None:
        expected_devices = "" if cpu else "0"
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_devices:
            raise ProbeError(f"probe requires CUDA_VISIBLE_DEVICES={expected_devices!r}")


def _prepare_record(run_dir: Path, *, mode: str) -> dict[str, Any]:
    work_dir = _work_dir(run_dir)
    source = _source_identity()
    smoke = _smoke_helpers()
    tokenizer_root = smoke.TOKENIZER_PATH
    try:
        tokenizer_hashes = {
            name: hashlib.sha256((tokenizer_root / name).read_bytes()).hexdigest()
            for name in TOKENIZER_FILES
        }
    except OSError as exc:
        raise ProbeError("cached Gemma tokenizer files are incomplete") from exc
    settings = smoke.settings()
    recipe = {
        "settings": {key: getattr(settings, key) for key in settings.__dataclass_fields__},
        "model_id": settings.model_id,
        "revision": settings.revision,
        "base_dtype": "fp32" if mode.startswith("cpu") else "bfloat16",
        "attention": "eager" if mode.startswith("cpu") else "sdpa",
        "adapter": {"kind": "LoRA", "rank": 8, "alpha": 16, "target_modules": "all-linear"},
        "loss": {"type": "dapo", "beta": 0, "scale_rewards": "none"},
        "optimizer": {"kind": "AdamW", "learning_rate": 1e-5, "schedule": "constant"},
        "gradient_checkpointing": {"enabled": True, "use_reentrant": False},
        "execution_mode": mode,
    }
    if mode == "gpu":
        model_files = tuple(tokenizer_root.glob("*.safetensors"))
        if not model_files:
            raise ProbeError("cached Gemma E2B safetensors are unavailable")
    return {
        "schema": 1,
        "run_id": run_dir.name,
        "prepared_at": time.time(),
        "source": source,
        "environment_digests": _environment_digests(work_dir),
        "task_graph_hashes": _task_graph_hashes(),
        "recipe": recipe,
        "tokenizer_root": str(tokenizer_root),
        "tokenizer_files_sha256": tokenizer_hashes,
        "n3_policy": N3_POLICY,
        "ceilings": CEILINGS,
        "initial_run_directory_bytes": 0,
        "work_item": str(work_dir),
    }


def _load_prepare(run_dir: Path, *, mode: str) -> dict[str, Any]:
    try:
        prepared = json.loads((run_dir / "prepare.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError("prepare.json is missing or unreadable") from exc
    if prepared.get("schema") != 1 or prepared.get("run_id") != run_dir.name:
        raise ProbeError("prepare.json does not bind this run directory")
    integrity = prepared.get("integrity_sha256")
    body = {key: value for key, value in prepared.items() if key != "integrity_sha256"}
    if not isinstance(integrity, str) or _prepare_digest(body) != integrity:
        raise ProbeError("prepare.json integrity differs; start a new run")
    if prepared.get("recipe", {}).get("execution_mode") != mode:
        raise ProbeError("execution mode differs from prepare.json")
    source = _source_identity()
    if prepared.get("source") != source:
        raise ProbeError("source commit/tree differs from prepare.json; start a new run")
    current = _prepare_record(run_dir, mode=mode)
    for key in ("environment_digests", "task_graph_hashes", "tokenizer_files_sha256", "recipe"):
        if prepared.get(key) != current.get(key):
            raise ProbeError(f"prepared {key} changed; start a new run")
    return prepared


def inspect(run_dir: Path, *, mode: str) -> dict[str, Any]:
    """Read-only orientation; it does not query a GPU or create the run directory."""
    if run_dir.exists():
        prepare_path = run_dir / "prepare.json"
        if prepare_path.exists():
            prepared = json.loads(prepare_path.read_text())
            if prepared.get("recipe", {}).get("execution_mode") != mode:
                raise ProbeError("execution mode differs from existing prepare.json")
    return {
        "phase": "inspect",
        "run_directory": str(run_dir),
        "run_directory_exists": run_dir.exists(),
        "mode": mode,
        "phases": ["inspect", "prepare", "preflight", "train", "resume", "inspect-run"],
        "writes": False,
        "gpu_queries": 0,
    }


def prepare(run_dir: Path, *, mode: str) -> dict[str, Any]:
    if run_dir.exists():
        raise ProbeError("probe run directory must not exist before prepare")
    _offline_requirements(cpu=mode.startswith("cpu"))
    inspection = inspect(run_dir, mode=mode)
    record = _prepare_record(run_dir, mode=mode)
    record["integrity_sha256"] = _prepare_digest(record)
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_json(run_dir / "inspect.json", inspection)
    save_json(run_dir / "prepare.json", record)
    return {"phase": "prepare", "path": str(run_dir / "prepare.json"), "source": record["source"]}


def _cpu_ownership() -> dict[str, Any]:
    return {"admitted": True, "stubbed_for_cpu": True, "policy": N3_POLICY}


def preflight(run_dir: Path, *, mode: str) -> dict[str, Any]:
    prepared = _load_prepare(run_dir, mode=mode)
    _offline_requirements(cpu=mode.startswith("cpu"))
    if "torch" in sys.modules:
        raise ProbeError("preflight must run before importing Torch")
    if mode.startswith("cpu"):
        ownership = _cpu_ownership()
        (run_dir / "preflight").mkdir(mode=0o700, exist_ok=False)
        save_json(run_dir / "preflight" / "ownership.json", ownership)
    else:
        from writing_agent.grpo_gpu import admit_gpu

        try:
            ownership = admit_gpu(run_dir / "preflight" / "ownership", policy=prepared["n3_policy"])
        except Exception as exc:
            evidence_path = run_dir / "preflight" / "ownership" / "ownership.json"
            ownership = (
                json.loads(evidence_path.read_text())
                if evidence_path.is_file()
                else {"admitted": False}
            )
            record = {
                "schema": 1,
                "source": prepared["source"],
                "environment_digests": prepared["environment_digests"],
                "gpu_ownership": ownership,
                "model_loaded": False,
                "failure": f"{type(exc).__name__}: {exc}",
            }
            save_json(run_dir / "preflight.json", record)
            raise ProbeError("N3 GPU ownership policy refused preflight") from exc
    record = {
        "schema": 1,
        "source": prepared["source"],
        "environment_digests": prepared["environment_digests"],
        "gpu_ownership": ownership,
        "model_loaded": False,
    }
    save_json(run_dir / "preflight.json", record)
    if ownership.get("admitted") is not True:
        raise ProbeError("N3 GPU ownership policy refused preflight")
    return {"phase": "preflight", "model_loaded": False, "gpu_admitted": True}


def _latest_checkpoint(training_root: Path) -> Path:
    checkpoints = []
    for path in training_root.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match and (path / "complete.json").is_file():
            checkpoints.append((int(match.group(1)), path.resolve()))
    if not checkpoints:
        raise ProbeError("no complete trainer checkpoint exists")
    return max(checkpoints)[1]


def _require_latest_checkpoint(training_root: Path, checkpoint: Path) -> int:
    try:
        actual = checkpoint.resolve(strict=True)
        root = training_root.resolve(strict=True)
    except OSError as exc:
        raise ProbeError("resume checkpoint is unavailable") from exc
    if root not in actual.parents or actual != _latest_checkpoint(training_root):
        raise ProbeError("resume must use the latest complete checkpoint")
    from writing_agent.grpo_task_graph import task_graph_resume_preflight

    return task_graph_resume_preflight(training_root, actual)


@contextmanager
def _stage_observers(stage_dir: Path, *, cpu: bool):
    """Record finite optimizer gradients and per-decision generation wall time."""
    import torch

    smoke = _smoke_helpers()
    from writing_agent.native_gemma import NativeGemmaSampleBackend

    gradient_steps: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    adamw_step = torch.optim.AdamW.step
    sample_methods = [(NativeGemmaSampleBackend, NativeGemmaSampleBackend.sample)]
    if cpu:
        sample_methods.append((smoke.ScriptedNativeBackend, smoke.ScriptedNativeBackend.sample))

    def checked_step(optimizer, *args, **kwargs):
        finite = True
        tensors = 0
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is not None:
                    tensors += 1
                    finite = finite and bool(torch.isfinite(gradient).all().item())
        gradient_steps.append(
            {"step": len(gradient_steps) + 1, "tensor_count": tensors, "finite": finite}
        )
        if not finite or tensors == 0:
            raise FloatingPointError("optimizer received missing or non-finite gradients")
        return adamw_step(optimizer, *args, **kwargs)

    torch.optim.AdamW.step = checked_step
    for cls, original in sample_methods:

        def timed_sample(instance, prepared, _original=original):
            started = time.perf_counter()
            result = _original(instance, prepared)
            usage = result.usage
            timings.append(
                {
                    "elapsed_seconds": time.perf_counter() - started,
                    "prefill_tokens": usage.get("prefill_tokens"),
                    "generated_tokens": len(result.generated_token_ids),
                }
            )
            return result

        cls.sample = timed_sample
    try:
        yield
    finally:
        torch.optim.AdamW.step = adamw_step
        for cls, original in sample_methods:
            cls.sample = original
        save_json(
            stage_dir / "gradient-observer.json",
            {
                "schema": 1,
                "step_count": len(gradient_steps),
                "gradient_tensor_count": sum(item["tensor_count"] for item in gradient_steps),
                "finite": bool(gradient_steps) and all(item["finite"] for item in gradient_steps),
                "steps": gradient_steps,
            },
        )
        save_json(stage_dir / "generation-times.json", {"schema": 1, "decisions": timings})


def _train_worker(
    run_dir: Path, *, mode: str, stage: str, checkpoint: Path | None
) -> dict[str, Any]:
    _load_prepare(run_dir, mode=mode)
    _offline_requirements(cpu=mode.startswith("cpu"))
    training_root = run_dir / "training"
    if stage == "resume":
        if checkpoint is None:
            raise ProbeError("resume stage requires --resume <checkpoint>")
        resume_step = _require_latest_checkpoint(training_root, checkpoint)
        if resume_step != 2:
            raise ProbeError("the Phase 8 resume boundary is checkpoint-2")
    elif checkpoint is not None:
        raise ProbeError("train stage does not accept a resume checkpoint")

    stage_dir = run_dir / "stages" / stage
    if mode.startswith("cpu"):
        result = _run_cpu_training(
            training_root,
            stage_dir,
            resume=checkpoint,
            stop_after_steps=2 if stage == "train" else None,
            all_tie=mode == "cpu-all-tie",
        )
    else:
        result = _run_gpu_training(
            training_root,
            stage_dir,
            resume=checkpoint,
            stop_after_steps=2 if stage == "train" else None,
        )
    save_json(stage_dir / "training-result.json", result)
    if stage == "train" and result.get("global_step") != 2:
        raise ProbeError("train stage did not stop at checkpoint-2")
    if stage == "resume" and result.get("global_step") != 3:
        raise ProbeError("resume stage did not reach optimizer step 3")
    return {
        "stage": stage,
        "global_step": result.get("global_step"),
        "checkpoint": result.get("checkpoint"),
    }


def _run_cpu_training(
    training_root: Path,
    stage_dir: Path,
    *,
    resume: Path | None,
    stop_after_steps: int | None,
    all_tie: bool,
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(2)
    smoke = _smoke_helpers()
    with _stage_observers(stage_dir, cpu=True):
        return smoke._make_run(
            training_root,
            resume=resume,
            stop_after_steps=stop_after_steps,
            all_tie=all_tie,
        )


def _run_gpu_training(
    training_root: Path,
    stage_dir: Path,
    *,
    resume: Path | None,
    stop_after_steps: int | None,
) -> dict[str, Any]:
    from writing_agent.grpo_gpu import configure_cuda_allocator

    configure_cuda_allocator()
    import torch
    from transformers import AutoModelForCausalLM

    smoke = _smoke_helpers()
    settings = smoke.settings()
    from writing_agent.native_gemma import NativeGemmaSampleBackend

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        )

    runtime_identity = {
        "model": settings.model_id,
        "revision": settings.revision,
        "execution": "phase8-p1-3090-v1",
    }
    with _stage_observers(stage_dir, cpu=False):
        return smoke._make_run(
            training_root,
            resume=resume,
            stop_after_steps=stop_after_steps,
            model_factory=load_base,
            sample_backend_factory=NativeGemmaSampleBackend,
            runtime_identity=runtime_identity,
        )


def _admit_stage(run_dir: Path, stage: str, prepared: dict[str, Any], *, cpu: bool):
    if cpu:
        path = run_dir / "ownership" / stage
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
        record = _cpu_ownership()
        save_json(path / "ownership.json", record)
        return record
    from writing_agent.grpo_gpu import admit_gpu

    return admit_gpu(run_dir / "ownership" / stage, policy=prepared["n3_policy"])


def _stage_limits(run_dir: Path, prepared: dict[str, Any], stage: str, *, cpu: bool) -> StageLimits:
    ceiling = prepared["ceilings"]
    wall = ceiling["train_seconds"] if stage == "train" else ceiling["resume_seconds"]
    # Keep room for the parent to attach inspect-run's own resource record to result.json.
    reserve = 64 * 1024 if stage == "inspect-run" else 0
    remaining_disk = max(
        0,
        ceiling["run_directory_growth_bytes"] - _disk_bytes(run_dir) - reserve,
    )
    return StageLimits(
        wall_time_seconds=wall if stage != "inspect-run" else 45 * 60,
        peak_rss_bytes=ceiling["peak_rss_bytes"],
        torch_peak_reserved_bytes=(
            None if cpu or stage == "inspect-run" else ceiling["peak_reserved_gpu_bytes"]
        ),
        disk_growth_bytes=remaining_disk,
        gpu_budget_seconds=(
            ceiling["aggregate_gpu_seconds"] if stage in {"train", "resume"} else None
        ),
    )


def _launch_stage(
    run_dir: Path,
    prepared: dict[str, Any],
    stage: str,
    *,
    mode: str,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    cpu = mode.startswith("cpu")
    if _disk_bytes(run_dir) >= prepared["ceilings"]["run_directory_growth_bytes"]:
        raise ProbeError("run-directory growth ceiling is already exhausted")
    if stage == "inspect-run":
        args = [str(run_dir), "--stage-worker", stage]
    else:
        args = [str(run_dir), "--stage-worker", stage]
        if checkpoint is not None:
            args.extend(("--stage-checkpoint", str(checkpoint)))
    if cpu:
        args.append("--cpu-dry-run")
        if mode == "cpu-all-tie":
            args.append("--cpu-all-tie")
    command = [sys.executable, str(SCRIPT_PATH), *args]
    limits = _stage_limits(run_dir, prepared, stage, cpu=cpu)
    try:
        status = run_stage(
            run_dir,
            stage,
            command,
            limits=limits,
            cwd=PROJECT_ROOT,
            gpu_stage=stage in {"train", "resume"},
            admit=(lambda: _admit_stage(run_dir, stage, prepared, cpu=cpu))
            if stage in {"train", "resume"}
            else None,
            nvml_reader=(lambda: {"stubbed_for_cpu": True, "device": "CPU dry run"})
            if cpu
            else None,
        )
    except StageFailure:
        if stage == "inspect-run":
            status_path = run_dir / "stages" / stage / "status.json"
            failed_status = json.loads(status_path.read_text())
            _attach_inspector_stage(run_dir, failed_status)
        raise
    if stage == "inspect-run":
        _attach_inspector_stage(run_dir, status)
    return status


def _attach_inspector_stage(run_dir: Path, status: dict[str, Any]) -> None:
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        return
    result = json.loads(result_path.read_text())
    resources_path = run_dir / "stages" / "inspect-run" / "resources.json"
    try:
        resources = json.loads(resources_path.read_text())
    except (OSError, json.JSONDecodeError):
        resources = None
    result.setdefault("measurements", {}).setdefault("stage_runtime_seconds", {})["inspect-run"] = (
        status.get("elapsed_seconds")
    )
    result.setdefault("measurements", {}).setdefault("stage_disk_growth_bytes", {})[
        "inspect-run"
    ] = resources.get("run_directory_growth_bytes") if resources else None
    result.setdefault("stages", {})["inspect-run"] = {
        "status": status.get("status"),
        "resources": resources,
    }
    _recompute_final_verdict(result)
    save_json(result_path, result)


def _recompute_final_verdict(result: dict[str, Any]) -> None:
    criteria = result.get("criteria", {})
    c5 = criteria.get("criterion_5", {})
    stages = result.get("stages", {})
    inspect_stage = stages.get("inspect-run", {})
    if inspect_stage.get("status") == "completed" and inspect_stage.get("resources"):
        c5.setdefault("evidence", {})["inspect_run_stage"] = {
            "status": inspect_stage["status"],
            "ceilings_exceeded": inspect_stage["resources"].get("ceilings_exceeded"),
        }
        c5["passed"] = c5.get("passed") is True and not inspect_stage["resources"].get(
            "ceilings_exceeded"
        )
    elif inspect_stage.get("status"):
        c5.setdefault("evidence", {})["inspect_run_stage"] = {
            "status": inspect_stage["status"],
            "ceilings_exceeded": (inspect_stage.get("resources") or {}).get("ceilings_exceeded"),
        }
        c5["passed"] = False
    result["verdict"] = _select_verdict(criteria, result.get("measurements", {}))


def _stage_worker(args) -> None:
    run_dir = args.run_dir.resolve()
    mode = _mode(args)
    if args.stage_worker in {"train", "resume"}:
        worker_main(
            lambda: _train_worker(
                run_dir,
                mode=mode,
                stage=args.stage_worker,
                checkpoint=args.stage_checkpoint,
            )
        )
        return
    if args.stage_worker == "inspect-run":
        from writing_agent.grpo_task_graph_probe_evidence import inspect_run

        worker_main(lambda: inspect_run(run_dir, mode=mode))
        result = json.loads((run_dir / "result.json").read_text())
        if result["verdict"] == "fail":
            raise ProbeError("inspect-run criterion failed; see result.json")
        return
    raise ProbeError("unknown internal stage")


def _mode(args) -> str:
    if args.cpu_all_tie:
        if not args.cpu_dry_run:
            raise ProbeError("--cpu-all-tie requires --cpu-dry-run")
        return "cpu-all-tie"
    return "cpu-dry-run" if args.cpu_dry_run else "gpu"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--phase",
        choices=("inspect", "prepare", "preflight", "train", "resume", "inspect-run"),
        default="inspect",
    )
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cpu-dry-run", action="store_true")
    parser.add_argument("--cpu-all-tie", action="store_true")
    parser.add_argument("--stage-worker", choices=("train", "resume", "inspect-run"))
    parser.add_argument("--stage-checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run_dir = args.run_dir.resolve()
        mode = _mode(args)
        if args.stage_worker:
            _stage_worker(args)
            return 0
        if args.stage_checkpoint is not None:
            raise ProbeError("--stage-checkpoint is reserved for the supervised worker")
        if args.phase == "inspect":
            if any((args.execute, args.resume, args.stop_after_step)):
                raise ProbeError("inspect accepts no execution options")
            record = inspect(run_dir, mode=mode)
        elif args.phase == "prepare":
            if any((args.execute, args.resume, args.stop_after_step)):
                raise ProbeError("prepare accepts no execution options")
            record = prepare(run_dir, mode=mode)
        elif args.phase == "preflight":
            if any((args.execute, args.resume, args.stop_after_step)):
                raise ProbeError("preflight accepts no execution options")
            record = preflight(run_dir, mode=mode)
        elif args.phase == "train":
            if args.execute is not True or args.stop_after_step != 2 or args.resume is not None:
                raise ProbeError("train requires --stop-after-step 2 --execute")
            prepared = _load_prepare(run_dir, mode=mode)
            _offline_requirements(cpu=mode.startswith("cpu"))
            if not (run_dir / "preflight.json").is_file():
                raise ProbeError("preflight.json is required before train")
            record = _launch_stage(run_dir, prepared, "train", mode=mode)
        elif args.phase == "resume":
            if args.execute is not True or args.resume is None or args.stop_after_step is not None:
                raise ProbeError("resume requires --resume <checkpoint> --execute")
            prepared = _load_prepare(run_dir, mode=mode)
            _offline_requirements(cpu=mode.startswith("cpu"))
            if not (run_dir / "preflight.json").is_file():
                raise ProbeError("preflight.json is required before resume")
            record = _launch_stage(
                run_dir,
                prepared,
                "resume",
                mode=mode,
                checkpoint=args.resume.resolve(),
            )
        else:
            if any((args.execute, args.resume, args.stop_after_step)):
                raise ProbeError("inspect-run accepts no execution options")
            prepared = _load_prepare(run_dir, mode=mode)
            _offline_requirements(cpu=None)
            record = _launch_stage(run_dir, prepared, "inspect-run", mode=mode)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    except (ProbeError, StageAttemptError, StageFailure, OSError, ValueError) as exc:
        print(f"task-graph probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


__all__ = ["ProbeError", "_require_latest_checkpoint", "_select_verdict", "inspect", "main"]
