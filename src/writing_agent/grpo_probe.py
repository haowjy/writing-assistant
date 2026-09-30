"""Frozen, inspect-first orchestration for the bounded Gemma GRPO probe.

Importing this module does not import optional ML packages, inspect a GPU, load a
tokenizer/model, create directories, or start processes. Execution is phase-specific.
"""

import copy
import fcntl
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from writing_agent.agent import SYSTEM_PROMPT
from writing_agent.catalog import fingerprint, save_json
from writing_agent.grpo import GRPOSettings, file_hashes, inspect_grpo, train_grpo
from writing_agent.grpo_gpu import DISPLAY_POLICY, _display_consumers
from writing_agent.inference import PROTOCOL, TransformersBackend, load_checkpoint, render_messages
from writing_agent.suite import run_selected, saved_results
from writing_agent.workspace import TOOL_SCHEMAS, Workspace, dispatch

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
DEVELOPMENT_SEEDS = (104729, 130363)
GPU_PHASES = frozenset({"base-eval", "train", "resume", "adapter-eval"})
PHASES = ("inspect", "prepare", "preflight", "base-eval", "train", "resume", "adapter-eval")
GPU_BUDGET_SECONDS = 3600.0
DEFAULT_TIMEOUT_SECONDS = 1200.0

SETTINGS = GRPOSettings(
    model_id=MODEL_ID,
    revision=MODEL_REVISION,
    context_tokens=4096,
    max_tokens=768,
    max_generated_tokens=1536,
    max_steps=3,
    max_invocations=3,
    group_size=4,
    microbatch_size=1,
    learning_rate=1e-5,
    lora_rank=8,
    seed=42,
    enable_thinking=True,
    gradient_checkpointing=True,
    gradient_checkpointing_use_reentrant=False,
)
REWARD_SPEC = {
    "id": "grpo-probe-v1-mechanical",
    "mode": "mechanical-only-smoke",
    "config": {
        "semantic_judgment": "unavailable",
        "ties": "stop-whole-group",
        "failed_groups": "preserve-and-stop",
    },
}


def _utc_now():
    return datetime.now(UTC).isoformat()


def _code_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(Path(__file__).parent.rglob("*.py"))
    paths.append(root / "scripts" / "run_grpo_probe.py")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }


def evaluation_config(seed: int, adapter: Path | None = None):
    config = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "kind": "transformers",
        "protocol": PROTOCOL,
        "prompt_format": "chat",
        "loader": "causal_lm",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "quantization": "none",
        "attention": "sdpa",
        "seed": seed,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": True,
        "context_tokens": SETTINGS.context_tokens,
        "max_tokens": SETTINGS.max_tokens,
        "max_generated_tokens": SETTINGS.max_generated_tokens,
    }
    if adapter is not None:
        config["adapter"] = {"id": str(adapter.resolve()), "revision": None}
    return config


def frozen_plan(run_dir: Path):
    run_dir = Path(run_dir).resolve()
    plan = {
        "schema_version": 1,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "local_files_only": True},
        "training": asdict(SETTINGS),
        "schedule": {
            "train": "fresh start through saved optimizer step 1",
            "resume": "verified checkpoint-1 through frozen total optimizer step 3",
            "task_order": "three frozen training tasks, shuffle disabled, one full group each",
        },
        "reward": copy.deepcopy(REWARD_SPEC),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "evaluation": {
            "base": [evaluation_config(seed) for seed in DEVELOPMENT_SEEDS],
            "adapter_condition_changes_only": ["adapter"],
            "retry_failed": False,
            "policy_limit": (
                "Base and adapter evaluation both re-render complete native history. "
                "Training retains exact sampled tokens and appends masked environment suffixes; "
                "training/evaluation token policies are deliberately not claimed identical."
            ),
        },
        "resource_policy": {
            "aggregate_gpu_seconds": GPU_BUDGET_SECONDS,
            "default_stage_timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
            "occupied_gpu": "refuse",
            "display_policy": DISPLAY_POLICY,
            "timeout": "SIGTERM child process group, grace, then SIGKILL same group only",
        },
        "paths": {
            "run": str(run_dir),
            "release": str(run_dir / "release"),
            "fixture_evidence": str(run_dir / "fixture-evidence"),
            "preflight": str(run_dir / "preflight.json"),
            "base_evaluation": str(run_dir / "base-eval"),
            "training": str(run_dir / "training"),
            "adapter_evaluation": str(run_dir / "adapter-eval"),
            "resource_ledger": str(run_dir / "resource-ledger.json"),
        },
        "code": _code_hashes(),
    }
    plan["identity"] = fingerprint(plan)
    return plan


def inspect_probe(run_dir: Path):
    """Readiness without optional ML imports, model/tokenizer loads, or writes."""
    from writing_agent.grpo_probe_data import inspect_probe_release

    plan = frozen_plan(run_dir)
    run_dir = Path(run_dir).resolve()
    release = run_dir / "release"
    checks = {
        "release": release.is_dir(),
        "fixture_evidence": (run_dir / "fixture-evidence").is_dir(),
        "preflight": (run_dir / "preflight.json").is_file(),
        "base_evaluation": (run_dir / "base-eval" / "complete.json").is_file(),
        "checkpoint_1": (run_dir / "training" / "checkpoint-1" / "complete.json").is_file(),
        "training_complete": (run_dir / "training" / "checkpoint-3" / "complete.json").is_file(),
        "adapter_evaluation": (run_dir / "adapter-eval" / "complete.json").is_file(),
    }
    release_record = None
    blockers = []
    if checks["release"]:
        try:
            release_record = inspect_probe_release(release)
        except Exception as exc:
            blockers.append(f"release-invalid: {type(exc).__name__}: {exc}")
    else:
        blockers.append("prepare phase has not produced a frozen release")
    if not checks["preflight"]:
        blockers.append("cached-tokenizer preflight has not passed")
    ledger = _load_ledger(run_dir)
    try:
        _require_ready(run_dir)
        _check_ledger(ledger)
        if (run_dir / "base-eval").exists():
            _require_base(run_dir)
        if (run_dir / "training").exists():
            _completed_training(run_dir, 1, 0)
            if (run_dir / "training" / "checkpoint-3").exists():
                _completed_training(run_dir, 3, 1)
        if (run_dir / "adapter-eval").exists():
            _require_adapter_evaluation(run_dir)
        if any(s["status"] != "completed" for s in ledger["stages"]):
            raise ValueError("A prior stage did not complete; inspect retained failure evidence")
    except Exception as exc:
        blockers.append(f"binding/resources: {type(exc).__name__}: {exc}")
    invocation_records = []
    for path in sorted((run_dir / "training" / "invocations").glob("*/complete.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        invocation_records.append(
            {
                "global_step": record.get("global_step"),
                "resumed_step": record.get("resumed_step"),
                "checkpoint": record.get("checkpoint"),
                "adapter": record.get("adapter"),
                "identity": record.get("identity"),
            }
        )
    return {
        "status": "offline-ready" if not blockers else "blocked",
        "gpu_readiness": "conditional: exclusivity and free VRAM checked only on execution",
        "execute": False,
        "plan": plan,
        "checks": checks,
        "release": release_record,
        "resource_ledger": ledger,
        "gpu_seconds_used": _gpu_seconds(ledger),
        "training_artifacts": invocation_records,
        "blockers": blockers,
    }


def prepare_probe(run_dir: Path):
    from writing_agent.grpo_probe_data import build_probe_release, validate_probe_fixtures

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    release = run_dir / "release"
    evidence = run_dir / "fixture-evidence"
    if any(
        (run_dir / name).exists()
        for name in (
            "release",
            "fixture-evidence",
            "prepare.json",
            "preparation-binding.json",
            "plan.json",
        )
    ):
        raise FileExistsError("Prepare destinations already exist; probe artifacts are immutable")
    record = build_probe_release(release)
    fixtures = validate_probe_fixtures(release, evidence)
    result = {
        "status": "completed",
        "phase": "prepare",
        "plan_identity": frozen_plan(run_dir)["identity"],
        "release": {
            "directory": record["directory"],
            "freeze": record["freeze"],
            "train_ids": [task["id"] for task in record["train"]],
            "development_ids": [task["id"] for task in record["development"]],
        },
        "fixtures": {
            "directory": str(evidence),
            "fixtures_hash": fixtures["fixtures_hash"],
            "task_count": len(fixtures["tasks"]),
            "case_count": sum(len(task["cases"]) for task in fixtures["tasks"]),
            "is_model_evaluation": fixtures["is_model_evaluation"],
        },
    }
    save_json(run_dir / "prepare.json", result)
    save_json(run_dir / "preparation-binding.json", _preparation_binding(run_dir))
    return result


def _template_ids(tokenizer, messages, tools, *, generation):
    text = tokenizer.apply_chat_template(
        render_messages(messages),
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=generation,
        enable_thinking=True,
    )
    return text, tokenizer.encode(text, add_special_tokens=False)


def _read(path):
    return json.loads(Path(path).read_text())


def _preparation_binding(run_dir):
    from writing_agent.grpo_probe_data import DEFAULT_PACKET, load_probe_release

    release = load_probe_release(run_dir / "release")
    if release["freeze"]["packet_hash"] != fingerprint(_read(DEFAULT_PACKET)):
        raise ValueError("Release does not match the current frozen packet")
    if release["freeze"]["goldens_hash"] != fingerprint(
        _read(DEFAULT_PACKET.with_name("goldens.json"))
    ):
        raise ValueError("Release does not match current golden fixtures")
    prepare = _read(run_dir / "prepare.json")
    if prepare["status"] != "completed" or prepare["release"]["freeze"] != release["freeze"]:
        raise ValueError("Prepare evidence does not match release")
    if prepare["plan_identity"] != frozen_plan(run_dir)["identity"]:
        raise ValueError("Settings or source code changed since prepare")
    fixtures = _read(run_dir / "fixture-evidence" / "evidence.json")
    if fixtures["freeze"] != release["freeze"] or fixtures["is_model_evaluation"] is not False:
        raise ValueError("Fixture evidence does not match release")
    if fixtures["fixtures_hash"] != release["freeze"]["goldens_hash"]:
        raise ValueError("Fixture golden identity changed")
    tasks = {task["id"]: task for task in release["tasks"]}
    if [row["id"] for row in fixtures["tasks"]] != list(tasks):
        raise ValueError("Fixture task selection differs")
    from writing_agent.grpo_probe_data import mechanical_probe_reward

    for task in fixtures["tasks"]:
        cases = {row["case"] for row in task["cases"]}
        if not {"golden", "untouched", "empty", "failed", "missing", "partial"} <= cases:
            raise ValueError("Required fixture cases missing")
        for row in task["cases"]:
            saved = _read(run_dir / "fixture-evidence" / task["id"] / f"{row['case']}.json")
            if fingerprint(saved["result"]) != row["result_hash"]:
                raise ValueError("Fixture result changed")
            actual = asdict(mechanical_probe_reward(tasks[task["id"]], saved["result"]))
            if actual != row["reward"] or actual != saved["reward"]:
                raise ValueError("Fixture reward/check evidence changed")
            if actual["status"] != "ok" or (actual["value"] == 1) != (row["case"] == "golden"):
                raise ValueError("Fixture outcome failed acceptance")
    return {
        "schema_version": 2,
        "plan": frozen_plan(run_dir),
        "release_files": file_hashes(run_dir / "release"),
        "fixture_files": file_hashes(run_dir / "fixture-evidence"),
        "prepare": fingerprint(prepare),
    }


def _validate_preparation(run_dir):
    current = _preparation_binding(run_dir)
    if _read(run_dir / "preparation-binding.json") != current:
        raise ValueError("Frozen preparation binding changed")
    return current


def _golden_token_evidence(tokenizer, task, responses):
    """Execute fixture tools while measuring native evaluation/training boundaries."""
    from tempfile import TemporaryDirectory

    from writing_agent.grpo_rollout import native_suffix

    allowed = set(task["visible"]["tools"])
    tools = [schema for schema in TOOL_SCHEMAS if schema["function"]["name"] in allowed]
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["visible"]["brief"]},
    ]
    initial_text, initial_ids = _template_ids(tokenizer, history, tools, generation=True)
    completion_ids = []
    generated = 0
    boundaries = []
    file_bodies = []
    followups = iter(task["visible"]["followups"])
    with TemporaryDirectory(prefix="grpo-probe-preflight-") as temporary:
        workspace = Workspace(
            Path(temporary), max_total_bytes=task["visible"]["budgets"]["max_total_bytes"]
        )
        for path, content in task["visible"]["initial_files"].items():
            workspace.write_file(path, content)
        for index, assistant in enumerate(copy.deepcopy(responses)):
            # Thinking is enabled in the frozen condition. The portable data fixture
            # intentionally contains only actions/content, so add a small disclosed
            # native reasoning span for boundary and feasibility measurement.
            assistant["thinking"] = "Follow the visible instructions and use the workspace."
            eval_text, eval_ids = _template_ids(tokenizer, history, tools, generation=True)
            full_text, full_ids = _template_ids(
                tokenizer, history + [assistant], tools, generation=False
            )
            if full_ids[: len(eval_ids)] != eval_ids:
                raise ValueError(
                    f"Golden native action is not prompt-prefix stable: {task['id']} "
                    f"boundary {index}"
                )
            action = full_ids[len(eval_ids) :]
            boundary_text = "<|tool_response>" if assistant.get("tool_calls") else "<turn|>"
            boundary_ids = tokenizer.encode(boundary_text, add_special_tokens=False)
            if len(boundary_ids) != 1 or boundary_ids[0] not in action:
                raise ValueError(f"Golden action has no native stop boundary: {task['id']}")
            action = action[: len(action) - action[::-1].index(boundary_ids[0])]
            if not action:
                raise ValueError(f"Golden action is empty: {task['id']}")
            remaining = SETTINGS.max_generated_tokens - generated
            generation_limit = min(SETTINGS.max_tokens, remaining)
            boundary = {
                "index": index,
                "evaluation_prompt_tokens": len(eval_ids),
                "training_input_tokens": len(initial_ids) + len(completion_ids),
                "generation_limit_tokens": generation_limit,
                "action_tokens": len(action),
                "environment_tokens": 0,
                "context_with_reserved_call": (
                    len(initial_ids) + len(completion_ids) + generation_limit
                ),
                "evaluation_context_with_reserved_call": len(eval_ids) + generation_limit,
                "prompt_sha256": hashlib.sha256(eval_text.encode()).hexdigest(),
                "rendered_action_sha256": hashlib.sha256(full_text.encode()).hexdigest(),
            }
            completion_ids.extend(action)
            generated += len(action)
            history.append(assistant)
            external = []
            for call in assistant.get("tool_calls") or []:
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if call["function"]["name"] in {"write_file", "patch_file"}:
                    key = "content" if "content" in arguments else "new"
                    body = arguments.get(key, "")
                    file_bodies.append(
                        {
                            "boundary": index,
                            "path": arguments.get("path"),
                            "argument": key,
                            "tokens": len(tokenizer.encode(body, add_special_tokens=False)),
                        }
                    )
                observation = dispatch(workspace, call["function"]["name"], arguments)
                external.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(observation, ensure_ascii=False),
                    }
                )
            if not (assistant.get("tool_calls") or []):
                followup = next(followups, None)
                if followup is not None:
                    external.append({"role": "user", "content": followup})
            if external:
                suffix = native_suffix(tokenizer, assistant, external, action, thinking=True)
                completion_ids.extend(suffix)
                boundary["environment_tokens"] = len(suffix)
                history.extend(external)
            boundaries.append(boundary)
    label_check = _label_leak_check(tokenizer, task, tools, initial_text, initial_ids)
    failures = []
    for boundary in boundaries:
        if boundary["action_tokens"] > SETTINGS.max_tokens:
            failures.append(f"action {boundary['index']} exceeds per-call budget")
        if boundary["context_with_reserved_call"] > SETTINGS.context_tokens:
            failures.append(f"training boundary {boundary['index']} cannot reserve a full call")
        if boundary["evaluation_context_with_reserved_call"] > SETTINGS.context_tokens:
            failures.append(f"evaluation boundary {boundary['index']} cannot reserve a full call")
    if generated > SETTINGS.max_generated_tokens:
        failures.append("golden sampled actions exceed trajectory budget")
    if not label_check["mutation_invariant"]:
        failures.append("changing private labels changed the initial model input")
    return {
        "task_id": task["id"],
        "initial_prompt_tokens": len(initial_ids),
        "initial_prompt_sha256": hashlib.sha256(initial_text.encode()).hexdigest(),
        "boundaries": boundaries,
        "file_bodies": file_bodies,
        "sampled_action_tokens": generated,
        "training_trajectory_tokens": len(initial_ids) + len(completion_ids),
        "labels_excluded": label_check["mutation_invariant"],
        "private_check_id_text_overlaps": label_check["check_id_text_overlaps"],
        "fits": not failures,
        "failures": failures,
    }


def _label_leak_check(tokenizer, task, tools, initial_text, initial_ids):
    mutated = copy.deepcopy(task)
    mutated["labels"] = {"private_sentinel": "SHOULD_NOT_REACH_MODEL_4e6c"}
    changed_text, changed_ids = _template_ids(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": mutated["visible"]["brief"]},
        ],
        tools,
        generation=True,
    )
    check_ids = [str(check.get("id", "")) for check in task["labels"].get("checks", [])]
    private_ids_present = [value for value in check_ids if value and value in initial_text]
    return {
        "mutation_invariant": changed_text == initial_text and changed_ids == initial_ids,
        # A phrase can independently occur in the visible brief and a private check ID.
        # Keep that overlap inspectable without misclassifying it as data-flow leakage.
        "check_id_text_overlaps": private_ids_present,
    }


def preflight_probe(run_dir: Path):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from writing_agent.grpo_probe_data import golden_messages, load_probe_release

    run_dir = Path(run_dir).resolve()
    destination = run_dir / "preflight.json"
    if destination.exists():
        raise FileExistsError("Preflight already exists; probe artifacts are immutable")
    preparation = _validate_preparation(run_dir)
    release = load_probe_release(run_dir / "release")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    tasks = [
        _golden_token_evidence(tokenizer, task, golden_messages(task)) for task in release["tasks"]
    ]
    result = {
        "schema_version": 1,
        "status": "completed" if all(task["fits"] for task in tasks) else "blocked",
        "phase": "preflight",
        "preparation_identity": fingerprint(preparation),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weights_loaded": False},
        "tokenizer": {
            "backend_sha256": fingerprint(tokenizer.backend_tokenizer.to_str()),
            "chat_template_sha256": fingerprint(tokenizer.chat_template),
            "response_template_sha256": fingerprint(tokenizer.response_template),
        },
        "cached_metadata_files": {
            hf_hub_download(
                MODEL_ID, name, revision=MODEL_REVISION, local_files_only=True
            ): hashlib.sha256(
                Path(
                    hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_files_only=True)
                ).read_bytes()
            ).hexdigest()
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
            )
        },
        "limits": {
            "context_tokens": SETTINGS.context_tokens,
            "max_tokens": SETTINGS.max_tokens,
            "max_generated_tokens": SETTINGS.max_generated_tokens,
        },
        "tasks": tasks,
        "limitations": (
            "Golden scripted/native fit is feasibility evidence. It does not guarantee that every "
            "sampled model trajectory terminates or fits. Portable golden actions are rendered "
            "with a disclosed short reasoning span because thinking is enabled."
        ),
    }
    save_json(destination, result)
    if result["status"] != "completed":
        raise RuntimeError("Cached-tokenizer preflight did not fit the frozen limits")
    binding = {"preparation": preparation, "preflight": result, "schema_version": 2}
    binding["identity"] = fingerprint(binding)
    if (run_dir / "plan.json").exists():
        raise FileExistsError("Final run binding already exists")
    save_json(run_dir / "plan.json", binding)
    return result


def _require_ready(run_dir):
    from writing_agent.grpo_probe_data import load_probe_release

    run_dir = Path(run_dir).resolve()
    preparation = _validate_preparation(run_dir)
    preflight = _read(run_dir / "preflight.json")
    binding = _read(run_dir / "plan.json")
    expected = {"preparation": preparation, "preflight": preflight, "schema_version": 2}
    expected["identity"] = fingerprint(expected)
    if binding != expected or preflight["preparation_identity"] != fingerprint(preparation):
        raise ValueError("Immutable run binding changed")
    for path, digest in preflight["cached_metadata_files"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Pinned cached tokenizer/model metadata changed")
    release = load_probe_release(run_dir / "release")
    if (
        preflight.get("status") != "completed"
        or [t["task_id"] for t in preflight["tasks"]] != [t["id"] for t in release["tasks"]]
        or not all(t["fits"] and t["labels_excluded"] for t in preflight["tasks"])
    ):
        raise ValueError("Completed nine-task tokenizer feasibility evidence required")
    return release


def _completed_training(run_dir, step, resumed_step):
    experiment = _read(run_dir / "training" / "experiment.json")
    if fingerprint(experiment["manifest"]) != experiment["identity"]:
        raise ValueError("Training experiment identity is invalid")
    release = _require_ready(run_dir)
    expected = inspect_grpo(
        release["train"],
        run_dir / "training",
        settings=SETTINGS,
        reward_spec=REWARD_SPEC,
        admission=release["admission"],
    )
    if experiment["manifest"]["plan"] != expected:
        raise ValueError("Training experiment differs from bound run")
    records = []
    for path in (run_dir / "training" / "invocations").glob("*/complete.json"):
        record = _read(path)
        if record["global_step"] == step and record["resumed_step"] == resumed_step:
            if record["identity"] != experiment["identity"]:
                raise ValueError("Completed invocation belongs to another experiment")
            if Path(record["adapter"]).resolve() != (path.parent / "adapter").resolve():
                raise ValueError("Invocation adapter path changed")
            seal = _verify_adapter(Path(record["adapter"]))
            if seal["identity"] != experiment["identity"]:
                raise ValueError("Export belongs to another experiment")
            evidence = _read(path.parent / "training-evidence.json")
            if evidence["run_identity"] != _read(run_dir / "plan.json")["identity"]:
                raise ValueError("Training evidence is not bound to this run")
            if not record["trainable_changed"]:
                raise ValueError("Completed training did not change parameters")
            if evidence["trainable_change"] != training_change_evidence(record):
                raise ValueError("Training parameter evidence differs from invocation metadata")
            if step == 3:
                initial = _completed_training(run_dir, 1, 0)
                if record["trainable_before"] != initial["trainable_after"]:
                    raise ValueError("Resume before-state differs from initial step-1 export")
                from writing_agent.grpo import verify_checkpoint

                if verify_checkpoint(Path(record["checkpoint"]), experiment["identity"]) != 3:
                    raise ValueError("Final training checkpoint is not step 3")
            records.append(record)
    if len(records) != 1:
        raise ValueError("Exactly one completed matching training invocation is required")
    return records[0]


def _require_base(run_dir):
    complete = _read(run_dir / "base-eval" / "complete.json")
    if complete["run_identity"] != _read(run_dir / "plan.json")["identity"]:
        raise ValueError("Base evaluation belongs to another run")
    if complete["status"] != "completed" or complete["planned_attempts"] != 12:
        raise ValueError("Completed base evaluation of all twelve planned attempts required")
    if complete["conditions"] != [evaluation_config(seed) for seed in DEVELOPMENT_SEEDS]:
        raise ValueError("Base evaluation conditions changed")
    if complete["files"] != file_hashes(run_dir / "base-eval", exclude=("complete.json",)):
        raise ValueError("Base evaluation artifacts changed")
    _verify_scores(run_dir, run_dir / "base-eval")
    return complete


def _verify_scores(run_dir, root):
    from writing_agent.grpo_probe_data import load_probe_release

    release = load_probe_release(run_dir / "release")
    expected = score_evaluation(release["development"], root)
    expected["run_identity"] = _read(run_dir / "plan.json")["identity"]
    if _read(root / "scores.json") != expected:
        raise ValueError("Evaluation scores differ from frozen reward applied to saved attempts")
    return expected


def _require_adapter_evaluation(run_dir):
    _require_base(run_dir)
    training = _completed_training(run_dir, 3, 1)
    adapter = Path(training["adapter"])
    root = run_dir / "adapter-eval"
    complete = _read(root / "complete.json")
    if (
        complete["status"] != "completed"
        or complete["run_identity"] != _read(run_dir / "plan.json")["identity"]
        or complete["planned_attempts"] != 12
        or complete["conditions"] != [evaluation_config(s, adapter) for s in DEVELOPMENT_SEEDS]
        or complete["adapter_seal"] != _verify_adapter(adapter)
        or complete["files"] != file_hashes(root, exclude=("complete.json",))
    ):
        raise ValueError("Adapter evaluation identity or artifacts changed")
    reload = _read(root / "adapter-reload.json")
    if (
        reload["exact"] is not True
        or reload["tensor_count"] <= 0
        or reload["adapter_files"] != file_hashes(adapter)
        or reload["base"] != {"id": MODEL_ID, "revision": MODEL_REVISION}
    ):
        raise ValueError("Verified resident adapter reload evidence required")
    scores = _verify_scores(run_dir, root)
    if _read(root / "paired.json") != pair_evaluations(
        _read(run_dir / "base-eval" / "scores.json"), scores
    ):
        raise ValueError("Paired evaluation report changed")
    return complete


def _verify_adapter(path: Path):
    path = Path(path).resolve()
    marker = json.loads((path / "complete.json").read_text())
    if marker.get("kind") != "inference-adapter":
        raise ValueError("Adapter is not a sealed inference artifact")
    if marker.get("files") != file_hashes(path, exclude=("complete.json",)):
        raise ValueError("Inference adapter changed after sealing")
    return marker


def admit_phase(run_dir: Path, phase: str, *, resume=None, adapter=None, stop_after_step=None):
    """Complete artifact/task admission before a GPU worker or CUDA context exists."""
    run_dir = Path(run_dir).resolve()
    if phase == "prepare":
        destinations = (
            "release",
            "fixture-evidence",
            "prepare.json",
            "preparation-binding.json",
            "plan.json",
        )
        if any((run_dir / name).exists() for name in destinations):
            raise FileExistsError("Prepare destinations already exist")
        return {
            "status": "admitted",
            "phase": phase,
            "plan_identity": frozen_plan(run_dir)["identity"],
        }
    if phase == "preflight":
        from writing_agent.grpo_probe_data import load_probe_release

        release = load_probe_release(run_dir / "release")
        _validate_preparation(run_dir)
        if (run_dir / "preflight.json").exists() or (run_dir / "plan.json").exists():
            raise FileExistsError("Preflight destination already exists")
        return {
            "status": "admitted",
            "phase": phase,
            "task_ids": [task["id"] for task in release["tasks"]],
            "freeze": release["freeze"],
        }
    release = _require_ready(run_dir)
    admission = {
        "status": "admitted",
        "phase": phase,
        "freeze": release["freeze"],
        "task_ids": [task["id"] for task in release["development"]],
    }
    if phase == "base-eval":
        if (run_dir / "base-eval").exists():
            raise FileExistsError("Base-evaluation destination already exists")
    elif phase == "adapter-eval":
        if adapter is None:
            raise ValueError("adapter-eval requires --adapter")
        if (run_dir / "adapter-eval").exists():
            raise FileExistsError("Adapter-evaluation destination already exists")
        _require_base(run_dir)
        record = _completed_training(run_dir, 3, 1)
        if Path(adapter).resolve() != Path(record["adapter"]).resolve():
            raise ValueError("Only this run's completed resumed step-3 export is admissible")
        admission["adapter"] = _verify_adapter(Path(adapter))
    elif phase == "train":
        _require_base(run_dir)
        if stop_after_step not in (None, 1):
            raise ValueError("The frozen initial training phase stops only at step 1")
        if (run_dir / "training").exists():
            raise FileExistsError("Training destination already exists")
        admission["task_ids"] = [task["id"] for task in release["train"]]
        admission["training_plan"] = inspect_grpo(
            release["train"],
            run_dir / "training",
            settings=SETTINGS,
            reward_spec=REWARD_SPEC,
            admission=release["admission"],
        )
        admission["stop_after_step"] = 1
    elif phase == "resume":
        if resume is None:
            raise ValueError("resume requires --resume")
        resume = Path(resume).resolve()
        if resume.parent != (run_dir / "training").resolve():
            raise ValueError("Resume checkpoint must belong to this run's training directory")
        _require_base(run_dir)
        record = _completed_training(run_dir, 1, 0)
        if resume != Path(record["checkpoint"]).resolve():
            raise ValueError("Resume must use the initial completed invocation's checkpoint")
        if any((run_dir / "training" / "groups").glob("*/stopped.json")):
            raise ValueError(
                "Stopped training group requires review; automatic resampling forbidden"
            )
        from writing_agent.grpo import verify_checkpoint

        admission["checkpoint_step"] = verify_checkpoint(
            resume, _read(run_dir / "training" / "experiment.json")["identity"]
        )
        if admission["checkpoint_step"] != 1:
            raise ValueError("The frozen resume phase requires the saved step-1 checkpoint")
        if any((run_dir / "training").glob("checkpoint-[23]/complete.json")):
            raise ValueError("A later complete checkpoint already exists")
        admission["task_ids"] = [task["id"] for task in release["train"]]
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return admission


def evaluate_probe(run_dir: Path, *, adapter: Path | None = None):
    admit_phase(run_dir, "adapter-eval" if adapter is not None else "base-eval", adapter=adapter)
    release = _require_ready(run_dir)
    run_dir = Path(run_dir).resolve()
    root = run_dir / ("adapter-eval" if adapter is not None else "base-eval")
    if root.exists():
        raise FileExistsError("Evaluation destination already exists; results are immutable")
    if adapter is not None:
        adapter = Path(adapter).resolve()
        adapter_marker = _verify_adapter(adapter)
    else:
        adapter_marker = None
    root.mkdir(parents=True)
    finished = False
    try:
        # One residency for both seeds; verify the actual loaded PEFT tensors before sampling.
        with load_checkpoint(evaluation_config(DEVELOPMENT_SEEDS[0], adapter)) as (
            model,
            tokenizer,
            record,
        ):
            if adapter is not None:
                reload = verify_loaded_adapter(model, adapter)
                reload["base"] = {"id": record["id"], "revision": record["revision"]}
                save_json(root / "adapter-reload.json", reload)
            for seed in DEVELOPMENT_SEEDS:
                condition = {**record, "seed": seed}
                run_selected(
                    release["development"],
                    condition,
                    lambda condition=condition: TransformersBackend(model, tokenizer, condition),
                    root / f"seed-{seed}",
                    execute=True,
                    retry_failed=False,
                )
        finished = True
    finally:
        scores = _save_evaluation_scores(run_dir, root)
    complete = {
        "status": "completed" if finished else "interrupted",
        "phase": "adapter-eval" if adapter is not None else "base-eval",
        "run_identity": _read(run_dir / "plan.json")["identity"],
        "planned_attempts": scores["planned_attempts"],
        "adapter_seal": adapter_marker,
        "conditions": [evaluation_config(seed, adapter) for seed in DEVELOPMENT_SEEDS],
        "policy_limit": frozen_plan(run_dir)["evaluation"]["policy_limit"],
        "files": file_hashes(root),
    }
    save_json(root / "complete.json", complete)
    return complete


def _save_evaluation_scores(run_dir, root):
    from writing_agent.grpo_probe_data import load_probe_release

    release = load_probe_release(run_dir / "release")
    scores = score_evaluation(release["development"], root)
    scores["run_identity"] = _read(run_dir / "plan.json")["identity"]
    save_json(root / "scores.json", scores)
    if root.name == "adapter-eval":
        base = _read(run_dir / "base-eval" / "scores.json")
        save_json(root / "paired.json", pair_evaluations(base, scores))
    return scores


def verify_loaded_adapter(model, adapter):
    """Compare saved export against the actual resident adapter, without another base."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    saved = load_file(str(Path(adapter) / "adapter_model.safetensors"))
    resident = get_peft_model_state_dict(model)
    if saved.keys() != resident.keys() or any(
        not saved[key].equal(resident[key].detach().cpu()) for key in saved
    ):
        raise ValueError("Loaded adapter tensor keys/values do not match the saved export")
    return {"exact": True, "tensor_count": len(saved), "adapter_files": file_hashes(adapter)}


def training_change_evidence(result):
    """Use the trainer's post-restore snapshot; checkpoint 1 may already be pruned."""
    before, after = result["trainable_before"], result["trainable_after"]
    if before["tensor_count"] != after["tensor_count"]:
        raise ValueError("Trainable tensor count changed")
    changed = before["sha256"] != after["sha256"]
    if not changed or after["lora_B_l1"] <= 0:
        raise RuntimeError("Optimizer steps did not further change trainable tensors")
    return {
        "changed": True,
        "before": before,
        "after": after,
        "resumed_step": result["resumed_step"],
        "global_step": result["global_step"],
    }


def score_evaluation(tasks, root):
    """Score every saved attempt; retain all twelve planned slots even on interruption."""
    from writing_agent.grpo_probe_data import mechanical_probe_reward

    rows = []
    for seed in DEVELOPMENT_SEEDS:
        records = saved_results(root / f"seed-{seed}")
        for task in tasks:
            matches = [r for r in records if r["scenario_id"] == task["id"]]
            if len(matches) > 1 or any(len(r.get("attempt_history", [])) > 1 for r in matches):
                raise ValueError("Unexpected repeated attempt in a no-retry evaluation")
            result = matches[0] if matches else None
            reward = None
            error = None
            if result is not None:
                try:
                    reward = asdict(mechanical_probe_reward(task, result))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "task_id": task["id"],
                    "seed": seed,
                    "status": result["status"] if result else "not-started",
                    "result_hash": fingerprint(result) if result else None,
                    "attempt_path": result.get("path") if result else None,
                    "reward": reward,
                    "reward_error": error,
                }
            )
    return {
        "planned_attempts": len(tasks) * len(DEVELOPMENT_SEEDS),
        "attempts": rows,
        "completed": sum(r["status"] == "completed" for r in rows),
        "unfinished": sum(r["status"] in {"not-started", "interrupted"} for r in rows),
        "failed": sum(r["status"] not in {"completed", "not-started", "interrupted"} for r in rows),
        "unavailable_rewards": sum(not r["reward"] or r["reward"]["status"] != "ok" for r in rows),
        "semantic_status": "unjudged",
    }


def pair_evaluations(base, adapter):
    if base["run_identity"] != adapter["run_identity"]:
        raise ValueError("Evaluation run identities differ")
    before = {(r["task_id"], r["seed"]): r for r in base["attempts"]}
    after = {(r["task_id"], r["seed"]): r for r in adapter["attempts"]}
    if before.keys() != after.keys():
        raise ValueError("Evaluation planned task/seed pairs differ")
    rows = []
    for key, left in before.items():
        right = after[key]
        usable = all(r["reward"] and r["reward"]["status"] == "ok" for r in (left, right))
        rows.append(
            {
                "task_id": key[0],
                "seed": key[1],
                "base": left,
                "adapter": right,
                "mechanical_delta": right["reward"]["value"] - left["reward"]["value"]
                if usable
                else None,
            }
        )
    return {
        "run_identity": base["run_identity"],
        "pairs": rows,
        "planned_pairs": len(rows),
        "base_counts": {k: v for k, v in base.items() if k != "attempts"},
        "adapter_counts": {k: v for k, v in adapter.items() if k != "attempts"},
        "interpretation": "Mechanical engineering signal only; semantics and quality unjudged",
    }


def train_probe(
    run_dir: Path,
    *,
    resume: Path | None = None,
    stop_after_step: int | None = None,
):
    from writing_agent.grpo_probe_data import mechanical_probe_reward

    admit_phase(
        run_dir,
        "resume" if resume is not None else "train",
        resume=resume,
        stop_after_step=stop_after_step,
    )
    release = _require_ready(run_dir)
    run_dir = Path(run_dir).resolve()
    output = run_dir / "training"
    phase = "resume" if resume is not None else "train"
    if resume is None:
        stop_after_step = 1
    initial = _completed_training(run_dir, 1, 0) if resume is not None else None
    plan = inspect_grpo(
        release["train"],
        output,
        settings=SETTINGS,
        reward_spec=REWARD_SPEC,
        admission=release["admission"],
    )
    if resume is None and output.exists():
        raise FileExistsError("Training destination already exists; use explicit resume")
    result = train_grpo(
        release["train"],
        output,
        settings=SETTINGS,
        reward_spec=REWARD_SPEC,
        admission=release["admission"],
        reward_callback=mechanical_probe_reward,
        execute=True,
        resume_from_checkpoint=resume,
        stop_after_steps=stop_after_step,
    )
    evidence = {
        "phase": phase,
        "status": "completed",
        "plan_identity": fingerprint(plan),
        "run_identity": _read(run_dir / "plan.json")["identity"],
        "global_step": result["global_step"],
        "trainable_changed": result["trainable_changed"],
        "trainable_change": training_change_evidence(result),
        "lora_B_zero_before": result["trainable_before"]["lora_B_l1"] == 0,
        "lora_B_l1_after": result["trainable_after"]["lora_B_l1"],
        "adapter": result["adapter"],
        "checkpoint": result["checkpoint"],
    }
    if not evidence["trainable_changed"] or evidence["lora_B_l1_after"] <= 0:
        raise RuntimeError("Optimizer stage completed without a verified LoRA parameter change")
    if resume is not None:
        if result["trainable_before"] != initial["trainable_after"]:
            raise RuntimeError("Restored before-state differs from saved step-1 adapter")
        evidence["change_after_resume"] = training_change_evidence(result)
        if not evidence["change_after_resume"]["changed"]:
            raise RuntimeError("Resumed optimizer steps did not further change adapter tensors")
    save_json(Path(result["adapter"]).parent / "training-evidence.json", evidence)
    return {**result, "evidence": evidence}


def execute_phase(run_dir: Path, phase: str, *, resume=None, adapter=None, stop_after_step=None):
    admit_phase(run_dir, phase, resume=resume, adapter=adapter, stop_after_step=stop_after_step)
    if phase == "prepare":
        return prepare_probe(run_dir)
    if phase == "preflight":
        return preflight_probe(run_dir)
    if phase == "base-eval":
        return evaluate_probe(run_dir)
    if phase == "adapter-eval":
        if adapter is None:
            raise ValueError("adapter-eval requires --adapter")
        return evaluate_probe(run_dir, adapter=Path(adapter))
    if phase == "train":
        if stop_after_step not in (None, 1):
            raise ValueError("The frozen initial training phase stops only at step 1")
        return train_probe(run_dir, stop_after_step=1)
    if phase == "resume":
        if resume is None:
            raise ValueError("resume requires --resume")
        return train_probe(run_dir, resume=Path(resume))
    raise ValueError(f"Phase is not executable: {phase}")


def _load_ledger(run_dir):
    path = Path(run_dir) / "resource-ledger.json"
    if not path.exists():
        return {"schema_version": 1, "gpu_budget_seconds": GPU_BUDGET_SECONDS, "stages": []}
    return json.loads(path.read_text())


def _gpu_seconds(ledger):
    return sum(
        float(stage.get("elapsed_seconds", stage["timeout_seconds"]))
        if stage.get("status") != "running"
        else float(stage.get("reserved_seconds", stage["timeout_seconds"]))
        for stage in ledger.get("stages", [])
        if stage.get("phase") in GPU_PHASES
    )


def _check_ledger(ledger):
    if any(s["status"] == "running" for s in ledger["stages"]):
        raise RuntimeError("Unreconciled running stage; allowance reserved pending manual review")
    if _gpu_seconds(ledger) >= GPU_BUDGET_SECONDS:
        raise RuntimeError("Aggregate GPU-stage budget is exhausted")


def _gpu_preflight():
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("Probe requires exactly one visible GPU")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    occupants = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    index, name, total, free = [part.strip() for part in rows[0].split(",", 3)]
    allowed = _display_consumers(occupants, int(free))
    return {
        "index": int(index),
        "name": name,
        "total_bytes": int(total) * 1024 * 1024,
        "free_bytes": int(free) * 1024 * 1024,
        "allowed_display_consumers": allowed,
        "display_policy": DISPLAY_POLICY,
    }


def _disk_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def supervise(
    run_dir: Path,
    phase: str,
    worker_command: list[str],
    *,
    admission: dict | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    grace_seconds: float = 10.0,
):
    """Run exactly one child process group and durably account for its resources."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "supervisor.lock").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another supervisor owns this run") from exc
        return _supervise_locked(
            run_dir,
            phase,
            worker_command,
            admission=admission,
            timeout_seconds=timeout_seconds,
            grace_seconds=grace_seconds,
        )


def _stop_child(process, grace_seconds):
    # The new session's process group belongs exclusively to this invocation.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # Also reap a surviving descendant if its group leader exited on SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _supervise_locked(run_dir, phase, worker_command, *, admission, timeout_seconds, grace_seconds):
    ledger = _load_ledger(run_dir)
    _check_ledger(ledger)
    if any(stage["phase"] == phase for stage in ledger["stages"]):
        raise RuntimeError("Stage already attempted; retries require a separately reviewed run")
    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or not math.isfinite(grace_seconds)
        or grace_seconds < 0
    ):
        raise ValueError("Timeout must be finite and positive; grace must be nonnegative")
    used = _gpu_seconds(ledger)
    remaining = GPU_BUDGET_SECONDS - used
    if phase in GPU_PHASES and remaining <= 0:
        raise RuntimeError("Aggregate GPU-stage budget is exhausted")
    timeout = (
        min(float(timeout_seconds), remaining - grace_seconds)
        if phase in GPU_PHASES
        else float(timeout_seconds)
    )
    if timeout <= 0:
        raise RuntimeError("Remaining budget cannot cover shutdown grace")
    gpu = _gpu_preflight() if phase in GPU_PHASES else None
    entry = {
        "phase": phase,
        "identity": fingerprint(
            {"phase": phase, "plan": frozen_plan(run_dir), "argv": worker_command}
        ),
        "started_at": _utc_now(),
        "timeout_seconds": timeout,
        "reserved_seconds": timeout + grace_seconds,
        "gpu_preflight": gpu,
        "admission": admission,
        "status": "running",
    }
    ledger["stages"].append(entry)
    save_json(run_dir / "resource-ledger.json", ledger)
    started = time.monotonic()
    disk_before = _disk_bytes(run_dir)
    existing_results = set((run_dir / "stages").glob(f"{phase}-*.json"))
    process = None
    status = "failed"
    returncode = None
    try:
        process = subprocess.Popen(worker_command, start_new_session=True)
        entry["child_pid"] = process.pid
        save_json(run_dir / "resource-ledger.json", ledger)
        returncode = process.wait(timeout=timeout)
        status = "completed" if returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except BaseException as exc:
        status = "interrupted"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process is not None and status in {"timeout", "interrupted"}:
            _stop_child(process, grace_seconds)
        if phase in {"base-eval", "adapter-eval"}:
            root = run_dir / phase
            if root.exists() and not (root / "scores.json").exists():
                try:
                    # SIGKILL cannot run the worker's finally. Score preserved attempts
                    # offline only after its process group has stopped.
                    _save_evaluation_scores(run_dir, root)
                except Exception as exc:
                    entry["scoring_error"] = f"{type(exc).__name__}: {exc}"
        entry.update(
            ended_at=_utc_now(),
            elapsed_seconds=time.monotonic() - started,
            status=status,
            returncode=returncode,
            process_peak_rss_bytes=None,
            rss_reason="Worker did not deliver a terminal RUSAGE_SELF measurement",
            torch_peak_allocated_bytes=None,
            torch_peak_reserved_bytes=None,
            torch_metrics_reason="No terminal worker CUDA measurement",
            whole_run_disk_bytes_before=disk_before,
            whole_run_disk_bytes_after=_disk_bytes(run_dir),
        )
        entry["whole_run_disk_growth_bytes"] = entry["whole_run_disk_bytes_after"] - disk_before
        result_files = set((run_dir / "stages").glob(f"{phase}-*.json")) - existing_results
        if len(result_files) == 1:
            try:
                result = _read(result_files.pop())
                entry["worker"] = result
                if result.get("status") not in {None, "running"}:
                    for key in (
                        "process_peak_rss_bytes",
                        "torch_peak_allocated_bytes",
                        "torch_peak_reserved_bytes",
                        "torch_metrics_reason",
                    ):
                        entry[key] = result.get(key)
                    entry["rss_reason"] = "Worker RUSAGE_SELF; one phase per process"
                    if status in {"completed", "failed"}:
                        entry["status"] = (
                            result["status"]
                            if returncode == 0
                            else (result["status"] if result["status"] != "completed" else "failed")
                        )
            except (OSError, ValueError) as exc:
                entry["worker_record_error"] = str(exc)
        save_json(run_dir / "resource-ledger.json", ledger)
    if entry["status"] != "completed":
        raise RuntimeError(f"Probe stage ended with status {entry['status']}")
    return entry


def worker_record(run_dir: Path, phase: str, call):
    """Capture structured worker status and Torch peaks without hiding exceptions."""
    run_dir = Path(run_dir).resolve()
    identity = fingerprint({"phase": phase, "plan": frozen_plan(run_dir)})
    destination = run_dir / "stages" / f"{phase}-{identity}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Stage worker record already exists")
    started = time.monotonic()
    record = {"phase": phase, "identity": identity, "started_at": _utc_now(), "status": "running"}
    save_json(destination, record)
    torch = None
    disk_before = _disk_bytes(run_dir)
    try:
        if phase in GPU_PHASES:
            import torch as torch_module

            torch = torch_module
            torch.cuda.reset_peak_memory_stats()
        result = call()
        record.update(status="completed", result=result)
        return result
    except BaseException as exc:
        name = type(exc).__name__
        message = str(exc)
        if name in {"OutOfMemoryError", "MemoryError"} or "out of memory" in message.casefold():
            status = "environment-oom"
        elif name == "GroupPending" and "All-tie" in message:
            status = "pending-tie"
        elif isinstance(exc, (KeyboardInterrupt, SystemExit)):
            status = "interrupted"
        else:
            status = "failed"
        record.update(status=status, error=f"{name}: {message}")
        raise
    finally:
        record.update(
            ended_at=_utc_now(),
            elapsed_seconds=time.monotonic() - started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            whole_run_disk_bytes_before=disk_before,
            whole_run_disk_bytes_after=_disk_bytes(run_dir),
            torch_peak_allocated_bytes=None,
            torch_peak_reserved_bytes=None,
            torch_metrics_reason="CPU stage or CUDA initialization unavailable",
        )
        record["whole_run_disk_growth_bytes"] = record["whole_run_disk_bytes_after"] - disk_before
        try:
            if torch is not None and torch.cuda.is_initialized():
                record["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                record["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                record["torch_metrics_reason"] = None
        except Exception as exc:
            record["torch_metrics_reason"] = str(exc)
        save_json(destination, record)
