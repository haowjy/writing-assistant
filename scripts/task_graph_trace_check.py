"""Run the one-group native Gemma CPU trace check for Phase 8 S11.

Production invocation (the model must already be cached):

    timeout 3600 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \\
      PYTHONPATH=src "$W/env-phase8/bin/python" scripts/task_graph_trace_check.py \\
      "$W/runs/trace-check-v1"

The command is terminal: on a halt, preserve its output directory and the exact delta,
termination, or parse shape in ``halt.json``. Never retry in the same directory. Fix S7a,
S7b, S4, or S10 in a new commit with a test, use a new path (for example ``trace-check-v2``),
and record every attempt in ``$W/runs/trace-check-attempts.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

TASK_CONFIG = "configs/phase8/probe-tasks/t1-lighthouse.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.task_graph_trace_check_support import (  # noqa: E402
    MAX_CONTEXT_TOKENS,
    MAX_GENERATED_TOKENS,
    MAX_TOKENS_PER_DECISION,
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_ROOT,
    SampleTrace,
    adapter_tensor_hash,
    classify_protocol_shape,
    decision_summaries,
    instrument_generation_time,
    load_model_and_tokenizer,
    member_summaries,
    offline_cpu_environment,
    on_policy_drift,
    peak_rss_bytes,
    prefill_metrics,
    same_incomplete_reason,
    tokenizer_files,
    tool_result_protocol_errors,
    trace_completion_outcome,
    trace_events_for_artifact,
    with_context_cap_for_local_model,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one two-member native task-graph group on CPU with cached Gemma E2B, "
            "then export, audit, admit, inspect twice, and report drift and timing."
        ),
        epilog=(
            "A halt is terminal evidence. Preserve the output directory; report the exact "
            "delta, termination, or parse shape. Never retry there. A fix belongs in S7a, "
            "S7b, S4, or S10 as a new commit with a test; use trace-check-v2 (or the next "
            "new path) and record every attempt in "
            "$W/runs/trace-check-attempts.md. The real invocation must be wrapped in "
            "timeout 3600 with HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, and "
            "CUDA_VISIBLE_DEVICES=''."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output", type=Path, help="new evidence directory; it must not exist")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without importing model libraries, reading weights, or writing",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="local Gemma4 checkpoint override, intended only for the tiny integration run",
    )
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        default=TOKENIZER_ROOT,
        help="local pinned Gemma tokenizer snapshot (no network access)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def claim_output_directory(path: Path) -> Path:
    """Atomically claim a new output directory, refusing even an existing symlink."""
    requested = path.expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"trace-check output already exists: {requested}")
    target = requested.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"trace-check output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(mode=0o700, exist_ok=False)
    return target


def _write_halt(
    output_dir: Path, state: dict[str, Any], failure: dict[str, Any], **details
) -> None:
    last_stage = failure.get("stage", state.get("stage"))
    state.update(stage="halt", protocol_shape=failure["protocol_shape"])
    write_json(
        output_dir / "halt.json",
        {
            "schema": 1,
            "status": "halt",
            "group_id": state.get("group_id"),
            "last_stage": last_stage,
            "failure": failure,
            **details,
        },
    )


def _run_trace(args, output_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    from dataclasses import replace
    from types import SimpleNamespace

    from writing_agent.grpo_task_graph import TaskGraphRollouts, TaskGraphTaskV1
    from writing_agent.native_audit import (
        TrainingAuditError,
        inspect_group_offline,
        require_training_admission,
    )
    from writing_agent.native_gemma import (
        NativeGemmaSampleBackend,
        make_native_manifest_descriptors,
    )
    from writing_agent.task_graph_environment import RolloutEnvironment
    from writing_agent.task_graph_gate import LineageGate
    from writing_agent.task_graph_group import GroupCoordinatorV1
    from writing_agent.task_graph_group_contract import derive_group_seed
    from writing_agent.task_graph_local import (
        DeterministicEvaluator,
        LocalTextToolProvider,
        LocalWorkspaceEnvironment,
    )
    from writing_agent.task_graph_ports import RuntimeDependenciesV1
    from writing_agent.task_graph_probe_experiment import (
        bind_native_tokenizer,
        build_admitted_entry,
        load_probe_task,
        persist_entry,
    )
    from writing_agent.task_graph_probe_experiment import (
        settings as probe_settings,
    )
    from writing_agent.task_graph_record_contracts import ContextPolicyV1
    from writing_agent.task_graph_store import TaskGraphStore
    from writing_agent.task_graph_training_records import TrainingBatchV1

    root = Path(__file__).resolve().parents[1]
    config_path = root / TASK_CONFIG
    config = load_probe_task(config_path)
    settings = config["probe_settings"]
    expected = {
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_generated_tokens": MAX_GENERATED_TOKENS,
        "max_tokens_per_decision": MAX_TOKENS_PER_DECISION,
        "max_tool_calls": 8,
        "max_author_calls": 2,
        "max_writer_turns": 6,
    }
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("t1 probe settings no longer match the frozen Phase 8 budgets")
    entry = build_admitted_entry(config)
    if args.model_path is not None:
        entry = with_context_cap_for_local_model(entry, MAX_CONTEXT_TOKENS)
    else:
        node = entry.graph.node(entry.node_id)
        if node.contract.budget_contract.max_context_tokens != MAX_CONTEXT_TOKENS:
            raise RuntimeError(
                "admitted t1 entry does not carry max_context_tokens=4096; "
                "port the S9b build.py fix before the E2B attempt"
            )

    state["stage"] = "load_model"
    torch, model, tokenizer, forced_stop_hook = load_model_and_tokenizer(args)
    model_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    model_id = MODEL_ID if args.model_path is None else "local-gemma4-test"
    model_revision = (
        MODEL_REVISION
        if args.model_path is None
        else hashlib.sha256(
            (args.model_path.expanduser().resolve() / "config.json").read_bytes()
        ).hexdigest()
    )
    tokenizer_hashes = tokenizer_files(args.tokenizer_root.expanduser().resolve())
    renderer, tokenizer_descriptor, decoding = make_native_manifest_descriptors(
        tokenizer,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        tokenizer_files_sha256=tokenizer_hashes,
        template_ref=entry.params.rendering["template_ref"],
        tool_schema_ref=entry.params.rendering["tool_schema_ref"],
        max_tokens_per_decision=MAX_TOKENS_PER_DECISION,
    )
    entry = bind_native_tokenizer(entry, tokenizer_descriptor)
    experiment_identity = {
        "task_id": config["id"],
        "group_size": 2,
        "budgets": expected,
        "model_id": model_id,
        "revision": model_revision,
        "native_sampling_only": True,
    }
    experiment_ref = hashlib.sha256(
        json.dumps(experiment_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    trace = SampleTrace()
    training_output = output_dir / "task-graph"
    gate = LineageGate()
    store = TaskGraphStore(training_output, verifier=gate)
    entry_checkpoint_id = persist_entry(store, entry)
    environment = RolloutEnvironment(store, entry.graph, None, gate, entry.graph.policy)
    task = TaskGraphTaskV1(config["id"], environment, entry_checkpoint_id)
    model_ref_value = {
        "kind": "trace-check-model-v1",
        "model_id": model_id,
        "revision": model_revision,
        "dtype": model_dtype,
        "device": "cpu",
        "checkpoint_path": str(args.model_path.resolve()) if args.model_path else None,
    }
    model_ref = store.put_artifact(model_ref_value)
    native_backend = NativeGemmaSampleBackend(
        model,
        tokenizer,
        manifest_descriptors=(renderer, tokenizer_descriptor, decoding),
        model_ref=model_ref,
    )
    instrument_generation_time(model, trace)

    tools = LocalTextToolProvider()
    dependencies = RuntimeDependenciesV1(
        native_backend,
        LocalWorkspaceEnvironment(tools),
        tools,
        DeterministicEvaluator(),
    )
    runtime_manifest = dependencies.manifest()
    adapter_hash_before = adapter_tensor_hash(model, torch)
    rollout_settings = replace(probe_settings(), group_size=2, max_steps=1, microbatch_size=1)
    group_seed = derive_group_seed(rollout_settings.seed, "task-graph-step", 0)
    trace.member_ordinals = {
        derive_group_seed(group_seed, "writer", ordinal): ordinal
        for ordinal in range(rollout_settings.group_size)
    }

    def traced_backend_factory(*backend_args, **backend_kwargs):
        return trace.backend(NativeGemmaSampleBackend(*backend_args, **backend_kwargs))

    rollouts = TaskGraphRollouts(
        "trace-check",
        tasks=[{"id": config["id"]}],
        settings=rollout_settings,
        output=training_output,
        task_entries=(task,),
        runtime_manifest=runtime_manifest,
        manifest_descriptors=(renderer, tokenizer_descriptor, decoding),
        model_ref=model_ref,
        experiment_identity=experiment_ref,
        base_revision=rollout_settings.revision,
        tokenizer_root=args.tokenizer_root.expanduser().resolve(),
        sample_backend_factory=traced_backend_factory,
    )
    trainer = SimpleNamespace(
        state=SimpleNamespace(global_step=0), model=model, processing_class=tokenizer
    )
    halt = None
    state["stage"] = "trainer_member_loop"
    try:
        rollouts([config["id"]] * rollout_settings.group_size, trainer)
    except Exception as exc:
        ordinal = next(
            (
                event["member_ordinal"]
                for event in reversed(trace.events)
                if event["member_ordinal"] is not None
            ),
            0,
        )
        failure = {
            "member_ordinal": ordinal,
            "type": type(exc).__name__,
            "message": str(exc),
            "stage": state.get("stage"),
            "sample_trace": trace_events_for_artifact(
                [event for event in trace.events if event["member_ordinal"] == ordinal]
            ),
        }
        failure["protocol_shape"] = classify_protocol_shape(failure)
        halt = failure
    spec = rollouts.last_spec
    if spec is None:
        raise RuntimeError("trainer rollout loop did not seal a trace-check group")
    state.update(group_id=spec.group_id, entry_checkpoint_id=entry_checkpoint_id)
    write_json(output_dir / "group.json", {"group_id": spec.group_id, "spec_ref": spec.identity()})
    coordinator = GroupCoordinatorV1(rollouts.environments[config["id"]], session=rollouts.session)
    decision = rollouts.last_decision or coordinator.finalize(spec)
    context_policy = ContextPolicyV1.from_dict(
        store.get_artifact(spec.policy["context_policy_ref"])
    )
    entry_view = environment.verify(environment.open(entry_checkpoint_id))
    scripted_author = store.get_artifact(spec.policy["simulator_ref"])
    if context_policy.mode != "carry" or scripted_author != {
        "implementation": "scripted-author-v1",
        "script_ref": entry_view.node.contract.interaction_contract.script_ref,
    }:
        raise RuntimeError("trainer rollout group differs from carry/scripted-author policy")
    members = member_summaries(
        store,
        coordinator,
        spec,
        decision,
        trace,
    )
    tool_result_errors = tool_result_protocol_errors(trace.events)
    tool_result_failure = (
        {
            "type": "ProtocolToolResultError",
            "message": "trace contains a protocol-shaped tool result error",
            "protocol_shape": "tool_result",
            "tool_result_errors": tool_result_errors,
        }
        if tool_result_errors
        else None
    )
    report: dict[str, Any] = {
        "schema": 1,
        "status": "halt" if halt else "running",
        "task_id": config["id"],
        "model_id": model_id,
        "model_revision": model_revision,
        "model_dtype": model_dtype,
        "real_weights": args.model_path is None,
        "model_checkpoint": str(args.model_path.resolve()) if args.model_path else None,
        "device": "cpu",
        "group_id": spec.group_id,
        "decision_ref": decision.identity(),
        "group_status": decision.status,
        "group_reason": decision.reason,
        "group_policy": {
            "context_policy": context_policy.mode,
            "scripted_author_ref": spec.policy["simulator_ref"],
        },
        "budgets": expected,
        "members": members,
        "tool_result_protocol_errors": tool_result_errors,
        "both_members_incomplete_same_reason": same_incomplete_reason(members) is not None,
        "same_incomplete_reason": same_incomplete_reason(members),
        "s10_revision_required": same_incomplete_reason(members) is not None,
        "decision_generate_times": decision_summaries(trace.events),
        "re_prefill": prefill_metrics(trace.events),
        "adapter_hash_before": adapter_hash_before,
        "adapter_hash_after": None,
        "inspector_byte_identical": None,
        "on_policy_drift": None,
        "halt": halt,
    }
    state["report"] = report
    write_json(output_dir / "report.json", report)
    if halt:
        if tool_result_failure is not None:
            halt = tool_result_failure
            report["halt"] = halt
        report["status"] = "halt"
        _write_halt(output_dir, state, halt, members=members)
        write_json(output_dir / "report.json", report)
        return report
    if decision.status not in {"ready", "tie"}:
        halt = tool_result_failure or {
            "type": "GroupDecisionV1",
            "message": f"group finalized {decision.status}: {decision.reason}",
            "protocol_shape": "non-protocol-infrastructure",
        }
        report["halt"] = halt
        report["status"] = "halt"
        _write_halt(output_dir, state, halt, members=members)
        write_json(output_dir / "report.json", report)
        return report

    state["stage"] = "inspect_trainer_admission"
    adapter_hash_after = adapter_tensor_hash(model, torch)
    report["adapter_hash_after"] = adapter_hash_after
    admission = rollouts.last_admission
    if admission is None:
        raise RuntimeError("trainer rollout loop did not audit its training batch")
    report["members"] = member_summaries(
        store,
        coordinator,
        spec,
        decision,
        trace,
        admission=admission,
    )
    report["both_members_incomplete_same_reason"] = (
        same_incomplete_reason(report["members"]) is not None
    )
    report["same_incomplete_reason"] = same_incomplete_reason(report["members"])
    report["s10_revision_required"] = report["same_incomplete_reason"] is not None
    report["batch_ref"] = admission.batch_ref
    report["admission_ref"] = admission.identity()

    try:
        require_training_admission(admission)
    except TrainingAuditError as exc:
        refused = [item for item in exc.admission.members if item["status"] != "admitted"]
        if tool_result_failure is not None:
            failure = tool_result_failure
        else:
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "failed_checks": [item["failed_check"] for item in refused],
            }
        failure["protocol_shape"] = getattr(exc, "protocol_shape", classify_protocol_shape(failure))
        report["halt"] = failure
        report["status"] = "halt"
        _write_halt(output_dir, state, failure, members=report["members"])
        write_json(output_dir / "report.json", report)
        return report

    batch = TrainingBatchV1.from_dict(store.get_artifact(admission.batch_ref))
    report["on_policy_drift"] = on_policy_drift(store, batch, model, torch)
    state["stage"] = "offline_inspector"
    inspect_one = inspect_group_offline(store.root, spec.group_id)
    inspect_two = inspect_group_offline(store.root, spec.group_id)
    report["inspector_byte_identical"] = inspect_one == inspect_two
    report["inspection_ref"] = hashlib.sha256(inspect_one).hexdigest()
    report["status"], report["halt"] = trace_completion_outcome(
        tool_result_errors,
        inspector_identical=report["inspector_byte_identical"],
        incomplete_reason=(
            report["same_incomplete_reason"] if report["s10_revision_required"] else None
        ),
    )
    if report["halt"] is None:
        state.update(stage="complete", protocol_shape=None)
    else:
        _write_halt(output_dir, state, report["halt"], members=report["members"])
    write_json(output_dir / "report.json", report)
    if forced_stop_hook is not None:
        forced_stop_hook.remove()
    return report


def _print_report(report: dict[str, Any], *, wall_seconds: float, peak_rss_bytes: int) -> None:
    report["total_wall_seconds"] = wall_seconds
    report["peak_rss_bytes"] = peak_rss_bytes
    report["peak_rss_mib"] = peak_rss_bytes / (1024 * 1024)
    if report.get("both_members_incomplete_same_reason"):
        print(
            "Both members ended incomplete for the same reason "
            f"({report['same_incomplete_reason']}); S10 must be revised before R3.",
            file=sys.stderr,
        )
    print(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "task_config": TASK_CONFIG,
        "device": "cpu",
        "offline": True,
        "group_size": 2,
        "budgets": {
            "max_tokens_per_decision": MAX_TOKENS_PER_DECISION,
            "max_generated_tokens": MAX_GENERATED_TOKENS,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
        },
        "output": str(args.output.expanduser().resolve()),
        "local_model_override": bool(args.model_path),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, sort_keys=True, indent=2))
        return 0

    try:
        output_dir = claim_output_directory(args.output)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    offline_cpu_environment()
    started = time.perf_counter()
    state: dict[str, Any] = {"stage": "startup", "started_unix": time.time(), **plan}
    write_json(output_dir / "run.json", state)

    def on_sigterm(_signum, _frame):
        previous_stage = state.get("stage")
        state.update(stage="halt", protocol_shape="wall_ceiling", signal="SIGTERM")
        report = {
            "schema": 1,
            "status": "halt",
            "halt": {
                "type": "WallCeiling",
                "message": "external timeout sent SIGTERM at the 60-minute ceiling",
                "protocol_shape": "wall_ceiling",
                "last_stage": previous_stage,
                "member_ordinal": state.get("member_ordinal"),
            },
            "total_wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_mib": peak_rss_bytes() / (1024 * 1024),
        }
        write_json(output_dir / "halt.json", {**state, **report})
        print(json.dumps(report, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(124)

    signal.signal(signal.SIGTERM, on_sigterm)
    report = None
    try:
        report = _run_trace(args, output_dir, state)
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "protocol_shape": classify_protocol_shape(
                {"type": type(exc).__name__, "message": str(exc)}
            ),
            "stage": state.get("stage"),
            "member_ordinal": state.get("member_ordinal"),
            "traceback": traceback.format_exc(),
        }
        state.update(stage="halt", protocol_shape=failure["protocol_shape"])
        write_json(output_dir / "halt.json", failure)
        report = {"schema": 1, "status": "halt", "halt": failure}
        print(json.dumps(report, sort_keys=True, indent=2), file=sys.stderr)
    wall_seconds = time.perf_counter() - started
    peak_rss = peak_rss_bytes()
    if report is None:
        report = {"schema": 1, "status": "halt", "halt": {"message": "no report generated"}}
    _print_report(report, wall_seconds=wall_seconds, peak_rss_bytes=peak_rss)
    write_json(output_dir / "report.json", report)
    write_json(output_dir / "summary.json", report)
    if report["status"] == "pass":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
