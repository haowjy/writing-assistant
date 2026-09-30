"""Offline criterion and measurement derivation for the Phase 8 probe runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from writing_agent.catalog import save_json
from writing_agent.task_graph import canonical_bytes, load_canonical_json

CRITERION_DESCRIPTIONS = {
    "criterion_1": (
        "Three ready/tie groups, 12 structurally eligible members, three all-admitted records, "
        "and at least one ready group."
    ),
    "criterion_2": (
        "The LoRA adapter changed, three optimizer steps completed, and every optimizer "
        "gradient was finite."
    ),
    "criterion_3": (
        "Checkpoints 1–3 verify, resume from 2 reaches 3, checkpoint/export tensors agree, "
        "and PEFT reload matches."
    ),
    "criterion_4": (
        "Offline inspection re-derives all ledgers/admissions with zero mismatches and all "
        "tamper controls refuse."
    ),
    "criterion_5": (
        "Every applicable run-time, GPU-memory, RSS, aggregate GPU-time, and run-directory "
        "ceiling held."
    ),
    "criterion_6": (
        "The two offline inspection outputs are byte-identical and private-data canaries "
        "are absent."
    ),
}


def _smoke_helpers():
    """Load the established tiny-Gemma fixture without importing runner orchestration."""
    module_name = "_task_graph_probe_evidence_cpu_smoke"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parents[2] / "scripts" / "smoke_task_graph_grpo_cpu.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("CPU smoke helpers are unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"required evidence is missing or unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"required evidence is not a JSON object: {path}")
    return value


def _group_paths(training_root: Path) -> list[Path]:
    root = training_root / "groups"
    if not root.is_dir():
        raise ValueError(f"task-graph group evidence is missing: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and not path.name.startswith("step-")
    )


def _store(training_root: Path):
    from writing_agent.task_graph_gate import LineageGate
    from writing_agent.task_graph_store import TaskGraphStore

    return TaskGraphStore(training_root, verifier=LineageGate())


def _collect_groups(training_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    from writing_agent.task_graph_group_records import (
        GroupAdvantageV1,
        GroupDecisionV1,
        GroupMemberResultV1,
    )
    from writing_agent.task_graph_records import TrainingAdmissionV1, WriterTurnV2

    paths = _group_paths(training_root)
    store = _store(training_root)
    gate = store.verifier
    groups = []
    errors = []
    for group_dir in paths:
        try:
            spec = _read_json(group_dir / "spec.json")
            consumed = _read_json(group_dir / "trainer-consumed.json")
            admission_receipt = _read_json(group_dir / "training-admission.json")
            batch_receipt = _read_json(group_dir / "training-batch.json")
            group_id = group_dir.name
            decision = GroupDecisionV1.from_dict(store.get_artifact(consumed["decision_ref"]))
            admission = TrainingAdmissionV1.from_dict(
                store.get_artifact(admission_receipt["admission_ref"])
            )
            if (
                decision.group_id != group_id
                or admission.group_id != group_id
                or admission.batch_ref != batch_receipt["batch_ref"]
                or admission.batch_ref != consumed["training_batch_ref"]
                or admission.identity() != admission_receipt["admission_ref"]
            ):
                raise ValueError("group receipts differ from their content-addressed records")
            result_members = []
            terminations: dict[str, int] = {}
            for result_ref in decision.member_result_refs:
                if result_ref is None:
                    raise ValueError("group decision has a missing member result")
                member = GroupMemberResultV1.from_dict(store.get_artifact(result_ref))
                if member.final_checkpoint_id is None:
                    raise ValueError("member has no final checkpoint")
                view = gate.view(store, member.final_checkpoint_id)
                # The member's terminal-outcome ref is an upstream pending snapshot;
                # the verified final state is derived by the lineage gate.
                eligibility = view.outcome.training_eligibility
                if not eligibility:
                    raise ValueError("member has no derived training eligibility")
                member_terminations = []
                for sample in view.samples:
                    turn = WriterTurnV2.from_dict(store.get_artifact(sample.turn_ref))
                    key = turn.termination["kind"]
                    if turn.termination.get("limit"):
                        key += ":" + turn.termination["limit"]
                    terminations[key] = terminations.get(key, 0) + 1
                    member_terminations.append(turn)
                result_members.append(
                    {
                        "member_id": member.member_id,
                        "eligibility": eligibility,
                        "turns": member_terminations,
                    }
                )
            advantages = [
                GroupAdvantageV1.from_dict(store.get_artifact(ref))
                for ref in decision.advantage_refs
            ]
            reward_values = [
                Fraction(item.reward["numerator"], item.reward["denominator"])
                for item in advantages
            ]
            groups.append(
                {
                    "group_id": group_id,
                    "spec": spec,
                    "decision": decision,
                    "admission": admission,
                    "members": result_members,
                    "terminations": terminations,
                    "rewards": reward_values,
                }
            )
        except Exception as exc:
            errors.append(f"{group_dir.name}: {type(exc).__name__}: {exc}")
    return groups, errors


def _inspections(training_root: Path, run_dir: Path, groups: list[dict[str, Any]]):
    from writing_agent.native_audit import inspect_group_offline

    out_dir = run_dir / "inspection"
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    passes: list[dict[str, bytes | str]] = [[], []]
    inspector_reports = []
    errors = []
    for round_number in (0, 1):
        for group in groups:
            group_id = group["group_id"]
            try:
                raw = inspect_group_offline(training_root, group_id)
                report = json.loads(raw)
                passes[round_number].append({"group_id": group_id, "bytes_hex": raw.hex()})
                inspector_reports.append(report)
            except Exception as exc:
                detail = f"{group_id}: {type(exc).__name__}: {exc}"
                errors.append(detail)
                passes[round_number].append({"group_id": group_id, "error": detail})
    payloads = [canonical_bytes({"schema": 1, "reports": rows}) for rows in passes]
    for round_number, raw in enumerate(payloads, start=1):
        (out_dir / f"inspect-{round_number}.json").write_bytes(raw + b"\n")
    identical = payloads[0] == payloads[1] and not errors and len(groups) == 3
    return inspector_reports, identical, errors


def _tamper_controls(training_root: Path, run_dir: Path, group: dict[str, Any]) -> dict[str, Any]:
    import base64

    from writing_agent.native_audit import inspect_group_offline
    from writing_agent.task_graph_errors import StoreError
    from writing_agent.task_graph_gate import LineageGate, StoreArtifactReader
    from writing_agent.task_graph_group_records import GroupDecisionV1, GroupMemberResultV1
    from writing_agent.task_graph_records import (
        RuntimeManifestV1,
        RuntimeManifestV2,
        RuntimePortDescriptorV1,
        WriterTurnV2,
    )
    from writing_agent.task_graph_store import TaskGraphStore

    store = TaskGraphStore(training_root, verifier=LineageGate())
    group_dir = training_root / "groups" / group["group_id"]
    consumed = _read_json(group_dir / "trainer-consumed.json")
    decision = GroupDecisionV1.from_dict(store.get_artifact(consumed["decision_ref"]))
    first_result = GroupMemberResultV1.from_dict(store.get_artifact(decision.member_result_refs[0]))
    first_view = store.verifier.view(store, first_result.final_checkpoint_id)
    if len(first_view.samples) < 2:
        raise ValueError("probe member lacks the required multi-turn tamper-control target")
    reader = StoreArtifactReader(store)
    first_turn = WriterTurnV2.from_dict(reader.artifact(first_view.samples[0].turn_ref))
    second_turn = WriterTurnV2.from_dict(reader.artifact(first_view.samples[1].turn_ref))
    spec_body = _read_json(group_dir / "spec.json")
    manifest = RuntimeManifestV2.from_dict(reader.artifact(spec_body["policy"]["adapter_ref"]))
    tamper_cases = (
        ("generated", first_turn.generated_token_ids_ref, "bytes"),
        ("external", second_turn.input_token_ids_ref, "bytes"),
        ("logprob", first_turn.logprobs["ref"], "bytes"),
        ("policy", spec_body["policy"]["behavior_policy_ref"], "json"),
        ("termination", first_view.samples[0].turn_ref, "termination"),
        ("v1_manifest", spec_body["policy"]["adapter_ref"], "manifest"),
    )
    store_dirs = (
        "artifacts",
        "checkpoints",
        "commits",
        "context_content",
        "context_revisions",
        "events",
        "groups",
        "instances",
        "operations",
        "private",
        "refs",
    )
    controls: dict[str, bool] = {}
    for label, ref, kind in tamper_cases:
        with tempfile.TemporaryDirectory(prefix="tamper-", dir=run_dir / "inspection") as tmp:
            copied = Path(tmp) / "store"
            copied.mkdir(mode=0o700)
            for name in store_dirs:
                source = training_root / name
                if source.exists():
                    shutil.copytree(source, copied / name)
            target = copied / "artifacts" / ref
            envelope = load_canonical_json(target.read_bytes())
            if kind == "bytes":
                raw = bytearray(base64.b64decode(envelope["body"]))
                if not raw:
                    raise ValueError(f"tamper target {label} has no bytes")
                raw[0] ^= 1
                envelope["body"] = base64.b64encode(raw).decode("ascii")
            elif kind == "json":
                envelope["body"] = {"policy": "tampered"}
            elif kind == "termination":
                envelope["body"]["termination"] = {
                    "kind": "token_limit",
                    "stop_token_id": None,
                    "limit": "decision",
                }
            else:
                ports = []
                for port in manifest.ports:
                    config = dict(port.configuration)
                    if port.role == "sampling":
                        config["capabilities"] = ["usage_reporting"]
                    ports.append(
                        RuntimePortDescriptorV1(
                            schema=1,
                            role=port.role,
                            implementation=port.implementation,
                            version=port.version,
                            configuration=config,
                        )
                    )
                envelope["body"] = RuntimeManifestV1(schema=1, ports=tuple(ports)).to_wire()
            target.write_bytes(canonical_bytes(envelope))
            try:
                inspect_group_offline(copied, group["group_id"])
            except StoreError:
                controls[label] = True
            else:
                controls[label] = False
    return {"controls": controls, "all_refused": len(controls) == 6 and all(controls.values())}


def _adapter_reload(training_root: Path, mode: str, complete: dict[str, Any]) -> dict[str, Any]:
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    adapter_dir = Path(complete["adapter"])
    checkpoint = training_root / "checkpoint-3"
    exported_path = adapter_dir / "adapter_model.safetensors"
    checkpoint_path = checkpoint / "adapter_model.safetensors"
    if not exported_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("checkpoint-3 or exported adapter tensors are missing")
    exported = load_file(str(exported_path), device="cpu")
    checkpoint_state = load_file(str(checkpoint_path), device="cpu")
    direct_match = _same_tensors(exported, checkpoint_state)
    smoke = _smoke_helpers()
    if mode.startswith("cpu"):
        tokenizer = AutoTokenizer.from_pretrained(
            str(smoke.TOKENIZER_PATH), local_files_only=True, trust_remote_code=False
        )
        base = smoke.tiny_gemma(tokenizer.vocab_size)
    else:
        from transformers import AutoModelForCausalLM

        settings = smoke.settings()
        base = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": "cpu"},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    try:
        reloaded = PeftModel.from_pretrained(
            base, str(adapter_dir), is_trainable=True, local_files_only=True
        )
        try:
            loaded_state = get_peft_model_state_dict(reloaded, adapter_name="default")
        except TypeError:
            loaded_state = get_peft_model_state_dict(reloaded)
        reload_match = _same_tensors(loaded_state, checkpoint_state)
        from writing_agent.grpo_trainer import trainable_evidence

        resident_hash = trainable_evidence(reloaded)["sha256"]
        return {
            "export_matches_checkpoint": direct_match,
            "peft_reload_matches_checkpoint": reload_match,
            "resident_trainable_hash_matches_export": resident_hash
            == complete.get("trainable_after", {}).get("sha256"),
            "tensor_count": len(checkpoint_state),
            "export_sha256": hashlib.sha256(exported_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        }
    finally:
        del base


def _same_tensors(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(
        left[name].shape == right[name].shape and left[name].equal(right[name]) for name in left
    )


def _checkpoint_evidence(training_root: Path, complete: dict[str, Any]) -> dict[str, Any]:
    from writing_agent.grpo_checkpoint import verify_checkpoint

    experiment = _read_json(training_root / "experiment.json")
    identity = experiment.get("identity")
    if not isinstance(identity, str):
        raise ValueError("training experiment identity is missing")
    verified = []
    for step in (1, 2, 3):
        path = training_root / f"checkpoint-{step}"
        verified.append(verify_checkpoint(path, identity))
    return {
        "verified_steps": verified,
        "resume_from_step": complete.get("resumed_step"),
        "global_step": complete.get("global_step"),
    }


def _criterion(computed: bool, passed: bool, evidence: dict[str, Any], missing=()):
    return {
        "computed": bool(computed),
        "passed": bool(computed and passed),
        "evidence": evidence,
        "missing_inputs": list(missing),
    }


def _resource_stages(run_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ("train", "resume"):
        result[name] = _read_json(run_dir / "stages" / name / "resources.json")
    return result


def _generation_metrics(run_dir: Path) -> dict[str, Any]:
    calls = []
    for stage in ("train", "resume"):
        data = _read_json(run_dir / "stages" / stage / "generation-times.json")
        calls.extend(data.get("decisions", ()))
    seconds = [value["elapsed_seconds"] for value in calls]
    if not seconds or any(not isinstance(value, (int, float)) or value < 0 for value in seconds):
        raise ValueError("per-decision generation timings are missing or invalid")
    return {
        "decision_count": len(seconds),
        "per_decision_generate_time_seconds": seconds,
        "mean_generate_time_seconds": sum(seconds) / len(seconds),
    }


def _observer_metrics(training_root: Path) -> dict[str, Any]:
    diffs = []
    for path in sorted((training_root / "observer").glob("step-*.json")):
        observer = _read_json(path)
        step = observer["step"]
        batch = _read_json(training_root / "batches" / f"step-{step:06d}.json")
        members = batch["members"]
        for record in observer["microbatches"]:
            completion = record["completion_ids"][0]
            valid = record["completion_mask"][0]
            active = record["loss_mask"][0]
            actual_completion = [
                token for token, keep in zip(completion, valid, strict=False) if keep
            ]
            member = next(
                (row for row in members if row["completion_ids"] == actual_completion), None
            )
            if member is None:
                raise ValueError("observer completion does not match the saved training batch")
            sampled = member["sampled_logprobs"]
            recomputed = record["recomputed_logprobs"][0]
            for index, (keep, mask) in enumerate(zip(valid, active, strict=False)):
                if keep and mask:
                    value = sampled[index]
                    if value is None:
                        raise ValueError("masked token lacks sampled logprob evidence")
                    diffs.append(abs(float(value) - float(recomputed[index])))
    if not diffs:
        raise ValueError("no masked-token sampled/recomputed logprob pairs")
    return {"mean": sum(diffs) / len(diffs), "max": max(diffs), "masked_token_count": len(diffs)}


def _ledger_metrics(groups: list[dict[str, Any]]) -> dict[str, Any]:
    prefill = 0
    unique = 0
    terminations: dict[str, int] = {}
    rewards = []
    for group in groups:
        rewards.extend(float(value) for value in group["rewards"])
        for kind, count in group["terminations"].items():
            terminations[kind] = terminations.get(kind, 0) + count
        for member in group["members"]:
            turns = member["turns"]
            if not turns:
                raise ValueError("group member has no native token ledger")
            for turn in turns:
                usage = turn.usage
                prefill += usage["prefill_tokens"]
            unique += turns[-1].input_token_count + turns[-1].generated_token_count
    if unique <= 0 or not rewards:
        raise ValueError("ledger or reward measurements are unavailable")
    return {
        "re_prefill_ratio": prefill / unique,
        "prefill_tokens": prefill,
        "unique_ledger_tokens": unique,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "reward_spread": max(rewards) - min(rewards),
        "termination_classes": terminations,
    }


def _checkpoint_disk_metrics(training_root: Path) -> list[dict[str, int]]:
    measurements = []
    previous = 0
    for step in (1, 2, 3):
        root = training_root / f"checkpoint-{step}"
        size = sum(
            path.lstat().st_size
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        measurements.append({"step": step, "bytes": size, "growth_bytes": size - previous})
        previous = size
    return measurements


def _privacy_scan(run_dir: Path) -> dict[str, Any]:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from tests.task_graph_rollout_fixtures import (
        AUTHOR_PACKET_CANARY,
        EVALUATOR_PACKET_CANARY,
        LEDGER_CANARY,
    )

    canaries = (AUTHOR_PACKET_CANARY, EVALUATOR_PACKET_CANARY, LEDGER_CANARY)
    hits = []
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            hits.append({"path": str(path.relative_to(run_dir)), "reason": "unreadable"})
            continue
        for canary in canaries:
            if canary.encode() in data:
                hits.append({"path": str(path.relative_to(run_dir)), "canary": canary})
    return {"checked_files": sum(1 for path in run_dir.rglob("*") if path.is_file()), "hits": hits}


def _failed_result(run_dir: Path, mode: str, error: str) -> dict[str, Any]:
    criteria = {
        f"criterion_{number}": {
            **_criterion(False, False, {}, [error]),
            "description": CRITERION_DESCRIPTIONS[f"criterion_{number}"],
        }
        for number in range(1, 7)
    }
    return {
        "schema": 1,
        "run_id": run_dir.name,
        "execution_mode": mode,
        "verdict": "fail",
        "criteria": criteria,
        "measurements": {},
        "failures": [error],
    }


def inspect_run(run_dir: Path, *, mode: str) -> dict[str, Any]:
    """Run two offline inspections and write a non-vacuous six-criterion verdict."""
    started = time.monotonic()
    result: dict[str, Any]
    try:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE"):
            if os.environ.get(name) != "1":
                raise ValueError(f"offline inspection requires {name}=1")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("offline inspection requires CUDA_VISIBLE_DEVICES=''")
        try:
            prepared = json.loads((run_dir / "prepare.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("prepare.json is missing or unreadable") from exc
        if prepared.get("recipe", {}).get("execution_mode") != mode:
            raise ValueError("inspect-run execution mode differs from prepare.json")
        training_root = run_dir / "training"
        group_paths = _group_paths(training_root)
        groups, group_errors = _collect_groups(training_root)
        reports, inspections_identical, inspection_errors = _inspections(
            training_root, run_dir, groups
        )
        all_group_inputs = not group_errors and len(groups) == len(group_paths) == 3
        group_statuses = [group["decision"].status for group in groups]
        structural_count = sum(
            member["eligibility"] == "structurally_eligible"
            for group in groups
            for member in group["members"]
        )
        admitted_records = [
            len(group["admission"].members) == 4
            and all(member["status"] == "admitted" for member in group["admission"].members)
            for group in groups
        ]
        ready_count = group_statuses.count("ready")
        tie_count = group_statuses.count("tie")
        criterion_1_evidence = {
            "group_count": len(group_paths),
            "group_statuses": group_statuses,
            "structurally_eligible_members": structural_count,
            "member_count": sum(len(group["members"]) for group in groups),
            "all_admitted_records": admitted_records,
            "ready_group_count": ready_count,
            "structural_and_admission": (
                all_group_inputs
                and len(group_statuses) == 3
                and all(status in {"ready", "tie"} for status in group_statuses)
                and structural_count == 12
                and sum(len(group["members"]) for group in groups) == 12
                and len(admitted_records) == 3
                and all(admitted_records)
            ),
        }
        criterion_1_pass = criterion_1_evidence["structural_and_admission"] and ready_count >= 1

        invocation_completes = sorted((training_root / "invocations").glob("*/complete.json"))
        complete_records = [_read_json(path) for path in invocation_completes]
        final = next((item for item in complete_records if item.get("global_step") == 3), None)
        initial = next((item for item in complete_records if item.get("global_step") == 2), None)
        gradient_reports = [
            _read_json(run_dir / "stages" / stage / "gradient-observer.json")
            for stage in ("train", "resume")
        ]
        gradients_finite = all(
            report.get("finite") is True and report.get("gradient_tensor_count", 0) > 0
            for report in gradient_reports
        )
        optimizer_steps = final.get("global_step") if final else None
        adapter_changed = bool(
            initial
            and final
            and initial.get("trainable_before", {}).get("sha256")
            != final.get("trainable_after", {}).get("sha256")
        )
        criterion_2_evidence = {
            "optimizer_steps": optimizer_steps,
            "gradients_finite": gradients_finite,
            "gradient_step_counts": [report.get("step_count") for report in gradient_reports],
            "adapter_changed": adapter_changed,
        }
        criterion_2_computed = (
            final is not None and len(gradient_reports) == 2 and len(complete_records) == 2
        )
        criterion_2_pass = (
            criterion_2_computed
            and optimizer_steps == 3
            and gradients_finite
            and adapter_changed
            and [report.get("step_count") for report in gradient_reports] == [2, 1]
        )

        criterion_3_evidence: dict[str, Any] = {}
        criterion_3_error = None
        try:
            checkpoint_evidence = _checkpoint_evidence(training_root, final or {})
            reload_evidence = _adapter_reload(training_root, mode, final or {})
            criterion_3_evidence = {**checkpoint_evidence, **reload_evidence}
        except Exception as exc:
            criterion_3_error = f"{type(exc).__name__}: {exc}"
        criterion_3_computed = not criterion_3_error
        criterion_3_pass = bool(
            criterion_3_computed
            and criterion_3_evidence.get("verified_steps") == [1, 2, 3]
            and criterion_3_evidence.get("resume_from_step") == 2
            and criterion_3_evidence.get("global_step") == 3
            and criterion_3_evidence.get("export_matches_checkpoint") is True
            and criterion_3_evidence.get("peft_reload_matches_checkpoint") is True
            and criterion_3_evidence.get("resident_trainable_hash_matches_export") is True
        )

        tamper_evidence: dict[str, Any] = {}
        tamper_error = None
        try:
            if not groups:
                raise ValueError("no group is available for tamper controls")
            tamper_evidence = _tamper_controls(training_root, run_dir, groups[0])
        except Exception as exc:
            tamper_error = f"{type(exc).__name__}: {exc}"
        mismatch_counts = [report.get("mismatch_count") for report in reports]
        inspector_ok = (
            all(
                report.get("status") == "admitted"
                and report.get("member_count") == 4
                and report.get("admitted_count") == 4
                and report.get("mismatch_count") == 0
                for report in reports
            )
            and len(reports) == 3
        )
        criterion_4_evidence = {
            "inspection_reports": reports,
            "mismatch_counts": mismatch_counts,
            "inspections_byte_identical": inspections_identical,
            "tamper_controls": tamper_evidence,
            "inspection_errors": inspection_errors + group_errors,
        }
        criterion_4_computed = (
            bool(reports)
            and not tamper_error
            and not inspection_errors
            and not group_errors
            and len(reports) == 3
        )
        criterion_4_pass = bool(
            criterion_4_computed
            and inspector_ok
            and inspections_identical
            and tamper_evidence.get("all_refused") is True
        )
        if tamper_error:
            criterion_4_evidence["tamper_error"] = tamper_error

        resources = _resource_stages(run_dir)
        ceilings = prepared["ceilings"]
        stage_resource_ok = True
        stage_evidence = {}
        for stage, record in resources.items():
            reserved = record.get("torch_peak_reserved_bytes")
            rss = record.get("peak_rss_bytes")
            gpu = mode == "gpu"
            ok = (
                record.get("status") == "completed"
                and not record.get("ceilings_exceeded")
                and isinstance(rss, int)
                and rss <= ceilings["peak_rss_bytes"]
                and record.get(
                    "run_directory_growth_bytes", ceilings["run_directory_growth_bytes"] + 1
                )
                <= ceilings["run_directory_growth_bytes"]
            )
            if gpu:
                ok = (
                    ok
                    and isinstance(reserved, int)
                    and reserved <= ceilings["peak_reserved_gpu_bytes"]
                    and record.get("nvml_before") is not None
                    and record.get("nvml_after") is not None
                    and record.get("admission", {}).get("admitted") is True
                    and record.get("gpu_seconds_after", ceilings["aggregate_gpu_seconds"] + 1)
                    <= ceilings["aggregate_gpu_seconds"]
                )
            stage_resource_ok = stage_resource_ok and ok
            stage_evidence[stage] = {
                "status": record.get("status"),
                "elapsed_seconds": record.get("elapsed_seconds"),
                "peak_rss_bytes": rss,
                "peak_reserved_gpu_bytes": reserved,
                "gpu_memory_applicable": gpu,
                "disk_growth_bytes": record.get("run_directory_growth_bytes"),
                "ceilings_exceeded": record.get("ceilings_exceeded"),
                "admission": record.get("admission"),
                "gpu_seconds_after": record.get("gpu_seconds_after"),
            }
        current_run_bytes = _disk_bytes(run_dir)
        disk_ok = current_run_bytes < ceilings["run_directory_growth_bytes"]
        criterion_5_evidence = {
            "ceilings": ceilings,
            "train_resume_stages": stage_evidence,
            "run_directory_bytes_before_result": current_run_bytes,
            "run_directory_bytes_limit": ceilings["run_directory_growth_bytes"],
            "gpu_memory_not_applicable": mode != "gpu",
        }
        criterion_5_computed = len(resources) == 2
        criterion_5_pass = criterion_5_computed and stage_resource_ok and disk_ok

        privacy = _privacy_scan(run_dir)
        source_scope_ok = len(prepared.get("task_graph_hashes", ())) == 3 and [
            item["task_id"] for item in prepared["task_graph_hashes"]
        ] == ["t1-lighthouse", "t2-winter-garden", "t3-coastal-post"]
        criterion_6_evidence = {
            "inspections_byte_identical": inspections_identical,
            "privacy_canary_hits": privacy["hits"],
            "privacy_files_scanned": privacy["checked_files"],
            "frozen_public_task_scope": source_scope_ok,
            "network_disabled": all(
                os.environ.get(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
            ),
        }
        criterion_6_pass = (
            inspections_identical
            and not privacy["hits"]
            and source_scope_ok
            and criterion_6_evidence["network_disabled"]
        )

        measurements = _ledger_metrics(groups)
        measurements.update(_generation_metrics(run_dir))
        measurements["on_policy_drift"] = _observer_metrics(training_root)
        measurements["tie_count"] = tie_count
        measurements["group_count"] = len(group_paths)
        measurements["stage_runtime_seconds"] = {
            stage: record.get("elapsed_seconds") for stage, record in resources.items()
        }
        measurements["stage_disk_growth_bytes"] = {
            stage: record.get("run_directory_growth_bytes") for stage, record in resources.items()
        }
        measurements["disk_growth_per_checkpoint"] = _checkpoint_disk_metrics(training_root)
        criteria = {
            "criterion_1": _criterion(
                all_group_inputs, criterion_1_pass, criterion_1_evidence, group_errors
            ),
            "criterion_2": _criterion(
                criterion_2_computed,
                criterion_2_pass,
                criterion_2_evidence,
                [] if criterion_2_computed else ["step-3 completion and gradient observer"],
            ),
            "criterion_3": _criterion(
                criterion_3_computed,
                criterion_3_pass,
                criterion_3_evidence,
                [] if criterion_3_computed else [criterion_3_error],
            ),
            "criterion_4": _criterion(
                criterion_4_computed,
                criterion_4_pass,
                criterion_4_evidence,
                [] if criterion_4_computed else [tamper_error or "offline inspection evidence"],
            ),
            "criterion_5": _criterion(criterion_5_computed, criterion_5_pass, criterion_5_evidence),
            "criterion_6": _criterion(True, criterion_6_pass, criterion_6_evidence),
        }
        for key, description in CRITERION_DESCRIPTIONS.items():
            criteria[key]["description"] = description
        failures = []
        if group_errors:
            failures.extend(group_errors)
        if inspection_errors:
            failures.extend(inspection_errors)
        if tamper_error:
            failures.append(tamper_error)
        if criterion_3_error:
            failures.append(criterion_3_error)
        result = {
            "schema": 1,
            "run_id": run_dir.name,
            "execution_mode": mode,
            "prepared_source": prepared.get("source"),
            "verdict": "fail",
            "criteria": criteria,
            "measurements": measurements,
            "stages": {
                name: {"status": record.get("status"), "resources": record}
                for name, record in resources.items()
            },
            "failures": failures,
            "inspect_run_elapsed_seconds": time.monotonic() - started,
        }
        result["verdict"] = _select_verdict(criteria, measurements)
        save_json(run_dir / "result.json", result)
        return result
    except Exception as exc:
        result = _failed_result(run_dir, mode, f"{type(exc).__name__}: {exc}")
        save_json(run_dir / "result.json", result)
        return result


def _disk_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            try:
                total += path.lstat().st_size
            except FileNotFoundError:
                continue
    return total


def _select_verdict(criteria: dict[str, Any], measurements: dict[str, Any]) -> str:
    if all(
        criteria.get(f"criterion_{number}", {}).get("computed") is True for number in range(1, 7)
    ):
        if all(criteria[f"criterion_{number}"].get("passed") is True for number in range(1, 7)):
            return "pass"
    c1 = criteria.get("criterion_1", {})
    c2 = criteria.get("criterion_2", {})
    all_tie = measurements.get("tie_count") == 3 and measurements.get("group_count") == 3
    if (
        all_tie
        and c1.get("computed") is True
        and c1.get("evidence", {}).get("structural_and_admission") is True
        and c2.get("computed") is True
        and c2.get("evidence", {}).get("optimizer_steps") == 3
        and c2.get("evidence", {}).get("gradients_finite") is True
        and all(
            criteria.get(f"criterion_{number}", {}).get("computed") is True
            for number in (3, 4, 5, 6)
        )
        and all(criteria[f"criterion_{number}"].get("passed") is True for number in (3, 4, 5, 6))
    ):
        return "inconclusive_no_signal"
    return "fail"


__all__ = ["inspect_run"]
