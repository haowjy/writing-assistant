"""Privacy-canary and own-lineage evidence for the Phase 8 probe."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from writing_agent.task_graph_probe_tasks import (
    AUTHOR_PACKET_CANARY,
    EVALUATOR_PACKET_CANARY,
)


def _criterion_result(computed: bool, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "computed": bool(computed),
        "passed": bool(computed and passed),
        "evidence": evidence,
        "missing_inputs": [],
    }


def scan_run_privacy(run_dir: Path) -> dict[str, Any]:
    canaries = {
        "unused_author_preference": AUTHOR_PACKET_CANARY,
        "private_evaluator_check_spec": EVALUATOR_PACKET_CANARY,
    }
    private_root = run_dir / "training" / "private"
    hits = []
    checked_files = 0
    for path in run_dir.rglob("*"):
        if path == private_root:
            if path.is_symlink() or not path.is_dir():
                hits.append(
                    {
                        "path": str(path.relative_to(run_dir)),
                        "reason": "private_store_area_not_a_directory",
                    }
                )
            continue
        if private_root in path.parents:
            continue
        if path.is_symlink():
            hits.append({"path": str(path.relative_to(run_dir)), "reason": "symlink_not_scanned"})
            continue
        if not path.is_file():
            continue
        checked_files += 1
        try:
            data = path.read_bytes()
        except OSError:
            hits.append({"path": str(path.relative_to(run_dir)), "reason": "unreadable"})
            continue
        for canary_id, canary in canaries.items():
            if canary.encode() in data:
                hits.append({"path": str(path.relative_to(run_dir)), "canary_id": canary_id})
    return {
        "checked_files": checked_files,
        "hits": hits,
        "canaries_scanned": [
            {"canary_id": canary_id, "sha256": hashlib.sha256(value.encode()).hexdigest()}
            for canary_id, value in canaries.items()
        ],
        "scan_scope": "all regular run-directory files outside training/private",
        "excluded_private_store_area": "training/private",
    }


def verify_member_input_scope(
    groups: list[dict[str, Any]],
    inspector_reports: list[dict[str, Any]],
    *,
    inspections_byte_identical: bool,
) -> dict[str, Any]:
    members = [member for group in groups for member in group["members"]]
    member_bindings_ok = bool(members) and all(
        member["sampler_inputs_bound_to_own_lineage"] for member in members
    )
    admission_members = [member for group in groups for member in group["admission"].members]
    admissions_ok = bool(admission_members) and all(
        member["status"] == "admitted" for member in admission_members
    )
    audit_reports_ok = (
        len(inspector_reports) == 6
        and all(
            report.get("status") == "admitted"
            and report.get("member_count") == 4
            and report.get("admitted_count") == 4
            and report.get("mismatch_count") == 0
            for report in inspector_reports
        )
        and inspections_byte_identical
    )
    return {
        "verified": member_bindings_ok and admissions_ok and audit_reports_ok,
        "member_count": len(members),
        "member_action_ids_bind_to_their_own_lineage": member_bindings_ok,
        "training_admissions_all_admitted": admissions_ok,
        "offline_native_audit_rederived_all_member_inputs": audit_reports_ok,
        "scope": (
            "Each member's native sampler inputs are reconstructed from that member's own "
            "verified start checkpoint, event chain, and context; exported prompt and completion "
            "bytes are checked by the offline native audit."
        ),
        "limits": (
            "This proves per-member lineage and input reconstruction, not absence of arbitrary "
            "shared public text or any unplanted sibling-derived content."
        ),
    }


def criterion_6(
    *,
    inspections_byte_identical: bool,
    privacy: dict[str, Any],
    frozen_public_task_scope: bool,
    network_disabled: bool,
    sibling_input_scope: dict[str, Any],
) -> dict[str, Any]:
    evidence = {
        "inspections_byte_identical": inspections_byte_identical,
        "privacy_canary_hits": privacy["hits"],
        "privacy_files_scanned": privacy["checked_files"],
        "privacy_canaries_scanned": privacy["canaries_scanned"],
        "privacy_scan_scope": privacy["scan_scope"],
        "excluded_private_store_area": privacy["excluded_private_store_area"],
        "frozen_public_task_scope": frozen_public_task_scope,
        "network_disabled": network_disabled,
        "sibling_input_scope": sibling_input_scope,
    }
    passed = (
        inspections_byte_identical
        and not privacy["hits"]
        and frozen_public_task_scope
        and network_disabled
        and sibling_input_scope["verified"]
    )
    return _criterion_result(True, passed, evidence)
