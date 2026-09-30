"""Strict index of sealed task-graph groups and trainer step reservations."""

from __future__ import annotations

import re
from pathlib import Path

from writing_agent.task_graph import load_canonical_json, validate_hash
from writing_agent.task_graph_record_contracts import GroupError, GroupSpecV1

_STEP_RESERVATION = re.compile(r"step-(\d{6})")
_STEP_LOCK = re.compile(r"\.step-\d{6}\.lock")
_RESERVATION_STATES = frozenset(
    {
        "sealing",
        "sealed",
        "pending",
        "invalid",
        "halted",
        "consumed",
        "audit-refused",
        "adapter-drift",
    }
)
_RESERVATION_FIELDS = frozenset(
    {
        "schema",
        "step",
        "task_id",
        "group_id",
        "status",
        "decision_ref",
        "training_batch_ref",
        "training_admission_ref",
        "failure",
    }
)


def groups_by_sequence(groups_root: Path | str) -> dict[int, GroupSpecV1 | None]:
    """Read sealed specs and step reservations, refusing every unknown root entry."""
    root = Path(groups_root)
    if root.is_symlink():
        raise GroupError("task-graph groups root is not a directory")
    if not root.exists():
        return {}
    if not root.is_dir():
        raise GroupError("task-graph groups root is not a directory")

    result: dict[int, GroupSpecV1 | None] = {}
    reservations: dict[int, dict] = {}
    for child in root.iterdir():
        if child.is_symlink():
            raise GroupError("symlink prevents task-graph group indexing")
        lock = _STEP_LOCK.fullmatch(child.name)
        if lock:
            if not child.is_file():
                raise GroupError("step lock entry is not a file")
            continue

        reservation = _STEP_RESERVATION.fullmatch(child.name)
        if reservation:
            if not child.is_file():
                raise GroupError("step reservation is not a file")
            sequence = int(reservation.group(1))
            try:
                body = load_canonical_json(child.read_bytes())
            except (OSError, TypeError, ValueError) as exc:
                raise GroupError("task-graph step reservation is invalid") from exc
            if (
                not isinstance(body, dict)
                or set(body) - _RESERVATION_FIELDS
                or type(body.get("schema")) is not int
                or body["schema"] != 1
                or type(body.get("step")) is not int
                or body.get("step") != sequence
                or not isinstance(body.get("task_id"), str)
                or re.fullmatch(r"[A-Za-z0-9_-]+", body["task_id"]) is None
                or not isinstance(body.get("status"), str)
                or body.get("status") not in _RESERVATION_STATES
                or ("failure" in body and not isinstance(body["failure"], str))
            ):
                raise GroupError("task-graph step reservation is malformed")
            try:
                for field in (
                    "group_id",
                    "decision_ref",
                    "training_batch_ref",
                    "training_admission_ref",
                ):
                    if field in body:
                        validate_hash(body[field])
            except (TypeError, ValueError) as exc:
                raise GroupError("task-graph step reservation is malformed") from exc
            if sequence in reservations:
                raise GroupError("duplicate task-graph group sequence")
            reservations[sequence] = body
            existing = result.get(sequence)
            if existing is not None and body.get("group_id") not in (None, existing.group_id):
                raise GroupError("step reservation differs from its sealed group")
            if sequence not in result:
                result[sequence] = None
            continue

        if not child.is_dir():
            raise GroupError("unrecognized task-graph group evidence")
        receipt = child / "spec.json"
        if receipt.is_symlink():
            raise GroupError("task-graph group spec receipt is a symlink")
        try:
            spec = GroupSpecV1.from_dict(load_canonical_json(receipt.read_bytes()))
        except (OSError, TypeError, ValueError) as exc:
            raise GroupError("task-graph group spec receipt is invalid") from exc
        if spec.group_id != child.name:
            raise GroupError("task-graph group directory differs from its sealed identity")
        sequence = spec.group_sequence
        if sequence in result and result[sequence] is not None:
            raise GroupError("duplicate task-graph group sequence")
        reservation_body = reservations.get(sequence)
        if reservation_body is not None and reservation_body.get("group_id") not in (
            None,
            spec.group_id,
        ):
            raise GroupError("step reservation differs from its sealed group")
        result[sequence] = spec
    return result


__all__ = ["groups_by_sequence"]
