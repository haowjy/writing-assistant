"""Admitted evaluator evidence families and replay-only verifiers.

A family is authoritative only when this registry can re-derive its claim from
frozen request inputs. Evaluator implementations never participate in replay.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from writing_agent.task_graph import canonical_bytes, canonical_json, file_hash, tree_hash
from writing_agent.task_graph_contracts import CheckContractV1
from writing_agent.task_graph_sampling import ProjectionError


@dataclass(frozen=True)
class EvaluationRequestV1:
    family: str | None
    target_checkpoint: str
    check: CheckContractV1
    evaluator_packet_ref: str
    files_json: str

    @classmethod
    def create(cls, family, target_checkpoint, check, evaluator_packet_ref, files):
        return cls(family, target_checkpoint, check, evaluator_packet_ref, canonical_json(files))

    def files(self) -> dict[str, str]:
        import json

        return json.loads(self.files_json)


@dataclass(frozen=True)
class EvaluationEvidenceV1:
    family: str
    status: str
    body: Mapping[str, Any]

    def to_wire(self, request: EvaluationRequestV1) -> dict[str, Any]:
        record_type = {
            "deterministic-file-v1": "DeterministicCheckEvidenceV1",
            "fixture-file-count-v1": "FixtureFileCountEvidenceV1",
        }.get(self.family)
        if record_type is None:
            raise ProjectionError("unknown evaluator evidence family")
        return {
            "record_type": record_type,
            "schema": 1,
            "target_checkpoint": request.target_checkpoint,
            "check_contract_hash": request.check.identity(),
            "evaluator_packet_ref": request.evaluator_packet_ref,
            "evidence": dict(self.body),
            "status": self.status,
        }


def deterministic_check(check: CheckContractV1, files: Mapping[str, str]) -> tuple[str, dict]:
    """Evaluate only the strict admitted file-target vocabulary."""
    spec = check.spec
    path = spec["path"]
    text = files.get(path)
    if text is None:
        return "fail", {"path": path, "subject_hash": None, "reason": "missing_path"}
    kind = spec["kind"]
    if kind == "nonempty":
        passed = bool(text.strip())
    elif kind == "contains":
        passed = spec["text"].casefold() in text.casefold()
    elif kind == "excludes":
        passed = spec["text"].casefold() not in text.casefold()
    elif kind == "excludes_all":
        passed = not any(item.casefold() in text.casefold() for item in spec["texts"])
    elif kind == "word_range":
        passed = spec["min"] <= len(text.split()) <= spec["max"]
    elif kind == "exact":
        passed = text == spec["text"]
    else:
        raise ValueError("check kind was not admitted for deterministic evaluator")
    return ("pass" if passed else "fail"), {
        "path": path,
        "subject_hash": file_hash(text),
        "reason": None,
    }


def _deterministic(request: EvaluationRequestV1) -> EvaluationEvidenceV1:
    status, body = deterministic_check(request.check, request.files())
    return EvaluationEvidenceV1("deterministic-file-v1", status, body)


def _fixture_count(request: EvaluationRequestV1) -> EvaluationEvidenceV1:
    files = request.files()
    # Explicit test family: a distinct, deterministic contract; never a semantic judge.
    return EvaluationEvidenceV1(
        "fixture-file-count-v1",
        "pass" if len(files) % 2 else "fail",
        {"tree_hash": tree_hash(files), "file_count": len(files), "rule": "odd-file-count-v1"},
    )


VERIFIERS: Mapping[str, Callable[[EvaluationRequestV1], EvaluationEvidenceV1]] = MappingProxyType(
    {
        "deterministic-file-v1": _deterministic,
        "fixture-file-count-v1": _fixture_count,
    }
)
RECORD_FAMILIES = MappingProxyType(
    {
        "DeterministicCheckEvidenceV1": "deterministic-file-v1",
        "FixtureFileCountEvidenceV1": "fixture-file-count-v1",
    }
)
FAMILY_CHECK_VERSIONS = MappingProxyType(
    {
        "deterministic-file-v1": "deterministic-v1",
        "fixture-file-count-v1": "fixture-file-count-v1",
    }
)


def verify_evaluation_evidence(request: EvaluationRequestV1, wire: Any) -> EvaluationEvidenceV1:
    if not isinstance(wire, dict):
        raise ProjectionError("evaluator evidence has wrong schema")
    family = RECORD_FAMILIES.get(wire.get("record_type"))
    if family is None or family not in VERIFIERS:
        raise ProjectionError("unknown evaluator evidence family")
    if request.family is not None and request.family != family:
        raise ProjectionError("evaluator used a different admitted family")
    if request.check.evaluator_version != FAMILY_CHECK_VERSIONS[family]:
        raise ProjectionError("evaluator family is not admitted by the check contract")
    expected = VERIFIERS[family](
        EvaluationRequestV1(
            family,
            request.target_checkpoint,
            request.check,
            request.evaluator_packet_ref,
            request.files_json,
        )
    )
    if canonical_bytes(wire) != canonical_bytes(expected.to_wire(request)):
        raise ProjectionError("evaluator evidence contradicts frozen request")
    return expected
