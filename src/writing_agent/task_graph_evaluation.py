"""Versioned evaluator codecs and offline evidence verifiers.

Verification establishes frozen-input provenance and internal consistency, not the
subjective correctness or authenticity of a transcript's judgment.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol

from writing_agent.task_graph import (
    canonical_bytes,
    canonical_json,
    domain_hash,
    file_hash,
    tree_hash,
)
from writing_agent.task_graph_contracts import CheckContractV1
from writing_agent.task_graph_errors import ProjectionError


@dataclass(frozen=True)
class EvaluationRequestV1:
    family: str | None
    target_checkpoint: str
    check: CheckContractV1
    evaluator_packet_ref: str
    files_json: str
    packet_json: str | None = None

    @classmethod
    def create(
        cls,
        family,
        target_checkpoint,
        check,
        evaluator_packet_ref,
        files,
        *,
        evaluator_packet=None,
    ):
        packet_json = canonical_json(evaluator_packet) if evaluator_packet is not None else None
        if (
            packet_json is not None
            and domain_hash("payload", evaluator_packet) != evaluator_packet_ref
        ):
            raise ProjectionError("evaluator packet differs from its frozen reference")
        return cls(
            family,
            target_checkpoint,
            check,
            evaluator_packet_ref,
            canonical_json(files),
            packet_json,
        )

    def files(self) -> dict[str, str]:
        return json.loads(self.files_json)

    def packet(self) -> dict[str, Any] | None:
        return json.loads(self.packet_json) if self.packet_json is not None else None


class EvidenceResolver(Protocol):
    """Only the request-authorized private evaluator packet may be read."""

    def read_evaluator_packet(self, ref: str) -> Mapping[str, Any]: ...


class StoreEvidenceResolver:
    def __init__(self, store, authorized_packet_ref: str):
        self._store = store
        self._authorized_packet_ref = authorized_packet_ref

    def read_evaluator_packet(self, ref: str) -> Mapping[str, Any]:
        if ref != self._authorized_packet_ref:
            raise ProjectionError("evaluator requested an unauthorized packet")
        return self._store.get_artifact(ref, expected_domain="payload", private=True)


@dataclass(frozen=True)
class EvaluationEvidenceV1:
    family: str
    status: str
    body: Mapping[str, Any]

    def to_wire(self, request: EvaluationRequestV1) -> dict[str, Any]:
        family = FAMILIES.get(self.family)
        if family is None:
            raise ProjectionError("unknown evaluator evidence family")
        return {
            "record_type": family.record_type,
            "schema": 1,
            "target_checkpoint": request.target_checkpoint,
            "check_contract_hash": request.check.identity(),
            "evaluator_packet_ref": request.evaluator_packet_ref,
            "evidence": dict(self.body),
            "status": self.status,
        }


@dataclass(frozen=True)
class DecodedEvidence:
    status: str
    body: Mapping[str, Any]


@dataclass(frozen=True)
class TranscriptEvidence:
    status: str
    declared_status: str
    request_payload: Mapping[str, Any]
    response_text: str
    body: Mapping[str, Any]


@dataclass(frozen=True)
class EvidenceFamily:
    name: str
    record_type: str
    check_version: str
    program_kind: str | None
    program_method: str | None
    decode: Callable[[Any, Any], DecodedEvidence | TranscriptEvidence]
    verify: Callable[
        [EvaluationRequestV1, DecodedEvidence | TranscriptEvidence, EvidenceResolver | None], None
    ]
    produce: Callable[[EvaluationRequestV1], EvaluationEvidenceV1] | None = None


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
    return EvaluationEvidenceV1(
        "fixture-file-count-v1",
        "pass" if len(files) % 2 else "fail",
        {"tree_hash": tree_hash(files), "file_count": len(files), "rule": "odd-file-count-v1"},
    )


def _decode_exact(status: Any, body: Any) -> DecodedEvidence:
    if not isinstance(status, str) or status not in {"pass", "fail"} or not isinstance(body, dict):
        raise ProjectionError("evaluator evidence has wrong family body")
    return DecodedEvidence(status, body)


def _verify_exact(
    request: EvaluationRequestV1,
    evidence: DecodedEvidence | TranscriptEvidence,
    resolver: EvidenceResolver | None,
) -> None:
    family = FAMILIES[request.family]
    expected = family.produce(request)
    if canonical_bytes({"status": evidence.status, "body": evidence.body}) != canonical_bytes(
        {"status": expected.status, "body": expected.body}
    ):
        raise ProjectionError("evaluator evidence contradicts frozen request")


def _decode_transcript(status: Any, body: Any) -> TranscriptEvidence:
    if (
        not isinstance(status, str)
        or status not in {"pass", "fail"}
        or not isinstance(body, dict)
        or set(body)
        != {
            "transcript",
            "declared_status",
        }
    ):
        raise ProjectionError("transcript evidence has wrong schema")
    transcript = body["transcript"]
    if (
        not isinstance(transcript, list)
        or len(transcript) != 2
        or not isinstance(transcript[0], dict)
        or set(transcript[0]) != {"role", "payload"}
        or transcript[0]["role"] != "request"
        or not isinstance(transcript[0]["payload"], dict)
        or not isinstance(transcript[1], dict)
        or set(transcript[1]) != {"role", "status", "text"}
        or transcript[1]["role"] != "response"
        or not isinstance(transcript[1]["status"], str)
        or transcript[1]["status"] not in {"pass", "fail"}
        or not isinstance(transcript[1]["text"], str)
        or not transcript[1]["text"].strip()
        or len(transcript[1]["text"]) > 100_000
        or body["declared_status"] != transcript[1]["status"]
        or status != body["declared_status"]
    ):
        raise ProjectionError("transcript evidence has inconsistent response")
    return TranscriptEvidence(
        status,
        body["declared_status"],
        transcript[0]["payload"],
        transcript[1]["text"],
        body,
    )


def _verify_transcript(
    request: EvaluationRequestV1,
    evidence: DecodedEvidence | TranscriptEvidence,
    resolver: EvidenceResolver | None,
) -> None:
    if resolver is None or request.packet_json is None:
        raise ProjectionError("transcript verification requires frozen evaluator packet")
    packet = resolver.read_evaluator_packet(request.evaluator_packet_ref)
    if canonical_json(packet) != request.packet_json:
        raise ProjectionError("resolved evaluator packet differs from frozen request")
    expected = {
        "target_checkpoint": request.target_checkpoint,
        "check_contract_hash": request.check.identity(),
        "evaluator_packet_ref": request.evaluator_packet_ref,
        "candidate_tree_hash": tree_hash(request.files()),
    }
    if evidence.request_payload != expected:
        raise ProjectionError("transcript provenance contradicts frozen request")


FAMILIES: Mapping[str, EvidenceFamily] = MappingProxyType(
    {
        family.name: family
        for family in (
            EvidenceFamily(
                "deterministic-file-v1",
                "DeterministicCheckEvidenceV1",
                "deterministic-v1",
                None,
                None,
                _decode_exact,
                _verify_exact,
                _deterministic,
            ),
            EvidenceFamily(
                "fixture-file-count-v1",
                "FixtureFileCountEvidenceV1",
                "fixture-file-count-v1",
                "fixture_file_count",
                "fixture",
                _decode_exact,
                _verify_exact,
                _fixture_count,
            ),
            EvidenceFamily(
                "transcript-review-v1",
                "TranscriptReviewEvidenceV1",
                "transcript-review-v1",
                "transcript_review",
                "offline_transcript",
                _decode_transcript,
                _verify_transcript,
            ),
        )
    }
)


def produce_evaluation_evidence(request: EvaluationRequestV1) -> EvaluationEvidenceV1:
    family = FAMILIES.get(request.family)
    if family is None or family.produce is None:
        raise ProjectionError("evaluator family has no local deterministic producer")
    return family.produce(request)


def verify_evaluation_evidence(
    request: EvaluationRequestV1,
    wire: Any,
    resolver: EvidenceResolver | None = None,
) -> EvaluationEvidenceV1:
    if not isinstance(wire, dict):
        raise ProjectionError("evaluator evidence has wrong schema")
    family = next(
        (item for item in FAMILIES.values() if item.record_type == wire.get("record_type")),
        None,
    )
    if family is None:
        raise ProjectionError("unknown evaluator evidence family")
    if request.family is not None and request.family != family.name:
        raise ProjectionError("evaluator used a different admitted family")
    if request.check.evaluator_version != family.check_version:
        raise ProjectionError("evaluator family is not admitted by the check contract")
    if (
        set(wire)
        != {
            "record_type",
            "schema",
            "target_checkpoint",
            "check_contract_hash",
            "evaluator_packet_ref",
            "evidence",
            "status",
        }
        or type(wire["schema"]) is not int
        or wire["schema"] != 1
        or wire["target_checkpoint"] != request.target_checkpoint
        or wire["check_contract_hash"] != request.check.identity()
        or wire["evaluator_packet_ref"] != request.evaluator_packet_ref
    ):
        raise ProjectionError("evaluator evidence contradicts frozen request")
    decoded = family.decode(wire["status"], wire["evidence"])
    family.verify(replace(request, family=family.name), decoded, resolver)
    return EvaluationEvidenceV1(family.name, decoded.status, decoded.body)
