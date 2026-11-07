"""Closed codecs for the shared payload shapes without Python record classes."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from writing_agent.task_graph_wire import (
    DictOf,
    Enum,
    Hash,
    Int,
    JsonValue,
    ListOf,
    PayloadCodec,
    Str,
    UnionOf,
    obj,
)


def _codec(record_type: str, **fields: Any) -> PayloadCodec:
    return PayloadCodec(record_type, obj(**fields))


_AUTHOR_PREREQUISITES = DictOf(
    obj(result_ref=Hash("artifact"), status=Enum(frozenset({"pass", "fail"})))
)
_REWARD_COMPONENTS = DictOf(
    obj(
        weight=Int(),
        earned=Int(),
        status=Enum(frozenset({"pass", "fail", "not_run"})),
    )
)
_EVALUATOR_EVIDENCE = PayloadCodec(
    "EvaluatorEvidenceV1",
    obj(
        record_type=Enum(
            frozenset(
                {
                    "DeterministicCheckEvidenceV1",
                    "FixtureFileCountEvidenceV1",
                    "TranscriptReviewEvidenceV1",
                }
            )
        ),
        schema=Int(equals=1),
        target_checkpoint=Hash("checkpoint"),
        check_contract_hash=Hash(None),
        evaluator_packet_ref=Hash("private"),
        evidence=JsonValue(),
        status=Enum(frozenset({"pass", "fail"})),
    ),
    tagged=False,
)

_PAYLOAD_RECORD_CODECS = MappingProxyType(
    {
        "DecisionLedgerV1": _codec(
            "DecisionLedgerV1",
            schema=Int(equals=1),
            values=DictOf(JsonValue()),
            proposals=DictOf(JsonValue()),
        ),
        "DisclosureLedgerV1": _codec(
            "DisclosureLedgerV1", schema=Int(equals=1), decisions=ListOf(JsonValue())
        ),
        "AuthorRequestV1": _codec(
            "AuthorRequestV1",
            schema=Int(equals=1),
            request_id=Str(nonempty=True, logical=True),
            source=Enum(frozenset({"writer_request", "mandatory_feedback"})),
            action_id=Str(logical=True, optional=True),
            call_id=Str(logical=True, optional=True),
            feedback_id=Str(logical=True, optional=True),
            arguments=UnionOf((JsonValue(), type(None))),
            decision_ids=ListOf(Str(nonempty=True, logical=True)),
            prerequisite_results=_AUTHOR_PREREQUISITES,
            requirement_version=Hash("artifact|private"),
            script_ref=Hash("private"),
            author_packet_ref=Hash("private", optional=True),
        ),
        "CheckRequestV1": _codec(
            "CheckRequestV1",
            schema=Int(equals=1),
            request_id=Str(nonempty=True, logical=True),
            target_checkpoint=Hash("checkpoint"),
            requirement_version=Hash("private"),
            check_contract_hash=Hash("private"),
            evaluator_packet_ref=Hash("private", optional=True),
            check_id=Str(nonempty=True, logical=True),
            purpose=Enum(frozenset({"progress", "completion"})),
        ),
        "RewardV1": _codec(
            "RewardV1",
            schema=Int(equals=1),
            terminal_outcome_ref=Hash("artifact"),
            reward_contract_ref=Hash("private"),
            candidate_checkpoint=Hash("checkpoint", optional=True),
            check_result_refs=ListOf(Hash("artifact")),
            components=_REWARD_COMPONENTS,
            numerator=Int(),
            normalization=Int(minimum=1),
            availability=Enum(frozenset({"available", "unavailable"})),
            eligibility_ref=Hash("artifact"),
        ),
        "TrainingEligibilityV1": _codec(
            "TrainingEligibilityV1",
            schema=Int(equals=1),
            terminal_outcome_ref=Hash("artifact"),
            status=Enum(frozenset({"eligible", "ineligible", "structurally_eligible"})),
            reason=Str(nonempty=True),
        ),
        **{
            name: _EVALUATOR_EVIDENCE
            for name in (
                "DeterministicCheckEvidenceV1",
                "FixtureFileCountEvidenceV1",
                "TranscriptReviewEvidenceV1",
            )
        },
        "GroupMemberSeedsV1": _codec(
            "GroupMemberSeedsV1",
            group_id=Hash(None),
            member_id=Str(nonempty=True, logical=True),
            derivation=Str(nonempty=True),
            writer_seed=Int(),
            environment_seed=Int(),
            parent_rng_ref=Hash("artifact"),
        ),
        "RequirementLedgerV1": _codec(
            "RequirementLedgerV1",
            schema=Int(equals=1),
            active=DictOf(Str()),
            superseded=DictOf(Str()),
        ),
    }
)


def payload_record_codecs() -> tuple[tuple[str, PayloadCodec], ...]:
    """Return shared payload codec entries for the central record registry."""
    return tuple(_PAYLOAD_RECORD_CODECS.items())
