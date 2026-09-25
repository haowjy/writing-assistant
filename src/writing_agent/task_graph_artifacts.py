"""Shared typing and visibility rules for Phase 3 artifact closure.

Storage and in-memory compilation use different byte loaders, but an artifact must
have the same typed references and visibility in both places.  This module is the
single interpretation point for those properties; it performs no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from writing_agent.task_graph_contracts import (
    CheckContractV1,
    GuardContractV1,
    NodeContractV1,
    ScriptContractV1,
)


@dataclass(frozen=True)
class TypedArtifactReference:
    identity: str
    private: bool


class TypedArtifactError(ValueError):
    """A typed payload has an unknown, invalid, or misrouted contract."""

    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind


_CONTRACT_TYPES = {
    NodeContractV1.ARTIFACT_TYPE: NodeContractV1,
    GuardContractV1.ARTIFACT_TYPE: GuardContractV1,
    CheckContractV1.ARTIFACT_TYPE: CheckContractV1,
    ScriptContractV1.ARTIFACT_TYPE: ScriptContractV1,
}
_PUBLIC_TYPES = frozenset({NodeContractV1.ARTIFACT_TYPE, GuardContractV1.ARTIFACT_TYPE})
_PRIVATE_TYPES = frozenset({CheckContractV1.ARTIFACT_TYPE, ScriptContractV1.ARTIFACT_TYPE})


def phase3_typed_artifact_references(
    body: Mapping[str, Any], *, private: bool
) -> tuple[TypedArtifactReference, ...]:
    """Validate one typed Phase 3 payload and return its complete direct edges."""
    artifact_type = body.get("artifact_type")
    if not isinstance(artifact_type, str):
        raise TypedArtifactError("invalid", "artifact_type must be a string")
    try:
        contract_type = _CONTRACT_TYPES[artifact_type]
    except KeyError as exc:
        raise TypedArtifactError(
            "unsupported", f"unsupported typed payload artifact: {artifact_type!r}"
        ) from exc
    if (private and artifact_type in _PUBLIC_TYPES) or (
        not private and artifact_type in _PRIVATE_TYPES
    ):
        required = "private" if artifact_type in _PRIVATE_TYPES else "public"
        raise TypedArtifactError("visibility", f"{artifact_type} requires {required} storage")
    try:
        contract = contract_type.from_dict(body)
    except (TypeError, ValueError) as exc:
        raise TypedArtifactError("invalid", f"invalid {artifact_type} envelope") from exc

    if isinstance(contract, CheckContractV1):
        return (
            *(
                TypedArtifactReference(identity, False)
                for identity in contract.public_evidence_refs
            ),
            *(
                TypedArtifactReference(identity, True)
                for identity in contract.private_evidence_refs
            ),
        )
    if not isinstance(contract, NodeContractV1):
        return ()

    entry = contract.entry_contract
    interaction = contract.interaction_contract
    references = [
        TypedArtifactReference(entry.request_ref, False),
        TypedArtifactReference(entry.files_ref, False),
    ]
    if entry.requirement_version is not None:
        references.append(TypedArtifactReference(entry.requirement_version, True))
    references.extend(
        TypedArtifactReference(identity, True)
        for identity in (*contract.mandatory_checks, *contract.optional_checks)
    )
    if interaction.script_ref is not None:
        references.append(TypedArtifactReference(interaction.script_ref, True))
    if interaction.author_packet_ref is not None:
        references.append(TypedArtifactReference(interaction.author_packet_ref, True))
    if interaction.interaction_policy_ref is not None:
        references.append(TypedArtifactReference(interaction.interaction_policy_ref, False))
    if interaction.decision_bindings_ref is not None:
        references.append(TypedArtifactReference(interaction.decision_bindings_ref, True))
    completion = contract.completion_contract
    if completion.evaluation_packet_ref is not None:
        references.append(TypedArtifactReference(completion.evaluation_packet_ref, True))
    return tuple(references)


def validate_phase3_artifact_closure(
    identity: str,
    *,
    private: bool,
    load: Callable[[str, bool], Any],
) -> Any:
    """Load and validate a complete Phase 3 typed closure with one traversal."""
    root = (identity, private)
    loaded: dict[tuple[str, bool], Any] = {}
    active: set[tuple[str, bool]] = set()
    completed: set[tuple[str, bool]] = set()
    stack: list[tuple[tuple[str, bool], bool]] = [(root, False)]
    while stack:
        key, exiting = stack.pop()
        if exiting:
            active.remove(key)
            completed.add(key)
            continue
        if key in completed:
            continue
        if key in active:
            raise TypedArtifactError("invalid", f"artifact reference cycle at {key[0]}")
        if key not in loaded:
            loaded[key] = load(*key)
        value = loaded[key]
        active.add(key)
        stack.append((key, True))
        if not isinstance(value, Mapping) or "artifact_type" not in value:
            continue
        references = phase3_typed_artifact_references(value, private=key[1])
        for reference in reversed(references):
            child = (reference.identity, reference.private)
            if child not in loaded:
                loaded[child] = load(*child)
            stack.append((child, False))
    return loaded[root]
