"""Typed closure traversal and inter-process lineage locking for the task-graph store."""

from __future__ import annotations

import fcntl
import os
import stat
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from writing_agent.task_graph import (
    CheckpointV1,
    CommitV1,
    EnvironmentStateV1,
    EventV1,
    GraphInstanceV1,
    MessageV1,
    domain_hash,
    domain_hash_bytes,
    validate_hash,
)
from writing_agent.task_graph_artifacts import (
    TypedArtifactError,
    phase3_typed_artifact_references,
)
from writing_agent.task_graph_errors import (
    CorruptRecordError,
    MissingReferenceError,
    ProjectionError,
    StoreError,
    WrongRecordDomainError,
)
from writing_agent.task_graph_record_contracts import SEMANTICS_V1, ExecutionVersionsV1
from writing_agent.task_graph_records import (
    RECORD_TYPES,
    ContextContentV1,
    ContextRevisionV1,
    record_reference_paths,
)

if TYPE_CHECKING:
    from writing_agent.task_graph_store import TaskGraphStore


@dataclass(frozen=True)
class _Artifact:
    value: Any
    domain: str
    private: bool


class ClosureValidator:
    """One linear, typed closure traversal for a public store operation.

    Loaded objects, active nodes, and completed nodes live only for this operation.
    The explicit stack keeps deep event, checkpoint, and commit ancestry off the
    Python call stack.  Every edge, including supplemental/imported references,
    resolves through this same domain-aware loader.
    """

    _RECORDS: dict[str, tuple[type[Any], str]] = {
        "instance": (GraphInstanceV1, "instances"),
        "event": (EventV1, "events"),
        "context_node": (ContextContentV1, "context_content"),
        "context_revision": (ContextRevisionV1, "context_revisions"),
        "checkpoint": (CheckpointV1, "checkpoints"),
        "commit": (CommitV1, "commits"),
    }
    _LOCATION_KINDS = {
        "instances": "instance",
        "events": "event",
        "checkpoints": "checkpoint",
        "commits": "commit",
        "context_content": "context_node",
        "context_revisions": "context_revision",
        "artifacts": "artifact",
        "private": "private",
    }

    def __init__(self, store: TaskGraphStore, parent: ClosureValidator | None = None) -> None:
        self.store = store
        self.parent = parent
        self.loaded: dict[tuple[str, str], Any] = {}
        self.virtual: dict[tuple[str, str], Any] = {}
        self.resolved: dict[tuple[str, str], tuple[str, str]] = {}
        self.active: set[tuple[str, str]] = set()
        self.completed: set[tuple[str, str]] = set()
        self.required_domain: dict[str, frozenset[str]] = {}
        self.gated_candidates: set[tuple[str, str]] = set()
        self.paths: dict[tuple[str, str], str] = {}
        if parent is not None:
            # Only completed (closure-validated, disk-backed) results are shared.
            self.completed = set(parent.completed)
            self.loaded = {key: parent.loaded[key] for key in parent.completed}
            self.resolved = dict(parent.resolved)
            self.required_domain = dict(parent.required_domain)

    def discard(self, keys: set[tuple[str, str]]) -> None:
        """Remove candidate and cache state associated with rejected keys."""
        for key in keys:
            self.virtual.pop(key, None)
            self.loaded.pop(key, None)
            self.completed.discard(key)
            self.active.discard(key)
            self.required_domain.pop(key[1], None)
            self.gated_candidates.discard(key)
        for requested, resolved in tuple(self.resolved.items()):
            if requested in keys or resolved in keys:
                self.resolved.pop(requested, None)

    def _merge_into_parent(self) -> None:
        parent = self.parent
        if parent is None or self.virtual or self.active:
            return
        for key in self.completed:
            if key not in parent.completed:
                parent.loaded[key] = self.loaded[key]
        parent.completed |= self.completed
        parent.resolved.update(self.resolved)
        parent.required_domain.update(self.required_domain)

    def add_virtual(self, kind: str, identity: str, value: Any) -> None:
        # Candidate objects must see fresh location resolution for their identity.
        for requested in [item for item in self.resolved if item[1] == identity]:
            del self.resolved[requested]
        key = (kind, identity)
        existing = self.virtual.get(key)
        if existing is not None and existing != value:
            raise CorruptRecordError(f"conflicting in-memory object for {identity}")
        self.virtual[key] = value

    def validate(self, requested: tuple[str, str]) -> Any:
        original_resolved = set(self.resolved)
        original_required_domain = dict(self.required_domain)
        try:
            root = self._resolve_key(requested)
            stack: list[tuple[tuple[str, str], bool]] = [(root, False)]
            while stack:
                raw_key, exiting = stack.pop()
                key = self._resolve_key(raw_key)
                if exiting:
                    try:
                        self._validate_after(key, self.loaded[key])
                    except StoreError as exc:
                        if key in self.gated_candidates:
                            path = {
                                "event": "event",
                                "checkpoint": "checkpoint",
                                "commit": "commit",
                            }[key[0]]
                            raise ProjectionError(
                                f"{path}: candidate closure failed: {exc}"
                            ) from exc
                        raise
                    self.active.remove(key)
                    self.completed.add(key)
                    continue
                if key in self.completed:
                    continue
                if key in self.active:
                    raise CorruptRecordError(f"reference cycle through {key[0]} {key[1]}")
                value = self._load(key)
                self.active.add(key)
                stack.append((key, True))
                edges = self._edges(key, value)
                stack.extend((edge, False) for edge in reversed(edges))
            self._merge_into_parent()
            return self.loaded[root]
        except BaseException:
            # Only completed closures are reusable; discard the partial walk's
            # decoded objects and location decisions before a caller retries.
            for key in tuple(self.loaded):
                if key not in self.completed:
                    self.loaded.pop(key, None)
            for key, resolved in tuple(self.resolved.items()):
                if key not in original_resolved and resolved not in self.completed:
                    self.resolved.pop(key, None)
            self.required_domain = {
                **original_required_domain,
                **{
                    identity: required
                    for identity, required in self.required_domain.items()
                    if ("artifact", identity) in self.completed
                },
            }
            self.active.clear()
            self._merge_into_parent()
            raise

    def audit_artifacts(self) -> None:
        """Debug audit that cached artifact bodies still match their addresses."""
        for kind, identity in self.completed:
            if kind not in {"artifact", "private"}:
                continue
            artifact = self.loaded[(kind, identity)]
            try:
                if artifact.domain.endswith(":bytes"):
                    actual = domain_hash_bytes(
                        artifact.domain.removesuffix(":bytes"), artifact.value
                    )
                else:
                    actual = domain_hash(artifact.domain, artifact.value)
            except (KeyError, TypeError, ValueError) as exc:
                raise CorruptRecordError(f"cached artifact identity mismatch: {identity}") from exc
            if actual != identity:
                raise CorruptRecordError(f"cached artifact identity mismatch: {identity}")

    def _path_locations(self, identity: str) -> list[str]:
        locations: list[str] = []
        for directory in (
            "instances",
            "events",
            "context_content",
            "context_revisions",
            "checkpoints",
            "commits",
            "artifacts",
            "private",
        ):
            path = (
                self.store._artifact_path(identity, directory == "private")
                if directory in {"artifacts", "private"}
                else self.store._record_path(directory, identity)
            )
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            locations.append(directory)
        return locations

    def _resolve_key(self, requested: tuple[str, str]) -> tuple[str, str]:
        if requested in self.resolved:
            self._require_domain(requested[0], requested[1])
            return self.resolved[requested]
        if requested in self.loaded:
            return requested
        kind, identity = requested
        validate_hash(identity)
        virtual_kinds = {candidate for candidate, item in self.virtual if item == identity}
        locations = self._path_locations(identity)
        if kind == "artifact|private":
            domains = set(virtual_kinds) | (set(locations) & {"artifacts", "private"})
            if not domains:
                raise MissingReferenceError(f"missing public or private artifact: {identity}")
            if len(domains) != 1:
                raise WrongRecordDomainError(
                    f"reference {identity} exists in ambiguous public/private locations"
                )
            resolved = next(iter(domains))
            resolved = {"artifacts": "artifact"}.get(resolved, resolved)
            self.resolved[requested] = (resolved, identity)
            return resolved, identity
        if kind in {"bytes", "artifact|bytes"}:
            if "artifacts" not in locations:
                if locations or "private" in locations or virtual_kinds:
                    raise WrongRecordDomainError(
                        f"reference {identity} is not stored as a public payload artifact"
                    )
                raise MissingReferenceError(f"missing public payload artifact: {identity}")
            other_locations = set(locations) - {"artifacts"}
            if other_locations:
                found_locations = sorted(("artifacts", *other_locations))
                raise WrongRecordDomainError(
                    f"reference {identity} exists in ambiguous locations: {found_locations}"
                )
            self._require_domain(kind, identity)
            self.resolved[requested] = ("artifact", identity)
            return ("artifact", identity)
        disk_kinds = {self._LOCATION_KINDS[location] for location in locations}

        if kind == "any":
            candidates = set(virtual_kinds)
            candidates.update(disk_kinds)
            if not candidates:
                raise MissingReferenceError(f"missing immutable reference: {identity}")
            if len(candidates) != 1:
                raise WrongRecordDomainError(
                    f"reference {identity} exists in ambiguous domains: {sorted(candidates)}"
                )
            resolved = next(iter(candidates))
            result = (resolved, identity)
            self.resolved[requested] = result
            return result

        expected_location = self._expected_location(kind)
        other_locations = {location for location in locations if location != expected_location}
        if other_locations:
            raise WrongRecordDomainError(
                f"reference {identity} exists in ambiguous locations: "
                f"{sorted({expected_location, *other_locations})}"
            )
        if kind in virtual_kinds:
            self.resolved[requested] = requested
            return requested
        if expected_location not in locations:
            if locations or virtual_kinds:
                raise WrongRecordDomainError(f"reference {identity} is not stored as {kind}")
            raise MissingReferenceError(f"missing {kind} reference: {identity}")
        self.resolved[requested] = requested
        return requested

    def _require_domain(self, edge_kind: str, identity: str) -> None:
        allowed = {
            "bytes": frozenset({"payload:bytes"}),
            "artifact|bytes": frozenset({"payload", "payload:bytes"}),
        }.get(edge_kind)
        if allowed is None:
            return
        previous = self.required_domain.get(identity)
        required = allowed if previous is None else previous & allowed
        if not required:
            raise WrongRecordDomainError(f"conflicting payload domain requirements for {identity}")
        if required != previous:
            self.required_domain[identity] = required
            self.completed.discard(("artifact", identity))

    @staticmethod
    def _expected_location(kind: str) -> str:
        if kind in {"artifact", "private"}:
            return "artifacts" if kind == "artifact" else "private"
        try:
            return ClosureValidator._RECORDS[kind][1]
        except KeyError as exc:
            raise AssertionError(f"unknown closure node kind: {kind}") from exc

    def _load(self, key: tuple[str, str]) -> Any:
        if key in self.loaded:
            return self.loaded[key]
        if key in self.virtual:
            value = self.virtual[key]
        else:
            kind, identity = key
            if kind in self._RECORDS:
                record_type, directory = self._RECORDS[kind]
                value = self.store._load_record(identity, record_type, directory)
            elif kind in {"artifact", "private"}:
                value, domain = self.store._decode_artifact(
                    self.store._artifact_path(identity, kind == "private"), identity
                )
                value = _Artifact(value, domain, kind == "private")
            else:
                raise AssertionError(f"unknown closure node kind: {kind}")
        self.loaded[key] = value
        return value

    def _remember(self, kind: str, identity: str, path: str) -> tuple[str, str]:
        key = (kind, identity)
        self.paths.setdefault(key, path)
        return key

    def _edges(self, key: tuple[str, str], value: Any) -> list[tuple[str, str]]:
        kind, _identity = key
        if kind == "instance":
            edges = [self._remember("artifact", value.template_ref, "instance.template_ref")]
            edges.extend(
                self._remember("artifact", identity, f"instance.source_refs[{index}]")
                for index, identity in enumerate(value.source_refs)
            )
            edges.extend(
                self._remember("artifact", identity, f"instance.request_refs[{index}]")
                for index, identity in enumerate(value.request_refs)
            )
            if value.requirements_ref is not None:
                edges.append(
                    self._remember("artifact", value.requirements_ref, "instance.requirements_ref")
                )
            edges.extend(
                self._remember(
                    "artifact", node.entry_contract, f"instance.nodes[{index}].entry_contract"
                )
                for index, node in enumerate(value.nodes)
            )
            return edges
        if kind == "event":
            edges = [
                self._remember("artifact", value.payload_ref, "event.payload_ref"),
                self._remember("artifact", value.versions_ref, "event.versions_ref"),
                self._remember("artifact", value.provenance_ref, "event.provenance_ref"),
            ]
            if value.previous is not None:
                edges.append(self._remember("event", value.previous, "event.previous"))
            edges.extend(
                self._remember("event", identity, f"event.caused_by[{index}]")
                for index, identity in enumerate(value.caused_by)
            )
            return edges
        if kind in {"context_revision", "context_node"}:
            return [
                self._remember(edge, identity, f"{kind}.{field_path}")
                for field_path, edge, identity in value._wire_edges
            ]
        if kind in {"artifact", "private"}:
            if value.domain == "payload" and isinstance(value.value, Mapping):
                if "record_type" in value.value:
                    record_type = value.value["record_type"]
                    base_path = self.paths.get(key, "artifact")
                    if not isinstance(record_type, str):
                        raise ProjectionError(
                            f"{base_path}.record_type: task-graph record type must be a string"
                        )
                    if record_type not in RECORD_TYPES:
                        raise ProjectionError(
                            f"{base_path}.record_type: unregistered task-graph record type "
                            f"{record_type!r}"
                        )
                    try:
                        refs = record_reference_paths(record_type, value.value)
                    except (TypeError, ValueError) as exc:
                        field_path = self._codec_field_path(record_type, str(exc))
                        suffix = f".{field_path}" if field_path else ""
                        raise ProjectionError(
                            f"{base_path}{suffix}: invalid task-graph payload record "
                            f"{record_type}: {exc}"
                        ) from exc
                    return [
                        self._remember(edge, identity, f"{base_path}.{field_path}")
                        for field_path, edge, identity in refs
                    ]
                if "artifact_type" in value.value:
                    return self._phase3_contract_edges(
                        value.value,
                        private=value.private,
                        path=self.paths.get(key, "artifact"),
                    )
            return []
        if kind == "checkpoint":
            edges = [
                self._remember("checkpoint", identity, f"checkpoint.parents[{index}]")
                for index, identity in enumerate(value.parents)
            ]
            edges.extend(self._state_edges(value.state))
            edges.extend(
                self._remember("any", identity, f"checkpoint.artifact_refs[{index}]")
                for index, identity in enumerate(value.artifact_refs)
            )
            return edges
        if kind == "commit":
            edges = [self._remember("checkpoint", value.checkpoint, "commit.checkpoint")]
            if value.parent_commit is not None:
                edges.append(self._remember("commit", value.parent_commit, "commit.parent_commit"))
            edges.extend(
                self._remember("event", identity, f"commit.events[{index}]")
                for index, identity in enumerate(value.events)
            )
            return edges
        return []

    @staticmethod
    def _codec_field_path(record_type: str, detail: str) -> str:
        prefix = f"{record_type}."
        if prefix in detail:
            field = detail.split(prefix, 1)[1].split(":", 1)[0].split(" ", 1)[0]
            return field.rstrip(".,)")
        marker = "unknown fields: "
        if marker in detail:
            import ast

            try:
                fields = ast.literal_eval(detail.split(marker, 1)[1])
            except (SyntaxError, ValueError):
                return ""
            if isinstance(fields, list) and fields and isinstance(fields[0], str):
                return fields[0]
        return ""

    def _versions_semantics(
        self, versions_ref: str
    ) -> tuple[ExecutionVersionsV1, Mapping[str, Any]]:
        key = self._resolve_key(("artifact", versions_ref))
        artifact = self._load(key)
        body = (
            artifact.value
            if artifact.domain == "payload" and isinstance(artifact.value, Mapping)
            else None
        )
        if body is None:
            raise ProjectionError("state.versions_ref: runtime lineage requires execution versions")
        if body.get("transition_semantics") != SEMANTICS_V1:
            raise ProjectionError(
                f"state.versions_ref.transition_semantics: runtime lineages require {SEMANTICS_V1}"
            )
        try:
            versions = ExecutionVersionsV1.from_dict(dict(body))
        except (TypeError, ValueError) as exc:
            raise ProjectionError(f"state.versions_ref: invalid execution versions: {exc}") from exc
        return versions, body

    def _state_edges(self, state: EnvironmentStateV1) -> list[tuple[str, str]]:
        versions, versions_body = self._versions_semantics(state.versions_ref)
        refs: list[tuple[str, str, str]] = [
            ("instance", state.instance_ref, "state.instance_ref"),
            ("context_revision", state.context_ref, "state.context_ref"),
            ("artifact", state.position["entry_contract"], "state.position.entry_contract"),
            ("private", state.requirements_ref, "state.requirements_ref"),
            ("artifact", state.decisions_ref, "state.decisions_ref"),
            ("artifact", state.disclosures_ref, "state.disclosures_ref"),
            ("artifact", state.versions_ref, "state.versions_ref"),
            ("artifact", state.budgets_ref, "state.budgets_ref"),
            ("artifact", state.rng_ref, "state.rng_ref"),
            ("artifact", state.external_inputs_ref, "state.external_inputs_ref"),
            ("artifact", state.outcome_ref, "state.outcome_ref"),
            ("artifact", state.provenance_ref, "state.provenance_ref"),
        ]
        refs.extend(
            ("any", identity, f"state.history.imported_refs[{index}]")
            for index, identity in enumerate(state.history["imported_refs"])
        )
        if state.author_packet_ref is not None:
            refs.append(("private", state.author_packet_ref, "state.author_packet_ref"))
        if state.continuation["author_request"] is not None:
            refs.append(
                (
                    "private",
                    state.continuation["author_request"],
                    "state.continuation.author_request",
                )
            )
        refs.extend(
            ("private", identity, f"state.continuation.check_requests[{index}]")
            for index, identity in enumerate(state.continuation["check_requests"])
        )
        if state.history["branch_base"] is not None:
            refs.append(("event", state.history["branch_base"], "state.history.branch_base"))
        if state.position["start_checkpoint"] is not None:
            refs.append(
                (
                    "checkpoint",
                    state.position["start_checkpoint"],
                    "state.position.start_checkpoint",
                )
            )
        if state.history["head"] is not None:
            refs.append(("event", state.history["head"], "state.history.head"))
        for field_path, edge, identity in record_reference_paths(
            "ExecutionVersionsV1", versions_body
        ):
            refs.append((edge, identity, f"state.versions_ref.{field_path}"))
        return [self._remember(kind, identity, path) for kind, identity, path in refs]

    def _phase3_contract_edges(
        self, body: Mapping[str, Any], *, private: bool, path: str
    ) -> list[tuple[str, str]]:
        """Validate supported Phase 3 envelopes and expose their typed closure."""
        try:
            references = phase3_typed_artifact_references(body, private=private)
        except TypedArtifactError as exc:
            raise ProjectionError(f"{path}: {exc}") from exc
        return [
            self._remember(
                "private" if reference.private else "artifact",
                reference.identity,
                f"{path}.{reference.path}" if reference.path else path,
            )
            for reference in references
        ]

    def _validate_after(self, key: tuple[str, str], value: Any) -> None:
        kind, identity = key
        if kind == "artifact" or kind == "private":
            self._validate_artifact(value, identity)
            required = self.required_domain.get(identity)
            if kind == "artifact" and required is not None and value.domain not in required:
                raise WrongRecordDomainError(
                    f"payload edge {identity} targets {value.domain!r}, expected {sorted(required)}"
                )
        elif kind == "event":
            payload = self.loaded[("artifact", value.payload_ref)]
            if payload.domain != "payload":
                raise WrongRecordDomainError(
                    f"event payload {value.payload_ref} has domain {payload.domain!r}"
                )
            if value.previous is not None:
                predecessor = self.loaded[("event", value.previous)]
                if predecessor.seq + 1 != value.seq:
                    raise CorruptRecordError("event predecessor sequence is not contiguous")
            for caused_by in value.caused_by:
                if self.loaded[("event", caused_by)].seq >= value.seq:
                    raise CorruptRecordError("an event cause must precede the caused event")
        elif kind == "context_revision":
            if ("context_node", value.content_ref) not in self.loaded:
                raise CorruptRecordError("context revision content was not validated")
        elif kind == "context_node":
            # The record codec enforces the root/child fields. All root pins and
            # any parent node have already been resolved by typed closure.
            return
        elif kind == "checkpoint":
            self._validate_checkpoint(value)
        elif kind == "commit":
            self._validate_commit(value)

    def _validate_artifact(self, artifact: _Artifact, identity: str) -> None:
        if artifact.domain in {"payload", "payload:bytes"}:
            return
        if artifact.domain == "message" and not artifact.private:
            try:
                MessageV1.from_dict(artifact.value)
            except (TypeError, ValueError) as exc:
                raise CorruptRecordError(f"invalid message artifact: {identity}") from exc
            return
        raise WrongRecordDomainError(
            f"unsupported artifact domain/location: {artifact.domain!r}, private={artifact.private}"
        )

    def _validate_checkpoint(self, checkpoint: CheckpointV1) -> None:
        head = checkpoint.event_head
        expected_seq = checkpoint.state.history["seq"]
        if head is None:
            if expected_seq != 0:
                raise CorruptRecordError("empty event history has nonzero sequence")
        else:
            event = self.loaded[("event", head)]
            if event.seq != expected_seq:
                raise CorruptRecordError("checkpoint event cursor sequence differs from its head")
            if event.lineage_id != checkpoint.state.position["lineage_id"]:
                raise CorruptRecordError("checkpoint event head and state lineages differ")
        if checkpoint.parents:
            parent = self.loaded[("checkpoint", checkpoint.parents[0])]
            ancestor = parent.event_head
            current = head
            while ancestor is not None and current is not None and current != ancestor:
                current = self.loaded[("event", current)].previous
            if ancestor is not None and current != ancestor:
                raise CorruptRecordError(
                    "checkpoint event history does not descend from its parent"
                )

    def _validate_commit(self, commit: CommitV1) -> None:
        checkpoint = self.loaded[("checkpoint", commit.checkpoint)]
        if not commit.events:
            raise CorruptRecordError("a commit must contain at least one event")
        if commit.parent_commit is not None:
            parent_commit = self.loaded[("commit", commit.parent_commit)]
            parent_id = parent_commit.checkpoint
            if checkpoint.parents != (parent_id,):
                raise CorruptRecordError("commit checkpoint does not descend from parent commit")
            parent_checkpoint = self.loaded[("checkpoint", parent_id)]
            if (
                parent_checkpoint.state.position["lineage_id"]
                != checkpoint.state.position["lineage_id"]
            ):
                raise CorruptRecordError("ordinary commit ancestry crosses lineages")
        elif checkpoint.parents:
            parent_id = checkpoint.parents[0]
        else:
            raise CorruptRecordError("a runtime commit requires a parent checkpoint")
        parent = self.loaded[("checkpoint", parent_id)]
        previous = parent.event_head
        sequence = parent.state.history["seq"]
        lineage = checkpoint.state.position["lineage_id"]
        for event_id in commit.events:
            event = self.loaded[("event", event_id)]
            if event.previous != previous or event.seq != sequence + 1:
                raise CorruptRecordError("commit events are not an ordered predecessor suffix")
            if event.lineage_id != lineage:
                raise CorruptRecordError("commit event lineage does not match checkpoint state")
            previous = event_id
            sequence = event.seq
        if previous != checkpoint.event_head or sequence != checkpoint.state.history["seq"]:
            raise CorruptRecordError("commit events do not end at the checkpoint event cursor")


class LineageLock:
    def __init__(self, store: TaskGraphStore, lineage_id: str) -> None:
        self.store = store
        self.lineage_id = lineage_id
        self.stream = None
        self.thread_lock: threading.Lock | None = None

    def __enter__(self) -> None:
        with self.store._thread_locks_guard:
            self.thread_lock = self.store._thread_locks.setdefault(
                self.lineage_id, threading.Lock()
            )
        self.thread_lock.acquire()
        path = self.store.root / "operations" / f"{self.store._lineage_name(self.lineage_id)}.lock"
        try:
            self.store._require_store_directory(path.parent)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                os.close(descriptor)
                raise CorruptRecordError("lineage lock is not a regular file")
            os.fchmod(descriptor, 0o600)
            self.stream = os.fdopen(descriptor, "r+b", buffering=0)
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        except Exception:
            self.thread_lock.release()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        assert self.stream is not None
        assert self.thread_lock is not None
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
        finally:
            self.thread_lock.release()


__all__ = ["ClosureValidator", "LineageLock"]
