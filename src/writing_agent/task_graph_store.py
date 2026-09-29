"""Private content-addressed persistence for task-graph records and artifacts.

The immutable files in this store are not authority by themselves.  A lineage's
canonical ``refs`` file is the only mutable authority, and replacing that file is
the publication linearization point.

Lexical parent traversal and static symlink ancestry are rejected before one absolute
path is used for checks and creation. This trusted-harness boundary does not attempt to
defeat a process that races path replacement after validation.
"""

from __future__ import annotations

import base64
import os
import re
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from writing_agent.task_graph import (
    CheckpointV1,
    CommitV1,
    EnvironmentStateV1,
    EventV1,
    GraphInstanceV1,
    MaterializedContextV1,
    MessageV1,
    Record,
    canonical_bytes,
    domain_hash,
    domain_hash_bytes,
    load_canonical_json,
    thaw,
    validate_hash,
)
from writing_agent.task_graph_closure import _ClosureValidator, _LineageLock
from writing_agent.task_graph_errors import (
    ConcurrentUpdateError,
    CorruptRecordError,
    MaterializationError,
    MissingReferenceError,
    ProjectionError,
    StoreError,
    WrongRecordDomainError,
)
from writing_agent.task_graph_operation import operation_scoped
from writing_agent.task_graph_records import (
    RECORD_TYPES,
    ContextContentV1,
    ContextRevisionV1,
    materialize_context_nodes,
    record_reference_edges,
)
from writing_agent.task_graph_wire import WireRecord

_LINEAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CommitVerifier(Protocol):
    """Semantic authority injected at publication and explicit view verification."""

    def view(self, store: TaskGraphStore, checkpoint_id: str) -> Any: ...

    def verify_commit(
        self,
        store: TaskGraphStore,
        base_checkpoint_id: str,
        events: Sequence[EventV1],
        next_state: EnvironmentStateV1,
    ) -> Any: ...


FaultHook = Callable[[str], None]


def _noop_fault(stage: str) -> None:
    del stage


class TaskGraphStore:
    """Private content-addressed storage for task-graph records and artifacts.

    ``expected_head`` exists only as a method argument.  It is never serialized
    into the authoritative ref, whose exact schema is ``{"head_commit": ...}``.
    """

    _RECORD_DIRS = {
        GraphInstanceV1: "instances",
        EventV1: "events",
        ContextContentV1: "context_content",
        ContextRevisionV1: "context_revisions",
        CheckpointV1: "checkpoints",
        CommitV1: "commits",
    }

    def __init__(
        self,
        root: Path | str,
        *,
        verifier: CommitVerifier,
        max_record_bytes: int = 16_000_000,
    ) -> None:
        if verifier is None:
            raise TypeError("verifier is required")
        if type(max_record_bytes) is not int or max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be a positive integer")
        self.root = self._verified_absolute_path(Path(root), MaterializationError)
        self.max_record_bytes = max_record_bytes
        self._verifier = verifier
        self._thread_locks: dict[str, threading.Lock] = {}
        self._thread_locks_guard = threading.Lock()
        self._session = threading.local()
        self._prepare_store()

    @property
    def verifier(self) -> CommitVerifier:
        """Return the configured semantic verifier without exposing store internals."""
        return self._verifier

    # -- immutable object codecs -------------------------------------------------

    @operation_scoped
    def put_artifact(self, value: Any, *, domain: str = "payload", private: bool = False) -> str:
        if domain not in {"payload", "message"} or (domain == "message" and private):
            raise ValueError("artifacts may use only payload or message domains")
        if domain == "payload" and isinstance(value, Mapping) and "record_type" in value:
            record_type = value["record_type"]
            if not isinstance(record_type, str):
                raise ValueError("record_type must be a string")
            if record_type in RECORD_TYPES:
                try:
                    body = load_canonical_json(canonical_bytes(value))
                    record_reference_edges(record_type, body)
                except Exception as exc:
                    raise ValueError(f"invalid task-graph payload record: {record_type}") from exc
        identity = domain_hash(domain, value)
        envelope = {
            "schema": 1,
            "domain": domain,
            "encoding": "json",
            "body": value,
        }
        self._write_immutable(self._artifact_path(identity, private), canonical_bytes(envelope))
        return identity

    @operation_scoped
    def put_bytes_artifact(
        self, value: bytes, *, domain: str = "payload", private: bool = False
    ) -> str:
        identity = domain_hash_bytes(domain, value)
        envelope = {
            "schema": 1,
            "domain": f"{domain}:bytes",
            "encoding": "base64",
            "body": base64.b64encode(value).decode("ascii"),
        }
        self._write_immutable(self._artifact_path(identity, private), canonical_bytes(envelope))
        return identity

    @operation_scoped
    def get_artifact(
        self,
        identity: str,
        *,
        expected_domain: str | None = None,
        private: bool = False,
    ) -> Any:
        validate_hash(identity)
        artifact = self._validator().validate(("private" if private else "artifact", identity))
        value, domain = artifact.value, artifact.domain
        if expected_domain is not None and domain != expected_domain:
            raise WrongRecordDomainError(
                f"artifact {identity} has domain {domain!r}, expected {expected_domain!r}"
            )
        return thaw(value)

    @operation_scoped
    def artifact_visibilities(self, identity: str) -> frozenset[str]:
        """Return storage visibility locations without decoding artifact contents."""
        validate_hash(identity)
        result = set()
        if self._artifact_path(identity, False).exists():
            result.add("public")
        if self._artifact_path(identity, True).exists():
            result.add("private")
        return frozenset(result)

    @operation_scoped
    def persist_artifact(self, artifact: Any) -> str:
        """Persist one identity-checked derived artifact in its declared domain."""
        value = artifact.value
        if artifact.kind in {"context_node", "context_revision"}:
            identity = self.persist(value)
        elif artifact.value_kind == "bytes":
            identity = self.put_bytes_artifact(value, private=artifact.kind == "private")
        else:
            if isinstance(value, WireRecord):
                body = value.to_wire()
            elif isinstance(value, Record):
                body = value.to_dict()
            else:
                body = load_canonical_json(value)
            identity = self.put_artifact(body, private=artifact.kind == "private")
        if identity != artifact.ref:
            raise ProjectionError("persisted artifact identity differs from its derive")
        return identity

    @operation_scoped
    def persist(self, record: Any) -> str:
        """Persist one typed immutable record, plus context content when needed."""
        if isinstance(record, MessageV1):
            return self.put_artifact(record.to_dict(), domain="message")
        for record_type, directory in self._RECORD_DIRS.items():
            if isinstance(record, record_type):
                return self._write_record(record, directory)
        raise TypeError(f"unsupported persisted record: {type(record).__name__}")

    @operation_scoped
    def load_instance(self, identity: str) -> GraphInstanceV1:
        return self._validator().validate(("instance", identity))

    @operation_scoped
    def load_event(self, identity: str) -> EventV1:
        return self._validator().validate(("event", identity))

    @operation_scoped
    def load_context_revision(self, identity: str) -> ContextRevisionV1:
        """Load a chained transition-seam context revision."""
        return self._validator().validate(("context_revision", identity))

    @operation_scoped
    def materialize_context(self, identity: str) -> MaterializedContextV1:
        """Flatten a new context chain after validating its complete closure."""
        validator = self._validator()
        revision = validator.validate(("context_revision", identity))
        nodes = {
            node_identity: value
            for (kind, node_identity), value in validator.loaded.items()
            if kind == "context_node"
        }
        return materialize_context_nodes(revision, nodes)

    @operation_scoped
    def load_checkpoint(self, identity: str) -> CheckpointV1:
        return self._validator().validate(("checkpoint", identity))

    @operation_scoped
    def load_commit(self, identity: str) -> CommitV1:
        return self._validator().validate(("commit", identity))

    @operation_scoped
    def save_checkpoint(
        self,
        state: EnvironmentStateV1,
        *,
        parent: str | None = None,
        artifact_refs: Sequence[str] = (),
    ) -> str:
        """Save a structurally validated checkpoint without publishing a lineage.

        Semantic verification belongs to publication and gate views. Checkpoint storage
        remains useful for unpublished roots and candidates.
        """
        if not isinstance(state, EnvironmentStateV1):
            raise TypeError("state must be EnvironmentStateV1")
        parents = () if parent is None else (parent,)
        checkpoint = CheckpointV1(
            parents=parents,
            state=state,
            event_head=state.history["head"],
            artifact_refs=tuple(artifact_refs),
        )
        validator = self._validator()
        validator.add_virtual("checkpoint", checkpoint.identity(), checkpoint)
        validator.validate(("checkpoint", checkpoint.identity()))
        return self.persist(checkpoint)

    # -- authority and atomic publication ---------------------------------------

    @operation_scoped
    def read_head(self, lineage_id: str) -> str | None:
        return self._read_head(lineage_id, self._validator())

    def _read_head(self, lineage_id: str, validator: _ClosureValidator) -> str | None:
        path = self._ref_path(lineage_id)
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        value = self._read_canonical(path)
        if not isinstance(value, dict) or set(value) != {"head_commit"}:
            raise CorruptRecordError(f"invalid lineage authority: {path}")
        head = value["head_commit"]
        try:
            validate_hash(head, optional=True)
        except (TypeError, ValueError) as exc:
            raise CorruptRecordError(f"invalid lineage head: {path}") from exc
        if head is not None:
            commit = validator.validate(("commit", head))
            checkpoint = validator.validate(("checkpoint", commit.checkpoint))
            if checkpoint.state.position["lineage_id"] != lineage_id:
                raise CorruptRecordError(
                    f"lineage authority {lineage_id!r} targets "
                    f"{checkpoint.state.position['lineage_id']!r}"
                )
        return head

    @operation_scoped
    def publish(
        self,
        lineage_id: str,
        expected_head: str | None,
        events: Sequence[EventV1],
        next_state: EnvironmentStateV1,
        *,
        artifact_refs: Sequence[str] = (),
        parent_checkpoint: str | None = None,
        fault: FaultHook | None = None,
        _validation: _ClosureValidator | None = None,
    ) -> str:
        """Atomically publish an event batch, checkpoint, commit, and lineage head.

        ``parent_checkpoint`` is only valid for a lineage's first commit. Repeating
        the identical CAS
        request after publication is idempotent and returns the existing commit.
        """
        self._lineage_name(lineage_id)
        validate_hash(expected_head, optional=True)
        validate_hash(parent_checkpoint, optional=True)
        if expected_head is not None and parent_checkpoint is not None:
            raise ValueError("parent_checkpoint is only valid for an initial lineage commit")
        if not isinstance(next_state, EnvironmentStateV1):
            raise TypeError("next_state must be EnvironmentStateV1")
        if next_state.position["lineage_id"] != lineage_id:
            raise ValueError("state lineage does not match publication lineage")
        batch = tuple(events)
        if any(not isinstance(event, EventV1) for event in batch):
            raise TypeError("events must contain EventV1 records")
        if not batch:
            raise ValueError("a published commit requires at least one event")
        hook = fault or _noop_fault
        validator = _validation or self._validator()
        base_checkpoint = parent_checkpoint
        if expected_head is not None:
            base_checkpoint = validator.validate(("commit", expected_head)).checkpoint
        if base_checkpoint is None:
            raise StoreError("publication requires an actual parent checkpoint")
        base = validator.validate(("checkpoint", base_checkpoint))
        if expected_head is None and base.parents:
            raise ProjectionError("checkpoint.parents: initial lineage commit must start at root")
        verifier = self.verifier
        if artifact_refs:
            raise ProjectionError("checkpoint.artifact_refs: runtime checkpoints cannot add refs")
        checkpoint = CheckpointV1(
            parents=(base_checkpoint,),
            state=next_state,
            event_head=next_state.history["head"],
            artifact_refs=tuple(artifact_refs),
        )
        commit = CommitV1(
            parent_commit=expected_head,
            events=tuple(event.identity() for event in batch),
            checkpoint=checkpoint.identity(),
        )

        with self._lineage_lock(lineage_id):
            current = self._read_head(lineage_id, validator)
            if current == commit.identity():
                # A prior replace may have succeeded while its directory fsync failed.
                # Visibility is not durability, so an idempotent success repeats the
                # required barrier.
                self._fsync_directory(self.root / "refs")
                return current
            if current != expected_head:
                raise ConcurrentUpdateError(
                    f"stale lineage head for {lineage_id!r}: "
                    f"expected {expected_head}, got {current}"
                )
            hook("before_immutable_writes")
            candidate_keys = {
                *(("event", event.identity()) for event in batch),
                ("checkpoint", checkpoint.identity()),
                ("commit", commit.identity()),
            }
            try:
                for event in batch:
                    validator.add_virtual("event", event.identity(), event)
                validator.add_virtual("checkpoint", checkpoint.identity(), checkpoint)
                validator.add_virtual("commit", commit.identity(), commit)
                validator.gated_candidates.update(candidate_keys)
                # Structural closure runs before the one semantic verifier.
                try:
                    validator.validate(("commit", commit.identity()))
                except WrongRecordDomainError as exc:
                    raise ProjectionError(
                        "artifact.visibility: candidate reference has the wrong visibility"
                    ) from exc
                verifier.verify_commit(self, base_checkpoint, batch, next_state)
                for event in batch:
                    self.persist(event)
                self.persist(checkpoint)
                self.persist(commit)
                hook("after_immutable_writes")
                hook("before_head_publication")
                self._replace_head(lineage_id, commit.identity())
                hook("after_head_publication")
                return commit.identity()
            except BaseException:
                validator.discard(candidate_keys)
                raise

    # -- operation-scoped closure validation ------------------------------------

    @contextmanager
    def operation(self):
        """Share closure results for one operation; never yield inside this context.

        The thread-local session remains active until the context exits, so yielding
        from a suspended generator would expose its verified cache to unrelated calls.
        """
        validator = getattr(self._session, "validator", None)
        if validator is not None:
            yield
            return
        validator = _ClosureValidator(self)
        self._session.validator = validator
        try:
            try:
                yield
            except BaseException:
                raise
            else:
                if os.environ.get("CWA_TASK_GRAPH_AUDIT_SCOPE_EXIT") == "1":
                    validator.audit_artifacts()
        finally:
            self._session.validator = None

    def _validator(self) -> _ClosureValidator:
        """Return a fresh validator seeded from the open operation session, if any."""
        return _ClosureValidator(self, parent=getattr(self._session, "validator", None))

    def _prepare_store(self) -> None:
        # The caller supplies one already-established durable parent.  We create
        # only the configured root and its fixed children, so every possible
        # visible mkdir has one unambiguous containing-directory barrier to retry.
        self._require_private_directory(self.root.parent)
        try:
            self.root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        self._require_private_directory(self.root)
        # Repeat even when root was already visible: a prior constructor may have
        # failed after mkdir and before this barrier completed.
        self._fsync_directory(self.root.parent)
        for name in (
            "instances",
            "checkpoints",
            "events",
            "context_content",
            "context_revisions",
            "artifacts",
            "private",
            "commits",
            "refs",
            "operations",
        ):
            path = self.root / name
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                pass
            self._require_private_directory(path)
            self._fsync_directory(self.root)

    @staticmethod
    def _require_private_directory(path: Path) -> None:
        try:
            info = path.lstat()
        except OSError as exc:
            raise MaterializationError(f"missing or inaccessible directory: {path}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise MaterializationError(f"not a real directory: {path}")
        if info.st_mode & 0o077:
            raise MaterializationError(f"directory is not private: {path}")

    @staticmethod
    def _verified_absolute_path(path: Path, error_type: type[Exception]) -> Path:
        """Reject lexical traversal, inspect static ancestry, and normalize once."""
        if ".." in path.parts:
            raise error_type(f"parent traversal is not allowed: {path}")
        absolute = Path(os.path.abspath(path))
        TaskGraphStore._reject_symlink_ancestry(absolute, error_type)
        return absolute

    @staticmethod
    def _reject_symlink_ancestry(path: Path, error_type: type[Exception]) -> None:
        """Reject existing symlinks in a lexical path; races are out of scope."""
        components = (path, *path.parents)
        for component in reversed(components):
            try:
                info = component.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise error_type(f"cannot inspect path ancestry: {component}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise error_type(f"symlink ancestry is not allowed: {component}")
            if component != path and not stat.S_ISDIR(info.st_mode):
                raise error_type(f"path ancestor is not a directory: {component}")

    def _record_path(self, directory: str, identity: str) -> Path:
        validate_hash(identity)
        return self.root / directory / f"{identity}.json"

    def _artifact_path(self, identity: str, private: bool) -> Path:
        validate_hash(identity)
        return self.root / ("private" if private else "artifacts") / identity

    @staticmethod
    def _lineage_name(lineage_id: str) -> str:
        if not isinstance(lineage_id, str) or not _LINEAGE_RE.fullmatch(lineage_id):
            raise ValueError("lineage_id is not safe for a ref filename")
        return lineage_id

    def _ref_path(self, lineage_id: str) -> Path:
        return self.root / "refs" / f"{self._lineage_name(lineage_id)}.json"

    def _write_record(self, record: Any, directory: str) -> str:
        identity = record.identity()
        wire = record.to_wire() if isinstance(record, WireRecord) else record
        self._write_immutable(self._record_path(directory, identity), canonical_bytes(wire))
        return identity

    def _write_immutable(self, path: Path, data: bytes) -> None:
        self._require_store_directory(path.parent)
        if len(data) > self.max_record_bytes:
            raise StoreError("immutable record exceeds storage byte limit")
        try:
            existing = self._read_bytes(path)
        except MissingReferenceError:
            existing = None
        if existing is not None:
            if existing != data:
                raise CorruptRecordError(f"immutable object collision at {path}")
            # A previous link may have succeeded while the containing-directory
            # fsync failed.  Equal visible bytes do not waive that barrier.
            self._fsync_directory(path.parent)
            return
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                if self._read_bytes(path) != data:
                    raise CorruptRecordError(f"immutable object collision at {path}") from exc
            self._fsync_directory(path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _read_bytes(self, path: Path) -> bytes:
        self._require_store_directory(path.parent)
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise MissingReferenceError(f"missing immutable reference: {path.name}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise CorruptRecordError(f"stored object is not a regular file: {path}")
        if info.st_size > self.max_record_bytes:
            raise CorruptRecordError(f"stored object exceeds byte limit: {path}")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise CorruptRecordError(f"cannot read stored object: {path}") from exc

    def _read_canonical(self, path: Path) -> Any:
        raw = self._read_bytes(path)
        try:
            return load_canonical_json(raw)
        except (TypeError, ValueError) as exc:
            raise CorruptRecordError(f"noncanonical stored JSON: {path}") from exc

    def _load_record(self, identity: str, record_type: type[Any], directory: str):
        validate_hash(identity)
        raw = self._read_bytes(self._record_path(directory, identity))
        try:
            record = record_type.from_json(raw)
        except (TypeError, ValueError) as exc:
            raise CorruptRecordError(
                f"stored {record_type.__name__} is invalid: {identity}"
            ) from exc
        if record.identity() != identity:
            raise CorruptRecordError(f"stored {record_type.__name__} hash mismatch")
        return record

    def _decode_artifact(self, path: Path, identity: str) -> tuple[Any, str]:
        envelope = self._read_canonical(path)
        if not isinstance(envelope, dict) or set(envelope) != {
            "schema",
            "domain",
            "encoding",
            "body",
        }:
            raise CorruptRecordError(f"invalid artifact envelope: {identity}")
        if envelope["schema"] != 1 or not isinstance(envelope["domain"], str):
            raise CorruptRecordError(f"invalid artifact envelope: {identity}")
        domain = envelope["domain"]
        encoding = envelope["encoding"]
        try:
            if encoding == "json" and not domain.endswith(":bytes"):
                value = envelope["body"]
                actual = domain_hash(domain, value)
            elif encoding == "base64" and domain.endswith(":bytes"):
                encoded = envelope["body"]
                if not isinstance(encoded, str):
                    raise ValueError("binary artifact body is not text")
                value = base64.b64decode(encoded, validate=True)
                actual = domain_hash_bytes(domain.removesuffix(":bytes"), value)
            else:
                raise ValueError("artifact encoding and domain disagree")
        except (TypeError, ValueError) as exc:
            raise CorruptRecordError(f"invalid artifact body: {identity}") from exc
        if actual != identity:
            raise CorruptRecordError(f"artifact hash mismatch: {identity}")
        return value, domain

    def _replace_head(self, lineage_id: str, head: str) -> None:
        destination = self._ref_path(lineage_id)
        self._require_store_directory(destination.parent)
        data = canonical_bytes({"head_commit": head})
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary_path = Path(temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
            self._fsync_directory(destination.parent)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _lineage_lock(self, lineage_id: str):
        return _LineageLock(self, lineage_id)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _require_store_directory(self, path: Path) -> None:
        try:
            info = path.lstat()
        except OSError as exc:
            raise CorruptRecordError(f"missing store directory: {path}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise CorruptRecordError(f"store path is not a real directory: {path}")
        if info.st_mode & 0o077:
            raise CorruptRecordError(f"store directory is not private: {path}")


__all__ = [
    "CommitVerifier",
    "ConcurrentUpdateError",
    "CorruptRecordError",
    "MaterializationError",
    "MissingReferenceError",
    "StoreError",
    "TaskGraphStore",
    "WrongRecordDomainError",
]
