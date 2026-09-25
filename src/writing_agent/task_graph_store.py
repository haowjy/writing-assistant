"""Private content-addressed persistence for task-graph checkpoints.

The immutable files in this store are not authority by themselves.  A lineage's
canonical ``refs`` file is the only mutable authority, and replacing that file is
the publication linearization point.  Writer workspaces are disposable, private
materializations and never contain store metadata or private artifacts.

Existing path components are checked for static symlinks and store/workspace trees
may not overlap. This trusted-harness boundary does not attempt to defeat a process
that races path replacement after validation.
"""

from __future__ import annotations

import base64
import difflib
import fcntl
import os
import re
import shutil
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from writing_agent.task_graph import (
    CheckpointV1,
    CommitV1,
    ContextContentV1,
    ContextRevisionV1,
    EnvironmentStateV1,
    EventV1,
    GraphInstanceV1,
    MessageV1,
    canonical_bytes,
    domain_hash,
    domain_hash_bytes,
    file_hash,
    load_canonical_json,
    safe_path,
    tree_hash,
    validate_file_tree,
    validate_hash,
)

DEFAULT_MAX_WORKSPACE_BYTES = 1_000_000
_LINEAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REPLAY_EFFECT = "Phase2RecordedEffectV1"


class StoreError(RuntimeError):
    """Base class for persistence failures."""


class MissingReferenceError(StoreError):
    """An immutable reference is absent."""


class CorruptRecordError(StoreError):
    """Stored bytes do not decode to the identity named by their path."""


class WrongRecordDomainError(CorruptRecordError):
    """A valid immutable object was used in a reference of the wrong type."""


class ConcurrentUpdateError(StoreError):
    """The lineage head did not match the compare-and-swap request."""


class MaterializationError(StoreError):
    """A checkpoint could not be safely materialized."""


class ReplayError(StoreError):
    """A recorded suffix did not deterministically reproduce its checkpoints."""


@dataclass(frozen=True)
class _Artifact:
    value: Any
    domain: str
    private: bool


@dataclass(frozen=True)
class RuntimeHandle:
    """Trusted restored state; it is never passed to a writer or model."""

    checkpoint_id: str
    state: EnvironmentStateV1
    context: ContextRevisionV1
    workspace: Path


@dataclass(frozen=True)
class FileDifference:
    path: str
    status: str
    before_hash: str | None
    after_hash: str | None
    text_diff: str | None


@dataclass(frozen=True)
class StateDifference:
    field: str
    before: Any
    after: Any


@dataclass(frozen=True)
class CheckpointDifference:
    files: tuple[FileDifference, ...]
    state: tuple[StateDifference, ...]


FaultHook = Callable[[str], None]


def _noop_fault(stage: str) -> None:
    del stage


class TaskGraphStore:
    """Deep, standard-library store for immutable task-graph state.

    ``expected_head`` exists only as a method argument.  It is never serialized
    into the authoritative ref, whose exact schema is ``{"head_commit": ...}``.
    """

    _RECORD_DIRS = {
        GraphInstanceV1: "instances",
        EventV1: "events",
        CheckpointV1: "checkpoints",
        CommitV1: "commits",
    }

    def __init__(
        self,
        root: Path | str,
        *,
        max_workspace_bytes: int = DEFAULT_MAX_WORKSPACE_BYTES,
        max_file_bytes: int | None = None,
        max_record_bytes: int = 16_000_000,
    ) -> None:
        if type(max_workspace_bytes) is not int or max_workspace_bytes < 0:
            raise ValueError("max_workspace_bytes must be a nonnegative integer")
        if max_file_bytes is None:
            max_file_bytes = max_workspace_bytes
        if type(max_file_bytes) is not int or max_file_bytes < 0:
            raise ValueError("max_file_bytes must be a nonnegative integer")
        if type(max_record_bytes) is not int or max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be a positive integer")
        self.root = Path(root)
        self.max_workspace_bytes = max_workspace_bytes
        self.max_file_bytes = max_file_bytes
        self.max_record_bytes = max_record_bytes
        self._thread_locks: dict[str, threading.Lock] = {}
        self._thread_locks_guard = threading.Lock()
        self._prepare_store()

    # -- immutable object codecs -------------------------------------------------

    def put_artifact(self, value: Any, *, domain: str = "payload", private: bool = False) -> str:
        identity = domain_hash(domain, value)
        envelope = {
            "schema": 1,
            "domain": domain,
            "encoding": "json",
            "body": value,
        }
        self._write_immutable(self._artifact_path(identity, private), canonical_bytes(envelope))
        return identity

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
        return value

    def persist(self, record: Any) -> str:
        """Persist one typed immutable record, plus context content when needed."""
        if isinstance(record, ContextRevisionV1):
            content = ContextContentV1.from_revision(record)
            self._write_record(content, "contexts")
            return self._write_record(record, "contexts")
        if isinstance(record, ContextContentV1):
            return self._write_record(record, "contexts")
        if isinstance(record, MessageV1):
            return self.put_artifact(record.to_dict(), domain="message")
        for record_type, directory in self._RECORD_DIRS.items():
            if isinstance(record, record_type):
                return self._write_record(record, directory)
        raise TypeError(f"unsupported persisted record: {type(record).__name__}")

    def load_instance(self, identity: str) -> GraphInstanceV1:
        return self._validator().validate(("instance", identity))

    def load_event(self, identity: str) -> EventV1:
        return self._validator().validate(("event", identity))

    def load_context(self, identity: str) -> ContextRevisionV1:
        return self._validator().validate(("context", identity))

    def load_checkpoint(self, identity: str) -> CheckpointV1:
        return self._validator().validate(("checkpoint", identity))

    def load_commit(self, identity: str) -> CommitV1:
        return self._validator().validate(("commit", identity))

    def save_checkpoint(
        self,
        state: EnvironmentStateV1,
        *,
        parent: str | None = None,
        artifact_refs: Sequence[str] = (),
    ) -> str:
        """Save a validated immutable checkpoint without changing any lineage head."""
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
            validator.validate(("commit", head))
        return head

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

        ``parent_checkpoint`` is only valid for a lineage's first commit (including
        a branch rooted at an immutable checkpoint).  Repeating the identical CAS
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
            raise ReplayError("publication requires an actual parent checkpoint")
        validator.validate(("checkpoint", base_checkpoint))
        parents = () if base_checkpoint is None else (base_checkpoint,)
        checkpoint = CheckpointV1(
            parents=parents,
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
            for event in batch:
                validator.add_virtual("event", event.identity(), event)
            validator.add_virtual("checkpoint", checkpoint.identity(), checkpoint)
            validator.add_virtual("commit", commit.identity(), commit)
            # Commit validation is the publication reducer seam.  It starts from
            # the actual parent state, rejects unsupported recorded transitions,
            # and requires exact full-state equality before any new immutable is
            # written.
            validator.validate(("commit", commit.identity()))
            for event in batch:
                self.persist(event)
            self.persist(checkpoint)
            self.persist(commit)
            hook("after_immutable_writes")
            hook("before_head_publication")
            self._replace_head(lineage_id, commit.identity())
            hook("after_head_publication")
            return commit.identity()

    def branch(
        self,
        parent_checkpoint: str,
        lineage_id: str,
        events: Sequence[EventV1],
        next_state: EnvironmentStateV1,
        *,
        artifact_refs: Sequence[str] = (),
        fault: FaultHook | None = None,
    ) -> str:
        """Publish the first commit of a new lineage from an immutable parent."""
        validator = self._validator()
        parent = validator.validate(("checkpoint", parent_checkpoint))
        if self._read_head(lineage_id, validator) is not None:
            raise ConcurrentUpdateError("a branch lineage must be new")
        if dict(next_state.files) != dict(parent.state.files):
            raise ValueError("branch initialization must start with the parent's full file state")
        if next_state.position["start_checkpoint"] != parent_checkpoint:
            raise ValueError("branch state must name its immutable start checkpoint")
        if next_state.history["branch_base"] != parent.event_head:
            raise ValueError("branch state must name the parent event head as branch_base")
        return self.publish(
            lineage_id,
            None,
            events,
            next_state,
            artifact_refs=artifact_refs,
            parent_checkpoint=parent_checkpoint,
            fault=fault,
            _validation=validator,
        )

    # -- materialization, restore, and inspection -------------------------------

    def materialize(
        self,
        checkpoint_id: str,
        fresh_root: Path | str,
        *,
        fault: FaultHook | None = None,
    ) -> Path:
        checkpoint = self.load_checkpoint(checkpoint_id)
        return self._materialize_checkpoint(checkpoint, fresh_root, fault=fault)

    def _materialize_checkpoint(
        self,
        checkpoint: CheckpointV1,
        fresh_root: Path | str,
        *,
        fault: FaultHook | None = None,
    ) -> Path:
        files = validate_file_tree(checkpoint.state.files)
        encoded = {path: text.encode("utf-8", "strict") for path, text in files.items()}
        total = sum(len(value) for value in encoded.values())
        if total > self.max_workspace_bytes:
            raise MaterializationError("workspace exceeds total UTF-8 byte limit")
        if any(len(value) > self.max_file_bytes for value in encoded.values()):
            raise MaterializationError("workspace file exceeds UTF-8 byte limit")

        destination = Path(fresh_root)
        self._reject_symlink_ancestry(destination, MaterializationError)
        store = Path(os.path.abspath(self.root))
        workspace = Path(os.path.abspath(destination))
        if workspace == store or workspace.is_relative_to(store) or store.is_relative_to(workspace):
            raise MaterializationError("workspace and canonical store trees must not overlap")
        hook = fault or _noop_fault
        created = False
        try:
            destination.mkdir(mode=0o700, parents=False, exist_ok=False)
            created = True
            self._require_private_directory(destination)
            hook("after_workspace_created")
            directories = sorted(
                {parent for path in files for parent in self._path_parents(path)},
                key=lambda value: (value.count("/"), value),
            )
            for directory in directories:
                target = destination.joinpath(*directory.split("/"))
                target.mkdir(mode=0o700)
                self._require_private_directory(target)
            for relative in sorted(files):
                target = destination.joinpath(*relative.split("/"))
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(target, flags, 0o600)
                try:
                    with os.fdopen(descriptor, "wb", closefd=True) as stream:
                        stream.write(encoded[relative])
                        stream.flush()
                        os.fsync(stream.fileno())
                except Exception:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    raise
                hook(f"after_file:{relative}")
            self._fsync_directory(destination)
            return destination
        except Exception as exc:
            if created:
                shutil.rmtree(destination, ignore_errors=True)
            if isinstance(exc, (StoreError, OSError)):
                raise
            raise MaterializationError("workspace materialization failed") from exc

    def restore(
        self,
        checkpoint_id: str,
        fresh_root: Path | str,
        *,
        fault: FaultHook | None = None,
    ) -> RuntimeHandle:
        checkpoint = self.load_checkpoint(checkpoint_id)
        workspace = self._materialize_checkpoint(checkpoint, fresh_root, fault=fault)
        try:
            context = self.load_context(checkpoint.state.context_ref)
            return RuntimeHandle(checkpoint_id, checkpoint.state, context, workspace)
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise

    def diff(self, before_id: str, after_id: str, *, text: bool = False) -> CheckpointDifference:
        before = self.load_checkpoint(before_id).state
        after = self.load_checkpoint(after_id).state
        paths = sorted(set(before.files) | set(after.files))
        file_changes: list[FileDifference] = []
        for path in paths:
            old = before.files.get(path)
            new = after.files.get(path)
            if old == new:
                continue
            status = "added" if old is None else "deleted" if new is None else "changed"
            rendered = None
            if text and old is not None and new is not None:
                rendered = "".join(
                    difflib.unified_diff(
                        old.splitlines(keepends=True),
                        new.splitlines(keepends=True),
                        fromfile=f"a/{path}",
                        tofile=f"b/{path}",
                    )
                )
            file_changes.append(
                FileDifference(
                    path,
                    status,
                    None if old is None else file_hash(old),
                    None if new is None else file_hash(new),
                    rendered,
                )
            )
        old_state = before.to_dict()
        new_state = after.to_dict()
        state_changes = tuple(
            StateDifference(field, old_state[field], new_state[field])
            for field in sorted(old_state)
            if field not in {"files", "tree_hash"} and old_state[field] != new_state[field]
        )
        return CheckpointDifference(tuple(file_changes), state_changes)

    # -- recorded replay ---------------------------------------------------------

    def replay(
        self,
        lineage_id: str,
        start_checkpoint: str,
        committed_suffix: Sequence[str],
    ) -> str:
        """Reduce a published suffix using recorded Phase 2 fixture effects only."""
        validator = self._validator()
        current_id = start_checkpoint
        current = validator.validate(("checkpoint", current_id))
        suffix = tuple(committed_suffix)
        if not suffix:
            return current_id
        if self._read_head(lineage_id, validator) != suffix[-1]:
            raise ReplayError("the supplied suffix is not the published lineage head")
        prior_commit: str | None = None
        for index, commit_id in enumerate(suffix):
            commit = validator.validate(("commit", commit_id))
            if index == 0:
                if commit.parent_commit is not None:
                    parent = validator.validate(("commit", commit.parent_commit))
                    if parent.checkpoint != current_id:
                        raise ReplayError("suffix does not start at the supplied checkpoint")
                elif tuple(validator.validate(("checkpoint", commit.checkpoint)).parents) != (
                    current_id,
                ):
                    raise ReplayError("root commit is not based on the supplied checkpoint")
            elif commit.parent_commit != prior_commit:
                raise ReplayError("commit suffix is not contiguous")
            target = validator.validate(("checkpoint", commit.checkpoint))
            if target.parents != (current_id,):
                raise ReplayError("checkpoint suffix is not contiguous")
            state = current.state
            for event_id in commit.events:
                event = validator.validate(("event", event_id))
                payload = validator.validate(("artifact", event.payload_ref))
                state = self._apply_recorded_effect_body(state, event, payload.value)
            if state != target.state:
                raise ReplayError("recorded effects do not reproduce the committed post-state")
            current_id = commit.checkpoint
            current = target
            prior_commit = commit_id
        return current_id

    # -- validation --------------------------------------------------------------

    def _validator(self) -> _ClosureValidator:
        """Return a fresh operation-scoped validator; nothing survives the call."""
        return _ClosureValidator(self)

    def _apply_recorded_effect_body(
        self, state: EnvironmentStateV1, event: EventV1, body: Any
    ) -> EnvironmentStateV1:
        """Future reducers register here; unknown transition envelopes fail closed."""
        if not isinstance(body, dict) or set(body) != {
            "artifact_type",
            "before_state_ref",
            "file_delta",
            "set",
            "history_set",
        }:
            raise ReplayError("event payload is not a Phase 2 recorded effect")
        if body["artifact_type"] != _REPLAY_EFFECT:
            raise ReplayError("event payload is outside the Phase 2 replay reducer")
        if body["before_state_ref"] != state.identity():
            raise ReplayError("recorded effect pre-state does not match replay state")
        changes = body["set"]
        history_changes = body["history_set"]
        deltas = body["file_delta"]
        if not isinstance(changes, dict) or not isinstance(history_changes, dict):
            raise ReplayError("recorded state changes must be objects")
        allowed = set(state.to_dict()) - {"schema", "files", "tree_hash", "history"}
        if not set(changes) <= allowed:
            raise ReplayError("recorded effect changes protected state fields")
        history_allowed = set(state.history) - {"head", "seq"}
        if not set(history_changes) <= history_allowed:
            raise ReplayError("recorded effect changes protected history fields")
        if not isinstance(deltas, dict):
            raise ReplayError("file_delta must be an object")
        files = dict(state.files)
        for path, delta in deltas.items():
            safe_path(path)
            if not isinstance(delta, dict) or set(delta) != {"before", "after"}:
                raise ReplayError("invalid recorded file delta")
            before = delta["before"]
            after = delta["after"]
            if before is not None and not isinstance(before, str):
                raise ReplayError("file delta before value must be text or null")
            if after is not None and not isinstance(after, str):
                raise ReplayError("file delta after value must be text or null")
            actual = files.get(path)
            if actual != before or (before is None and path in files):
                raise ReplayError("recorded file delta precondition failed")
            if after is None:
                files.pop(path, None)
            else:
                files[path] = after
        validate_file_tree(files)
        value = state.to_dict()
        value.update(changes)
        value["files"] = files
        value["tree_hash"] = tree_hash(files)
        history = state.to_dict()["history"]
        history.update(history_changes)
        if event.previous != history["head"] or event.seq != history["seq"] + 1:
            raise ReplayError("recorded event does not extend the replay cursor")
        history["head"] = event.identity()
        history["seq"] = event.seq
        value["history"] = history
        try:
            result = EnvironmentStateV1.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise ReplayError("recorded effect produces an invalid state") from exc
        if event.lineage_id != result.position["lineage_id"]:
            raise ReplayError("recorded event and resulting state lineages differ")
        return result

    # -- filesystem internals ----------------------------------------------------

    def _prepare_store(self) -> None:
        self._reject_symlink_ancestry(self.root, MaterializationError)
        absolute = Path(os.path.abspath(self.root))
        missing: list[Path] = []
        cursor = absolute
        while not cursor.exists():
            missing.append(cursor)
            cursor = cursor.parent
        for path in reversed(missing):
            path.mkdir(mode=0o700)
            self._require_private_directory(path)
            self._fsync_directory(path.parent)
        self._require_private_directory(absolute)
        # Reopening repairs an initialization whose mkdir became visible but whose
        # parent-directory barrier previously failed.
        self._fsync_directory(absolute.parent)
        self.root = absolute
        for name in (
            "instances",
            "checkpoints",
            "events",
            "contexts",
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
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise MaterializationError(f"not a real directory: {path}")
        if info.st_mode & 0o077:
            raise MaterializationError(f"directory is not private: {path}")

    @staticmethod
    def _reject_symlink_ancestry(path: Path, error_type: type[Exception]) -> None:
        """Reject existing symlinks in a lexical path; races are out of scope."""
        absolute = Path(os.path.abspath(path))
        components = (absolute, *absolute.parents)
        for component in reversed(components):
            try:
                info = component.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise error_type(f"cannot inspect path ancestry: {component}") from exc
            if stat.S_ISLNK(info.st_mode):
                raise error_type(f"symlink ancestry is not allowed: {component}")
            if component != absolute and not stat.S_ISDIR(info.st_mode):
                raise error_type(f"path ancestor is not a directory: {component}")

    @staticmethod
    def _path_parents(path: str) -> tuple[str, ...]:
        pieces = path.split("/")[:-1]
        return tuple("/".join(pieces[:index]) for index in range(1, len(pieces) + 1))

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
        self._write_immutable(self._record_path(directory, identity), canonical_bytes(record))
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
            raise WrongRecordDomainError(
                f"{identity} is not a valid {record_type.__name__}"
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


class _ClosureValidator:
    """One linear, typed closure traversal for a public store operation.

    Loaded objects, active nodes, and completed nodes live only for this operation.
    The explicit stack keeps deep event, checkpoint, and commit ancestry off the
    Python call stack.  Every edge, including supplemental/imported references,
    resolves through this same domain-aware loader.
    """

    _RECORDS: dict[str, tuple[type[Any], str]] = {
        "instance": (GraphInstanceV1, "instances"),
        "event": (EventV1, "events"),
        "context": (ContextRevisionV1, "contexts"),
        "context_content": (ContextContentV1, "contexts"),
        "checkpoint": (CheckpointV1, "checkpoints"),
        "commit": (CommitV1, "commits"),
    }
    _LOCATION_KINDS = {
        "instances": "instance",
        "events": "event",
        "checkpoints": "checkpoint",
        "commits": "commit",
        "artifacts": "artifact",
        "private": "private",
    }

    def __init__(self, store: TaskGraphStore) -> None:
        self.store = store
        self.loaded: dict[tuple[str, str], Any] = {}
        self.virtual: dict[tuple[str, str], Any] = {}
        self.resolved: dict[tuple[str, str], tuple[str, str]] = {}
        self.active: set[tuple[str, str]] = set()
        self.completed: set[tuple[str, str]] = set()

    def add_virtual(self, kind: str, identity: str, value: Any) -> None:
        key = (kind, identity)
        existing = self.virtual.get(key)
        if existing is not None and existing != value:
            raise CorruptRecordError(f"conflicting in-memory object for {identity}")
        self.virtual[key] = value

    def validate(self, requested: tuple[str, str]) -> Any:
        root = self._resolve_key(requested)
        stack: list[tuple[tuple[str, str], bool]] = [(root, False)]
        while stack:
            raw_key, exiting = stack.pop()
            key = self._resolve_key(raw_key)
            if exiting:
                self._validate_after(key, self.loaded[key])
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
        return self.loaded[root]

    def _path_locations(self, identity: str) -> list[str]:
        validate_hash(identity)
        locations: list[str] = []
        for directory in (
            "instances",
            "events",
            "contexts",
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
            return self.resolved[requested]
        if requested in self.loaded:
            return requested
        kind, identity = requested
        validate_hash(identity)
        virtual_kinds = {candidate for candidate, item in self.virtual if item == identity}
        locations = self._path_locations(identity)
        disk_kinds: set[str] = set()
        for location in locations:
            if location == "contexts":
                disk_kinds.add("contexts")
            else:
                disk_kinds.add(self._LOCATION_KINDS[location])

        if kind == "any":
            candidates = set(virtual_kinds)
            candidates.update(disk_kinds)
            if not candidates:
                raise MissingReferenceError(f"missing immutable reference: {identity}")
            # A virtual value and its same typed persisted location are one
            # candidate; distinct domains/locations are always ambiguous.
            normalized = {
                "contexts" if value.startswith("context") else value for value in candidates
            }
            if len(normalized) != 1:
                raise WrongRecordDomainError(
                    f"reference {identity} exists in ambiguous domains: {sorted(normalized)}"
                )
            resolved = next(iter(normalized))
            if resolved == "contexts":
                resolved = self._infer_context_kind(identity)
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
        if expected_location == "contexts":
            actual = self._infer_context_kind(identity)
            if actual != kind:
                raise WrongRecordDomainError(f"reference {identity} is {actual}, expected {kind}")
        self.resolved[requested] = requested
        return requested

    @staticmethod
    def _expected_location(kind: str) -> str:
        if kind in {"artifact", "private"}:
            return "artifacts" if kind == "artifact" else "private"
        try:
            return _ClosureValidator._RECORDS[kind][1]
        except KeyError as exc:
            raise AssertionError(f"unknown closure node kind: {kind}") from exc

    def _infer_context_kind(self, identity: str) -> str:
        for kind in ("context", "context_content"):
            virtual = self.virtual.get((kind, identity))
            if virtual is not None:
                return kind
        raw = self.store._read_bytes(self.store._record_path("contexts", identity))
        matches: list[str] = []
        for kind in ("context", "context_content"):
            record_type = self._RECORDS[kind][0]
            try:
                value = record_type.from_json(raw)
            except (TypeError, ValueError):
                continue
            if value.identity() == identity:
                matches.append(kind)
                self.loaded[(kind, identity)] = value
        if len(matches) != 1:
            raise WrongRecordDomainError(f"ambiguous or invalid context record: {identity}")
        return matches[0]

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

    def _edges(self, key: tuple[str, str], value: Any) -> list[tuple[str, str]]:
        kind, _ = key
        if kind == "instance":
            refs = [value.template_ref, *value.source_refs, *value.request_refs]
            if value.requirements_ref is not None:
                refs.append(value.requirements_ref)
            refs.extend(node.entry_contract for node in value.nodes)
            return [("artifact", identity) for identity in refs]
        if kind == "event":
            edges = [
                ("artifact", value.payload_ref),
                ("artifact", value.versions_ref),
                ("artifact", value.provenance_ref),
            ]
            if value.previous is not None:
                edges.append(("event", value.previous))
            edges.extend(("event", identity) for identity in value.caused_by)
            return edges
        if kind == "context":
            edges = [("context_content", value.content_hash)]
            if value.event_head is not None:
                edges.append(("event", value.event_head))
            edges.extend(("event", identity) for identity in value.provenance_refs)
            edges.extend(
                ("artifact", value.rendering[name])
                for name in ("template_ref", "tokenizer_ref", "tool_schema_ref")
            )
            return edges
        if kind == "context_content":
            return [
                ("artifact", getattr(value, name))
                for name in ("template_ref", "tokenizer_ref", "tool_schema_ref")
            ]
        if kind == "checkpoint":
            state = value.state
            edges = [("checkpoint", identity) for identity in value.parents]
            edges.extend(
                [
                    ("instance", state.instance_ref),
                    ("context", state.context_ref),
                    *(
                        ("artifact", identity)
                        for identity in (
                            state.position["entry_contract"],
                            state.requirements_ref,
                            state.decisions_ref,
                            state.disclosures_ref,
                            state.versions_ref,
                            state.budgets_ref,
                            state.rng_ref,
                            state.external_inputs_ref,
                            state.outcome_ref,
                            state.provenance_ref,
                        )
                    ),
                    *(("any", identity) for identity in state.history["imported_refs"]),
                    *(("any", identity) for identity in value.artifact_refs),
                ]
            )
            if state.author_packet_ref is not None:
                edges.append(("private", state.author_packet_ref))
            if state.continuation["author_request"] is not None:
                edges.append(("private", state.continuation["author_request"]))
            edges.extend(("private", identity) for identity in state.continuation["check_requests"])
            if state.history["branch_base"] is not None:
                edges.append(("event", state.history["branch_base"]))
            if state.position["start_checkpoint"] is not None:
                edges.append(("checkpoint", state.position["start_checkpoint"]))
            if value.event_head is not None:
                edges.append(("event", value.event_head))
            return edges
        if kind == "commit":
            edges = [("checkpoint", value.checkpoint)]
            if value.parent_commit is not None:
                edges.append(("commit", value.parent_commit))
            edges.extend(("event", identity) for identity in value.events)
            return edges
        return []

    def _validate_after(self, key: tuple[str, str], value: Any) -> None:
        kind, identity = key
        if kind == "artifact" or kind == "private":
            self._validate_artifact(value, identity)
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
        elif kind == "context":
            content = self.loaded[("context_content", value.content_hash)]
            if content != ContextContentV1.from_revision(value):
                raise CorruptRecordError("context content does not match its revision")
        elif kind == "checkpoint":
            self._validate_checkpoint(value)
        elif kind == "commit":
            self._validate_commit(value)

    @staticmethod
    def _validate_artifact(artifact: _Artifact, identity: str) -> None:
        if artifact.domain in {"payload", "payload:bytes"}:
            body = artifact.value
            if (
                artifact.domain == "payload"
                and isinstance(body, Mapping)
                and "artifact_type" in body
            ):
                if body["artifact_type"] != _REPLAY_EFFECT:
                    raise WrongRecordDomainError(
                        f"unsupported typed payload artifact: {body['artifact_type']!r}"
                    )
                required = {
                    "artifact_type",
                    "before_state_ref",
                    "file_delta",
                    "set",
                    "history_set",
                }
                if set(body) != required:
                    raise CorruptRecordError("invalid Phase2RecordedEffectV1 envelope")
                try:
                    validate_hash(body["before_state_ref"])
                except (TypeError, ValueError) as exc:
                    raise CorruptRecordError("invalid recorded-effect state reference") from exc
                if not all(
                    isinstance(body[name], Mapping) for name in ("file_delta", "set", "history_set")
                ):
                    raise CorruptRecordError("invalid recorded-effect edge fields")
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
        elif checkpoint.parents:
            parent_id = checkpoint.parents[0]
        else:
            raise ReplayError("a recorded transition commit requires a parent checkpoint")
        parent = self.loaded[("checkpoint", parent_id)]
        previous = parent.event_head
        sequence = parent.state.history["seq"]
        lineage = checkpoint.state.position["lineage_id"]
        state = parent.state
        for event_id in commit.events:
            event = self.loaded[("event", event_id)]
            if event.previous != previous or event.seq != sequence + 1:
                raise CorruptRecordError("commit events are not an ordered predecessor suffix")
            if event.lineage_id != lineage:
                raise CorruptRecordError("commit event lineage does not match checkpoint state")
            payload = self.loaded[("artifact", event.payload_ref)]
            try:
                state = self.store._apply_recorded_effect_body(state, event, payload.value)
            except ReplayError:
                raise
            previous = event_id
            sequence = event.seq
        if previous != checkpoint.event_head or sequence != checkpoint.state.history["seq"]:
            raise CorruptRecordError("commit events do not end at the checkpoint event cursor")
        if state != checkpoint.state:
            raise ReplayError("recorded effects do not reproduce the committed post-state")


class _LineageLock:
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


__all__ = [
    "DEFAULT_MAX_WORKSPACE_BYTES",
    "CheckpointDifference",
    "ConcurrentUpdateError",
    "CorruptRecordError",
    "FileDifference",
    "MaterializationError",
    "MissingReferenceError",
    "ReplayError",
    "RuntimeHandle",
    "StateDifference",
    "StoreError",
    "TaskGraphStore",
    "WrongRecordDomainError",
]
