"""Semantic verification for typed task-graph lineages."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from threading import RLock
from typing import Any

from writing_agent.task_graph import CheckpointV1, EventV1, MaterializedContextV1
from writing_agent.task_graph_admission import (
    AdmissionError,
    AdmittedGraphV1,
    StoreArtifactResolver,
    admit_graph,
)
from writing_agent.task_graph_admission import (
    AdmissionPolicyV1 as RuntimeAdmissionPolicy,
)
from writing_agent.task_graph_derive_author import DERIVES as AUTHOR_DERIVES
from writing_agent.task_graph_derive_context import DERIVES as CONTEXT_DERIVES
from writing_agent.task_graph_derive_entry import derive_entry, params_of
from writing_agent.task_graph_derive_outcome import DERIVES as OUTCOME_DERIVES
from writing_agent.task_graph_derive_writer import DERIVES as WRITER_DERIVES
from writing_agent.task_graph_errors import (
    CorruptRecordError,
    ProjectionError,
    StoreError,
)
from writing_agent.task_graph_record_contracts import (
    SEMANTICS_V1,
    ExecutionVersionsV1,
)
from writing_agent.task_graph_records import (
    RECORD_TYPES,
    AdmissionPolicyV1,
    EnvironmentStepV1,
)
from writing_agent.task_graph_transition import (
    ArtifactReader,
    LineageView,
    Transition,
    first_difference,
)

DeriveKey = str | tuple[str, str]
Derive = Callable[[LineageView, Any, ArtifactReader], Transition]


def _assemble_derives() -> dict[DeriveKey, Derive]:
    routes: dict[DeriveKey, Derive] = {}
    for mapping in (WRITER_DERIVES, AUTHOR_DERIVES, OUTCOME_DERIVES, CONTEXT_DERIVES):
        for key, derive in mapping.items():
            if key in routes:
                raise RuntimeError(f"duplicate task-graph derive route: {key!r}")
            routes[key] = derive
    return routes


DERIVE = _assemble_derives()


def derive_input(view: LineageView, input_record: Any, reader: ArtifactReader) -> Transition:
    """Dispatch one typed transition input through the shared derive registry."""
    try:
        record_type = getattr(input_record, "RECORD_TYPE", None)
        codec = RECORD_TYPES.get(record_type) if isinstance(record_type, str) else None
        if not isinstance(codec, type) or not isinstance(input_record, codec):
            raise ProjectionError("input.record_type: input is not a registered transition record")
        key: DeriveKey = record_type
        if isinstance(input_record, EnvironmentStepV1):
            key = (record_type, input_record.directive["kind"])
        derive = DERIVE.get(key)
        if derive is None:
            raise ProjectionError(f"input.record_type: no derive route for {key!r}")
        return derive(view, input_record, reader)
    except (ProjectionError, CorruptRecordError):
        raise
    except Exception as exc:
        raise ProjectionError(
            f"input.record_type: recorded input cannot be derived: {exc}"
        ) from exc


class StoreArtifactReader:
    """Read verified payloads and chained context through a task-graph store."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def artifact(self, ref: str, *, domain: str = "payload", private: bool = False) -> Any:
        if domain == "event":
            if private:
                raise ValueError("events cannot be private artifacts")
            return self.store.load_event(ref).to_dict()
        if domain == "checkpoint":
            if private:
                raise ValueError("checkpoints cannot be private artifacts")
            return self.store.load_checkpoint(ref)
        if domain == "context_revision":
            if private:
                raise ValueError("context revisions cannot be private artifacts")
            return self.store.load_context_revision(ref).to_wire()
        if domain not in {"payload", "message", "payload:bytes"}:
            raise ValueError(f"unsupported artifact domain: {domain!r}")
        return self.store.get_artifact(ref, expected_domain=domain, private=private)

    def bytes_artifact(self, ref: str) -> bytes:
        value = self.store.get_artifact(ref, expected_domain="payload:bytes")
        if not isinstance(value, bytes):
            raise TypeError("bytes artifact did not contain bytes")
        return value

    def checkpoint(self, ref: str) -> CheckpointV1:
        return self.store.load_checkpoint(ref)

    def context(self, ref: str) -> MaterializedContextV1:
        return self.store.materialize_context(ref)


class ViewCache:
    """Thread-safe LRU of immutable views for already-published checkpoints."""

    def __init__(self, capacity: int = 64) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("view cache capacity must be a positive integer")
        self.capacity = capacity
        self._views: OrderedDict[tuple[str, str], LineageView] = OrderedDict()
        self._lock = RLock()

    def get(self, root_id: str, checkpoint_id: str) -> LineageView | None:
        key = (root_id, checkpoint_id)
        with self._lock:
            view = self._views.get(key)
            if view is not None:
                self._views.move_to_end(key)
            return view

    def _insert(self, view: LineageView) -> None:
        """Insert a view after LineageGate has verified its publication."""
        key = (view.root_checkpoint_id, view.checkpoint_id)
        with self._lock:
            existing = self._views.get(key)
            if existing is not None and existing != view:
                raise ProjectionError(
                    "checkpoint.identity: published checkpoint produced conflicting views"
                )
            self._views[key] = view
            self._views.move_to_end(key)
            while len(self._views) > self.capacity:
                self._views.popitem(last=False)


class LineageGate:
    """Fold recorded input events through the single transition registry."""

    def __init__(self, *, cache: ViewCache | None = None) -> None:
        self.cache = cache or ViewCache()
        self._admitted: dict[tuple[str, str, str], AdmittedGraphV1] = {}
        self._admission_lock = RLock()
        self._pending: OrderedDict[str, LineageView] = OrderedDict()
        self._pending_lock = RLock()

    def view(self, store: Any, checkpoint_id: str, *, root: str | None = None) -> LineageView:
        """Verify a checkpoint from its admitted entry and return its immutable view."""
        target = store.load_checkpoint(checkpoint_id)
        chain = [(checkpoint_id, target)]
        while chain[-1][1].parents:
            parent_id = chain[-1][1].parents[0]
            chain.append((parent_id, store.load_checkpoint(parent_id)))
        chain.reverse()
        root_id, root_checkpoint = chain[0]
        if root is not None and root != root_id:
            raise ProjectionError("checkpoint.parents: checkpoint does not descend from root")

        cached_index = None
        view = None
        for index in range(len(chain) - 1, -1, -1):
            cached = self.cache.get(root_id, chain[index][0])
            if cached is not None:
                cached_index, view = index, cached
                break
        if view is None:
            reader = StoreArtifactReader(store)
            try:
                graph, _ = self._admitted_graph(
                    store,
                    root_checkpoint.state.instance_ref,
                    root_checkpoint.state.versions_ref,
                    reader=reader,
                )
                entry = derive_entry(
                    graph,
                    root_checkpoint.state.position["node_id"],
                    params_of(root_checkpoint.state, reader),
                    reader,
                )
            except (ProjectionError, CorruptRecordError):
                raise
            except Exception as exc:
                raise ProjectionError(
                    f"state.versions_ref: lineage root cannot be derived: {exc}"
                ) from exc
            if root_checkpoint.artifact_refs:
                raise ProjectionError(
                    "checkpoint.artifact_refs: lineage root has supplemental refs"
                )
            if entry.state.identity() != root_checkpoint.state.identity():
                path = first_difference(entry.state, root_checkpoint.state, path="state") or "state"
                raise ProjectionError(f"lineage root is not an admitted entry: {path}")
            view = entry.view
            cached_index = 0

        for index in range(cached_index + 1, len(chain)):
            _, checkpoint = chain[index]
            parent = chain[index - 1][1]
            view = self._fold_checkpoint(store, view, parent, checkpoint)
        return view

    def verify_commit(
        self,
        store: Any,
        base_checkpoint_id: str,
        events: Sequence[EventV1],
        next_state: Any,
    ) -> LineageView:
        """Verify one candidate event against a verified base; return its gate view."""
        if len(events) != 1:
            raise ProjectionError("commit.events: runtime commits must contain one event")
        base = store.load_checkpoint(base_checkpoint_id)
        before = self.view(store, base_checkpoint_id)
        candidate = self._step(store, before, base, events[0], next_state).view
        with self._pending_lock:
            existing = self._pending.get(candidate.checkpoint_id)
            if existing is not None and existing != candidate:
                raise ProjectionError(
                    "checkpoint.identity: candidate checkpoint produced conflicting views"
                )
            self._pending[candidate.checkpoint_id] = candidate
            self._pending.move_to_end(candidate.checkpoint_id)
            while len(self._pending) > self.cache.capacity:
                self._pending.popitem(last=False)
        return candidate

    def record_published(self, store: Any, commit_id: str) -> LineageView:
        """Cache the verified view of a commit that is the current published head."""
        try:
            commit = store.load_commit(commit_id)
            checkpoint = store.load_checkpoint(commit.checkpoint)
            if store.read_head(checkpoint.state.position["lineage_id"]) != commit_id:
                raise ProjectionError("commit.head: candidate is not the published lineage head")
        except CorruptRecordError:
            raise
        except StoreError as exc:
            raise ProjectionError(
                f"commit.id: published candidate cannot be loaded: {exc}"
            ) from exc
        with self._pending_lock:
            candidate = self._pending.pop(commit.checkpoint, None)
        if candidate is None:
            candidate = self.view(store, commit.checkpoint)
        self.cache._insert(candidate)
        return candidate

    def _admitted_graph(
        self,
        store: Any,
        instance_ref: str,
        versions_ref: str,
        *,
        reader: StoreArtifactReader | None = None,
    ) -> tuple[AdmittedGraphV1, ExecutionVersionsV1]:
        reader = reader or StoreArtifactReader(store)
        raw_versions = reader.artifact(versions_ref)
        try:
            versions = ExecutionVersionsV1.from_dict(raw_versions)
            if versions.transition_semantics != SEMANTICS_V1:
                raise ValueError("unsupported transition semantics")
            key = (str(store.root), instance_ref, versions.admission_policy_ref)
            with self._admission_lock:
                graph = self._admitted.get(key)
            if graph is None:
                instance = store.load_instance(instance_ref)
                wire = AdmissionPolicyV1.from_dict(reader.artifact(versions.admission_policy_ref))
                policy = wire.to_admission_policy(RuntimeAdmissionPolicy)
                graph = admit_graph(instance, StoreArtifactResolver(store), policy=policy)
                with self._admission_lock:
                    graph = self._admitted.setdefault(key, graph)
            return graph, versions
        except (AdmissionError, KeyError, TypeError, ValueError) as exc:
            raise ProjectionError(
                f"state.versions_ref: lineage admission or transition semantics is invalid: {exc}"
            ) from exc

    def _fold_checkpoint(
        self,
        store: Any,
        before: LineageView,
        parent: CheckpointV1,
        checkpoint: CheckpointV1,
    ) -> LineageView:
        if checkpoint.artifact_refs:
            raise ProjectionError(
                "checkpoint.artifact_refs: runtime checkpoints have no supplemental refs"
            )
        event = store.load_event(checkpoint.event_head)
        return self._step(store, before, parent, event, checkpoint.state).view

    def _step(
        self,
        store: Any,
        before: LineageView,
        parent: CheckpointV1,
        event: EventV1,
        state: Any,
    ) -> Transition:
        if event.previous != parent.event_head:
            raise ProjectionError("event.previous: candidate is not the parent's successor")
        if event.seq != parent.state.history["seq"] + 1:
            raise ProjectionError("event.seq: candidate is not the parent's successor")
        transition = self._derive(store, before, event, StoreArtifactReader(store))
        self._compare_event(transition.event, event)
        self._compare_state(transition.state, state)
        return transition

    def _derive(
        self, store: Any, view: LineageView, event: EventV1, reader: StoreArtifactReader
    ) -> Transition:
        try:
            payload = reader.artifact(event.payload_ref)
            if not isinstance(payload, Mapping):
                raise TypeError("event payload must be an object")
            record_type = payload.get("record_type")
            codec = RECORD_TYPES.get(record_type) if isinstance(record_type, str) else None
            if not isinstance(codec, type):
                raise ValueError("unknown input record type")
            input_record = codec.from_dict(dict(payload))
        except (ProjectionError, CorruptRecordError):
            raise
        except Exception as exc:
            raise ProjectionError(
                f"event.payload_ref: recorded input cannot be derived: {exc}"
            ) from exc
        return derive_input(view, input_record, reader)

    @staticmethod
    def _compare_event(expected: EventV1, actual: EventV1) -> None:
        if expected.id != actual.id:
            expected_body = expected.to_dict()
            actual_body = actual.to_dict()
            expected_body.pop("id", None)
            actual_body.pop("id", None)
            path = first_difference(expected_body, actual_body, path="event") or "event.id"
            raise ProjectionError(f"derived event differs at {path}")

    @staticmethod
    def _compare_state(expected: Any, actual: Any) -> None:
        if expected.identity() != actual.identity():
            path = first_difference(expected, actual, path="state") or "state"
            raise ProjectionError(f"derived state differs at {path}")


__all__ = ["DERIVE", "LineageGate", "StoreArtifactReader", "ViewCache", "derive_input"]
