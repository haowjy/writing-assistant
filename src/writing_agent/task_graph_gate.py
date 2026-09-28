"""Semantic verification for typed task-graph lineages."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from threading import RLock
from typing import Any

from writing_agent.task_graph import CheckpointV1, EventV1
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
from writing_agent.task_graph_errors import ProjectionError
from writing_agent.task_graph_record_contracts import (
    SEMANTICS_V1,
    ExecutionVersionsV1,
)
from writing_agent.task_graph_records import (
    RECORD_TYPES,
    AdmissionPolicyV1,
    ContextRevisionV1,
    MaterializedContextV1,
    OutcomeV1,
)
from writing_agent.task_graph_transition import (
    ArtifactReader,
    CheckpointChain,
    ContextView,
    LineageMode,
    LineageView,
    ToolSpec,
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

    def insert(self, view: LineageView) -> None:
        """Insert only after the caller has confirmed checkpoint publication."""
        key = (view.root_checkpoint_id, view.checkpoint_id)
        with self._lock:
            existing = self._views.get(key)
            if existing is not None and existing != view:
                raise ProjectionError("published checkpoint produced conflicting cached views")
            self._views[key] = view
            self._views.move_to_end(key)
            while len(self._views) > self.capacity:
                self._views.popitem(last=False)


class LineageGate:
    """Fold recorded input events through the single transition registry."""

    def __init__(self, *, cache: ViewCache | None = None) -> None:
        self.cache = cache or ViewCache()
        self._admitted: dict[tuple[str, str], AdmittedGraphV1] = {}
        self._admission_lock = RLock()

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
            raise ProjectionError("checkpoint does not descend from the requested root")

        cached_index = None
        view = None
        for index in range(len(chain) - 1, -1, -1):
            cached = self.cache.get(root_id, chain[index][0])
            if cached is not None:
                cached_index, view = index, cached
                break
        if view is None:
            view = self._entry_view(store, root_id, root_checkpoint)
            cached_index = 0

        for checkpoint_id, checkpoint in chain[cached_index + 1 :]:
            view = self._fold_checkpoint(store, view, checkpoint_id, checkpoint)
        return view

    def verify_commit(
        self,
        store: Any,
        base_checkpoint_id: str,
        events: Sequence[EventV1],
        next_state: Any,
    ) -> None:
        """Verify one candidate event against a verified base; never cache it."""
        if len(events) != 1:
            raise ProjectionError("runtime commits must contain exactly one event")
        base = store.load_checkpoint(base_checkpoint_id)
        before = self.view(store, base_checkpoint_id)
        event = events[0]
        if event.previous != base.event_head or event.seq != base.state.history["seq"] + 1:
            raise ProjectionError("candidate event is not the next event after its base")
        transition = self._derive(store, before, event)
        self._compare_event(transition.event, event)
        self._compare_state(transition.state, next_state)

    def verify_checkpoint(self, store: Any, checkpoint_id: str) -> None:
        """Verify the checkpoint's full lineage before the store materializes it."""
        self.view(store, checkpoint_id)

    def _entry_view(self, store: Any, root_id: str, root: CheckpointV1) -> LineageView:
        reader = StoreArtifactReader(store)
        state = root.state
        if root.artifact_refs:
            raise ProjectionError("lineage root differs at checkpoint.artifact_refs")
        try:
            graph, versions = self._admitted_graph(store, state.instance_ref, state.versions_ref)
            entry = derive_entry(
                graph,
                state.position["node_id"],
                params_of(state, reader),
                reader,
            )
        except ProjectionError:
            raise
        except (AdmissionError, KeyError, TypeError, ValueError) as exc:
            raise ProjectionError("lineage root is not an admitted entry") from exc
        if entry.state.identity() != state.identity():
            path = first_difference(entry.state, state, path="state") or "state"
            raise ProjectionError(f"lineage root is not an admitted entry: {path}")

        node = graph.node(state.position["node_id"])
        materialized = reader.context(state.context_ref)
        revision = ContextRevisionV1.from_dict(
            reader.artifact(state.context_ref, domain="context_revision")
        )
        context = ContextView(
            messages=materialized.messages,
            sources=tuple(None for _ in materialized.messages),
            tools=materialized.tools,
            rendering=materialized.rendering,
            content_ref=revision.content_ref,
            revision_ref=state.context_ref,
        )
        return LineageView(
            root_checkpoint_id=root_id,
            checkpoint_id=root_id,
            head_event_id=None,
            state=state,
            budget=reader.artifact(state.budgets_ref),
            outcome=OutcomeV1.from_dict(reader.artifact(state.outcome_ref)),
            check_statuses={},
            context=context,
            raw_call_ids=frozenset(),
            call_sources={},
            samples=(),
            ancestry=CheckpointChain(root_id, context),
            node=node,
            mode=LineageMode.for_node(node),
            tool_spec=ToolSpec(**dict(versions.tool_spec)),
        )

    def _admitted_graph(
        self, store: Any, instance_ref: str, versions_ref: str
    ) -> tuple[AdmittedGraphV1, ExecutionVersionsV1]:
        reader = StoreArtifactReader(store)
        raw_versions = reader.artifact(versions_ref)
        try:
            versions = ExecutionVersionsV1.from_dict(raw_versions)
            if versions.transition_semantics != SEMANTICS_V1:
                raise ValueError("unsupported transition semantics")
            key = (instance_ref, versions.admission_policy_ref)
            with self._admission_lock:
                graph = self._admitted.get(key)
            if graph is None:
                instance = store.load_instance(instance_ref)
                wire = AdmissionPolicyV1.from_dict(reader.artifact(versions.admission_policy_ref))
                policy = RuntimeAdmissionPolicy(
                    writer_family=wire.writer_family,
                    allowed_tools=frozenset(wire.allowed_tools),
                    controller_versions=frozenset(wire.controller_versions),
                    check_versions=frozenset(wire.check_versions),
                )
                graph = admit_graph(instance, StoreArtifactResolver(store), policy=policy)
                with self._admission_lock:
                    graph = self._admitted.setdefault(key, graph)
            return graph, versions
        except (AdmissionError, KeyError, TypeError, ValueError) as exc:
            raise ProjectionError("lineage admission or transition semantics is invalid") from exc

    def _fold_checkpoint(
        self,
        store: Any,
        before: LineageView,
        checkpoint_id: str,
        checkpoint: CheckpointV1,
    ) -> LineageView:
        parent_id = before.checkpoint_id
        parent = store.load_checkpoint(parent_id)
        if checkpoint.parents != (parent_id,):
            raise ProjectionError("checkpoint ancestry is not contiguous")
        if checkpoint.state.history["seq"] != parent.state.history["seq"] + 1:
            raise ProjectionError("runtime checkpoint must advance exactly one event")
        event = store.load_event(checkpoint.event_head)
        if event.previous != parent.event_head or event.seq != parent.state.history["seq"] + 1:
            raise ProjectionError("checkpoint event is not the next event after its parent")
        transition = self._derive(store, before, event)
        self._compare_event(transition.event, event)
        self._compare_state(transition.state, checkpoint.state)
        if transition.view.checkpoint_id != checkpoint_id:
            path = (
                first_difference(
                    CheckpointV1(
                        parents=(parent_id,), state=transition.state, event_head=transition.event.id
                    ),
                    checkpoint,
                    path="checkpoint",
                )
                or "checkpoint"
            )
            raise ProjectionError(f"derived checkpoint differs at {path}")
        return transition.view

    def _derive(self, store: Any, view: LineageView, event: EventV1) -> Transition:
        reader = StoreArtifactReader(store)
        try:
            payload = reader.artifact(event.payload_ref)
            if not isinstance(payload, Mapping):
                raise TypeError("event payload must be an object")
            record_type = payload.get("record_type")
            codec = RECORD_TYPES.get(record_type) if isinstance(record_type, str) else None
            if not isinstance(codec, type):
                raise ValueError("unknown input record type")
            input_record = codec.from_dict(dict(payload))
            key: DeriveKey = record_type
            if record_type == "EnvironmentStepV1":
                key = (record_type, input_record.directive["kind"])
            derive = DERIVE.get(key)
            if derive is None:
                raise ValueError("input record has no derive route")
            return derive(view, input_record, reader)
        except ProjectionError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectionError("recorded input cannot be derived") from exc

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


__all__ = ["DERIVE", "LineageGate", "StoreArtifactReader", "ViewCache"]
