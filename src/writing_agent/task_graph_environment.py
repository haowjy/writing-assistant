"""Environment-owned atomic event batches for admitted task-graph runtimes.

Producers choose domain records and intended state changes. This service owns the
recorded-effect envelope, runtime log, causal event sequence, visible context,
candidate semantic validation, CAS publication, and fresh restored result.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from writing_agent import task_graph_projection
from writing_agent.task_graph import ContextRevisionV1, EnvironmentStateV1, EventV1, MessageV1
from writing_agent.task_graph_accounting import charge_context_append
from writing_agent.task_graph_store import RuntimeHandle, TaskGraphStore


class WriterRuntimeError(ValueError):
    """The supplied action or restored state violates the graph runtime contract."""


@dataclass(frozen=True)
class WriterStepV1:
    runtime: RuntimeHandle
    commit_id: str
    event_id: str
    record_ref: str


class EnvironmentTransactionService:
    """One composition-owned transaction entry point; no role-specific policy."""

    def __init__(self, store: TaskGraphStore, rollout_id: str, entry_checkpoint_id: str):
        self.store = store
        self.rollout_id = rollout_id
        self.entry_checkpoint_id = entry_checkpoint_id

    def head(self, runtime: RuntimeHandle) -> tuple[str | None, str | None]:
        lineage = runtime.state.position["lineage_id"]
        head = self.store.read_head(lineage)
        if head is None:
            return None, runtime.checkpoint_id
        if self.store.load_commit(head).checkpoint != runtime.checkpoint_id:
            raise WriterRuntimeError("stale runtime handle: lineage head moved")
        return head, None

    def runtime_log(self, state: EnvironmentStateV1) -> list[dict[str, Any]]:
        body = self.store.get_artifact(state.external_inputs_ref, expected_domain="payload")
        if isinstance(body, dict) and body.get("record_type") == "WriterRuntimeLogV1":
            if body.get("rollout_id") != self.rollout_id:
                raise WriterRuntimeError("restored runtime belongs to another rollout")
            return list(body["entries"])
        if state.history["action_ids"] or state.history["tool_result_ids"]:
            raise WriterRuntimeError("missing writer runtime log")
        return []

    def batch(self, runtime: RuntimeHandle, *, restore_prefix: str) -> EnvironmentBatch:
        return EnvironmentBatch(self, runtime, restore_prefix)

    def publish_record(
        self,
        runtime: RuntimeHandle,
        kind: str,
        actor: str,
        *,
        changes: Mapping[str, Any],
        record: dict[str, Any] | None = None,
        record_ref: str | None = None,
        private: bool = False,
        audience: tuple[str, ...] = ("controller", "evaluator", "trainer"),
        extra_refs: tuple[str, ...] = (),
        restore_prefix: str = "environment",
        result_ref: str | None = None,
        result_effect: bool = False,
    ) -> WriterStepV1:
        batch = self.batch(runtime, restore_prefix=restore_prefix)
        event, recorded_ref = batch.append_record(
            kind,
            actor,
            record=record,
            record_ref=record_ref,
            changes=changes,
            private=private,
            audience=audience,
        )
        return batch.publish(
            event,
            event.payload_ref if result_effect else result_ref or recorded_ref,
            extra_refs=extra_refs,
        )


class EnvironmentBatch:
    """Stage typed records and effects, then publish exactly one atomic CAS."""

    def __init__(
        self, service: EnvironmentTransactionService, runtime: RuntimeHandle, restore_prefix: str
    ):
        self.service = service
        self.store = service.store
        self.runtime = runtime
        self.restore_prefix = restore_prefix
        self.head, self.parent = service.head(runtime)
        self.current = runtime.state
        self.events: list[EventV1] = []
        self.refs: list[str] = []

    def _effect(
        self,
        changes: Mapping[str, Any],
        history: Mapping[str, Any],
        delta: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "artifact_type": "Phase2RecordedEffectV1",
            "before_state_ref": self.current.identity(),
            "file_delta": dict(delta),
            "set": dict(changes),
            "history_set": dict(history),
        }

    def _append_effect(
        self,
        kind: str,
        actor: str,
        audience: tuple[str, ...],
        changes: Mapping[str, Any],
        history: Mapping[str, Any] = (),
        delta: Mapping[str, Any] = (),
    ) -> EventV1:
        state = self.current
        effect = self._effect(changes, history, delta)
        effect_ref = self.store.put_artifact(effect)
        event = EventV1(
            previous=state.history["head"],
            seq=state.history["seq"] + 1,
            lineage_id=state.position["lineage_id"],
            rollout_id=self.service.rollout_id,
            node_visit_id=state.position["visit_id"],
            kind=kind,
            actor=actor,
            audience=audience,
            payload_ref=effect_ref,
            versions_ref=state.versions_ref,
            provenance_ref=state.provenance_ref,
        )
        self.current = self.store._apply_recorded_effect_body(state, event, effect)
        self.store.persist(event)
        self.events.append(event)
        self.refs.append(effect_ref)
        return event

    def append_record(
        self,
        kind: str,
        actor: str,
        *,
        changes: Mapping[str, Any],
        record: dict[str, Any] | None = None,
        record_ref: str | None = None,
        message: MessageV1 | None = None,
        message_ref: str | None = None,
        history: Mapping[str, Any] = (),
        delta: Mapping[str, Any] = (),
        private: bool = False,
        audience: tuple[str, ...] | None = None,
    ) -> tuple[EventV1, str]:
        if (record is None) == (record_ref is None):
            raise ValueError("supply exactly one record or record reference")
        if message is not None and message_ref is not None:
            raise ValueError("supply a message or its reference, not both")
        if record_ref is None:
            record_ref = self.store.put_artifact(record, private=private)
        if message is not None:
            message_ref = self.store.persist(message)
        entries = self.service.runtime_log(self.current)
        entry = {"seq": self.current.history["seq"] + 1, "kind": kind, "record_ref": record_ref}
        if message_ref is not None:
            entry["message_ref"] = message_ref
        entries.append(entry)
        log_ref = self.store.put_artifact(
            {
                "record_type": "WriterRuntimeLogV1",
                "rollout_id": self.service.rollout_id,
                "entries": entries,
            }
        )
        if audience is None:
            audience = (
                ("controller", "trainer", "writer")
                if message_ref is not None
                else ("controller", "evaluator", "trainer")
            )
        event = self._append_effect(
            kind, actor, audience, {**changes, "external_inputs_ref": log_ref}, history, delta
        )
        self.refs.extend((record_ref, log_ref))
        if message_ref is not None:
            self.refs.append(message_ref)
        return event, record_ref

    def append_visible(self, messages: tuple[MessageV1, ...], source: EventV1) -> EventV1:
        """Append one or more messages after their source events in the same batch."""
        context = ContextRevisionV1(
            messages=(*self.runtime.context.messages, *messages),
            tools=self.runtime.context.tools,
            event_head=source.id,
            provenance_refs=(source.id,),
            rendering=self.runtime.context.rendering,
        )
        self.store.persist(context)
        changes = {"context_ref": context.identity()}
        charged = charge_context_append(
            self.store.get_artifact(self.current.budgets_ref, expected_domain="payload"), context
        )
        if charged is not None:
            changes["budgets_ref"] = self.store.put_artifact(charged)
            self.refs.append(changes["budgets_ref"])
        self.refs.append(context.identity())
        return self._append_effect(
            "context_changed", "environment", ("controller", "trainer"), changes
        )

    def publish(
        self, primary: EventV1, record_ref: str, *, extra_refs: tuple[str, ...] = ()
    ) -> WriterStepV1:
        if not self.events or primary not in self.events:
            raise ValueError("primary event must be part of the staged batch")
        task_graph_projection.project_writer_context(
            self.store,
            self.service.entry_checkpoint_id,
            self.runtime.checkpoint_id,
            candidate_events=tuple(self.events),
            candidate_state=self.current,
        )
        commit = self.store.publish(
            self.runtime.state.position["lineage_id"],
            self.head,
            tuple(self.events),
            self.current,
            parent_checkpoint=self.parent,
            artifact_refs=tuple(dict.fromkeys((*self.refs, *extra_refs))),
        )
        checkpoint_id = self.store.load_commit(commit).checkpoint
        fresh = self.runtime.workspace.parent / f"{self.restore_prefix}-{uuid.uuid4().hex}"
        return WriterStepV1(
            self.store.restore(checkpoint_id, fresh), commit, primary.id, record_ref
        )
