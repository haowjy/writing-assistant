"""Single-step environment boundary for typed task-graph transitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias

from writing_agent.task_graph import (
    EnvironmentStateV1,
    MessageV1,
    Phase,
    Record,
    freeze,
    load_canonical_json,
)
from writing_agent.task_graph_admission import (
    AdmissionPolicyV1 as AdmissionPolicy,
)
from writing_agent.task_graph_admission import (
    AdmittedGraphV1,
)
from writing_agent.task_graph_controller import Directive, next_step
from writing_agent.task_graph_derive_entry import EntryParamsV1, derive_entry
from writing_agent.task_graph_errors import (
    AdapterContractError,
    AdapterContractProjectionError,
    ConcurrentUpdateError,
    MissingReferenceError,
    ProjectionError,
    WriterRuntimeError,
)
from writing_agent.task_graph_gate import LineageGate, StoreArtifactReader, derive_input
from writing_agent.task_graph_operation import operation_scoped
from writing_agent.task_graph_record_contracts import ExecutionVersionsV1
from writing_agent.task_graph_records import (
    AdmissionPolicyV1 as AdmissionPolicyRecord,
)
from writing_agent.task_graph_records import (
    MaterializedContextV1,
    MemberStartV1,
)
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_transition import InputRecord, LineageView, ToolSpec, Transition
from writing_agent.task_graph_wire import WireRecord


@dataclass(frozen=True)
class RuntimeHandle:
    """New-core handle: a checkpoint plus its flattened, verified context."""

    checkpoint_id: str
    state: EnvironmentStateV1
    context: MaterializedContextV1
    workspace: Path


class SessionSeals(Protocol):
    sealed_adapter_ref: str | None

    def require_seal(self, adapter_ref: str) -> None: ...

    def require_member_seal(self, store: TaskGraphStore, runtime: RuntimeHandle) -> None: ...


@dataclass(frozen=True)
class SamplerInput:
    messages: tuple[MessageV1, ...]
    tools: tuple[Mapping[str, Any], ...]
    rendering: Mapping[str, Any]
    context_revision_ref: str
    action_id: str
    writer_seed: int | None
    model_ref: str | None
    behavior_policy_ref: str | None
    tokenizer_ref: str | None
    template_ref: str | None
    decoding_ref: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(freeze(self.tools)))
        object.__setattr__(self, "rendering", freeze(self.rendering))


@dataclass(frozen=True)
class ToolInput:
    files: Mapping[str, str]
    queue_entry: Mapping[str, Any]
    tool_spec: ToolSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", freeze(self.files))
        object.__setattr__(self, "queue_entry", freeze(self.queue_entry))


@dataclass(frozen=True)
class AuthorInput:
    request: Mapping[str, Any]
    script: Any
    decisions: Mapping[str, Any]
    disclosures: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("request", "decisions", "disclosures"):
            object.__setattr__(self, name, freeze(getattr(self, name)))


@dataclass(frozen=True)
class CheckInput:
    request: Mapping[str, Any]
    files: Mapping[str, str]
    evaluator_packet: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request", freeze(self.request))
        object.__setattr__(self, "files", freeze(self.files))
        if self.evaluator_packet is not None:
            object.__setattr__(self, "evaluator_packet", freeze(self.evaluator_packet))


PortInput: TypeAlias = SamplerInput | ToolInput | AuthorInput | CheckInput


@dataclass(frozen=True)
class StepResult:
    runtime: RuntimeHandle
    commit_id: str
    event_id: str
    phase: Phase
    directive: Directive


class RolloutEnvironment:
    """Own entry, verification, producer derivation, persistence, and publication."""

    def __init__(
        self,
        store: TaskGraphStore,
        graph: AdmittedGraphV1,
        session: SessionSeals | None,
        gate: LineageGate,
        admission_policy: AdmissionPolicy,
    ):
        if store.verifier is not gate:
            raise ValueError("rollout store must publish through its lineage gate")
        if graph.policy != admission_policy:
            raise ValueError("rollout graph must be admitted under its pinned admission policy")
        self.store = store
        self.graph = graph
        self.session = session
        self.gate = gate
        self.admission_policy = admission_policy
        self.admission_policy_ref = AdmissionPolicyRecord.from_admission_policy(
            admission_policy
        ).identity()
        self.reader = StoreArtifactReader(store)

    @operation_scoped
    def enter(self, node_id: str, params: EntryParamsV1, workspace_root: Path) -> RuntimeHandle:
        if not isinstance(params, EntryParamsV1):
            raise TypeError("entry parameters must be EntryParamsV1")
        self._require_admission_policy(params.versions_ref, path="params.versions_ref")
        if self.store.read_head(params.lineage_id) is not None:
            raise ConcurrentUpdateError("entry lineage already has a published head")
        entry = derive_entry(self.graph, node_id, params, self.reader)
        self.store.persist(self.graph.instance)
        for artifact in entry.artifacts:
            self._persist_artifact(artifact)
        self.store.save_checkpoint(entry.state)
        runtime = self._materialize(entry.view, workspace_root)
        self._check_session_seals(runtime)
        return runtime

    @operation_scoped
    def open(self, checkpoint_id: str, workspace_root: Path) -> RuntimeHandle:
        return self._open(checkpoint_id, workspace_root)

    @operation_scoped
    def verify(self, runtime: RuntimeHandle) -> LineageView:
        return self._verify(runtime)

    @operation_scoped
    def port_input(self, view: LineageView, directive: Directive) -> PortInput:
        self._require_published_view(view)
        self._check_session_seals(view)
        if next_step(view) != directive:
            raise ProjectionError("port directive differs from the verified view")
        if directive.kind == "sample_writer":
            member = None
            if view.group is not None:
                lineage = view.state.position["lineage_id"]
                member = next(
                    (item for item in view.group.members if item.member_id == lineage), None
                )
                if member is None:
                    raise ProjectionError("group view has no sealed member sampling pins")
            policy = {} if view.group is None else view.group.policy
            return SamplerInput(
                messages=view.context.messages,
                tools=view.context.tools,
                rendering=view.context.rendering,
                context_revision_ref=view.context.revision_ref,
                action_id=(
                    f"{view.state.position['lineage_id']}:action:"
                    f"{len(view.state.history['action_ids'])}"
                ),
                writer_seed=None if member is None else member.writer_seed,
                model_ref=policy.get("model_ref"),
                behavior_policy_ref=policy.get("behavior_policy_ref"),
                tokenizer_ref=policy.get("tokenizer_ref"),
                template_ref=policy.get("template_ref"),
                decoding_ref=policy.get("decoding_ref"),
            )
        if directive.kind == "execute_tool":
            cursor = directive.call_index
            queue = view.state.continuation["tool_queue"]
            if cursor is None or not 0 <= cursor < len(queue):
                raise ProjectionError("tool directive has no queued call")
            return ToolInput(view.state.files, queue[cursor], view.tool_spec)
        if directive.kind == "await_author_reply":
            request_ref = view.state.continuation["author_request"]
            request = self.reader.artifact(request_ref, private=True)
            return AuthorInput(
                request,
                view.node.script,
                self.reader.artifact(view.state.decisions_ref),
                self.reader.artifact(view.state.disclosures_ref),
            )
        if directive.kind == "await_check_result":
            request_ref = view.state.continuation["check_requests"][0]
            request = self.reader.artifact(request_ref, private=True)
            packet_ref = request["evaluator_packet_ref"]
            packet = None if packet_ref is None else self.reader.artifact(packet_ref, private=True)
            return CheckInput(request, view.state.files, packet)
        raise ValueError(f"directive {directive.kind!r} has no external port")

    @operation_scoped
    def commit(self, runtime: RuntimeHandle, input_record: InputRecord) -> StepResult:
        return self._commit(runtime, input_record)

    @operation_scoped
    def start_member(
        self, entry_checkpoint_id: str, start: MemberStartV1, workspace_root: Path
    ) -> RuntimeHandle:
        if not isinstance(start, MemberStartV1):
            raise TypeError("member start must be MemberStartV1")
        runtime = self._open(entry_checkpoint_id, workspace_root)
        return self._commit(runtime, start).runtime

    def _open(self, checkpoint_id: str, workspace_root: Path) -> RuntimeHandle:
        checkpoint = self._current_checkpoint(checkpoint_id)
        self._require_admission_policy(checkpoint.state.versions_ref, path="state.versions_ref")
        if checkpoint.state.instance_ref != self.graph.instance.identity():
            raise WriterRuntimeError("checkpoint graph identity differs from the admitted graph")
        view = self.gate.view(self.store, checkpoint_id)
        runtime = self._materialize(view, workspace_root)
        self._check_session_seals(runtime)
        return runtime

    def _materialize(self, view: LineageView, workspace_root: Path) -> RuntimeHandle:
        workspace = self.store.materialize(view.checkpoint_id, workspace_root)
        context = self.store.materialize_context(view.state.context_ref)
        return RuntimeHandle(view.checkpoint_id, view.state, context, workspace)

    def _verify(self, runtime: RuntimeHandle) -> LineageView:
        if not isinstance(runtime, RuntimeHandle):
            raise TypeError("runtime must be a RolloutEnvironment RuntimeHandle")
        self._check_session_seals(runtime)
        checkpoint = self._current_checkpoint(runtime.checkpoint_id)
        self._require_admission_policy(checkpoint.state.versions_ref, path="state.versions_ref")
        if checkpoint.state != runtime.state:
            raise WriterRuntimeError("runtime handle state differs from its checkpoint")
        context = self.store.materialize_context(runtime.state.context_ref)
        if context != runtime.context:
            raise WriterRuntimeError("runtime handle context differs from its checkpoint")
        if runtime.state.instance_ref != self.graph.instance.identity():
            raise WriterRuntimeError("runtime graph identity differs from the admitted graph")
        return self.gate.view(self.store, runtime.checkpoint_id)

    def _current_checkpoint(self, checkpoint_id: str):
        checkpoint = self.store.load_checkpoint(checkpoint_id)
        lineage = checkpoint.state.position["lineage_id"]
        head = self.store.read_head(lineage)
        if head is None:
            if checkpoint.parents:
                raise WriterRuntimeError("unpublished checkpoint is not a lineage entry")
        elif self.store.load_commit(head).checkpoint != checkpoint_id:
            raise WriterRuntimeError("runtime checkpoint is not the published lineage head")
        return checkpoint

    def _require_published_view(self, view: LineageView) -> None:
        try:
            checkpoint = self._current_checkpoint(view.checkpoint_id)
        except MissingReferenceError as exc:
            raise ProjectionError("candidate view has not been published") from exc
        if checkpoint.state.position["lineage_id"] != view.state.position["lineage_id"]:
            raise ProjectionError("view lineage differs from its published checkpoint")
        if self.gate.view(self.store, view.checkpoint_id) != view:
            raise ProjectionError("port view differs from the verified published checkpoint")

    def _check_session_seals(self, runtime: RuntimeHandle) -> None:
        if self.session is None:
            return
        sealed_ref = self.session.sealed_adapter_ref
        if sealed_ref is not None:
            self.session.require_seal(sealed_ref)
        self.session.require_member_seal(self.store, runtime)

    def _require_admission_policy(self, versions_ref: str, *, path: str) -> None:
        try:
            versions = ExecutionVersionsV1.from_dict(self.reader.artifact(versions_ref))
        except Exception as exc:
            raise ProjectionError(f"{path}: pinned execution versions are invalid: {exc}") from exc
        if versions.admission_policy_ref != self.admission_policy_ref:
            raise ProjectionError(f"{path}.admission_policy_ref: differs from environment policy")

    def _commit(self, runtime: RuntimeHandle, input_record: InputRecord) -> StepResult:
        view = self._verify(runtime)
        head = self.store.read_head(view.state.position["lineage_id"])
        if head is not None and self.store.load_commit(head).checkpoint != runtime.checkpoint_id:
            raise ConcurrentUpdateError("runtime checkpoint is no longer the lineage head")
        transition = self._derive(view, input_record)
        self._persist_transition(transition)
        lineage = transition.state.position["lineage_id"]
        if lineage != view.state.position["lineage_id"]:
            head = None
        commit_id = self.store.publish(
            lineage,
            head,
            (transition.event,),
            transition.state,
            parent_checkpoint=runtime.checkpoint_id if head is None else None,
        )
        published_view = self.gate.record_published(self.store, commit_id, transition.view)
        context = MaterializedContextV1(
            published_view.context.messages,
            published_view.context.tools,
            published_view.context.rendering,
        )
        next_runtime = RuntimeHandle(
            published_view.checkpoint_id, transition.state, context, runtime.workspace
        )
        return StepResult(
            next_runtime,
            commit_id,
            transition.event.id,
            Phase(transition.state.position["phase"]),
            next_step(published_view),
        )

    def _derive(self, view: LineageView, input_record: InputRecord) -> Transition:
        try:
            return derive_input(view, input_record, self.reader)
        except AdapterContractProjectionError as exc:
            raise AdapterContractError("port input violates its adapter contract") from exc

    def _persist_transition(self, transition: Transition) -> None:
        input_artifact = next(
            (item for item in transition.artifacts if item.ref == transition.event.payload_ref),
            None,
        )
        if input_artifact is None:
            raise ProjectionError("transition does not return its input payload artifact")
        self._persist_artifact(input_artifact)
        for artifact in transition.artifacts:
            if artifact is not input_artifact and artifact.kind != "context_revision":
                self._persist_artifact(artifact)
        self.store.persist(transition.event)
        for artifact in transition.artifacts:
            if artifact.kind == "context_revision":
                self._persist_artifact(artifact)

    def _persist_artifact(self, artifact) -> None:
        value = artifact.value
        if artifact.kind in {"context_node", "context_revision"}:
            identity = self.store.persist(value)
        elif artifact.value_kind == "bytes":
            identity = self.store.put_bytes_artifact(value, private=artifact.kind == "private")
        else:
            body = (
                value.to_wire()
                if isinstance(value, WireRecord)
                else value.to_dict()
                if isinstance(value, Record)
                else load_canonical_json(value)
            )
            identity = self.store.put_artifact(body, private=artifact.kind == "private")
        if identity != artifact.ref:
            raise ProjectionError("persisted artifact identity differs from its derive")


__all__ = [
    "AuthorInput",
    "CheckInput",
    "PortInput",
    "RolloutEnvironment",
    "RuntimeHandle",
    "SamplerInput",
    "StepResult",
    "ToolInput",
]
