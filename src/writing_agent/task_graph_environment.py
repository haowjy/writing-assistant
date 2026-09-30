"""Single-step environment boundary for typed task-graph transitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

from writing_agent.task_graph import (
    CheckpointV1,
    EnvironmentStateV1,
    MaterializedContextV1,
    MessageV1,
    Phase,
    freeze,
    thaw,
)
from writing_agent.task_graph_admission import (
    AdmissionPolicyV1 as AdmissionPolicy,
)
from writing_agent.task_graph_admission import (
    AdmittedGraphV1,
)
from writing_agent.task_graph_controller import Directive, next_step
from writing_agent.task_graph_derive_entry import EntryParamsV1, derive_entry
from writing_agent.task_graph_derive_writer import (
    group_member,
    tool_dispatch_error,
    writer_action_id,
)
from writing_agent.task_graph_errors import (
    AdapterContractError,
    AdapterContractProjectionError,
    ConcurrentUpdateError,
    CorruptRecordError,
    MissingReferenceError,
    ProjectionError,
    WriterRuntimeError,
)
from writing_agent.task_graph_gate import LineageGate, StoreArtifactReader, derive_input
from writing_agent.task_graph_native_contracts import NativeSamplingBudget
from writing_agent.task_graph_operation import operation_scoped
from writing_agent.task_graph_record_contracts import ExecutionVersionsV1
from writing_agent.task_graph_records import (
    AdmissionPolicyV1 as AdmissionPolicyRecord,
)
from writing_agent.task_graph_records import MemberStartV1, WriterTurnV2
from writing_agent.task_graph_sampling import NativeSamplingHistory
from writing_agent.task_graph_store import TaskGraphStore
from writing_agent.task_graph_transition import InputRecord, LineageView, ToolSpec, Transition


@dataclass(frozen=True)
class RuntimeHandle:
    """New-core handle: a checkpoint plus its verified flattened context."""

    checkpoint_id: str
    state: EnvironmentStateV1
    context: MaterializedContextV1


class SessionSeals(Protocol):
    sealed_adapter_ref: str | None

    def require_seal(
        self,
        adapter_ref: str,
        *,
        token_limited: bool = False,
        training_mode: str | None = None,
        policy=None,
        rendering=None,
    ) -> None: ...

    def require_member_seal(self, view: LineageView) -> None: ...


@dataclass(frozen=True)
class SamplerInput:
    messages: tuple[MessageV1, ...]
    tools: tuple[Mapping[str, Any], ...]
    rendering: Mapping[str, Any]
    context_content_hash: str
    context_revision_ref: str
    action_id: str
    writer_seed: int | None
    model_ref: str | None
    behavior_policy_ref: str | None
    tokenizer_ref: str | None
    template_ref: str | None
    decoding_ref: str | None
    native_sampling_budget: NativeSamplingBudget | None = None
    adapter_ref: str | None = None
    decision_ordinal: int | None = None
    native_history: NativeSamplingHistory | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(freeze(self.tools)))
        object.__setattr__(self, "rendering", freeze(self.rendering))
        if self.decision_ordinal is not None and (
            type(self.decision_ordinal) is not int or self.decision_ordinal < 0
        ):
            raise ValueError("sampling decision ordinal must be nonnegative")
        if self.native_history is not None and not isinstance(
            self.native_history, NativeSamplingHistory
        ):
            raise TypeError("sampling history must be committed native ledger evidence")


@dataclass(frozen=True)
class ToolInput:
    files: Mapping[str, str]
    queue_entry: Mapping[str, Any]
    tool_spec: ToolSpec
    dispatch_permitted: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", freeze(self.files))
        object.__setattr__(self, "queue_entry", freeze(self.queue_entry))


@dataclass(frozen=True)
class AuthorInput:
    request_ref: str
    request: Mapping[str, Any]
    script: Any
    decisions: Mapping[str, Any]
    disclosures: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("request", "decisions", "disclosures"):
            object.__setattr__(self, name, freeze(getattr(self, name)))


@dataclass(frozen=True)
class CheckInput:
    request_ref: str
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


@dataclass(frozen=True)
class _CheckpointHead:
    checkpoint: CheckpointV1
    commit_id: str | None
    is_published_head: bool


@dataclass(frozen=True)
class _CommitBase:
    view: LineageView
    head: _CheckpointHead


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
        self.commit_observer: Callable[[StepResult], None] | None = None

    @operation_scoped
    def enter(self, node_id: str, params: EntryParamsV1) -> RuntimeHandle:
        if not isinstance(params, EntryParamsV1):
            raise TypeError("entry parameters must be EntryParamsV1")
        self._require_admission_policy(params.versions_ref, path="params.versions_ref")
        if self.store.read_head(params.lineage_id) is not None:
            raise ConcurrentUpdateError("entry lineage already has a published head")
        entry = derive_entry(self.graph, node_id, params, self.reader)
        self.store.persist(self.graph.instance)
        for artifact in entry.artifacts:
            self.store.persist_artifact(artifact)
        self.store.save_checkpoint(entry.state)
        runtime = self._runtime(entry.view)
        self._check_session_seals(entry.view)
        return runtime

    @operation_scoped
    def open(self, checkpoint_id: str) -> RuntimeHandle:
        return self._open(checkpoint_id)

    @operation_scoped
    def open_head(self, lineage_id: str) -> RuntimeHandle:
        """Resume the verified lineage head through its published commit."""
        commit_id = self.store.read_head(lineage_id)
        if commit_id is None:
            raise ConcurrentUpdateError("lineage has no published head")
        commit = self.store.load_commit(commit_id)
        checkpoint = self.store.load_checkpoint(commit.checkpoint)
        if checkpoint.state.position["lineage_id"] != lineage_id:
            raise ProjectionError("published head belongs to another lineage")
        return self._open(commit.checkpoint)

    @operation_scoped
    def verify(self, runtime: RuntimeHandle) -> LineageView:
        return self._verify(runtime)

    @operation_scoped
    def step_input(self, runtime: RuntimeHandle) -> tuple[LineageView, Directive, PortInput | None]:
        """Verify once, then return the directive and its authorized port input."""
        view = self._verify(runtime)
        directive = next_step(view)
        requires_group_session = view.group is not None and (
            view.group.runner_mode == "real" or view.group.training_mode == "native"
        )
        if requires_group_session and self.session is None:
            raise AdapterContractError("runtime view requires a sealed runtime session")
        if (
            directive.kind == "sample_writer"
            and view.node.contract.budget_contract.usage_charged_limits()
        ):
            if self.session is None or self.session.sealed_adapter_ref is None:
                raise AdapterContractError(
                    "token-limited entries require a sealed usage-reporting session"
                )
        return view, directive, self._build_port_input(view, directive)

    def _build_port_input(self, view: LineageView, directive: Directive) -> PortInput | None:
        if directive.kind == "sample_writer":
            member = group_member(view)
            policy = {} if view.group is None else view.group.policy
            native_budget = None
            native_history = None
            native_mode = view.group is not None and view.group.training_mode == "native"
            if native_mode:
                limits = view.budget["limits"]
                consumed = view.budget["consumed"]
                generated_limit = limits.get("generated_tokens")
                native_budget = NativeSamplingBudget(
                    remaining_generated_tokens=(
                        None
                        if generated_limit is None
                        else max(0, generated_limit - consumed.get("generated_tokens", 0))
                    ),
                    max_context_tokens=limits.get("context_tokens"),
                )
                native_history = self._native_sampling_history(view)
            return SamplerInput(
                messages=view.context.messages,
                tools=view.context.tools,
                rendering=view.context.rendering,
                context_content_hash=view.context.content_ref,
                context_revision_ref=view.context.revision_ref,
                action_id=writer_action_id(view),
                writer_seed=None if member is None else member.writer_seed,
                model_ref=policy.get("model_ref"),
                behavior_policy_ref=policy.get("behavior_policy_ref"),
                tokenizer_ref=policy.get("tokenizer_ref"),
                template_ref=policy.get("template_ref"),
                decoding_ref=policy.get("decoding_ref"),
                native_sampling_budget=native_budget,
                adapter_ref=policy.get("adapter_ref") if native_mode else None,
                decision_ordinal=(view.state.history["action_count"] if native_mode else None),
                native_history=native_history,
            )
        if directive.kind == "execute_tool":
            cursor = directive.call_index
            queue = view.state.continuation["tool_queue"]
            if cursor is None or not 0 <= cursor < len(queue):
                raise ProjectionError("tool directive has no queued call")
            queue_entry = queue[cursor]
            dispatch_permitted = tool_dispatch_error(view.budget, queue_entry) is None
            return ToolInput(view.state.files, queue_entry, view.tool_spec, dispatch_permitted)
        if directive.kind == "await_author_reply":
            request_ref = view.state.continuation["author_request"]
            request = self.reader.artifact(request_ref, private=True)
            return AuthorInput(
                request_ref,
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
            return CheckInput(request_ref, request, view.state.files, packet)
        return None

    def _native_sampling_history(self, view: LineageView) -> NativeSamplingHistory | None:
        if not view.samples:
            return None
        previous = view.samples[-1]
        if _context_root_changed_after(self.reader, view.head_event_id, previous.event_id):
            return None
        try:
            turn = WriterTurnV2.from_dict(self.reader.artifact(previous.turn_ref))
            input_ids = _read_u32_tokens(
                self.reader.bytes_artifact(turn.input_token_ids_ref), turn.input_token_count
            )
            generated_ids = _read_u32_tokens(
                self.reader.bytes_artifact(turn.generated_token_ids_ref), turn.generated_token_count
            )
            return NativeSamplingHistory(turn, input_ids, generated_ids)
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterContractError(
                "native sampler cannot rebuild its committed token prefix"
            ) from exc

    @operation_scoped
    def commit(self, runtime: RuntimeHandle, input_record: InputRecord) -> StepResult:
        result = self._commit(runtime, input_record)
        if self.commit_observer is not None:
            self.commit_observer(result)
        return result

    @operation_scoped
    def start_member(self, entry_checkpoint_id: str, start: MemberStartV1) -> RuntimeHandle:
        if not isinstance(start, MemberStartV1):
            raise TypeError("member start must be MemberStartV1")
        runtime = self._open(entry_checkpoint_id)
        return self._commit(runtime, start).runtime

    def _open(self, checkpoint_id: str) -> RuntimeHandle:
        self._published_checkpoint(checkpoint_id)
        view = self.gate.view(self.store, checkpoint_id)
        runtime = self._runtime(view)
        self._check_session_seals(view)
        return runtime

    @staticmethod
    def _runtime(view: LineageView) -> RuntimeHandle:
        context = MaterializedContextV1(
            view.context.messages, view.context.tools, view.context.rendering
        )
        return RuntimeHandle(view.checkpoint_id, view.state, context)

    def _verify(self, runtime: RuntimeHandle) -> LineageView:
        if not isinstance(runtime, RuntimeHandle):
            raise TypeError("runtime must be a RolloutEnvironment RuntimeHandle")
        checkpoint = self._published_checkpoint(runtime.checkpoint_id)
        return self._verified_view(runtime, checkpoint)

    def _verify_commit_base(self, runtime: RuntimeHandle) -> _CommitBase:
        if not isinstance(runtime, RuntimeHandle):
            raise TypeError("runtime must be a RolloutEnvironment RuntimeHandle")
        head = self._inspect_checkpoint(runtime.checkpoint_id)
        return _CommitBase(
            self._verified_view(runtime, head.checkpoint, require_group_session=True), head
        )

    def _verified_view(
        self,
        runtime: RuntimeHandle,
        checkpoint: CheckpointV1,
        *,
        require_group_session: bool = False,
    ) -> LineageView:
        if checkpoint.state != runtime.state:
            raise WriterRuntimeError("runtime handle state differs from its checkpoint")
        view = self.gate.view(self.store, runtime.checkpoint_id)
        if runtime.context != MaterializedContextV1(
            view.context.messages, view.context.tools, view.context.rendering
        ):
            raise WriterRuntimeError("runtime handle context differs from its verified checkpoint")
        self._check_session_seals(view, require_group_session=require_group_session)
        return view

    def _published_checkpoint(self, checkpoint_id: str) -> CheckpointV1:
        checked = self._inspect_checkpoint(checkpoint_id)
        if not checked.is_published_head:
            raise ConcurrentUpdateError("checkpoint is not the current published lineage head")
        return checked.checkpoint

    def _inspect_checkpoint(self, checkpoint_id: str) -> _CheckpointHead:
        try:
            checkpoint = self.store.load_checkpoint(checkpoint_id)
        except MissingReferenceError as exc:
            raise ConcurrentUpdateError("checkpoint is not persisted") from exc
        if checkpoint.state.instance_ref != self.graph.instance.identity():
            raise WriterRuntimeError("checkpoint graph identity differs from the admitted graph")
        self._require_admission_policy(
            checkpoint.state.versions_ref,
            path="state.versions_ref",
            mismatch_error=WriterRuntimeError,
        )
        lineage = checkpoint.state.position["lineage_id"]
        head = self.store.read_head(lineage)
        if head is None:
            is_head = not checkpoint.parents
        else:
            is_head = self.store.load_commit(head).checkpoint == checkpoint_id
        return _CheckpointHead(checkpoint, head, is_head)

    def _check_session_seals(
        self, view: LineageView, *, require_group_session: bool = False
    ) -> None:
        budget = view.node.contract.budget_contract
        token_limited = bool(budget.usage_charged_limits())
        requires_group_session = view.group is not None and (
            view.group.runner_mode == "real" or view.group.training_mode == "native"
        )
        if self.session is None:
            if requires_group_session and require_group_session:
                raise AdapterContractError("runtime view requires a sealed runtime session")
            return

        sealed_ref = self.session.sealed_adapter_ref
        try:
            if view.group is not None:
                self.session.require_member_seal(view)
            elif sealed_ref is not None:
                self.session.require_seal(
                    sealed_ref,
                    token_limited=token_limited,
                    rendering=thaw(view.context.rendering),
                )
        except Exception as exc:
            raise AdapterContractError(
                "runtime session seal does not match the verified view"
            ) from exc

    def _require_admission_policy(
        self,
        versions_ref: str,
        *,
        path: str,
        mismatch_error: type[Exception] = ProjectionError,
    ) -> None:
        try:
            versions = ExecutionVersionsV1.from_dict(self.reader.artifact(versions_ref))
        except (CorruptRecordError, MissingReferenceError):
            raise
        except Exception as exc:
            raise ProjectionError(f"{path}: pinned execution versions are invalid: {exc}") from exc
        if versions.admission_policy_ref != self.admission_policy_ref:
            raise mismatch_error(f"{path}.admission_policy_ref: differs from environment policy")

    def _commit(self, runtime: RuntimeHandle, input_record: InputRecord) -> StepResult:
        base = self._verify_commit_base(runtime)
        view = base.view
        head, base_is_head = base.head.commit_id, base.head.is_published_head
        transition = self._derive(view, input_record)
        self._check_session_seals(transition.view, require_group_session=True)
        lineage = transition.state.position["lineage_id"]
        if lineage == view.state.position["lineage_id"] and not base_is_head:
            if head is not None and self._is_published_retry(
                head, runtime.checkpoint_id, transition
            ):
                published_view = self.gate.record_published(self.store, head)
                return self._step_result(published_view, head, transition.event.id)
            raise ConcurrentUpdateError("runtime checkpoint is no longer the lineage head")

        self._persist_transition(transition)
        if lineage != view.state.position["lineage_id"]:
            head = None
        commit_id = self.store.publish(
            lineage,
            head,
            (transition.event,),
            transition.state,
            parent_checkpoint=runtime.checkpoint_id if head is None else None,
        )
        published_view = self.gate.record_published(self.store, commit_id)
        return self._step_result(published_view, commit_id, transition.event.id)

    def _is_published_retry(
        self, commit_id: str, base_checkpoint_id: str, transition: Transition
    ) -> bool:
        if self.store.read_head(transition.state.position["lineage_id"]) != commit_id:
            raise ConcurrentUpdateError("lineage head changed while retrying a published step")
        commit = self.store.load_commit(commit_id)
        checkpoint = self.store.load_checkpoint(commit.checkpoint)
        return (
            checkpoint.identity() == transition.view.checkpoint_id
            and checkpoint.parents == (base_checkpoint_id,)
            and commit.events == (transition.event.id,)
        )

    @staticmethod
    def _step_result(view: LineageView, commit_id: str, event_id: str) -> StepResult:
        return StepResult(
            RolloutEnvironment._runtime(view),
            commit_id,
            event_id,
            Phase(view.state.position["phase"]),
            next_step(view),
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
        self.store.persist_artifact(input_artifact)
        for artifact in transition.artifacts:
            if artifact is not input_artifact and artifact.kind != "context_revision":
                self.store.persist_artifact(artifact)
        self.store.persist(transition.event)
        for artifact in transition.artifacts:
            if artifact.kind == "context_revision":
                self.store.persist_artifact(artifact)


def _context_root_changed_after(reader, head_event_id: str | None, sample_event_id: str) -> bool:
    current = head_event_id
    while current is not None and current != sample_event_id:
        event = reader.artifact(current, domain="event")
        if event.get("kind") == "context_changed":
            return True
        current = event.get("previous")
    if current != sample_event_id:
        raise AdapterContractError("native sampler history is outside the active event ancestry")
    return False


def _read_u32_tokens(data: bytes, count: int) -> tuple[int, ...]:
    if not isinstance(data, bytes) or len(data) != 4 * count:
        raise ValueError("native token bytes do not match their committed count")
    return tuple(
        int.from_bytes(data[offset : offset + 4], "little") for offset in range(0, len(data), 4)
    )


__all__ = [
    "AuthorInput",
    "CheckInput",
    "NativeSamplingBudget",
    "PortInput",
    "RolloutEnvironment",
    "RuntimeHandle",
    "SamplerInput",
    "StepResult",
    "ToolInput",
]
