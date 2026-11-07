"""Immutable values shared by the transition-seam derives and gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal, Protocol, TypeAlias

from writing_agent.task_graph import (
    CheckpointV1,
    EnvironmentStateV1,
    EventV1,
    MaterializedContextV1,
    MessageV1,
    Phase,
    Record,
    canonical_bytes,
    domain_hash,
    domain_hash_bytes,
    freeze,
    load_canonical_json,
    validate_hash,
)
from writing_agent.task_graph_admission import AdmittedNodeV1
from writing_agent.task_graph_contracts import RewardContractV1
from writing_agent.task_graph_record_contracts import GroupSpecV1
from writing_agent.task_graph_records import (
    AuthorReplyV1,
    ContextOperationInputV1,
    EnvironmentStepV1,
    EvaluatorResultV1,
    MemberStartV1,
    OutcomeV1,
    ToolObservationV1,
    WriterTurnV1,
    WriterTurnV2,
)
from writing_agent.task_graph_wire import WireRecord

Hash: TypeAlias = str


@dataclass(frozen=True)
class LineageMode:
    interaction: Literal["none", "scripted_author"]
    ask_semantics: bool
    feedback_rules: tuple[Mapping[str, Any], ...]
    evaluation: bool
    reward: RewardContractV1 | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "feedback_rules", tuple(freeze(rule) for rule in self.feedback_rules)
        )
        if self.interaction not in {"none", "scripted_author"}:
            raise ValueError("unsupported lineage interaction mode")
        if type(self.ask_semantics) is not bool or type(self.evaluation) is not bool:
            raise TypeError("lineage capabilities must be booleans")

    @staticmethod
    def for_node(node: AdmittedNodeV1) -> LineageMode:
        """Resolve the runtime capabilities admitted by one node."""
        interaction = node.contract.interaction_contract
        script = node.script
        feedback = () if script is None else script.feedback
        mode = "scripted_author" if interaction.mode == "scripted_author" else "none"
        evaluation = node.evaluator_packet is not None and bool(node.checks)
        return LineageMode(
            interaction=mode,
            ask_semantics=interaction.mode == "scripted_author",
            feedback_rules=feedback,
            evaluation=evaluation,
            reward=node.reward_contract if evaluation else None,
        )


@dataclass(frozen=True)
class ContextView:
    messages: tuple[MessageV1, ...]
    sources: tuple[Hash | None, ...]
    tools: tuple[Mapping[str, Any], ...]
    rendering: Mapping[str, Any]
    content_ref: Hash
    revision_ref: Hash

    def __post_init__(self) -> None:
        for source in self.sources:
            validate_hash(source, optional=True)
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "messages", tuple(freeze(message) for message in self.messages))
        object.__setattr__(self, "tools", tuple(freeze(tool) for tool in self.tools))
        object.__setattr__(self, "rendering", freeze(self.rendering))
        if len(self.messages) != len(self.sources):
            raise ValueError("context messages and sources must align")
        for identity in (self.content_ref, self.revision_ref):
            validate_hash(identity)


@dataclass(frozen=True)
class CheckpointChain:
    """Persistent ancestry index; it retains only verified checkpoint contexts."""

    checkpoint_id: Hash
    context: ContextView
    parent: CheckpointChain | None = None

    def __post_init__(self) -> None:
        validate_hash(self.checkpoint_id)

    def context_at(self, checkpoint_id: Hash) -> ContextView:
        current: CheckpointChain | None = self
        while current is not None:
            if current.checkpoint_id == checkpoint_id:
                return current.context
            current = current.parent
        raise KeyError(checkpoint_id)


@dataclass(frozen=True)
class CallSource:
    action_id: str
    queue_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError("action_id must be nonempty")
        if type(self.queue_index) is not int or self.queue_index < 0:
            raise ValueError("queue_index must be nonnegative")


@dataclass(frozen=True)
class SampleRef:
    action_id: str
    event_id: Hash
    turn_ref: Hash
    outcome: Literal["action", "budget_stop"]

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError("action_id must be nonempty")
        validate_hash(self.event_id)
        validate_hash(self.turn_ref)
        if self.outcome not in {"action", "budget_stop"}:
            raise ValueError("unsupported sample outcome")


@dataclass(frozen=True)
class ToolSpec:
    max_file_bytes: int
    max_workspace_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (self.max_file_bytes, self.max_workspace_bytes)
        ):
            raise ValueError("tool limits must be positive integers")


@dataclass(frozen=True)
class LineageView:
    root_checkpoint_id: Hash
    checkpoint_id: Hash
    head_event_id: Hash | None
    state: EnvironmentStateV1
    budget: Mapping[str, Any]
    outcome: OutcomeV1
    check_statuses: Mapping[str, str | None]
    context: ContextView
    raw_call_ids: frozenset[str]
    call_sources: Mapping[str, CallSource]
    samples: tuple[SampleRef, ...]
    ancestry: CheckpointChain
    node: AdmittedNodeV1
    mode: LineageMode
    tool_spec: ToolSpec
    group: GroupSpecV1 | None = None

    def __post_init__(self) -> None:
        for identity in (self.root_checkpoint_id, self.checkpoint_id):
            validate_hash(identity)
        validate_hash(self.head_event_id, optional=True)
        object.__setattr__(self, "state", freeze(self.state))
        object.__setattr__(self, "budget", freeze(self.budget))
        object.__setattr__(self, "outcome", freeze(self.outcome))
        object.__setattr__(self, "check_statuses", freeze(self.check_statuses))
        object.__setattr__(self, "raw_call_ids", frozenset(self.raw_call_ids))
        object.__setattr__(self, "call_sources", freeze(self.call_sources))
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "node", freeze(self.node))
        object.__setattr__(self, "mode", freeze(self.mode))
        object.__setattr__(self, "tool_spec", freeze(self.tool_spec))
        object.__setattr__(self, "group", freeze(self.group))


class ArtifactReader(Protocol):
    """Read-only access to hash-addressed store objects."""

    def artifact(self, ref: Hash, *, domain: str = "payload", private: bool = False) -> Any: ...

    def context(self, ref: Hash) -> MaterializedContextV1: ...

    def bytes_artifact(self, ref: Hash) -> bytes: ...

    def checkpoint(self, ref: Hash) -> CheckpointV1: ...


@dataclass(frozen=True)
class DerivedArtifact:
    ref: Hash
    value: bytes | Record | WireRecord
    kind: str
    value_kind: Literal["record", "canonical_json", "bytes"]

    def __post_init__(self) -> None:
        validate_hash(self.ref)
        record_types = (Record, WireRecord)
        if not isinstance(self.value, (bytes, *record_types)):
            raise TypeError("derived artifact values must be canonical bytes or typed records")
        if self.kind not in {"artifact", "private", "context_revision", "context_node"}:
            raise ValueError("unsupported derived artifact kind")
        value_kind = self.value_kind
        if value_kind not in {"record", "canonical_json", "bytes"}:
            raise ValueError("unsupported derived artifact value kind")
        if value_kind == "record" and not isinstance(self.value, record_types):
            raise TypeError("record artifacts require a typed record")
        if value_kind == "canonical_json" and isinstance(self.value, bytes):
            identity = domain_hash("payload", load_canonical_json(self.value))
        elif value_kind == "bytes" and isinstance(self.value, bytes):
            identity = domain_hash_bytes("payload", self.value)
        elif value_kind == "record" and isinstance(self.value, record_types):
            identity = self.value.identity()
        else:
            raise ValueError("artifact value kind does not match its value")
        if self.ref != identity:
            raise ValueError("derived artifact ref does not match its value identity")


InputRecord: TypeAlias = (
    WriterTurnV1
    | WriterTurnV2
    | ToolObservationV1
    | AuthorReplyV1
    | EvaluatorResultV1
    | ContextOperationInputV1
    | EnvironmentStepV1
    | MemberStartV1
)


@dataclass(frozen=True)
class Transition:
    event: EventV1
    input: InputRecord
    state: EnvironmentStateV1
    artifacts: tuple[DerivedArtifact, ...]
    view: LineageView

    def __post_init__(self) -> None:
        if not isinstance(self.event, EventV1):
            raise TypeError("transition event must be EventV1")
        if not isinstance(self.input, InputRecord):
            raise TypeError("transition input must be an input record")
        if not isinstance(self.state, EnvironmentStateV1):
            raise TypeError("transition state must be EnvironmentStateV1")
        if any(not isinstance(artifact, DerivedArtifact) for artifact in self.artifacts):
            raise TypeError("transition artifacts must be DerivedArtifact records")
        if not isinstance(self.view, LineageView):
            raise TypeError("transition view must be LineageView")
        object.__setattr__(self, "event", freeze(self.event))
        object.__setattr__(self, "input", freeze(self.input))
        object.__setattr__(self, "state", freeze(self.state))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "view", freeze(self.view))


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    for method_name in ("to_wire", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            result = method()
            if isinstance(result, Mapping):
                return result
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return None


def _key_path(path: str, key: Any) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def first_difference(expected: Any, actual: Any, *, path: str = "state") -> str | None:
    """Return the first stable path where two nested values differ."""
    left_map = _as_mapping(expected)
    right_map = _as_mapping(actual)
    if left_map is not None or right_map is not None:
        if left_map is None or right_map is None:
            return path
        keys = set(left_map) | set(right_map)
        for key in sorted(keys, key=lambda item: (str(type(item)), repr(item))):
            child_path = _key_path(path, key)
            if key not in left_map or key not in right_map:
                return child_path
            mismatch = first_difference(left_map[key], right_map[key], path=child_path)
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or not isinstance(actual, (list, tuple)):
            return path
        for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
            mismatch = first_difference(left, right, path=f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return (
            None if len(expected) == len(actual) else f"{path}[{min(len(expected), len(actual))}]"
        )

    try:
        equal = canonical_bytes(expected) == canonical_bytes(actual)
    except (TypeError, ValueError):
        equal = type(expected) is type(actual) and expected == actual
    if not equal:
        return path
    return None


__all__ = [
    "ArtifactReader",
    "CallSource",
    "CheckpointChain",
    "ContextView",
    "DerivedArtifact",
    "Hash",
    "InputRecord",
    "LineageMode",
    "LineageView",
    "Phase",
    "SampleRef",
    "ToolSpec",
    "Transition",
    "first_difference",
]
