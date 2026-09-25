"""Versioned execution contracts for admitted task graphs.

Phase 1 froze the identities of :class:`GraphInstanceV1` and :class:`NodeSpecV1`.
The executable node configuration therefore lives in immutable artifacts named by
``NodeSpecV1.entry_contract``.  These records are deliberately data-only: admission
interprets their small vocabulary, while writer and author models never execute it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, ClassVar, Self

from writing_agent.task_graph import domain_hash, validate_hash

FILE_TOOLS = frozenset({"list_dir", "read_file", "search", "write_file", "patch_file"})
GRAPH_TOOLS = FILE_TOOLS | {"ask_author"}
WRITER_FAMILIES = frozenset({"F1", "F2", "F3", "F4", "F5"})
CHECK_APPLICABILITY = frozenset({"each_turn", "node_exit_candidate"})
TASK_STATUSES = frozenset({"complete", "accepted_partial", "incomplete", "unknown"})
EXECUTION_STATUSES = frozenset(
    {
        "running",
        "valid",
        "interrupted",
        "environment_error",
        "backend_error",
        "simulator_error",
        "controller_error",
        "external_cancelled",
    }
)
CHECK_STATUSES = frozenset({"pass", "fail", "unavailable"})
GUARD_BUDGETS = frozenset(
    {
        "writer_turns",
        "author_calls",
        "tool_calls",
        "read_tokens",
        "generated_tokens",
        "total_tokens",
        "graph_hops",
        "wall_time",
    }
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _nonnegative(value: int, label: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be a {qualifier} integer")


def _strings(value: tuple[str, ...], label: str, *, unique: bool = True) -> None:
    if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"{label} must be an array of nonempty strings")
    if unique and len(value) != len(set(value)):
        raise ValueError(f"{label} must be unique")


@dataclass(frozen=True)
class _Contract:
    """Strict artifact codec shared by every Phase 3 contract."""

    schema: int = 1
    ARTIFACT_TYPE: ClassVar[str]

    def __post_init__(self) -> None:
        if type(self.schema) is not int or self.schema != 1:
            raise ValueError(f"unsupported {self.ARTIFACT_TYPE} schema")
        for field in fields(self):
            if field.name != "schema":
                object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))
        self.validate()

    def validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.ARTIFACT_TYPE,
            **{field.name: _thaw(getattr(self, field.name)) for field in fields(self)},
        }

    def identity(self) -> str:
        return domain_hash("payload", self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.ARTIFACT_TYPE} must be an object")
        expected = {field.name for field in fields(cls)} | {"artifact_type"}
        if set(value) != expected:
            missing = sorted(expected - set(value))
            unknown = sorted(set(value) - expected)
            raise ValueError(
                f"invalid {cls.ARTIFACT_TYPE} fields; missing={missing}, unknown={unknown}"
            )
        if value["artifact_type"] != cls.ARTIFACT_TYPE:
            raise ValueError(f"expected {cls.ARTIFACT_TYPE}")
        kwargs = {key: value[key] for key in expected - {"artifact_type"}}
        result = cls(**kwargs)
        if result.to_dict() != dict(value):
            raise ValueError(f"{cls.ARTIFACT_TYPE} is not in canonical typed form")
        return result


@dataclass(frozen=True)
class NodeEntryV1(_Contract):
    request_ref: str = ""
    files_ref: str = ""
    context_policy: str = "carry"
    requirement_version: str | None = None
    tool_allowlist: tuple[str, ...] = ()
    family: str | None = None
    ARTIFACT_TYPE: ClassVar[str] = "NodeEntryV1"

    def validate(self) -> None:
        validate_hash(self.request_ref)
        validate_hash(self.files_ref)
        validate_hash(self.requirement_version, optional=True)
        if self.context_policy not in {"carry", "seed", "drop"}:
            raise ValueError("unsupported entry context policy")
        _strings(self.tool_allowlist, "tool_allowlist")
        if set(self.tool_allowlist) - GRAPH_TOOLS:
            raise ValueError("entry declares an unsupported writer tool")
        if self.family is not None and self.family not in WRITER_FAMILIES:
            raise ValueError("entry declares an unsupported writer family")


@dataclass(frozen=True)
class InteractionContractV1(_Contract):
    mode: str = "none"
    script_ref: str | None = None
    author_packet_ref: str | None = None
    interaction_policy_ref: str | None = None
    decision_bindings_ref: str | None = None
    fallback_policy: str = "none"
    required_script_keys: tuple[str, ...] = ()
    scripted_turns: int = 0
    mandatory_feedback: tuple[str, ...] = ()
    ARTIFACT_TYPE: ClassVar[str] = "InteractionContractV1"

    def validate(self) -> None:
        if self.mode not in {"none", "scripted", "scripted_author", "simulated_author"}:
            raise ValueError("unsupported interaction mode")
        for ref in (
            self.script_ref,
            self.author_packet_ref,
            self.interaction_policy_ref,
            self.decision_bindings_ref,
        ):
            validate_hash(ref, optional=True)
        if self.fallback_policy not in {"none", "pinned_author_adapter"}:
            raise ValueError("unsupported interaction fallback")
        _strings(self.required_script_keys, "required_script_keys")
        _strings(self.mandatory_feedback, "mandatory_feedback")
        _nonnegative(self.scripted_turns, "scripted_turns")
        author_contracts = (
            self.script_ref,
            self.author_packet_ref,
            self.interaction_policy_ref,
            self.decision_bindings_ref,
            self.required_script_keys,
            self.scripted_turns,
            self.mandatory_feedback,
        )
        if self.mode == "none" and (self.fallback_policy != "none" or any(author_contracts)):
            raise ValueError("interaction mode none cannot carry author contracts")
        if self.mode == "scripted":
            if self.script_ref is None:
                raise ValueError("scripted interaction requires script_ref")
            if (
                any(
                    (
                        self.author_packet_ref,
                        self.interaction_policy_ref,
                        self.decision_bindings_ref,
                    )
                )
                or self.fallback_policy != "none"
            ):
                raise ValueError("scripted interaction cannot carry simulator contracts")
        if self.mode == "scripted_author" and (
            self.script_ref is None
            or self.author_packet_ref is None
            or self.interaction_policy_ref is None
            or self.decision_bindings_ref is None
            or self.fallback_policy != "none"
            or self.scripted_turns
            or self.required_script_keys
        ):
            raise ValueError("scripted_author requires a complete deterministic role contract")
        if self.mode == "simulated_author" and (
            self.author_packet_ref is None or self.interaction_policy_ref is None
        ):
            raise ValueError("simulated_author requires author packet and interaction policy")
        if self.mode == "simulated_author" and (
            self.script_ref is not None or self.scripted_turns or self.required_script_keys
        ):
            raise ValueError("simulated_author cannot carry scripted interaction")


@dataclass(frozen=True)
class BudgetContractV1(_Contract):
    max_steps: int = 0
    max_tool_calls: int = 0
    max_read_tokens: int = 0
    max_total_bytes: int = 0
    max_author_calls: int = 0
    max_graph_hops: int = 0
    max_visits: int = 0
    ARTIFACT_TYPE: ClassVar[str] = "BudgetContractV1"

    def validate(self) -> None:
        _nonnegative(self.max_steps, "max_steps", positive=True)
        _nonnegative(self.max_tool_calls, "max_tool_calls")
        _nonnegative(self.max_read_tokens, "max_read_tokens")
        _nonnegative(self.max_total_bytes, "max_total_bytes", positive=True)
        _nonnegative(self.max_author_calls, "max_author_calls")
        _nonnegative(self.max_graph_hops, "max_graph_hops", positive=True)
        _nonnegative(self.max_visits, "max_visits", positive=True)

    def legacy_agent_budgets(self) -> dict[str, int]:
        return {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_read_tokens": self.max_read_tokens,
        }


@dataclass(frozen=True)
class CompletionContractV1(_Contract):
    controller_version: str = "deterministic-v1"
    strategy: str = "required_checks"
    required_check_ids: tuple[str, ...] = ()
    accepted_partial: bool = False
    repair_turns: int = 0
    required_script_turns: int = 0
    evaluation_packet_ref: str | None = None
    ARTIFACT_TYPE: ClassVar[str] = "CompletionContractV1"

    def validate(self) -> None:
        if not isinstance(self.controller_version, str) or not self.controller_version:
            raise TypeError("controller_version must be a nonempty string")
        if self.strategy not in {"required_checks", "environment_operation"}:
            raise ValueError("unsupported completion strategy")
        _strings(self.required_check_ids, "required_check_ids")
        if not isinstance(self.accepted_partial, bool):
            raise TypeError("accepted_partial must be bool")
        _nonnegative(self.repair_turns, "repair_turns")
        _nonnegative(self.required_script_turns, "required_script_turns")
        validate_hash(self.evaluation_packet_ref, optional=True)


@dataclass(frozen=True)
class CheckContractV1(_Contract):
    id: str = ""
    evaluator_version: str = "deterministic-v1"
    applicability: str = "node_exit_candidate"
    required: bool = True
    public_evidence_refs: tuple[str, ...] = ()
    private_evidence_refs: tuple[str, ...] = ()
    spec: Mapping[str, Any] = MappingProxyType({})
    ARTIFACT_TYPE: ClassVar[str] = "CheckContractV1"

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("check id is required")
        if not isinstance(self.evaluator_version, str) or not self.evaluator_version:
            raise TypeError("evaluator_version must be a nonempty string")
        if self.applicability not in CHECK_APPLICABILITY and not self.applicability.startswith(
            "before_feedback:"
        ):
            raise ValueError("unsupported check applicability")
        if (
            self.applicability.startswith("before_feedback:")
            and not self.applicability.split(":", 1)[1]
        ):
            raise ValueError("before_feedback scope requires an id")
        if not isinstance(self.required, bool):
            raise TypeError("check required must be bool")
        for refs, label in (
            (self.public_evidence_refs, "public_evidence_refs"),
            (self.private_evidence_refs, "private_evidence_refs"),
        ):
            if not isinstance(refs, tuple):
                raise TypeError(f"{label} must be an array")
            for ref in refs:
                validate_hash(ref)
            if len(refs) != len(set(refs)):
                raise ValueError(f"{label} must be unique")
        if set(self.public_evidence_refs) & set(self.private_evidence_refs):
            raise ValueError("check evidence cannot be both public and private")
        if not isinstance(self.spec, Mapping):
            raise TypeError("check spec must be an object")


@dataclass(frozen=True)
class ScriptContractV1(_Contract):
    fixed_followups: tuple[str, ...] = ()
    responses: Mapping[str, str] = MappingProxyType({})
    ARTIFACT_TYPE: ClassVar[str] = "ScriptContractV1"

    def validate(self) -> None:
        if not isinstance(self.fixed_followups, tuple) or any(
            not isinstance(item, str) for item in self.fixed_followups
        ):
            raise TypeError("fixed_followups must be an array of strings")
        if not isinstance(self.responses, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.responses.items()
        ):
            raise TypeError("script responses must map nonempty string keys to text")


@dataclass(frozen=True)
class AuthorPacketV1(_Contract):
    """Private author facts, never an evaluator packet or a writer message."""

    preferences: Mapping[str, str] = MappingProxyType({})
    requirements: Mapping[str, str] = MappingProxyType({})
    ARTIFACT_TYPE: ClassVar[str] = "AuthorPacketV1"

    def validate(self) -> None:
        for name in ("preferences", "requirements"):
            values = getattr(self, name)
            if not isinstance(values, Mapping) or any(
                not isinstance(key, str) or not key or not isinstance(value, str) or not value
                for key, value in values.items()
            ):
                raise ValueError(f"{name} must map nonempty ids to nonempty text")


@dataclass(frozen=True)
class InteractionPolicyV1(_Contract):
    """Public request vocabulary; contains no answer, requirement, or rubric text."""

    public_decisions: tuple[Mapping[str, str], ...] = ()
    max_decisions_per_request: int = 1
    max_question_chars: int = 1000
    max_proposals: int = 8
    max_proposal_chars: int = 1000
    repeat: str = "replay_disclosed_answer"
    mandatory_feedback: tuple[str, ...] = ()
    ARTIFACT_TYPE: ClassVar[str] = "InteractionPolicyV1"

    def validate(self) -> None:
        ids = []
        for item in self.public_decisions:
            if (
                not isinstance(item, Mapping)
                or set(item) != {"id", "label"}
                or any(not isinstance(value, str) or not value for value in item.values())
            ):
                raise ValueError("public decisions need exactly id and label")
            ids.append(item["id"])
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("public decision ids must be nonempty and unique")
        for name in (
            "max_decisions_per_request",
            "max_question_chars",
            "max_proposals",
            "max_proposal_chars",
        ):
            _nonnegative(getattr(self, name), name, positive=True)
        if self.max_decisions_per_request > len(ids):
            raise ValueError("request decision limit exceeds declared topics")
        if self.repeat != "replay_disclosed_answer":
            raise ValueError("unsupported repeat policy")
        _strings(self.mandatory_feedback, "mandatory_feedback")


@dataclass(frozen=True)
class DecisionBindingsV1(_Contract):
    bindings: Mapping[str, str] = MappingProxyType({})
    ARTIFACT_TYPE: ClassVar[str] = "DecisionBindingsV1"

    def validate(self) -> None:
        if not isinstance(self.bindings, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.bindings.items()
        ):
            raise ValueError("decision bindings must map public ids to private ids")


@dataclass(frozen=True)
class RequirementUpdateV1(_Contract):
    id: str = ""
    supersedes: str = ""
    replacement: str = ""
    ARTIFACT_TYPE: ClassVar[str] = "RequirementUpdateV1"

    def validate(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.id, self.supersedes, self.replacement)
        ):
            raise ValueError("requirement update needs id, supersedes, and replacement")


@dataclass(frozen=True)
class ScriptedAuthorV1(_Contract):
    """Exact prepared answers and ordered feedback, never executable code."""

    answers: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
    feedback: tuple[Mapping[str, Any], ...] = ()
    ARTIFACT_TYPE: ClassVar[str] = "ScriptedAuthorV1"

    def validate(self) -> None:
        if not isinstance(self.answers, Mapping):
            raise TypeError("answers must be a map")
        for key, rule in self.answers.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(rule, Mapping)
                or set(rule) != {"mode", "utterance", "value", "selector", "prerequisite_check_ids"}
            ):
                raise ValueError("invalid scripted answer rule")
            if rule["mode"] not in {
                "fixed_answer",
                "declared_option_id",
                "declared_option_position",
            }:
                raise ValueError("unsupported answer mode")
            if not isinstance(rule["utterance"], str) or not rule["utterance"]:
                raise ValueError("script utterance must be nonempty")
            if not isinstance(rule["value"], str) or not rule["value"]:
                raise ValueError("script decision value must be nonempty")
            if rule["mode"] == "fixed_answer" and rule["selector"] is not None:
                raise ValueError("fixed answer cannot select a proposal")
            if rule["mode"] == "declared_option_id" and (
                not isinstance(rule["selector"], str) or not rule["selector"]
            ):
                raise ValueError("option-id selector must be an id")
            if rule["mode"] == "declared_option_position" and (
                type(rule["selector"]) is not int or rule["selector"] < 0
            ):
                raise ValueError("option-position selector must be nonnegative")
            if not isinstance(rule["prerequisite_check_ids"], (list, tuple)):
                raise TypeError("answer prerequisites must be an array")
            _strings(tuple(rule["prerequisite_check_ids"]), "answer prerequisites")
        feedback_ids = []
        for rule in self.feedback:
            if not isinstance(rule, Mapping) or set(rule) != {
                "id",
                "utterance",
                "prerequisite_check_ids",
                "requirement_update_ref",
            }:
                raise ValueError("invalid feedback rule")
            if (
                not isinstance(rule["id"], str)
                or not rule["id"]
                or not isinstance(rule["utterance"], str)
                or not rule["utterance"]
            ):
                raise ValueError("feedback id and utterance must be nonempty")
            if not isinstance(rule["prerequisite_check_ids"], (list, tuple)):
                raise TypeError("feedback prerequisites must be an array")
            _strings(tuple(rule["prerequisite_check_ids"]), "feedback prerequisites")
            validate_hash(rule["requirement_update_ref"], optional=True)
            feedback_ids.append(rule["id"])
        if len(feedback_ids) != len(set(feedback_ids)):
            raise ValueError("feedback ids must be unique")


@dataclass(frozen=True)
class RewardContractV1(_Contract):
    """Integer basis-point weights make scoring exact and replayable."""

    components: Mapping[str, int] = MappingProxyType({})
    normalization: int = 10000
    incomplete_score: int = 0
    ARTIFACT_TYPE: ClassVar[str] = "RewardContractV1"

    def validate(self) -> None:
        _nonnegative(self.normalization, "normalization", positive=True)
        _nonnegative(self.incomplete_score, "incomplete_score")
        if (
            not isinstance(self.components, Mapping)
            or not self.components
            or any(
                not isinstance(key, str) or not key or type(value) is not int or value < 0
                for key, value in self.components.items()
            )
        ):
            raise ValueError("reward components need nonnegative integer weights")
        if sum(self.components.values()) != self.normalization:
            raise ValueError("reward weights must sum to normalization")
        if self.incomplete_score > self.normalization:
            raise ValueError("incomplete score cannot exceed normalization")


@dataclass(frozen=True)
class EvaluatorPacketV1(_Contract):
    reward_contract_ref: str = ""
    check_ids: tuple[str, ...] = ()
    ARTIFACT_TYPE: ClassVar[str] = "EvaluatorPacketV1"

    def validate(self) -> None:
        validate_hash(self.reward_contract_ref)
        _strings(self.check_ids, "evaluator check ids")


@dataclass(frozen=True)
class GuardContractV1(_Contract):
    kind: str = "always"
    arguments: Mapping[str, Any] = MappingProxyType({})
    ARTIFACT_TYPE: ClassVar[str] = "GuardContractV1"

    def validate(self) -> None:
        if self.kind not in {
            "always",
            "never",
            "task_status",
            "execution_status",
            "check_status",
            "interaction_complete",
            "budget_remaining",
        }:
            raise ValueError("unsupported guard kind")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("guard arguments must be an object")
        expected = {
            "always": set(),
            "never": set(),
            "task_status": {"status"},
            "execution_status": {"status"},
            "check_status": {"check_id", "status"},
            "interaction_complete": {"value"},
            "budget_remaining": {"budget"},
        }[self.kind]
        if set(self.arguments) != expected:
            raise ValueError("guard arguments do not match its declared kind")
        if self.kind in {"task_status", "execution_status", "check_status"} and any(
            not isinstance(value, str) or not value for value in self.arguments.values()
        ):
            raise TypeError("status guard arguments must be nonempty strings")
        if self.kind == "interaction_complete" and type(self.arguments["value"]) is not bool:
            raise TypeError("interaction_complete guard requires a bool")
        if self.kind == "budget_remaining" and (
            not isinstance(self.arguments["budget"], str) or not self.arguments["budget"]
        ):
            raise TypeError("budget_remaining guard requires a budget name")
        if self.kind == "task_status" and self.arguments["status"] not in TASK_STATUSES:
            raise ValueError("task_status guard uses an unsupported status")
        if self.kind == "execution_status" and self.arguments["status"] not in EXECUTION_STATUSES:
            raise ValueError("execution_status guard uses an unsupported status")
        if self.kind == "check_status" and self.arguments["status"] not in CHECK_STATUSES:
            raise ValueError("check_status guard uses an unsupported status")
        if self.kind == "budget_remaining" and self.arguments["budget"] not in GUARD_BUDGETS:
            raise ValueError("budget_remaining guard uses an unsupported counter")


@dataclass(frozen=True)
class NodeContractV1(_Contract):
    node_id: str = ""
    node_kind: str = "writer"
    entry: Mapping[str, Any] = MappingProxyType({})
    interaction: Mapping[str, Any] = MappingProxyType({})
    budgets: Mapping[str, Any] = MappingProxyType({})
    completion: Mapping[str, Any] = MappingProxyType({})
    mandatory_checks: tuple[str, ...] = ()
    optional_checks: tuple[str, ...] = ()
    ARTIFACT_TYPE: ClassVar[str] = "NodeContractV1"

    def __post_init__(self) -> None:
        for name, contract_type in (
            ("entry", NodeEntryV1),
            ("interaction", InteractionContractV1),
            ("budgets", BudgetContractV1),
            ("completion", CompletionContractV1),
        ):
            value = getattr(self, name)
            if isinstance(value, contract_type):
                value = value.to_dict()
            object.__setattr__(self, name, value)
        super().__post_init__()

    @property
    def entry_contract(self) -> NodeEntryV1:
        return NodeEntryV1.from_dict(_thaw(self.entry))

    @property
    def interaction_contract(self) -> InteractionContractV1:
        return InteractionContractV1.from_dict(_thaw(self.interaction))

    @property
    def budget_contract(self) -> BudgetContractV1:
        return BudgetContractV1.from_dict(_thaw(self.budgets))

    @property
    def completion_contract(self) -> CompletionContractV1:
        return CompletionContractV1.from_dict(_thaw(self.completion))

    def validate(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id is required")
        if self.node_kind not in {"writer", "environment"}:
            raise ValueError("unsupported node kind")
        _ = (
            self.entry_contract,
            self.interaction_contract,
            self.budget_contract,
            self.completion_contract,
        )
        _strings(self.mandatory_checks, "mandatory_checks")
        _strings(self.optional_checks, "optional_checks")
        if set(self.mandatory_checks) & set(self.optional_checks):
            raise ValueError("a check cannot be both mandatory and optional")


@dataclass(frozen=True)
class EdgeContractV1:
    """Typed view of the edge mapping embedded in the frozen Phase 1 node."""

    edge_id: str
    guard_ref: str
    target_node: str | None
    effect: str
    precedence: int | None
    schema: int = 1

    def __post_init__(self) -> None:
        if type(self.schema) is not int or self.schema != 1:
            raise ValueError("unsupported edge schema")
        if not isinstance(self.edge_id, str) or not self.edge_id:
            raise ValueError("edge_id is required")
        validate_hash(self.guard_ref)
        if self.target_node is not None and (
            not isinstance(self.target_node, str) or not self.target_node
        ):
            raise ValueError("target_node must be a nonempty string or null")
        if self.effect not in {"advance", "branch", "terminate"}:
            raise ValueError("unsupported edge effect")
        if self.precedence is not None and (
            type(self.precedence) is not int or self.precedence < 0
        ):
            raise ValueError("edge precedence must be a nonnegative integer or null")
        if self.effect == "terminate" and self.target_node is not None:
            raise ValueError("terminate edges cannot name a target")
        if self.effect != "terminate" and self.target_node is None:
            raise ValueError("advance and branch edges require a target")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EdgeContractV1:
        if not isinstance(value, Mapping):
            raise TypeError("edge must be an object")
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise ValueError("edge must contain exactly the EdgeContractV1 fields")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}
