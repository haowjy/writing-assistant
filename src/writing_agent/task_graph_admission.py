"""Fail-closed admission for immutable task-graph instances.

Admission resolves the complete executable contract before any writer is called.
It is intentionally stricter than the Phase 1 value constructors: Phase 1 preserves
wire identities, while this module decides whether those values are runnable.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from writing_agent.task_graph import GraphInstanceV1, NodeSpecV1, domain_hash, safe_path
from writing_agent.task_graph_artifacts import (
    TypedArtifactError,
    validate_phase3_artifact_closure,
)
from writing_agent.task_graph_contracts import (
    GRAPH_TOOLS,
    WRITER_FAMILIES,
    AuthorPacketV1,
    CheckContractV1,
    DecisionBindingsV1,
    EdgeContractV1,
    EvaluatorPacketV1,
    GuardContractV1,
    InteractionPolicyV1,
    NodeContractV1,
    RequirementUpdateV1,
    RewardContractV1,
    ScriptContractV1,
    ScriptedAuthorV1,
)

SUPPORTED_CONTROLLER_VERSIONS = frozenset({"deterministic-v1"})
SUPPORTED_CHECK_VERSIONS = frozenset({"deterministic-v1", "legacy-check-v1"})


class AdmissionError(ValueError):
    """An instance cannot be sampled safely."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class ArtifactResolver(Protocol):
    def resolve(self, identity: str, *, private: bool) -> Any:
        """Return the hash-verified artifact from the required visibility domain."""


class MappingArtifactResolver:
    """Hermetic resolver useful to compilers and tests before store publication."""

    def __init__(
        self,
        public: Mapping[str, Any],
        private: Mapping[str, Any] | None = None,
    ) -> None:
        self.public = dict(public)
        self.private = dict(private or {})

    def resolve(self, identity: str, *, private: bool) -> Any:
        try:
            return validate_phase3_artifact_closure(
                identity,
                private=private,
                load=lambda child, child_private: self._load(child, private=child_private),
            )
        except TypedArtifactError as exc:
            code = "artifact_routing" if exc.kind == "visibility" else "invalid_contract"
            raise AdmissionError(code, f"invalid artifact {identity}: {exc}") from exc

    def _load(self, identity: str, *, private: bool) -> Any:
        expected = self.private if private else self.public
        opposite = self.public if private else self.private
        if identity in expected and identity in opposite:
            raise AdmissionError(
                "artifact_routing", f"{identity} exists in both visibility domains"
            )
        if identity in opposite:
            visibility = "private" if private else "public"
            raise AdmissionError("artifact_routing", f"{identity} is not in {visibility} storage")
        try:
            value = expected[identity]
        except KeyError as exc:
            raise AdmissionError("missing_reference", f"missing artifact {identity}") from exc
        if domain_hash("payload", value) != identity:
            raise AdmissionError("corrupt_reference", f"artifact hash mismatch for {identity}")
        return _wire_copy(value)


def _wire_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _wire_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_copy(item) for item in value]
    return value


class StoreArtifactResolver:
    """Visibility-aware adapter over :class:`TaskGraphStore`."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def resolve(self, identity: str, *, private: bool) -> Any:
        try:
            return validate_phase3_artifact_closure(
                identity,
                private=private,
                load=self._load,
            )
        except TypedArtifactError as exc:
            code = "artifact_routing" if exc.kind == "visibility" else "invalid_contract"
            raise AdmissionError(code, f"invalid artifact {identity}: {exc}") from exc

    def _load(self, identity: str, private: bool) -> Any:
        from writing_agent.task_graph_store import MissingReferenceError, StoreError

        required = "private" if private else "public"
        opposite = "public" if private else "private"
        locations = self.store.artifact_visibilities(identity)
        if required in locations and opposite in locations:
            raise AdmissionError(
                "artifact_routing", f"{identity} exists in both visibility domains"
            )
        if opposite in locations:
            raise AdmissionError("artifact_routing", f"{identity} is not in {required} storage")
        try:
            return self.store.get_artifact(identity, expected_domain="payload", private=private)
        except MissingReferenceError as exc:
            raise AdmissionError("missing_reference", f"missing artifact {identity}") from exc
        except StoreError as exc:
            raise AdmissionError("invalid_contract", f"invalid artifact {identity}") from exc


@dataclass(frozen=True)
class AdmissionPolicyV1:
    writer_family: str | None = None
    allowed_tools: frozenset[str] = GRAPH_TOOLS
    controller_versions: frozenset[str] = SUPPORTED_CONTROLLER_VERSIONS
    check_versions: frozenset[str] = SUPPORTED_CHECK_VERSIONS

    def __post_init__(self) -> None:
        if self.writer_family is not None and self.writer_family not in WRITER_FAMILIES:
            raise ValueError("unsupported requested writer family")
        if not self.allowed_tools <= GRAPH_TOOLS:
            raise ValueError("admission policy cannot authorize unknown tools")
        if not self.controller_versions <= SUPPORTED_CONTROLLER_VERSIONS:
            raise ValueError("admission policy cannot authorize unknown controllers")
        if not self.check_versions <= SUPPORTED_CHECK_VERSIONS:
            raise ValueError("admission policy cannot authorize unknown check versions")


@dataclass(frozen=True)
class AdmittedNodeV1:
    spec: NodeSpecV1
    contract: NodeContractV1
    edges: tuple[EdgeContractV1, ...]
    guards: Mapping[str, GuardContractV1]
    checks: Mapping[str, CheckContractV1]
    script: ScriptContractV1 | ScriptedAuthorV1 | None
    author_packet: AuthorPacketV1 | None = None
    interaction_policy: InteractionPolicyV1 | None = None
    decision_bindings: DecisionBindingsV1 | None = None
    evaluator_packet: EvaluatorPacketV1 | None = None
    reward_contract: RewardContractV1 | None = None


@dataclass(frozen=True)
class AdmittedGraphV1:
    instance: GraphInstanceV1
    nodes: Mapping[str, AdmittedNodeV1]
    policy: AdmissionPolicyV1

    def node(self, node_id: str) -> AdmittedNodeV1:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise AdmissionError("unknown_target", f"unknown admitted node {node_id}") from exc


def _contract(
    resolver: ArtifactResolver,
    identity: str,
    contract_type: type,
    *,
    private: bool,
) -> Any:
    try:
        value = resolver.resolve(identity, private=private)
        return contract_type.from_dict(value)
    except AdmissionError:
        raise
    except (TypeError, ValueError) as exc:
        raise AdmissionError(
            "invalid_contract", f"{identity} is not a valid {contract_type.ARTIFACT_TYPE}"
        ) from exc


def _resolve_plain(resolver: ArtifactResolver, identity: str, *, private: bool) -> Any:
    try:
        return resolver.resolve(identity, private=private)
    except AdmissionError:
        raise
    except (TypeError, ValueError) as exc:
        raise AdmissionError("invalid_contract", f"cannot resolve artifact {identity}") from exc


def admit_graph(
    instance: GraphInstanceV1,
    resolver: ArtifactResolver,
    *,
    policy: AdmissionPolicyV1 | None = None,
) -> AdmittedGraphV1:
    """Resolve and validate one graph completely, without executing any role."""
    if not isinstance(instance, GraphInstanceV1):
        raise TypeError("instance must be GraphInstanceV1")
    policy = policy or AdmissionPolicyV1()
    _validate_global_budget(instance)

    # Phase 1 deliberately places these references in public artifact closure.
    _resolve_plain(resolver, instance.template_ref, private=False)
    for identity in (*instance.source_refs, *instance.request_refs):
        _resolve_plain(resolver, identity, private=False)
    if instance.requirements_ref is not None:
        _resolve_plain(resolver, instance.requirements_ref, private=False)

    specs = {node.id: node for node in instance.nodes}
    if not specs:
        raise AdmissionError("missing_node", "a runnable graph needs at least one node")
    admitted: dict[str, AdmittedNodeV1] = {}
    edge_ids: set[str] = set()
    writer_nodes = 0
    for spec in instance.nodes:
        contract = _contract(resolver, spec.entry_contract, NodeContractV1, private=False)
        _validate_node_identity(spec, contract, policy)
        if spec.kind == "writer":
            writer_nodes += 1
        entry = contract.entry_contract
        if entry.request_ref not in instance.request_refs:
            raise AdmissionError(
                "request_routing",
                f"node {spec.id} request_ref is not declared by the graph instance",
            )
        _resolve_plain(resolver, entry.request_ref, private=False)
        if entry.files_ref not in instance.source_refs:
            raise AdmissionError(
                "source_routing",
                f"node {spec.id} files_ref is not declared by the graph instance",
            )
        _resolve_plain(resolver, entry.files_ref, private=False)
        if entry.requirement_version is not None:
            _resolve_plain(resolver, entry.requirement_version, private=True)
        if set(entry.tool_allowlist) - policy.allowed_tools:
            raise AdmissionError("invalid_tool", f"node {spec.id} requests a disabled tool")

        interaction = contract.interaction_contract
        script = _validate_interaction(resolver, spec, contract, policy)
        checks = _validate_checks(resolver, spec, contract, policy)
        if interaction.mode == "scripted_author" and any(
            check.evaluator_version != "deterministic-v1"
            or check.spec.get("kind")
            not in {"nonempty", "contains", "excludes", "excludes_all", "word_range", "exact"}
            or "path" not in check.spec
            or check.public_evidence_refs
            or check.private_evidence_refs
            for check in checks.values()
        ):
            raise AdmissionError(
                "unsupported_check", "scripted author permits only strict deterministic file checks"
            )
        if interaction.mode == "scripted_author" and not any(
            check.required and check.applicability in {"each_turn", "node_exit_candidate"}
            for check in checks.values()
        ):
            raise AdmissionError(
                "check_coverage", "scripted author needs a terminal required check"
            )
        author_packet = interaction_policy = decision_bindings = None
        if interaction.mode == "scripted_author":
            author_packet, interaction_policy, decision_bindings = _validate_scripted_author(
                resolver, spec, contract, script, checks
            )
        evaluator_packet = reward_contract = None
        if (
            interaction.mode == "scripted_author"
            and contract.completion_contract.evaluation_packet_ref is not None
        ):
            evaluator_packet = _contract(
                resolver,
                contract.completion_contract.evaluation_packet_ref,
                EvaluatorPacketV1,
                private=True,
            )
            reward_contract = _contract(
                resolver, evaluator_packet.reward_contract_ref, RewardContractV1, private=True
            )
            if set(evaluator_packet.check_ids) != set(checks):
                raise AdmissionError("evaluator_coverage", "evaluator packet check IDs differ")
            if set(reward_contract.components) - set(checks):
                raise AdmissionError("reward_coverage", "reward names undeclared checks")
            if any(
                checks[check_id].applicability not in {"each_turn", "node_exit_candidate"}
                for check_id in reward_contract.components
            ):
                raise AdmissionError(
                    "reward_coverage", "reward component lacks terminal check evidence"
                )
        edges, guards = _validate_edges(resolver, spec, specs, edge_ids)
        if interaction.mode == "scripted_author" and any(
            edge.effect != "terminate" for edge in edges
        ):
            raise AdmissionError("unsupported_edge", "scripted-author v1 is single-node terminal")
        if interaction.mode == "scripted_author":
            complete_status = (
                "accepted_partial" if contract.completion_contract.accepted_partial else "complete"
            )
            if not any(
                guard.kind == "always"
                or guard.kind == "task_status"
                and guard.arguments["status"] == complete_status
                or guard.kind == "execution_status"
                and guard.arguments["status"] == "valid"
                or guard.kind == "interaction_complete"
                and guard.arguments["value"] is True
                or guard.kind == "check_status"
                and guard.arguments["status"] == "pass"
                for guard in guards.values()
            ):
                raise AdmissionError("guard_coverage", "no terminal guard can pass on completion")
        for guard in guards.values():
            if guard.kind == "check_status" and guard.arguments["check_id"] not in checks:
                raise AdmissionError(
                    "guard_contract",
                    f"node {spec.id} guard names undeclared check {guard.arguments['check_id']}",
                )
        admitted[spec.id] = AdmittedNodeV1(
            spec=spec,
            contract=contract,
            edges=edges,
            guards=MappingProxyType(guards),
            checks=MappingProxyType(checks),
            script=script,
            author_packet=author_packet,
            interaction_policy=interaction_policy,
            decision_bindings=decision_bindings,
            evaluator_packet=evaluator_packet,
            reward_contract=reward_contract,
        )
        if interaction.mode == "none" and contract.budget_contract.max_author_calls:
            raise AdmissionError(
                "interaction_budget", f"node {spec.id} has author budget but no interaction"
            )
        if interaction.mode == "scripted_author" and "ask_author" not in entry.tool_allowlist:
            raise AdmissionError("invalid_tool", "scripted author node must expose ask_author")
        if interaction.mode == "scripted_author" and contract.completion_contract.repair_turns:
            raise AdmissionError("unsupported_repair", "scripted-author v1 has no repair loop")
        if interaction.mode != "scripted_author" and "ask_author" in entry.tool_allowlist:
            raise AdmissionError("invalid_tool", "ask_author requires scripted_author mode")
        if contract.budget_contract.max_graph_hops != instance.budgets["max_graph_hops"]:
            raise AdmissionError(
                "budget_contract", f"node {spec.id} disagrees with the graph hop bound"
            )

    if not writer_nodes:
        raise AdmissionError("missing_writer", "a runnable graph needs a writer node")
    _validate_reachability_and_cycles(instance, admitted)
    return AdmittedGraphV1(instance, MappingProxyType(admitted), policy)


def _validate_global_budget(instance: GraphInstanceV1) -> None:
    if "max_graph_hops" not in instance.budgets:
        raise AdmissionError("missing_budget", "graph budgets require max_graph_hops")
    value = instance.budgets["max_graph_hops"]
    if type(value) is not int or value < 1:
        raise AdmissionError("unbounded_graph", "max_graph_hops must be positive")


def _validate_node_identity(
    spec: NodeSpecV1,
    contract: NodeContractV1,
    policy: AdmissionPolicyV1,
) -> None:
    if contract.node_id != spec.id or contract.node_kind != spec.kind:
        raise AdmissionError("contract_mismatch", f"node contract does not describe {spec.id}")
    entry = contract.entry_contract
    if spec.kind == "environment":
        if spec.families or entry.family is not None or entry.tool_allowlist:
            raise AdmissionError(
                "node_family", f"environment node {spec.id} declares writer capabilities"
            )
        if contract.interaction_contract.mode != "none":
            raise AdmissionError(
                "node_family", f"environment node {spec.id} cannot interact as a writer"
            )
        if contract.completion_contract.strategy != "environment_operation":
            raise AdmissionError(
                "node_family", f"environment node {spec.id} needs environment completion"
            )
        return
    if not spec.families or set(spec.families) - WRITER_FAMILIES:
        raise AdmissionError("node_family", f"writer node {spec.id} needs valid families")
    if len(spec.families) != len(set(spec.families)):
        raise AdmissionError("node_family", f"writer node {spec.id} repeats a family")
    if entry.family is None or entry.family not in spec.families:
        raise AdmissionError("node_family", f"node {spec.id} entry family is inconsistent")
    if policy.writer_family is not None and policy.writer_family not in spec.families:
        raise AdmissionError(
            "node_family", f"node {spec.id} does not admit requested family {policy.writer_family}"
        )
    if contract.completion_contract.strategy == "environment_operation":
        raise AdmissionError("node_family", f"writer node {spec.id} has environment completion")


def _validate_interaction(
    resolver: ArtifactResolver,
    spec: NodeSpecV1,
    contract: NodeContractV1,
    policy: AdmissionPolicyV1,
) -> ScriptContractV1 | ScriptedAuthorV1 | None:
    interaction = contract.interaction_contract
    completion = contract.completion_contract
    budget = contract.budget_contract
    if completion.controller_version not in policy.controller_versions:
        raise AdmissionError(
            "unsupported_controller", f"node {spec.id} uses {completion.controller_version}"
        )
    if interaction.mode == "simulated_author":
        raise AdmissionError(
            "unsupported_interaction",
            f"node {spec.id} uses simulated_author before typed role contracts exist",
        )
    if interaction.mandatory_feedback and interaction.mode != "scripted_author":
        raise AdmissionError(
            "unsupported_feedback",
            f"node {spec.id} declares mandatory feedback before feedback rules exist",
        )
    script = None
    if interaction.script_ref is not None:
        contract_type = (
            ScriptedAuthorV1 if interaction.mode == "scripted_author" else ScriptContractV1
        )
        script = _contract(resolver, interaction.script_ref, contract_type, private=True)
    for identity, private in (
        (interaction.author_packet_ref, True),
        (interaction.interaction_policy_ref, False),
        (interaction.decision_bindings_ref, True),
    ):
        if identity is not None:
            _resolve_plain(resolver, identity, private=private)
    if interaction.mode == "scripted":
        assert isinstance(script, ScriptContractV1)
        if len(script.fixed_followups) != interaction.scripted_turns:
            raise AdmissionError(
                "script_coverage", f"node {spec.id} fixed follow-up coverage is inconsistent"
            )
        missing = set(interaction.required_script_keys) - set(script.responses)
        if missing:
            raise AdmissionError(
                "script_coverage", f"node {spec.id} lacks scripted responses for {sorted(missing)}"
            )
    elif interaction.mode != "scripted_author" and script is not None:
        raise AdmissionError("script_coverage", f"node {spec.id} has an inapplicable script")
    if interaction.mode == "scripted_author":
        if completion.evaluation_packet_ref is None:
            raise AdmissionError("evaluator_coverage", "scripted author needs an evaluator packet")
        if budget.max_tool_calls < 1:
            raise AdmissionError("interaction_budget", "scripted author needs an ask tool call")
        if budget.max_author_calls < len(interaction.mandatory_feedback) + 1:
            raise AdmissionError(
                "interaction_budget", "author budget cannot cover feedback and ask"
            )
        if budget.max_steps < len(interaction.mandatory_feedback) + 2:
            raise AdmissionError("interaction_budget", "writer budget cannot cover author turns")
    required_author_calls = interaction.scripted_turns + len(interaction.mandatory_feedback)
    required_author_calls += int(bool(interaction.required_script_keys))
    if interaction.mode == "simulated_author" and budget.max_author_calls < 1:
        raise AdmissionError(
            "interaction_budget", f"node {spec.id} cannot afford simulated interaction"
        )
    if required_author_calls > budget.max_author_calls:
        raise AdmissionError(
            "script_coverage", f"node {spec.id} cannot afford its required interactions"
        )
    if completion.required_script_turns != interaction.scripted_turns:
        raise AdmissionError(
            "script_coverage", f"node {spec.id} completion disagrees with script coverage"
        )
    required_writer_turns = 1 + interaction.scripted_turns + len(interaction.mandatory_feedback)
    if required_writer_turns > budget.max_steps:
        raise AdmissionError(
            "script_coverage", f"node {spec.id} cannot afford its required writer turns"
        )
    if completion.evaluation_packet_ref is not None:
        _resolve_plain(resolver, completion.evaluation_packet_ref, private=True)
    return script


def _validate_scripted_author(
    resolver: ArtifactResolver,
    spec: NodeSpecV1,
    contract: NodeContractV1,
    script: ScriptContractV1 | ScriptedAuthorV1 | None,
    checks: Mapping[str, CheckContractV1],
) -> tuple[AuthorPacketV1, InteractionPolicyV1, DecisionBindingsV1]:
    interaction = contract.interaction_contract
    assert isinstance(script, ScriptedAuthorV1)
    assert interaction.author_packet_ref and interaction.interaction_policy_ref
    assert interaction.decision_bindings_ref
    packet = _contract(resolver, interaction.author_packet_ref, AuthorPacketV1, private=True)
    policy = _contract(
        resolver, interaction.interaction_policy_ref, InteractionPolicyV1, private=False
    )
    bindings = _contract(
        resolver, interaction.decision_bindings_ref, DecisionBindingsV1, private=True
    )
    decision_ids = {item["id"] for item in policy.public_decisions}
    if set(bindings.bindings) != decision_ids or set(script.answers) != decision_ids:
        raise AdmissionError("script_coverage", f"node {spec.id} needs exact decision coverage")
    if set(bindings.bindings.values()) - set(packet.preferences):
        raise AdmissionError("script_coverage", "decision binding names missing author preference")
    public_values = [value for item in policy.public_decisions for value in item.values()]
    if any(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", item["id"]) is None
        for item in policy.public_decisions
    ):
        raise AdmissionError("visibility", "public decision ID must use bounded safe syntax")
    if any(
        len(item["label"]) > 256
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in item["label"])
        for item in policy.public_decisions
    ):
        raise AdmissionError("visibility", "public decision label must be bounded printable text")
    private_values = [
        *packet.preferences.keys(),
        *packet.preferences.values(),
        *packet.requirements.keys(),
        *packet.requirements.values(),
        *(rule["utterance"] for rule in script.answers.values()),
        *(rule["value"] for rule in script.answers.values()),
        *(rule["utterance"] for rule in script.feedback),
    ]
    if tuple(rule["id"] for rule in script.feedback) != policy.mandatory_feedback or (
        policy.mandatory_feedback != interaction.mandatory_feedback
    ):
        raise AdmissionError("script_coverage", "mandatory feedback routes disagree")
    for public_id, rule in script.answers.items():
        if rule["value"] not in rule["utterance"]:
            raise AdmissionError("script_coverage", "author answer must state its bound value")
        if (
            rule["mode"] == "fixed_answer"
            and rule["value"] != packet.preferences[bindings.bindings[public_id]]
        ):
            raise AdmissionError("script_coverage", "fixed answer differs from author packet")
        if set(rule["prerequisite_check_ids"]) - set(checks):
            raise AdmissionError("script_coverage", "answer prerequisite names unknown check")
    superseded_ids: set[str] = set()
    replacement_ids: set[str] = set()
    for feedback in script.feedback:
        available = {
            check.id
            for check in checks.values()
            if check.applicability in {"each_turn", f"before_feedback:{feedback['id']}"}
        }
        if set(feedback["prerequisite_check_ids"]) - available:
            raise AdmissionError("script_coverage", "feedback prerequisite names unknown check")
        update_ref = feedback["requirement_update_ref"]
        if update_ref is not None:
            update = _contract(resolver, update_ref, RequirementUpdateV1, private=True)
            private_values.append(update.id)
            private_values.append(update.replacement)
            if (
                update.supersedes not in packet.requirements
                or update.supersedes in superseded_ids
                or update.id in packet.requirements
                or update.id in replacement_ids
            ):
                raise AdmissionError("requirement_update", "unauthorized superseded requirement")
            superseded_ids.add(update.supersedes)
            replacement_ids.add(update.id)
    if any(
        private_text and private_text in public
        for public in public_values
        for private_text in private_values
    ):
        raise AdmissionError(
            "visibility", "public decision vocabulary contains private author text"
        )
    return packet, policy, bindings


def _validate_checks(
    resolver: ArtifactResolver,
    spec: NodeSpecV1,
    contract: NodeContractV1,
    policy: AdmissionPolicyV1,
) -> dict[str, CheckContractV1]:
    checks: dict[str, CheckContractV1] = {}
    mandatory_ids: list[str] = []
    feedback = set(contract.interaction_contract.mandatory_feedback)
    for required, refs in ((True, contract.mandatory_checks), (False, contract.optional_checks)):
        for identity in refs:
            check = _contract(resolver, identity, CheckContractV1, private=True)
            if check.id in checks:
                raise AdmissionError("duplicate_check", f"duplicate check id {check.id}")
            if check.required is not required:
                raise AdmissionError(
                    "check_contract", f"check {check.id} required flag is misrouted"
                )
            if check.evaluator_version not in policy.check_versions:
                raise AdmissionError(
                    "unsupported_check", f"check {check.id} uses {check.evaluator_version}"
                )
            _validate_check_program(check)
            for identity in check.public_evidence_refs:
                _resolve_plain(resolver, identity, private=False)
            for identity in check.private_evidence_refs:
                _resolve_plain(resolver, identity, private=True)
            if check.applicability.startswith("before_feedback:"):
                feedback_id = check.applicability.split(":", 1)[1]
                if feedback_id not in feedback:
                    raise AdmissionError(
                        "check_applicability",
                        f"check {check.id} names unknown feedback {feedback_id}",
                    )
            checks[check.id] = check
            if required:
                mandatory_ids.append(check.id)
    completion = contract.completion_contract
    if tuple(mandatory_ids) != completion.required_check_ids:
        raise AdmissionError(
            "check_contract", f"node {spec.id} completion does not name every mandatory check"
        )
    return checks


_LEGACY_CHECK_BASE = frozenset({"id", "metric", "kind", "method", "required"})
_TEXT_TARGET_KINDS = frozenset(
    {"nonempty", "contains", "excludes", "excludes_all", "word_range", "exact", "alias_answer"}
)
_MECHANICAL_KINDS = _TEXT_TARGET_KINDS | frozenset(
    {"protected", "allowed_changes", "evidence_exposed", "kb_word_budget", "wiki_links"}
)


def _validate_check_program(check: CheckContractV1) -> None:
    try:
        _validate_legacy_check_shape(
            check,
            deterministic_only=check.evaluator_version == "deterministic-v1",
        )
    except (TypeError, ValueError) as exc:
        raise AdmissionError(
            "invalid_check_program",
            f"check {check.id} has no supported {check.evaluator_version} program",
        ) from exc


def _validate_legacy_check_shape(check: CheckContractV1, *, deterministic_only: bool) -> None:
    spec = dict(check.spec)
    if not _LEGACY_CHECK_BASE <= spec.keys():
        raise ValueError("missing check program fields")
    if spec["id"] != check.id or spec["required"] is not check.required:
        raise ValueError("check envelope and program disagree")
    if spec["metric"] not in {f"Q{number}" for number in range(1, 14)}:
        raise ValueError("unsupported check metric")
    kind = spec["kind"]
    method = spec["method"]
    if kind == "semantic":
        if deterministic_only or method != "llm_judge":
            raise ValueError("semantic program is not deterministic")
        allowed = _LEGACY_CHECK_BASE | {"text", "weight"}
        if set(spec) - allowed or not isinstance(spec.get("text"), str) or not spec["text"]:
            raise ValueError("invalid semantic program")
        if "weight" in spec and (
            isinstance(spec["weight"], bool)
            or not isinstance(spec["weight"], (int, float))
            or not math.isfinite(spec["weight"])
            or spec["weight"] <= 0
        ):
            raise ValueError("semantic weight must be positive")
        return
    if kind not in _MECHANICAL_KINDS or method != "deterministic":
        raise ValueError("unsupported deterministic check kind")

    operands: set[str]
    if kind == "nonempty" or kind == "wiki_links":
        operands = set()
    elif kind in {"contains", "excludes", "exact", "evidence_exposed"}:
        operands = {"text"}
    elif kind == "excludes_all":
        operands = {"texts"}
    elif kind == "word_range":
        operands = {"min", "max"}
    elif kind == "protected":
        operands = {"text"}
    elif kind == "allowed_changes":
        operands = {"paths"}
    elif kind == "alias_answer":
        operands = {"aliases"}
    elif kind == "kb_word_budget":
        operands = {"max"}
    else:  # pragma: no cover - exhaustive above
        raise AssertionError(kind)

    target_fields = set(spec) & {"path", "artifact"}
    if len(target_fields) > 1:
        raise ValueError("check program has ambiguous target")
    if kind == "protected" and target_fields != {"path"}:
        raise ValueError("protected checks require a file target")
    if (
        kind in {"allowed_changes", "evidence_exposed", "kb_word_budget", "wiki_links"}
        and target_fields
    ):
        raise ValueError("check kind does not accept a text target")
    if kind not in _TEXT_TARGET_KINDS and kind != "protected" and target_fields:
        raise ValueError("unsupported target")
    allowed = _LEGACY_CHECK_BASE | operands | target_fields
    if set(spec) != allowed:
        raise ValueError("check program has missing or unknown operands")

    for field in ("text", "path", "artifact"):
        if field in spec and (not isinstance(spec[field], str) or not spec[field]):
            raise TypeError(f"{field} must be nonempty text")
    if "path" in spec:
        safe_path(spec["path"])
    for field in ("texts", "paths", "aliases"):
        if field in spec and (
            not isinstance(spec[field], (list, tuple))
            or not spec[field]
            or any(not isinstance(value, str) or not value for value in spec[field])
        ):
            raise TypeError(f"{field} must be a nonempty text array")
    if kind == "word_range" and (
        type(spec["min"]) is not int
        or type(spec["max"]) is not int
        or not 0 <= spec["min"] <= spec["max"]
    ):
        raise ValueError("invalid word range")
    if kind == "kb_word_budget" and (type(spec["max"]) is not int or spec["max"] < 0):
        raise ValueError("invalid KB word budget")


def _validate_edges(
    resolver: ArtifactResolver,
    spec: NodeSpecV1,
    specs: Mapping[str, NodeSpecV1],
    graph_edge_ids: set[str],
) -> tuple[tuple[EdgeContractV1, ...], dict[str, GuardContractV1]]:
    if not spec.exits:
        raise AdmissionError("missing_edge", f"node {spec.id} has no exit")
    edges: list[EdgeContractV1] = []
    guards: dict[str, GuardContractV1] = {}
    try:
        for value in spec.exits:
            edge = EdgeContractV1.from_dict(value)
            if edge.edge_id in graph_edge_ids:
                raise AdmissionError("duplicate_edge", f"duplicate edge id {edge.edge_id}")
            graph_edge_ids.add(edge.edge_id)
            if edge.target_node is not None and edge.target_node not in specs:
                raise AdmissionError(
                    "unknown_target", f"edge {edge.edge_id} targets {edge.target_node}"
                )
            guard = _contract(resolver, edge.guard_ref, GuardContractV1, private=False)
            edges.append(edge)
            guards[edge.edge_id] = guard
    except AdmissionError:
        raise
    except (TypeError, ValueError) as exc:
        raise AdmissionError("invalid_edge", f"node {spec.id} contains an invalid edge") from exc
    if len(edges) > 1:
        precedences = [edge.precedence for edge in edges]
        if any(value is None for value in precedences) or len(set(precedences)) != len(precedences):
            raise AdmissionError(
                "guard_precedence",
                f"node {spec.id} has potentially simultaneous guards without unique precedence",
            )
    return tuple(sorted(edges, key=lambda edge: edge.precedence or 0)), guards


def _validate_reachability_and_cycles(
    instance: GraphInstanceV1,
    admitted: Mapping[str, AdmittedNodeV1],
) -> None:
    adjacency = {
        node_id: tuple(
            edge.target_node
            for edge in node.edges
            if edge.target_node is not None and edge.effect in {"advance", "branch"}
        )
        for node_id, node in admitted.items()
    }
    reachable: set[str] = set()
    stack = [instance.entry_node]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(adjacency[current])
    unreachable = set(admitted) - reachable
    if unreachable:
        raise AdmissionError("unreachable_node", f"unreachable nodes: {sorted(unreachable)}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            # Both the graph-wide hop bound and each node visit bound are immutable.
            if instance.budgets["max_graph_hops"] < 1 or any(
                node.contract.budget_contract.max_visits < 1 for node in admitted.values()
            ):
                raise AdmissionError("unbounded_cycle", "cycle lacks immutable visit/hop bounds")
            return
        if node_id in visited:
            return
        visiting.add(node_id)
        for target in adjacency[node_id]:
            visit(target)
        visiting.remove(node_id)
        visited.add(node_id)

    visit(instance.entry_node)
