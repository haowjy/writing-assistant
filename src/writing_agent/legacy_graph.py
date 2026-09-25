"""Opt-in, behavior-preserving adapter from legacy scenarios to task graphs.

The existing suite continues to call :func:`agent.run_agent` directly.  This module
only compiles an equivalent immutable contract for callers that explicitly request
it; it does not execute a writer or evaluate checks.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from writing_agent.task_graph import GraphInstanceV1, NodeSpecV1, domain_hash
from writing_agent.task_graph_admission import (
    AdmissionPolicyV1,
    AdmittedGraphV1,
    MappingArtifactResolver,
    admit_graph,
)
from writing_agent.task_graph_contracts import (
    BudgetContractV1,
    CheckContractV1,
    CompletionContractV1,
    EdgeContractV1,
    GuardContractV1,
    InteractionContractV1,
    NodeContractV1,
    NodeEntryV1,
    ScriptContractV1,
)


def _artifact(body: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    value = copy.deepcopy(dict(body))
    return domain_hash("payload", value), value


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


@dataclass(frozen=True)
class LegacyGraphBundleV1:
    """Compiled graph plus its immutable objects and exact legacy projections."""

    instance: GraphInstanceV1
    public_artifacts: Mapping[str, Any]
    private_artifacts: Mapping[str, Any]
    initial_files: Mapping[str, str]
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[str, ...]
    followups: tuple[str, ...]
    budgets: Mapping[str, int]
    prose: tuple[Mapping[str, Any], ...]
    check_package: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "public_artifacts",
            "private_artifacts",
            "initial_files",
            "budgets",
            "check_package",
        ):
            object.__setattr__(self, name, _freeze(copy.deepcopy(dict(getattr(self, name)))))
        object.__setattr__(
            self,
            "messages",
            tuple(_freeze(copy.deepcopy(dict(message))) for message in self.messages),
        )
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "followups", tuple(self.followups))
        object.__setattr__(
            self,
            "prose",
            tuple(_freeze(copy.deepcopy(dict(selector))) for selector in self.prose),
        )

    def admission(self) -> AdmittedGraphV1:
        family = self.instance.nodes[0].families[0]
        return admit_graph(
            self.instance,
            MappingArtifactResolver(self.public_artifacts, self.private_artifacts),
            policy=AdmissionPolicyV1(writer_family=family),
        )

    def run_agent_inputs(self) -> dict[str, Any]:
        """Return byte-for-byte-equivalent logical inputs used by ``run_selected``."""
        return {
            "messages": _thaw(self.messages),
            "tools": list(self.tools),
            "followups": list(self.followups),
            **{
                name: self.budgets[name]
                for name in ("max_steps", "max_tool_calls", "max_read_tokens")
            },
        }

    def workspace_files(self) -> dict[str, str]:
        return _thaw(self.initial_files)

    def private_check_inputs(self) -> dict[str, Any]:
        return _thaw(self.check_package)

    def prose_selectors(self) -> list[dict[str, Any]]:
        return _thaw(self.prose)

    @property
    def workspace_max_total_bytes(self) -> int:
        return self.budgets["max_total_bytes"]

    def persist(self, store: Any) -> str:
        """Persist all referenced artifacts and the instance, preserving identities."""
        for identity, body in self.public_artifacts.items():
            if store.put_artifact(body) != identity:
                raise ValueError("public artifact identity changed during persistence")
        for identity, body in self.private_artifacts.items():
            if store.put_artifact(body, private=True) != identity:
                raise ValueError("private artifact identity changed during persistence")
        if store.persist(self.instance) != self.instance.identity():
            raise ValueError("graph identity changed during persistence")
        return self.instance.identity()


def compile_legacy_scenario(scenario: Mapping[str, Any]) -> LegacyGraphBundleV1:
    """Compile one loaded legacy scenario without changing its legacy projections."""
    if not isinstance(scenario, Mapping):
        raise TypeError("scenario must be an object")
    try:
        scenario_id = scenario["id"]
        family = scenario["family"]
        visible = copy.deepcopy(scenario["visible"])
        labels = copy.deepcopy(scenario["labels"])
    except KeyError as exc:
        raise ValueError(f"legacy scenario is missing {exc.args[0]}") from exc
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("legacy scenario id is required")
    if not isinstance(visible, dict) or not isinstance(labels, dict):
        raise TypeError("legacy visible and labels packages must be objects")
    expected_visible = {"brief", "initial_files", "followups", "tools", "budgets", "prose"}
    if set(visible) != expected_visible:
        raise ValueError("legacy visible package shape is unsupported")
    if not isinstance(visible["brief"], str) or not visible["brief"].strip():
        raise ValueError("legacy brief must be nonempty text")
    if not isinstance(visible["initial_files"], dict) or any(
        not isinstance(path, str) or not isinstance(text, str)
        for path, text in visible["initial_files"].items()
    ):
        raise TypeError("legacy initial_files must map paths to text")
    if not isinstance(visible["followups"], list) or any(
        not isinstance(text, str) for text in visible["followups"]
    ):
        raise TypeError("legacy followups must be an array of text")
    if not isinstance(visible["tools"], list) or any(
        not isinstance(name, str) for name in visible["tools"]
    ):
        raise TypeError("legacy tools must be an array of names")
    budget_names = {"max_steps", "max_tool_calls", "max_read_tokens", "max_total_bytes"}
    if not isinstance(visible["budgets"], dict) or set(visible["budgets"]) != budget_names:
        raise ValueError("legacy budgets are incomplete")
    if not isinstance(labels.get("checks"), list):
        raise ValueError("legacy private checks must be explicit")

    public: dict[str, Any] = {}
    private: dict[str, Any] = {}

    template_ref, template = _artifact({"kind": "legacy-single-writer-template", "schema": 1})
    public[template_ref] = template
    request_ref, request = _artifact(
        {"kind": "legacy-visible-brief", "schema": 1, "text": visible["brief"]}
    )
    public[request_ref] = request
    guard = GuardContractV1(kind="always")
    public[guard.identity()] = guard.to_dict()

    mandatory_refs: list[str] = []
    optional_refs: list[str] = []
    mandatory_ids: list[str] = []
    for check in labels["checks"]:
        if not isinstance(check, dict) or not isinstance(check.get("id"), str):
            raise ValueError("legacy check needs an id")
        required = check.get("required") is True
        contract = CheckContractV1(
            id=check["id"],
            evaluator_version="legacy-check-v1",
            applicability="node_exit_candidate",
            required=required,
            spec=check,
        )
        private[contract.identity()] = contract.to_dict()
        (mandatory_refs if required else optional_refs).append(contract.identity())
        if required:
            mandatory_ids.append(check["id"])

    followups = tuple(visible["followups"])
    script_ref = None
    if followups:
        script = ScriptContractV1(fixed_followups=followups)
        script_ref = script.identity()
        private[script_ref] = script.to_dict()
    interaction = InteractionContractV1(
        mode="scripted" if followups else "none",
        script_ref=script_ref,
        scripted_turns=len(followups),
    )
    budget = BudgetContractV1(
        **visible["budgets"],
        max_author_calls=len(followups),
        max_graph_hops=1,
        max_visits=1,
    )
    completion = CompletionContractV1(
        required_check_ids=tuple(mandatory_ids),
        required_script_turns=len(followups),
    )
    node_id = "legacy-writer"
    node_contract = NodeContractV1(
        node_id=node_id,
        node_kind="writer",
        entry=NodeEntryV1(
            request_ref=request_ref,
            context_policy="carry",
            tool_allowlist=tuple(visible["tools"]),
            family=family,
        ),
        interaction=interaction,
        budgets=budget,
        completion=completion,
        mandatory_checks=tuple(mandatory_refs),
        optional_checks=tuple(optional_refs),
    )
    public[node_contract.identity()] = node_contract.to_dict()
    edge = EdgeContractV1(
        edge_id="legacy-terminate",
        guard_ref=guard.identity(),
        target_node=None,
        effect="terminate",
        precedence=0,
    )
    instance = GraphInstanceV1(
        template_ref=template_ref,
        entry_node=node_id,
        nodes=(
            NodeSpecV1(
                id=node_id,
                kind="writer",
                families=(family,),
                entry_contract=node_contract.identity(),
                exits=(edge.to_dict(),),
            ),
        ),
        request_refs=(request_ref,),
        budgets={"max_graph_hops": 1},
    )
    return LegacyGraphBundleV1(
        instance=instance,
        public_artifacts=public,
        private_artifacts=private,
        initial_files=visible["initial_files"],
        messages=({"role": "user", "content": visible["brief"]},),
        tools=tuple(visible["tools"]),
        followups=followups,
        budgets=visible["budgets"],
        prose=tuple(visible["prose"]),
        check_package=labels,
    )
