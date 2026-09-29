"""Composition root for sealed runtime adapter manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass

from writing_agent.task_graph_errors import AdapterContractError
from writing_agent.task_graph_local import LocalWorkspaceEnvironment
from writing_agent.task_graph_ports import (
    USAGE_REPORTING_CAPABILITY,
    RuntimeDependenciesV1,
    manifest_supports_usage_reporting,
    require_native_manifest_binding,
)
from writing_agent.task_graph_records import RuntimeManifestV2
from writing_agent.task_graph_transition import LineageView


@dataclass(frozen=True)
class RuntimeSession:
    dependencies: RuntimeDependenciesV1
    manifest_ref: str
    sealed_adapter_ref: str | None = None

    @classmethod
    def create(cls, store, dependencies: RuntimeDependenciesV1) -> RuntimeSession:
        if (
            isinstance(dependencies.environment, LocalWorkspaceEnvironment)
            and dependencies.environment.provider is not dependencies.tools
        ):
            raise ValueError("local environment and declared tool provider differ")
        evaluator_family = json.loads(dependencies.evaluator.descriptor.configuration_json).get(
            "family"
        )
        if evaluator_family != dependencies.evaluator.family:
            raise ValueError("evaluator descriptor does not declare its evidence family")
        manifest = dependencies.manifest()
        ref = store.put_artifact(manifest.to_wire())
        if ref != manifest.identity():
            raise ValueError("runtime manifest persistence changed identity")
        return cls(
            dependencies,
            ref,
        )

    def bind(self, store, adapter_ref: str) -> RuntimeSession:
        if adapter_ref != self.manifest_ref or store.get_artifact(adapter_ref) != (
            self.dependencies.manifest().to_wire()
        ):
            raise ValueError("sealed adapter manifest differs from executing runtime")
        return RuntimeSession(self.dependencies, self.manifest_ref, adapter_ref)

    def require_seal(
        self,
        adapter_ref: str,
        *,
        token_limited: bool = False,
        training_mode: str | None = None,
        policy=None,
        rendering=None,
    ) -> None:
        manifest = self.dependencies.manifest()
        if (
            self.sealed_adapter_ref != adapter_ref
            or self.manifest_ref != adapter_ref
            or manifest.identity() != adapter_ref
        ):
            raise AdapterContractError("sealed adapter manifest differs from executing runtime")
        if training_mode == "native" and not isinstance(manifest, RuntimeManifestV2):
            raise AdapterContractError("native training requires RuntimeManifestV2")
        if isinstance(manifest, RuntimeManifestV2):
            if rendering is None:
                raise AdapterContractError("V2 runtime binding requires context pins")
            require_native_manifest_binding(
                manifest,
                adapter_ref,
                policy=policy,
                rendering=rendering,
                require_capabilities=training_mode == "native",
            )
        if token_limited and not manifest_supports_usage_reporting(manifest):
            raise AdapterContractError(
                f"token-limited entry requires {USAGE_REPORTING_CAPABILITY} capability"
            )

    def require_member_seal(self, view: LineageView) -> None:
        """Check the gate-verified group contract against the executing manifest."""
        spec = view.group
        if spec is None:
            manifest = self.dependencies.manifest()
            if isinstance(manifest, RuntimeManifestV2):
                require_native_manifest_binding(
                    manifest,
                    self.manifest_ref,
                    policy=None,
                    rendering=view.context.rendering,
                    require_capabilities=False,
                )
            return
        lineage = view.state.position["lineage_id"]
        if lineage not in {member.member_id for member in spec.members}:
            raise ValueError("group member differs from its verified group contract")
        budget = view.node.contract.budget_contract
        token_limited = bool(budget.usage_charged_limits())
        self.require_seal(
            spec.policy["adapter_ref"],
            token_limited=token_limited,
            training_mode=spec.training_mode,
            policy=spec.policy,
            rendering=view.context.rendering,
        )
