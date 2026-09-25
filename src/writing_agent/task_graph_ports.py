"""Runtime ports and immutable composition manifest for task-graph experiments.

These contracts inject policy-preserving implementations; admission and semantic
replay still validate persisted evidence independently. A port cannot grant native
eligibility or make an unadmitted check program authoritative by itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from writing_agent.task_graph import ContextRevisionV1, canonical_json, domain_hash
from writing_agent.task_graph_environment import EnvironmentBatch, WriterStepV1
from writing_agent.task_graph_sampling import PreparedRequestV1, SamplingEvidenceV1
from writing_agent.task_graph_store import RuntimeHandle


@dataclass(frozen=True)
class PortDescriptorV1:
    role: str
    implementation: str
    version: str
    configuration_json: str = "{}"

    def __post_init__(self) -> None:
        if self.role not in {"sampling", "environment", "tools", "evaluator"}:
            raise ValueError("unknown runtime port role")
        if not self.implementation or not self.version:
            raise ValueError("runtime port needs implementation and version")
        configuration = json.loads(self.configuration_json)
        if (
            not isinstance(configuration, dict)
            or canonical_json(configuration) != self.configuration_json
        ):
            raise ValueError("port configuration must be canonical JSON object")

    def to_wire(self) -> dict[str, Any]:
        return {
            "record_type": "RuntimePortDescriptorV1",
            "schema": 1,
            "role": self.role,
            "implementation": self.implementation,
            "version": self.version,
            "configuration": json.loads(self.configuration_json),
        }

    def identity(self) -> str:
        return domain_hash("payload", self.to_wire())


@dataclass(frozen=True)
class RuntimeManifestV1:
    descriptors: tuple[PortDescriptorV1, ...]

    def __post_init__(self) -> None:
        if {item.role for item in self.descriptors} != {
            "sampling",
            "environment",
            "tools",
            "evaluator",
        } or len(self.descriptors) != 4:
            raise ValueError("runtime manifest requires one descriptor for each port")

    def to_wire(self) -> dict[str, Any]:
        return {
            "record_type": "RuntimeManifestV1",
            "schema": 1,
            "ports": [item.to_wire() for item in sorted(self.descriptors, key=lambda d: d.role)],
        }

    def identity(self) -> str:
        return domain_hash("payload", self.to_wire())


class SamplingHarness(Protocol):
    descriptor: PortDescriptorV1

    def prepared_request(
        self, context: ContextRevisionV1, payload_ref: str, *, verified: bool
    ) -> PreparedRequestV1: ...

    def evidence(
        self,
        action_id: str,
        context: ContextRevisionV1,
        request_ref: str | None,
        prepared_request_ref: str | None,
        raw_output_ref: str | None,
        usage: Mapping[str, Any],
        adapter_trace: Mapping[str, Any] | None,
    ) -> SamplingEvidenceV1: ...


class ExecutionEnvironment(Protocol):
    descriptor: PortDescriptorV1

    def head(self, runtime: RuntimeHandle) -> tuple[str | None, str | None]: ...

    def runtime_log(self, state: Any) -> list[dict[str, Any]]: ...

    def batch(self, runtime: RuntimeHandle, *, restore_prefix: str) -> EnvironmentBatch: ...

    def publish_record(
        self, runtime: RuntimeHandle, kind: str, actor: str, **kwargs: Any
    ) -> WriterStepV1: ...


@dataclass(frozen=True)
class ToolExecution:
    observation: dict[str, Any]
    files: dict[str, str]


class ToolProvider(Protocol):
    descriptor: PortDescriptorV1

    def schemas(
        self, allowlist: tuple[str, ...], interaction_policy: Any = None
    ) -> tuple[dict[str, Any], ...]: ...

    def execute(
        self,
        files: Mapping[str, str],
        name: str,
        arguments: dict[str, str],
        *,
        stage_parent: Path,
        max_file_bytes: int,
        max_workspace_bytes: int,
    ) -> ToolExecution: ...


class Evaluator(Protocol):
    descriptor: PortDescriptorV1

    def evaluate(self, check: Any, files: Mapping[str, str]) -> tuple[str, dict]: ...


@dataclass(frozen=True)
class RuntimeDependenciesV1:
    sampling: SamplingHarness
    environment: ExecutionEnvironment
    tools: ToolProvider
    evaluator: Evaluator

    def manifest(self) -> RuntimeManifestV1:
        return RuntimeManifestV1(
            (
                self.sampling.descriptor,
                self.environment.descriptor,
                self.tools.descriptor,
                self.evaluator.descriptor,
            )
        )


def local_runtime_dependencies(
    store, rollout_id: str, entry_checkpoint_id: str
) -> RuntimeDependenciesV1:
    """Current offline composition root; no model, network, or shell capability."""
    from writing_agent.task_graph_environment import EnvironmentTransactionService
    from writing_agent.task_graph_local import (
        DeterministicEvaluator,
        LocalSamplingHarness,
        LocalTextToolProvider,
    )

    return RuntimeDependenciesV1(
        sampling=LocalSamplingHarness(),
        environment=EnvironmentTransactionService(store, rollout_id, entry_checkpoint_id),
        tools=LocalTextToolProvider(),
        evaluator=DeterministicEvaluator(),
    )
