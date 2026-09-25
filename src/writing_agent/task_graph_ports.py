"""Immutable runtime descriptors and remote-capable sampling/execution contracts.

Ports describe capabilities and effects, not local persistence. Composition owns the
transaction publisher and seals these descriptors before running a member.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from writing_agent.task_graph import canonical_json, domain_hash
from writing_agent.task_graph_evaluation import EvaluationEvidenceV1, EvaluationRequestV1


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


@dataclass(frozen=True)
class PreparedSamplingInput:
    request_ref: str
    prepared_request_ref: str
    context_content_hash: str
    context_revision_ref: str
    messages_json: str
    tools_json: str
    rendering_json: str


@dataclass(frozen=True)
class SampleResult:
    message: Mapping[str, Any]
    raw_output: str | bytes | None = None
    usage: Mapping[str, Any] | None = None
    trace: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, Mapping):
            raise TypeError("sample message must be a parsed object")
        if self.raw_output is not None and not isinstance(self.raw_output, (str, bytes)):
            raise TypeError("sample raw output must be exact text or bytes")
        for field in ("message", "usage", "trace"):
            value = getattr(self, field)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise TypeError(f"sample {field} must be a parsed object")
                object.__setattr__(self, field, json.loads(canonical_json(value)))


class SampleBackend(Protocol):
    descriptor: PortDescriptorV1

    def sample(self, prepared: PreparedSamplingInput) -> SampleResult: ...


@dataclass(frozen=True)
class ToolManifest:
    schemas_json: str
    capabilities: tuple[str, ...] = ("text_workspace",)

    def __post_init__(self) -> None:
        schemas = json.loads(self.schemas_json)
        if (
            not isinstance(schemas, list)
            or canonical_json(schemas) != self.schemas_json
            or not set(self.capabilities) <= {"text_workspace"}
        ):
            raise ValueError("tool manifest must declare only supported text workspace tools")

    def schemas(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self.schemas_json))


@dataclass(frozen=True)
class EnvironmentSpec:
    max_file_bytes: int
    max_workspace_bytes: int


@dataclass(frozen=True)
class EnvironmentHandle:
    identity: str


@dataclass(frozen=True)
class EnvironmentSnapshot:
    files_json: str

    @classmethod
    def from_files(cls, files: Mapping[str, str]) -> EnvironmentSnapshot:
        return cls(canonical_json(dict(files)))

    def files(self) -> dict[str, str]:
        return json.loads(self.files_json)


@dataclass(frozen=True)
class EnvironmentAction:
    name: str
    arguments_json: str

    @classmethod
    def from_arguments(cls, name: str, arguments: Mapping[str, Any]) -> EnvironmentAction:
        return cls(name, canonical_json(dict(arguments)))

    def arguments(self) -> dict[str, Any]:
        return json.loads(self.arguments_json)


@dataclass(frozen=True)
class EnvironmentResult:
    observation: Mapping[str, Any]
    snapshot: EnvironmentSnapshot
    infrastructure: str = "ok"

    def __post_init__(self) -> None:
        if self.infrastructure not in {"ok", "transient_failure", "permanent_failure"}:
            raise ValueError("unknown infrastructure classification")


class ExecutionEnvironment(Protocol):
    descriptor: PortDescriptorV1

    def tool_manifest(
        self, allowlist: tuple[str, ...], interaction_policy: Any = None
    ) -> ToolManifest: ...

    def execute(
        self,
        spec: EnvironmentSpec,
        handle: EnvironmentHandle,
        snapshot: EnvironmentSnapshot,
        action: EnvironmentAction,
    ) -> EnvironmentResult: ...


class ToolProvider(Protocol):
    descriptor: PortDescriptorV1

    def tool_manifest(
        self, allowlist: tuple[str, ...], interaction_policy: Any = None
    ) -> ToolManifest: ...

    def execute(
        self, spec: EnvironmentSpec, snapshot: EnvironmentSnapshot, action: EnvironmentAction
    ) -> EnvironmentResult: ...


class Evaluator(Protocol):
    descriptor: PortDescriptorV1
    family: str

    def evaluate(self, request: EvaluationRequestV1) -> EvaluationEvidenceV1: ...


@dataclass(frozen=True)
class RuntimeDependenciesV1:
    sampling: SampleBackend
    environment: ExecutionEnvironment
    tools: ToolProvider
    evaluator: Evaluator

    def __post_init__(self) -> None:
        for role in ("sampling", "environment", "tools", "evaluator"):
            if getattr(self, role).descriptor.role != role:
                raise ValueError(f"runtime {role} has a descriptor for another port")

    def manifest(self) -> RuntimeManifestV1:
        return RuntimeManifestV1(
            (
                self.sampling.descriptor,
                self.environment.descriptor,
                self.tools.descriptor,
                self.evaluator.descriptor,
            )
        )
