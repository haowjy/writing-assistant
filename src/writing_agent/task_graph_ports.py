"""Immutable runtime descriptors and remote-capable sampling/execution contracts.

Ports describe capabilities and effects, not local persistence. Composition owns the
transaction publisher and seals these descriptors before running a member.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from writing_agent.task_graph import canonical_json, validate_hash
from writing_agent.task_graph_errors import AdapterContractError
from writing_agent.task_graph_evaluation import EvaluationEvidenceV1, EvaluationRequestV1
from writing_agent.task_graph_records import (
    NATIVE_RUNTIME_CAPABILITIES,
    DecodingDescriptorV1,
    RendererDescriptorV1,
    RuntimeManifestV1,
    RuntimeManifestV2,
    RuntimePortDescriptorV1,
    TokenizerDescriptorV1,
)
from writing_agent.task_graph_sampling import NativeSamplingBudget, NativeSamplingHistory

USAGE_REPORTING_CAPABILITY = "usage_reporting"
NATIVE_TOKEN_LEDGER_CAPABILITY = "native_token_ledger"
SAMPLED_LOGPROBS_CAPABILITY = "sampled_logprobs"
NATIVE_TRAINING_CAPABILITIES = NATIVE_RUNTIME_CAPABILITIES


@dataclass(frozen=True)
class PortDescriptorV1:
    role: str
    implementation: str
    version: str
    configuration_json: str = "{}"
    capabilities: tuple[str, ...] = ()

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
        if "capabilities" in configuration:
            raise ValueError("port capabilities must use the descriptor field")
        if (
            not isinstance(self.capabilities, tuple)
            or self.capabilities != tuple(sorted(set(self.capabilities)))
            or set(self.capabilities) - NATIVE_RUNTIME_CAPABILITIES
            or (self.capabilities and self.role != "sampling")
        ):
            raise ValueError("port capabilities must be supported, sorted, and role-specific")

    def to_wire(self) -> dict[str, Any]:
        return self.to_record().to_wire()

    def to_record(self) -> RuntimePortDescriptorV1:
        configuration = json.loads(self.configuration_json)
        if self.capabilities:
            configuration["capabilities"] = list(self.capabilities)
        return RuntimePortDescriptorV1(
            schema=1,
            role=self.role,
            implementation=self.implementation,
            version=self.version,
            configuration=configuration,
        )

    def identity(self) -> str:
        return self.to_record().identity()


@dataclass(frozen=True)
class PreparedSamplingInput:
    context_content_hash: str
    context_revision_ref: str
    writer_seed: int | None
    model_ref: str | None
    behavior_policy_ref: str | None
    decoding_ref: str | None
    tokenizer_ref: str | None
    template_ref: str | None
    messages_json: str
    tools_json: str
    rendering_json: str
    native_sampling_budget: NativeSamplingBudget | None = None
    adapter_ref: str | None = None
    decision_ordinal: int | None = None
    native_history: NativeSamplingHistory | None = None

    def __post_init__(self) -> None:
        validate_hash(self.context_content_hash)
        validate_hash(self.context_revision_ref)
        if self.writer_seed is not None and (
            type(self.writer_seed) is not int or self.writer_seed < 0
        ):
            raise ValueError("prepared sampling seed must be a nonnegative integer")
        for reference in (
            self.model_ref,
            self.behavior_policy_ref,
            self.decoding_ref,
            self.tokenizer_ref,
            self.template_ref,
            self.adapter_ref,
        ):
            if reference is not None:
                validate_hash(reference)
        if self.decision_ordinal is not None and (
            type(self.decision_ordinal) is not int or self.decision_ordinal < 0
        ):
            raise ValueError("prepared decision ordinal must be a nonnegative integer")
        if self.native_history is not None and not isinstance(
            self.native_history, NativeSamplingHistory
        ):
            raise TypeError("prepared native history must be typed committed sampling evidence")


@dataclass(frozen=True)
class BinaryLogprobEvidence:
    data: bytes
    codec: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.data, bytes)
            or self.codec != "f32-le"
            or not isinstance(self.shape, tuple)
            or len(self.shape) != 1
            or type(self.shape[0]) is not int
            or self.shape[0] < 0
            or len(self.data) != 4 * self.shape[0]
        ):
            raise ValueError("logprob evidence requires one-dimensional f32-le bytes")


@dataclass(frozen=True)
class SampleResult:
    message: Mapping[str, Any]
    raw_output: str | bytes | None = None
    usage: Mapping[str, Any] | None = None
    trace: Mapping[str, Any] | None = None
    logprobs: BinaryLogprobEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, Mapping):
            raise AdapterContractError("sample message must be a parsed object")
        if self.raw_output is not None and not isinstance(self.raw_output, (str, bytes)):
            raise AdapterContractError("sample raw output must be exact text or bytes")
        if self.logprobs is not None and not isinstance(self.logprobs, BinaryLogprobEvidence):
            raise AdapterContractError("sample logprobs must be typed binary evidence")

        # Tool-call values are model output, not adapter metadata. Leave their nested
        # values for intake_message, which records noncanonical values as an invalid call.
        try:
            message = dict(self.message)
        except (TypeError, ValueError) as exc:
            raise AdapterContractError("sample message is not a readable object") from exc
        object.__setattr__(self, "message", message)
        for field in ("usage", "trace"):
            value = getattr(self, field)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise AdapterContractError(f"sample {field} must be a parsed object")
            try:
                copied = json.loads(canonical_json(value))
            except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
                raise AdapterContractError(f"sample {field} is not canonical JSON") from exc
            object.__setattr__(self, field, copied)


@dataclass(frozen=True)
class SampleResultV2:
    """Port result with typed token and sampled-logprob evidence, before persistence."""

    message: Mapping[str, Any]
    input_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    usage: Mapping[str, Any]
    logprobs: BinaryLogprobEvidence
    termination: Mapping[str, Any]
    sampling_pins: Mapping[str, Any]
    raw_output: str | bytes | None = None
    trace: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, Mapping):
            raise AdapterContractError("sample message must be a parsed object")
        for field in ("input_token_ids", "generated_token_ids"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(
                type(token) is not int or token < 0 for token in values
            ):
                raise AdapterContractError(f"sample {field} must be a tuple of token IDs")
        if self.logprobs.shape != (len(self.generated_token_ids),):
            raise AdapterContractError("sample logprob shape differs from generated token IDs")
        if self.raw_output is not None and not isinstance(self.raw_output, (str, bytes)):
            raise AdapterContractError("sample raw output must be exact text or bytes")

        for field in ("message", "usage", "termination", "sampling_pins", "trace"):
            value = getattr(self, field)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise AdapterContractError(f"sample {field} must be a parsed object")
            try:
                copied = json.loads(canonical_json(value))
            except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
                raise AdapterContractError(f"sample {field} is not canonical JSON") from exc
            object.__setattr__(self, field, copied)


class SampleBackend(Protocol):
    descriptor: PortDescriptorV1

    def sample(self, prepared: PreparedSamplingInput) -> SampleResult: ...


class SampleBackendV2(Protocol):
    descriptor: PortDescriptorV1
    manifest_descriptors: tuple[RendererDescriptorV1, TokenizerDescriptorV1, DecodingDescriptorV1]

    def sample(self, prepared: PreparedSamplingInput) -> SampleResultV2: ...


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


class ExecutionInfrastructureError(RuntimeError):
    """Remote execution failed before a tool observation could be committed."""

    def __init__(self, classification: str):
        self.classification = classification
        super().__init__(f"execution environment infrastructure {classification}")


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
    sampling: SampleBackend | SampleBackendV2
    environment: ExecutionEnvironment
    tools: ToolProvider
    evaluator: Evaluator

    def __post_init__(self) -> None:
        for role in ("sampling", "environment", "tools", "evaluator"):
            if getattr(self, role).descriptor.role != role:
                raise ValueError(f"runtime {role} has a descriptor for another port")

    def manifest(self) -> RuntimeManifestV1 | RuntimeManifestV2:
        ports = tuple(
            sorted(
                (
                    port.descriptor.to_record()
                    for port in (self.sampling, self.environment, self.tools, self.evaluator)
                ),
                key=lambda item: item.role,
            )
        )
        native = getattr(self.sampling, "manifest_descriptors", None)
        if native is None:
            return RuntimeManifestV1(schema=1, ports=ports)
        if (
            not isinstance(native, tuple)
            or len(native) != 3
            or not isinstance(native[0], RendererDescriptorV1)
            or not isinstance(native[1], TokenizerDescriptorV1)
            or not isinstance(native[2], DecodingDescriptorV1)
        ):
            raise ValueError("native sampler manifest descriptors are invalid")
        renderer, tokenizer, decoding = native
        return RuntimeManifestV2(
            schema=2,
            ports=ports,
            capabilities=tuple(sorted(NATIVE_RUNTIME_CAPABILITIES)),
            renderer=renderer,
            tokenizer=tokenizer,
            decoding=decoding,
        )


def manifest_sampling_capabilities(
    manifest: RuntimeManifestV1 | RuntimeManifestV2,
) -> frozenset[str]:
    """Read the capabilities that both the manifest and sampling port declare."""
    if not isinstance(manifest, (RuntimeManifestV1, RuntimeManifestV2)):
        raise AdapterContractError("runtime manifest has an unsupported record type")
    sampler = next((port for port in manifest.ports if port.role == "sampling"), None)
    if sampler is None:
        raise AdapterContractError("runtime manifest has no sampling port")
    capabilities = sampler.configuration.get("capabilities", ())
    known = (
        NATIVE_TRAINING_CAPABILITIES
        if isinstance(manifest, RuntimeManifestV2)
        else {USAGE_REPORTING_CAPABILITY}
    )
    if (
        type(capabilities) not in (tuple, list)
        or any(type(capability) is not str for capability in capabilities)
        or tuple(capabilities) != tuple(sorted(set(capabilities)))
        or set(capabilities) - known
    ):
        raise AdapterContractError("sampling port capabilities are invalid")
    port_capabilities = frozenset(capabilities)
    if isinstance(manifest, RuntimeManifestV1):
        return port_capabilities

    declared = manifest.capabilities
    if (
        type(declared) not in (tuple, list)
        or any(type(capability) is not str for capability in declared)
        or tuple(declared) != tuple(sorted(set(declared)))
        or set(declared) - NATIVE_TRAINING_CAPABILITIES
    ):
        raise AdapterContractError("manifest capabilities are invalid")
    return frozenset(declared) & port_capabilities


def manifest_supports_usage_reporting(manifest: RuntimeManifestV1 | RuntimeManifestV2) -> bool:
    """Whether the sealed sampling descriptor promises token usage evidence."""
    return USAGE_REPORTING_CAPABILITY in manifest_sampling_capabilities(manifest)


def require_native_manifest_binding(
    manifest: RuntimeManifestV1 | RuntimeManifestV2,
    manifest_ref: str,
    *,
    policy: Mapping[str, str] | None,
    rendering: Mapping[str, Any],
    require_capabilities: bool = True,
) -> None:
    """Fail closed unless a V2 manifest pins the group's rendering and decoding."""
    if not isinstance(manifest, RuntimeManifestV2):
        raise AdapterContractError("native training requires RuntimeManifestV2")

    declared_capabilities = manifest_sampling_capabilities(manifest)
    if require_capabilities and NATIVE_TRAINING_CAPABILITIES - declared_capabilities:
        raise AdapterContractError("native sampling manifest lacks required capabilities")
    if require_capabilities and manifest.decoding.processors:
        raise AdapterContractError("native training requires neutral decoding processors")

    if policy is not None:
        expected_policy = {
            "template_ref": manifest.renderer.template_ref,
            "tokenizer_ref": manifest.tokenizer.identity(),
            "decoding_ref": manifest.decoding.identity(),
            "adapter_ref": manifest_ref,
        }
        for field, expected in expected_policy.items():
            if policy.get(field) != expected:
                raise AdapterContractError(f"native group {field} differs from its manifest pin")

    expected_rendering = {
        "template_ref": manifest.renderer.template_ref,
        "tokenizer_ref": manifest.renderer.tokenizer_ref,
        "tool_schema_ref": manifest.renderer.tool_schema_ref,
    }
    for field, expected in expected_rendering.items():
        if rendering.get(field) != expected:
            raise AdapterContractError(f"native renderer {field} differs from the context root")
