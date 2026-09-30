"""Pure shared contracts for native task-graph sampling and runtime binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from writing_agent.task_graph_errors import AdapterContractError
from writing_agent.task_graph_records import (
    NATIVE_RUNTIME_CAPABILITIES,
    RuntimeManifestV1,
    RuntimeManifestV2,
    WriterTurnV2,
)

NATIVE_TRAINING_CAPABILITIES = NATIVE_RUNTIME_CAPABILITIES
USAGE_REPORTING_CAPABILITY = "usage_reporting"
NATIVE_TOKEN_LEDGER_CAPABILITY = "native_token_ledger"
SAMPLED_LOGPROBS_CAPABILITY = "sampled_logprobs"


@dataclass(frozen=True)
class NativeSamplingBudget:
    """Public allocation inputs for a native sampler, never the full runtime ledger."""

    remaining_generated_tokens: int | None
    max_context_tokens: int | None

    def __post_init__(self) -> None:
        for name in ("remaining_generated_tokens", "max_context_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")


@dataclass(frozen=True)
class NativeSamplingHistory:
    """Committed native token prefix for continuing a sampled writer turn."""

    turn: WriterTurnV2
    input_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.turn, WriterTurnV2):
            raise TypeError("native sampling history requires a committed WriterTurnV2")
        for name in ("input_token_ids", "generated_token_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                type(token) is not int or token < 0 for token in values
            ):
                raise TypeError(f"native history {name} must be a tuple of token IDs")
        if len(self.input_token_ids) != self.turn.input_token_count:
            raise ValueError("native history input count differs from its turn")
        if len(self.generated_token_ids) != self.turn.generated_token_count:
            raise ValueError("native history generated count differs from its turn")


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
