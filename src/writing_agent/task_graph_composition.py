"""Composition root for sealed runtime adapters and the local publication service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from writing_agent.task_graph import (
    EnvironmentStateV1,
    canonical_json,
    load_canonical_json,
    validate_hash,
)
from writing_agent.task_graph_environment import EnvironmentTransactionService
from writing_agent.task_graph_local import (
    DeterministicEvaluator,
    LocalTextToolProvider,
    LocalWorkspaceEnvironment,
    ScriptedSampleBackend,
)
from writing_agent.task_graph_ports import (
    PreparedSamplingInput,
    RuntimeDependenciesV1,
    SampleResult,
)
from writing_agent.task_graph_record_contracts import GroupSpecV1


@dataclass(frozen=True)
class RuntimeSession:
    dependencies: RuntimeDependenciesV1
    publisher: EnvironmentTransactionService
    manifest_ref: str
    sealed_adapter_ref: str | None = None

    @classmethod
    def create(cls, store, rollout_id, entry_checkpoint_id, dependencies):
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
            EnvironmentTransactionService(store, rollout_id, entry_checkpoint_id),
            ref,
        )

    def bind(self, store, adapter_ref: str) -> RuntimeSession:
        if adapter_ref != self.manifest_ref or store.get_artifact(adapter_ref) != (
            self.dependencies.manifest().to_wire()
        ):
            raise ValueError("sealed adapter manifest differs from executing runtime")
        return RuntimeSession(self.dependencies, self.publisher, self.manifest_ref, adapter_ref)

    def require_seal(self, adapter_ref: str) -> None:
        if (
            self.sealed_adapter_ref != adapter_ref
            or self.manifest_ref != adapter_ref
            or self.dependencies.manifest().identity() != adapter_ref
        ):
            raise ValueError("sealed adapter manifest differs from executing runtime")

    def require_member_seal(self, store, state: EnvironmentStateV1) -> None:
        """Bind a started group member to its immutable group policy before effects."""
        lineage = state.position["lineage_id"]
        if not lineage.startswith("grp-"):
            return
        seed = store.get_artifact(state.rng_ref)
        if (
            not isinstance(seed, dict)
            or seed.get("record_type") != "GroupMemberSeedsV1"
            or seed.get("member_id") != lineage
            or not isinstance(seed.get("group_id"), str)
        ):
            raise ValueError("group member lacks its sealed seed witness")
        validate_hash(seed["group_id"])
        path = store.root / "groups" / seed["group_id"] / "spec.json"
        try:
            spec = GroupSpecV1.from_dict(load_canonical_json(path.read_bytes()))
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("group member lacks a valid sealed group receipt") from exc
        if spec.group_id != seed["group_id"] or lineage not in {
            member.member_id for member in spec.members
        }:
            raise ValueError("group member differs from sealed group receipt")
        adapter_ref = spec.policy["adapter_ref"]
        declared = store.get_artifact(adapter_ref)
        if isinstance(declared, dict) and declared.get("record_type") == "RuntimeManifestV1":
            self.require_seal(adapter_ref)
        elif self.sealed_adapter_ref is not None:
            # A bound runtime cannot execute against an untyped legacy adapter pin.
            self.require_seal(adapter_ref)


def local_unbound_session(store, rollout_id, entry_checkpoint_id) -> RuntimeSession:
    """Explicit legacy/offline direct-submit composition; never claims group binding."""
    tools = LocalTextToolProvider()
    dependencies = RuntimeDependenciesV1(
        ScriptedSampleBackend(()),
        LocalWorkspaceEnvironment(tools),
        tools,
        DeterministicEvaluator(),
    )
    return RuntimeSession.create(store, rollout_id, entry_checkpoint_id, dependencies)


class RuntimeRunner:
    """Prepare the current projected context, invoke a backend, commit its sample."""

    def __init__(self, writer, session: RuntimeSession):
        if writer.session is not session:
            raise ValueError("runner and writer must share the same runtime session")
        if session.sealed_adapter_ref is None:
            raise ValueError("sampling runner requires a bound runtime session")
        self.writer = writer
        self.session = session

    def sample(self, runtime, *, request_extras: dict[str, Any] | None = None):
        self.session.require_seal(self.session.sealed_adapter_ref)
        self.writer.validate_runtime(runtime)
        request = {
            "messages": [message.to_dict() for message in runtime.context.messages],
            **(request_extras or {}),
        }
        request_json = canonical_json(request)
        request = json.loads(request_json)
        prepared_ref = self.writer.prepare_verified_messages(runtime, request)
        prepared = self.writer.store.get_artifact(prepared_ref, expected_domain="payload")
        input_value = PreparedSamplingInput(
            request_ref=prepared["payload_ref"],
            prepared_request_ref=prepared_ref,
            context_content_hash=runtime.context.content_hash,
            context_revision_ref=runtime.context.identity(),
            messages_json=canonical_json(request["messages"]),
            tools_json=canonical_json(runtime.context.tools),
            rendering_json=canonical_json(runtime.context.rendering),
            request_json=request_json,
        )
        result = self.session.dependencies.sampling.sample(input_value)
        if not isinstance(result, SampleResult):
            raise TypeError("sample backend must return SampleResult")
        trace = dict(result.trace or {})
        if result.logprobs is not None:
            if "per_token_logprobs_ref" in trace or "per_token_logprobs" in trace:
                raise ValueError("sample supplied both binary and ref-only logprobs")
            tokens = trace.get("generated_token_ids")
            if (
                not isinstance(tokens, list)
                or any(type(token) is not int or token < 0 for token in tokens)
                or len(tokens) != result.logprobs.shape[0]
            ):
                raise ValueError("binary logprobs must align with generated token IDs")
            trace["per_token_logprobs_ref"] = self.writer.store.put_bytes_artifact(
                result.logprobs.data
            )
            trace["per_token_logprobs_codec"] = result.logprobs.codec
            trace["per_token_logprobs_shape"] = list(result.logprobs.shape)
        return self.writer.submit_action(
            runtime,
            result.message,
            prepared_request_ref=prepared_ref,
            raw_output=result.raw_output,
            usage=result.usage,
            trace=trace or None,
        )
