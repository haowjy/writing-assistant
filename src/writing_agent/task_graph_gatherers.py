"""Port adapters that turn allowlisted inputs into typed transition records.

Gatherers do not receive lineage views or publish events. Adapter responses are
checked here; ``RolloutEnvironment.commit`` remains the recording boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from writing_agent.task_graph import canonical_json
from writing_agent.task_graph_calls import intake_message
from writing_agent.task_graph_contracts import CheckContractV1
from writing_agent.task_graph_environment import AuthorInput, CheckInput, SamplerInput, ToolInput
from writing_agent.task_graph_errors import AdapterContractError
from writing_agent.task_graph_evaluation import (
    FAMILIES,
    EvaluationEvidenceV1,
    EvaluationRequestV1,
)
from writing_agent.task_graph_ports import (
    EnvironmentAction,
    EnvironmentResult,
    EnvironmentSnapshot,
    EnvironmentSpec,
    Evaluator,
    ExecutionInfrastructureError,
    PreparedSamplingInput,
    SampleBackend,
    SampleResult,
    ToolProvider,
)
from writing_agent.task_graph_records import (
    AuthorReplyV1,
    EvaluatorResultV1,
    ToolObservationV1,
    WriterTurnV1,
)
from writing_agent.task_graph_sampling import ArtifactSink, persist_logprob_trace
from writing_agent.task_graph_scripted import scripted_author_reply


class SamplingRunner:
    """Run one sampling port call and produce its fully bound writer-turn input."""

    def __init__(
        self,
        artifacts: ArtifactSink,
        backend: SampleBackend,
        input_observer: Callable[[SamplerInput], None] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.backend = backend
        self.input_observer = input_observer

    def turn(self, port: SamplerInput) -> WriterTurnV1:
        if not isinstance(port, SamplerInput):
            raise TypeError("sampling runner requires SamplerInput")
        if self.input_observer is not None:
            self.input_observer(port)
        prepared = PreparedSamplingInput(
            context_content_hash=port.context_content_hash,
            context_revision_ref=port.context_revision_ref,
            writer_seed=port.writer_seed,
            model_ref=port.model_ref,
            behavior_policy_ref=port.behavior_policy_ref,
            decoding_ref=port.decoding_ref,
            tokenizer_ref=port.tokenizer_ref,
            template_ref=port.template_ref,
            messages_json=canonical_json([message.to_dict() for message in port.messages]),
            tools_json=canonical_json(port.tools),
            rendering_json=canonical_json(port.rendering),
            native_sampling_budget=port.native_sampling_budget,
        )
        result = self.backend.sample(prepared)
        if not isinstance(result, SampleResult):
            raise AdapterContractError("sample backend must return SampleResult")
        message = result.message
        if message.get("role") != "assistant":
            raise AdapterContractError("sampled message must have assistant role")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise AdapterContractError("sampled assistant content must be text or null")
        if not isinstance(message.get("tool_calls", []), list):
            raise AdapterContractError("sampled tool_calls must be an array")
        usage = dict(result.usage or {})
        trace = dict(result.trace or {})
        try:
            persist_logprob_trace(self.artifacts, result, trace)
        except ValueError as exc:
            raise AdapterContractError("binary logprobs do not align with sampled tokens") from exc

        raw_output_ref = None
        if result.raw_output is not None:
            raw_output_ref = (
                self.artifacts.put_bytes_artifact(result.raw_output)
                if isinstance(result.raw_output, bytes)
                else self.artifacts.put_artifact(result.raw_output)
            )
        try:
            turn = WriterTurnV1(
                action_id=port.action_id,
                context_revision_ref=port.context_revision_ref,
                raw_output_ref=raw_output_ref,
                usage=usage,
                adapter_trace=trace or None,
                message=intake_message(dict(message)),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise AdapterContractError("sample response violates the writer-turn contract") from exc
        return turn


class ToolRunner:
    """Execute one file-tool input through the existing tool-provider port."""

    def __init__(self, provider: ToolProvider) -> None:
        self.provider = provider

    def observe(self, port: ToolInput) -> ToolObservationV1:
        if not isinstance(port, ToolInput):
            raise TypeError("tool runner requires ToolInput")
        call_id = port.queue_entry["call_id"]
        if not port.dispatch_permitted:
            return ToolObservationV1(call_id, None)
        name = port.queue_entry["name"]
        arguments = dict(port.queue_entry["arguments"])
        before = dict(port.files)
        spec = EnvironmentSpec(port.tool_spec.max_file_bytes, port.tool_spec.max_workspace_bytes)
        try:
            result = self.provider.execute(
                spec,
                EnvironmentSnapshot.from_files(before),
                EnvironmentAction.from_arguments(name, arguments),
            )
            if not isinstance(result, EnvironmentResult):
                raise AdapterContractError("tool provider must return EnvironmentResult")
            if result.infrastructure != "ok":
                raise ExecutionInfrastructureError(result.infrastructure)
            observation = dict(result.observation)
            after = result.snapshot.files()
            effect = {
                path: {"before": before.get(path), "after": after.get(path)}
                for path in sorted(set(before) | set(after))
                if before.get(path) != after.get(path)
            }
            return ToolObservationV1(
                call_id,
                {
                    "spec": {
                        "max_file_bytes": port.tool_spec.max_file_bytes,
                        "max_workspace_bytes": port.tool_spec.max_workspace_bytes,
                    },
                    "observation": observation,
                    "effect": effect,
                },
            )
        except ExecutionInfrastructureError:
            raise
        except AdapterContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterContractError("tool response violates its effect contract") from exc


class ScriptedAuthorSource:
    """Resolve one authorized scripted author request without a live model."""

    def reply(self, port: AuthorInput) -> AuthorReplyV1:
        if not isinstance(port, AuthorInput):
            raise TypeError("scripted author source requires AuthorInput")
        try:
            return scripted_author_reply(
                port.script,
                port.request,
                port.request_ref,
                port.decisions,
                port.disclosures,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterContractError(
                "scripted author response violates its source contract"
            ) from exc


class CheckRunner:
    """Check evaluator family before calling the evaluator port, then freeze evidence."""

    def __init__(
        self,
        artifacts: ArtifactSink,
        evaluator: Evaluator,
        check_contracts: Mapping[str, CheckContractV1],
    ) -> None:
        self.artifacts = artifacts
        self.evaluator = evaluator
        self.check_contracts = dict(check_contracts)

    def result(self, port: CheckInput) -> EvaluatorResultV1:
        if not isinstance(port, CheckInput):
            raise TypeError("check runner requires CheckInput")
        request = dict(port.request)
        check_ref = request["check_contract_hash"]
        check = self.check_contracts.get(check_ref)
        if check is None or check.id != request["check_id"]:
            raise AdapterContractError("check request has no admitted contract")
        family = FAMILIES.get(self.evaluator.family)
        if (
            family is None
            or family.check_version != check.evaluator_version
            or self.evaluator.family != family.name
        ):
            raise AdapterContractError("evaluator family differs from the admitted check")
        packet_ref = request["evaluator_packet_ref"]
        if packet_ref is None or port.evaluator_packet is None:
            raise AdapterContractError("check request has no authorized evaluator packet")
        evaluation_request = EvaluationRequestV1.create(
            family.name,
            request["target_checkpoint"],
            check,
            packet_ref,
            port.files,
            evaluator_packet=dict(port.evaluator_packet),
        )
        evidence = self.evaluator.evaluate(evaluation_request)
        if not isinstance(evidence, EvaluationEvidenceV1):
            raise AdapterContractError("evaluator must return EvaluationEvidenceV1")
        try:
            wire = evidence.to_wire(evaluation_request)
            evidence_ref = self.artifacts.put_artifact(wire)
            return EvaluatorResultV1(port.request_ref, evidence.status, evidence_ref)
        except AdapterContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterContractError("evaluator response violates its evidence contract") from exc


@dataclass(frozen=True)
class Gatherers:
    """The four producer adapters consumed by :class:`RolloutDriver`."""

    sampler: SamplingRunner
    tools: ToolRunner
    author: ScriptedAuthorSource
    evaluator: CheckRunner


__all__ = [
    "CheckRunner",
    "Gatherers",
    "SamplingRunner",
    "ScriptedAuthorSource",
    "ToolRunner",
]
