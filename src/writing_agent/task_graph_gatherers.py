"""Port adapters that turn allowlisted inputs into typed transition records.

Gatherers do not receive lineage views or publish events. Adapter responses are
checked here; ``RolloutEnvironment.commit`` remains the recording boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from writing_agent.task_graph import canonical_bytes, canonical_json, context_content_hash
from writing_agent.task_graph_calls import intake_message, tool_effect_contract
from writing_agent.task_graph_contracts import CheckContractV1
from writing_agent.task_graph_errors import AdapterContractError
from writing_agent.task_graph_evaluation import (
    FAMILIES,
    EvaluationEvidenceV1,
    EvaluationRequestV1,
    StoreEvidenceResolver,
    verify_evaluation_evidence,
)
from writing_agent.task_graph_gate import StoreArtifactReader
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
    WriterRequestV1,
    WriterTurnV1,
)
from writing_agent.task_graph_rollout_env import AuthorInput, CheckInput, SamplerInput, ToolInput
from writing_agent.task_graph_sampling import (
    WriterTurnSamplingBindingV1,
    decode_and_bind_sampling,
)
from writing_agent.task_graph_scripted import ScriptCoverageError, resolve_script_reply


class SamplingRunner:
    """Run one sampling port call and produce its fully bound writer-turn input."""

    def __init__(self, store: Any, backend: SampleBackend) -> None:
        self.store = store
        self.backend = backend

    def turn(self, port: SamplerInput) -> WriterTurnV1:
        if not isinstance(port, SamplerInput):
            raise TypeError("sampling runner requires SamplerInput")
        request: dict[str, Any] = {
            "messages": [message.to_dict() for message in port.messages],
        }
        pins = {
            name: value
            for name, value in (
                ("writer_seed", port.writer_seed),
                ("model_ref", port.model_ref),
                ("behavior_policy_ref", port.behavior_policy_ref),
                ("tokenizer_ref", port.tokenizer_ref),
                ("template_ref", port.template_ref),
                ("decoding_ref", port.decoding_ref),
            )
            if value is not None
        }
        if pins:
            request["sampling_pins"] = pins
        request_json = canonical_json(request)
        request_ref = self.store.put_artifact(request)
        prepared_ref = self.store.put_artifact(
            WriterRequestV1(port.context_revision_ref, request_ref, True).to_wire()
        )
        prepared = PreparedSamplingInput(
            request_ref=request_ref,
            prepared_request_ref=prepared_ref,
            context_content_hash=context_content_hash(
                port.messages, tools=port.tools, rendering=port.rendering
            ),
            context_revision_ref=port.context_revision_ref,
            messages_json=canonical_json(request["messages"]),
            tools_json=canonical_json(port.tools),
            rendering_json=canonical_json(port.rendering),
            request_json=request_json,
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
        usage = dict(result.usage or {})
        if "completion_tokens" in port.usage_requirements and "completion_tokens" not in usage:
            raise AdapterContractError("token-limited sample is missing completion usage")
        if "total_tokens" in port.usage_requirements and not (
            "total_tokens" in usage or {"prompt_tokens", "completion_tokens"} <= usage.keys()
        ):
            raise AdapterContractError("token-limited sample is missing total usage")

        trace = dict(result.trace or {})
        if result.logprobs is not None:
            if "per_token_logprobs_ref" in trace or "per_token_logprobs" in trace:
                raise AdapterContractError("sample supplied duplicate logprob evidence")
            tokens = trace.get("generated_token_ids")
            if (
                not isinstance(tokens, list)
                or any(type(token) is not int or token < 0 for token in tokens)
                or len(tokens) != result.logprobs.shape[0]
            ):
                raise AdapterContractError("binary logprobs do not align with sampled tokens")
            trace["per_token_logprobs_ref"] = self.store.put_bytes_artifact(result.logprobs.data)
            trace["per_token_logprobs_codec"] = result.logprobs.codec
            trace["per_token_logprobs_shape"] = list(result.logprobs.shape)
        self._check_pins(port, trace)

        raw_output_ref = None
        if result.raw_output is not None:
            raw_output_ref = (
                self.store.put_bytes_artifact(result.raw_output)
                if isinstance(result.raw_output, bytes)
                else self.store.put_artifact(result.raw_output)
            )
        try:
            turn = WriterTurnV1(
                action_id=port.action_id,
                context_revision_ref=port.context_revision_ref,
                request_ref=request_ref,
                prepared_request_ref=prepared_ref,
                raw_output_ref=raw_output_ref,
                usage=usage,
                adapter_trace=trace or None,
                message=intake_message(dict(message)),
            )
            context = SimpleNamespace(
                revision_ref=port.context_revision_ref,
                messages=port.messages,
            )
            decode_and_bind_sampling(
                WriterTurnSamplingBindingV1(turn, context, StoreArtifactReader(self.store))
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise AdapterContractError("sample response violates the writer-turn contract") from exc
        return turn

    @staticmethod
    def _check_pins(port: SamplerInput, trace: Mapping[str, Any]) -> None:
        expected = {
            "seed": port.writer_seed,
            "model_ref": port.model_ref,
            "behavior_policy_ref": port.behavior_policy_ref,
            "tokenizer_ref": port.tokenizer_ref,
            "template_ref": port.template_ref,
            "decoding_ref": port.decoding_ref,
        }
        for name, value in expected.items():
            if (
                value is not None
                and name in trace
                and canonical_bytes(trace[name]) != canonical_bytes(value)
            ):
                raise AdapterContractError(f"sample response contradicts pinned {name}")


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
            tool_effect_contract(
                name,
                arguments,
                before,
                after,
                observation.get("ok"),
                max_file_bytes=port.tool_spec.max_file_bytes,
                max_workspace_bytes=port.tool_spec.max_workspace_bytes,
                storage_bytes_limit=port.tool_spec.max_workspace_bytes,
            )
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
        request = {**dict(port.request), "request_ref": port.request_ref}
        if request["source"] == "mandatory_feedback":
            rule = next(
                (item for item in port.script.feedback if item["id"] == request["feedback_id"]),
                None,
            )
            if rule is None:
                raise AdapterContractError("script lacks the requested feedback response")
            result = {
                "utterance": rule["utterance"],
                "decision_ids": [],
                "selected_proposals": {},
            }
        else:
            result = None
        try:
            if result is None:
                _, _, result = resolve_script_reply(
                    port.script,
                    request,
                    dict(port.decisions),
                    dict(port.disclosures),
                )
        except ScriptCoverageError:
            return AuthorReplyV1(port.request_ref, "unsupported_coverage", None, (), {})
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterContractError(
                "scripted author response violates its source contract"
            ) from exc
        selected = {
            decision: [] if proposal is None else [proposal]
            for decision, proposal in result["selected_proposals"].items()
        }
        return AuthorReplyV1(
            port.request_ref,
            "answered",
            result["utterance"],
            result["decision_ids"],
            selected,
        )


class CheckRunner:
    """Check evaluator family before calling the evaluator port, then freeze evidence."""

    def __init__(
        self,
        store: Any,
        evaluator: Evaluator,
        check_contracts: Mapping[str, CheckContractV1],
    ) -> None:
        self.store = store
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
        if not isinstance(evidence, EvaluationEvidenceV1) or evidence.family != family.name:
            raise AdapterContractError("evaluator returned evidence for another family")
        try:
            wire = evidence.to_wire(evaluation_request)
            verified = verify_evaluation_evidence(
                evaluation_request,
                wire,
                StoreEvidenceResolver(self.store, packet_ref),
            )
            evidence_ref = self.store.put_artifact(wire)
            return EvaluatorResultV1(port.request_ref, verified.status, evidence_ref)
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
