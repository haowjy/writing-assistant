"""Opt-in transactional text-tool stepping for an admitted writer node.

The Phase 2 reducer remains the publication authority. Each writer observation is
followed by a context_changed effect *in the same commit*: the latter can name the
already-hashed observation as provenance without making a hash cycle.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from writing_agent.task_graph import (
    ContextRevisionV1,
    EnvironmentStateV1,
    MessageV1,
    canonical_json,
    safe_path,
    tree_hash,
    validate_hash,
)
from writing_agent.task_graph_accounting import (
    READ_TOOLS,
    exhausted_stop_reason,
    observation_read_tokens,
    sampled_usage_charge,
    tool_error,
    tool_result_charge,
)
from writing_agent.task_graph_admission import AdmittedGraphV1
from writing_agent.task_graph_compaction import (
    ContextPolicyV1,
    make_record,
    require_quiescent,
    select_context,
)
from writing_agent.task_graph_environment import (
    WriterRuntimeError,
    WriterStepV1,
)
from writing_agent.task_graph_ports import RuntimeDependenciesV1, local_runtime_dependencies
from writing_agent.task_graph_projection import (
    execution_value,
    project_writer_context,
)
from writing_agent.task_graph_sampling import (
    PreparedRequestV1,
    VerifiedMessagesStaleError,
    _validate_prepared_request,
)
from writing_agent.task_graph_store import RuntimeHandle, TaskGraphStore

_MAX_ARGUMENT_BYTES = 65_536
_MAX_CALL_EVIDENCE = 131_072


def _count_whitespace(value: str) -> int:
    return len(value.split())


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _valid_utf8(value: str) -> bool:
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return False
    return True


def _safe_evidence(value: Any) -> str:
    """Keep backend syntax inspectable without making it a canonical message value."""
    try:
        result = json.dumps(value, ensure_ascii=True, default=repr)
    except Exception:
        try:
            result = repr(value).encode("utf-8", "backslashreplace").decode("utf-8")
        except Exception:
            result = f"<unserializable {type(value).__name__}>"
    return result[:_MAX_CALL_EVIDENCE] + ("<truncated>" if len(result) > _MAX_CALL_EVIDENCE else "")


def _bounded_call_evidence(value: Any) -> tuple[bool, str]:
    """Reject oversized/deep Python envelopes before serializing backend evidence."""
    stack = [(value, 0)]
    remaining_bytes = _MAX_CALL_EVIDENCE
    remaining_nodes = 4096
    seen: set[int] = set()
    while stack:
        item, depth = stack.pop()
        remaining_nodes -= 1
        if remaining_nodes < 0 or depth > 64:
            return False, "<oversized or deeply nested tool call envelope>"
        if isinstance(item, (dict, list, tuple)):
            if len(item) > remaining_nodes:
                return False, "<oversized or deeply nested tool call envelope>"
            identity = id(item)
            if identity in seen:
                return False, "<cyclic tool call envelope>"
            seen.add(identity)
            children = item.items() if isinstance(item, dict) else item
            if isinstance(item, dict):
                for key, child in children:
                    stack.append((key, depth + 1))
                    stack.append((child, depth + 1))
            else:
                stack.extend((child, depth + 1) for child in children)
        elif isinstance(item, str):
            remaining_bytes -= len(item.encode("utf-8", "backslashreplace"))
        elif isinstance(item, bytes):
            remaining_bytes -= len(item)
        elif item is None or type(item) in {bool, int, float}:
            remaining_bytes -= 64
        else:
            return False, f"<unsupported tool call envelope {type(item).__name__}>"
        if remaining_bytes < 0:
            return False, "<oversized or deeply nested tool call envelope>"
    return True, _safe_evidence(value)


class TransactionalWriterV1:
    """A serial, explicit writer adapter; never invokes a model or author service."""

    def __init__(
        self,
        store: TaskGraphStore,
        graph: AdmittedGraphV1,
        rollout_id: str,
        entry_checkpoint_id: str,
        *,
        count_tokens: Callable[[str], int] = _count_whitespace,
        read_tokenizer: str = "whitespace-v1",
        dependencies: RuntimeDependenciesV1 | None = None,
    ) -> None:
        if not isinstance(graph, AdmittedGraphV1):
            raise TypeError("writer runtime requires an admitted graph")
        if (
            not isinstance(rollout_id, str)
            or not rollout_id
            or not _valid_utf8(rollout_id)
            or any(character.isspace() or ord(character) < 0x20 for character in rollout_id)
        ):
            raise ValueError("rollout_id must be a nonempty logical id")
        self.store = store
        self.graph = graph
        self.rollout_id = rollout_id
        self.count_tokens = count_tokens
        if not isinstance(read_tokenizer, str) or not read_tokenizer:
            raise ValueError("read_tokenizer must be a versioned nonempty name")
        self.read_tokenizer = read_tokenizer
        if count_tokens is not _count_whitespace and read_tokenizer == "whitespace-v1":
            raise WriterRuntimeError("custom read tokenizer requires a verifiable registry")
        entry = store.load_checkpoint(entry_checkpoint_id)
        if (
            entry.state.instance_ref != graph.instance.identity()
            or entry.state.position["phase"] != "ready_writer"
            or entry.state.history["action_ids"]
            or entry.state.history["tool_result_ids"]
        ):
            raise WriterRuntimeError("entry must be an admitted, unsampled ready-writer checkpoint")
        node = graph.node(entry.state.position["node_id"])
        self.phase5 = node.contract.interaction_contract.mode == "scripted_author"
        if node.contract.interaction_contract.mode == "scripted_author":
            if (
                entry.state.author_packet_ref
                != node.contract.interaction_contract.author_packet_ref
            ):
                raise WriterRuntimeError("entry author packet differs from admitted private packet")
            decisions = store.get_artifact(entry.state.decisions_ref, expected_domain="payload")
            disclosures = store.get_artifact(entry.state.disclosures_ref, expected_domain="payload")
            requirements = store.get_artifact(
                entry.state.requirements_ref, expected_domain="payload"
            )
            if (
                decisions
                != {
                    "record_type": "DecisionLedgerV1",
                    "schema": 1,
                    "values": {},
                    "proposals": {},
                }
                or disclosures
                != {
                    "record_type": "DisclosureLedgerV1",
                    "schema": 1,
                    "decisions": [],
                }
                or requirements
                != {
                    "record_type": "RequirementLedgerV1",
                    "schema": 1,
                    "active": dict(node.author_packet.requirements),
                    "superseded": {},
                }
            ):
                raise WriterRuntimeError("scripted-author entry ledgers are not initialized")
        context = store.load_context(entry.state.context_ref)
        request = store.get_artifact(node.contract.entry_contract.request_ref)
        if (
            isinstance(request, dict)
            and isinstance(request.get("text"), str)
            and (
                len(context.messages) < 2
                or canonical_json(context.messages[1].content)
                != canonical_json(({"type": "text", "text": request["text"]},))
            )
        ):
            raise WriterRuntimeError("entry request differs from admitted node request")
        self.entry_checkpoint_id = entry_checkpoint_id
        self.dependencies = dependencies or local_runtime_dependencies(
            store, rollout_id, entry_checkpoint_id
        )
        self.dependencies.manifest()
        self.environment = self.dependencies.environment
        entry_budget = store.get_artifact(entry.state.budgets_ref, expected_domain="payload")
        if isinstance(entry_budget, dict) and "context_tokens" in entry_budget.get("limits", {}):
            raise WriterRuntimeError("context_tokens cannot be enforced before sampling")
        if (
            isinstance(entry_budget, dict)
            and entry_budget.get("read_tokenizer") == read_tokenizer
            and read_tokenizer != "whitespace-v1"
        ):
            raise WriterRuntimeError("unsupported read tokenizer for semantic validation")

    def validate_runtime(self, runtime: RuntimeHandle) -> tuple[Any, dict[str, Any]]:
        checkpoint = self.store.load_checkpoint(runtime.checkpoint_id)
        if (
            checkpoint.state != runtime.state
            or self.store.load_context(runtime.state.context_ref) != runtime.context
        ):
            raise WriterRuntimeError("runtime handle does not match its checkpoint")
        if runtime.state.instance_ref != self.graph.instance.identity():
            raise WriterRuntimeError("runtime graph identity differs from admitted graph")
        project_writer_context(self.store, self.entry_checkpoint_id, runtime.checkpoint_id)
        node = self.graph.node(runtime.state.position["node_id"])
        if (
            node.spec.kind != "writer"
            or runtime.state.position["entry_contract"] != node.spec.entry_contract
        ):
            raise WriterRuntimeError("not an admitted writer-node entry")
        expected_tools = self.dependencies.tools.schemas(
            node.contract.entry_contract.tool_allowlist, node.interaction_policy
        )
        if canonical_json(runtime.context.tools) != canonical_json(expected_tools):
            raise WriterRuntimeError(
                "context advertises tools outside the admitted text-tool schema"
            )
        budget = self.store.get_artifact(runtime.state.budgets_ref, expected_domain="payload")
        if not isinstance(budget, dict) or set(budget) != {
            "schema",
            "limits",
            "consumed",
            "read_tokenizer",
        }:
            raise WriterRuntimeError("writer budget artifact has an invalid schema")
        if budget["schema"] != 1 or not all(
            isinstance(budget[k], dict) for k in ("limits", "consumed")
        ):
            raise WriterRuntimeError("invalid writer budget artifact")
        if budget["read_tokenizer"] != self.read_tokenizer:
            raise WriterRuntimeError("read tokenizer differs from the pinned budget contract")
        contract = node.contract.budget_contract
        required = {
            "writer_turns": contract.max_steps,
            "tool_calls": contract.max_tool_calls,
            "read_tokens": contract.max_read_tokens,
            "storage_bytes": contract.max_total_bytes,
        }
        if any(budget["limits"].get(k) != v for k, v in required.items()):
            raise WriterRuntimeError("budget limits differ from admitted node")
        if any(type(v) is not int or v < 0 for v in budget["consumed"].values()) or any(
            type(v) is not int or v < 0 for v in budget["limits"].values()
        ):
            raise WriterRuntimeError("budget counters must be nonnegative integers")
        for name in budget["limits"]:
            permitted_overrun = name in {"context_bytes", "context_storage_bytes"} or (
                runtime.state.position["phase"] == "terminal"
                and name in {"generated_tokens", "total_tokens"}
            )
            if budget["consumed"].get(name, 0) > budget["limits"][name] and not permitted_overrun:
                raise WriterRuntimeError(f"{name} budget is already over limit")
        if budget["consumed"].get("storage_bytes", 0) != sum(
            len(text.encode("utf-8")) for text in runtime.state.files.values()
        ):
            raise WriterRuntimeError("storage counter differs from checkpoint files")
        return node, budget

    def change_context(self, runtime: RuntimeHandle, policy: ContextPolicyV1) -> WriterStepV1:
        """Publish one fixed, zero-mask context operation at a completed exchange."""
        if not isinstance(policy, ContextPolicyV1):
            raise TypeError("context policy must be ContextPolicyV1")
        _, budget = self.validate_runtime(runtime)
        require_quiescent(runtime.state)
        self.environment.head(runtime)
        sources: list[str | None] = []
        project_writer_context(
            self.store,
            self.entry_checkpoint_id,
            runtime.checkpoint_id,
            source_event_ids=sources,
        )
        seed = None
        seed_sources: list[str | None] = []
        if policy.operation == "seed":
            ancestor = runtime.checkpoint_id
            while ancestor != policy.seed_checkpoint_ref:
                checkpoint = self.store.load_checkpoint(ancestor)
                if not checkpoint.parents:
                    raise WriterRuntimeError("named seed is not an ancestor checkpoint")
                ancestor = checkpoint.parents[0]
            seed = self.store.load_context(self.store.load_checkpoint(ancestor).state.context_ref)
            project_writer_context(
                self.store,
                self.entry_checkpoint_id,
                ancestor,
                source_event_ids=seed_sources,
            )
        policy_ref = self.store.put_artifact(policy.to_dict())
        origin = (
            f"{runtime.state.position['lineage_id']}:context:{runtime.state.history['seq'] + 1}"
        )
        messages, _, summary, _ = select_context(
            runtime.context,
            tuple(sources),
            policy,
            seed=seed,
            seed_sources=tuple(seed_sources),
            summary_origin=origin,
        )
        provenance = (
            (runtime.state.history["head"],) if runtime.state.history["head"] is not None else ()
        )
        context = ContextRevisionV1(
            messages=messages,
            tools=runtime.context.tools,
            rendering=runtime.context.rendering,
            event_head=runtime.state.history["head"],
            provenance_refs=provenance,
        )
        if context.identity() == runtime.state.context_ref:
            raise WriterRuntimeError("context operation would not create a new revision")
        summary_ref = (
            self.store.put_bytes_artifact(summary.encode("utf-8")) if summary is not None else None
        )
        self.store.persist(context)
        record, new_budget, _ = make_record(
            self.store,
            runtime.state,
            runtime.context,
            tuple(sources),
            policy,
            policy_ref,
            context,
            summary_ref,
            budget,
            seed=seed,
            seed_sources=tuple(seed_sources),
        )
        budget_ref = self.store.put_artifact(new_budget)
        batch = self.environment.batch(runtime, restore_prefix="context")
        event, record_ref = batch.append_record(
            "context_changed",
            "environment",
            record=record,
            changes={"context_ref": context.identity(), "budgets_ref": budget_ref},
            audience=("controller", "trainer"),
        )
        refs = [policy_ref, budget_ref]
        if summary_ref is not None:
            refs.append(summary_ref)
        if policy.seed_checkpoint_ref is not None:
            refs.append(policy.seed_checkpoint_ref)
        return batch.publish(event, record_ref, extra_refs=tuple(refs))

    @staticmethod
    def _parsed_calls(
        raw_calls: Any,
        rollout_id: str,
        ordinal: int,
        allowed: frozenset[str],
        prior_raw_ids: set[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(raw_calls, list):
            raise WriterRuntimeError("tool_calls must be an array")
        queue: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        seen = set(prior_raw_ids)
        atomic_reject = len(raw_calls) > 1 and any(
            isinstance(call, dict)
            and isinstance(call.get("function"), dict)
            and call["function"].get("name") == "ask_author"
            for call in raw_calls
        )
        for index, raw in enumerate(raw_calls):
            bounded, evidence = _bounded_call_evidence(raw)
            call_id = f"{rollout_id}:call:{ordinal}:{index}"
            raw_id = raw.get("id") if isinstance(raw, dict) else None
            function = raw.get("function") if isinstance(raw, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            reason = None
            if not bounded:
                reason = "Tool call envelope exceeds size or nesting limit"
                raw_id = name = arguments = None
            if (
                isinstance(raw_id, str)
                and len(raw_id) > 256
                or isinstance(name, str)
                and len(name) > 256
                or isinstance(arguments, dict)
                and (
                    len(arguments) > 64
                    or sum(
                        len(value.encode("utf-8", "surrogatepass"))
                        for value in arguments.values()
                        if isinstance(value, str)
                    )
                    > _MAX_ARGUMENT_BYTES
                )
            ):
                reason = "Tool call envelope exceeds size limit"
            if bounded and (
                not isinstance(raw, dict)
                or set(raw) != {"id", "type", "function"}
                or raw.get("type") != "function"
            ):
                reason = "Invalid tool call envelope"
            if bounded and (
                not isinstance(function, dict) or set(function) != {"name", "arguments"}
            ):
                reason = reason or "Invalid tool function envelope"
            if (
                not isinstance(raw_id, str)
                or not raw_id
                or not raw_id.isprintable()
                or any(ch.isspace() for ch in raw_id)
                or not _valid_utf8(raw_id)
            ):
                reason = reason or "Tool call needs an id"
            elif raw_id in seen:
                reason = reason or "Duplicate tool call id"
            else:
                seen.add(raw_id)
            if (
                not isinstance(name, str)
                or not name
                or not _valid_utf8(name)
                or any(ch.isspace() or ord(ch) < 0x20 for ch in name)
            ):
                reason = reason or "Invalid tool function"
            if isinstance(arguments, str) and bounded:
                try:
                    if len(arguments.encode("utf-8", "surrogatepass")) > _MAX_ARGUMENT_BYTES:
                        raise ValueError("tool arguments JSON exceeds size limit")
                    arguments = json.loads(
                        arguments,
                        object_pairs_hook=_pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(
                            ValueError("invalid JSON constant")
                        ),
                    )
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    reason = reason or "Invalid tool arguments JSON"
            if not isinstance(arguments, dict):
                reason = reason or "Tool arguments must be an object"
            elif name == "ask_author":
                try:
                    from writing_agent.task_graph_scripted import validate_ask_shape

                    validate_ask_shape(arguments)
                except (TypeError, ValueError) as exc:
                    reason = reason or str(exc)
            elif not all(
                isinstance(key, str)
                and _valid_utf8(key)
                and isinstance(value, str)
                and _valid_utf8(value)
                for key, value in arguments.items()
            ):
                reason = reason or "Tool arguments must be an object of UTF-8 strings"
            if reason is None and "path" in arguments:
                path = arguments["path"]
                if not (name in {"list_dir", "search"} and path == "."):
                    candidate = (
                        path[:-1] if name in {"list_dir", "search"} and path.endswith("/") else path
                    )
                    try:
                        safe_path(candidate)
                    except (TypeError, ValueError):
                        reason = "Unsafe or noncanonical workspace path"
            if not isinstance(name, str) or name not in allowed:
                reason = reason or "Tool is not available in this condition"
            if atomic_reject:
                reason = "Mixed control and file-tool batch is forbidden"
            queue.append(
                {
                    "call_id": call_id,
                    "name": name if reason is None else "invalid_call",
                    "arguments": arguments if reason is None else {},
                }
            )
            metadata.append(
                {
                    "call_id": call_id,
                    "raw_id": raw_id
                    if isinstance(raw_id, str) and raw_id.isprintable() and _valid_utf8(raw_id)
                    else None,
                    "validation_error": reason,
                    "parsed_call_json": evidence,
                }
            )
        return queue, metadata

    def prepare_request(self, runtime: RuntimeHandle, exact_request: Any) -> str:
        """Durably pin caller-owned backend input, without verifying its messages.

        Preparation does not dispatch or reserve a paid call. The returned ref
        binds the request to this exact context; an unused artifact is an orphan.
        """
        if exact_request is None:
            raise WriterRuntimeError("prepared request requires exact backend input")
        _, budget = self.validate_runtime(runtime)
        state = runtime.state
        if (
            state.position["phase"] != "ready_writer"
            or state.continuation["next_call"] != len(state.continuation["tool_queue"])
            or budget["consumed"].get("writer_turns", 0) >= budget["limits"]["writer_turns"]
        ):
            raise WriterRuntimeError("request preparation requires an available writer turn")
        for name in ("generated_tokens", "total_tokens"):
            if (
                name in budget["limits"]
                and budget["consumed"].get(name, 0) >= budget["limits"][name]
            ):
                raise WriterRuntimeError(f"{name} budget exhausted before sampling")
        for name in ("context_bytes", "context_storage_bytes"):
            if (
                name in budget["limits"]
                and budget["consumed"].get(name, 0) >= budget["limits"][name]
            ):
                raise WriterRuntimeError(f"{name} budget exhausted before sampling")
        self.environment.head(runtime)
        payload_ref = (
            self.store.put_bytes_artifact(exact_request)
            if isinstance(exact_request, bytes)
            else self.store.put_artifact(exact_request)
        )
        return self.store.put_artifact(
            self.dependencies.sampling.prepared_request(
                runtime.context, payload_ref, verified=False
            ).to_wire()
        )

    def prepare_verified_messages(
        self, runtime: RuntimeHandle, exact_request: Mapping[str, Any]
    ) -> str:
        """Pin a request whose typed message sequence is the current projection.

        Other adapter-specific fields are persisted but not interpreted or
        verified. This does not imply exact backend bytes or native eligibility.
        """
        if not isinstance(exact_request, Mapping) or canonical_json(
            exact_request.get("messages")
        ) != canonical_json([message.to_dict() for message in runtime.context.messages]):
            raise WriterRuntimeError("request messages differ from the current writer context")
        prepared_ref = self.prepare_request(runtime, dict(exact_request))
        prepared = self.store.get_artifact(prepared_ref, expected_domain="payload")
        return self.store.put_artifact(
            self.dependencies.sampling.prepared_request(
                runtime.context,
                PreparedRequestV1.from_wire(prepared).payload_ref,
                verified=True,
            ).to_wire()
        )

    def submit_action(
        self,
        runtime: RuntimeHandle,
        message: Mapping[str, Any],
        *,
        exact_request: Any | None = None,
        prepared_request_ref: str | None = None,
        raw_output: str | bytes | None = None,
        trace: Mapping[str, Any] | None = None,
        usage: Mapping[str, int] | None = None,
    ) -> WriterStepV1:
        node, budget = self.validate_runtime(runtime)
        state = runtime.state
        if state.position["phase"] != "ready_writer" or state.continuation["next_call"] != len(
            state.continuation["tool_queue"]
        ):
            raise WriterRuntimeError("writer action requires a ready, drained tool queue")
        if budget["consumed"].get("writer_turns", 0) >= budget["limits"]["writer_turns"]:
            raise WriterRuntimeError("writer-turn budget exhausted")
        for name in ("context_bytes", "context_storage_bytes"):
            if (
                name in budget["limits"]
                and budget["consumed"].get(name, 0) >= budget["limits"][name]
            ):
                raise WriterRuntimeError(f"{name} budget exhausted before sampling")
        self.environment.head(runtime)
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise WriterRuntimeError("backend adapter must supply an assistant message")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise WriterRuntimeError("assistant content must be text or null")
        action_ordinal = len(state.history["action_ids"])
        action_id = f"{self.rollout_id}:action:{action_ordinal}"
        prior_raw_ids = {
            call["raw_id"]
            for entry in self.environment.runtime_log(state)
            if entry["kind"] == "writer_action"
            for call in self.store.get_artifact(entry["record_ref"])["calls"]
            if call["raw_id"] is not None
        }
        parse_error = None
        try:
            queue, call_metadata = self._parsed_calls(
                message.get("tool_calls", []),
                self.rollout_id,
                action_ordinal,
                frozenset(node.contract.entry_contract.tool_allowlist),
                prior_raw_ids,
            )
        except WriterRuntimeError as exc:
            # A sampled token overrun must still be journaled even if the
            # adapter also supplied an unusable batch envelope.
            parse_error = exc
            queue, call_metadata = [], []
        if node.contract.interaction_contract.mode == "scripted_author":
            from writing_agent.task_graph_scripted import validate_ask_semantics

            decisions = self.store.get_artifact(state.decisions_ref, expected_domain="payload")
            for call, metadata in zip(queue, call_metadata, strict=True):
                if call["name"] == "ask_author" and metadata["validation_error"] is None:
                    try:
                        validate_ask_semantics(call["arguments"], node, decisions)
                    except (TypeError, ValueError) as exc:
                        metadata["validation_error"] = str(exc)
                        call["name"] = "invalid_call"
                        call["arguments"] = {}
        if usage is None:
            usage = {}
        if not isinstance(usage, Mapping) or any(
            name in usage and (type(usage[name]) is not int or usage[name] < 0)
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise WriterRuntimeError("top-level token usage must contain nonnegative integers")
        # Provider detail fields are retained verbatim, never charged twice.
        canonical_json(usage)
        next_budget, exceeded = sampled_usage_charge(budget, usage)
        if exceeded is None:
            if parse_error is not None:
                raise parse_error
            if not queue and not content.strip():
                raise WriterRuntimeError("final response must contain text")
        if trace is not None and not isinstance(trace, Mapping):
            raise WriterRuntimeError("trace metadata must be an object")
        if trace is not None and "per_token_logprobs" in trace:
            raise WriterRuntimeError("native logprob arrays require a binary artifact reference")
        if trace is not None and "per_token_logprobs_ref" in trace:
            validate_hash(trace["per_token_logprobs_ref"])
            self.store.get_artifact(
                trace["per_token_logprobs_ref"], expected_domain="payload:bytes"
            )
        if (
            trace is not None
            and "generated_token_ids" in trace
            and (
                not isinstance(trace["generated_token_ids"], list)
                or any(
                    type(token) is not int or token < 0 for token in trace["generated_token_ids"]
                )
            )
        ):
            raise WriterRuntimeError("generated token IDs must be nonnegative integers")
        if ("generated_tokens" in budget["limits"] and "completion_tokens" not in usage) or (
            "total_tokens" in budget["limits"]
            and "total_tokens" not in usage
            and not {"prompt_tokens", "completion_tokens"} <= usage.keys()
        ):
            raise WriterRuntimeError("token-limited action requires backend usage evidence")
        if exact_request is not None and prepared_request_ref is not None:
            raise WriterRuntimeError("supply either an exact or a prepared request, not both")
        if prepared_request_ref is not None:
            validate_hash(prepared_request_ref)
            prepared = self.store.get_artifact(prepared_request_ref, expected_domain="payload")
            try:
                decoded = PreparedRequestV1.from_wire(prepared)
                _validate_prepared_request(
                    self.store,
                    prepared_request_ref,
                    runtime.context.content_hash,
                    runtime.context.identity(),
                    runtime.context.rendering,
                    decoded.payload_ref,
                )
            except VerifiedMessagesStaleError as exc:
                raise WriterRuntimeError("verified request messages are stale") from exc
            except ValueError as exc:
                raise WriterRuntimeError(
                    "prepared request does not match the sampling context"
                ) from exc
            request_ref = decoded.payload_ref
        else:
            # The direct path still persists supplied evidence before publication.
            request_ref = (
                self.store.put_bytes_artifact(exact_request)
                if isinstance(exact_request, bytes)
                else self.store.put_artifact(exact_request)
                if exact_request is not None
                else None
            )
        if raw_output is not None and not isinstance(raw_output, (str, bytes)):
            raise WriterRuntimeError("raw output must be exact text or bytes when supplied")
        raw_output_ref = (
            self.store.put_bytes_artifact(raw_output)
            if isinstance(raw_output, bytes)
            else self.store.put_artifact(raw_output)
            if raw_output is not None
            else None
        )
        trace_body = self.dependencies.sampling.evidence(
            action_id,
            runtime.context,
            request_ref,
            prepared_request_ref,
            raw_output_ref,
            usage,
            trace,
        ).to_wire()
        trace_ref = self.store.put_artifact(trace_body)
        if exceeded is not None:
            return self._sampled_budget_stop(
                runtime,
                budget,
                exceeded,
                usage,
                message,
                trace_ref,
                request_ref,
                prepared_request_ref,
                raw_output_ref,
            )
        parts: list[dict[str, Any]] = []
        if content:
            parts.append({"type": "text", "text": content})
        parts.extend(
            {
                "type": "tool_call",
                "id": call["call_id"],
                "name": call["name"],
                "arguments": call["arguments"],
            }
            for call in queue
        )
        assistant = MessageV1(
            role="assistant", content=tuple(parts), origin=action_id, loss_eligible=True
        )
        budget_ref = self.store.put_artifact(next_budget)
        continuation = state.to_dict()["continuation"]
        continuation.update(tool_queue=queue, next_call=0)
        position = state.to_dict()["position"]
        if not queue:
            position["phase"] = "checking"
        record = {
            "record_type": "WriterActionV1",
            "action_id": action_id,
            "trace_ref": trace_ref,
            "request_ref": request_ref,
            "prepared_request_ref": prepared_request_ref,
            "raw_output_ref": raw_output_ref,
            "logprob_ref": trace_body["logprob_ref"],
            "calls": call_metadata,
            "usage": dict(usage),
            "model": trace_body["model"],
            "seed": trace_body["seed"],
            "loss_eligibility": {
                "assistant_text": bool(content),
                "tool_syntax": bool(queue),
                "assistant_ending": True,
                "system": False,
                "user": False,
                "author": False,
                "tool_observation": False,
                "seed": False,
                "environment": False,
            },
        }
        batch = self.environment.batch(runtime, restore_prefix="writer")
        event, record_ref = batch.append_record(
            "writer_action",
            "writer",
            record=record,
            message=assistant,
            changes={"continuation": continuation, "position": position, "budgets_ref": budget_ref},
            history={"action_ids": [*state.history["action_ids"], action_id]},
            audience=("controller", "trainer", "writer"),
        )
        batch.append_visible((assistant,), event)
        return batch.publish(
            event,
            record_ref,
            extra_refs=tuple(
                record[name]
                for name in (
                    "trace_ref",
                    "request_ref",
                    "prepared_request_ref",
                    "raw_output_ref",
                    "logprob_ref",
                )
                if isinstance(record.get(name), str)
            ),
        )

    def _sampled_budget_stop(
        self,
        runtime,
        budget,
        exceeded,
        usage,
        message,
        trace_ref,
        request_ref,
        prepared_request_ref,
        raw_output_ref,
    ) -> WriterStepV1:
        """Account a sampled overrun without accepting call syntax or executing tools."""
        state = runtime.state
        self.environment.head(runtime)
        next_budget, actual_exceeded = sampled_usage_charge(budget, usage)
        if actual_exceeded != exceeded:
            raise WriterRuntimeError("sampled stop no longer matches its budget")
        budget_ref = self.store.put_artifact(next_budget)
        outcome = {
            "schema": 1,
            "task_status": "incomplete",
            "execution_status": "valid",
            "stop_reason": f"{exceeded}_budget",
            "reward_status": "pending",
            "training_eligibility": "pending",
        }
        if self.phase5:
            outcome.update(
                record_type="TerminalOutcomeV1",
                candidate_checkpoint=runtime.checkpoint_id,
                check_result_refs=[],
                transition_edge_id=None,
                requirement_version=state.requirements_ref,
            )
        outcome_ref = self.store.put_artifact(outcome)
        record = {
            "record_type": "WriterSampledBudgetStopV1",
            "action_id": self.store.get_artifact(trace_ref)["action_id"],
            "reason": f"{exceeded}_budget",
            "usage": dict(usage),
            "model": self.store.get_artifact(trace_ref)["model"],
            "seed": self.store.get_artifact(trace_ref)["seed"],
            "parsed_message_json": _safe_evidence(message),
            "trace_ref": trace_ref,
            "request_ref": request_ref,
            "prepared_request_ref": prepared_request_ref,
            "raw_output_ref": raw_output_ref,
            "logprob_ref": self.store.get_artifact(trace_ref)["logprob_ref"],
        }
        position = state.to_dict()["position"]
        position["phase"] = "terminal"
        batch = self.environment.batch(runtime, restore_prefix="writer")
        event, record_ref = batch.append_record(
            "budget_charged",
            "writer_runtime",
            record=record,
            changes={"position": position, "budgets_ref": budget_ref, "outcome_ref": outcome_ref},
            audience=("controller", "evaluator", "trainer"),
        )
        return batch.publish(
            event,
            record_ref,
            extra_refs=tuple(
                ref
                for ref in (
                    outcome_ref,
                    budget_ref,
                    trace_ref,
                    request_ref,
                    prepared_request_ref,
                    raw_output_ref,
                    record["logprob_ref"],
                )
                if ref is not None
            ),
        )

    def step_tool(self, runtime: RuntimeHandle) -> WriterStepV1:
        _, budget = self.validate_runtime(runtime)
        self.environment.head(runtime)
        state = runtime.state
        queue = state.continuation["tool_queue"]
        cursor = state.continuation["next_call"]
        if state.position["phase"] != "ready_writer" or cursor >= len(queue):
            raise WriterRuntimeError("no queued tool call to resume")
        entries = self.environment.runtime_log(state)
        action = next(
            (
                self.store.get_artifact(entry["record_ref"])
                for entry in reversed(entries)
                if entry["kind"] == "writer_action"
            ),
            None,
        )
        if action is None:
            raise WriterRuntimeError("queued calls have no committed writer action")
        call = queue[cursor]
        metadata = action["calls"][cursor]
        if call["call_id"] != metadata["call_id"]:
            raise WriterRuntimeError("queued call does not match committed action")
        predispatch_error = tool_error(budget, metadata["validation_error"], call["name"])
        if (
            call["name"] == "ask_author"
            and metadata["validation_error"] is None
            and predispatch_error is None
        ):
            from writing_agent.task_graph_scripted import ScriptedAuthorRuntimeV1

            return ScriptedAuthorRuntimeV1(self).request(runtime, call, action)
        observation: dict[str, Any]
        files = dict(state.files)
        read_charge = 0
        if predispatch_error is not None:
            observation = predispatch_error
        else:
            execution = self.dependencies.tools.execute(
                files,
                call["name"],
                dict(call["arguments"]),
                stage_parent=runtime.workspace.parent,
                max_file_bytes=self.store.max_file_bytes,
                max_workspace_bytes=min(
                    self.store.max_workspace_bytes, budget["limits"]["storage_bytes"]
                ),
            )
            observation = execution.observation
            if observation["ok"] and call["name"] in READ_TOOLS:
                charge = (
                    observation_read_tokens(observation, call["name"], self.read_tokenizer)
                    if self.read_tokenizer == "whitespace-v1"
                    else self.count_tokens(json.dumps(observation["result"], ensure_ascii=False))
                )
                if type(charge) is not int or charge < 0:
                    raise WriterRuntimeError("read tokenizer returned an invalid charge")
                if (
                    budget["consumed"].get("read_tokens", 0) + charge
                    > budget["limits"]["read_tokens"]
                ):
                    observation = {
                        "ok": False,
                        "valid": True,
                        "error": "Read-token budget exceeded",
                    }
                else:
                    read_charge = charge
                    files = execution.files
            elif observation["ok"]:
                files = execution.files
        delta = {
            path: {"before": state.files.get(path), "after": files.get(path)}
            for path in sorted(set(state.files) | set(files))
            if state.files.get(path) != files.get(path)
        }
        next_budget, budget_charge = tool_result_charge(budget, state.files, files, read_charge)
        budget_ref = self.store.put_artifact(next_budget)
        continuation = state.to_dict()["continuation"]
        continuation["next_call"] = cursor + 1
        position = state.to_dict()["position"]
        # Turn exhaustion is a controller stop boundary, not a fabricated
        # termination/check outcome. Phase 5 will publish that decision.
        result_id = f"{self.rollout_id}:tool_result:{len(state.history['tool_result_ids'])}"
        tool_message = MessageV1(
            role="tool",
            call_id=call["call_id"],
            origin=action["action_id"],
            content=({"type": "tool_result", "call_id": call["call_id"], "content": observation},),
            loss_eligible=False,
        )
        prospective = ContextRevisionV1(
            messages=(*runtime.context.messages, tool_message),
            tools=runtime.context.tools,
            rendering=runtime.context.rendering,
        )
        after_value = state.to_dict()
        after_value.update(
            files=files,
            tree_hash=tree_hash(files),
            continuation=continuation,
            position=position,
            budgets_ref=budget_ref,
        )
        after_value["history"]["tool_result_ids"] = [*state.history["tool_result_ids"], result_id]
        prospective_state = EnvironmentStateV1.from_dict(after_value)
        record = {
            "record_type": "WriterToolResultV1",
            "result_id": result_id,
            "call_id": call["call_id"],
            "action_id": action["action_id"],
            "observation": observation,
            "file_delta": delta,
            "before_execution_hash": execution_value(self.store, state, runtime.context, budget),
            "after_execution_hash": execution_value(
                self.store, prospective_state, prospective, next_budget
            ),
            "budget_charge": budget_charge,
            "loss_eligibility": {"tool_observation": False},
        }
        batch = self.environment.batch(runtime, restore_prefix="writer")
        event, record_ref = batch.append_record(
            "tool_result",
            "environment",
            record=record,
            message=tool_message,
            changes={"continuation": continuation, "position": position, "budgets_ref": budget_ref},
            history={"tool_result_ids": [*state.history["tool_result_ids"], result_id]},
            delta=delta,
            audience=("controller", "trainer", "writer"),
        )
        batch.append_visible((tool_message,), event)
        return batch.publish(event, record_ref)

    def drain_tools(self, runtime: RuntimeHandle) -> RuntimeHandle:
        """Resume only uncommitted queued calls; each iteration publishes one result."""
        while runtime.state.continuation["next_call"] < len(
            runtime.state.continuation["tool_queue"]
        ):
            if runtime.state.position["phase"] != "ready_writer":
                break
            runtime = self.step_tool(runtime).runtime
        return runtime

    def stop_exhausted(self, runtime: RuntimeHandle) -> WriterStepV1:
        """Seal a drained writer entry with no remaining turn or token capacity.

        A final assistant reply is in ``checking`` and cannot take this path: its
        checks must run before anyone decides whether the task was complete.
        """
        _, budget = self.validate_runtime(runtime)
        state = runtime.state
        reason = exhausted_stop_reason(budget)
        if (
            state.position["phase"] != "ready_writer"
            or state.continuation["next_call"] != len(state.continuation["tool_queue"])
            or reason is None
        ):
            raise WriterRuntimeError("writer budget is not exhausted at a drained boundary")
        self.environment.head(runtime)
        outcome = {
            "schema": 1,
            "task_status": "incomplete",
            "execution_status": "valid",
            "stop_reason": reason,
            "reward_status": "pending",
            "training_eligibility": "pending",
        }
        if self.phase5:
            outcome.update(
                record_type="TerminalOutcomeV1",
                candidate_checkpoint=runtime.checkpoint_id,
                check_result_refs=[],
                transition_edge_id=None,
                requirement_version=state.requirements_ref,
            )
        outcome_ref = self.store.put_artifact(outcome)
        record = {"record_type": "WriterExhaustedStopV1", "reason": outcome["stop_reason"]}
        position = state.to_dict()["position"]
        position["phase"] = "terminal"
        batch = self.environment.batch(runtime, restore_prefix="writer")
        event, record_ref = batch.append_record(
            "termination_recorded",
            "writer_runtime",
            record=record,
            changes={"position": position, "outcome_ref": outcome_ref},
            audience=("controller", "evaluator", "trainer"),
        )
        return batch.publish(event, outcome_ref, extra_refs=(outcome_ref,))
