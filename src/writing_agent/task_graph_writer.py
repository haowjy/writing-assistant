"""Opt-in transactional text-tool stepping for an admitted writer node.

The Phase 2 reducer remains the publication authority. Each writer observation is
followed by a context_changed effect *in the same commit*: the latter can name the
already-hashed observation as provenance without making a hash cycle.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from writing_agent.task_graph import (
    ContextRevisionV1,
    EnvironmentStateV1,
    EventV1,
    MessageV1,
    canonical_json,
    domain_hash,
    safe_path,
    tree_hash,
    validate_hash,
)
from writing_agent.task_graph_admission import AdmittedGraphV1
from writing_agent.task_graph_projection import project_writer_context
from writing_agent.task_graph_store import RuntimeHandle, TaskGraphStore
from writing_agent.workspace import TOOL_SCHEMAS, Workspace, dispatch

_READ_TOOLS = frozenset({"read_file", "search", "list_dir"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _value_only(value: Any) -> Any:
    """Remove embedded identity/provenance edges from opaque value artifacts."""
    if isinstance(value, Mapping):
        return {
            key: _value_only(item)
            for key, item in value.items()
            if not (key.endswith("_ref") or key.endswith("_refs") or "provenance" in key)
        }
    if isinstance(value, (tuple, list)):
        return [_value_only(item) for item in value]
    if isinstance(value, str) and _HASH.fullmatch(value):
        return "<identity>"
    return value


class WriterRuntimeError(ValueError):
    """The supplied action or restored state violates the graph writer contract."""


@dataclass(frozen=True)
class WriterStepV1:
    runtime: RuntimeHandle
    commit_id: str
    event_id: str
    record_ref: str


class TransactionalWriterV1:
    """A serial, explicit writer adapter; never invokes a model or author service."""

    def __init__(
        self,
        store: TaskGraphStore,
        graph: AdmittedGraphV1,
        rollout_id: str,
        entry_checkpoint_id: str,
        *,
        count_tokens: Callable[[str], int] = lambda value: len(value.split()),
        read_tokenizer: str = "whitespace-v1",
    ) -> None:
        if not isinstance(graph, AdmittedGraphV1):
            raise TypeError("writer runtime requires an admitted graph")
        if not rollout_id or any(character.isspace() for character in rollout_id):
            raise ValueError("rollout_id must be a nonempty logical id")
        self.store = store
        self.graph = graph
        self.rollout_id = rollout_id
        self.count_tokens = count_tokens
        if not isinstance(read_tokenizer, str) or not read_tokenizer:
            raise ValueError("read_tokenizer must be a versioned nonempty name")
        self.read_tokenizer = read_tokenizer
        entry = store.load_checkpoint(entry_checkpoint_id)
        if (
            entry.state.instance_ref != graph.instance.identity()
            or entry.state.position["phase"] != "ready_writer"
            or entry.state.history["action_ids"]
            or entry.state.history["tool_result_ids"]
        ):
            raise WriterRuntimeError("entry must be an admitted, unsampled ready-writer checkpoint")
        node = graph.node(entry.state.position["node_id"])
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

    def _check(self, runtime: RuntimeHandle) -> tuple[Any, dict[str, Any]]:
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
        expected_tools = tuple(
            schema
            for schema in TOOL_SCHEMAS
            if schema["function"]["name"] in node.contract.entry_contract.tool_allowlist
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
        for name in ("writer_turns", "tool_calls", "read_tokens", "storage_bytes"):
            if budget["consumed"].get(name, 0) > budget["limits"][name]:
                raise WriterRuntimeError(f"{name} budget is already over limit")
        if budget["consumed"].get("storage_bytes", 0) != sum(
            len(text.encode("utf-8")) for text in runtime.state.files.values()
        ):
            raise WriterRuntimeError("storage counter differs from checkpoint files")
        return node, budget

    def _head(self, runtime: RuntimeHandle) -> tuple[str | None, str | None]:
        lineage = runtime.state.position["lineage_id"]
        head = self.store.read_head(lineage)
        if head is None:
            return None, runtime.checkpoint_id
        if self.store.load_commit(head).checkpoint != runtime.checkpoint_id:
            raise WriterRuntimeError("stale runtime handle: lineage head moved")
        return head, None

    @staticmethod
    def _effect(
        state: EnvironmentStateV1,
        *,
        changes: Mapping[str, Any],
        history: Mapping[str, Any] = (),
        delta: Mapping[str, Any] = (),
    ) -> dict[str, Any]:
        return {
            "artifact_type": "Phase2RecordedEffectV1",
            "before_state_ref": state.identity(),
            "file_delta": dict(delta),
            "set": dict(changes),
            "history_set": dict(history),
        }

    def _event(
        self,
        state: EnvironmentStateV1,
        kind: str,
        payload_ref: str,
        *,
        audience: tuple[str, ...],
        actor: str,
    ) -> EventV1:
        return EventV1(
            previous=state.history["head"],
            seq=state.history["seq"] + 1,
            lineage_id=state.position["lineage_id"],
            rollout_id=self.rollout_id,
            node_visit_id=state.position["visit_id"],
            kind=kind,
            actor=actor,
            audience=audience,
            payload_ref=payload_ref,
            versions_ref=state.versions_ref,
            provenance_ref=state.provenance_ref,
        )

    def _reduced(
        self, state: EnvironmentStateV1, event: EventV1, effect: dict[str, Any]
    ) -> EnvironmentStateV1:
        # Use the actual Phase 2 reducer, not a parallel state-transition model.
        return self.store._apply_recorded_effect_body(state, event, effect)

    def _ledger(self, state: EnvironmentStateV1) -> list[dict[str, Any]]:
        body = self.store.get_artifact(state.external_inputs_ref, expected_domain="payload")
        if isinstance(body, dict) and body.get("record_type") == "WriterRuntimeLogV1":
            if body.get("rollout_id") != self.rollout_id:
                raise WriterRuntimeError("restored runtime belongs to another rollout")
            return list(body["entries"])
        if state.history["action_ids"] or state.history["tool_result_ids"]:
            raise WriterRuntimeError("missing writer runtime log")
        return []

    def _execution_value(
        self,
        state: EnvironmentStateV1,
        context: ContextRevisionV1,
        budget: Mapping[str, Any],
    ) -> str:
        # Deliberately use values, not context/event/checkpoint/provenance identities.
        return domain_hash(
            "state",
            {
                "files": state.files,
                "position": {
                    key: state.position[key]
                    for key in ("node_id", "visit_id", "phase", "loop_counts", "lineage_id")
                },
                "messages": [message.to_dict() for message in context.messages],
                "tool_queue": state.continuation["tool_queue"],
                "next_call": state.continuation["next_call"],
                "action_ids": state.history["action_ids"],
                "tool_result_ids": state.history["tool_result_ids"],
                "budgets": budget,
                "requirements": _value_only(self.store.get_artifact(state.requirements_ref)),
                "decisions": _value_only(self.store.get_artifact(state.decisions_ref)),
                "disclosures": _value_only(self.store.get_artifact(state.disclosures_ref)),
                "outcome": _value_only(self.store.get_artifact(state.outcome_ref)),
                "rng": _value_only(self.store.get_artifact(state.rng_ref)),
            },
        )

    @staticmethod
    def _context(old: ContextRevisionV1, message: MessageV1, source: EventV1) -> ContextRevisionV1:
        return ContextRevisionV1(
            messages=(*old.messages, message),
            tools=old.tools,
            event_head=source.id,
            provenance_refs=(source.id,),
            rendering=old.rendering,
        )

    def _publish(
        self,
        runtime: RuntimeHandle,
        *,
        kind: str,
        actor: str,
        audience: tuple[str, ...],
        message: MessageV1,
        record: dict[str, Any],
        changes: Mapping[str, Any],
        history: Mapping[str, Any],
        delta: Mapping[str, Any] = (),
    ) -> WriterStepV1:
        state = runtime.state
        head, parent = self._head(runtime)
        record_ref = self.store.put_artifact(record)
        message_ref = self.store.persist(message)
        entries = self._ledger(state)
        entries.append(
            {
                "seq": state.history["seq"] + 1,
                "kind": kind,
                "record_ref": record_ref,
                "message_ref": message_ref,
            }
        )
        log_ref = self.store.put_artifact(
            {"record_type": "WriterRuntimeLogV1", "rollout_id": self.rollout_id, "entries": entries}
        )
        primary_changes = {**changes, "external_inputs_ref": log_ref}
        effect = self._effect(state, changes=primary_changes, history=history, delta=delta)
        effect_ref = self.store.put_artifact(effect)
        primary = self._event(state, kind, effect_ref, audience=audience, actor=actor)
        intermediate = self._reduced(state, primary, effect)
        context = self._context(runtime.context, message, primary)
        self.store.persist(context)
        context_effect = self._effect(intermediate, changes={"context_ref": context.identity()})
        context_effect_ref = self.store.put_artifact(context_effect)
        context_event = self._event(
            intermediate,
            "context_changed",
            context_effect_ref,
            audience=("controller", "trainer"),
            actor="environment",
        )
        final = self._reduced(intermediate, context_event, context_effect)
        extra_refs = tuple(
            record[name]
            for name in ("trace_ref", "request_ref", "raw_output_ref", "logprob_ref")
            if isinstance(record.get(name), str)
        )
        commit = self.store.publish(
            state.position["lineage_id"],
            head,
            (primary, context_event),
            final,
            parent_checkpoint=parent,
            artifact_refs=(
                record_ref,
                message_ref,
                log_ref,
                effect_ref,
                context_effect_ref,
                *extra_refs,
            ),
        )
        checkpoint_id = self.store.load_commit(commit).checkpoint
        fresh = runtime.workspace.parent / f"writer-{uuid.uuid4().hex}"
        restored = self.store.restore(checkpoint_id, fresh)
        return WriterStepV1(restored, commit, primary.id, record_ref)

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
        seen: set[str] = set(prior_raw_ids)
        atomic_reject = len(raw_calls) > 1 and any(
            isinstance(call, dict)
            and isinstance(call.get("function"), dict)
            and call["function"].get("name") == "ask_author"
            for call in raw_calls
        )
        for index, raw in enumerate(raw_calls):
            call_id = f"{rollout_id}:call:{ordinal}:{index}"
            raw_id = raw.get("id") if isinstance(raw, dict) else None
            function = raw.get("function") if isinstance(raw, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            reason = None
            if (
                not isinstance(raw_id, str)
                or not raw_id
                or not raw_id.isprintable()
                or any(ch.isspace() for ch in raw_id)
            ):
                reason = "Tool call needs an id"
            elif raw_id in seen:
                reason = "Duplicate tool call id"
            else:
                seen.add(raw_id)
            if not isinstance(name, str) or not name:
                reason = reason or "Invalid tool function"
                name = "invalid_call"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError):
                    reason = reason or "Invalid tool arguments JSON"
            if not isinstance(arguments, dict):
                reason = reason or "Tool arguments must be an object"
                arguments = {
                    "_invalid_raw": json.dumps(arguments, ensure_ascii=False, default=repr)
                }
            elif not all(
                isinstance(key, str) and isinstance(value, str) for key, value in arguments.items()
            ):
                reason = reason or "Tool arguments must be an object of strings"
                arguments = {
                    "_invalid_raw": json.dumps(arguments, ensure_ascii=False, default=repr)
                }
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
            if name not in allowed:
                reason = reason or "Tool is not available in this condition"
            if atomic_reject:
                reason = "Mixed control and file-tool batch is forbidden"
            queue.append({"call_id": call_id, "name": name, "arguments": arguments})
            metadata.append(
                {
                    "call_id": call_id,
                    "raw_id": raw_id if isinstance(raw_id, str) and raw_id.isprintable() else None,
                    "validation_error": reason,
                    "parsed_call_json": json.dumps(raw, ensure_ascii=True, default=repr),
                }
            )
        return queue, metadata

    def submit_action(
        self,
        runtime: RuntimeHandle,
        message: Mapping[str, Any],
        *,
        exact_request: Any | None = None,
        raw_output: str | bytes | None = None,
        trace: Mapping[str, Any] | None = None,
        usage: Mapping[str, int] | None = None,
    ) -> WriterStepV1:
        node, budget = self._check(runtime)
        state = runtime.state
        if state.position["phase"] != "ready_writer" or state.continuation["next_call"] != len(
            state.continuation["tool_queue"]
        ):
            raise WriterRuntimeError("writer action requires a ready, drained tool queue")
        if budget["consumed"].get("writer_turns", 0) >= budget["limits"]["writer_turns"]:
            raise WriterRuntimeError("writer-turn budget exhausted")
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
            for entry in self._ledger(state)
            if entry["kind"] == "writer_action"
            for call in self.store.get_artifact(entry["record_ref"])["calls"]
            if call["raw_id"] is not None
        }
        queue, call_metadata = self._parsed_calls(
            message.get("tool_calls") or [],
            self.rollout_id,
            action_ordinal,
            frozenset(node.contract.entry_contract.tool_allowlist),
            prior_raw_ids,
        )
        if not queue and not content.strip():
            raise WriterRuntimeError("final response must contain text")
        if usage is None:
            usage = {}
        if not isinstance(usage, Mapping) or any(
            name in usage and (type(usage[name]) is not int or usage[name] < 0)
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise WriterRuntimeError("top-level token usage must contain nonnegative integers")
        # Provider detail fields are retained verbatim, never charged twice.
        canonical_json(usage)
        total_usage = usage.get(
            "total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
        for name, increment in (
            ("generated_tokens", usage.get("completion_tokens", 0)),
            ("total_tokens", total_usage),
        ):
            limit = budget["limits"].get(name)
            if limit is not None and budget["consumed"].get(name, 0) + increment > limit:
                raise WriterRuntimeError(f"{name} budget exhausted before action commit")
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
        if "context_tokens" in budget["limits"]:
            if trace is None or any(
                type(trace.get(name)) is not int or trace[name] < 0
                for name in ("input_tokens", "reserved_output_tokens")
            ):
                raise WriterRuntimeError("context-limited action requires exact token counts")
            if (
                trace["input_tokens"] + trace["reserved_output_tokens"]
                > budget["limits"]["context_tokens"]
            ):
                raise WriterRuntimeError("context capacity exceeded")
        # Exact request/rendering metadata is immutable before the action event.
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
        trace_body = {
            "record_type": "WriterActionTraceV1",
            "action_id": action_id,
            "context_content_hash": runtime.context.content_hash,
            "context_revision_ref": runtime.context.identity(),
            "rendering": dict(runtime.context.rendering),
            "exact_request_ref": request_ref,
            "raw_output_ref": raw_output_ref,
            "raw_output_evidence": "supplied" if raw_output is not None else "missing",
            "logprob_ref": trace.get("per_token_logprobs_ref") if trace is not None else None,
            "adapter_trace": dict(trace) if trace is not None else None,
            "token_evidence": "supplied"
            if trace is not None and "generated_token_ids" in trace
            else "missing",
            "logprob_evidence": "supplied"
            if trace is not None and "per_token_logprobs_ref" in trace
            else "missing",
            "native_on_policy_eligible": False,
            "reason": "native token alignment and loss masks are not implemented in Phase 4",
        }
        trace_ref = self.store.put_artifact(trace_body)
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
        next_budget = json.loads(canonical_json(budget))
        consumed = next_budget["consumed"]
        consumed["writer_turns"] = consumed.get("writer_turns", 0) + 1
        consumed["model_calls"] = consumed.get("model_calls", 0) + 1
        consumed["generated_tokens"] = consumed.get("generated_tokens", 0) + usage.get(
            "completion_tokens", 0
        )
        consumed["total_tokens"] = consumed.get("total_tokens", 0) + total_usage
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
            "raw_output_ref": raw_output_ref,
            "logprob_ref": trace_body["logprob_ref"],
            "calls": call_metadata,
            "usage": dict(usage),
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
        return self._publish(
            runtime,
            kind="writer_action",
            actor="writer",
            audience=("controller", "trainer", "writer"),
            message=assistant,
            record=record,
            changes={"continuation": continuation, "position": position, "budgets_ref": budget_ref},
            history={"action_ids": [*state.history["action_ids"], action_id]},
        )

    def step_tool(self, runtime: RuntimeHandle) -> WriterStepV1:
        _, budget = self._check(runtime)
        state = runtime.state
        queue = state.continuation["tool_queue"]
        cursor = state.continuation["next_call"]
        if state.position["phase"] != "ready_writer" or cursor >= len(queue):
            raise WriterRuntimeError("no queued tool call to resume")
        entries = self._ledger(state)
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
        before_bytes = sum(len(text.encode("utf-8")) for text in state.files.values())
        observation: dict[str, Any]
        files = dict(state.files)
        read_charge = 0
        if budget["consumed"].get("tool_calls", 0) >= budget["limits"]["tool_calls"]:
            observation = {"ok": False, "valid": True, "error": "Tool-call budget exceeded"}
        elif metadata["validation_error"] is not None:
            observation = {"ok": False, "valid": False, "error": metadata["validation_error"]}
        else:
            stage = Path(tempfile.mkdtemp(prefix="writer-stage-", dir=runtime.workspace.parent))
            try:
                for path, text in files.items():
                    target = stage / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(text.encode("utf-8"))
                workspace = Workspace(
                    stage,
                    self.store.max_file_bytes,
                    min(self.store.max_workspace_bytes, budget["limits"]["storage_bytes"]),
                )
                observation = dispatch(workspace, call["name"], dict(call["arguments"]))
                if not observation["ok"] and "error" in observation:
                    observation["error"] = observation["error"].replace(str(stage), "<workspace>")
                if observation["ok"] and call["name"] in _READ_TOOLS:
                    charge = self.count_tokens(
                        json.dumps(observation["result"], ensure_ascii=False)
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
                        files = workspace.snapshot()
                elif observation["ok"]:
                    files = workspace.snapshot()
            finally:
                shutil.rmtree(stage)
        after_bytes = sum(len(text.encode("utf-8")) for text in files.values())
        delta = {
            path: {"before": state.files.get(path), "after": files.get(path)}
            for path in sorted(set(state.files) | set(files))
            if state.files.get(path) != files.get(path)
        }
        next_budget = json.loads(canonical_json(budget))
        consumed = next_budget["consumed"]
        consumed["attempted_tool_calls"] = consumed.get("attempted_tool_calls", 0) + 1
        consumed["tool_calls"] = min(
            consumed.get("tool_calls", 0) + 1, budget["limits"]["tool_calls"]
        )
        consumed["read_tokens"] = consumed.get("read_tokens", 0) + read_charge
        consumed["storage_bytes"] = after_bytes
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
            "before_execution_hash": self._execution_value(state, runtime.context, budget),
            "after_execution_hash": self._execution_value(
                prospective_state, prospective, next_budget
            ),
            "budget_charge": {
                "attempted_tool_calls": 1,
                "tool_calls": int(
                    budget["consumed"].get("tool_calls", 0) < budget["limits"]["tool_calls"]
                ),
                "read_tokens": read_charge,
                "read_tokenizer": self.read_tokenizer,
                "file_bytes_before": before_bytes,
                "file_bytes_after": after_bytes,
                "file_byte_delta": after_bytes - before_bytes,
            },
            "loss_eligibility": {"tool_observation": False},
        }
        return self._publish(
            runtime,
            kind="tool_result",
            actor="environment",
            audience=("controller", "trainer", "writer"),
            message=tool_message,
            record=record,
            changes={"continuation": continuation, "position": position, "budgets_ref": budget_ref},
            history={"tool_result_ids": [*state.history["tool_result_ids"], result_id]},
            delta=delta,
        )

    def drain_tools(self, runtime: RuntimeHandle) -> RuntimeHandle:
        """Resume only uncommitted queued calls; each iteration publishes one result."""
        while runtime.state.continuation["next_call"] < len(
            runtime.state.continuation["tool_queue"]
        ):
            runtime = self.step_tool(runtime).runtime
        return runtime

    def stop_exhausted(self, runtime: RuntimeHandle) -> WriterStepV1:
        """Seal a tool-only last turn that cannot produce another writer action.

        A final assistant reply is in ``checking`` and cannot take this path: its
        checks must run before anyone decides whether the task was complete.
        """
        _, budget = self._check(runtime)
        state = runtime.state
        if (
            state.position["phase"] != "ready_writer"
            or state.continuation["next_call"] != len(state.continuation["tool_queue"])
            or budget["consumed"].get("writer_turns", 0) < budget["limits"]["writer_turns"]
        ):
            raise WriterRuntimeError("writer budget is not exhausted at a drained boundary")
        head, parent = self._head(runtime)
        outcome = {
            "schema": 1,
            "task_status": "incomplete",
            "execution_status": "valid",
            "stop_reason": "writer_budget",
            "reward_status": "pending",
            "training_eligibility": "pending",
        }
        outcome_ref = self.store.put_artifact(outcome)
        position = state.to_dict()["position"]
        position["phase"] = "terminal"
        effect = self._effect(state, changes={"position": position, "outcome_ref": outcome_ref})
        effect_ref = self.store.put_artifact(effect)
        event = self._event(
            state,
            "termination_recorded",
            effect_ref,
            audience=("controller", "evaluator", "trainer"),
            actor="environment",
        )
        final = self._reduced(state, event, effect)
        commit = self.store.publish(
            state.position["lineage_id"],
            head,
            (event,),
            final,
            parent_checkpoint=parent,
            artifact_refs=(effect_ref, outcome_ref),
        )
        checkpoint_id = self.store.load_commit(commit).checkpoint
        fresh = runtime.workspace.parent / f"writer-{uuid.uuid4().hex}"
        return WriterStepV1(self.store.restore(checkpoint_id, fresh), commit, event.id, outcome_ref)
