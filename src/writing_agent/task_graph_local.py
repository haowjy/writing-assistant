"""Current local, offline implementations of the runtime ports."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from writing_agent.task_graph import ContextRevisionV1, canonical_json
from writing_agent.task_graph_ports import PortDescriptorV1, ToolExecution
from writing_agent.task_graph_sampling import (
    AdapterEvidenceV1,
    PreparedRequestV1,
    SamplingEvidenceV1,
)
from writing_agent.workspace import TOOL_SCHEMAS, Workspace

ASK_AUTHOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_author",
        "description": "Ask about declared public decision IDs. This must be the only tool call.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "decision_ids": {"type": "array", "items": {"type": "string"}},
                "proposals": {"type": "array", "items": {"type": "object"}},
                "option_refs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "decision_ids", "proposals", "option_refs"],
            "additionalProperties": False,
        },
    },
}


def writer_tool_schemas(
    allowlist: tuple[str, ...], interaction_policy=None
) -> tuple[dict[str, Any], ...]:
    schemas = tuple(schema for schema in TOOL_SCHEMAS if schema["function"]["name"] in allowlist)
    if "ask_author" not in allowlist:
        return schemas
    if interaction_policy is None:
        raise ValueError("ask_author schema requires admitted public decision declarations")
    schema = json.loads(canonical_json(ASK_AUTHOR_SCHEMA))
    declared = interaction_policy.public_decisions
    schema["function"]["description"] += " Public decisions: " + "; ".join(
        f"{item['id']}: {item['label']}" for item in declared
    )
    schema["function"]["parameters"]["properties"]["decision_ids"]["items"]["enum"] = [
        item["id"] for item in declared
    ]
    return (*schemas, schema)


def _graph_dispatch(workspace: Workspace, name: str, arguments: dict[str, str]) -> dict:
    """Legacy dispatch catches all OSError; this boundary must not hide I/O failure."""
    schemas = {schema["function"]["name"]: schema["function"] for schema in TOOL_SCHEMAS}
    if name not in schemas:
        return {"ok": False, "valid": False, "error": f"Unknown tool: {name}"}
    params = schemas[name]["parameters"]
    if set(arguments) - params["properties"].keys() or set(params["required"]) - arguments.keys():
        return {"ok": False, "valid": False, "error": "Tool arguments do not match schema"}
    try:
        return {"ok": True, "valid": True, "result": getattr(workspace, name)(**arguments)}
    except UnicodeError:
        raise
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError) as exc:
        return {"ok": False, "valid": True, "error": str(exc)}
    except FileExistsError as exc:
        # mkdir on a writer-selected child of an existing file is a path conflict.
        if name in {"write_file", "patch_file"} and "path" in arguments:
            parent = (workspace.root / arguments["path"]).parent
            if any(
                ancestor.is_file()
                for ancestor in (parent, *parent.parents)
                if ancestor != workspace.root
            ):
                return {"ok": False, "valid": True, "error": str(exc)}
        raise


class LocalSamplingHarness:
    descriptor = PortDescriptorV1("sampling", "caller-supplied-v1", "1")

    def prepared_request(
        self, context: ContextRevisionV1, payload_ref: str, *, verified: bool
    ) -> PreparedRequestV1:
        return PreparedRequestV1(
            "VerifiedWriterMessagesV1" if verified else "PreparedWriterRequestV1",
            context.content_hash,
            context.identity(),
            canonical_json(context.rendering),
            payload_ref,
        )

    def evidence(
        self,
        action_id,
        context,
        request_ref,
        prepared_request_ref,
        raw_output_ref,
        usage,
        adapter_trace,
    ):
        return SamplingEvidenceV1(
            action_id=action_id,
            context_content_hash=context.content_hash,
            context_revision_ref=context.identity(),
            rendering_json=canonical_json(context.rendering),
            exact_request_ref=request_ref,
            prepared_request_ref=prepared_request_ref,
            raw_output_ref=raw_output_ref,
            logprob_ref=adapter_trace.get("per_token_logprobs_ref")
            if adapter_trace is not None
            else None,
            usage_json=canonical_json(usage),
            model=adapter_trace.get("model") if adapter_trace is not None else None,
            seed=adapter_trace.get("seed") if adapter_trace is not None else None,
            adapter=AdapterEvidenceV1.from_wire(adapter_trace),
        )


class LocalTextToolProvider:
    descriptor = PortDescriptorV1("tools", "local-text-workspace-v1", "1")

    def schemas(self, allowlist: tuple[str, ...], interaction_policy=None):
        return writer_tool_schemas(allowlist, interaction_policy)

    def execute(
        self,
        files: Mapping[str, str],
        name: str,
        arguments: dict[str, str],
        *,
        stage_parent: Path,
        max_file_bytes: int,
        max_workspace_bytes: int,
    ) -> ToolExecution:
        stage = Path(tempfile.mkdtemp(prefix="writer-stage-", dir=stage_parent))
        try:
            for path, text in files.items():
                target = stage / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            workspace = Workspace(stage, max_file_bytes, max_workspace_bytes, strict_decode=True)
            observation = _graph_dispatch(workspace, name, arguments)
            if not observation["ok"] and "error" in observation:
                observation["error"] = observation["error"].replace(str(stage), "<workspace>")
            result_files = workspace.snapshot() if observation["ok"] else dict(files)
            return ToolExecution(observation=observation, files=result_files)
        finally:
            shutil.rmtree(stage)


class DeterministicEvaluator:
    descriptor = PortDescriptorV1("evaluator", "deterministic-file-checks-v1", "1")

    def evaluate(self, check, files: Mapping[str, str]) -> tuple[str, dict]:
        from writing_agent.task_graph_checks import deterministic_check

        return deterministic_check(check, files)
