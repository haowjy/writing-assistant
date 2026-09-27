"""Current local, offline implementations of the runtime ports."""

from __future__ import annotations

import base64
import shutil
import tempfile
from pathlib import Path

from writing_agent.task_graph import canonical_json, domain_hash
from writing_agent.task_graph_contracts import TOOL_SCHEMAS, writer_tool_schemas
from writing_agent.task_graph_ports import (
    EnvironmentAction,
    EnvironmentHandle,
    EnvironmentResult,
    EnvironmentSnapshot,
    EnvironmentSpec,
    PortDescriptorV1,
    SampleResult,
    ToolManifest,
)
from writing_agent.workspace import Workspace


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


class ScriptedSampleBackend:
    """Offline sample source; each invocation consumes one predetermined result."""

    def __init__(self, results):
        if not isinstance(results, (list, tuple)) or any(
            not isinstance(result, SampleResult) for result in results
        ):
            raise TypeError("scripted backend requires a finite sequence of SampleResult values")
        normalized = tuple(
            SampleResult(
                result.message, result.raw_output, result.usage, result.trace, result.logprobs
            )
            for result in results
        )
        script = [
            {
                "message": result.message,
                "raw_output": (
                    {"bytes_base64": base64.b64encode(result.raw_output).decode("ascii")}
                    if isinstance(result.raw_output, bytes)
                    else {"text": result.raw_output}
                ),
                "usage": result.usage,
                "trace": result.trace,
                **(
                    {
                        "logprobs": {
                            "bytes_base64": base64.b64encode(result.logprobs.data).decode("ascii"),
                            "codec": result.logprobs.codec,
                            "shape": list(result.logprobs.shape),
                        }
                    }
                    if result.logprobs is not None
                    else {}
                ),
            }
            for result in normalized
        ]
        self.descriptor = PortDescriptorV1(
            "sampling",
            "scripted-offline-v1",
            "1",
            canonical_json({"script_ref": domain_hash("payload", script)}),
        )
        self._results = iter(normalized)
        self.calls = 0

    def sample(self, prepared):
        self.calls += 1
        return next(self._results)


class LocalTextToolProvider:
    descriptor = PortDescriptorV1("tools", "local-text-workspace-v1", "1")

    def tool_manifest(self, allowlist: tuple[str, ...], interaction_policy=None):
        return ToolManifest(canonical_json(writer_tool_schemas(allowlist, interaction_policy)))

    def execute(
        self,
        spec: EnvironmentSpec,
        snapshot: EnvironmentSnapshot,
        action: EnvironmentAction,
    ) -> EnvironmentResult:
        files = snapshot.files()
        name = action.name
        arguments = action.arguments()
        stage = Path(tempfile.mkdtemp(prefix="writer-stage-"))
        try:
            for path, text in files.items():
                target = stage / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            workspace = Workspace(
                stage, spec.max_file_bytes, spec.max_workspace_bytes, strict_decode=True
            )
            observation = _graph_dispatch(workspace, name, arguments)
            if not observation["ok"] and "error" in observation:
                observation["error"] = observation["error"].replace(str(stage), "<workspace>")
            result_files = workspace.snapshot() if observation["ok"] else dict(files)
            return EnvironmentResult(observation, EnvironmentSnapshot.from_files(result_files))
        finally:
            shutil.rmtree(stage)


class DeterministicEvaluator:
    descriptor = PortDescriptorV1(
        "evaluator",
        "deterministic-file-checks-v1",
        "1",
        '{"family":"deterministic-file-v1"}',
    )

    family = "deterministic-file-v1"

    def evaluate(self, request):
        from writing_agent.task_graph_evaluation import produce_evaluation_evidence

        return produce_evaluation_evidence(request)


class LocalWorkspaceEnvironment:
    """Workspace execution composes a provider; persistence stays outside the port."""

    descriptor = PortDescriptorV1("environment", "local-workspace-v1", "1")

    def __init__(self, provider):
        self.provider = provider

    def tool_manifest(self, allowlist, interaction_policy=None):
        return self.provider.tool_manifest(allowlist, interaction_policy)

    def execute(self, spec, handle: EnvironmentHandle, snapshot, action):
        return self.provider.execute(spec, snapshot, action)
