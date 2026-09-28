"""Synchronous rollout sequencing over verified single-event commits."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from writing_agent.task_graph_controller import Directive, next_step
from writing_agent.task_graph_errors import DriverBudgetError
from writing_agent.task_graph_gatherers import Gatherers
from writing_agent.task_graph_records import EnvironmentStepV1
from writing_agent.task_graph_rollout_env import (
    AuthorInput,
    CheckInput,
    PortInput,
    RolloutEnvironment,
    RuntimeHandle,
    SamplerInput,
    ToolInput,
)

Alternative = Callable[[Directive, PortInput | None], Any | None]


def _no_alternative(_directive: Directive, _port: PortInput | None) -> None:
    return None


class RolloutDriver:
    """Gather one legal input and commit it before asking the controller again."""

    def __init__(
        self,
        env: RolloutEnvironment,
        gatherers: Gatherers,
        alternatives: Alternative = _no_alternative,
    ) -> None:
        self.env = env
        self.gatherers = gatherers
        self.alternatives = alternatives

    def run(self, runtime: RuntimeHandle, *, max_steps: int) -> RuntimeHandle:
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        steps = 0
        while True:
            # Each environment call owns and closes its operation scope. Never keep
            # store.operation() open while an adapter or caller callback runs.
            view = self.env.verify(runtime)
            directive = next_step(view)
            if directive.kind in {"done", "halt"}:
                return runtime
            if steps >= max_steps:
                raise DriverBudgetError(max_steps)

            port = self._port_input(view, directive)
            input_record = self.alternatives(directive, port) if directive.alternatives else None
            if input_record is None:
                input_record = self._gather(directive, port)
            runtime = self.env.commit(runtime, input_record).runtime
            steps += 1

    def _port_input(self, view: Any, directive: Directive) -> PortInput | None:
        if directive.kind not in {
            "sample_writer",
            "execute_tool",
            "await_author_reply",
            "await_check_result",
        }:
            return None
        return self.env.port_input(view, directive)

    def _gather(self, directive: Directive, port: PortInput | None) -> Any:
        match directive.kind:
            case "sample_writer":
                if not isinstance(port, SamplerInput):
                    raise TypeError("sample_writer requires SamplerInput")
                return self.gatherers.sampler.turn(port)
            case "execute_tool":
                if not isinstance(port, ToolInput):
                    raise TypeError("execute_tool requires ToolInput")
                return self.gatherers.tools.observe(port)
            case "await_author_reply":
                if not isinstance(port, AuthorInput):
                    raise TypeError("await_author_reply requires AuthorInput")
                return self.gatherers.author.reply(port)
            case "await_check_result":
                if not isinstance(port, CheckInput):
                    raise TypeError("await_check_result requires CheckInput")
                return self.gatherers.evaluator.result(port)
            case _:
                return _environment_step(directive)


def _environment_step(directive: Directive) -> EnvironmentStepV1:
    """Translate only controller-owned operations to their exact wire directive."""
    body: dict[str, Any] = {"kind": directive.kind}
    if directive.kind == "request_author":
        body["source"] = directive.source
    elif directive.kind == "commit_transition":
        body["edge_id"] = directive.edge_id
    elif directive.kind == "seal_outcome":
        body.update(task_status=directive.task_status, stop_reason=directive.stop_reason)
    elif directive.kind == "stop_exhausted":
        body["stop_reason"] = directive.stop_reason
    return EnvironmentStepV1(directive=body)


__all__ = ["Alternative", "RolloutDriver"]
