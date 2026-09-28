"""Synchronous rollout sequencing over verified single-event commits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from writing_agent.task_graph_controller import Directive
from writing_agent.task_graph_environment import (
    AuthorInput,
    CheckInput,
    PortInput,
    RolloutEnvironment,
    RuntimeHandle,
    SamplerInput,
    ToolInput,
)
from writing_agent.task_graph_errors import DriverBudgetError
from writing_agent.task_graph_gatherers import Gatherers
from writing_agent.task_graph_records import EnvironmentStepV1

Alternative = Callable[[Directive, PortInput | None], Any | None]


@dataclass(frozen=True)
class RunResult:
    """Small named result preserving whether execution finished or halted."""

    runtime: RuntimeHandle
    directive: Directive


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

    def run(self, runtime: RuntimeHandle, *, max_steps: int) -> RunResult:
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        steps = 0
        while True:
            # Each environment call owns and closes its operation scope. Never keep
            # store.operation() open while an adapter or caller callback runs.
            _view, directive, port = self.env.step_input(runtime)
            if directive.kind in {"done", "halt"}:
                return RunResult(runtime, directive)
            if steps >= max_steps:
                raise DriverBudgetError(max_steps, runtime)

            input_record = self.alternatives(directive, port) if directive.alternatives else None
            if input_record is None:
                input_record = self._gather(directive, port)
            runtime = self.env.commit(runtime, input_record).runtime
            steps += 1

    def _gather(self, directive: Directive, port: PortInput | None) -> Any:
        match port:
            case SamplerInput():
                return self.gatherers.sampler.turn(port)
            case ToolInput():
                return self.gatherers.tools.observe(port)
            case AuthorInput():
                return self.gatherers.author.reply(port)
            case CheckInput():
                return self.gatherers.evaluator.result(port)
            case None:
                return EnvironmentStepV1.of(directive)


__all__ = ["Alternative", "RolloutDriver", "RunResult"]
