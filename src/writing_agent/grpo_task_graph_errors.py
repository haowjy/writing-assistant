"""Failure types for native task-graph GRPO admission."""


class TaskGraphTrainingError(RuntimeError):
    """Task-graph rollout or admission stopped before an optimizer update."""


class TaskGraphResumeRefused(TaskGraphTrainingError):
    """A saved group/reservation makes ordinary checkpoint resume unsafe."""


class TaskGraphResumeLocationRefused(TaskGraphResumeRefused):
    """A resume checkpoint is outside the experiment output directory."""


class TaskGraphGroupPending(TaskGraphTrainingError):
    """A native task-graph group is pending or invalid and cannot be resampled."""
