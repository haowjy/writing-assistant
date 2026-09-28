"""Exception types shared by the task-graph runtime.

This module is deliberately a leaf: it defines runtime failures without importing
any other task-graph module.
"""


class ProjectionError(ValueError):
    """A causal event chain cannot be safely rendered as writer context."""


class AdapterContractProjectionError(ProjectionError):
    """A recorded input violates a contract that the producer must satisfy."""


class VerifiedMessagesStaleError(ProjectionError):
    """A verified request no longer describes the active visible messages."""


class WriterRuntimeError(ValueError):
    """The supplied action or restored state violates the graph runtime contract."""


class StoreError(RuntimeError):
    """Base class for persistence failures."""


class MissingReferenceError(StoreError):
    """An immutable reference is absent."""


class CorruptRecordError(StoreError):
    """Stored bytes do not decode to the identity named by their path."""


class WrongRecordDomainError(CorruptRecordError):
    """A valid immutable object was used in a reference of the wrong type."""


class ConcurrentUpdateError(StoreError):
    """The lineage head did not match the compare-and-swap request."""


class MaterializationError(StoreError):
    """A checkpoint could not be safely materialized."""


class AdapterContractError(RuntimeError):
    """An adapter returned output that violates its declared contract."""


class DriverBudgetError(RuntimeError):
    """The rollout driver reached its operational step limit before halting."""

    def __init__(self, max_steps: int, runtime: object) -> None:
        self.max_steps = max_steps
        self.runtime = runtime
        super().__init__(f"rollout driver exceeded max_steps={max_steps}")


__all__ = [
    "AdapterContractError",
    "AdapterContractProjectionError",
    "ConcurrentUpdateError",
    "CorruptRecordError",
    "DriverBudgetError",
    "MaterializationError",
    "MissingReferenceError",
    "ProjectionError",
    "StoreError",
    "VerifiedMessagesStaleError",
    "WrongRecordDomainError",
    "WriterRuntimeError",
]
