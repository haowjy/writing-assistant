"""Leaf decorator for one re-entrant task-graph store operation."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

_Result = TypeVar("_Result")


def operation_scoped(method: Callable[..., _Result]) -> Callable[..., _Result]:
    """Run a store or role entry point inside its store's operation scope."""

    @wraps(method)
    def wrapped(owner: Any, *args: Any, **kwargs: Any) -> _Result:
        store = getattr(owner, "store", owner)
        with store.operation():
            return method(owner, *args, **kwargs)

    return wrapped
