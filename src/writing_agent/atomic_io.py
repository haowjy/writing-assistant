"""Atomic, durable file replacement shared by manifests and run evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_bytes(
    path: Path | str,
    data: bytes,
    *,
    mode: int = 0o600,
    create_parent: bool = False,
    parent_mode: int = 0o700,
    replace: bool = True,
) -> bool:
    """Atomically replace a file, fsyncing both the file and containing directory."""
    path = Path(path)
    if not isinstance(data, bytes):
        raise TypeError("atomic file contents must be bytes")
    if create_parent:
        path.parent.mkdir(mode=parent_mode, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, mode)
        stream = os.fdopen(descriptor, "wb", closefd=True)
        descriptor = -1
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary_path, path)
            written = True
        else:
            try:
                os.link(temporary_path, path)
                written = True
            except FileExistsError:
                written = False
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return written
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(
    path: Path | str,
    value: Any,
    *,
    canonical: bool = False,
    sort_keys: bool = True,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    allow_nan: bool = False,
    mode: int = 0o600,
    create_parent: bool = False,
    parent_mode: int = 0o700,
) -> None:
    """Serialize and atomically replace JSON using canonical or readable formatting."""
    if canonical:
        from writing_agent.task_graph import canonical_bytes

        encoded = canonical_bytes(value)
    else:
        encoded = (
            json.dumps(
                value,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                indent=indent,
                allow_nan=allow_nan,
            )
            + "\n"
        ).encode("utf-8", "strict")
    atomic_write_bytes(
        path,
        encoded,
        mode=mode,
        create_parent=create_parent,
        parent_mode=parent_mode,
    )


__all__ = ["atomic_write_bytes", "atomic_write_json"]
