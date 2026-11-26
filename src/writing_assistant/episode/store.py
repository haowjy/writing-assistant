"""Task rows and episodes on disk, split assignment, and the local_dev guard.

Task rows and saved episodes are files another process (or an older build) wrote, so
loading validates them; everything after loading trusts the types.

Where task rows live (they hold book text, so never in git):
- public rows: `TASKS_DIR/<suite>.jsonl`, written by scripts/build_seed_tasks.py, each row
  `train`/`dev`/`test` by `split_of`;
- personal rows: `LOCAL_DEV_DIR/tasks/<suite>.jsonl`. The local_dev loader stamps every row
  there `split = "local_dev"` whatever the file says; nothing hashes, syncs or exports them.
  A public file carrying a `local_dev` row is refused.

`local_only()` is the one guard on `local_dev` text. Reading it (these rows, or a
`local_dev` work's text in `Corpus.text`) outside the guard raises `LocalOnly`, and inside
it every model call to an endpoint without `local = true` raises `LocalOnly` before
anything is sent (`llm.client`). So text that was read was read under the guard.
`WA_DATA` overrides the data root (tests, another machine).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from writing_assistant.episode.model import Episode

DATA = Path(os.environ.get("WA_DATA", "/data/jimyao/research/writing-assistant"))
TASKS_DIR = DATA / "tasks"
LOCAL_DEV_DIR = DATA / "local_dev"
LOCAL_DEV = "local_dev"
SPLITS = ("train", "dev", "test")
# split_of buckets: sha256(source_group) mod 100 -> train [0, 70), dev [70, 85), test [85, 100).
# Changing these moves rows between splits; the held-out test split must never move.
SPLIT_EDGES = (70, 85)
EPISODES_FILE, RUN_FILE = "episodes.jsonl", "run.json"  # in a run directory (cli.py)
# Every run on the held-out `test` split, one line each: the record that "never used for
# selection" can be checked against. Not in TASKS_DIR: it names local paths and is no task.
TEST_USES = DATA / "test-split-uses.jsonl"


_LOCAL_ONLY: contextvars.ContextVar[bool] = contextvars.ContextVar("local_only", default=False)


class LocalOnly(RuntimeError):
    """local_dev text was read outside `local_only()`, or a hosted endpoint was asked to
    receive it."""


@contextmanager
def local_only(active: bool = True) -> Iterator[None]:
    """Within this block (and the asyncio tasks it starts), local_dev text may be read and
    only `local = true` endpoints may be called. cli.py enters it around everything that
    handles a local_dev row."""
    token = _LOCAL_ONLY.set(active or _LOCAL_ONLY.get())
    try:
        yield
    finally:
        _LOCAL_ONLY.reset(token)


def under_local_only() -> bool:
    return _LOCAL_ONLY.get()


def require_local_only(what: str) -> None:
    if not _LOCAL_ONLY.get():
        raise LocalOnly(f"{what} is local_dev text: read it only inside episode.store.local_only()")


@dataclass(frozen=True)
class Task:
    task_id: str
    suite: str
    kind: str
    split: str  # its works' split; a task mounting two works needs both in one split
    work_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


def split_of(source_group: str) -> str:
    """"train" | "dev" | "test", stable across runs and machines. `local_dev` rows are
    stamped by the local_dev loader, never hashed."""
    key = source_group.strip().casefold().encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100
    return "train" if bucket < SPLIT_EDGES[0] else "dev" if bucket < SPLIT_EDGES[1] else "test"


def assert_train_split(rows: Iterable[Any]) -> None:
    """Refuse anything that is not `train`: the guard every training loader calls."""
    bad = [_get(r, "task_id") or _get(r, "work_id") or repr(r)[:60] for r in rows if _get(r, "split") != "train"]
    if bad:
        raise ValueError(f"{len(bad)} non-train rows reached a training path: {bad[:5]}")


def _get(row: Any, key: str) -> Any:
    return row.get(key) if isinstance(row, dict) else getattr(row, key, None)


_CLAUSE_SEP = re.compile(r",(?=\s*\w+=)")


def matches(row: dict[str, Any], where: str | None) -> bool:
    """`where` is comma-separated `key=value` equality tests (`kind=pov_write,source=gutenberg`);
    a key is looked up in the row, then in its `payload`. Values compare as strings. A comma
    starts a new test only when `key=` follows it, so `author=Austen, Jane` is one test."""
    if not where:
        return True
    for clause in _CLAUSE_SEP.split(where):
        key, sep, value = clause.partition("=")
        if not sep:
            raise ValueError(f"bad where clause {clause!r}: expected key=value")
        key = key.strip()
        got = row[key] if key in row else row.get("payload", {}).get(key)
        if got is None or str(got) != value.strip():
            return False
    return True


def task_rows(suite: str, split: str, where: str | None = None) -> list[Task]:
    """A suite's task rows for one split: `local_dev` from the local_dev loader, the rest from
    the public rows. No file means no rows."""
    if split == LOCAL_DEV:
        return load_local_dev(suite, where)
    path = TASKS_DIR / f"{suite}.jsonl"
    return load_rows(path, suite=suite, split=split, where=where) if path.exists() else []


def load_local_dev(suite: str, where: str | None = None, root: Path | None = None) -> list[Task]:
    """Rows under `local_dev/tasks/<suite>.jsonl`, every one stamped `split = "local_dev"`."""
    require_local_only(f"local_dev task rows for suite {suite}")
    path = (root or LOCAL_DEV_DIR) / "tasks" / f"{suite}.jsonl"
    if not path.exists():
        return []
    rows = [r | {"split": LOCAL_DEV} for r in read_jsonl(path)]
    return _tasks(rows, path, suite, LOCAL_DEV, where, allowed=(LOCAL_DEV,))


def load_rows(path: str | Path, suite: str | None = None, split: str | None = None, where: str | None = None) -> list[Task]:
    """Public task rows from a JSONL file, filtered by suite, split and `where`. A
    `local_dev` row here is refused: those come only from the local_dev loader."""
    path = Path(path)
    return _tasks(read_jsonl(path), path, suite, split, where, allowed=SPLITS)


def _tasks(raw: list[dict], path: Path, suite: str | None, split: str | None, where: str | None,
           allowed: tuple[str, ...]) -> list[Task]:
    names = {f.name for f in fields(Task)}
    tasks = []
    for n, row in enumerate(raw, 1):
        missing = {"task_id", "suite", "kind", "split"} - set(row)
        if missing or row["split"] not in allowed:
            raise ValueError(f"{path}:{n}: bad task row (missing {sorted(missing)}, split {row.get('split')!r})")
        if (suite is None or row["suite"] == suite) and (split is None or row["split"] == split) and matches(row, where):
            tasks.append(Task(**{k: v for k, v in row.items() if k in names}))
    return tasks


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSONL file another process may have been writing: [] when it does not exist,
    and a last line without its newline (a write cut short) is ignored."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines[:-1] if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]], keep: int | None = None) -> None:
    """Append rows, one per line, after cutting the file back to its first `keep` bytes
    (default: its complete lines), so a line a killed writer left unfinished is not glued
    to the next one."""
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with path.open("ab") as f:
        f.truncate(_complete_bytes(path) if keep is None else keep)
        f.write(text.encode("utf-8"))


def _complete_bytes(path: Path) -> int:
    """Bytes up to the end of the last complete line; reads only the last byte when it is a
    newline, as it is after every write that finished."""
    if not path.exists() or (size := path.stat().st_size) == 0:
        return 0
    with path.open("rb") as f:
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return size
        f.seek(0)
        return f.read().rfind(b"\n") + 1


_TASK_ID = re.compile(r'"task_id": "((?:[^"\\]|\\.)*)"')


def _episodes_bytes(path: Path) -> int:
    """How much of episodes.jsonl holds whole tasks. A task's episodes are written in one
    append, so a last line cut short means its task is incomplete: the earlier lines of that
    task are dropped with it, and a resume plays the task again."""
    keep = _complete_bytes(path)
    if keep == path.stat().st_size:
        return keep
    data = path.read_bytes()
    m = _TASK_ID.search(data[keep:].decode("utf-8", "replace"))
    if m is None:
        return keep
    lines = data[:keep].split(b"\n")[:-1]
    while lines and json.loads(lines[-1]).get("task_id") == json.loads(f'"{m.group(1)}"'):
        keep -= len(lines.pop()) + 1
    return keep


def save_episodes(episodes: Iterable[Episode], out: str | Path) -> Path:
    """Append one task's episodes to `<out>/episodes.jsonl`, one complete episode per line,
    first cutting off an incomplete task a killed run left at the end."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / EPISODES_FILE
    append_jsonl(path, (ep.to_dict() for ep in episodes), _episodes_bytes(path) if path.exists() else 0)
    return path


def load_episodes(run_dir: str | Path) -> list[Episode]:
    """Every episode saved in a run, parent_id/condition intact for re-pairing twins; an
    incomplete task at the end (a run killed mid-write) is left out, as a resume replays it."""
    path = Path(run_dir) / EPISODES_FILE
    if not path.exists():
        return []
    lines = path.read_bytes()[: _episodes_bytes(path)].decode("utf-8").split("\n")[:-1]
    episodes = [Episode.from_dict(json.loads(line)) for line in lines if line.strip()]
    seen: set[str] = set()
    for ep in episodes:
        if ep.ep_id in seen:
            raise ValueError(f"{path}: episode {ep.ep_id} saved twice")
        seen.add(ep.ep_id)
    return episodes
