"""Task rows and episodes on disk, and split assignment.

Task rows and saved episodes are files another process (or an older build) wrote, so
loading validates them; everything after loading trusts the types.

Where task rows live (they hold book text, so never in git):
- public rows: `TASKS_DIR/<suite>.jsonl`, written by scripts/build_seed_tasks.py, each row
  `train`/`dev`/`test` by `split_of`;
- personal rows: `LOCAL_DEV_DIR/tasks/<suite>.jsonl`. The local_dev loader stamps every row
  there `split = "local_dev"` whatever the file says; nothing hashes, syncs or exports them,
  and cli.py runs them inside `llm.client.local_only()`, so their text reaches no hosted
  endpoint. A public file carrying a `local_dev` row is refused.
`WA_DATA` overrides the data root (tests, another machine).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
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


def save_episodes(episodes: Iterable[Episode], out: str | Path) -> Path:
    """Append episodes to `<out>/episodes.jsonl`, one complete episode per line."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / EPISODES_FILE
    lines = "".join(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n" for ep in episodes)
    with path.open("a", encoding="utf-8") as f:
        f.write(lines)
    return path


def load_episodes(run_dir: str | Path) -> list[Episode]:
    """Every episode saved in a run, parent_id/condition intact for re-pairing twins."""
    path = Path(run_dir) / EPISODES_FILE
    episodes = [Episode.from_dict(d) for d in read_jsonl(path)]
    seen: set[str] = set()
    for ep in episodes:
        if ep.ep_id in seen:
            raise ValueError(f"{path}: episode {ep.ep_id} saved twice")
        seen.add(ep.ep_id)
    return episodes
