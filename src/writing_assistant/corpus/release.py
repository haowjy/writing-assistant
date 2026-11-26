"""What text may leave this machine: the licence rule, in one place.

Text from a work leaves (in `wa-bench export --public`, or to the private bucket by `wa-bench
sync`) only when the work's licence is in PERMISSIVE. Anything else, a missing licence or a
work the corpus does not know included, leaves as `url` + `content_hash` (a work) or as
`{"sha256": ...}` (text inside a task payload). `local_dev` rows, works and episodes never
leave.

- Works: `works.jsonl` with the text inline only for permissive works.
- Task rows: the payload fields that carry work text are marked per task kind
  (`Suite.text_fields`, paths such as `passage.text`, `exemplars[].text`, `files.*`). A
  marked field leaves as text only when every work the row draws on is permissive; the
  unmarked fields are the benchmark's own text and leave whole. A row draws on the works in
  `work_ids` and every `work_id` a payload passage names (a Suite 3 impostor is by another
  author and is not in `work_ids`). A row whose kind has no marks may leave only if all its
  works are permissive; otherwise export refuses it rather than guess which fields are text.
- Episodes, and the scores and judgments rows about them, leave whole or not at all: only
  when every work the episode draws on is permissive (DECISIONS #12: left out, not hashed).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from writing_assistant.episode.store import EPISODES_FILE, LOCAL_DEV, RUN_FILE, TASKS_DIR, read_jsonl

if TYPE_CHECKING:
    from writing_assistant.corpus.rows import Corpus

PERMISSIVE = frozenset({"public-domain", "cc-by", "cc-by-sa", "author-granted"})
WORKS_FILE = "works.jsonl"
TextFields = dict[str, list[str]]  # task kind -> marked payload paths


def works_in(value: Any) -> set[str]:
    """Every work a row or episode state draws on: its `work_ids` and every nested `work_id`."""
    found: set[str] = set()
    if isinstance(value, dict):
        found |= set(value.get("work_ids", []))
        if isinstance(value.get("work_id"), str):
            found.add(value["work_id"])
        for v in value.values():
            found |= works_in(v)
    elif isinstance(value, list):
        for v in value:
            found |= works_in(v)
    return found


def may_leave(corpus: Corpus, work_ids: set[str]) -> bool:
    """True when the text of every one of these works may leave the machine."""
    return all(w in corpus.works and corpus.works[w].split != LOCAL_DEV and corpus.works[w].licence in PERMISSIVE
               for w in work_ids)


def _hashed(value: Any, path: list[str]) -> Any:
    """`value` with the strings at `path` replaced by their sha256 (a missing path is skipped)."""
    if not path:
        if not isinstance(value, str):
            raise ValueError(f"a marked text field holds {type(value).__name__}, not text")
        return {"sha256": hashlib.sha256(value.encode()).hexdigest()}
    head, rest = path[0], path[1:]
    if head.endswith("[]"):
        key = head[:-2]
        if not isinstance(value, dict) or key not in value:
            return value
        return value | {key: [_hashed(v, rest) for v in value[key]]}
    if head == "*":
        return {k: _hashed(v, rest) for k, v in value.items()}
    if not isinstance(value, dict) or head not in value:
        return value
    return value | {head: _hashed(value[head], rest)}


def public_row(corpus: Corpus, row: dict, text_fields: TextFields) -> dict:
    """The task row as it may leave: marked fields hashed unless every work is permissive."""
    if may_leave(corpus, works_in(row)):
        return row
    if row["kind"] not in text_fields:
        raise ValueError(f"{row['task_id']}: kind {row['kind']!r} marks no text fields and draws on "
                         f"non-permissive works; mark them in its suite's text_fields")
    payload = row.get("payload", {})
    for field in text_fields[row["kind"]]:
        payload = _hashed(payload, field.split("."))
    return row | {"payload": payload}


def public_task_rows(corpus: Corpus, text_fields: TextFields, tasks_dir: Path = TASKS_DIR) -> dict[str, list[dict]]:
    """Every public task row (`<tasks_dir>/<suite>.jsonl`; local_dev rows live elsewhere and
    a stray one is dropped), as it may leave."""
    return {path.stem: [public_row(corpus, row, text_fields) for row in read_jsonl(path) if row.get("split") != LOCAL_DEV]
            for path in sorted(tasks_dir.glob("*.jsonl"))}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def export_public(corpus: Corpus, out: Path, text_fields: TextFields, tasks_dir: Path = TASKS_DIR) -> None:
    """`wa-bench export --public <out>`: `<out>/works.jsonl` (every non-local_dev work, text
    inline only when permissive; the rest carry `url` and `content_hash` for the reader to
    fetch and check) and `<out>/tasks/<suite>.jsonl`."""
    works = []
    for work in corpus.select():
        row = asdict(work)
        if may_leave(corpus, {work.work_id}):
            row["text"] = corpus.text(work)
        works.append(row)
    _write_jsonl(Path(out) / WORKS_FILE, works)
    for suite, rows in public_task_rows(corpus, text_fields, tasks_dir).items():
        _write_jsonl(Path(out) / "tasks" / f"{suite}.jsonl", rows)


def stage_sync(stage: Path, runs: Path, corpus: Corpus, text_fields: TextFields) -> Counter[str]:
    """What `wa-bench sync` uploads, staged under `stage`: the public export under `corpus/`
    and `tasks/`, and each run's files with episodes, scores and judgments filtered."""
    counts: Counter[str] = Counter()
    export_public(corpus, stage / "corpus", text_fields)
    shutil.move(stage / "corpus" / "tasks", stage / "tasks")
    for run_dir in sorted(p for p in runs.glob("*") if (p / RUN_FILE).exists()):
        dest = stage / "runs" / run_dir.name
        episodes = [d for d in read_jsonl(run_dir / EPISODES_FILE)
                    if d["split"] != LOCAL_DEV and may_leave(corpus, works_in(d["state"]))]
        counts["episodes_left_out"] += len(read_jsonl(run_dir / EPISODES_FILE)) - len(episodes)
        counts["episodes"] += len(episodes)
        keep = {d["ep_id"] for d in episodes}
        _write_jsonl(dest / EPISODES_FILE, episodes)
        for path in [*run_dir.glob("scores-*.jsonl"), *run_dir.glob("judgments-*.jsonl")]:
            _write_jsonl(dest / path.name, [r for r in read_jsonl(path)
                                            if r["ep_id"] in keep and r.get("scored", r["ep_id"]) in keep])
        for name in (RUN_FILE, "guardrails.json"):
            if (run_dir / name).exists():
                shutil.copy(run_dir / name, dest / name)
    return counts
