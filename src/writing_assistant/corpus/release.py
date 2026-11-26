"""What text may leave this machine: the licence rule, in one place.

Text from a work leaves (in `wa-bench export --public`, or to the private bucket by `wa-bench
sync`) only when the work's licence is in PERMISSIVE. Anything else, a missing licence or a
work the corpus does not know included, leaves as `url` + `content_hash` (a work) or as
`{"sha256": ...}` (text inside a task payload). `local_dev` rows, works and episodes never
leave.

- Works: `works.jsonl` with the text inline only for permissive works.
- Task rows: only `<tasks dir>/<suite>.jsonl` for the suites named, each line validated as
  a task row of that suite and of one of its kinds (anything else in the directory, a log or
  a cache, never leaves). The payload fields that carry work text are marked per task kind
  (`Suite.text_fields`, paths such as `passage.text`, `exemplars[].text`, `files.*`). A
  marked field leaves as text only when every work the row draws on is permissive; the
  unmarked fields are the benchmark's own text and leave whole. A row draws on the works in
  `work_ids` and every `work_id` a payload passage names (a Suite 3 impostor is by another
  author and is not in `work_ids`). A mark that is mistyped, or missing for a new field,
  would let text out silently, so a hashed row that still shares a `RUN`-word run with a
  work it draws on is refused.
- Episodes, and the score rows and judge items about them, leave whole or not at all: only
  when every work the episode draws on is permissive (DECISIONS #12: left out, not hashed).
  A run's store file never leaves: `sync` writes a new store holding the closed episodes
  that may, each with the ancestors it was forked from and its rows (`Store.copy`). The
  test is made of every segment in an episode's chain, over the state at each of its
  checkpoints (state changes along a chain, and an ancestor is copied whole), and one
  segment failing keeps the episode back. A judge item quotes the episode it judged, which
  for a twin's text is not the episode the row is about: the chains of the episodes an
  episode's items quote are tested with its own.
"""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from writing_assistant.corpus.text import norm_words
from writing_assistant.episode.model import strings
from writing_assistant.episode.segments import STORE_FILE, Store, has_store
from writing_assistant.episode.store import LOCAL_DEV, RUN_FILE, TASKS_DIR, load_rows, local_only

if TYPE_CHECKING:
    from writing_assistant.corpus.rows import Corpus

PERMISSIVE = frozenset({"public-domain", "cc-by", "cc-by-sa", "author-granted"})
WORKS_FILE = "works.jsonl"
TextFields = dict[str, list[str]]  # task kind -> marked payload paths
SuiteFields = dict[str, TextFields]  # suite name -> its kinds' marks (every kind it has)
# Words: a run this long shared with a work is that work's text (style/tasks.py's leak rule).
# 8 words is too short: a Suite 3 beat paraphrase keeps "to the Fifth Northumberland
# Fusiliers as Assistant Surgeon".
RUN = 12


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


def _runs(text: str) -> set[tuple[str, ...]]:
    words = norm_words(text)
    return {tuple(words[i : i + RUN]) for i in range(len(words) - RUN + 1)}


def public_row(corpus: Corpus, row: dict, text_fields: TextFields,
               work_runs: Callable[[str], set[tuple[str, ...]]]) -> dict:
    """The task row as it may leave: marked fields hashed unless every work is permissive.
    `work_runs(work_id)`: that work's `RUN`-word runs."""
    works = works_in(row)
    if may_leave(corpus, works):
        return row
    payload = row.get("payload", {})
    for field in text_fields[row["kind"]]:
        payload = _hashed(payload, field.split("."))
    runs = set().union(*map(_runs, strings(payload)))
    left = [w for w in sorted(works) if w in corpus.works and runs & work_runs(w)]
    if left:
        raise ValueError(f"{row['task_id']}: with its marked fields {text_fields[row['kind']]} hashed, the payload "
                         f"still holds text of {left[0]}; mark the field that carries it in its suite's text_fields")
    return row | {"payload": payload}


def public_task_rows(corpus: Corpus, suites: SuiteFields, tasks_dir: Path = TASKS_DIR) -> dict[str, list[dict]]:
    """Suite name -> its public task rows (`<tasks_dir>/<suite>.jsonl`), as they may leave.
    Refuses a line that is not a task row of that suite (a local_dev row included: those
    live elsewhere)."""
    @functools.cache
    def work_runs(work_id: str) -> set[tuple[str, ...]]:
        with local_only():  # compared on this machine; a local_dev work's text goes nowhere
            return _runs(corpus.text(corpus.get(work_id)))

    out = {}
    for suite, text_fields in suites.items():
        path = tasks_dir / f"{suite}.jsonl"
        if not path.exists():
            continue
        rows = []
        for task in load_rows(path):
            if task.suite != suite or task.kind not in text_fields:
                raise ValueError(f"{path}: {task.task_id} is not a task row of suite {suite} "
                                 f"(suite {task.suite!r}, kind {task.kind!r}; kinds {sorted(text_fields)})")
            rows.append(public_row(corpus, asdict(task), text_fields, work_runs))
        out[suite] = rows
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def export_public(corpus: Corpus, out: Path, suites: SuiteFields, tasks_dir: Path = TASKS_DIR) -> None:
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
    (Path(out) / "tasks").mkdir(exist_ok=True)  # there with no task file too: `stage_sync` moves it
    for suite, rows in public_task_rows(corpus, suites, tasks_dir).items():
        _write_jsonl(Path(out) / "tasks" / f"{suite}.jsonl", rows)


def stage_sync(stage: Path, runs: Path, corpus: Corpus, suites: SuiteFields) -> Counter[str]:
    """What `wa-bench sync` uploads, staged under `stage`: the public export under `corpus/`
    and `tasks/`, and of each run a new `store.db` holding the episodes that may leave and
    their rows, written here, never copied. Whatever `stage` held is replaced; a run
    directory this build does not read (`has_store`) stops the sync before that."""
    counts: Counter[str] = Counter()
    stored = {p: has_store(p) for p in sorted(runs.glob("*")) if (p / RUN_FILE).exists()}
    for run_dir in (p for p, has in stored.items() if has):
        Store(run_dir / STORE_FILE, readonly=True).db.close()  # opening it is the format check
    if stage.exists():
        shutil.rmtree(stage)
    export_public(corpus, stage / "corpus", suites)
    shutil.move(stage / "corpus" / "tasks", stage / "tasks")
    for run_dir in stored:
        dest = stage / "runs" / run_dir.name
        dest.mkdir(parents=True)
        if stored[run_dir]:
            with Store(run_dir / STORE_FILE, readonly=True) as store, Store(dest / STORE_FILE) as staged:
                def leaves(name: str) -> bool:
                    for seg in store.chain(name):
                        split, states = store.snapshots(seg)
                        if split == LOCAL_DEV or not all(may_leave(corpus, works_in(state)) for state in states):
                            return False
                    return True

                names, kept = store.names(), 0
                for ep_id in names:
                    # A run may be writing: an open ancestor of a closed episode can gain a
                    # checkpoint at any moment, and the episode a row. What is copied is
                    # what was tested, so both read one state of the file, and no answer about
                    # a segment is kept for the next episode. One episode at a time, so a
                    # run's write waits for one chain.
                    with store.reading():
                        quoted = {j["ep_id"] for j in store.judgments(scored=ep_id, superseded=True)}
                        if all(leaves(name) for name in sorted({ep_id} | quoted)):
                            store.copy([ep_id], staged)
                            kept += 1
                counts["episodes_left_out"] += len(names) - kept
                counts["episodes"] += kept
        for name in (RUN_FILE, "guardrails.json"):
            if (run_dir / name).exists():
                shutil.copy(run_dir / name, dest / name)
    return counts
