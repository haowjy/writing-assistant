"""`Work` rows, the corpus index and text cache on disk, and the public export.

On disk, under one cache directory (never committed, never synced whole):

    works.jsonl             one Work per line, append-only; the last line for a work_id wins
    text/ab/abcdef...       the bytes as fetched, named by their sha256 (`content_hash`)

Personal works live apart, in the same layout under `local_dev/corpus/` (store.LOCAL_DEV_DIR):
the loader stamps each of them `split = "local_dev"` whatever its row says, and their text is
read from there. `select` leaves them out unless asked; export never writes them.

A collector writes the text first, then appends the row, so an interrupted run never leaves
a row without its text; a re-run appends a row only when something in it changed.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from writing_assistant.corpus import overlap as _overlap
from writing_assistant.corpus.text import decode, gutenberg_body, split_chapters
from writing_assistant.episode.store import DATA, LOCAL_DEV, LOCAL_DEV_DIR, matches

if TYPE_CHECKING:
    from writing_assistant.episode import Episode
    from writing_assistant.judges import Judges

CACHE_DIR = DATA / "corpus-cache"
INDEX_FILE = "works.jsonl"
# Licences whose text may leave this machine. Anything else, a missing licence included,
# leaves as url + content_hash only.
PERMISSIVE = frozenset({"public-domain", "cc-by", "cc-by-sa", "author-granted"})


@dataclass(frozen=True)
class Work:
    work_id: str  # "<source>:<id at the source>", e.g. "gutenberg:1342"
    source: str
    source_group: str  # the author for fiction (first listed); `split` hashes from it
    title: str
    licence: str | None  # one of PERMISSIVE, "copyrighted", or None when the source does not say
    url: str  # where the cached bytes came from
    content_hash: str  # sha256 of those bytes
    split: str  # split_of(source_group), or "local_dev" stamped by the local_dev loader
    authors: list[str] = field(default_factory=list)
    familiarity: float | None = None  # filled by Corpus from familiarity_parts; never stored
    familiarity_parts: dict[str, float] = field(default_factory=dict)  # e.g. {"downloads": 182729}
    content_rating: str | None = None  # the source's own rating; no content filter anywhere
    published_date: str | None = None
    snapshot_date: str | None = None
    tags: list[str] = field(default_factory=list)
    synopsis: str | None = None
    synopsis_source: str | None = None  # "author" | "catalogue" | "generated"
    seed_papers: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        del row["familiarity"]
        return row


class Corpus:
    """The works index and text cache under `cache_dir`. Built once in cli.py and carried in
    `Services.corpus`; `judges` is for build-time callers (authors.py, kb.py)."""

    def __init__(self, cache_dir: Path = CACHE_DIR, judges: Judges | None = None,
                 local_dev_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.local_dev_dir = Path(local_dev_dir) if local_dev_dir else LOCAL_DEV_DIR / "corpus"
        self.judges = judges
        self._works: dict[str, Work] | None = None
        self._texts: dict[str, str] = {}
        self._shingles: dict[str, frozenset[int]] = {}

    # -- index ------------------------------------------------------------------------------

    @property
    def works(self) -> dict[str, Work]:
        if self._works is None:
            local = {i: replace(w, split=LOCAL_DEV) for i, w in _load_index(self.local_dev_dir / INDEX_FILE).items()}
            public = _load_index(self.cache_dir / INDEX_FILE)
            if clash := local.keys() & public.keys():
                raise ValueError(f"work ids in both the public and the local_dev corpus: {sorted(clash)[:5]}")
            self._works = _with_familiarity(public | local)
        return self._works

    def get(self, work_id: str) -> Work:
        return self.works[work_id]

    def select(self, where: str | None = None, include_local_dev: bool = False) -> list[Work]:
        """Works matching `where` (comma-separated `key=value`, as `episode.store.matches`), in
        work_id order; `local_dev` works only when asked."""
        return [
            w
            for _, w in sorted(self.works.items())
            if (include_local_dev or w.split != "local_dev") and matches(asdict(w), where)
        ]

    def store_text(self, raw: bytes) -> str:
        """Put `raw` in the cache under its sha256 and return that `content_hash`."""
        digest = hashlib.sha256(raw).hexdigest()
        path = self._text_path(digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_bytes(raw)
            tmp.replace(path)
        return digest

    def add(self, work: Work) -> Work:
        """Record `work`, whose text must already be cached (`store_text`), unless an identical
        row is indexed. Never deletes: an older row for the same work_id stays on disk,
        shadowed by the new one, and its bytes stay in the cache."""
        if not self.has_text(work):
            raise ValueError(f"{work.work_id}: no cached text for {work.content_hash}")
        old = self.works.get(work.work_id)
        if old is None or old.to_row() != work.to_row():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with (self.cache_dir / INDEX_FILE).open("a", encoding="utf-8") as f:
                f.write(json.dumps(work.to_row(), ensure_ascii=False) + "\n")
            self._works = None  # familiarity percentiles move with every addition
        return self.works[work.work_id]

    def has_text(self, work: Work) -> bool:
        return self._text_path(work.content_hash).exists()

    def _text_path(self, digest: str) -> Path:
        return self.cache_dir / "text" / digest[:2] / digest

    # -- text -------------------------------------------------------------------------------

    def text(self, work: Work) -> str:
        """The work's text from the cache: decoded, and for Gutenberg the book between the
        licence header and footer."""
        if work.content_hash not in self._texts:
            root = self.local_dev_dir if work.split == LOCAL_DEV else self.cache_dir
            raw = (root / "text" / work.content_hash[:2] / work.content_hash).read_bytes()
            text = decode(raw)
            self._texts[work.content_hash] = gutenberg_body(text) if work.source == "gutenberg" else text
        return self._texts[work.content_hash]

    def chapters(self, work: Work) -> list[str]:
        """The one chapter splitter every caller uses (`text.split_chapters`)."""
        return split_chapters(self.text(work))

    # -- overlap ----------------------------------------------------------------------------

    def overlap(self, output: str, ep: Episode) -> dict[str, tuple[int, float]]:
        """`context_copy`, `tool_copy`, `memorization` -> (longest shared run in words, share of
        output words inside a shared 8-gram). `memorization` is absent when the episode names
        no work (`ep.state["work_ids"]`). See `overlap.py` for who-wrote-what."""
        works = [self.get(w) for w in ep.state.get("work_ids", [])]
        return _overlap.overlap(output, ep, [self._work_shingles(w) for w in works] if works else None)

    def _work_shingles(self, work: Work) -> frozenset[int]:
        if work.content_hash not in self._shingles:
            self._shingles[work.content_hash] = _overlap.shingles(self.text(work))
        return self._shingles[work.content_hash]

    flag = staticmethod(_overlap.flag)


def _load_index(path: Path) -> dict[str, Work]:
    """The index is a file earlier runs wrote, so rows are checked here. A final line without
    its newline is a write cut short and is ignored."""
    if not path.exists():
        return {}
    names = {f.name for f in fields(Work)}
    works: dict[str, Work] = {}
    lines = path.read_text(encoding="utf-8").split("\n")
    for n, line in enumerate(lines[:-1], 1):  # lines[-1] is "" or an unterminated write
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {"work_id", "source", "source_group", "title", "url", "content_hash", "split"} - set(row)
        if missing:
            raise ValueError(f"{path}:{n}: work row missing {sorted(missing)}")
        works[row["work_id"]] = Work(**{k: v for k, v in row.items() if k in names and k != "familiarity"})
    return works


def _with_familiarity(works: dict[str, Work]) -> dict[str, Work]:
    """Per source, each part as a percentile over that source's works (rank-based, so the log
    scaling the design names changes nothing), and `familiarity` the mean of the parts a work
    has. Computed on load because every percentile moves when a work is added."""
    by_part: dict[tuple[str, str], list[float]] = {}
    for w in works.values():
        for part, value in w.familiarity_parts.items():
            by_part.setdefault((w.source, part), []).append(value)
    for values in by_part.values():
        values.sort()
    out = {}
    for work_id, w in works.items():
        pcts = []
        for part, value in w.familiarity_parts.items():
            values = by_part[(w.source, part)]
            # mid-rank percentile: ties share one value, a lone work sits at 0.5
            lo, hi = bisect.bisect_left(values, value), bisect.bisect_right(values, value)
            pcts.append((lo + hi) / 2 / len(values))
        out[work_id] = replace(w, familiarity=sum(pcts) / len(pcts) if pcts else None)
    return out


# -- export -----------------------------------------------------------------------------------


def export_public(corpus: Corpus, out: Path) -> Path:
    """`wa-bench export --public <out>`: every non-`local_dev` work as one JSONL row in
    `<out>/works.jsonl`, with the text inline only when its licence is in PERMISSIVE; the
    rest carry `url` and `content_hash` for the reader to fetch and check."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / INDEX_FILE
    with path.open("w", encoding="utf-8") as f:
        for work in corpus.select():
            row = asdict(work)
            if work.licence in PERMISSIVE:
                row["text"] = corpus.text(work)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
