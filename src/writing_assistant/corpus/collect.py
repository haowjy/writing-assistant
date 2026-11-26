"""Collectors: one function per source behind `collect()`, which `wa-bench collect` runs.

Slice 1 has the Gutenberg importer: metadata from the Gutendex API, text from the
`aleph.gutenberg.org` mirror, over plain httpx with a fixed delay between requests (the
search gateway's per-host buckets arrive in Slice 3). Never `www.gutenberg.org`: its robot
policy forbids automated access.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx

from writing_assistant.corpus.rows import Corpus, Work
from writing_assistant.corpus.text import decode, gutenberg_body
from writing_assistant.episode.store import split_of

if TYPE_CHECKING:
    from writing_assistant.search.gateway import SearchGateway

log = logging.getLogger(__name__)

GUTENDEX = "https://gutendex.com/books/"  # the trailing slash matters: without it, a redirect
MIRROR = "http://aleph.gutenberg.org"
CATALOGUE = f"{MIRROR}/cache/epub/feeds/pg_catalog.csv"
DELAY = 1.0  # seconds between requests, to every host
RETRIES = 3  # for 429, 5xx and transport errors, backing off 10, 20 delays (or Retry-After)
# Gutendex's stand-ins for "no single author": each such work is its own source_group.
NO_AUTHOR = frozenset({"anonymous", "various", "unknown"})


class _Polite:
    """GETs with a fixed delay between requests and a short, slow retry on 429/5xx. Every
    wait scales with `delay`, so `delay=0` (tests) never sleeps."""

    def __init__(self, http: httpx.Client, delay: float) -> None:
        self.http, self.delay, self._last = http, delay, float("-inf")

    def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        time.sleep(self.delay * (float(retry_after) if retry_after and retry_after.isdigit() else 10 * attempt))

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response | None:
        """The response, or None on 404. Raises once retries are spent."""
        for attempt in range(1, RETRIES + 1):
            time.sleep(max(0.0, self._last + self.delay - time.monotonic()))
            self._last = time.monotonic()
            try:
                resp = self.http.get(url, params=params)
            except httpx.TransportError as e:
                if attempt == RETRIES:
                    raise
                log.warning("%s: %s; retry %d", url, e, attempt)
                self._backoff(attempt)
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == RETRIES:
                    resp.raise_for_status()
                log.warning("%s: HTTP %d; retry %d", url, resp.status_code, attempt)
                self._backoff(attempt, resp.headers.get("retry-after"))
                continue
            resp.raise_for_status()
            return resp
        raise AssertionError("unreachable")


def mirror_urls(book_id: int) -> list[str]:
    """Candidate plain-text files on the mirror, best first: `-0.txt` is UTF-8, `-8.txt`
    ISO-8859-1, bare `.txt` ASCII. Directory: every digit but the last, then the id
    (1342 -> 1/3/4/1342/; single-digit ids live under 0/)."""
    digits = str(book_id)
    base = f"{MIRROR}/{'/'.join(digits[:-1]) or '0'}/{book_id}/{book_id}"
    return [f"{base}-0.txt", f"{base}-8.txt", f"{base}.txt"]


def gutenberg_work(book: dict[str, Any], *, url: str, content_hash: str, snapshot_date: str) -> Work:
    """A Gutendex record as a Work. The first listed author is the source_group, so a
    co-written book stays with its lead author's other works in one split."""
    authors = [a["name"] for a in book.get("authors") or []]
    named = [a for a in authors if a.casefold() not in NO_AUTHOR]
    work_id = f"gutenberg:{book['id']}"
    group = named[0] if named else work_id
    copyright_ = book.get("copyright")
    summaries = book.get("summaries") or []
    downloads = book.get("download_count")
    return Work(
        work_id=work_id,
        source="gutenberg",
        source_group=group,
        title=book.get("title") or "",
        licence={False: "public-domain", True: "copyrighted"}.get(copyright_),
        url=url,
        content_hash=content_hash,
        split=split_of(group),
        authors=authors,
        familiarity_parts={"downloads": float(downloads)} if downloads is not None else {},
        snapshot_date=snapshot_date,
        tags=[*(book.get("subjects") or []), *(book.get("bookshelves") or []),
              *(f"language:{lang}" for lang in book.get("languages") or [])],
        synopsis=summaries[0] if summaries else None,
        synopsis_source="catalogue" if summaries else None,
    )


def collect_gutenberg(corpus: Corpus, fetch: _Polite, query: dict[str, Any]) -> list[Work]:
    """Import the books a Gutendex query returns (`ids`, `search`, `languages`, `topic`, ...;
    plus `limit`), following its pagination. A book already cached gets its metadata
    refreshed without refetching the text; a book with no usable mirror file, or whose file
    still fails after the retries, is skipped and logged."""
    query = dict(query)
    limit = query.pop("limit", None)
    works: list[Work] = []
    url: str | None = GUTENDEX
    params: dict[str, Any] | None = {k: ",".join(map(str, v)) if isinstance(v, list) else v for k, v in query.items()}
    while url and (limit is None or len(works) < limit):
        page = fetch.get(url, params)
        if page is None:
            raise RuntimeError(f"Gutendex returned 404 for {url}")
        data = page.json()
        url, params = data.get("next"), None  # `next` carries the query
        for book in data["results"]:
            if limit is not None and len(works) >= limit:
                break
            if book.get("media_type") != "Text":
                continue
            work = _import_book(corpus, fetch, book)
            if work is not None:
                works.append(work)
    return works


def _import_book(corpus: Corpus, fetch: _Polite, book: dict[str, Any]) -> Work | None:
    old = corpus.indexed().get(f"gutenberg:{book['id']}")
    if old is not None and corpus.has_text(old):
        work = gutenberg_work(book, url=old.url, content_hash=old.content_hash, snapshot_date=old.snapshot_date)
        return corpus.add(work)
    for url in mirror_urls(book["id"]):
        try:
            resp = fetch.get(url)
        except httpx.HTTPError as e:  # retries spent: skip this book, not the batch
            log.warning("gutenberg:%s: %s: %s after %d attempts; skipped", book["id"], url, type(e).__name__, RETRIES)
            return None
        if resp is None:
            continue
        try:
            gutenberg_body(decode(resp.content))
        except ValueError as e:
            log.warning("gutenberg:%s: %s: %s; skipped", book["id"], url, e)
            return None
        today = dt.datetime.now(dt.UTC).date().isoformat()
        work = gutenberg_work(book, url=url, content_hash=corpus.store_text(resp.content), snapshot_date=today)
        return corpus.add(work)
    log.warning("gutenberg:%s: no plain-text file on the mirror; skipped", book["id"])
    return None


def gutenberg_catalogue(*, http: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Every catalogue record from the mirror's `pg_catalog.csv` (metadata only):
    `{id, type, language, title, authors: [name], translated: bool, subjects, bookshelves}`.
    Names drop the dates and roles the catalogue appends ("Wells, H. G. (Herbert George),
    1866-1946" -> "Wells, H. G. (Herbert George)", as Gutendex writes it); `authors` lists
    only authors, and `translated` says whether a translator is credited."""
    own = http is None
    http = http or httpx.Client(timeout=900, headers={"User-Agent": "writing-assistant-bench/0"})
    try:
        resp = http.get(CATALOGUE)
        resp.raise_for_status()
    finally:
        if own:
            http.close()
    out = []
    for row in csv.DictReader(io.StringIO(resp.text)):
        people = [p.strip() for p in row["Authors"].split(";") if p.strip()]
        roles = [re.search(r"\[([^\]]+)\]\s*$", p) for p in people]
        names = [re.sub(r"(,\s*(\d{3,4}\??|-|BCE|BC| )+(-\d{0,4}\??)?)?\s*(\[[^\]]+\])?\s*$", "", p) for p in people]
        out.append({
            "id": int(row["Text#"]), "type": row["Type"], "language": row["Language"], "title": row["Title"],
            "authors": [n for n, r in zip(names, roles, strict=True) if r is None],
            "translated": any(r and "translator" in r.group(1).casefold() for r in roles),
            "subjects": row["Subjects"], "bookshelves": row["Bookshelves"],
        })
    return out


COLLECTORS: dict[str, Callable[[Corpus, _Polite, dict[str, Any]], list[Work]]] = {"gutenberg": collect_gutenberg}


def collect(
    corpus: Corpus,
    sources: dict[str, dict[str, Any]],
    since: dt.date | None = None,
    gateway: SearchGateway | None = None,
    *,
    http: httpx.Client | None = None,
    delay: float = DELAY,
) -> list[Work]:
    """`wa-bench collect`: run each named source's collector with its query and return the
    works it wrote or refreshed. Writes rows, cache entries and familiarity parts; never
    deletes. `since` and `gateway` are for the live scrapers of Slice 3; the Gutenberg
    importer has no date filter and fetches over its own client."""
    del since, gateway
    unknown = set(sources) - set(COLLECTORS)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}; known: {sorted(COLLECTORS)}")
    own = http is None
    http = http or httpx.Client(timeout=120, follow_redirects=False, headers={"User-Agent": "writing-assistant-bench/0"})
    try:
        fetch = _Polite(http, delay)
        written = [w.work_id for name, query in sources.items() for w in COLLECTORS[name](corpus, fetch, query)]
    finally:
        if own:
            http.close()
    # Re-read: each add moved every familiarity percentile, so earlier returns are stale.
    return [corpus.get(work_id) for work_id in written]

