"""Verbatim overlap of an output with what the episode showed and with its works.

Three sets, defined by who wrote the text (design/benchmark.md, "The Corpus"):

- `context_copy`: text the benchmark put in front of a model actor. Every delivered message
  (the Turn binding delivers prompts and scripted actors' lines), the system prompts, files as
  a non-model writer left them (setup, scripted actors), events logged by non-model actors,
  and results of suite tools other than fetches. Only what some model actor could perceive
  counts: a file its ACL hides or an event no model seat sees was never in front of it.
- `tool_copy`: text fetch tools (search, read page, read paper) brought into the episode.
- `memorization`: the cached text of the works in `ep.state["work_ids"]`, minus both.

Nothing a model actor wrote is in any set: its file writes, its tool arguments, the results
of workspace tools (which only return workspace files, already accounted for by who wrote
them). So a recited target saved with `episode_write` and read back does not subtract itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from writing_assistant.corpus.text import norm_words
from writing_assistant.episode.tools import WORKSPACE_TOOLS

if TYPE_CHECKING:
    from writing_assistant.episode import Episode

N = 8
FLAG_RUN = 50  # words; provisional until Slice 1's calibration pairs give a distribution
FLAG_COVERAGE = 0.2
# Tools whose results are text fetched from outside the episode (search/gateway.py's table,
# which lands in Slice 3; take the names from there once it exists).
FETCH_TOOLS = frozenset({"episode_search_web", "episode_read_page", "episode_search_papers", "episode_read_paper"})


def _grams(words: list[str]) -> Iterator[int]:
    for i in range(len(words) - N + 1):
        yield hash(tuple(words[i : i + N]))


def shingles(text: str) -> frozenset[int]:
    """Hashed word 8-grams of normalised text (in-process only: str hashes are salted)."""
    return frozenset(_grams(norm_words(text)))


def _score(hits: list[bool]) -> tuple[int, float]:
    """(longest run, coverage) in words, from which of the output's 8-grams are shared."""
    if not hits:
        return 0, 0.0
    n_words = len(hits) + N - 1
    covered = [False] * n_words
    longest = run = 0
    for i, hit in enumerate(hits):
        if hit:
            run += 1
            longest = max(longest, run + N - 1)
            covered[i : i + N] = [True] * N
        else:
            run = 0
    return longest, sum(covered) / n_words


def overlap(output: str, ep: Episode, work_shingles: list[frozenset[int]] | None) -> dict[str, tuple[int, float]]:
    grams = list(_grams(norm_words(output)))
    context = frozenset().union(*map(shingles, benchmark_texts(ep)))
    tool = frozenset().union(*map(shingles, fetched_texts(ep)))
    out = {
        "context_copy": _score([g in context for g in grams]),
        "tool_copy": _score([g in tool for g in grams]),
    }
    if work_shingles is not None:
        out["memorization"] = _score(
            [g not in context and g not in tool and any(g in work for work in work_shingles) for g in grams]
        )
    return out


def flag(overlap: dict[str, tuple[int, float]]) -> bool:
    """The pre-registered rule: a `memorization` run >= 50 words or coverage >= 0.2."""
    if "memorization" not in overlap:
        return False
    run, coverage = overlap["memorization"]
    return run >= FLAG_RUN or coverage >= FLAG_COVERAGE


# -- who wrote what ---------------------------------------------------------------------------


def _models(ep: Episode) -> set[str]:
    return {a for a, endpoint in ep.actors.items() if endpoint != "scripted"}


def _perceived(event_visible_to: dict[str, str], models: set[str]) -> str | None:
    """The most a model actor saw of an event: "full", "redacted", or None."""
    modes = {m for seat, m in event_visible_to.items() if seat == "*" or seat in models}
    return "full" if "full" in modes else "redacted" if "redacted" in modes else None


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def benchmark_texts(ep: Episode) -> Iterable[str]:
    models = _models(ep)
    yield from (p for a, p in ep.state.get("system_prompts", {}).items() if a in models)
    first_write: dict[str, int] = {}
    for ev in ep.events:
        if ev.kind == "file_write":
            path = ev.data["path"]
            if not any(ep.readable(m, path) for m in models):
                continue
            if path not in first_write:
                first_write[path] = ev.idx
                if ev.data["before"] is not None:
                    yield ev.data["before"]  # placed in ep.files before any logged write
            if ev.actor not in models:
                yield _added(ev.data["before"], ev.data["after"])
        elif ev.kind == "deliver":
            if ev.actor in models:
                yield ev.data["message"]
        elif ev.kind == "tool":
            if ev.actor in models and ev.data["name"] not in WORKSPACE_TOOLS and ev.data["name"] not in FETCH_TOOLS:
                # A suite tool may echo its arguments back; those words are the model's own.
                echoed = set(_strings(ev.data["args"]))
                yield from (s for s in _strings(ev.data["result"]) if s not in echoed)
        elif ev.actor not in models:
            mode = _perceived(ev.visible_to, models)
            if mode == "full":
                yield from _strings(ev.data)
            elif mode == "redacted" and "redacted" in ev.data:
                yield from _strings(ev.data["redacted"])
    for path, text in ep.files.items():  # files setup() placed without logging a write
        if path not in first_write and any(ep.readable(m, path) for m in models):
            yield text


def _added(before: str | None, after: str) -> str:
    """The lines a non-model writer added. Lines it carried over from `before` were already
    counted when they were first written, or were written by a model and must not count."""
    if before is None:
        return after
    kept = set(before.split("\n"))
    return "\n".join(line for line in after.split("\n") if line not in kept)


def fetched_texts(ep: Episode) -> Iterable[str]:
    for ev in ep.events:
        if ev.kind == "tool" and ev.data["name"] in FETCH_TOOLS:
            yield from _strings(ev.data["result"])
