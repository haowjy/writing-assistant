"""The within-author partition behind Suite 3 (design/benchmark.md, Suite 3 item 1).

One author's works become three disjoint sets of chapters: the exemplar pool (passages put
in the prompt, `k` in {0, 1, 3, 10}), the targets (a beat sheet plus the author's true
passage y*), and the verifier's held-out references. Every passage is cut to the same
token budget, so `k` counts examples rather than prompt length.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from writing_assistant.corpus.text import count_tokens, truncate_tokens

if TYPE_CHECKING:
    from writing_assistant.corpus.rows import Corpus, Work

K_VALUES = (0, 1, 3, 10)
PASSAGE_TOKENS = 600
_MIN_PARAGRAPH = 15  # words; shorter opening paragraphs are chapter titles or epigraph credits


@dataclass(frozen=True)
class Passage:
    work_id: str
    chapter: int  # index into corpus.chapters(work)
    text: str


@dataclass(frozen=True)
class Target:
    beat_sheet: list[dict[str, Any]]  # judges.outline over y_star: [{role, paraphrase}, ...]
    y_star: Passage


@dataclass(frozen=True)
class AuthorSplit:
    source_group: str
    exemplar_pool: list[Passage]
    targets: list[Target]
    verifier_refs: list[Passage]

    def draw(self, k: int, seed: int) -> list[Passage]:
        """`k` exemplars: the first `k` of one seeded order of the pool, so for every seed the
        draws nest (k=1 within k=3 within k=10)."""
        return sorted(self.exemplar_pool, key=lambda p: _rank(seed, p.work_id, p.chapter))[:k]


def _rank(seed: Any, *key: Any) -> bytes:
    return hashlib.sha256("\0".join(map(str, (seed, *key))).encode()).digest()


def _prose(chapter: str) -> str:
    """A chapter without its heading line and any short title paragraphs after it."""
    paragraphs = [p.strip() for p in chapter.split("\n\n")[1:] if p.strip()] or [chapter]
    while len(paragraphs) > 1 and len(paragraphs[0].split()) < _MIN_PARAGRAPH:
        paragraphs.pop(0)
    return "\n\n".join(paragraphs)


async def partition_author_corpus(
    corpus: Corpus,
    works: list[Work],
    *,
    passage_tokens: int = PASSAGE_TOKENS,
    n_targets: int = 5,
    n_refs: int = 5,
    seed: int = 0,
) -> AuthorSplit:
    """Partition one author's works. With three or more works the roles are split by book
    (targets from one, verifier references from another, exemplars from the rest); with
    fewer, by chapter across the pooled chapters. Either way no chapter serves two roles, and
    only chapters with at least `passage_tokens` of prose serve at all.
    Beat sheets come from `corpus.judges.outline`, so this runs at build time. ValueError
    when the works span authors or leave fewer than max(K_VALUES) exemplar chapters."""
    groups = {w.source_group for w in works}
    if len(groups) != 1:
        raise ValueError(f"partition_author_corpus takes one author's works, got {sorted(groups)}")
    (group,) = groups
    order = sorted(works, key=lambda w: _rank(seed, w.work_id))

    def keyed(ws: list[Work]) -> list[tuple[str, int, str]]:
        # A chapter whose prose cannot fill the budget (a prologue, a short letter) serves no
        # role: a short exemplar would make `k` measure prompt length again.
        chs = [
            (w.work_id, i, ch)
            for w in ws
            for i, ch in enumerate(corpus.chapters(w))
            if count_tokens(_prose(ch)) >= passage_tokens
        ]
        return sorted(chs, key=lambda c: _rank(seed, c[0], c[1]))

    if len(order) >= 3:
        target_chs = keyed(order[:1])[:n_targets]
        ref_chs = keyed(order[1:2])[:n_refs]
        pool_chs = keyed(order[2:])
    else:
        chs = keyed(order)
        target_chs, ref_chs, pool_chs = chs[:n_targets], chs[n_targets : n_targets + n_refs], chs[n_targets + n_refs :]
    if len(pool_chs) < max(K_VALUES):
        raise ValueError(f"{group}: {len(pool_chs)} exemplar chapters, need {max(K_VALUES)}")

    def passages(chs: list[tuple[str, int, str]]) -> list[Passage]:
        return [Passage(work_id, i, truncate_tokens(_prose(ch), passage_tokens)) for work_id, i, ch in chs]

    outline = corpus.judges.outline
    targets = [Target(await outline(p.text), p) for p in passages(target_chs)]
    return AuthorSplit(group, passages(pool_chs), targets, passages(ref_chs))


async def retrieve(
    pool: list[Passage], query: str, k: int, embed_style: Callable[[list[str]], Awaitable[np.ndarray]]
) -> list[Passage]:
    """The `k` exemplars closest to `query` by style-embedding cosine (ties in pool order), so
    retrieved draws nest like seeded ones."""
    if k == 0 or not pool:
        return []
    vectors = await embed_style([query, *(p.text for p in pool)])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = vectors[1:] @ vectors[0]
    return [pool[i] for i in np.argsort(-sims, kind="stable")[:k]]
