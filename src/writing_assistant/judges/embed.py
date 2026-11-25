"""Embeddings from an external `/v1/embeddings` endpoint (never loaded in-process), and the
reference index novelty and same-split comparisons are measured against."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from writing_assistant.llm.client import LLMClient


async def embed(client: LLMClient, endpoint: str, texts: list[str]) -> np.ndarray:
    """Unit-norm rows, one per text, so a dot product is a cosine. The server's vectors are
    untrusted: a zero or non-finite row raises instead of turning into NaN downstream."""
    v = await client.embed(texts, endpoint=endpoint)
    if not texts:
        return v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if not np.all(np.isfinite(v)) or np.any(norms == 0):
        raise ValueError(f"{endpoint}: zero or non-finite embedding")
    return v / norms


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of rows of `a` against rows of `b` ((d,) inputs act as one row)."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T


@dataclass(frozen=True)
class ReferenceIndex:
    """Unit-norm vectors of one split's references, from one embedding model. Stored as one
    `.npz` (`vectors`, `ids`, `model`); the train-time index holds `train` rows only."""

    ids: list[str]
    vectors: np.ndarray  # (n, d), unit norm
    model: str  # the embedding model the vectors came from; queries must use the same one

    def nearest_distance(self, queries: np.ndarray) -> np.ndarray:
        """1 - cosine to the nearest reference, per query row."""
        if not self.ids:
            raise ValueError("empty reference index")
        return 1.0 - cosine(queries, self.vectors).max(axis=1)

    def save(self, path: str | Path) -> None:
        np.savez(path, vectors=self.vectors, ids=np.asarray(self.ids, dtype=str), model=np.asarray(self.model))

    @classmethod
    def load(cls, path: str | Path) -> ReferenceIndex:
        with np.load(path, allow_pickle=False) as z:
            return cls(ids=z["ids"].tolist(), vectors=z["vectors"], model=str(z["model"]))
