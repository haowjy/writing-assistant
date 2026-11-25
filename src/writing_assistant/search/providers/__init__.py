"""One upstream per file, each implementing `Provider`. Only `search.gateway` calls a provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class Hit:
    id: str  # canonical id (`arxiv:2610.02185`, `doi:...`, `openalex:W...`, or the page URL)
    title: str
    url: str
    snippet: str = ""
    published: str | None = None  # ISO date when the upstream gives one
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    markdown: str
    meta: dict[str, Any] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)  # canonical ids
    cited_by: list[str] = field(default_factory=list)


class Provider(Protocol):
    name: str
    kind: Literal["web", "paper"]

    async def search(self, query: str, **facets: Any) -> list[Hit]: ...

    async def fetch(self, canonical_id: str) -> Doc | None:
        """None on 403, 429, robots disallow, paywall or CAPTCHA; never retries aggressively."""
        ...
