"""The one shared dataset of works every suite draws on.

`Corpus` is built once in cli.py and reaches suites as `Services.corpus`. Works (`rows.Work`)
and task rows (`episode.store.Task`) are two schemas; a task names its works by `work_id`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from writing_assistant.corpus.rows import Work
    from writing_assistant.episode import Episode
    from writing_assistant.judges import Judges


class Corpus:
    def __init__(self, cache_dir: Path, judges: Judges | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.judges = judges

    def select(self, where: str | None = None, include_local_dev: bool = False) -> list[Work]:
        """Works matching `where` (same syntax as `episode.store.matches`); `local_dev` only when asked."""
        raise NotImplementedError

    def text(self, work: Work) -> str:
        """The work's full text from the content-addressed cache."""
        raise NotImplementedError

    def chapters(self, work: Work) -> list[str]:
        """The one chapter splitter every caller uses."""
        raise NotImplementedError

    def overlap(self, output: str, ep: Episode) -> dict[str, tuple[int, float]]:
        """`context_copy`, `tool_copy`, `memorization` -> (longest 8-gram run, coverage).

        `memorization` is absent when the episode names no work (`ep.state["work_ids"]`).
        """
        raise NotImplementedError

    def flag(self, overlap: dict[str, tuple[int, float]]) -> bool:
        """The pre-registered rule: a `memorization` run >= 50 words or coverage >= 0.2."""
        raise NotImplementedError
