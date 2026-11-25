"""Judges keyed by calibrated use: calibration is per use (200 yes/no, 40/60/80 pairs).

`judges_from(models_toml, client, profile)` builds a `Judges` from `[judges.eval]` or
`[judges.train]` in models.toml; every value there is an endpoint name.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Judges:
    fact: Callable[[str, str], Awaitable[bool]]  # (question, evidence) -> yes/no  [200-label gate]
    outline: Callable[[str], Awaitable[list[dict]]]  # text -> IdeaLens {role, paraphrase} items
    craft_pair: Callable[[str, str, str], Awaitable[float]]  # (a, b, reference) -> P(a)  [80-pair gate]
    edit_pair: Callable[[str, str, str], Awaitable[float]]  # (edited, original, objection) -> P(edited answers it)  [40-pair gate]
    secret_2afc: Callable[[str, str, str], Awaitable[float]]  # (transcript, secret_a, secret_b) -> P(a)
    author_verify: Callable[[str, str, list[str]], Awaitable[float]]  # (a, b, held-out passages) -> P(a is the author)  [60-pair gate]
    embed_semantic: Callable[[list[str]], Awaitable[np.ndarray]]
    embed_style: Callable[[list[str]], Awaitable[np.ndarray]]
    forecast: Callable[[str, int], Awaitable[list[str]]]  # (prefix, n) -> n short ending forecasts
    readability: Callable[[str], dict[str, float]]  # pure spaCy + wordfreq, no model
