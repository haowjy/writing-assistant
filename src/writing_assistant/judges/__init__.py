"""Judges keyed by calibrated use: calibration is per use (200 yes/no, 40/60/80 pairs).

`judges_from(client, profile)` builds a `Judges` from `[judges.eval]` or `[judges.train]` in
the models.toml the client parsed; every value there is an endpoint name. A use the profile
does not name is unavailable: its judge raises `Unscored("unavailable")` when called.

A model judge that cannot score an item raises `Unscored` (`reason` "refusal",
"unparseable", "truncated" or "unavailable"); catch it and leave the item unscored, never
default it. Each field is a `Judge` whose `stamp` ({use, endpoint, model, prompt}) is what a
stored judgment is traced to.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from writing_assistant.judges import readability
from writing_assistant.judges.embed import ReferenceIndex, cosine, embed
from writing_assistant.judges.extract import FACT, OUTLINE, outline, yes_no
from writing_assistant.judges.generate import FORECAST, forecasts
from writing_assistant.judges.pairwise import AUTHOR, CRAFT, EDIT, SECRET, ab
from writing_assistant.judges.reply import Unscored

if TYPE_CHECKING:
    import numpy as np

    from writing_assistant.llm.client import LLMClient

__all__ = ["Judge", "Judges", "ReferenceIndex", "Unscored", "cosine", "judges_from"]


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


@dataclass(frozen=True)
class Judge:
    """One calibrated use: calls `fn`, and names what produced its judgments."""

    fn: Callable[..., Any]
    stamp: dict[str, str]  # use, endpoint, model, prompt (version)

    def __call__(self, *args: Any) -> Any:
        return self.fn(*args)


PASSAGE_BREAK = "\n\n* * *\n\n"


def judges_from(client: LLMClient, profile: str) -> Judges:
    uses: dict[str, str] = client.config["judges"][profile]
    model_uses = {f.name for f in fields(Judges)} - {"readability"}
    if unknown := uses.keys() - model_uses:
        raise KeyError(f"[judges.{profile}] names unknown uses {sorted(unknown)}")

    def judge(use: str, prompt: str, fn: Callable[..., Any]) -> Judge:
        ep = uses.get(use)
        if ep is None:
            return Judge(_unavailable, {"use": use, "endpoint": "", "model": "", "prompt": prompt})
        return Judge(lambda *args: fn(ep, *args), {"use": use, "endpoint": ep, "model": client.endpoints[ep]["model"], "prompt": prompt})

    return Judges(
        fact=judge("fact", FACT.version, lambda ep, q, ev: yes_no(client, ep, q, ev)),
        outline=judge("outline", OUTLINE.version, lambda ep, text: outline(client, ep, text)),
        craft_pair=judge("craft_pair", CRAFT.version, lambda ep, a, b, ref: ab(client, ep, CRAFT, a, b, ref)),
        edit_pair=judge("edit_pair", EDIT.version, lambda ep, ed, orig, obj: ab(client, ep, EDIT, ed, orig, obj)),
        secret_2afc=judge("secret_2afc", SECRET.version, lambda ep, tr, sa, sb: ab(client, ep, SECRET, sa, sb, tr)),
        author_verify=judge(
            "author_verify", AUTHOR.version,
            lambda ep, a, b, passages: ab(client, ep, AUTHOR, a, b, PASSAGE_BREAK.join(passages)),
        ),
        embed_semantic=judge("embed_semantic", "", lambda ep, texts: embed(client, ep, texts)),
        embed_style=judge("embed_style", "", lambda ep, texts: embed(client, ep, texts)),
        forecast=judge("forecast", FORECAST.version, lambda ep, prefix, n: forecasts(client, ep, prefix, n)),
        readability=Judge(
            readability.profile,
            {"use": "readability", "endpoint": "", "model": readability.SPACY_MODEL, "prompt": readability.VERSION},
        ),
    )


async def _unavailable(*args: Any) -> Any:
    raise Unscored("unavailable", [])
