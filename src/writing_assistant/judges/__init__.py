"""Judges keyed by calibrated use: calibration is per use (200 yes/no, 40/60/80 pairs).

`judges_from(client, profile)` builds a `Judges` from `[judges.eval]` or `[judges.train]` in
the models.toml the client parsed; every value there is an endpoint name. A use the profile
does not name is unavailable: its judge raises `Unscored("unavailable")` when called.

A model judge that cannot score an item raises `Unscored` (`reason` "refusal",
"unparseable", "truncated" or "unavailable"); catch it and leave the item unscored, never
default it. Each model field is a `Judge` whose `stamp` ({use, endpoint, model, prompt}) is
what a stored judgment is traced to: `await judge(*args)` gives the value, and
`await judge.judge(*args)` a `Judgment` with the replies behind it and, for the uses a
human can label, the item in `label_ui.py`'s shape (question, evidence or a/b/context).

`Judges.findings` holds the judges that read a whole stored rollout (`judges/findings.py`):
one use per prompt file in `judges/rubrics/`, each answering with a list of findings.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from writing_assistant.episode.store import LocalOnly
from writing_assistant.judges import readability
from writing_assistant.judges.embed import ReferenceIndex, cosine, embed
from writing_assistant.judges.extract import FACT, OUTLINE, outline, yes_no
from writing_assistant.judges.findings import PROMPTS, findings, version
from writing_assistant.judges.generate import FORECAST, forecasts
from writing_assistant.judges.pairwise import AUTHOR, CRAFT, EDIT, SECRET, ab
from writing_assistant.judges.reply import Unscored

if TYPE_CHECKING:
    from writing_assistant.episode import CallRecord
    from writing_assistant.llm.client import LLMClient

__all__ = ["Judge", "Judges", "Judgment", "Measure", "ReferenceIndex", "Unscored", "cosine", "judges_from"]


@dataclass(frozen=True)
class Judges:
    fact: Judge  # (question, evidence) -> yes/no  [200-label gate]
    outline: Judge  # text -> IdeaLens {role, paraphrase} items
    craft_pair: Judge  # (a, b, reference) -> P(a)  [80-pair gate]
    edit_pair: Judge  # (edited, original, objection) -> P(edited answers it)  [40-pair gate]
    secret_2afc: Judge  # (transcript, secret_a, secret_b) -> P(a)
    author_verify: Judge  # (a, b, held-out passages) -> P(a is the author)  [60-pair gate]
    embed_semantic: Judge  # texts -> unit-norm rows
    embed_style: Judge
    forecast: Judge  # (prefix, n) -> n short ending forecasts
    readability: Measure  # pure spaCy + wordfreq, no model
    findings: dict[str, Judge]  # use -> (a rendered rollout) -> findings; one per prompt in judges/rubrics/ (judges/findings.py)


@dataclass(frozen=True)
class Judgment:
    """One judged item: the value (None when unscored), why it went unscored, the judge's
    replies, and the item as `label_ui.py` reads it ({} for uses nobody labels)."""

    value: Any
    unscored: str | None
    replies: list[str]
    item: dict[str, Any]


@dataclass(frozen=True)
class Judge:
    """One calibrated use. `fn(*args)` returns (value, the calls behind it) or raises
    `Unscored`; `item(*args)` gives the labelling fields for a stored judgment."""

    fn: Callable[..., Awaitable[tuple[Any, list[CallRecord]]]]
    stamp: dict[str, str]  # use, endpoint, model, prompt (version)
    item: Callable[..., dict[str, Any]] = lambda *args: {}

    async def __call__(self, *args: Any) -> Any:
        return (await self.fn(*args))[0]

    async def judge(self, *args: Any) -> Judgment:
        try:
            value, records = await self.fn(*args)
        except Unscored as e:
            return Judgment(None, e.reason, [r.response for r in e.records], self.item(*args))
        except LocalOnly:  # local_dev text and a hosted judge: refused before sending
            return Judgment(None, "local_only", [], self.item(*args))
        return Judgment(value, None, [r.response for r in records], self.item(*args))


@dataclass(frozen=True)
class Measure:
    """A deterministic measure (no model), stamped like a judge."""

    fn: Callable[[str], dict[str, float]]
    stamp: dict[str, str]

    def __call__(self, text: str) -> dict[str, float]:
        return self.fn(text)


PASSAGE_BREAK = "\n\n* * *\n\n"


def judges_from(client: LLMClient, profile: str) -> Judges:
    uses: dict[str, str] = client.config["judges"][profile]
    model_uses = {f.name for f in fields(Judges)} - {"readability", "findings"} | PROMPTS.keys()
    if unknown := uses.keys() - model_uses:
        raise KeyError(f"[judges.{profile}] names unknown uses {sorted(unknown)}")

    def judge(use: str, prompt: str, fn: Callable[..., Any], item: Callable[..., dict] = lambda *args: {}) -> Judge:
        ep = uses.get(use)
        if ep is None:
            return Judge(_unavailable, {"use": use, "endpoint": "", "model": "", "prompt": prompt}, item)
        return Judge(lambda *args: fn(ep, *args), {"use": use, "endpoint": ep, "model": client.endpoints[ep]["model"], "prompt": prompt}, item)

    async def vectors(ep: str, texts: list[str]) -> tuple[np.ndarray, list]:
        return await embed(client, ep, texts), []

    def pair(question: str) -> Callable[[str, str, Any], dict[str, Any]]:
        return lambda a, b, context: {"question": question, "context": context, "a": a, "b": b}

    return Judges(
        fact=judge("fact", FACT.version, lambda ep, q, ev: yes_no(client, ep, q, ev),
                   lambda q, ev: {"question": q, "evidence": ev}),
        outline=judge("outline", OUTLINE.version, lambda ep, text: outline(client, ep, text)),
        craft_pair=judge("craft_pair", CRAFT.version, lambda ep, a, b, ref: ab(client, ep, CRAFT, a, b, ref),
                         pair("Which passage is better fiction for this place in the story?")),
        edit_pair=judge("edit_pair", EDIT.version, lambda ep, ed, orig, obj: ab(client, ep, EDIT, ed, orig, obj),
                        pair("Which version better answers the objection, leaving the rest as it was?")),
        secret_2afc=judge("secret_2afc", SECRET.version, lambda ep, tr, sa, sb: ab(client, ep, SECRET, sa, sb, tr),
                          lambda tr, sa, sb: pair("Which statement is the secret the transcript points to?")(sa, sb, tr)),
        author_verify=judge(
            "author_verify", AUTHOR.version,
            lambda ep, a, b, passages: ab(client, ep, AUTHOR, a, b, PASSAGE_BREAK.join(passages)),
            pair("Which candidate did the author of the context passages write?"),
        ),
        embed_semantic=judge("embed_semantic", "", vectors),
        embed_style=judge("embed_style", "", vectors),
        forecast=judge("forecast", FORECAST.version, lambda ep, prefix, n: forecasts(client, ep, prefix, n)),
        readability=Measure(
            readability.profile,
            {"use": "readability", "endpoint": "", "model": readability.SPACY_MODEL, "prompt": readability.VERSION},
        ),
        findings={use: judge(use, version(use), lambda ep, record, use=use: findings(client, ep, use, record)) for use in PROMPTS},
    )


async def _unavailable(*args: Any) -> Any:
    raise Unscored("unavailable", [])
