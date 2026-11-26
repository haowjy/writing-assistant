"""What every model judge shares: versioned prompts, the `Unscored` signal, reply cleaning.

A judge that cannot score an item raises `Unscored` instead of returning a value, so a
refusal or an unparseable reply never becomes a yes, a no or a P(a). The settled `Judges`
signatures return bare `bool` / `float` / `list`, and an exception is the one signal that
fits all of them; callers catch `Unscored` and leave the item unscored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from writing_assistant.episode import CallRecord

# Judges answer deterministically; thinking models need room before the final line.
TEMPERATURE = 0.0
MAX_TOKENS = 4096


@dataclass(frozen=True)
class Prompt:
    """A judge prompt. Bump `version` on any wording change: a stored judgment is traced to
    (endpoint, model, prompt version), and calibration holds only for the version labelled."""

    version: str
    system: str
    user: str  # both are str.format templates; inserted text is not re-interpreted

    def messages(self, **fields: object) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system.format(**fields)},
            {"role": "user", "content": self.user.format(**fields)},
        ]


Reason = Literal["refusal", "unparseable", "truncated", "unavailable", "local_only"]


class Unscored(Exception):
    """The judge could not score this item. `reason` is "refusal" (the model or its host
    declined), "unparseable" (no answer in the expected form, including an empty reply),
    "truncated" (cut off at max_tokens before answering), "unavailable" (models.toml names
    no endpoint for this use) or "local_only" (the item is local_dev text and the judge's
    endpoint is hosted); `records` are the calls that produced it, for storing beside
    the unscored item."""

    def __init__(self, reason: Reason, records: list[CallRecord]) -> None:
        self.reason = reason
        self.records = records
        tail = " | ".join(r.response[:160].replace("\n", " ") for r in records)
        super().__init__(f"{reason}: {tail!r}")


REFUSAL = re.compile(
    r"\b(I\s+can(?:no|['’])?t|I\s+(?:am|['’]m)\s+(?:not\s+able|unable)|I\s+won['’]t|I\s+will\s+not"
    r"|I\s+(?:must|have\s+to)\s+decline|I['’]m\s+sorry|I\s+apologi[sz]e|not\s+(?:comfortable|appropriate)"
    r"|against\s+(?:my|the)\s+(?:guidelines|polic(?:y|ies))|content\s+polic(?:y|ies)|sexually\s+explicit)",
    re.IGNORECASE,
)
_MARKUP = " \t*_#>`.,:;!?\"'“”‘’()[]"


def answer_text(rec: CallRecord) -> str:
    """The reply. The client has already moved any thinking to `rec.reasoning`."""
    return rec.response.strip()


def pick[T](rec: CallRecord, options: dict[str, T]) -> T:
    """The option a reply chose. `options` maps an answer word (lower case) to its value.
    Reads the last `ANSWER: <word>`; failing that, a last line that is only the word (bold,
    quoted or punctuated). Never the first word: "No doubt..." and "Now, ..." are reasoning.
    Raises `Unscored` otherwise."""
    text = answer_text(rec)
    words = "|".join(sorted(map(re.escape, options), key=len, reverse=True))
    found = re.findall(rf"answer[*_\s]*[:：][*_`\"'\s]*({words})\b", text, re.IGNORECASE)
    if found:
        return options[found[-1].lower()]
    lines = [ln.strip(_MARKUP).lower() for ln in text.splitlines() if ln.strip(_MARKUP)]
    if lines and lines[-1] in options:
        return options[lines[-1]]
    raise Unscored(why_unanswered(rec), [rec])


def why_unanswered(rec: CallRecord) -> Reason:
    """Why a reply holds no usable answer: the host's stop reason first, then the text."""
    if rec.finish_reason == "refusal":
        return "refusal"
    if rec.finish_reason == "length":
        return "truncated"
    return "refusal" if REFUSAL.search(rec.response) else "unparseable"
