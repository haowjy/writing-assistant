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


class Unscored(Exception):
    """The judge could not score this item. `reason` is "refusal" (the model declined) or
    "unparseable" (no answer in the expected form, including an empty or truncated reply);
    `records` are the calls that produced it, for storing beside the unscored item."""

    def __init__(self, reason: Literal["refusal", "unparseable"], records: list[CallRecord]) -> None:
        self.reason = reason
        self.records = records
        tail = " | ".join(r.response[:160].replace("\n", " ") for r in records)
        super().__init__(f"{reason}: {tail!r}")


_REFUSAL = re.compile(
    r"\b(I\s+can(?:no|['’])?t|I\s+(?:am|['’]m)\s+(?:not\s+able|unable)|I\s+won['’]t|I\s+will\s+not"
    r"|I\s+(?:must|have\s+to)\s+decline|I['’]m\s+sorry|I\s+apologi[sz]e|not\s+(?:comfortable|appropriate)"
    r"|against\s+(?:my|the)\s+(?:guidelines|polic(?:y|ies))|content\s+polic(?:y|ies)|sexually\s+explicit)",
    re.IGNORECASE,
)
_MARKUP = " \t*_#>`.,:;!?\"'“”‘’()[]"


def answer_text(rec: CallRecord) -> str:
    """The reply with thinking removed. Text after the last `</think>` is the answer (some
    servers drop the opening tag); a `<think>` never closed means the reply was cut off
    while thinking, so there is no answer."""
    text = rec.response
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return ""
    return text.strip()


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
    raise Unscored(refusal_or_garbage(text), [rec])


def refusal_or_garbage(text: str) -> Literal["refusal", "unparseable"]:
    return "refusal" if _REFUSAL.search(text) else "unparseable"
