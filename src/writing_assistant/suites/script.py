"""What scripted seats say: a table of lines in `ep.state["script"]`, shared by every suite.

Scripted seats never go through `Turn`; the suite's step asks this table what the author
(or a scripted player) says next, and that line is delivered to the model. The table and
its position are all the step knows of where it is, and they are in `state`: a fork, or an
episode reopened after a delivery, faces exactly the turns its original would have faced.
One table serves every scripted seat of an episode: entries are taken in the order the
seats come to speak.

    state["script"] = {
        "pos": 0,                       # next entry
        "turns": [entry, ...],          # keyed by turn index
        "topics": {keyword: reply},     # keyed by question topic
    }

An entry is `{"say": str}` plus, per task, what happens before the line is delivered:
`decision` (a pivot: `{superseded, adopted}`), `edit` (a collaborative edit:
`{find, text}`), `scope` (the paragraphs a request covers), `as` (the character speaking),
`dm` (a scripted DM's tool calls and narration). This module only walks the table; each
task applies its own entry keys. `remaining` says when it has run out.
"""

from __future__ import annotations

import re
from typing import Any


def script(turns: list[dict[str, Any]], topics: dict[str, str] | None = None) -> dict[str, Any]:
    return {"pos": 0, "turns": turns, "topics": topics or {}}


def remaining(state: dict) -> int:
    s = state["script"]
    return len(s["turns"]) - s["pos"]


def previous(state: dict) -> dict[str, Any] | None:
    """The entry `next_entry` returned last; None before the first."""
    s = state["script"]
    return s["turns"][s["pos"] - 1] if s["pos"] else None


def next_entry(state: dict, last_reply: str = "") -> dict[str, Any]:
    """The next entry, advancing the script. When the model's last reply asked a question
    whose sentence names a topic keyword, the topic's reply is put before the entry's line
    (first matching keyword in table order), so a question gets its scripted answer."""
    s = state["script"]
    entry = dict(s["turns"][s["pos"]])
    s["pos"] += 1
    answer = topic_reply(s["topics"], last_reply)
    if answer and "say" in entry:
        entry["say"] = f"{answer}\n\n{entry['say']}"
    return entry


def topic_reply(topics: dict[str, str], reply: str) -> str | None:
    questions = [q.lower() for q in re.findall(r"[^.?!\n]*\?", reply)]
    for keyword, answer in topics.items():
        if any(keyword.lower() in q for q in questions):
            return answer
    return None
