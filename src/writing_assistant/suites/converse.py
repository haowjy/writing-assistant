"""A conversation among any number of seats: the step a task kind's `next_delivery` calls.

    state["converse"] = {
        "seats":   [seat, ...],                  # everyone who speaks, in table order
        "order":   "round_robin",                # the only order built
        "opening": {"seat": seat, "text": str},  # the first line, said by a seat
        "max_turns": int,                        # lines spoken, all seats together, the opening included
        "turns": 0, "ended": None,               # the step's own place
    }

Every line is a `say` event by the seat that spoke, visible to the seats that hear it
(`hears(ep, speaker)`; everyone when the kind gives none) and to the speaker. A kind built
on this step logs no other `say` event. One call of `converse` is one step:

1. At the first step the opening is logged as its seat's line. After that, the reply the
   step is given is logged as the line of the model seat it was delivered to.
2. Play is over when a seat has ended it (`conversation_end`), `turns` has reached
   `max_turns`, a scripted seat is to speak and the script has run out, nobody left has
   anything to answer, or the episode is no longer running.
3. The next speaker is the next seat in table order after the last one.
4. A scripted seat (`ep.actors[seat] == "scripted"`) speaks inside the step, from
   `state["script"]` (`suites/script.py`), and the step goes round again. A model seat is
   delivered the lines `Episode.view(seat)` shows it since its last delivery, others' only,
   each as `<speaker>: <text>`; a seat that has heard nothing is passed over.

`ended` says who or what ended it: a seat's name, `turn_budget`, `script`, or `silence`
(every seat was passed over, which a `hears` that leaves seats deaf can cause). It stays
None when the episode stopped running instead.

The step keeps its place in `state` and reads the rest from the log, so an episode saved
when a delivery returns, or forked there, goes on exactly as the original would have.

A delivery that leaves the episode truncated ends play (a model call cut off at
`max_tokens`, a seat past `max_steps`). When the seat was not the subject's, the step
fails the episode with `<seat>: <reason>`: a failed episode is never scored, so the subject
is not marked down for what another seat did.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from writing_assistant.episode import Episode
from writing_assistant.episode.tools import ToolTable, schema
from writing_assistant.suites import SUBJECT, script

Hears = Callable[[Episode, str], list[str]]  # (episode, speaker) -> the seats that hear the line


def conversation(seats: list[str], opening: dict[str, str], max_turns: int) -> dict[str, Any]:
    """`state["converse"]` at the start: nothing said yet."""
    return {"seats": seats, "order": "round_robin", "opening": opening, "max_turns": max_turns, "turns": 0, "ended": None}


async def converse(ep: Episode, last_reply: str, *, hears: Hears | None = None) -> tuple[str, str] | None:
    conv = ep.state["converse"]
    seats = conv["seats"]
    # The model seat `last_reply` answers for: the latest delivery, unless a line was said since.
    answering = next((e.actor if e.kind == "deliver" else None for e in reversed(ep.events) if e.kind in ("deliver", "say")), None)
    if ep.status != "running":
        if answering and ep.actors[answering] != SUBJECT:
            ep.fail(f"{answering}: {ep.failure_reason}")
        return None

    def say(seat: str, text: str) -> str:
        heard = hears(ep, seat) if hears else seats
        ep.log(seat, "say", {"text": text}, {s: "full" for s in seats if s == seat or s in heard})
        conv["turns"] += 1
        return seat

    last = say(**conv["opening"]) if conv["turns"] == 0 else say(answering, last_reply)
    while conv["ended"] is None:
        if conv["turns"] >= conv["max_turns"]:
            conv["ended"] = "turn_budget"
            break
        at = seats.index(last) + 1
        for seat in seats[at:] + seats[:at]:
            if ep.actors[seat] == "scripted":
                if not script.remaining(ep.state):
                    conv["ended"] = "script"
                else:
                    last = say(seat, script.next_entry(ep.state, last_reply if last == answering else "")["say"])
                break
            if message := _heard(ep, seat):
                return seat, message
        else:
            conv["ended"] = "silence"
    return None


def _heard(ep: Episode, seat: str) -> str:
    """What `seat` is delivered: the lines of others it was shown since its last delivery."""
    events = ep.view(seat).events
    since = max((e.idx for e in events if e.kind == "deliver" and e.actor == seat), default=-1)
    return "\n\n".join(f"{e.actor}: {e.data['text']}" for e in events[since + 1:] if e.kind == "say" and e.actor != seat)


def _end(ep: Episode, actor: str, args: dict[str, Any]) -> dict[str, Any]:
    conv = ep.state["converse"]
    if conv["ended"] is None:
        conv["ended"] = actor
    return {"ok": True}


# A seat that holds this tool ends the conversation by calling it; the reason is on the tool event.
TOOLS: ToolTable = {
    "conversation_end": (
        schema("conversation_end",
               "End the conversation for everyone. Your reply after calling it is the last line spoken; "
               "nobody answers it.",
               {"reason": {"type": "string", "minLength": 1, "description": "Why the conversation is over, in a sentence."}},
               ["reason"]),
        _end,
    ),
}
