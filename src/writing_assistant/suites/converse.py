"""A conversation among any number of seats: the step a task kind's `next_delivery` calls.

    state["converse"] = {
        "seats":   [seat, ...],                  # everyone who speaks, in table order
        "order":   "round_robin",                # the only order built
        "opening": {"seat": seat, "text": str, "to": [seat, ...]?},  # the first line, said by a seat
        "max_turns": int,                        # turns taken, all seats together, the opening included
        "turns": 0, "ended": None,               # the step's own place
    }

Every line is a `say` event by the seat that spoke. A public line is visible to the seats
that hear its speaker (`hears(ep, speaker)`; everyone when the kind gives none) and to the
speaker. An addressed line names its listeners in `data["to"]` and is visible to them and
its speaker only, whatever `hears` says: a model seat makes one by calling `say_to` during
its turn, a scripted line or the opening by a `to` on its entry. A kind built on this step
logs no other `say` event. One call of `converse` is one step:

1. At the first step the opening is logged as its seat's line. After that, the reply the
   step is given is the public line of the seat the latest delivery went to; an empty
   reply logs no line.
2. Play is over when a seat has ended it (`conversation_end`), `turns` has reached
   `max_turns`, a scripted seat is to speak and the script has run out, nobody left has
   anything to answer, or the episode is no longer running.
3. The next speaker is the next seat in table order after the last one.
4. A scripted seat (`ep.actors[seat] == "scripted"`) speaks inside the step, from
   `state["script"]` (`suites/script.py`), and the step goes round again. A model seat is
   delivered the lines `Episode.view(seat)` shows it since its last delivery, others' only,
   each as `<speaker>: <text>`, or `<speaker> (to <names>): <text>` for an addressed one; a
   seat that has heard nothing is passed over.

`turns` counts the opening, each scripted line and each delivery answered, whatever was or
was not said in it, so seats that only speak to each other with `say_to` still reach
`max_turns`.

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

Hears = Callable[[Episode, str], list[str]]  # (episode, speaker) -> the seats that hear its public lines


def conversation(seats: list[str], opening: dict[str, Any], max_turns: int) -> dict[str, Any]:
    """`state["converse"]` at the start: nothing said yet."""
    return {"seats": seats, "order": "round_robin", "opening": opening, "max_turns": max_turns, "turns": 0, "ended": None}


async def converse(ep: Episode, last_reply: str, *, hears: Hears | None = None) -> tuple[str, str] | None:
    conv = ep.state["converse"]
    seats = conv["seats"]
    answering = next((e.actor for e in reversed(ep.events) if e.kind == "deliver"), None)  # whose reply `last_reply` is
    if ep.status != "running":
        if answering and ep.actors[answering] != SUBJECT:
            ep.fail(f"{answering}: {ep.failure_reason}")
        return None

    def turn(seat: str, text: str, to: list[str] | None = None) -> str:
        """`seat` takes a turn, saying `text` (nothing, when it is empty) to everyone who hears it or to `to`."""
        if to is not None:
            if refused := _refused(seats, seat, to):
                raise ValueError(f"{ep.ep_id}: {seat}'s line: {refused}")
            _say(ep, seat, text, to)
        elif text:
            heard = hears(ep, seat) if hears else seats
            ep.log(seat, "say", {"text": text}, {s: "full" for s in seats if s == seat or s in heard})
        conv["turns"] += 1
        return seat

    last = turn(**conv["opening"]) if conv["turns"] == 0 else turn(answering, last_reply)
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
                    entry = script.next_entry(ep.state, last_reply if last == answering else "")
                    last = turn(seat, entry["say"], entry.get("to"))
                break
            if message := _heard(ep, seat):
                return seat, message
        else:
            conv["ended"] = "silence"
    return None


def _refused(seats: list[str], speaker: str, to: list[str]) -> str | None:
    """Why `speaker` cannot address a line to `to`; None when it can."""
    if not to:
        return "it is addressed to nobody"
    if unknown := [s for s in to if s not in seats]:
        return f"no seat named {', '.join(unknown)}; the seats are {', '.join(seats)}"
    if speaker in to:
        return "a seat cannot address itself"
    return None


def _say(ep: Episode, speaker: str, text: str, to: list[str]) -> None:
    ep.log(speaker, "say", {"text": text, "to": to}, {s: "full" for s in (speaker, *to)})


def _heard(ep: Episode, seat: str) -> str:
    """What `seat` is delivered: the lines of others it was shown since its last delivery."""
    events = ep.view(seat).events
    since = max((e.idx for e in events if e.kind == "deliver" and e.actor == seat), default=-1)
    return "\n\n".join(f"{e.actor}{' (to ' + ', '.join(e.data['to']) + ')' if 'to' in e.data else ''}: {e.data['text']}"
                       for e in events[since + 1:] if e.kind == "say" and e.actor != seat)


def _end(ep: Episode, actor: str, args: dict[str, Any]) -> dict[str, Any]:
    conv = ep.state["converse"]
    if conv["ended"] is None:
        conv["ended"] = actor
    return {"ok": True}


def _say_to(ep: Episode, actor: str, args: dict[str, Any]) -> dict[str, Any]:
    to = list(dict.fromkeys(args["seats"]))
    if refused := _refused(ep.state["converse"]["seats"], actor, to):
        return {"ok": False, "error": refused}
    _say(ep, actor, args["text"], to)
    return {"ok": True}


TOOLS: ToolTable = {
    # A seat that holds this tool ends the conversation by calling it; the reason is on the tool event.
    "conversation_end": (
        schema("conversation_end",
               "End the conversation for everyone. Your reply after calling it is the last line spoken; "
               "nobody answers it.",
               {"reason": {"type": "string", "minLength": 1, "description": "Why the conversation is over, in a sentence."}},
               ["reason"]),
        _end,
    ),
    "say_to": (
        schema("say_to",
               "Say a line that only the seats you name hear; they are told it was addressed to them. Your reply "
               "at the end of your turn is still what you say to everyone who hears you: leave it empty to say nothing more. "
               "Name at least one seat at the table, and not yourself.",
               {"seats": {"type": "array", "items": {"type": "string"}, "description": "The seats that hear the line."},
                "text": {"type": "string", "minLength": 1, "description": "The line."}},
               ["seats", "text"]),
        _say_to,
    ),
}
