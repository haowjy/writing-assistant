"""A made-up conversation (placeholder suite; the benchmark's new task kinds will replace it).

One kind, `converse`: seats talk in turn about a request, on whatever files the position
holds, until the turn budget, a seat or the script ends it. A row holds no book text, so
the suite declares no `text_fields`.

Rubrics. `counts` scores what needs no judge and is the kind's default. The others are
scored by a judge that reads the stored rollout (`READS`; the judge use of the same name,
`judges/findings.py`) and run only on a row that lists them (`payload["rubrics"]`). Each
is asked once per seat the subject held, about the whole task or about each of that seat's
turns, on that seat's view of the rollout (`episode.rollout_evidence`); its row's metrics
are `<dimension>.<criterion>` on 0 to 1, per seat (`.<seat>`) when the subject held several.

    payload = {
        "request": str,                 # the opening line
        "opening": seat,                # who says it (default: the first seat)
        "to": [seat, ...],              # the seats it is addressed to (default: everyone who hears the speaker)
        "max_turns": int,               # lines spoken by all seats together, the request included
        "seats": {seat: {               # in table order, which is speaking order
            "cast": "scripted" | "subject" | another cast name,
            "brief": str,               # a model seat's system prompt
            "heard_by": [seat, ...],    # who hears this seat's lines (default: everyone)
            "reads": [path, ...],       # restricted files of the position this seat may read
            "tools": [name, ...],       # default: the workspace tools and conversation_end
            "sampling": {...},          # temperature, max_tokens, max_steps of a model seat
        }},
        "script": [{"say": str, "to": [seat, ...]?}, ...],  # the scripted seats' lines, in the order they come to speak
    }

Set up at a position (`wa-bench situate`), the seats work on that episode's files; a
restricted file stays closed to every seat `reads` does not name. A task row played in
place (`wa-bench run --suite situations`) starts with no files.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass

from writing_assistant.episode import Episode, rollout_evidence
from writing_assistant.episode.store import Task, task_rows
from writing_assistant.episode.tools import WORKSPACE_TOOLS
from writing_assistant.judges.findings import render
from writing_assistant.suites import SUBJECT, KindAbout, Services, Tally, converse, script

ABOUT = {"converse": KindAbout(
    title="A conversation among seats",
    tests="Nothing yet: a placeholder that exercises seats, casts and turn-taking with no judge.",
    given="Each model seat gets its brief as a system prompt, then the lines it hears since it last spoke.",
    kept=None,
    arms={"base": "Every seat as the row describes it."},
)}
METRIC_ABOUT = {
    "turns": "Turns taken by all seats together, the opening request included.",
    "subject_lines": "Lines the model under test spoke, to everyone or to named seats, over all the seats it held.",
    "subject_words": "Words in those lines.",
    "ended_by_": "1 when the conversation ended that way: a seat called conversation_end, the turn budget ran out, "
                 "the script ran out, or nobody had anything left to answer.",
}


@dataclass(frozen=True)
class Reads:
    """A rubric a judge scores by reading the rollout: what each call covers ("task", or
    "turn": one call per turn of the seat) and the (dimension, criterion) pairs it answers."""

    covers: str
    criteria: tuple[tuple[str, str], ...]


READS = {
    "world_coherence": Reads("task", (("output", "continuity"), ("output", "visibility"))),
    "character_voice_consistency": Reads("task", (("output", "voice_consistency"),)),
    "thinking_about_prose": Reads("turn", (("thinking", "writing_decision"),)),
}
METRIC_ABOUT |= {
    "output.continuity": "World coherence: the work carries the established facts and the requested changes through (0 to 1).",
    "output.visibility": "World coherence: the seat acted only on what it could know (0 to 1).",
    "output.voice_consistency": "The seat's character stays recognisable across its turns (0 to 1).",
    "thinking.writing_decision": "Per turn: the recorded thinking makes a writing decision the prose carries through (0 to 1).",
}


class Situations:
    tools = converse.TOOLS
    judged_by = {f"{dimension}.{criterion}": rubric for rubric, reads in READS.items() for dimension, criterion in reads.criteria}
    about = ABOUT
    metric_about = METRIC_ABOUT
    text_fields: dict[str, list[str]] = {}
    rubrics: dict[str, tuple[str, ...]] = {"counts": ("converse",)} | dict.fromkeys(READS, ())

    def tasks(self, split: str, where: str | None = None) -> Iterable[Task]:
        return task_rows("situations", split, where)

    def setup(self, task: Task, services, at: Episode | None = None) -> list[Episode]:
        p = task.payload
        ep = at.begin(task.task_id, task.task_id, task.suite) if at else Episode(task.task_id, task.task_id, task.suite, task.split)
        ep.state["work_ids"] = list(task.work_ids)
        seats = p["seats"]
        ep.actors = {seat: s["cast"] for seat, s in seats.items()}
        models = {seat: s for seat, s in seats.items() if s["cast"] != "scripted"}
        ep.actor_tools = {seat: s.get("tools", [*WORKSPACE_TOOLS, *converse.TOOLS]) for seat, s in models.items()}
        ep.state["system_prompts"] = {seat: s["brief"] for seat, s in models.items()}
        ep.state["sampling"] = {seat: s["sampling"] for seat, s in models.items() if "sampling" in s}
        ep.state["heard_by"] = {seat: s["heard_by"] for seat, s in seats.items() if "heard_by" in s}
        restricted = ep.state.get("file_acl", {})
        for seat, s in seats.items():
            for path in restricted.keys() & set(s.get("reads", [])):  # any other path is open to every seat already
                restricted[path].append(seat)
        ep.state["script"] = script.script(p.get("script", []))
        opening = {"seat": p.get("opening", next(iter(seats))), "text": p["request"]} | ({"to": p["to"]} if "to" in p else {})
        ep.state["converse"] = converse.conversation(list(seats), opening, p["max_turns"])
        return [ep]

    async def next_delivery(self, ep: Episode, last_reply: str) -> tuple[str, str] | None:
        return await converse.converse(ep, last_reply, hears=_hears)

    async def score(self, ep: Episode, services, tallies, twin: Episode | None = None) -> dict[str, dict[str, float]]:
        read = [rubric for rubric in tallies if rubric in READS]
        await asyncio.gather(*(_read(ep, services, rubric, tallies[rubric]) for rubric in read))
        return {rubric: tallies[rubric].rates() if rubric in READS else COUNTED[rubric](ep) for rubric in tallies}


async def _read(ep: Episode, services: Services, rubric: str, tally: Tally) -> None:
    """Ask the rubric's judge about each seat the subject held and record its findings, one
    per criterion and call, in an order that depends on the rollout alone. A criterion about
    thinking is not asked where none was recorded (`no_thinking`), and a call with nothing
    left to ask is not made. Whatever the judge returns, a finding is about the seat and the
    scope it was asked about; a criterion it left out is `unparseable`."""
    reads, judge = READS[rubric], services.judges.findings[rubric]
    seats = [seat for seat, cast in ep.actors.items() if cast == SUBJECT]
    said = ep.meta.get("cast", {}).get(SUBJECT, {}).get("thinking")  # what this endpoint's thinking is, when known
    calls = []
    for seat in seats:
        ev = rollout_evidence(ep, seat)
        own = [t for t in ev.turns if not t.inherited]
        for turn in [None] if reads.covers == "task" else [t.turn_idx for t in own]:
            thought = any(t.thinking for t in own if turn in (None, t.turn_idx))
            ask = [(d, c) for d, c in reads.criteria if d == "output" or thought]
            calls.append((seat, turn, ask, judge.judge(render(ev, turn, said)) if ask else None))
    answers = await asyncio.gather(*(call or asyncio.sleep(0) for *_, call in calls))
    for (seat, turn, ask, _), j in zip(calls, answers, strict=True):
        about = {"covers": "task" if turn is None else {"turn": turn}, "seat": seat}
        empty = about | {"cites": [], "evidence": []}
        if j:
            got = {} if j.unscored else {(f["dimension"], f["criterion"]): f for f in j.value if f.get("seat", seat) == seat}
            tally.found([got[d, c] | about if (d, c) in got else
                         empty | {"dimension": d, "criterion": c, "unscored": j.unscored or "unparseable"} for d, c in ask],
                        judge, j.replies, by_seat=len(seats) > 1)
        tally.found([empty | {"dimension": d, "criterion": c, "unscored": "no_thinking"} for d, c in reads.criteria if (d, c) not in ask],
                    by_seat=len(seats) > 1)


def _counts(ep: Episode) -> dict[str, float]:
    conv = ep.state["converse"]
    mine = [e.data["text"] for e in ep.events if e.kind == "say" and ep.actors[e.actor] == SUBJECT]
    ended = "seat" if conv["ended"] in conv["seats"] else conv["ended"]
    return {"turns": float(conv["turns"]), "subject_lines": float(len(mine)),
            "subject_words": float(sum(len(text.split()) for text in mine))} | {
        f"ended_by_{how}": float(ended == how) for how in ("seat", "turn_budget", "script", "silence")}


COUNTED = {"counts": _counts}


def _hears(ep: Episode, speaker: str) -> list[str]:
    return ep.state["heard_by"].get(speaker, ep.state["converse"]["seats"])


suite = Situations()
