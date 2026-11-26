"""A made-up conversation (placeholder suite; the benchmark's new task kinds will replace it).

One kind, `converse`: seats talk in turn about a request, on whatever files the position
holds, until the turn budget, a seat or the script ends it. It is scored on what needs no
judge. A row holds no book text, so the suite declares no `text_fields`.

    payload = {
        "request": str,                 # the opening line
        "opening": seat,                # who says it (default: the first seat)
        "max_turns": int,               # lines spoken by all seats together, the request included
        "seats": {seat: {               # in table order, which is speaking order
            "cast": "scripted" | "subject" | another cast name,
            "brief": str,               # a model seat's system prompt
            "heard_by": [seat, ...],    # who hears this seat's lines (default: everyone)
            "reads": [path, ...],       # restricted files of the position this seat may read
            "tools": [name, ...],       # default: the workspace tools and conversation_end
            "sampling": {...},          # temperature, max_tokens, max_steps of a model seat
        }},
        "script": [{"say": str}, ...],  # the scripted seats' lines, in the order they come to speak
    }

Set up at a position (`wa-bench situate`), the seats work on that episode's files; a
restricted file stays closed to every seat `reads` does not name. A task row played in
place (`wa-bench run --suite madeup`) starts with no files.
"""

from __future__ import annotations

from collections.abc import Iterable

from writing_assistant.episode import Episode
from writing_assistant.episode.store import Task, task_rows
from writing_assistant.episode.tools import WORKSPACE_TOOLS
from writing_assistant.suites import SUBJECT, KindAbout, converse, script

ABOUT = {"converse": KindAbout(
    title="A conversation among seats",
    tests="Nothing yet: a placeholder that exercises seats, casts and turn-taking with no judge.",
    given="Each model seat gets its brief as a system prompt, then the lines it hears since it last spoke.",
    kept=None,
    arms={"base": "Every seat as the row describes it."},
)}
METRIC_ABOUT = {
    "turns": "Lines spoken by all seats together, the opening request included.",
    "subject_lines": "Lines the model under test spoke, over all the seats it held.",
    "subject_words": "Words in those lines.",
    "ended_by_": "1 when the conversation ended that way: a seat called conversation_end, the turn budget ran out, "
                 "the script ran out, or nobody had anything left to answer.",
}


class MadeUp:
    tools = converse.TOOLS
    judged_by: dict[str, str] = {}
    about = ABOUT
    metric_about = METRIC_ABOUT
    text_fields: dict[str, list[str]] = {}

    def tasks(self, split: str, where: str | None = None) -> Iterable[Task]:
        return task_rows("madeup", split, where)

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
        for seat, s in seats.items():
            for path in s.get("reads", []):
                ep.state["file_acl"][path].append(seat)  # only a path the position restricts needs granting
        ep.state["script"] = script.script(p.get("script", []))
        ep.state["converse"] = converse.conversation(list(seats), {"seat": p.get("opening", next(iter(seats))), "text": p["request"]},
                                                     p["max_turns"])
        return [ep]

    async def next_delivery(self, ep: Episode, last_reply: str) -> tuple[str, str] | None:
        return await converse.converse(ep, last_reply, hears=_hears)

    async def score(self, ep: Episode, services, tally, twin: Episode | None = None) -> dict[str, float]:
        conv = ep.state["converse"]
        mine = [e.data["text"] for e in ep.events if e.kind == "say" and ep.actors[e.actor] == SUBJECT]
        ended = "seat" if conv["ended"] in conv["seats"] else conv["ended"]
        return {"turns": float(conv["turns"]), "subject_lines": float(len(mine)),
                "subject_words": float(sum(len(text.split()) for text in mine))} | {
            f"ended_by_{how}": float(ended == how) for how in ("seat", "turn_budget", "script", "silence")}


def _hears(ep: Episode, speaker: str) -> list[str]:
    return ep.state["heard_by"].get(speaker, ep.state["converse"]["seats"])


suite = MadeUp()
