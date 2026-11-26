"""Suite 3 tasks: `setup` builds one episode per task row, `play` delivers its one message.

The writer is the only model actor (`suites.SUBJECT`). What it must never see stays in
`ep.state` and is never delivered: the author's true passage `y_star`, the verifier's
held-out `verifier_refs` and the `impostor` passage. Setup refuses a row whose exemplars
share a chapter or a sentence-length run of text with any of those, so a target can never
be shown as an exemplar. The author is never named to the writer.

Task row payloads (`Task.payload`), by kind (a passage is `{work_id, chapter, text}`):

k_exemplar_scene: {mode: "instruction" | "continuation" | "style_retrieved", k: int,
    author: source_group, source, familiarity?, exemplars: [passage] (k of them),
    beat_sheet: [{role, paraphrase}], y_star: passage, verifier_refs: [passage],
    impostor: passage + {source_group}}
    The scene is the writer's reply (no tools). `instruction` asks for the author's voice
    with the exemplars as samples; `continuation` frames them as excerpts of a manuscript
    to continue; `style_retrieved` is `instruction` over exemplars retrieved by
    `embed_style` similarity to the beat sheet at build time.
critique_edit: {author, source, familiarity?, passage: passage, objection: str,
    scope: [paragraph number, 1-based], path}
    The passage is mounted at `path`; the writer edits it in place with the workspace tools,
    and the scorer reads the file.
"""

from __future__ import annotations

import asyncio
from typing import Any

from writing_assistant.corpus.text import norm_words
from writing_assistant.episode import Episode, norm_path
from writing_assistant.suites import SUBJECT

WRITER = "writer"
EDIT_TOOLS = ["episode_read", "episode_edit", "episode_write"]
MODES = ("instruction", "continuation", "style_retrieved")
_RUN = 12  # words: a shared run this long between an exemplar and a hidden passage is a leak


def _episode(task, state: dict[str, Any], system: str) -> Episode:
    ep = Episode(ep_id=task.task_id, task_id=task.task_id, suite="3", split=task.split,
                 state={"kind": task.kind, "work_ids": list(task.work_ids), **state})
    ep.actors[WRITER] = SUBJECT
    ep.actor_tools[WRITER] = []
    ep.state["system_prompts"] = {WRITER: system}
    return ep


# -- k_exemplar_scene ------------------------------------------------------------------------

def setup_scene(task) -> list[Episode]:
    p = task.payload
    if p["mode"] not in MODES:
        raise ValueError(f"{task.task_id}: unknown mode {p['mode']!r}; modes: {MODES}")
    if len(p["exemplars"]) != p["k"]:
        raise ValueError(f"{task.task_id}: k={p['k']} but {len(p['exemplars'])} exemplars")
    hidden = [p["y_star"], *p["verifier_refs"], p["impostor"]]
    _refuse_leaks(task.task_id, p["exemplars"], hidden)
    state = {key: p[key] for key in ("mode", "k", "exemplars", "beat_sheet", "y_star", "verifier_refs", "impostor")}
    return [_episode(task, {"scene": state}, "You are a fiction writer.")]


def _refuse_leaks(task_id: str, shown: list[dict], hidden: list[dict]) -> None:
    hidden_chapters = {(h["work_id"], h["chapter"]) for h in hidden}
    hidden_runs = set().union(*(_runs(h["text"]) for h in hidden))
    for x in shown:
        if (x["work_id"], x["chapter"]) in hidden_chapters or _runs(x["text"]) & hidden_runs:
            raise ValueError(f"{task_id}: exemplar {x['work_id']} ch. {x['chapter']} overlaps a target, "
                             "verifier or impostor passage")


def _runs(text: str) -> set[tuple[str, ...]]:
    words = norm_words(text)
    return {tuple(words[i : i + _RUN]) for i in range(len(words) - _RUN + 1)}


def beats(beat_sheet: list[dict]) -> str:
    return "\n".join(f"{i}. ({b['role']}) {b['paraphrase']}" for i, b in enumerate(beat_sheet, 1))


def target_words(ep: Episode) -> int:
    """The length asked for: the true passage's, rounded, so candidates and y* compare."""
    return max(100, round(len(ep.state["scene"]["y_star"]["text"].split()), -1))


def scene_message(ep: Episode) -> str:
    s = ep.state["scene"]
    words, sheet = target_words(ep), beats(s["beat_sheet"])
    texts = [x["text"] for x in s["exemplars"]]
    if s["mode"] == "continuation":
        excerpts = "\n\n* * *\n\n".join(texts)
        head = (f"Below are excerpts from a manuscript.\n\n<<<\n{excerpts}\n>>>\n\n" if texts
                else "")
        return (f"{head}Notes for the next scene of the manuscript (the notes are not part of the text):\n"
                f"{sheet}\n\nWrite that scene, about {words} words"
                + (", continuing the manuscript in the same voice" if texts else "")
                + ". Reply with the scene's text only: no title, notes or commentary.")
    if not texts:
        return (f"Write a scene of about {words} words that covers these beats, in order:\n{sheet}\n\n"
                "Reply with the scene's text only: no title, notes or commentary.")
    samples = "\n\n".join(f"Passage {i}:\n<<<\n{t}\n>>>" for i, t in enumerate(texts, 1))
    return (f"Below {'is one passage' if len(texts) == 1 else f'are {len(texts)} passages'} by one author. "
            "Study the voice: sentence length and rhythm, diction, punctuation, how dialogue is handled, "
            f"what gets described and how.\n\n{samples}\n\n"
            f"Write a new scene of about {words} words in this author's voice. It covers these beats, in order:\n"
            f"{sheet}\n\nReply with the scene's text only: no title, notes or commentary.")


# -- critique_edit ---------------------------------------------------------------------------

def setup_edit(task) -> list[Episode]:
    p = task.payload
    path = norm_path(p["path"])
    n = len(paragraphs(p["passage"]["text"]))
    if not p["scope"] or not all(1 <= i <= n for i in p["scope"]):
        raise ValueError(f"{task.task_id}: scope {p['scope']} outside the passage's {n} paragraphs")
    ep = _episode(task, {"edit": {"path": path, "original": p["passage"]["text"], "objection": p["objection"],
                                  "scope": p["scope"]}},
                  "You are a fiction editor working on files in a project folder.")
    ep.files[path], ep.file_revs[path] = p["passage"]["text"], 1
    ep.actor_tools[WRITER] = list(EDIT_TOOLS)
    return [ep]


def edit_message(ep: Episode) -> str:
    s = ep.state["edit"]
    return (f"An editor left this note on {s['path']}:\n\n{s['objection']}\n\n"
            f"Revise {s['path']} in place with the tools to answer the note. Paragraphs are separated by "
            "blank lines. Change only what the note asks for and leave every other paragraph exactly as "
            "it is. When the file is revised, reply with one line saying what you changed.")


def paragraphs(text: str) -> list[str]:
    """Blank-line separated paragraphs, stripped, empty ones dropped."""
    return [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]


# -- dispatch --------------------------------------------------------------------------------

SETUP = {"k_exemplar_scene": setup_scene, "critique_edit": setup_edit}
MESSAGE = {"k_exemplar_scene": scene_message, "critique_edit": edit_message}


def setup(task, services) -> list[Episode]:
    if task.kind not in SETUP:
        raise ValueError(f"{task.task_id}: suite 3 has no task kind {task.kind!r}; kinds: {sorted(SETUP)}")
    return SETUP[task.kind](task)


async def play(eps: list[Episode], services, turn) -> list[Episode]:
    await asyncio.gather(*(turn(ep, WRITER, MESSAGE[ep.state["kind"]](ep)) for ep in eps))
    return eps
