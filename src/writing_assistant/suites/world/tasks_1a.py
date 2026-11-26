"""Suite 1a tasks: `setup` builds the episodes a task row describes, `next_delivery` steps one.

Every model-driven actor is the model under test's (the cast name `suites.SUBJECT`).
Files a task mounts are written straight into `ep.files` at rev 1 with no event, which is
how the record says the benchmark wrote them.

A twin is forked from the episode it is the baseline of (`fork(..., baseline=True)`), so it
names that episode in `baseline_of`: both POV arms have a `twin_swap`, and the name is what
tells them apart.

Task row payloads (`Task.payload`), by kind:

pivot_unbinding: {brief, files: {path: text}, direction_a: [line, 2-4], pivot: str,
    direction_b: str, decision: {superseded, adopted}, final: [line], scene_path,
    residue_probes: [question], topics?: {keyword: reply}}
    The pivot episode is sent direction_a, then `pivot + "\\n\\n" + direction_b` with a
    `decision` event, then `final`; its `clean_b` twin is sent `direction_b` alone, then
    the same `final`. A residue probe is a yes/no question asking whether the scene
    carries over an element of direction A.
dm_play: {brief, world: <srd world skeleton>, turns: [{as: character id, say: str}],
    dm_script?: [{calls: [{name, arguments}], narration: str}] (one per turn), topics?}
    With `dm_script` the DM is scripted (no model): how build_seed_tasks.py produces the
    fixed logs pov_write reads.
pov_write: {world, pov: character id, log: [{actor, kind, data, visible_to}],
    hidden_event_idx: int (into log), swap: {data, state},
    hidden_probes: [question], permitted_probes: [question]}
    The request is `POV_REQUEST`, which asks for `POV_WORDS` words.
    `log` is the world/say events of a scripted dm_play run; the secret must appear only
    in the hidden event (and in the world state `swap.state` replaces).
    Rows built from SRD worlds (worlds.py) also carry `template`, the secret template whose
    split they inherit, and `world_seed`; nothing here reads them.
rule_bend: {chapters: int (default 3), rules_path, bent_rules: str, kept_rules: str,
    request: str, out_path, canon_probes: [question], bent_probes: [question]}
    Mounts the first `chapters` chapters of each work in `task.work_ids` as
    `canon/<work_id>/NN.md`. The `bent` arm's rules file is `bent_rules`; its `twin_swap`
    twin keeps canon (`kept_rules`). A canon probe's yes means the chapter follows the
    canon rule; a bent probe's yes means it follows the changed one.
collab_edit: {brief, path, text, files?: {path: text}, rounds: [{request: str,
    scope: [phrase], edit?: {find: phrase, text: paragraph}}]}
    Before a round's request, the author replaces the paragraph containing `edit.find`
    with `edit.text`; the request covers the paragraphs containing a `scope` phrase.
"""

from __future__ import annotations

import copy
import json

from writing_assistant.episode import Episode, Event, norm_path, strings
from writing_assistant.episode.tools import dispatch_tool
from writing_assistant.suites import SUBJECT, KindAbout, one_message, paragraphs
from writing_assistant.suites import script
from writing_assistant.suites.world import srd

WRITER, AUTHOR, DM = "writer", "author", srd.DM
WORKSPACE = ["episode_read", "episode_write", "episode_edit", "episode_grep", "episode_list"]
# Payload fields that may carry work text (corpus/release.py). pov_write and dm_play logs and
# rule_bend rules are written by the benchmark; rule_bend's canon is mounted from the corpus.
TEXT_FIELDS = {"pivot_unbinding": ["files.*"], "collab_edit": ["text", "files.*", "rounds[].edit.text"],
               "pov_write": [], "dm_play": [], "rule_bend": []}
# pov_write's request; the scorer's length floor reads the same POV_WORDS.
POV_WORDS = 700
POV_REQUEST = ("Write the next chapter of the novel, about {words} words, covering this session in close third person "
               "from {pov}'s point of view. Reply with the chapter text only.")
# Sampling temperature per task kind, stamped on each episode (`ep.meta`) and read by drivers.
TEMPERATURE = {"pivot_unbinding": 0.7, "dm_play": 0.7, "pov_write": 0.7, "rule_bend": 0.7, "collab_edit": 0.7}
PUBLIC = {"*": "full"}


def _episode(task, condition: str) -> Episode:
    return Episode(ep_id=task.task_id, task_id=task.task_id, suite="1a", split=task.split, condition=condition,
                   meta={"temperature": TEMPERATURE[task.kind]}, state={"kind": task.kind, "work_ids": list(task.work_ids)})


def _mount(ep: Episode, files: dict[str, str]) -> None:
    for path, text in files.items():
        path = norm_path(path)
        ep.files[path], ep.file_revs[path] = text, 1


def _writer(ep: Episode, system: str, tools: list[str], scripted_author: bool = True) -> None:
    ep.actors[WRITER] = SUBJECT
    if scripted_author:
        ep.actors[AUTHOR] = "scripted"
    ep.actor_tools[WRITER] = list(tools)
    ep.state["system_prompts"] = {WRITER: system}


# -- pivot_unbinding -----------------------------------------------------------------------

def setup_pivot(task, services) -> list[Episode]:
    p = task.payload
    ep = _episode(task, "pivot")
    _mount(ep, p["files"])
    _writer(ep, f"You are co-writing a story with its author in a shared project folder.\n\n{p['brief']}", WORKSPACE)
    ep.state["pivot"] = {"scene_path": norm_path(p["scene_path"]), "residue_probes": p["residue_probes"]}
    clean = ep.fork("clean_b", baseline=True)
    final = [{"say": line} for line in p["final"]]
    topics = p.get("topics", {})
    ep.state["script"] = script.script(
        [{"say": line} for line in p["direction_a"]]
        + [{"say": f"{p['pivot']}\n\n{p['direction_b']}", "decision": p["decision"]}] + final, topics)
    clean.state["script"] = script.script([{"say": p["direction_b"]}] + final, topics)
    return [ep, clean]


def _pivot_entry(ep: Episode, entry: dict) -> None:
    if "decision" in entry:
        ep.log(AUTHOR, "decision", entry["decision"], PUBLIC)


# -- collab_edit ---------------------------------------------------------------------------

def setup_collab(task, services) -> list[Episode]:
    p = task.payload
    ep = _episode(task, "base")
    path = norm_path(p["path"])
    _mount(ep, p.get("files", {}) | {path: p["text"]})
    _writer(ep, f"You are co-writing a story with its author in a shared project folder; they edit the files too.\n\n{p['brief']}", WORKSPACE)
    ep.state["collab"] = {"path": path, "initial": p["text"]}
    ep.state["script"] = script.script([
        {"say": r["request"], "scope": r["scope"], **({"edit": r["edit"]} if "edit" in r else {})} for r in p["rounds"]
    ])
    return [ep]


def _collab_entry(ep: Episode, entry: dict) -> None:
    if "edit" not in entry:
        return
    path = ep.state["collab"]["path"]
    text, find = ep.files[path], entry["edit"]["find"]
    hit = next((par for par in paragraphs(text) if find in par), None)
    if hit is None:
        # The model rewrote the paragraph the author meant to edit; the edit cannot land.
        ep.log(AUTHOR, "say", {"text": f"(edit skipped: no paragraph contains {find!r})"}, {AUTHOR: "full"})
        return
    i = text.index(hit)
    ep.write_file(AUTHOR, path, text[:i] + entry["edit"]["text"] + text[i + len(hit):], PUBLIC)


# -- rule_bend -----------------------------------------------------------------------------

def setup_rule_bend(task, services) -> list[Episode]:
    p = task.payload
    ep = _episode(task, "bent")
    for work_id in task.work_ids:
        [work] = services.corpus.select(f"work_id={work_id}")
        chapters = services.corpus.chapters(work)[: p.get("chapters", 3)]
        _mount(ep, {f"canon/{work_id}/{n:02d}.md": text for n, text in enumerate(chapters, 1)})
    _writer(ep, "You are co-writing a story in a project folder. The files under canon/ are the source; "
                f"{p['rules_path']} states how this story departs from it, and it overrides the source.", WORKSPACE)
    ep.state["rule_bend"] = {"out_path": norm_path(p["out_path"]), "canon_probes": p["canon_probes"],
                             "bent_probes": p["bent_probes"]}
    ep.state["script"] = script.script([{"say": p["request"]}])
    twin = ep.fork("twin_swap", baseline=True)
    _mount(ep, {p["rules_path"]: p["bent_rules"]})
    _mount(twin, {p["rules_path"]: p["kept_rules"]})
    return [ep, twin]


# -- the scripted-author conversation (pivot, collab, rule_bend) ---------------------------

async def next_line(ep: Episode, last_reply: str, apply=None) -> tuple[str, str] | None:
    if not script.remaining(ep.state) or ep.status != "running":
        return None
    entry = script.next_entry(ep.state, last_reply)
    if apply:
        apply(ep, entry)
    return WRITER, entry["say"]


# -- dm_play -------------------------------------------------------------------------------

def setup_dm(task, services) -> list[Episode]:
    p = task.payload
    ep = _episode(task, "base")
    world = copy.deepcopy(p["world"])
    world |= {"roll_count": 0, "rules_rejections": 0, "facts": world.get("facts", [])}
    ep.state["world"] = world
    scripted = "dm_script" in p
    if scripted and len(p["dm_script"]) != len(p["turns"]):
        raise ValueError(f"{task.task_id}: dm_script has {len(p['dm_script'])} entries for {len(p['turns'])} turns")
    turns = [t | ({"dm": p["dm_script"][i]} if scripted else {}) for i, t in enumerate(p["turns"])]
    ep.state["script"] = script.script(turns, p.get("topics", {}))
    ep.actors = {DM: "scripted" if scripted else SUBJECT} | {c: "scripted" for c in world["party"]}
    ep.actor_tools[DM] = list(srd.TOOLS)
    ep.state["system_prompts"] = {DM: dm_prompt(p["brief"], world)}
    return [ep]


def dm_prompt(brief: str, world: dict) -> str:
    return (
        "You are the game master of a tabletop role-playing session. Each message is what a player "
        "character says or does. Every change to the world (dice, HP, conditions, items, movement, "
        "events, facts, revealed secrets) goes through the srd_ tools; narrate only what the tools "
        "established, and give each event to exactly the characters who could perceive it. Keep "
        "secrets from characters until their reveal condition holds. Reply with your narration to the players.\n\n"
        f"{brief}\n\nWorld (JSON):\n{json.dumps({k: world[k] for k in ('party', 'locations', 'characters', 'items', 'secrets')}, indent=1)}"
    )


async def next_dm(ep: Episode, last_reply: str) -> tuple[str, str] | None:
    """A player's line goes to the DM, and the DM's narration of it is logged by the step
    that follows the reply. A scripted DM answers inside the step, so its whole session is
    one step and no delivery."""
    world = ep.state["world"]
    while True:
        answered = script.previous(ep.state)
        if answered:
            # The narration answers the speaker: whoever is with them now hears it.
            location = world["characters"][answered["as"]]["location"]
            ep.log(DM, "say", {"text": last_reply}, srd.sees(world, location))
        if not script.remaining(ep.state) or ep.status != "running":
            return None
        entry = script.next_entry(ep.state, last_reply)
        speaker = world["characters"][entry["as"]]
        line = f"{speaker['name']}: {entry['say']}"
        ep.log(entry["as"], "say", {"text": line}, srd.sees(world, speaker["location"]))
        if ep.actors[DM] != "scripted":
            return DM, line
        for call in entry["dm"]["calls"]:
            await dispatch_tool(ep, DM, call["name"], call["arguments"], srd.TOOLS)
        last_reply = entry["dm"]["narration"]


# -- pov_write -----------------------------------------------------------------------------

def setup_pov(task, services) -> list[Episode]:
    p = task.payload
    base = _episode(task, "base")
    base.state["world"] = copy.deepcopy(p["world"])
    for ev in p["log"]:
        base.log(ev["actor"], ev["kind"], ev["data"], ev["visible_to"])
    pov, i = p["pov"], p["hidden_event_idx"]
    name = p["world"]["characters"][pov]["name"]
    _writer(base, f"You are a novelist turning a tabletop session into a book. Write in close third person "
                  f"from {name}'s point of view.", [], scripted_author=False)
    base.state["pov"] = {"pov": pov, "request": POV_REQUEST.format(words=POV_WORDS, pov=name),
                         "hidden_probes": p["hidden_probes"], "permitted_probes": p["permitted_probes"]}
    swap = {"event_idx": i, "data": p["swap"]["data"], "state": p["swap"].get("state", {}), "restricted": [pov]}
    # The strings the swap removes from the hidden event must not survive anywhere else.
    secret = set(strings(base.events[i].data)) - set(strings(swap["data"]))
    eps = []
    for arm in ("full_view", "pov_view"):
        ep = base.fork(arm)
        ep.state["pov"]["arm"] = arm
        twin = ep.fork("twin_swap", swap=swap, baseline=True)
        reachable = _reachablestrings(twin)
        left = [s for s in secret if any(s in text for text in reachable)]
        if left:
            raise ValueError(f"{task.task_id}: the swap twin still contains hidden-event text {left[0]!r}")
        eps += [ep, twin]
    return eps


def _reachablestrings(ep: Episode) -> list[str]:
    """Every string a prompt could be built from: events, files, world state, system prompts.
    Compared as strings, never as JSON text, whose escapes would hide a quote or an accent."""
    return [*strings([e.data for e in ep.events]), *ep.files.values(),
            *strings(ep.state.get("world", {})), *strings(ep.state.get("system_prompts", {}))]


def render_log(events: list[Event], pov_name: str) -> str:
    """The story log as the writer reads it: world events and speech, in order."""
    lines = []
    for ev in events:
        if ev.kind not in ("world", "say"):
            continue
        if "text" in ev.data:
            lines.append(ev.data["text"])
        elif "redacted" in ev.data:
            lines.append(ev.data["redacted"])
        else:
            lines.append(f"[something happened that {pov_name} did not witness]")
    return "\n".join(lines)


def pov_message(ep: Episode) -> str:
    s = ep.state["pov"]
    name = ep.state["world"]["characters"][s["pov"]]["name"]
    events = ep.events if s["arm"] == "full_view" else ep.view(s["pov"]).events
    return f"{s['request']}\n\nSession log:\n{render_log(events, name)}"


async def next_pov(ep: Episode, last_reply: str) -> tuple[str, str] | None:
    return one_message(ep, WRITER, pov_message)


# -- dispatch ------------------------------------------------------------------------------

SETUP = {
    "pivot_unbinding": setup_pivot,
    "dm_play": setup_dm,
    "pov_write": setup_pov,
    "rule_bend": setup_rule_bend,
    "collab_edit": setup_collab,
}
NEXT = {
    "pivot_unbinding": lambda ep, last_reply: next_line(ep, last_reply, _pivot_entry),
    "dm_play": next_dm,
    "pov_write": next_pov,
    "rule_bend": next_line,
    "collab_edit": lambda ep, last_reply: next_line(ep, last_reply, _collab_entry),
}


def setup(task, services) -> list[Episode]:
    if task.kind not in SETUP:
        raise ValueError(f"{task.task_id}: suite 1a has no task kind {task.kind!r}; kinds: {sorted(SETUP)}")
    return SETUP[task.kind](task, services)


async def next_delivery(ep: Episode, last_reply: str) -> tuple[str, str] | None:
    return await NEXT[ep.state["kind"]](ep, last_reply)
ABOUT = {
    "pov_write": KindAbout(
        title="POV writing",
        tests="Does the model keep a character from knowing what they never witnessed?",
        given=f"The log of a tabletop session already played (speech and world events), and a request for the next "
              f"chapter, about {POV_WORDS} words, in close third person from one character's point of view. One event "
              "is a secret that character never saw.",
        kept="In pov_view, every event the character did not see is left out of the log, the secret among them.",
        arms={"full_view": "The whole log, the secret included: a good writer still keeps it out of the chapter.",
              "pov_view": "Only what the character saw. It cannot leak what it was never shown, so it is the floor."},
        twin="Each arm is played again with the secret swapped for another one the character also never sees. The "
             "judge is asked the original secret's questions of that chapter: a yes there is the judge or chance, "
             "not a leak, and net_pov_leak subtracts it."),
    "dm_play": KindAbout(
        title="Running the game",
        tests="Can the model run a consistent world as the game master?",
        given="A world (party, places, characters, items, secrets) and the rules tools (srd_*); scripted players "
              "speak one line at a time and the model replies with narration. Every change to the world must go "
              "through the tools, and each event goes only to the characters who could perceive it.",
        kept=None,
        arms={"base": "One episode per task. (With no model, a scripted game master plays the same task to make "
                      "the fixed logs pov_write reads.)"}),
    "pivot_unbinding": KindAbout(
        title="Changing direction",
        tests="When the author changes their mind, does the model drop the old idea?",
        given="A shared project folder. The author gives direction A over a few messages, then scraps it for "
              "direction B, then asks for the scene in a file.",
        kept=None,
        arms={"pivot": "Direction A, the change to B, then the request."},
        twin="clean_b: the author only ever gives direction B, then the same request. Questions about direction A "
             "asked of that scene measure what shows up anyway; post_pivot_residue subtracts it."),
    "rule_bend": KindAbout(
        title="Bending canon",
        tests="Can the model write in an existing book's world with one rule deliberately changed?",
        given="The first chapters of a real book under canon/, a rules file saying how this story departs from it "
              "(and that it overrides the book), and a request for a chapter in a file.",
        kept=None,
        arms={"bent": "The rules file changes one of the book's rules."},
        twin="The rules file keeps the book's rule. How often that chapter uses the changed rule anyway is "
             "subtracted in net_canon_snapback."),
    "collab_edit": KindAbout(
        title="Editing together",
        tests="Can the model edit a shared document without trampling the author's work?",
        given="A document in a shared folder and an author who, over several rounds, edits a paragraph themselves "
              "and then asks the model to change specific paragraphs.",
        kept=None,
        arms={"base": "One episode per task; each round is scored on its own, with no judge."}),
}
