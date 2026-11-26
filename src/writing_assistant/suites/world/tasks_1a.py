"""Suite 1a tasks: `setup` builds the episodes a task row describes, `play` runs them.

Every model-driven actor is mapped to the endpoint placeholder `suites.SUBJECT`; cli.py
replaces it with the endpoint under test. Files a task mounts are written straight into `ep.files`
at rev 1 with no event, which is how the record says the benchmark wrote them.

Twins are children of the episode they are paired with (`parent_id` = that episode's id,
condition from `TWINS`), because both POV arms have a `twin_swap` and could not be told
apart as siblings.

Task row payloads (`Task.payload`), by kind:

pivot_unbinding: {brief, files: {path: text}, direction_a: [line, 2-4], pivot: str,
    direction_b: str, decision: {superseded, adopted}, final: [line], scene_path,
    residue_probes: [question], topics?: {keyword: reply}}
    The pivot play delivers direction_a, then `pivot + "\\n\\n" + direction_b` with a
    `decision` event, then `final`; its `clean_b` twin delivers `direction_b` alone, then
    the same `final`. A residue probe is a yes/no question asking whether the scene
    carries over an element of direction A.
dm_play: {brief, world: <srd world skeleton>, turns: [{as: character id, say: str}],
    dm_script?: [{calls: [{name, arguments}], narration: str}] (one per turn), topics?}
    With `dm_script` the DM is scripted (no model): how build_seed_tasks.py produces the
    fixed logs pov_write reads.
pov_write: {world, pov: character id, log: [{actor, kind, data, visible_to}],
    hidden_event_idx: int (into log), swap: {data, state}, request: str,
    hidden_probes: [question], permitted_probes: [question]}
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

import asyncio
import copy
import json

from writing_assistant.episode import Episode, Event, norm_path, strings
from writing_assistant.episode.tools import dispatch_tool
from writing_assistant.suites import SUBJECT, paragraphs
from writing_assistant.suites.world import author, srd

WRITER, AUTHOR, DM = "writer", "author", srd.DM
WORKSPACE = ["episode_read", "episode_write", "episode_edit", "episode_grep", "episode_list"]
TWINS = {"pivot": "clean_b", "full_view": "twin_swap", "pov_view": "twin_swap", "bent": "twin_swap"}
# Payload fields that may carry work text (corpus/release.py). pov_write and dm_play logs and
# rule_bend rules are written by the benchmark; rule_bend's canon is mounted from the corpus.
TEXT_FIELDS = {"pivot_unbinding": ["files.*"], "collab_edit": ["text", "files.*", "rounds[].edit.text"],
               "pov_write": [], "dm_play": [], "rule_bend": []}
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
    clean = ep.fork("clean_b")
    final = [{"say": line} for line in p["final"]]
    topics = p.get("topics", {})
    ep.state["script"] = author.script(
        [{"say": line} for line in p["direction_a"]]
        + [{"say": f"{p['pivot']}\n\n{p['direction_b']}", "decision": p["decision"]}] + final, topics)
    clean.state["script"] = author.script([{"say": p["direction_b"]}] + final, topics)
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
    ep.state["script"] = author.script([
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
    ep.state["script"] = author.script([{"say": p["request"]}])
    twin = ep.fork("twin_swap")
    _mount(ep, {p["rules_path"]: p["bent_rules"]})
    _mount(twin, {p["rules_path"]: p["kept_rules"]})
    return [ep, twin]


# -- the scripted-author conversation (pivot, collab, rule_bend) ---------------------------

async def converse(ep: Episode, turn, apply=None) -> None:
    reply = ""
    while author.remaining(ep.state) and ep.status == "running":
        entry = author.next_entry(ep.state, reply)
        if apply:
            apply(ep, entry)
        reply = await turn(ep, WRITER, entry["say"])


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
    ep.state["script"] = author.script(turns, p.get("topics", {}))
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


async def play_dm(ep: Episode, turn) -> None:
    world = ep.state["world"]
    reply = ""
    while author.remaining(ep.state) and ep.status == "running":
        entry = author.next_entry(ep.state, reply)
        speaker = world["characters"][entry["as"]]
        ep.log(entry["as"], "say", {"text": f"{speaker['name']}: {entry['say']}"}, srd.sees(world, speaker["location"]))
        if ep.actors[DM] == "scripted":
            for call in entry["dm"]["calls"]:
                await dispatch_tool(ep, DM, call["name"], call["arguments"], srd.TOOLS)
            reply = entry["dm"]["narration"]
        else:
            reply = await turn(ep, DM, f"{speaker['name']}: {entry['say']}")
        # The narration answers the speaker: whoever is with them now hears it.
        ep.log(DM, "say", {"text": reply}, srd.sees(world, speaker["location"]))


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
    base.state["pov"] = {"pov": pov, "request": p["request"],
                         "hidden_probes": p["hidden_probes"], "permitted_probes": p["permitted_probes"]}
    swap = {"event_idx": i, "data": p["swap"]["data"], "state": p["swap"].get("state", {}), "restricted": [pov]}
    # The strings the swap removes from the hidden event must not survive anywhere else.
    secret = set(strings(base.events[i].data)) - set(strings(swap["data"]))
    eps = []
    for arm in ("full_view", "pov_view"):
        ep = base.fork(arm)
        ep.state["pov"]["arm"] = arm
        twin = ep.fork("twin_swap", swap=swap)
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


async def play_pov(ep: Episode, turn) -> None:
    await turn(ep, WRITER, pov_message(ep))


# -- dispatch ------------------------------------------------------------------------------

SETUP = {
    "pivot_unbinding": setup_pivot,
    "dm_play": setup_dm,
    "pov_write": setup_pov,
    "rule_bend": setup_rule_bend,
    "collab_edit": setup_collab,
}
PLAY = {
    "pivot_unbinding": lambda ep, turn: converse(ep, turn, _pivot_entry),
    "dm_play": play_dm,
    "pov_write": play_pov,
    "rule_bend": converse,
    "collab_edit": lambda ep, turn: converse(ep, turn, _collab_entry),
}


def setup(task, services) -> list[Episode]:
    if task.kind not in SETUP:
        raise ValueError(f"{task.task_id}: suite 1a has no task kind {task.kind!r}; kinds: {sorted(SETUP)}")
    return SETUP[task.kind](task, services)


async def play(eps: list[Episode], services, turn) -> list[Episode]:
    """Episodes play concurrently: twins share no play and each has its own session. They
    start in list order, so a scripted Turn that never suspends still answers them in order."""
    await asyncio.gather(*(PLAY[ep.state["kind"]](ep, turn) for ep in eps))
    return eps
