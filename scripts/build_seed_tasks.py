"""Reproducible seed task rows for Suites 1a and 3, written to `store.TASKS_DIR`
(`/data/jimyao/research/writing-assistant/tasks/{1a,3}.jsonl`; they hold book text, so they
never go into git).

    uv run python scripts/build_seed_tasks.py collect          # Gutenberg authors for Suite 3 (network, minutes)
    uv run python scripts/build_seed_tasks.py build [--suites 1a,3] [--judges eval]

`collect` imports enough further Gutenberg authors that `dev` and `test` each hold several.
It reads the mirror's catalogue (metadata only), keeps English fiction authors with at least
three untranslated fiction books whose `split_of` is `dev` or `test`, walks them in a fixed
order (sha256 of the name, so famous and never-downloaded authors are equally likely; the
catalogue is not ranked by popularity) and imports up to four books each through the
Gutendex `ids` query, which brings their download counts. Gutendex's `topic` and `search`
filters and its deep ranking pages time out server-side, so nothing here depends on them.
Which authors land in which split is the hash's call, never ours.
Which authors land in which split is the hash's call, never ours.

`build` is deterministic given the corpus and the judges' answers, which are cached in
`TASKS_DIR/.judge-cache.jsonl` (keyed by prompt version and input), so a rebuild reproduces
the same rows without calling a model again:
- Suite 1a: `pov_write` and `dm_play` rows from procedurally generated SRD worlds (seed ->
  world; split hashed from `srd-world-<seed>`). Each world's fixed log is one scripted-DM
  `dm_play` run through the suite's own code, with the secret confined to the one hidden
  `reveal` event; the `fact` judge checks that no later event states it, and a world that
  fails the check is dropped. Plus hand-written `pivot_unbinding`, `collab_edit` and
  `rule_bend` rows (`rule_bend` rows take their split from their work).
- Suite 3: the within-author partition (corpus/authors.py) of every author the splitter
  can serve; per target, `k_exemplar_scene` for k in {0, 1, 3, 10} x modes `instruction`
  and `continuation` (and `style_retrieved` when an `embed_style` endpoint exists), plus
  `critique_edit` rows over exemplar-pool passages.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from writing_assistant.cli import load_dotenv
from writing_assistant.corpus import Corpus
from writing_assistant.corpus.authors import K_VALUES, AuthorSplit, Passage, partition_author_corpus, retrieve
from writing_assistant.corpus.collect import NO_AUTHOR, collect, gutenberg_catalogue
from writing_assistant.episode.store import TASKS_DIR, Task, split_of
from writing_assistant.judges import Unscored, judges_from
from writing_assistant.judges.extract import FACT, OUTLINE
from writing_assistant.llm.client import LLMClient
from writing_assistant.suites.style.tasks import beats, paragraphs
from writing_assistant.suites.world import srd, tasks_1a

ROOT = Path(__file__).resolve().parents[1]
CACHE = TASKS_DIR / ".judge-cache.jsonl"


def _h(*parts: Any) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


class CachedJudges:
    """`fact` and `outline` through a JSONL cache keyed by prompt version and input, so a
    rebuild never asks a model the same question twice and reproduces the same rows."""

    def __init__(self, judges) -> None:
        self.judges = judges
        self.cache: dict[str, Any] = {}
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["value"]
        self.calls = 0

    async def _get(self, key: str, call) -> Any:
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = await call()
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with CACHE.open("a") as f:
                f.write(json.dumps({"key": key, "value": self.cache[key]}) + "\n")
        return self.cache[key]

    async def fact(self, question: str, evidence: str) -> bool:
        return await self._get(_h(FACT.version, question, evidence), lambda: self.judges.fact(question, evidence))

    async def outline(self, text: str) -> list[dict]:
        return await self._get(_h(OUTLINE.version, text), lambda: self.judges.outline(text))


# == Suite 1a: SRD worlds and their pre-played logs ============================================

SETTINGS = [
    {"hub": ("quay", "the quay"), "far": ("net_shed", "the net shed"), "side": ("chapel", "the harbour chapel"),
     "work": "mending a net", "place": "a coil of rope", "clue": ("lantern", "cracked lantern"),
     "passer": "A fishwife with an empty basket", "door": "harbour steps"},
    {"hub": ("courtyard", "the monastery courtyard"), "far": ("scriptorium", "the scriptorium"),
     "side": ("bell_tower", "the bell tower"), "work": "copying a ledger", "place": "a loose flagstone",
     "clue": ("seal", "broken wax seal"), "passer": "A novice carrying firewood", "door": "cloister arch"},
    {"hub": ("camp", "the caravan camp"), "far": ("cistern", "the old cistern"), "side": ("dune", "the high dune"),
     "work": "watering the camels", "place": "a saddle blanket", "clue": ("compass", "dented brass compass"),
     "passer": "A boy leading a lame mule", "door": "picket line"},
    {"hub": ("market", "the river market"), "far": ("boathouse", "the boathouse"), "side": ("bridge", "the stone bridge"),
     "work": "caulking a hull", "place": "a stack of crates", "clue": ("ring", "tarnished signet ring"),
     "passer": "A ferryman with a split lip", "door": "market gate"},
    {"hub": ("parade_ground", "the fort's parade ground"), "far": ("stables", "the stables"),
     "side": ("watchtower", "the watchtower"), "work": "oiling a saddle", "place": "a water trough",
     "clue": ("letter", "half-burned letter"), "passer": "A sergeant counting rations", "door": "main gate"},
    {"hub": ("hall", "the manor's great hall"), "far": ("kitchen", "the kitchen"), "side": ("gallery", "the portrait gallery"),
     "work": "polishing the silver", "place": "the hearth", "clue": ("key", "small iron key"),
     "passer": "A maid with a covered tray", "door": "hall doors"},
]
# (secret, swapped secret, hidden probes); every probe asks about the original, so the twin's
# rate is the floor a writer reaches without the secret in view.
SECRETS = [
    ("{npc} set the fire that burned the old granary.", "{npc} saw the reeve's son set the fire that burned the old granary.",
     ["Does the text state or imply that {npc} set the granary fire?", "Does the text suggest {npc} is guilty of arson?"]),
    ("{npc} is the lost heir of House Varrow.", "{npc} is a smuggler who owes money to the harbour guild.",
     ["Does the text state or imply that {npc} is an heir of a noble house?", "Does the text mention House Varrow?"]),
    ("{npc} hid the stolen chalice under the chapel floor.", "{npc} sold the stolen chalice to a passing trader.",
     ["Does the text state or imply that {npc} hid the chalice?", "Does the text mention a hiding place under a floor?"]),
    ("{npc} poisoned the well on the night of the festival.", "{npc} found the well already poisoned on the night of the festival.",
     ["Does the text state or imply that {npc} poisoned the well?", "Does the text suggest {npc} is a poisoner?"]),
    ("{npc} is a spy paid by Baron Oskel.", "{npc} is a deserter from Baron Oskel's army.",
     ["Does the text state or imply that {npc} is a spy?", "Does the text suggest {npc} takes money from Baron Oskel?"]),
    ("{npc} buried the missing courier in the orchard.", "{npc} helped the missing courier escape over the hills.",
     ["Does the text state or imply that the courier is dead?", "Does the text state or imply that {npc} buried someone?"]),
]
PARTY_NAMES = ["Mira", "Tam", "Bren", "Ysolde", "Corin", "Ash", "Lio", "Wren", "Dace", "Nell", "Hale", "Iva"]
NPC_NAMES = ["Orin", "Maud", "Fennick", "Old Sabe", "Jory", "Petronel"]
POV_REQUEST = ("Write the next chapter of the novel, about 700 words, covering this session in close third person "
               "from {pov}'s point of view. Reply with the chapter text only.")


def _character(name: str, location: str, cha: int = 1) -> dict:
    return {"name": name, "location": location, "hp": 10, "max_hp": 10, "ac": 12, "attack_bonus": 3,
            "abilities": {"str": 1, "dex": 1, "con": 1, "int": 0, "wis": 1, "cha": cha}, "conditions": [], "inventory": []}


def make_world(seed: int) -> dict[str, Any]:
    """World seed -> a world, its scripted session (player lines and DM calls), probes."""
    rng = random.Random(f"srd-world-{seed}")
    s = SETTINGS[rng.randrange(len(SETTINGS))]
    secret, swapped, hidden = SECRETS[rng.randrange(len(SECRETS))]
    a, b, c = rng.sample(PARTY_NAMES, 3)
    npc = rng.choice(NPC_NAMES)
    ids = {n: n.lower() for n in (a, b, c)} | {npc: "npc"}
    (hub, hub_name), (far, far_name), (side, side_name) = s["hub"], s["far"], s["side"]
    clue_id, clue_name = s["clue"]
    world = {
        "seed": seed, "party": [ids[a], ids[b], ids[c]],
        "locations": {hub: {"name": hub_name, "exits": [far, side], "items": [clue_id]},
                      far: {"name": far_name, "exits": [hub], "items": []},
                      side: {"name": side_name, "exits": [hub], "items": []}},
        "characters": {ids[a]: _character(a, hub, cha=3), ids[b]: _character(b, hub), ids[c]: _character(c, hub),
                       "npc": _character(npc, far)},
        "items": {clue_id: {"name": clue_name, "kind": "gear"}},
        "secrets": {"truth": {"text": secret.format(npc=npc), "owner": "npc", "known_by": ["npc"], "reveal": {"location": far}}},
        "facts": [],
    }
    A, B, C = ids[a], ids[b], ids[c]

    def resolve(**action):
        return {"name": "srd_resolve", "arguments": action}

    def event(text, location, who):
        return {"name": "srd_propose_event", "arguments": {"text": text, "location": location, "visible_to": who}}

    session = [
        (A, f"I'll go and speak with {npc} at {far_name}. Wait for me here.",
         [resolve(action="move", actor=A, to=far)],
         f"{a} crosses to {far_name}, where {npc} looks up from {s['work']}."),
        (B, f"I'll search {hub_name} while we wait.",
         [resolve(action="check", actor=B, ability="wis", dc=2),
          event(f"{b} finds a {clue_name} hidden under {s['place']}.", hub, [B, C])],
         f"Under {s['place']}, {b} finds a {clue_name}."),
        (A, f"{npc}, I know you are hiding something. Tell me the truth.",
         [resolve(action="check", actor=A, ability="cha", dc=2), {"name": "srd_reveal", "arguments": {"secret_id": "truth", "to": [A]}}],
         f"{npc} hesitates, glances at the door, then leans close and tells {a} everything, and makes {a} swear to keep it."),
        (C, f"I'll keep watch by the {s['door']} and note who passes.",
         [event(f"{s['passer']} passes {hub_name} and stares at {b} and {c} for a long moment.", hub, [B, C])],
         f"{s['passer']} goes by and stares at you both for a long moment."),
        (A, f"Thank you, {npc}. I will keep my word.",
         [resolve(action="move", actor=A, to=hub)],
         f"{a} walks back into {hub_name}."),
        (B, f"{a}, what did {npc} tell you?", [],
         f"{a} only shrugs and says that {npc} was tired and had little to say."),
        (C, f"Then let's take the {clue_name} and go.",
         [resolve(action="take", actor=C, item=clue_id)],
         f"{c} pockets the {clue_name}, and the three of you set off along the road."),
    ]
    permitted = [f"Does the text mention {b} finding a {clue_name}?",
                 f"Does the text mention someone passing by and staring at {b} and {c}?",
                 f"Does the text mention {c} taking the {clue_name}?"]
    return {"seed": seed, "world": world, "session": session, "names": {A: a, B: b, C: c, "npc": npc},
            "secret": secret.format(npc=npc), "swapped": swapped.format(npc=npc),
            "hidden_probes": [q.format(npc=npc) for q in hidden], "permitted_probes": permitted,
            "povs": [B, C], "brief": f"A small party has come to {hub_name} looking for answers."}


async def pre_played_log(w: dict) -> list[dict]:
    """One scripted-DM dm_play run through Suite 1a's own setup and play: the fixed log."""
    task = Task(f"build:w{w['seed']}", "1a", "dm_play", "train", [], {
        "brief": w["brief"], "world": w["world"], "turns": [{"as": who, "say": say} for who, say, _, _ in w["session"]],
        "dm_script": [{"calls": calls, "narration": text} for _, _, calls, text in w["session"]]})
    [ep] = tasks_1a.setup_dm(task, None)
    await tasks_1a.play_dm(ep, None)
    if ep.state["world"]["rules_rejections"]:
        bad = [e.data for e in ep.events if e.kind == "tool" and not e.data["result"]["ok"]]
        raise ValueError(f"world {w['seed']}: the scripted DM was refused: {bad}")
    # Tool events repeat the reveal's text in their result; the story log is world and say events.
    return [{"actor": e.actor, "kind": e.kind, "data": e.data, "visible_to": e.visible_to}
            for e in ep.events if e.kind in ("world", "say")]


async def world_rows(n_worlds: int, judges: CachedJudges) -> list[dict]:
    rows = []
    for seed in range(n_worlds):
        w = make_world(seed)
        split = split_of(f"srd-world-{seed}")
        log = await pre_played_log(w)
        [hidden_idx] = [i for i, e in enumerate(log) if e["data"].get("tool") == "reveal"]
        if [i for i, e in enumerate(log) if w["secret"] in json.dumps(e["data"])] != [hidden_idx]:
            raise ValueError(f"world {seed}: the secret text appears outside the hidden event")
        later = tasks_1a.render_log([SimpleNamespace(kind=e["kind"], data=e["data"]) for e in log[hidden_idx + 1 :]], "the party")
        stated = [q for q in w["hidden_probes"] if await judges.fact(q, later)]
        if stated:
            print(f"  world {seed}: dropped, the fact judge finds a later event answers {stated}")
            continue
        hidden = log[hidden_idx]["data"]
        swap_world = {"secrets": {"truth": {"text": w["swapped"]}}}
        for pov in w["povs"]:
            rows.append({"task_id": f"1a:pov_write:w{seed}:{pov}", "suite": "1a", "kind": "pov_write", "split": split,
                         "work_ids": [], "payload": {
                             "world_seed": seed, "world": w["world"], "pov": pov, "log": log, "hidden_event_idx": hidden_idx,
                             "swap": {"data": hidden | {"text": w["swapped"]}, "state": {"world": swap_world}},
                             "request": POV_REQUEST.format(pov=w["names"][pov]),
                             "hidden_probes": w["hidden_probes"], "permitted_probes": w["permitted_probes"]}})
        rows.append({"task_id": f"1a:dm_play:w{seed}", "suite": "1a", "kind": "dm_play", "split": split, "work_ids": [],
                     "payload": {"world_seed": seed, "brief": w["brief"], "world": w["world"],
                                 "turns": [{"as": who, "say": say} for who, say, _, _ in w["session"]]}})
    return rows


# -- hand-written 1a rows --------------------------------------------------------------------

PIVOTS = [
    {"id": "lighthouse", "brief": "The story: a lighthouse keeper on a storm-bound island, told in close third person.",
     "files": {"notes/premise.md": "Keeper: Ansel Brack, sixty, alone on Skerry Light since his wife died.\n",
               "chapters/03.md": "The storm came in at dusk. Ansel trimmed the wick and listened to the shutters.\n"},
     "direction_a": ["Next is chapter 4. I want Ansel's estranged brother Pell to row out through the storm; he has come to borrow money.",
                     "Pell should carry a pistol inside his coat, and Ansel should notice the weight of it when they embrace."],
     "pivot": "Change of plan: drop Pell entirely. No brother, no money, no pistol.",
     "direction_b": "Chapter 4 instead: a half-drowned girl knocks at the door during the storm and cannot remember her own name.",
     "decision": {"superseded": "Pell rows out to borrow money, armed", "adopted": "an amnesiac girl arrives in the storm"},
     "final": ["Write chapter 4 now in chapters/04.md, about 600 words."],
     "residue_probes": ["Does the text include Ansel's brother?", "Does the text mention a pistol or gun?",
                        "Does anyone in the text ask to borrow money?"]},
    {"id": "orchard", "brief": "The story: two sisters run a failing cider orchard in a valley town.",
     "files": {"notes/cast.md": "Hester (older, careful) and June (younger, reckless). Their father left the orchard in debt.\n"},
     "direction_a": ["Scene for chapter 2: June sells the family's prize press to a rival grower, Mr. Coyle, without telling Hester.",
                     "Hester should find out from Coyle himself at the market, in front of half the town."],
     "pivot": "I've changed my mind about chapter 2: June does not sell the press, and Coyle is not in this story at all.",
     "direction_b": "Chapter 2 instead: a late frost threatens the blossom, and the sisters spend the night lighting smudge pots between the rows.",
     "decision": {"superseded": "June secretly sells the press to Coyle", "adopted": "a night fighting frost in the orchard"},
     "final": ["Write chapter 2 in chapters/02.md, about 600 words."],
     "residue_probes": ["Does the text mention Mr. Coyle?", "Does the text mention selling the cider press?",
                        "Does the text include a scene at a market?"]},
    {"id": "freighter", "brief": "The story: the crew of an aging ore freighter between the outer moons.",
     "files": {"notes/ship.md": "The Tamsin Ray: six crew, captain Odile Marr, engineer Kasimir.\n"},
     "direction_a": ["For the next scene, Kasimir discovers a stowaway child hiding in the reactor crawlspace.",
                     "The child should be carrying a sealed data spike that the mining company wants back."],
     "pivot": "Scrap the stowaway. There is no child and no data spike.",
     "direction_b": "Next scene instead: the freighter's air recycler fails, and Odile must decide who goes into the cold-sleep pods first.",
     "decision": {"superseded": "a stowaway child with a data spike", "adopted": "an air recycler failure and the cold-sleep choice"},
     "final": ["Write that scene in chapters/05.md, about 600 words."],
     "residue_probes": ["Does the text include a child or stowaway?", "Does the text mention a data spike?",
                        "Does the text mention the mining company wanting something back?"]},
]

COLLAB = [
    {"id": "winter-letter", "brief": "A short story about a widower writing to his estranged son. The author edits alongside you.",
     "path": "story.md",
     "text": ("The kettle had gone cold an hour ago, and Abel had not noticed.\n\n"
              "He wrote the first line of the letter four times. Each time it seemed to say more than he meant and less than he felt, "
              "and each time he scored it through with a single neat line, the way he had once corrected invoices.\n\n"
              "Outside, the snow was coming down in a slow, deliberate way, as if it too were choosing its words.\n\n"
              "His son had not written in six years. Abel did not blame him; he had stopped blaming anyone some time ago.\n\n"
              "In the end he wrote only: Your mother's roses came up again this spring. I thought you should know."),
     "rounds": [{"request": "The second paragraph is too long-winded. Tighten it, and leave the rest alone.",
                 "scope": ["first line of the letter"]},
                {"request": "Now make the snow paragraph less on-the-nose; I don't want the weather 'choosing its words'.",
                 "scope": ["the snow was coming down"],
                 "edit": {"find": "His son had not written", "text": "His son had not written in six years. Abel had kept every one of the old letters anyway, in a biscuit tin under the stairs."}}]},
    {"id": "river-crossing", "brief": "A story about a ferry crossing during a flood. The author edits alongside you.",
     "path": "draft.md",
     "text": ("The river had risen in the night and taken the lower jetty with it.\n\n"
              "Nadia stood at the rail and counted the passengers: nine, and a goat that belonged to no one.\n\n"
              "The ferryman said the crossing was safe. He said it the way people say things they need to be true.\n\n"
              "Halfway across, the engine coughed, caught, and coughed again.\n\n"
              "Nobody spoke. The goat, alone among them, looked entirely calm."),
     "rounds": [{"request": "Paragraph four needs more tension: make the engine failure feel dangerous. Don't touch the other paragraphs.",
                 "scope": ["Halfway across"]},
                {"request": "Give the last paragraph a little more weight; end on Nadia rather than the goat.",
                 "scope": ["Nobody spoke"],
                 "edit": {"find": "Nadia stood at the rail", "text": "Nadia stood at the rail with her daughter's hand in hers and counted the passengers: nine, and a goat that belonged to no one."}}]},
]

RULE_BENDS = [
    {"id": "pride-poor-darcy", "work": "gutenberg:1342", "rules_path": "rules.md",
     "bent_rules": "In this version Mr. Darcy has no fortune at all: he is a poor relation living on Bingley's charity. Everything else follows the source.",
     "kept_rules": "This version follows the source exactly.",
     "request": "Write the next chapter (chapter 4) in chapters/04.md, about 600 words, continuing from the canon chapters under the rules in rules.md.",
     "out_path": "chapters/04.md",
     "canon_probes": ["Does the text present Mr. Darcy as rich or as having a large estate or income?"],
     "bent_probes": ["Does the text present Mr. Darcy as poor or dependent on Bingley?"]},
    {"id": "scarlet-watson-detective", "work": "gutenberg:244", "rules_path": "rules.md",
     "bent_rules": "In this version Dr. Watson is the brilliant consulting detective, and Sherlock Holmes is the slow-witted army doctor who narrates. Everything else follows the source.",
     "kept_rules": "This version follows the source exactly.",
     "request": "Write the next chapter in chapters/04.md, about 600 words, continuing from the canon chapters under the rules in rules.md.",
     "out_path": "chapters/04.md",
     "canon_probes": ["Does the text show Holmes making the clever deductions?"],
     "bent_probes": ["Does the text show Watson making the clever deductions?"]},
    {"id": "time-machine-past", "work": "gutenberg:35", "rules_path": "rules.md",
     "bent_rules": "In this version the machine travels only into the past: the Time Traveller has been to the distant past, never the future. Everything else follows the source.",
     "kept_rules": "This version follows the source exactly.",
     "request": "Write the next chapter in chapters/04.md, about 600 words: the Time Traveller continues his account of where his machine took him, under the rules in rules.md.",
     "out_path": "chapters/04.md",
     "canon_probes": ["Does the text describe the Time Traveller in the far future?"],
     "bent_probes": ["Does the text describe the Time Traveller in the distant past?"]},
]


def hand_rows(corpus: Corpus) -> list[dict]:
    rows = []
    for p in PIVOTS:
        payload = {k: v for k, v in p.items() if k != "id"} | {"scene_path": p["final"][0].split(" in ")[1].split(",")[0]}
        rows.append({"task_id": f"1a:pivot_unbinding:{p['id']}", "suite": "1a", "kind": "pivot_unbinding",
                     "split": split_of(f"story-{p['id']}"), "work_ids": [], "payload": payload})
    for c in COLLAB:
        rows.append({"task_id": f"1a:collab_edit:{c['id']}", "suite": "1a", "kind": "collab_edit",
                     "split": split_of(f"story-{c['id']}"), "work_ids": [], "payload": {k: v for k, v in c.items() if k != "id"}})
    for r in RULE_BENDS:
        if r["work"] not in corpus.works:
            print(f"  rule_bend {r['id']}: {r['work']} is not in the corpus; run `collect` first")
            continue
        rows.append({"task_id": f"1a:rule_bend:{r['id']}", "suite": "1a", "kind": "rule_bend",
                     "split": corpus.get(r["work"]).split, "work_ids": [r["work"]],
                     "payload": {k: v for k, v in r.items() if k not in ("id", "work")} | {"chapters": 3}})
    return rows


# == Suite 3: the author split ===================================================================

N_TARGETS, N_REFS = 3, 5
OBJECTIONS = [
    "Paragraph {n} tells the reader what the characters feel instead of letting it show. Revise paragraph {n} so the feeling comes through action, gesture or speech.",
    "Paragraph {n} is slower than the passage around it. Tighten paragraph {n}: cut what the reader does not need and keep what happens.",
    "Paragraph {n} leans on general, abstract wording. Revise paragraph {n} with concrete, specific detail.",
]


def _passage(p: Passage, **extra: Any) -> dict:
    return {"work_id": p.work_id, "chapter": p.chapter, "text": p.text, **extra}


def by_author(corpus: Corpus) -> dict[str, list]:
    """Public Gutenberg works grouped by author (`select`'s key=value syntax cannot carry the
    comma in "Austen, Jane"). Works with no named author are their own group and left out."""
    out: dict[str, list] = defaultdict(list)
    for w in corpus.select():
        if w.source == "gutenberg" and not w.source_group.startswith("gutenberg:"):
            out[w.source_group].append(w)
    return dict(sorted(out.items()))


async def author_splits(corpus: Corpus) -> dict[str, AuthorSplit]:
    out = {}
    for author, works in by_author(corpus).items():
        try:
            out[author] = await partition_author_corpus(corpus, works, n_targets=N_TARGETS, n_refs=N_REFS)
        except ValueError as e:
            print(f"  {author}: no partition ({e})")
        except Unscored as e:
            print(f"  {author}: the outline judge left a target unscored ({e.reason}); author skipped")
    return out


async def style_rows(corpus: Corpus, judges) -> list[dict]:
    splits = await author_splits(corpus)
    groups = by_author(corpus)
    by_split: dict[str, list[str]] = defaultdict(list)
    for author in splits:
        by_split[groups[author][0].split].append(author)
    retrieval = True
    rows = []
    for author, sp in splits.items():
        works = groups[author]
        split, slug = works[0].split, _h(author)[:8]
        familiarity = [w.familiarity for w in works if w.familiarity is not None]
        common = {"author": author, "source": "gutenberg",
                  "familiarity": round(sum(familiarity) / len(familiarity), 4) if familiarity else None}
        work_ids = sorted({p.work_id for p in sp.exemplar_pool + sp.verifier_refs} | {t.y_star.work_id for t in sp.targets})
        others = sorted(a for a in by_split[split] if a != author)
        if not others:
            print(f"  {author}: the only author in {split}; no impostor, no scene rows")
        for i, target in enumerate(sp.targets):
            if not others:
                break
            rival = splits[others[int(_h(author, i), 16) % len(others)]]
            impostor = rival.exemplar_pool[int(_h(author, i, "p"), 16) % len(rival.exemplar_pool)]
            scene = {"beat_sheet": target.beat_sheet, "y_star": _passage(target.y_star),
                     "verifier_refs": [_passage(p) for p in sp.verifier_refs],
                     "impostor": _passage(impostor, source_group=rival.source_group)}
            draws = {"instruction": sp.draw, "continuation": sp.draw}
            if retrieval:
                try:
                    ranked = await retrieve(sp.exemplar_pool, beats(target.beat_sheet), max(K_VALUES), judges.embed_style)
                    draws["style_retrieved"] = lambda k, seed, ranked=ranked: ranked[:k]
                except Unscored:
                    retrieval = False
                    print("  style_retrieved rows skipped: no embed_style endpoint in models.toml")
            for mode, draw in draws.items():
                for k in K_VALUES:
                    if mode == "style_retrieved" and k == 0:
                        continue
                    rows.append({"task_id": f"3:{slug}:t{i}:{mode}:k{k}", "suite": "3", "kind": "k_exemplar_scene",
                                 "split": split, "work_ids": work_ids, "payload": common | scene | {
                                     "mode": mode, "k": k, "target": i,
                                     "exemplars": [_passage(p) for p in draw(k, i)]}})
        ranked = sorted(sp.exemplar_pool, key=lambda p: _h(author, "edit", p.work_id, p.chapter))
        made = 0
        for p in ranked:
            pars = paragraphs(p.text)
            cands = [n for n, par in enumerate(pars, 1) if n > 1 and n < len(pars) and len(par.split()) >= 40]
            if len(pars) < 4 or not cands:
                continue
            n = cands[int(_h(author, "par", made), 16) % len(cands)]
            rows.append({"task_id": f"3:{slug}:edit{made}", "suite": "3", "kind": "critique_edit", "split": split,
                         "work_ids": [p.work_id], "payload": common | {
                             "passage": _passage(p), "objection": OBJECTIONS[made % len(OBJECTIONS)].format(n=n),
                             "scope": [n], "path": "passage.md"}})
            made += 1
            if made == 2:
                break
    return rows


# == collect ====================================================================================

EXTRA = [35, 155]  # works the hand-written rule_bend rows mount (The Time Machine, The Moonstone)


def cmd_collect(args: argparse.Namespace) -> None:
    corpus = Corpus()
    collect(corpus, {"gutenberg": {"ids": EXTRA}})
    have = Counter(works[0].split for works in by_author(corpus).values())
    print(f"authors already in the corpus by split: {dict(have)}", flush=True)
    catalogue = defaultdict(list)
    for rec in gutenberg_catalogue():
        if rec["type"] == "Text" and rec["language"] == "en" and rec["authors"] and not rec["translated"] and _fiction(rec):
            catalogue[rec["authors"][0]].append(rec["id"])
    candidates = sorted((a for a, ids in catalogue.items() if len(ids) >= 3 and a.casefold() not in NO_AUTHOR
                         and split_of(a) != "train" and a not in by_author(corpus)), key=_h)
    print(f"{len(candidates)} dev/test fiction authors with 3+ books in the catalogue", flush=True)
    picked: Counter[str] = Counter()
    for author in candidates:
        split = split_of(author)
        if picked[split] + have[split] >= args.per_split:
            if all(picked[s] + have[s] >= args.per_split for s in ("dev", "test")):
                break
            continue
        try:
            works = collect(corpus, {"gutenberg": {"ids": sorted(catalogue[author])[:4]}})
        except Exception as e:  # a flaky Gutendex call loses this author, not the run
            print(f"  {author}: import failed ({type(e).__name__}: {e})", flush=True)
            continue
        probe = Corpus(judges=SimpleNamespace(outline=_no_outline))
        try:
            asyncio.run(partition_author_corpus(probe, by_author(probe)[author], n_targets=N_TARGETS, n_refs=N_REFS))
        except (ValueError, KeyError) as e:
            print(f"  {author} ({split}): imported {len(works)} books, not usable ({e})", flush=True)
            continue
        picked[split] += 1
        downloads = [w.familiarity_parts.get("downloads") for w in works]
        print(f"  {author} ({split}): {len(works)} books, downloads {downloads}", flush=True)
    print(f"added authors by split: {dict(picked)}")


def _fiction(book: dict) -> bool:
    """Fiction by the record's subjects and bookshelves (lists from Gutendex, text from the catalogue)."""
    labels = " ".join(str(book.get(k, "")) if isinstance(book.get(k), str) else " ".join(book.get(k) or [])
                      for k in ("subjects", "bookshelves")).casefold()
    return "fiction" in labels and "poetry" not in labels and "drama" not in labels


async def _no_outline(text: str) -> list[dict]:
    return []


# == build ======================================================================================

async def cmd_build(args: argparse.Namespace) -> None:
    client = LLMClient(args.models)
    judges = CachedJudges(judges_from(client, args.judges))
    corpus = Corpus(judges=judges)
    suites = args.suites.split(",")
    try:
        if "1a" in suites:
            rows = await world_rows(args.worlds, judges) + hand_rows(corpus)
            _write("1a", rows)
        if "3" in suites:
            _write("3", await style_rows(corpus, judges.judges))
    finally:
        await client.aclose()
    print(f"judge calls made: {judges.calls} (the rest from {CACHE}); spent ${sum(client.spent.values()):.4f}")


def _write(suite: str, rows: list[dict]) -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    path = TASKS_DIR / f"{suite}.jsonl"
    if len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError(f"suite {suite}: duplicate task ids")
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)
    counts = Counter((r["split"], r["kind"]) for r in rows)
    print(f"{path}: {len(rows)} rows; " + ", ".join(f"{s}/{k} {n}" for (s, k), n in sorted(counts.items())))


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", type=Path, default=ROOT / "models.toml")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--per-split", type=int, default=7, help="authors wanted in each of dev and test")
    b = sub.add_parser("build")
    b.add_argument("--suites", default="1a,3")
    b.add_argument("--judges", default="eval")
    b.add_argument("--worlds", type=int, default=40)
    args = p.parse_args()
    if args.cmd == "collect":
        cmd_collect(args)
    else:
        asyncio.run(cmd_build(args))


if __name__ == "__main__":
    sys.exit(main())
