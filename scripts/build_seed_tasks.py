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

`build` is deterministic given the corpus and the judges' answers, which are cached in
`TASKS_DIR/.judge-cache.jsonl` (keyed by judge model, prompt version and input), so a rebuild reproduces
the same rows without calling a model again:
- Suite 1a: `pov_write` and `dm_play` rows from `--per-template` SRD worlds of every secret
  template (suites/world/worlds.py; a world's split is its template's). Each world's fixed
  log is one scripted-DM `dm_play` run through the suite's own code, with the secret
  confined to one hidden event; the `fact` judge checks that no later event states it, and
  a world that fails the check is dropped. Plus hand-written `pivot_unbinding`,
  `collab_edit` and `rule_bend` rows (`rule_bend` rows take their split from their work).
- Suite 3: the within-author partition (corpus/authors.py) of every author the splitter
  can serve; per target, `k_exemplar_scene` for k in {0, 1, 3, 10} x modes `instruction`
  and `continuation` (and `style_retrieved` when an `embed_style` endpoint exists), plus
  `critique_edit` rows over exemplar-pool passages.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from writing_assistant.cli import load_dotenv
from writing_assistant.corpus import Corpus
from writing_assistant.corpus.authors import (
    K_VALUES,
    AuthorSplit,
    Passage,
    partition_author_corpus,
    retrieve,
)
from writing_assistant.corpus.collect import NO_AUTHOR, collect, gutenberg_catalogue
from writing_assistant.episode.store import TASKS_DIR, split_of
from writing_assistant.judges import Unscored, judges_from
from writing_assistant.judges.extract import FACT, OUTLINE
from writing_assistant.llm.client import LLMClient
from writing_assistant.suites import paragraphs
from writing_assistant.suites.style.tasks import beats
from writing_assistant.suites.world import worlds

ROOT = Path(__file__).resolve().parents[1]
CACHE = TASKS_DIR / ".judge-cache.jsonl"


def _h(*parts: Any) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


class CachedJudges:
    """`fact` and `outline` through a JSONL cache keyed by judge model, prompt version and
    input, so a rebuild never asks a model the same question twice and reproduces the same
    rows, and `--judges train` never reuses another model's answers."""

    def __init__(self, judges) -> None:
        self.judges = judges
        self.cache: dict[str, Any] = {}
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["value"]
        self.calls = 0

    async def _get(self, model: str, key: str, call) -> Any:
        key = _h(model, key)
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = await call()
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with CACHE.open("a") as f:
                f.write(json.dumps({"key": key, "model": model, "value": self.cache[key]}) + "\n")
        return self.cache[key]

    async def fact(self, question: str, evidence: str) -> bool:
        return await self._get(self.judges.fact.stamp["model"], _h(FACT.version, question, evidence),
                               lambda: self.judges.fact(question, evidence))

    async def outline(self, text: str) -> list[dict]:
        return await self._get(self.judges.outline.stamp["model"], _h(OUTLINE.version, text),
                               lambda: self.judges.outline(text))


# == Suite 1a: hand-written rows (the SRD worlds are in suites/world/worlds.py) =================

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

# Imported besides the authors walk: The Time Machine (35), which a hand-written rule_bend row
# mounts, and The Moonstone (155): Wilkie Collins hashes to `dev`, and his book is where the
# dev critique_edit rows come from.
EXTRA = [35, 155]


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
            rows = await worlds.world_rows(args.per_template, judges.fact) + hand_rows(corpus)
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
    b.add_argument("--per-template", type=int, default=2, help="worlds per secret template")
    args = p.parse_args()
    if args.cmd == "collect":
        cmd_collect(args)
    else:
        asyncio.run(cmd_build(args))


if __name__ == "__main__":
    sys.exit(main())
