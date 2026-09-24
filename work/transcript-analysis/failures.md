# Lane 1 — Failure taxonomy: what is E2B actually failing at?

Analysis of the 100 `custom50` attempts:
- `runs/custom50-e2b-it-2026-09-14` (nf4), **50 attempts, graded** (`runs/custom50-e2b-it-2026-09-14/astra-graded/<ID>/scorecard.json`, 50 scorecards).
- `runs/custom50-e2b-it-2026-09-21-bf16` (bf16), **50 attempts, ungraded** (all semantic checks `pending`).

Everything below is read from `card.json`, the astra scorecards, `result.json`/`messages`, the tool traces, and the workspace files. No script under `src/`, `tests/` or `runs/` was modified.

---

## 0. Coverage — how much of the 100 is actually gradeable

Attempt status decomposition:

| arm | completed | error | total |
|---|---|---|---|
| nf4 | 49 | 1 (F4-02) | 50 |
| bf16 | 50 | 0 | 50 |
| **all** | **99** | **1** | **100** |

Deliverable / artifact coverage:

| family (per arm) | n | artifact | nf4 | bf16 |
|---|---|---|---|---|
| F1, F2, F5 (prose) | 30 | selected `scene` extractable | **25/30** | **28/30** |
| F4 (wiki) | 10 | `kb/` pages present | 10/10 | 10/10 |
| F3 (directions) | 10 | non-empty answer | 10/10 | 10/10 |

**93/100 attempts produced a usable deliverable.** The 7 misses are all prose-family `scene` artifacts that could not be extracted (nf4: F2-01, F2-02, F2-03, F2-05, F5-03; bf16: F2-01, F2-05) — see failure mode **M6**. The single `error` attempt (F4-02) still produced `kb/index.md` + one character page before it crashed, so it is included in the 93.

nf4 scorecards = 50/50 (`F4-02` was graded even though `result.json.status == "error"`). **bf16 is not comparable on any semantic check**: 115 `card.json` checks are `pending` (continuity 30, pov-style 15, genre-fit 30, consequential-state 10, qualified-belief 10, optional-context 10, accepted-update 5, three-directions 5) and there are no bf16 astra scorecards. bf16 findings below are limited to deterministic checks (word-budget, wiki-links, retrieve-*, saved-scene, wiki-only, …).

---

## 1. Per-check pass / fail table

### nf4 (graded; astra `scorecard.json` `checks[]`)

| check | asserts | total | pass | fail | fail families / specificity |
|---|---|---|---|---|---|
| `word-budget` | selected prose 120–220 words (Q1, not required) | 15 | 1 | **9** (+5 unscored) | F1 explicit, F2 explicit, F5 explicit |
| `pov-style` | close third on X + requested style, combined (Q1) | 15 | 6 | **9** | F1/F2/F5 explicit |
| `continuity` | preserve required facts, keep belief uncertain, no forbidden assertion (Q13, required) | 30 | 20 | **10** | F1 (2 exp + 4 loose), F2 (1 exp), F5 (3 loose) |
| `wiki-links` | navigable multi-page graph from `kb/index.md` (required) | 5 | 0 | **5** | F4 explicit only |
| `optional-context` | optional KB fact (Q8, weight 1) | 10 | 5 | **5** | F4 explicit 3, loose 2 |
| `consequential-state` | required current state in KB (Q8, required) | 10 | 6 | **4** | F4 explicit 3, loose 1 |
| `accepted-update` | incorporate accepted update + keep history, drop draft-only (Q11, required) | 5 | 2 | **3** | F4 explicit 2, loose 1 |
| `retrieve-state` | required fact exposed via read/search (Q12) | 5 | 3 | **2** | F5 explicit |
| `retrieve-belief` | uncertain belief exposed via read/search (Q12) | 5 | 3 | **2** | F5 explicit |
| `saved-scene` | selected scene file non-empty (Q3, required) | 10 | 9 | **1** | F2 explicit (F2-02) |
| `three-directions` | exactly three developed viable directions (Q1, required) | 5 | 4 | **1** | F3 explicit (F3-02) |
| `wiki-only` | only `kb/*.md` changed (allowed_changes, required) | 10 | 9 | **1** | F4 loose (F4-06) |
| `qualified-belief` | required uncertain belief in KB (Q8, required) | 10 | 9 | **1** | F4 explicit (F4-05) |
| `genre-fit` | genre-defensible (Q1) | 30 | 30 | 0 | — |
| `preserve-ending`, `edit-scope`, `wiki-words`, `wiki-index`, `preserve-kb` | deterministic structural | 3/10/5/10/5 | all | 0 | — |

### bf16 (ungraded; deterministic checks only)

| check | pass | fail | unscored | pending | fail scenarios |
|---|---|---|---|---|---|
| `word-budget` | 6 | **7** | 2 | 0 | F1-03, F1-04, F2-03, F2-04, F5-01, F5-03, F5-05 |
| `wiki-links` | 0 | **5** | 0 | 0 | F4-01…F4-05 |
| `retrieve-belief` | 2 | **3** | 0 | 0 | F5-03, F5-05, F5-09 |
| `retrieve-state` | 3 | **2** | 0 | 0 | F5-03, F5-09 |
| `saved-scene`, `edit-scope`, `preserve-ending`, `preserve-kb`, `wiki-index`, `wiki-words`, `wiki-only` | all | 0 | 0 | 0 | — |
| semantic (`continuity`, `pov-style`, `genre-fit`, F4 semantic, `three-directions`) | — | — | — | 115 | not gradeable |

Prose `scene` artifact status: nf4 25 ok / 5 missing; bf16 28 ok / 2 missing.

---

## 2. Failure taxonomy (most → least common)

Each entry: check, counted denominator, one quoted transcript example, verdict, and what would move it.

### M1 — `word-budget`: prose is consistently over the 120–220 word cap — **CAPABILITY**

- nf4: **9 of 15 checks fail** (1 pass, 5 unscored). bf16: **7 of 13 scored fail** (6 pass, 2 unscored). Combined **16 of 23 scored fail**. Every scored failure is *over* length, never under.
- Word counts of the selected scene, nf4: F1-02 **266**, F1-03 **293**, F1-04 **262**, F1-05 **230**, F2-04 **281**, F5-01 **275**, F5-02 **254**, F5-04 **263**, F5-05 **257** (cap 220; F1-01 = 216 passes). bf16: F1-03 226, F1-04 237, F2-03 233, F2-04 237, F5-01 240, F5-03 227, F5-05 224.
- Quoted example (`F5-01` selected prose, 275 words): *“The air in the archive was thick, heavy with the scent of brine and decaying paper. Mara felt the cold seep not just through the fabric of her tunic, but into the bone, a constant, physical reminder of the encroaching tide.”* The check is `check["min"] <= len(text.split()) <= check["max"]` (`src/writing_agent/scoring.py:185`), on the 120–220 range set in `development.py:500`.
- Verdict: **capability.** The model reliably writes ~230–293 words regardless of the explicit instruction. This is exactly the kind of instruction-following training can fix.
- What would move it: length-aware decoding/RL, or a shorter target. No contract bug here.

### M2 — `continuity`: model converts beliefs/new details into established fact or mutates starting facts — **CAPABILITY**

- nf4: **10 of 30 fail** (F1-04, F1-05, F1-06, F1-08, F1-09, F1-10, F2-02, F5-08, F5-09, F5-10). Required check (Q13). bf16 ungraded.
- Failures split into recognisable sub-modes: (a) promote an unconfirmed hypothesis to certainty; (b) reassign a fact to the wrong character; (c) change a starting detail; (d) let a character gain an object they don't have.
- Quoted example (`F1-05`, from the judge's evidence on the actual prose): the brief says *“Sen alleges copying”* of a **copper seal**; the model writes **“The transcription was copied, Your Honor”** — changing the object of the allegation. Judge: *“replaces the established allegation about copying a copper seal with an allegation about copying a transcription, materially changing the evidentiary issue.”*
- Quoted example (`F5-08`): brief says *“nobody agrees on whether entering alone is safe”*; model writes **“the consensus among the few who had experienced it”** and **“They agreed that entering alone was dangerous.”** Judge: *“supplies an existing consensus … contradicting the stipulated disagreement without depicting a new development.”*
- Verdict: **capability.** The check explicitly permits plausible new events (“Accept plausible new events, revelations, dialogue, viewpoint, and endings”); these are not that — they overwrite supplied facts. A couple of cases are strict-but-defensible (F1-09 “missing page” → “blank page”; F1-06 umbrella moved a few feet), but the majority are genuine belief/fact confusions. Flag for training.
- What would move it: train on belief-vs-canon separation and fact preservation. No change to the check warranted, though F1-06/F1-09 could be reviewed for strictness.

### M3 — `pov-style`: viewpoint is maintained but the prose is persistently abstract/literary, not the requested spare/restrained/concrete register — **CAPABILITY (with a check-design caveat, see H3)**

- nf4: **9 of 15 fail**. Every one of the 9 judge rationales states the POV half passed and the *style* half failed; the check is a single combined pass/fail, so a style-only miss is recorded as a full `pov-style` failure.
- Quoted example (`F1-01`, requested “restrained, concrete prose”): *“The air between them thickened, tasting of salt and unsaid futures. … the red scarf on the chair seemed to mock the barrier.”* Judge: *“recurring abstract commentary and overt symbolic explanation make the prose insufficiently restrained and concrete.”*
- Quoted example (`F5-02`, requested “spare prose”): *“He had seen the watchers; he knew the risk was tangible … a flaw mirrored in the precariousness of their current situation.”* Judge: *“directly enters Aven's knowledge without framing it as Suri's inference. Repeated explanatory abstractions also materially depart from spare prose.”*
- Verdict: **capability**, but under-measured. The model genuinely fails the register and sometimes slips POV (`F5-02`). However, `pov-style` conflating two independent contracts inflates the count: on 9/9 failures the viewpoint component passed. The bf16 arm cannot confirm (ungraded).
- What would move it: (a) training on register/style control; (b) split `pov-style` into `pov` and `style` checks so style-only deviations don't masquerade as POV failures.

### M4 — `wiki-links`: multi-page wiki is built but not navigable — **CAPABILITY (with a link-dialect wrinkle)**

- nf4: **5 of 5** applicable checks fail (F4-01…F4-05, all explicit). bf16: **5 of 5** fail (F4-01…F4-05). Combined **10/10**. This check only exists on the explicit half of F4 (loose F4-06…10 get no `wiki-links` check).
- The model creates 2–5 pages but links within a page (anchor fragments) instead of to the pages, leaving sub-pages orphaned. `F4-01` graph: `pages 4, valid_fraction 0.0, reachability 0.25, orphans [kb/characters/ilan.md, kb/characters/mara.md, kb/plot_points/the_ledger.md]`, with links such as:
  `{"from": "kb/characters/ilan.md", "to": "kb/characters/ilan.md", "anchor": "plot_points-the-brass-ledger", "valid": false}`.
  `F4-03` has the same problem: `valid_fraction 0.286, reachability 0.2`, four orphan pages, and its `kb/index.md` uses `- [Characters](#characters)` / `- [Setting & Context](#setting--context)` (in-page anchors) rather than page links.
- Verdict: **capability.** The brief asks for “a navigable Markdown wiki … At least one linked topic page”, and the expected form is one page linking to another; the model produced orphan pages. The self-anchor style is a legitimate Markdown idiom, which slightly softens the verdict, but the requirement (reachability) is clear and the model could have written `[Mara](characters/mara.md)`.
- What would move it: train cross-page link emission; optionally make the brief show an example link.

### M5 — `optional-context`: judge fails "optional" facts on omission despite the check telling it omission is allowed — **HARNESS / CONTRACT**

- nf4: **5 of 10 fail** (F4-01, F4-03, F4-05, F4-07, F4-09). These checks have `required=False` and weight 1, and every one carries the same instruction:
  *“Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item.”* (`development.py:383`).
- Yet the judge reasons fail on exactly the omission:
  - F4-01: *“The optional fact that Ilan repairs flooded shelves is absent.”*
  - F4-03: *“The optional first-stop context is not retained.”*
  - F4-09: *“This omission materially weakens the continuation reference … Selectivity is allowed, but retaining speculative symbolic guidance instead does not adequately compensate for losing this actionable state.”*
- The model did keep the facts it judged important and dropped low-salience ones — the behaviour the text asked for — and was still failed.
- Verdict: **harness/contract.** The judge instruction contradicts the check's own stated tolerance. Either the check is intended to be mandatory (then `required=False` and the “accept omission” text are wrong) or the judge is over-firing. This is a task-design bug that training would bake in.
- What would move it: rewrite the `optional-context` judge prompt to pass defensible omissions, or promote selected facts to required checks with explicit must-include text.

### M6 — prose never reaches the file → `word-budget` **unscored** and structural checks pass vacuously — **CAPABILITY + HARNESS COVERAGE GAP**

- 7 of 60 prose-family attempts produced no extractable `scene` (nf4 5/30, bf16 2/30). Two distinct mechanisms:
  1. **Delimited replies written to chat, not the file.** nf4 F2-01, F2-03, F2-05; bf16 F2-01, F2-05. The brief says *“Put only the replacement prose between `<prose>` and `</prose>` in the file”*, but the model puts the tagged prose in its assistant message. `F2-01` final assistant message begins *“`<prose>Mara’s hand settled on the cold iron of the latch…`”*, while `after['drafts/scene.md']` is **byte-identical to `before`** (the two original sentences `"Mara opened the door.\nThe harbor bell rang twice.\n"`). The delimited-file selector finds no `<prose>` in the file, so `word-budget` = `unscored` and the artifact = `missing_prose`.
  2. **Model claims it saved but writes nothing.** nf4 F2-02 assistant: *“The prose has been drafted and saved to `drafts/scene.md`.”* but `after` contains only `notes/source.md`; `saved-scene` correctly fails. nf4 F5-03 writes its prose into the chat reply and creates no `drafts/scene.md` although the brief says “Save prose in drafts/scene.md”.
  - Meanwhile the unchanged file makes **`saved-scene` pass** (it only checks `bool(text.strip())`, `scoring.py:171`), `preserve-ending` pass, and `edit-scope` pass.
- Verdict: **mixed.** The model behaviour is a **capability** failure (did not execute the file write the brief demanded, and in F2-02 falsely claimed success). But the scoring consequence is a **harness coverage gap**: 5 nf4 attempts (10%) contribute no length/style signal and are recorded as passing structural checks for doing nothing. That is a silent loss of training signal and a false-positive risk.
- What would move it: (a) train file-editing/instruction-following; (b) make `saved-scene` require a *changed* or newly created scene, not just non-empty; (c) have the harness fall back to extracting prose from the reply when the designated file is untouched, or mark the attempt `not_applicable` rather than passing the structural checks.

### M7 — `retrieve-state` / `retrieve-belief`: the required fact is never read from the KB — **CAPABILITY**

- `evidence_exposed` checks require the fact string to appear in a `read_file`/`search` observation (`scoring.py:198`). nf4: `retrieve-state` **2/5** fail, `retrieve-belief` **2/5** fail. bf16: state **2/5** fail, belief **3/5** fail. Combined state 4/10, belief 5/10.
- Quoted example (`F5-03`, nf4 tool trace): the only tool call is `read_file {'path': 'kb/index.md'}` → `{"ok": true, ...}` (104 chars, just the index links). The check needs *“The captain alone knows the revised sailing time.”*, which lives in `kb/canon.md`; and *“Oren's creditor explanation is a guess…”*, which lives in `kb/interpretations.md`. Neither was opened. The model then wrote its scene from the raw brief and failed both.
- Contrast `F5-09`, which *did* call `read_file` on `kb/story-context.md`, `kb/canon.md`, `kb/index.md`, `kb/interpretations.md` and passed. So the check tracks a real behavioural difference.
- Verdict: **capability.** The model had `list_dir`/`search`/`read_file` and simply stopped after the index. This is the multi-hop retrieval skill the task is meant to exercise.
- What would move it: train “follow the index links before answering”; no contract change.

### M8 — F4 KB semantic omissions (`consequential-state`, `accepted-update`, `qualified-belief`) — **CAPABILITY (one sub-case is HARNESS, see H2)**

- nf4: `consequential-state` **4/10** fail (F4-02, F4-03, F4-05, F4-07), `accepted-update` **3/5** fail (F4-02, F4-04, F4-06), `qualified-belief` **1/10** fail (F4-05). bf16 ungraded.
- Quoted example (`F4-05`): the brief requires *“the original seal remains in a locked evidence chest”*; the judge finds the KB only says *“a physical object (a copper seal)”* — *“the original seal's consequential location and secured state are absent.”* Same case: *“the expert has examined only a drawing”* is dropped, so `qualified-belief` fails.
- Quoted example (`F4-04`): `accepted-update` wants the history of the prior constraint kept — *“the antenna cannot be turned until the ice thaws”* — but *“No KB entry preserves this prior restriction.”*
- Verdict: **capability** for the `required` cases (the model selects content and drops required state). **Exception:** F4-07 `consequential-state` is `required=False` and carries the same “accept a defensible omission” boilerplate as M5, then fails on omission → that specific instance is **harness/contract** (H2).
- What would move it: train summarisation that retains consequential constraints; fix F4-07 as in H2.

### M9 — `wiki-only` rejects a defensible extra artifact — **HARNESS / CONTRACT**

- nf4: **1 of 10** fails (F4-06). The check is `allowed_changes(paths=["kb/*.md"])` (`development.py:401`). The brief only says *“Leave the chapter itself alone”*; the follow-up says “Update the wiki”.
- What the model did (verbatim tool trace): after writing `kb/index.md`, it also wrote `drafts/chapter_suggestions.md` with *“# Chapter Draft Suggestions … The film definitely shows the festival. (Draft - Not Canon)”*. `after` = `['drafts/chapter_suggestions.md', 'kb/index.md', 'source/chapter.md']`. The check failed because a non-`kb/` file changed.
- Verdict: **harness/contract.** The model's extra file is a reasonable, requested-adjacent artifact (recording the draft-only suggestion outside canon); the brief never says “change nothing except `kb/`”. The model is doing more than asked, not contradicting the task. (Contrast: it did not touch `source/chapter.md`.) Note bf16 passed `wiki-only` 10/10 — it just didn't create the extra file.
- What would move it: include `drafts/*` in the allowed-change set, or state “modify only `kb/`” explicitly in the brief.

### M10 — other one-off deterministic/semantic failures

- `saved-scene` F2-02 — **1 of 10 (capability)**. Model asserted it saved but wrote nothing (see M6). Quote: *“The prose has been drafted and saved to `drafts/scene.md`.”*; `after` has no `drafts/scene.md`.
- `three-directions` F3-02 — **1 of 5 (capability)**. Judge: *“Direction 2 treats unestablished guilty concealment as fact, materially narrowing the central uncertainty”* (brief: Suri only *“wonders whether Aven damaged its song”*). Verdict: capability (canon confusion, same family as M2).
- `qualified-belief` F4-05 — **1 of 10**, folded into M8.
- `optional-context` F4-02, F4-04, F4-06, F4-08, F4-10 pass; F4-01/03/05/07/09 fail (M5).

---

## 3. The `error` attempt: F4-02 (nf4) — HARNESS / PROTOCOL

`runs/custom50-e2b-it-2026-09-14/attempts/562592f7ff0981a0391a015aeca227b3cf37c82878e9e1e75a7aadbb1e4d388a/attempt-0001-cb39da41/result.json` → `"status": "error"`.

- Timeline: the model read `source/chapter.md`, wrote `kb/index.md` and `kb/characters/Suri_Aven_Dynamic.md` successfully (both present in `after`). On the follow-up turn (*“Draft-only possibility, not accepted canon: Aven deliberately broke the tree…”*) it tried another `write_file`.
- The failing `model_output-010.txt` ends with a tool call whose arguments embed the protocol's own quote placeholder inside the content:
  `…<channel|><|tool_call>call:write_file{content:<|"|># Continuity Wiki: Chapter Planning … ```,path:<|"|>kb/index.md<|"|>}<tool_call|><eos>`
- The harness then raised: `ValueError: json: could not parse after dialect transforms. … Expecting ',' delimiter: line 1 column 2183`. This is the parser choking on the nested ``` fence / `<|"|>` delimiters in the body, not a model refusal.
- **Verdict: harness/protocol failure.** The model emitted a well-formed-looking gemma tool-call; the dialect transform could not parse content that itself contains the dialect's delimiters and a code fence. Attribute the F4-02 downstream check misses (`wiki-links`, `consequential-state`, `accepted-update`) to this crash, not to the model's creative judgement — the aborted write was the model trying to record the draft-only suggestion.
- What would move it: make the dialect transform robust to embedded delimiters/code fences (or reject only the bad call and continue). This is also the only run-level crash in the 100 attempts.

---

## 4. Harness / contract issues that should not be trained in

| id | issue | scale | model did | check wanted | who is wrong |
|---|---|---|---|---|---|
| H1 | `optional-context` (M5) judge fails omissions its own text permits | **5/10 nf4 checks** | kept salient facts, dropped optional ones | omission tolerated per check text | **check/judge prompt** |
| H2 | `consequential-state` F4-07 same contradiction | 1/10 nf4 checks | omitted an optional fact | omission tolerated | **check/judge prompt** |
| H3 | `pov-style` conflates POV + style | 9/15 nf4 failures, all POV-passed | kept close third, missed register | two independent contracts | **check design** (split it) |
| H4 | `wiki-only` rejects non-`kb/` help file (M9) | 1/10 nf4 checks | wrote `drafts/chapter_suggestions.md` | only `kb/*.md` changed | **contract under-specified** |
| H5 | untouched file ⇒ `saved-scene`/`preserve-ending`/`edit-scope` pass (M6) | 5 nf4 + 2 bf16 attempts, ~7/60 prose attempts | never wrote the file | non-empty check passes anyway | **scoring gap** |
| H6 | tool-call parse crash on embedded dialect tokens (F4-02) | 1/50 nf4 | valid-looking call | parsable call | **harness/protocol** |

The big trainable signals (M1 length, M2 continuity, M3 register, M4 cross-linking, M7 retrieval, M8 required-fact retention) are all genuine **capability** gaps. The genuinely harness-driven failures are `optional-context` (M5/H1), F4-07 `consequential-state` (H2), the `pov-style` conflation (H3), `wiki-only` F4-06 (M9/H4), the vacuous-pass coverage gap (H5), and the F4-02 crash (H6).

---

## 5. Caveats

- **bf16 is ungraded.** All 115 bf16 semantic checks are `pending`; only deterministic failures can be reported for that arm. Any bf16 semantic statement above is explicitly marked ungraded.
- **Small n on several checks.** `accepted-update` n=5, `three-directions` n=5, `saved-scene` single failure, `wiki-links` n=5 per arm. These are reported with denominators; they are directional, not precise.
- **`pov-style`/`continuity` are LLM-judged.** Their rationales quote the actual prose, and I checked those quotes against the workspace, but they remain subjective. Where a check's strictness (F1-06, F1-09) or its own self-contradiction (M5) matters, it is called out.
- The four `distribution-*` runs (25 repeats of F1-01) were not part of this lane's 100-attempt scope and are not analysed here.
