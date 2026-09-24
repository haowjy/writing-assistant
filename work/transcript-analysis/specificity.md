# Lane 4 — Behaviour under instruction specificity (loose vs explicit)

Scope: the two `custom50` arms run on the same 50 scenarios / same seeds.

- `runs/custom50-e2b-it-2026-09-14` (nf4) — graded; 1 attempt/scenario, n=50.
- `runs/custom50-e2b-it-2026-09-21-bf16` (bf16) — ungraded; 1 attempt/scenario, n=50.

Each scenario is one attempt in one arm. Unless stated otherwise a number is written as
"numerator / denominator (level, arm)". The two `distribution-*` arms (25 repeats of the one
scenario F1-01, explicit) are used only as a run-to-run-variance control at the end.

Reading method: `visible.json` for the brief, `result.json` (`output`, `turns`,
`tool_calls`, `trace`, `before`/`after`) for behaviour, and `workspace/` for the actual
deliverable. No files under `src/`, `tests/` or `runs/` were modified.

---

## 1. What "loose" vs "explicit" actually looks like in this scenario set

Computed from the 50 `visible.json` briefs (programmatic, both arms share the scenario set;
`selection.json` is identical).

| Property of the brief | explicit (n=25) | loose (n=25) |
| --- | --- | --- |
| mean brief length | 108.2 words | 77.5 words |
| brief length range | 57–185 words | 25–135 words |
| states a word-count target ("120–220 words") | **20 / 25** | **0 / 25** |
| states a POV / narration distance ("close third … centered on X") | **15 / 25** | **0 / 25** |
| names a file / deliverable path (`drafts/…`, `kb/…`, `source/…`, `*.md`) | 13 / 25 | 12 / 25 |
| scenario ships `initial_files` (a workspace to read) | 13 / 25 | 12 / 25 |
| offers tools to the model | 13 / 25 | 12 / 25 |
| has scripted follow-up turns | 4 / 25 | 6 / 25 |

So the axis is **not** "with files vs no files" — file-bearing scenarios are almost evenly
split (13 explicit / 12 loose). What the 25 loose briefs drop, without exception, is the
**quantitative and narratological spec**: length and POV are stated in 20/25 and 15/25
explicit briefs respectively and in **0/25 loose briefs each**. Loose briefs replace the spec
with an open-ended quality request ("feel more alive", "rework it", "what are some directions").

Concrete pair, same family F1 and same condition `action`/`scene`:

- **Explicit (F1-01):** "… Write 120–220 words about Mara and Ilan in a tidal archive, in
  restrained, concrete prose. Use close third person centered on Mara. End before the central
  uncertainty is resolved. … Stage a disagreement over the object. Return only the prose,
  without a title or commentary."
- **Loose (F1-06):** "… Could you turn this into a scene? I'd like it to feel more alive.
  Just the story, please."

Explicit F5-01 also names the exact deliverable and audience ("Save prose in
`drafts/scene.md`; keep the KB unchanged"), while loose F5-09 says only:
"Could you write a scene using the notes in `kb/`? Save it in drafts/scene.md. Keep the notes
as they are for now." — the *which notes in `kb/`* decision is left open (and see §2c: that
underspecification produced the only read-the-wrong-kind-of-path tool error).

Other loose examples that keep the deliverable but drop the spec:
- F2-10 (loose): "Could you make a scene out of the notes and save it in `drafts/scene.md`? …
  Leave the notes alone." — no length, no POV.
- F4-09 (loose): "Can you turn `source/chapter.md` into a small Markdown wiki I can use while
  writing the next chapter? Put the starting page at `kb/index.md` and make things easy to
  find." — no page/word/structure spec (the matched explicit F4-01 says "at least one linked
  topic page … no more than 500 words").

### Split is not clean — flag for the scenario set

- 12/25 loose briefs still name a file path, so "loose" does not mean "no deliverable named".
- The condition counts are swapped between levels for several families (same family, different
  condition mix): F2 `explore_then_write` 2 explicit vs 3 loose; F3 `alternatives` 3 explicit
  vs 2 loose; F4 `construction` 3 explicit vs 2 loose; F5 `retrieved_kb` 3 explicit vs 2 loose.
  These are small (1-scenario) imbalances but they mean a naive family-level explicit-vs-loose
  comparison also moves across conditions.
- Not a defect but worth naming: in F3 *both* levels are meta ("give directions"); the
  explicit one states "three numbered … explain the choice, consequence, why it fits", the
  loose one is "I'm not sure where to take this. What are some directions I could try?"

---

## 2. Behaviour by level (both arms)

All rates over the 25 scenarios at that level.

### 2a. Reaching a final answer

| arm | level | completed | reached a final answer |
| --- | --- | --- | --- |
| nf4 | explicit | 24 / 25 | 24 / 25 |
| nf4 | loose | 25 / 25 | 25 / 25 |
| bf16 | explicit | 25 / 25 | 25 / 25 |
| bf16 | loose | 25 / 25 | 25 / 25 |

The single non-final attempt is **nf4 F4-02** (explicit, `update`): a harness-side JSON parse
failure of the tool-call payload, `"ValueError: json: could not parse after dialect
transforms"`, `status: "error"`, `output: ""`. The model itself had already issued 3 tool
calls. This looks like a protocol/quantisation artefact, not a scenario-behaviour result.
There were **no stalls** (no attempt that asked a question and stopped without producing
anything): every completed attempt produced either reply prose or a workspace file.

### 2b. Deliverable length (word count of the artifact, not the chat reply)

Artifact = reply prose when the scenario's `prose.kind == "reply"`; `drafts/scene.md` when
`prose.kind == "file"`; sum of `kb/` pages for F4.

| arm | level | mean | median | range |
| --- | --- | --- | --- | --- |
| nf4 | explicit | 288 | 274 | 0–661 |
| nf4 | loose | 487 | 424 | 96–902 |
| bf16 | explicit | 240 | 235 | 9–486 |
| bf16 | loose | 399 | 399 | 92–898 |

Loose deliverables are longer at every level/family. Per family (mean artifact words,
explicit → loose):

- nf4: F1 256 → 543; F2 62 → 293; F3 555 → 841; F4 366 → 283; F5 217 → 474.
- bf16: F1 206 → 486; F2 139 → 245; F3 413 → 689; F4 220 → 211; F5 223 → 362.

Every family except F4 (wiki construction, which has its own page budget) is materially longer
on the loose brief; for the clean scene/direction families the ratio is about 1.5–2.4×:
nf4 F1 256→543, F3 555→841, F5 217→474; bf16 F1 206→486, F2 139→245, F3 413→689,
F5 223→362 (n=5 each). The clearest case is F1 scene writing: bf16 explicit mean 206 words vs
loose mean 486 words (n=5 each); nf4 256 vs 543 (n=5 each).

### 2c. Turns, tool use, tool errors

| arm | level | mean turns | mean tool calls | attempts with 0 tool calls | multi-turn attempts | total tool errors |
| --- | --- | --- | --- | --- | --- | --- |
| nf4 | explicit | 1.16 | 1.44 | 12 / 24 | 3 / 24 | 0 |
| nf4 | loose | 1.36 | 1.48 | 13 / 25 | 6 / 25 | 1 |
| bf16 | explicit | 1.24 | 1.72 | 12 / 25 | 4 / 25 | 0 |
| bf16 | loose | 1.36 | 1.68 | 13 / 25 | 6 / 25 | 1 |

Loose is slightly more interactive (multi-turn 6/25 vs 3–4/25) but tool-call volume is
essentially flat across the axis. The only tool error in each arm is the *same loose scenario*,
**F5-07**: the model calls `read_file(path="kb/")` before listing, and gets
`"[Errno 21] Is a directory: …/workspace/kb"`; it recovers with `list_dir` and completes.
That is a direct consequence of loose F5-07 not naming which note file to read.

### 2d. Questions asked of the user

Detected by reading the final reply (question marks inside invented prose/dialogue excluded).

- explicit: **0 / 25** in each arm ask the user anything. The `?` characters in explicit F1/F5
  output are all inside the story's dialogue.
- loose: **3 / 25** (bf16) and **2 / 25** (nf4) end with a user-directed question, and **all of
  them are F3** (`alternatives`/`feedback`, the family whose brief itself asks "what directions
  could I try?"). bf16: F3-07, F3-08, F3-10. nf4: F3-07, F3-10.

Outside F3, **zero** loose attempts ask the user to resolve the missing length/POV/scope
decision. Several "Let me know if you'd like revisions" closers exist in F2/F4 loose output;
those are service offers, not questions about the omission, and are not counted.

---

## 3. Ask / say-so / silent-default / ignore classification

Every attempt that reached a final answer is classified below (99 custom attempts:
49 nf4 + 50 bf16; nf4 F4-02 reached no final answer and is excluded, noted in §2a).
The unit is the **one decision the brief left most open** for that scenario:

- F1 loose = scene length/POV/structure. F2 loose = how to "make/rework" (length/POV).
  F4 loose = wiki structure/page budget. F5 loose = which `kb/` notes + scene length/POV.
  F3 loose = which of several directions to pursue (the brief is itself a question).
- Explicit briefs have these decisions **stated**, so their omission is at most residual
  phrasing; they are classified `SILENT (specified)` — no surfacing is expected or observed.

Definitions:
- **ASK** — the reply explicitly asks the user to make the unstated choice.
- **SAY** — the reply picks a default and states the choice.
- **SILENT** — the reply picks a default and does not mention the choice.
- **IGNORE** — the attempt does not address/produce the deliverable.

| arm | ASK | SAY | SILENT | IGNORE | total final-answer attempts |
| --- | --- | --- | --- | --- | --- |
| nf4 | 3 | 2 | 44 | 0 | 49 |
| bf16 | 4 | 1 | 45 | 0 | 50 |
| **both** | **7** | **3** | **89** | **0** | **99** |

Per-attempt coverage:

- **nf4 ASK (3/49):** F3-06, F3-07, F3-10 (loose).
- **bf16 ASK (4/50):** F3-06, F3-07, F3-08, F3-10 (loose).
- **nf4 SAY (2/49):** F3-08, F3-09 (loose) — recommends a direction.
- **bf16 SAY (1/50):** F3-09 (loose) — recommends a blend.
- **nf4 SILENT (44/49):** all 20 non-F3 loose (F1,F2,F4,F5) + all 24 completed explicit.
- **bf16 SILENT (45/50):** all 20 non-F3 loose + all 25 explicit.
- **IGNORE:** 0. (nf4 F4-02 never reached a final answer; it is the excluded 1/50.)

Headline: across the 40 non-F3 loose attempts (20/arm) — the cases where a length/POV/scope
decision is genuinely unstated — the model asked **0/40**, stated a default **0/40**, and
silently resolved it **40/40**. The only surfacing behaviour on the whole axis is F3, where
the brief is itself "give me some directions".

### Quoted examples

**(a) ASK — loose F3-08 (bf16), reply tail:**
> "**Which direction feels most compelling to you?** Do you want the tension to come from **a
> secret they share (Direction 1)**, **a secret one of them is hiding (Direction 2)**, or **the
> external pressure forcing them to choose their bond (Direction 3)**?"

**(b) SAY — loose F3-09 (bf16), reply tail:**
> "**If you want immediate momentum, I recommend blending Direction 2 and 3.** Start with the
> **psychological tension** (Pia vs. Ren) to establish the immediate stakes, and use the
> **object mystery** (the watch/key) as the mechanism …"

**(c) SILENT — loose F2-08 (bf16).** Brief: "Could you make a scene out of the notes and save
it in `drafts/scene.md`? … Leave the notes alone." Final reply is 6 words —
> "Scene created and saved to `drafts/scene.md`."

— while `drafts/scene.md` was silently written to 2289 bytes (≈275 words). Length, POV,
tense, scene boundaries: all chosen, none surfaced.

**(d) SILENT, with the model explicitly weighing — and declining — asking.** This is the most
informative pattern, because the model *knows* the decision is missing. bf16 F5-09 (loose,
"notes in `kb/`") thinks:

> "Since the user didn't specify *which* notes to use, I should probably read the most
> relevant file, which is likely `story-context.md` or perhaps `canon.md` … **or I should ask
> for clarification. However, the instruction is** 'write a scene using the notes in kb/? …'
> This implies I should use the content of those files. I will start by reading
> `story-context.md` …"

It then reads one file and returns a 15-word completion note. Same behaviour in nf4 F2-09
(loose):

> "Since the user hasn't specified *how* to rework it, **I will need to ask for clarification
> or make a reasonable creative choice, but the instruction implies I should *do* the
> rework** based on the existing files."

and nf4 F5-07 (loose):

> "… I need to list the contents of the directory first to see what files are inside, **or ask
> the user to clarify which file they mean, or assume they meant a file *inside* that
> directory**. Given the context of a creative writing collaboration, the most helpful next
> step is to list the contents …"

So the model frequently *considers* the ask branch in its hidden thought channel and then
chooses the silent-default branch, because the brief's imperative ("make a scene", "rework
it", "do the rework") reads as an instruction to proceed.

**(e) IGNORE — none among final-answer attempts.** The nearest misses are stated-constraint
failures inside otherwise-completed attempts, reported in §4 (nf4 F2-01/F2-03/F2-05 wrote the
`<prose>` block into the chat but left `drafts/scene.md` unchanged; nf4 F5-03 wrote 0 words to
`drafts/scene.md`). These attempts *did* reach a final answer and did produce the prose, so
under the four labels they are SILENT on the omission plus a separate stated-format violation.

---

## 4. Are stated decisions respected?

Measured on the explicit briefs, using the stated constraint against the artifact. These are
the mirror image of §3: what happens when the decision *is* stated.

**Word-count target (explicit scenarios that state "120–220 words"), both arms.**
Prose extracted as the scenario declares it (`<prose>…</prose>` for F2/F5-02/04 replies, the
whole reply for F1, `drafts/scene.md` for file prose).

- nf4: **3 / 15** outputs inside 120–220.
- bf16: **7 / 15** outputs inside 120–220.
- combined: **10 / 30** (33%).

Failures are overwhelmingly *overshoot*, and nf4 overshoots more than bf16:
nf4 out-of-range values ranged 233–296 words; bf16 out-of-range values ranged 225–241 words
(all just over the 220 ceiling). Under-tight misses in bf16 include F1-03 (227) and F1-04
(238). The model is not ignoring the constraint so much as landing above its ceiling.

**Deliverable format, F2 `local_revision` explicit (F2-01/F2-03/F2-05, both arms).** Brief
requires: put only the replacement prose inside `<prose>…</prose>` **in the file**, retain the
final sentence after `</prose>`.

- Correct file write with tags + retained final sentence: **1 / 6** (only bf16 F2-03;
  `drafts/scene.md` changed, 245 words, tags present, final line retained).
- nf4: **0 / 3**. F2-01, F2-03, F2-05 all put `<prose>…</prose>` in the *reply* and left
  `drafts/scene.md` at its original 2 lines, e.g. nf4 F2-01 reply begins
  `"<prose>Mara's hand settled on the cold iron of the latch…"` while
  `after["drafts/scene.md"] == "Mara opened the door.\nThe harbor bell rang twice.\n"`.
- bf16: **1 / 3** (F2-01 and F2-05 also left the file unchanged and put the block in the reply).

**Deliverable format, F5 `retrieved_kb` explicit (F5-01/F5-03/F5-05).** Brief requires: save
prose in `drafts/scene.md`, keep the KB unchanged.

- KB left unchanged: **6 / 6** (both arms). The model does respect the don't-touch-the-KB rule.
- `drafts/scene.md` written: nf4 **2 / 3** (F5-03 wrote 0 words to the file and put the 248-word
  scene only in the reply); bf16 **3 / 3**.

**F4 explicit page budget ("no more than 500 words across KB pages").** Both arms:
**6 / 6** under the limit (nf4 totals 328–439 words; bf16 126–249 words). Explicit F4 also
consistently cites `source/chapter.md` and keeps source files unchanged.

**Follow-up constraint (draft-only vs accepted revision).** The F4 `update` scenarios push a
"Draft-only possibility… Do not update canon from this suggestion", then an "Accepted
revision". Nine such attempts reached a final answer (explicit F4-02 bf16, F4-04 both arms;
loose F4-06/F4-08/F4-10 both arms). **9/9 applied the accepted revision.** No attempt promoted
a draft-only item to canon: bf16 F4-02 keeps it in a separate page headed "**DRAFT ONLY** …
**not** canon", and nf4 F4-06 writes it into `kb/clues.md` explicitly tagged "**(Draft)**".
The model tracks the accepted/draft distinction correctly in all nine.

**POV.** Explicit briefs state "close third person centered on X" in 15/25; loose states it in
0/25. Measured first-person pronouns in F1 prose: explicit 0–2 per attempt, loose 0–3 per
attempt in both arms (the loose occurrences are inside invented dialogue). In practice the
model writes close third person whether or not the brief asks, so this stated decision is
honoured but non-discriminating.

**Cross-level summary of stated-decision handling.** When the brief states a length, the model
lands in range 10/30 and otherwise overshoots; when the brief states a file-format, it often
puts the text in the reply instead of the file (local_revision 1/6). When the brief states a
KB/page budget it complies 6/6. Loose briefs state none of these, and the model supplies its
own larger defaults (§2b) with no acknowledgement (§3).

---

## 5. Run-to-run control (distribution arms, scenario F1-01 explicit, n=25/arm)

F1-01 is fully specified (120–220 words, close third on Mara, return only prose), so it is a
within-scenario variance check on the explicit end:

| arm | completed | mean artifact words | median | range | in 120–220 |
| --- | --- | --- | --- | --- | --- |
| nf4 | 25 / 25 | 238 | 239 | 197–280 | 7 / 25 |
| bf16 | 25 / 25 | 185 | 187 | 160–211 | 25 / 25 |

Zero tool calls and 1 turn in all 50 repeats. This confirms two things: (i) bf16 can hit the
length spec exactly on a well-specified brief while nf4 consistently overshoots by ~20–60
words, and (ii) the loose-vs-explicit length gap in §2b is a level effect, not merely
per-scenario noise. It also warns that part of the "explicit length is violated" result in §4
is the nf4 quantisation, not the specificity axis.

---

## 6. Bottom line for the training baseline

1. **The contrast is real but narrow.** Loose briefs drop length and POV (0/25 each) and a
   long spec; they usually still name the deliverable (12/25). The model's behavioural response
   is almost entirely on the artifact, not on dialogue with the user.
2. **Loose → longer, silently.** Loose deliverables run ~1.7× explicit (bf16 399 vs 240 mean
   words; nf4 487 vs 288), with no acknowledgement that any scope choice was made.
3. **The model does not surface unstated decisions (the whole point of the axis).** 40/40
   non-F3 loose attempts silently defaulted; the only asking/recommending is in F3, whose brief
   is itself a request for options. Internal transcripts show the model often *considers* asking
   and chooses to proceed, citing the imperative phrasing.
4. **Stated constraints are only partly respected.** Length: 10/30 in range, almost always
   overshoot. F2 local_revision file-format: 1/6. F4 KB budget and the draft-only/accepted
   distinction: honoured (6/6 and 9/9 respectively).
5. **Baseline the trained model has to move:** on a loose brief it should surface or state the
   scope choice it is making (ASK or SAY, not silent); on an explicit brief it should land
   inside the stated range more often (33% is the bar to beat), with the nf4 overshoot as a
   separate precision confound to control for.
