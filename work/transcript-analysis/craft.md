# LANE 2 — The prose itself: why does the writing read as weak?

Analysis of generated prose from `google/gemma-4-E2B-it` ("E2B") across the
custom50 and distribution runs. Graded arm = nf4 (`runs/custom50-e2b-it-2026-09-14`),
whose prose profile is **2.20/5 overall**, with the worst named dimensions
**redundancy 2.00, language 2.20, characterization 2.24, pacing 2.28**
(`astra-graded/*/scorecard.json`, `scores.Q2.dimensions`).

This report is written from the text, not from scores. Every habit below carries
a count over the primary sample and at least one verbatim quotation.

---

## 0. Corpus and method

**Primary sample (read in full, quoted below): 25 distinct pieces of generated
prose** — the prose actually selected and graded by the ASTRA pipeline, one per
scenario at `runs/custom50-e2b-it-2026-09-14/astra-graded/<ID>/scorecard.json`
(`artifacts[0].text`), dumped to `graded_prose/<ID>.md`:

- F1 (all 10): `F1-01 … F1-10`
- F2 (6 with selected prose): `F2-04, F2-06, F2-07, F2-08, F2-09, F2-10`
- F5 (9): `F5-01, F5-02, F5-04, F5-05, F5-06, F5-07, F5-08, F5-09, F5-10`

**Corroborating sample (read to check persistence):** the same 25 scenarios in
the bf16 arm (`bf16_prose.json`), plus all **50** distribution outputs
(`distribution-e2b-it-2026-09-21-nf4|bf16`, 25 repeats of F1-01 each) =
**100 pieces of generated prose** available, 25 read line-by-line and ~35 read
closely.

**F3 and F4 produce no prose.** F3 asks for "three numbered … next-scene
directions" (no `Q2` grade), and F4 asks for a Markdown continuity wiki. They are
therefore out of scope for a *prose* study and are treated only as controls
below. So the "five families" claim has to be qualified: prose quality is only
measurable in F1, F2 and F5, and the prose habits are the same in all three.

**Counting note.** Unless stated otherwise, every fraction is "N of 25 graded
pieces". Regexes were run with `PYTHONPATH=src uv run python3` over the dumped
texts; the raw counts are reproducible from `corpus.json`, `bf16_prose.json` and
`graded_prose/`.

---

## 1. Ranked habits

### H1. The atmospheric-opening template — **21/25 (84%)**
**Model default (not requested).**

The overwhelming majority of pieces open with `The <air|silence|static|light|heat|bell|chill> …`
followed by two or three stacked sensory abstractions, usually with `thick`/`heavy`.
It is the single most visible signature in the corpus.

- `F1-01`: "The damp air of the archive clung to Mara, heavy with the scent of brine and decaying paper."
- `F1-03`: "The salt-laced air of the night ferry hung thick, heavy with the scent of brine and something older—something tied to the stone."
- `F1-09`: "The air in the clockmaker's workshop was thick—a cloying mixture of fine brass polish, aged oil, and the dry, metallic scent of centuries of stopped time."
- `F1-10`: "The air in the restaurant was thick, heavy with the ghosts of last night's patrons—a cloying mix of fryer oil, stale linen, and something vaguely sweet, like old sugar."
- `F2-04`: "The static was the only constant, a thin, high shiver against the silence of the cabin."
- `F5-01`: "The air in the archive was thick, heavy with the scent of brine and decaying paper."

The template is so fixed that openings converge verbatim across independent
repeats of the same scenario. In the 25 bf16 distribution outputs of F1-01:

| opening stem | count |
|---|---|
| begins `The damp air of the archive` | **12/25** |
| contains `clung to Mara` | **12/25** |
| contains `decaying paper` | **12/25** |

("The damp air of the archive clung to Mara's coat, heavy with the smell of brine
and decaying paper…" appears in near-identical form many times.) The nf4
distribution arm is more varied but still opens 11/25 with the same
`The <air/silence/static/…>` construction and uses `brine and` 11/25.

**Instruction vs default:** the explicit briefs explicitly ask for *concrete*
prose ("restrained, concrete prose", "spare prose"). The default opening is the
opposite — an atmospheric abstraction before any person acts. This is a model
habit, and it directly feeds **language 2.20** (diffuse, competing sense-images).

**Line-level intuition:** delete the weather. Open on the first concrete action
or line of dialogue that carries the conflict: not "The damp air of the archive
clung to Mara…" but "Mara set the brass ledger's empty tray down on Ilan's
workbench and said, 'You sold it.'"

---

### H2. The appositive/participial afterthought — **25/25 (100%), 122 occurrences**
**Model default.**

Nearly every sentence is `[main clause], [elaborating fragment]` — a trailing
present participle, a possessive appositive, or an em-dash gloss. The same
syntactic shape recurs ~5 times per 120–220-word piece.

| construction | docs | hits |
|---|---|---|
| trailing `, …ing` clause | **25/25** | **122** |
| comma + possessive appositive (`, his/her/its/their …`) | **23/25** | **74** |
| em-dash, any use | 23/25 | 75 |
| em-dash appositive `—a/an/the …` | **20/25** | **44** |

Examples:
- `F1-01`: "The air between them thickened, **tasting of salt and unsaid futures**."
- `F1-08`: "The atmosphere thickened, **turning the sterile archive into a stage for inevitable, charmingly clumsy misunderstandings**."
- `F1-06`: "…something faintly metallic**—the ghost scent of old film**."
- `F5-01`: "Mara felt the familiar knot tighten in her chest**—the suspicion that lingered like the moisture in the air**."
- `F5-10`: "He smiled**—a wide, genuine smile that didn't quite reach his eyes**."

**Instruction vs default:** default. It explains both **language 2.20** (every
noun is immediately re-described by a second image, so focus diffuses: "shields,
knots, ghosts, snapping threads, shattering, drowning" per the F1-03 grader) and
**redundancy 2.00** (the afterthought usually restates the clause it follows).

**Line-level intuition:** allow one image per sentence to stand unexplained.
Prefer "The air between them thickened." full stop, and let the next action show
what it tasted of.

---

### H3. Telling interiority via `felt/knew/seemed` + abstract emotion — **`felt` 21/25 (47 hits), `seemed` 13/25, `knew/knowledge` 12/25**
**Model default.**

Emotion is announced rather than dramatised. The prose repeatedly says a
character "felt" a named abstraction, or "knew" a fact the scene could have shown.

- `F1-01`: "Mara **felt the familiar, hollow accusation rise, unproven and unverified**."
- `F1-07`: "She watched the sky… and **felt the familiar knot of worry tighten in her chest**."
- `F1-09`: "Pia **felt the familiar, sharp prickle of suspicion, a feeling she couldn't quite name, only identify**."
- `F5-08`: "The knowledge was heavy, **a lead weight in Kavi's chest**."
- `F1-08`: "She **knew** Kavi was already spiraling into the trap—the perfect defense mechanism for an overwhelmed clerk."

**Instruction vs default:** default. Directly explains **characterization 2.24**
— the grader's recurring note is that "narration explains feelings rather than
revealing individual motives" (F5-08) and that dialogue "largely delivers premise
information" (F1-08/F5-08).

**Line-level intuition:** replace the feeling-noun with a physical action that
only that character would do. "Mara felt the accusation rise" → "Mara turned the
ledger tray face-down so Ilan could not see it was empty."

---

### H4. Memorised stock phrases — **`felt the familiar knot tighten` 4/25; `familiar knot` 6/25; `heavy with the scent of` 4/25; `her gaze fixed on the` 4/25**
**Model default.**

Phrases recur verbatim across unrelated scenarios, including across genres and
across the two arms:

- `F1-03`: "Oren **felt the familiar knot tighten**."
- `F1-04`: "Nessa **felt the familiar knot tighten in her chest**."
- `F5-01`: "Mara **felt the familiar knot tighten in her chest**—the suspicion that lingered like the moisture in the air."
- `F5-04`: "She **felt the familiar knot tighten in her chest**."
- `F5-09` (also bf16): "Pia felt a familiar knot tighten in her chest."

This is the clearest evidence of a memorised emotional idiom: the same body
("chest"), the same metaphor ("knot"), the same verb ("tighten") across a
literary-fiction archive, a sci-fi radio station and a fantasy courthouse.
Shared 5-grams across the 25 graded pieces include **`felt the familiar knot tighten` (4 docs)**,
**`familiar knot tighten in her` (4)**, **`her gaze fixed on the` (4)**,
**`heavy with the scent of` (4)**, and **`the scent of brine and` (3)**.

**Instruction vs default:** default. Contributes to **language 2.20**
(unoriginal phrasing) and **redundancy 2.00** (same idiom re-used as if new).

**Line-level intuition:** ban the phrase. If two characters are both
apprehensive, find two different, character-specific tells.

---

### H5. Restating the same emotional condition every sentence — **abstract-noun density 16.8/1000 words; `thick|heavy` 22/25 docs (42 hits); `tension|pressure|dread` 17/25; `the silence` 20/25 (29 hits)**
**Model default.** *(This is the habit behind redundancy 2.00.)*

Scenes state the emotional weather, then restate it, then restate it again. The
F5-08 opening is a textbook case — four consecutive sentences all asserting dread:

> "The **silence** of the apartment was the worst part… the heavy, thick **quiet** that seemed to absorb every other noise… The **knowledge was heavy**, a lead weight in Kavi's chest… the **dread** was starting to seep past the logic… The **implication hung heavy**…"

The grader flagged exactly this on F5-08: *"Recurrent heavy silence, cold,
waiting, and dread often restate the same emotional condition."* On F1-07:
*"repeated assertions of mystery substitute for developing tension."*

**Instruction vs default:** default. Note the brief's own set-ups use abstract
framing ("Their everyday work carries the weight of that unspoken choice"), but
the model amplifies rather than resolves it. This is the largest single driver
of **redundancy 2.00**.

**Line-level intuition:** one assertion of mood per scene, maximum, and only
after an action has earned it. If a second mood sentence is needed, replace it
with a new fact or a new obstacle.

---

### H6. Ending on a named abstraction / suspended "waiting for" — **22/25**
**Mixed: the *unresolved* ending is instructed; *naming* the unresolvedness is default.**

Explicit briefs say "End before the central uncertainty is resolved." The model
complies, but almost always by *telling the reader the uncertainty is
unresolved*, rather than stopping on a concrete image or line.

- `F1-02`: "…the question of what had happened that day—who disappeared, and why—**hanging unresolved in the humid air**…"
- `F1-03`: "…**waiting for the moment when the uncertainty would either shatter the calm or drown them both**."
- `F5-01`: "…the tide remained indifferent, **promising only more saturation, never resolution**."
- `F5-04`: "…the real story **suspended, waiting for the thaw that might never come**."
- `F5-05`: "The question of true culpability… **remained suspended, unresolved**…"
- `F5-08`: "They were **always waiting for someone to speak the truth**."

Only 3 of 25 final sentences end on a concrete image or an action
(`F2-08`: "The stranger was close."; `F1-06`: "The cinema wasn't just closing
down; it was waking up."; `F1-09` ends in dialogue). 7/25 end literally on
"waiting for …".

**Instruction vs default:** the *non-resolution* is instructed; the *abstract
summary sentence naming the theme* is the model's own reflex. This is why
readers get exposition instead of image at the point of highest pressure, and it
feeds **pacing 2.28** (no concrete beat lands at the end) and **language 2.20**.

**Line-level intuition:** cut the closing sentence that explains what remains
unresolved. End one sentence earlier, on the object or gesture.

---

### H7. The "not just X; it was Y" / ", but" antithesis — **`not just` 10/25 (17 hits); `, but` 20/25 (35 hits); `wasn't just … it was` 7/25 (8 hits)**
**Model default.**

A balancing rhetoric that sounds profound and means little:

- `F1-06`: "The cinema **wasn't just** closing down; **it was** waking up."
- `F1-07`: "The heat in 1936 **was not just** a temperature; **it was** a physical weight…"
- `F2-07`: "It **wasn't just** dust; **it was** the palpable memory of scarcity."
- `F5-07`: "The reunion **wasn't just** a memory; **it was** the seed of a new hope…"
- `F5-01`: "Mara felt the cold seep **not just** through the fabric of her tunic, **but into the bone**…"

**Instruction vs default:** default. This is the mechanism by which the prose
"language 2.20" problem reads as *portentous but vague* — each balanced clause
adds an abstraction without adding information. Also contributes to redundancy.

**Line-level intuition:** choose the concrete half and delete the abstraction.
"The heat was a physical weight" is stronger than the paired construction.

---

### H8. Summarising/montage instead of dramatising — **iterative markers (`every/always/each/often/would`) 22/25 (42 hits); dialogue only 137/597 sentences = 23%**
**Model default.** *(This is the habit behind pacing 2.28 and characterization 2.24.)*

The briefs ask for a *scene*; the model frequently gives an iterated summary of
a situation. Examples of summary register:

- `F1-10`: "**For the next week**, Ada and Milo lived inside the walls of the family restaurant… **Every** clatter of dishes sounded amplified, **every** distant siren a potential alarm."
- `F2-07`: "**Every grain of dust** that stirred **seemed to carry the weight** of the drought…"
- `F5-07`: "The communal garden, once a vibrant tapestry of green, was now a landscape of resignation."

Dialogue is thin even when explicitly demanded. Across the 25 pieces only
**23%** of sentences are dialogue; the three briefs that explicitly say
"Let most of the scene unfold through dialogue" / "dialogue-led prose"
(`F1-03`, `F1-04`, `F5-04`) land at **24%, 26% and 37%** respectively — i.e. the
model largely ignores the "dialogue-led" instruction. `F1-05` is 11% dialogue
despite asking to "Show an interrupted practical task rather than summarizing."

**Instruction vs default:** default. Explains the grader's repeated note that
"dialogue largely delivers premise information" (F5-08) and that scenes "stall"
(F5-10, F5-09).

**Line-level intuition:** for every narrated beat, ask "who could say this
instead?" Convert summary sentences into a two-line exchange that changes what a
character knows.

---

### H9. Over-explaining motive and subtext — **`knew/knowledge` 12/25 docs; representative examples in F1-06, F1-08, F5-10**
**Model default.** *(Feeds characterization 2.24.)*

The narrator pre-chews the subtext:

- `F1-06`: "He **was supposed to be methodical, practical**—a skeptic facing the inexplicable."
- `F1-08`: "She **knew** Kavi was already spiraling into the trap—**the perfect defense mechanism for an overwhelmed clerk**."
- `F5-10`: "He **knew the script. He knew the mistake was inevitable**, and the ensuing chaos was the main event."

The grader's F5-10 note: *"narration repeatedly announces confusion and Milo's
success instead of making them evident through action."*

**Instruction vs default:** default. It substitutes for the interior
idiosyncrasy the briefs ask the writer to preserve.

**Line-level intuition:** delete the explaining clause after the em-dash. Trust
the preceding action ("He smiled—a wide, genuine smile…" is enough; the gloss
`that didn't quite reach his eyes` is already a cliché).

---

### H10. Brief/notes leakage, meta-commentary and screenplay format — **verbatim ≥6-word brief reuse 4/25; meta stage directions 3/25; script format 4/25 (all loose)**
**Mostly instruction-sensitive: it appears when the prompt is loose, and is suppressed when explicit.**

When the prompt is a raw notes dump, the model pastes the notes back into the
fiction:

- `F5-06` (loose): "The projector **must be checked before anything is screened.** … The cinema **is scheduled to close on Friday.**" — both sentences lifted verbatim from the brief.
- `F5-08` (loose): "people near the office are hearing **their own voices answering from empty rooms**."
- `F1-08` (explicit): "Only the supervisor can **authorize a corrected official copy**." — verbatim from the brief's fact list.
- `F1-01`: "I moved it **above the tide line**." — verbatim.

When constrained only by "make a scene" / "could you write a scene", the model
defaults to a screenplay:

- `F2-06` (loose): "**Setting:** The dusty, slightly melancholic main hall…
  **Characters:** *EMIL:* earnestly… **MR. HENDERSON** (Clearing his throat loudly)…"
- `F2-10` (loose): "**Setting:** The back kitchen… **(Scene Start)** … **(Scene End)**"
- `F2-08` (loose): "**Setting:** A sparsely lit, dusty archive room…"
- `F5-07` (loose) and `F2-06` end with meta commentary: "**(Scene End)**" / "The scene ends on the tension between formality and absurdity.)"

One piece (`F5-09`) leaks the artifact frame directly into the prose: "The
collector's square key, **mentioned in the notes**, felt heavy **in the context of**
the delicate escapement."

**Instruction vs default:** *instruction-sensitive.* Explicit briefs ("Return
only the prose, without a title or commentary"; "Put only the replacement prose
between `<prose>`") suppress the script format — none of the four script-format
cases is an explicit brief. So this is a genuine model default that competent
prompting contains, but a *loose* brief reliably triggers it.

**Line-level intuition:** treat the notes as canon to obey, not text to copy;
never write "mentioned in the notes" or "the scene ends".

---

### H11. Length overrun on an explicit word budget — **9/10 explicit pieces exceed 220 words; median 268 vs requested 120–220**
**Instruction non-compliance, not a style choice.**

Every explicit F1/F2/F5 brief specifies "Write 120–220 words" (or 120–220 for a
sentence expansion). The selected prose:

| spec | n | median words | over 220 words |
|---|---|---|---|
| explicit | 10 | **268** | **9/10** |
| loose | 15 | 428 | 13/15 |

Even the shortest explicit piece (`F1-01`, 218 words) only just fits.
Over-writing is the mechanical substrate of the redundancy score: there is more
text asserting the same mood than the brief asked for, and the extra text is
almost entirely H2/H3/H5 filler.

**Line-level intuition:** enforce the ceiling by cutting one atmospheric opener,
one appositive afterthought and one mood sentence per scene.

---

## 2. What the evidence does **not** support

- **"Conflict resolved too fast"** — not supported as the dominant failure. The
  endings are overwhelmingly *suspended* or *escalating* (22/25 end on named
  unresolvedness or a new threat). If anything the model under-resolves while
  over-explaining. The one clear early-resolution is `F5-07`, a romance that
  consummates the relationship: "The touch was everything—a bridge between the
  past longing and the uncertain future… the seed of a new hope, planted right
  there in the cracked soil."
- **"Hedge/adverb stacking"** — present but secondary: -ly adverbs run
  **17.9 per 1000 words** (166/9,259) and `seemed` appears in 13/25. This is
  real but orders of magnitude less defining than H2/H3/H5.
- **"Flat/absent dialogue"** — *supported*, but as part of H8, not separately:
  dialogue is 23% of sentences overall and fails to rise when the brief asks for
  dialogue-led prose.
- **n is small for genre effects.** With 25 graded pieces spread over ~8 genres,
  the sample cannot support per-genre claims. All the habits above were checked
  across F1/F2/F5 and both arms and are consistent.

---

## 3. Loose vs explicit: does specificity change the *writing quality*?

**No — it changes content discipline, not craft.** The same habits fire at
similar or higher rates in loose briefs.

| habit | explicit (n=10) | loose (n=15) |
|---|---|---|
| `The <air/silence/static>…` opening | 8/10 | 13/15 |
| `thick`/`heavy` | 9/10 | 13/15 |
| `felt` | 9/10 | 12/15 |
| `the silence` | 8/10 | 12/15 |
| `, but` antithesis | 7/10 | 13/15 |
| trailing `, …ing` | 10/10 | 15/15 |
| simile `like/as if` | 4/10 | 10/15 |
| `tension/pressure/dread` | 6/10 | 11/15 |
| `familiar knot` | 4/10 | 2/15 |

What specificity buys:

1. **Format compliance.** Explicit briefs suppress screenplay format and title
   commentary; loose briefs trigger them (H10).
2. **Fewer verbatim note pastes** in the most constrained explicit pieces (though
   `F1-01` and `F1-08` still paste).
3. **A (mostly ignored) word ceiling.**

What specificity does **not** buy: it does not make the prose more concrete. The
brief's explicit instruction — "restrained, concrete prose", "spare prose" —
is precisely the property the prose lacks. Both arms open with weather, restate
affect, and end on abstraction. If instruction specificity had fixed the craft,
the explicit pieces would look different from the loose ones; they don't.

Median length actually *rises* in loose briefs (428 vs 268 words), i.e. removing
the word ceiling lets the redundancy grow.

---

## 4. Synthesis: the causal story for 2.20/5

The rubric dimensions are downstream of two mechanical facts about the text:

1. **The model writes a mood, not an event.** H1 (atmospheric opening) + H5
   (mood restated) + H8 (summary instead of scene) mean a piece asserts how the
   scene feels rather than staging what happens. → pacing 2.28, redundancy 2.00.
2. **The model elaborates every statement instead of choosing.** H2 (appositive/
   participial afterthought, 122 occurrences) + H7 (balanced antithesis) +
   H3/H9 (telling and explaining) mean each sentence adds another gloss, image
   or abstraction rather than a new fact. → language 2.20, characterization 2.24.

Underneath both sits a small set of memorised phrases (`felt the familiar knot
tighten`, `heavy with the scent of`, `her gaze fixed on the`) that reappear as if
fresh, and a habitual over-writing past every word budget. The fix is editorial
and local: remove weather, remove afterthoughts, remove mood restatement, and
convert one narrated beat per scene into dialogue or physical action.

---

## Appendix — reproducibility

- `corpus.json` — every attempt (150) with output, brief, workspace files, metadata.
- `graded_prose/<ID>.md` — the 25 graded pieces.
- `bf16_prose.json` — the 25 bf16 counterparts.
- `graded_dims.json` — per-scenario Q2 rationale + dimension reasons.
- `all_texts.md` — all 150 attempt texts concatenated for manual reading.

Key machine counts (25 graded pieces unless noted): opening template 21/25;
trailing `, …ing` 25/25 docs / 122 hits; possessive appositive 23/25 / 74;
em-dash appositive 20/25 / 44; `felt` 21/25 / 47; `thick|heavy` 22/25 / 42;
`the silence` 20/25 / 29; `familiar knot` 6/25; `felt the familiar knot tighten`
4/25; ending-on-abstraction 22/25; explicit over-length 9/10; bf16-distribution
opening stem `The damp air of the archive` 12/25.
