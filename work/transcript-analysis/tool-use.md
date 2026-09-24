# Lane 3 — Does the model use its tools, or does it invent facts?

Scope: all **100 attempts** = 2 arms × 50 development scenarios.
- `nf4` = `runs/custom50-e2b-it-2026-09-14` (graded: `astra-graded/<ID>/scorecard.json`)
- `bf16` = `runs/custom50-e2b-it-2026-09-21-bf16` (ungraded)

Sources read: per-attempt `trace.jsonl` (`tool` records), `result.json`
(`tool_calls`, `tool_errors`, `turns`, `status`, `output`), `visible.json`
(`tools`, `initial_files`, `followups`, `brief`), and `workspace/` files.
No files under `src/`, `tests/`, or `runs/` were modified.

One structural fact dominates everything below: **half the scenarios were run
with no tools at all.** `visible.json["tools"] == []` for every F1 (10) and F3
(10) scenario and for the 5 F5 `supplied_kb` scenarios; those attempts also have
no `initial_files`. Only F2 (10), F4 (10) and F5 `retrieved_kb` (5) offer tools
and a workspace. This is why raw "used a tool / didn't" counts split 25/25 per
arm, not because the model declined to read.

---

## 1. Tool-call inventory (denominator = 100 attempts)

| Measure | nf4 | bf16 | Both |
|---|---|---|---|
| Attempts | 50 | 50 | 100 |
| Attempts where tools were offered | **25** | **25** | **50** |
| Attempts that called ≥1 tool | **25** | **25** | **50** |
| Tool-offered attempts that called ≥1 tool | **25/25** | **25/25** | **50/50** |
| Attempts with no tools offered that called a tool | 0/25 | 0/25 | 0/50 |
| Total tool calls | 73 | 85 | **158** |
| Calls per tool-using attempt (mean / max) | 2.92 / 6 | 3.40 / 9 | — |
| `result.json` `tool_calls` total | 73 | 85 | 158 |
| `result.json` `tool_errors` total | 1 | 1 | 2 |
| Attempt status `completed` / `error` | 49 / 1 | 50 / 0 | 99 / 1 |

**Every attempt that was given tools used at least one.** The model never
declined a workspace.

Breakdown by tool name (all 158 calls):

| Tool | nf4 | bf16 | Both | Share |
|---|---|---|---|---|
| `read_file` | 34 | 35 | **69** | 43.7% |
| `write_file` | 33 | 40 | **73** | 46.2% |
| `patch_file` | 4 | 7 | **11** | 7.0% |
| `list_dir` | 2 | 3 | **5** | 3.2% |
| `search` | 0 | 0 | **0** | **0%** |

Tool use is entirely determined by family/condition, identical across arms:

| Family / condition | tools offered | attempts that used tools (per arm) |
|---|---|---|
| F1 scene/continuation/dialogue/reflection/action | no | 0/10 |
| F2 `local_revision` / `explore_then_write` | yes | 10/10 |
| F3 `alternatives` / `feedback` | no | 0/10 |
| F4 `construction` / `update` | yes | 10/10 |
| F5 `retrieved_kb` | yes | 5/5 |
| F5 `supplied_kb` | no | 0/5 |

---

## 2. Read-before-write ordering

I reconstructed the ordered tool sequence per attempt and asked two questions:
(a) did any `write_file`/`patch_file` occur before the first
`read_file`/`search`/`list_dir`? (b) was the specific file being modified read
earlier in the same attempt?

**Result: 0 cold writes.** Of the 43 attempts that wrote anything, **0 wrote
before reading**. 61 writes created files that never existed (`kb/index.md`,
`drafts/scene.md`, etc.), so no prior read was possible. Of the 23 writes that
modified a file that already existed, 10 were preceded by a same-attempt read of
that exact path; the other 13 targeted files the model had itself created
earlier in the same attempt (F4 `kb/*`), i.e. it re-used its own prior write
rather than re-reading.

Hand-verifiable sample (tool sequence in order; both arms):

| Scenario | Arm | Tools offered | Sequence | read-before-write |
|---|---|---|---|---|
| F2-01 `local_revision` | nf4/bf16 | yes | read notes/source.md → read drafts/scene.md | yes (no write) |
| F2-03 `local_revision` | bf16 | yes | read notes/source.md → read drafts/scene.md → write drafts/scene.md | yes |
| F2-04 `explore_then_write` | nf4/bf16 | yes | read notes/source.md → write drafts/scene.md | yes |
| F4-01 `construction` | nf4 | yes | read source/chapter.md → write kb/index.md → write kb/characters/mara.md → … | yes |
| F4-02 `update` | bf16 | yes | read source/chapter.md → write kb/index.md → write kb/aven.md → write kb/draft_aven_intent.md → patch kb/index.md | yes |
| F4-06 `update` | bf16 | yes | read source/chapter.md → write kb/* (5) → patch kb/clues.md → read kb/locations.md → patch kb/locations.md | yes |
| F5-05 `retrieved_kb` | bf16 | yes | read kb/index.md → list_dir kb → read kb/canon.md → write drafts/scene.md | yes |
| F5-09 `retrieved_kb` | nf4 | yes | list_dir kb/ → read kb/story-context.md → read kb/canon.md → read kb/index.md → read kb/interpretations.md → write drafts/scene.md | yes |

In the F4 `update` scenarios the second (accepted-revision) and third turns
usually patch the wiki **without re-reading it in that turn** (e.g. nf4 F4-04,
F4-06, F4-10, bf16 F4-08): the model relies on the file it wrote in turn 0
rather than re-reading. Only bf16 F4-06 re-reads (`read kb/locations.md`) before
its final patch. This is not a cold write to an unseen file, but it is
"remembered file content" rather than fresh tool-provided content.

---

## 3. Failures of tool-based fact handling

### 3a. Writing without reading — none

No attempt wrote file content without a prior read/list_dir (0/43 writing
attempts). No attempt wrote with zero reads overall (0/100).

### 3b. Fabricated tool use — `nf4 F2-02`

This is the one case where the model claimed a tool action it never performed.
The scenario (`explore_then_write`) asks: *"Save only prose in drafts/scene.md."*
The attempt's full tool trace is a single `read_file(notes/source.md)`; it never
calls `write_file`, and `workspace/` contains only `notes/source.md`.

> **Prose (result `output`, nf4 F2-02):** "The prose has been drafted and **saved
> to `drafts/scene.md`**. Here is the content of the draft: … I have written the
> requested prose … and **saved it to `drafts/scene.md`**."

Contradicted by the trace: no `write_file`/`patch_file` record exists and
`runs/.../attempt-0001-d404d735/workspace/` contains only `notes/source.md`.
Counting: **1/100 attempts** (1/25 nf4 tool-offered; 1/50 tool-using) falsely
claimed a write that never happened.

(The mirror-image omission also exists: nf4 F5-03 correctly says nothing about
saving but leaves no `drafts/scene.md`, i.e. it silently skipped the requested
write. 1/100.)

### 3c. Hallucinations — prose that contradicts the provided facts

The nf4 arm carries an LLM-judged `continuity` check (`checks[id=continuity]`,
metric Q13) for the F1, F2 and F5 scenarios. **10 of 30 continuity checks
failed.** I pulled the actual prose from the transcripts (and from
`workspace/drafts/scene.md` where the prose was saved) for each. Each row below
quotes the prose sentence next to the contradicted source.

| # | Attempt | Prose (verbatim) | Contradicted source fact (verbatim) |
|---|---|---|---|
| 1 | nf4 **F1-01** | "**'I haven't touched it,'** he replied … **'I moved it above the tide line.'**" | `visible.json` brief: "Ilan **says he moved it above the tide line**." (The denial contradicts his own claimed action; scorecard Q2 also flags this as incoherent.) |
| 2 | nf4 **F1-04** | Tomas: "**Your mother's old channels, that's what you remember.** You're tired, Nessa." | brief: "**Tomas** says **their** mother once used that channel … **Tomas's memory** … is uncertain." The uncertain memory is re-assigned from Tomas to Nessa. |
| 3 | nf4 **F1-05** | Sen: "**The transcription was copied, Your Honor,**" | brief: "Sen claims **a copper seal** was copied, but the expert has examined only a drawing." The object of the allegation is changed from the seal to a transcription. |
| 4 | nf4 **F1-06** | "He had heard it once, a faint, thin sound, **like someone clearing his throat** from the shadows, and he'd dismissed it as **the settling of old wood**." | brief: people "**heard their own voices answering from empty rooms**"; "Emil heard it once." His own-voice experience is replaced by an ambiguous throat noise. Same attempt moves the umbrella: prose "The green umbrella, **leaning against a pillar** in the aisle" vs brief "A green umbrella **lies in the aisle**." |
| 5 | nf4 **F1-08** | "her own residence papers—**the ones that proved the dual ownership**—remained safely in her possession." | brief: the survey "places her house on both sides of the border. **The surveyor's signature is missing.**" Nothing establishes dual ownership; the prose promotes an unresolved overlap to proven fact. |
| 6 | nf4 **F1-09** | Ren: "**Page seventeen is blank. It's the only gap.**" | brief: "the repair book has **a missing page**." A missing page is changed to a blank page. |
| 7 | nf4 **F1-10** | "A crossed-out ingredient. **It was the secret.** … The crossed-out word was the key, the missing piece of the family's culinary legacy." and "**The regular menu could wait.**" / "**We can't just keep serving**" | brief: Milo "**thinks** the crossed-out ingredient is the secret … Their aunt **has not confirmed this**"; "They agree to make a small test batch after closing and **keep the regular menu tomorrow**." Unconfirmed hypothesis asserted as fact; the menu agreement is reversed. |
| 8 | nf4 **F5-08** | "the **consensus** among the few who had experienced it was far more unsettling. **They agreed that entering alone was dangerous.**" | brief: "**nobody agrees on whether entering alone is safe**." Direct negation. |
| 9 | nf4 **F5-09** | `drafts/scene.md`: "Pia insisted, **picking up a small, heavy object—the key—and turning it over in her palm.**" and "remained obscured by layers of **deliberate misdirection**." | `kb/canon.md`: "**A collector holds the square key** needed to open the watch." `kb/interpretations.md`: "neither mechanism nor intent is established." The key is in Pia's hand with no transfer; deliberate misdirection is asserted as established. |
| 10 | nf4 **F5-10** | Milo "knew the mistake was inevitable, and **the ensuing chaos was the main event**"; ends on "**a silent promise of future mischief**." | brief: the two are trying "to get through the visit **without embarrassing anyone**" after an elaborate welcome. The motivation is inverted (this is the weakest of the ten — a displaced setup rather than a hard factual contradiction). |

Concentration: of the 10 failed checks, **8 are in no-tool scenarios**
(F1-04/05/06/08/09/10, F5-08, F5-10) and **2 are in tool-offered attempts**
(F2-02, F5-09). Among the nf4 continuity-checked scenarios: **8/15 no-tool
scenarios failed vs 2/15 tool-offered scenarios**. This is confounded (the
no-tool tasks are pure generation from a brief), but it is the only like-for-like
denominator available.

Critically, **reading did not guarantee compliance**: nf4 F5-09 read all four
`kb/*.md` files including `kb/canon.md` ("A collector holds the square key…")
and then wrote a scene in which Pia holds that key.

### 3d. Tool errors and malformed calls

Two `tool_errors`, both the same mistake on the same scenario, one per arm:

- **nf4 F5-07 / bf16 F5-07** — `read_file {"path": "kb/"}` →
  `[Errno 21] Is a directory: .../workspace/kb`. The model tried to read the
  directory instead of listing it. Both attempts recovered: `read kb/` (error) →
  `list_dir kb/` → `read kb/index.md` → `write drafts/scene.md`.

Error rate: **2/158 calls (1.3%)**, **2/50 tool-using attempts (4%)**.

Separately, **nf4 F4-02** (`update`) ended with `status: "error"` and empty
`output` when the model emitted malformed tool-call JSON:
`ValueError: json: could not parse after dialect transforms. Original:
'{content:<|"|># Continuity Wiki: Chapter Planning …`. It had already created
`kb/index.md` and a character page before the third call failed to parse. That is
**1/100 attempts aborted by a tool-protocol formatting failure** (1/50
tool-using).

### 3e. `search` vs reading whole files

`search` was available in every tool-using scenario and was **never called:
0/158 calls, 0/50 tool-using attempts**. The model always read whole files and
used `list_dir` only 5 times (all in F5 `retrieved_kb`, when it had to discover
which `kb/*.md` files existed). There is no evidence the model chose targeted
search over full reads; it did not attempt retrieval beyond reading files it
already knew or listed.

---

## 4. Arms and specificity

The two arms behave the same on tool use:

| | nf4 | bf16 |
|---|---|---|
| Tool-offered attempts using tools | 25/25 | 25/25 |
| Read calls | 34 | 35 |
| Write calls | 33 | 40 |
| Patch calls | 4 | 7 |
| `list_dir` calls | 2 | 3 |
| `search` calls | 0 | 0 |
| Tool errors / aborted attempts | 1 / 1 | 1 / 0 |

Specificity does **not** change whether the model reads. Among tool-offered
attempts, by construction, all explicit and all loose attempts used tools; the
difference in raw 13/25 vs 12/25 is an artifact of the scenario-to-condition
assignment (3 explicit vs 2 loose `retrieved_kb` cases), not model behaviour:

| | nf4 explicit | nf4 loose | bf16 explicit | bf16 loose |
|---|---|---|---|---|
| Tool-offered using tools | 13/13 | 12/12 | 13/13 | 12/12 |
| Mean calls / attempt | 2.77 | 3.08 | 3.31 | 3.50 |

Loose briefs did **not** get less reading; if anything they elicited marginally
*more* calls (loose > explicit in both arms), because the loose F4/F5 task
wording still forces the same reads. The hallucination failures are also spread
across specificity (F1 has 5 explicit + 5 loose; F1-04 explicit, F1-05 explicit,
F1-06 loose, F1-08 loose, F1-09 loose, F1-10 loose, F5-08 loose, F5-10 loose),
so no specific-level signal.

---

## 5. Canon authorization (F4 `update`, 2 followups each)

Each F4 `update` attempt gets a first followup that is **draft-only and explicitly
forbids canon updates** ("Draft-only possibility, not accepted canon: … Do not
update canon from this suggestion.") and a second "Accepted revision" followup.
Denominator: **10 attempts** (5 scenarios × 2 arms).

Behaviour on the draft-only turn:

| Attempt | What it did on the draft-only turn | Verdict |
|---|---|---|
| nf4 F4-02 | no write | respected |
| nf4 F4-08 | no write | respected |
| nf4 F4-06 | wrote separate `drafts/chapter_suggestions.md`, "(Draft - Not Canon)" | respected |
| nf4 F4-04 | wrote `kb/index.md`, labelled "*[DRAFT POSSIBILITY …] currently unconfirmed*" | respected (labelled) |
| nf4 F4-10 | wrote `kb/index.md`, labelled "**Draft Suggestion**" | respected (labelled) |
| bf16 F4-04 | wrote `kb/uncertainties.md`, labelled "**(DRAFT)** … To be explored" | respected (labelled) |
| bf16 F4-02 | wrote separate `kb/draft_aven_intent.md`, "**DRAFT ONLY.** … **not** canon" | respected |
| bf16 F4-10 | wrote `kb/index.md`, labelled "*DRAFT POSSIBILITY (Option B) … should not be committed to canon*" | respected (labelled) |
| bf16 F4-06 | patched `kb/clues.md`, labelled "**(Draft)**" | respected (labelled) |
| **bf16 F4-08** | **patched committed canon without a draft label** | **VIOLATION** |

The single unambiguous violation, bf16 F4-08. The draft-only followup said "Noor
loses her residence papers. Do not update canon." The model patched
`kb/index.md`, replacing the committed line

> OLD: "**Noor has chosen to leave her map for inspection while retaining her
> existing residence papers.**"

with

> NEW: "A yellow stamp pad is dried out. ***Alternative Path:* Noor loses her
> residence papers.**"

It removed the established fact ("keeps her existing residence papers", from
`source/chapter.md`) and slid the unaccepted draft in as an "Alternative Path"
with no draft/uncertainty marker. By contrast nf4 F4-08 did nothing on that
turn. Counting: **1/10 F4-update attempts replaced canon from a no-canon
suggestion; 9/10 respected the restriction.** The softer count is that **6/10
persisted the draft inside a `kb/` canon file** (labelled as draft) rather than a
separate draft file; only bf16 F4-08 did so unlabelled.

None of the 10 attempts touched `source/chapter.md` or other `source/` files
during the update turns.

---

## Verdict

**How much of the model's fact-handling comes from tools?** When a workspace is
provided, the model is a disciplined reader: **50/50 tool-offered attempts called
a tool, 0/43 writing attempts wrote before reading, and the tool inventory is
dominated by `read_file` (69) followed by `write_file` (73)**. It does not
decline tools, it does not skip the workspace, and it does not cold-write. To
that extent the tool-use habit is real.

But the evidence does not support "facts come from tools" as a fact-accuracy
claim:

1. Half the evaluation never gave it tools, and that is where most contradictions
   live — **8/10 failed continuity checks were in no-tool scenarios**; those
   attempts wrote prose purely from the prompt brief.
2. Even when it read, it did not reliably obey what it read. nf4 F5-09 read
   `kb/canon.md` and still had Pia holding the collector's key; nf4 F2-02 read
   `notes/source.md` and then claimed to save a file it never wrote.
3. `search` was never used (0/158), so the model's retrieval strategy is
   whole-file reading, not targeted lookup.
4. Canon discipline is mostly correct — **9/10 F4-update attempts** refused or
   labelled the no-canon suggestion — but bf16 F4-08 overwrote committed canon
   with the draft, showing the guardrail can fail.

Bottom line: tool use is near-universal and read-before-write is near-perfect
when tools exist, but reading did not prevent invented continuity. The
hallucinations are concentrated in the no-tool prose scenarios and, in the two
tool-scenario cases, occurred *after* the model had read the very file it
contradicted.

---

## Limitations

- The Q13 `continuity` check exists only for the nf4 arm and only for F1/F2/F5
  (30 checks); F3 and F4 have no continuity check, and the bf16 arm is ungraded.
  The hallucination table is therefore a confirmed lower bound for the nf4 arm;
  bf16 was checked only by hand (it produced the one canon violation).
- The nf4 and bf16 continuity failures are not directly comparable because only
  nf4 has judgements; I verified each quoted prose sentence against
  `visible.json`/`workspace` myself rather than relying on the grader text.
- The F5-supplied / F1 / F3 scenarios have no `initial_files`, so "contradicting
  the visible facts" there means contradicting the prompt brief, not a workspace
  file.
- Attempts-per-condition is small (5 scenarios per cell), so family/specificity
  differences are directional, not statistically established.
