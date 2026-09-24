# The model reads files, but reading does not ensure factual accuracy

This report asks whether the model uses project files before writing and whether it obeys what it reads. Every attempt given tools used them, and every attempt that wrote a file first read or inspected the workspace; nevertheless, the model sometimes contradicted supplied facts, falsely claimed a save, or replaced established story state with an unaccepted suggestion.

The evidence covers 100 attempts by Google's `google/gemma-4-E2B-it`: 50 development scenarios in each of two weight formats. `nf4` uses compressed 4-bit weights; `bf16` is bfloat16, the full-precision comparison. The nf4 run, `runs/custom50-e2b-it-2026-09-14`, has ASTRA scorecards, meaning language-model judgements saved at `astra-graded/<ID>/scorecard.json`, where `<ID>` denotes a scenario identifier. The bf16 run, `runs/custom50-e2b-it-2026-09-21-bf16`, has no such grading.

There are five task families: F1 asks for a scene in the reply; F2 for revision or drafting saved to a file; F3 for alternative directions or feedback; F4 for building or updating a Markdown knowledge base (KB); and F5 for using project facts. In F5, `retrieved_kb` means facts must be read from files; `supplied_kb` means the facts are included in the brief. An identifier such as F5-09 names case 09 in family F5. “Canon” means established story facts, as distinct from a character's belief or a proposed revision. Explicit briefs specify more choices; loose briefs leave more to the model.

The analysis reads ordered tool records in `trace.jsonl`; tool counts, errors, turns, status and output in `result.json`; offered tools, starting files, follow-ups and briefs in `visible.json`; and workspace files. It changes no source, tests or run files.

## Half the attempts could not use tools because none were offered

All 10 F1 cases, all 10 F3 cases and the five F5 supplied-fact cases in each run have `tools: []` and no `initial_files`. Only the 10 F2 cases, 10 F4 cases and five F5 retrieval cases offer tools and a workspace. Therefore the raw split—25 attempts use tools and 25 do not in each format—comes from the test design, not from the model refusing to read.

Every tool-enabled attempt makes at least one call. The table separates access from actual use so the denominator is visible.

| Measure | nf4 | bf16 | Both |
|---|---|---|---|
| Attempts | 50 | 50 | 100 |
| Attempts where tools were offered | 25 | 25 | 50 |
| Attempts that called ≥1 tool | 25 | 25 | 50 |
| Tool-offered attempts that called ≥1 tool | 25/25 | 25/25 | 50/50 |
| Attempts with no tools offered that called a tool | 0/25 | 0/25 | 0/50 |
| Total tool calls | 73 | 85 | 158 |
| Calls per tool-using attempt (mean / max) | 2.92 / 6 | 3.40 / 9 | — |
| `result.json` `tool_calls` total | 73 | 85 | 158 |
| `result.json` `tool_errors` total | 1 | 1 | 2 |
| Attempt status `completed` / `error` | 49 / 1 | 50 / 0 | 99 / 1 |

The tool names describe five operations: `read_file` returns a file's contents; `write_file` writes a file; `patch_file` changes existing text; `list_dir` lists a directory; and `search` finds matching text. Reading and writing account for most calls; search is never used.

| Tool | nf4 | bf16 | Both | Share |
|---|---|---|---|---|
| `read_file` | 34 | 35 | 69 | 43.7% |
| `write_file` | 33 | 40 | 73 | 46.2% |
| `patch_file` | 4 | 7 | 11 | 7.0% |
| `list_dir` | 2 | 3 | 5 | 3.2% |
| `search` | 0 | 0 | 0 | 0% |

Within F2, `local_revision` means changing an existing draft and `explore_then_write` means reading supporting material before creating prose. F3's `alternatives` and `feedback` are requests for directions or responses to feedback. F4's `construction` creates a wiki and `update` revises it after follow-ups. Tool availability is identical in both formats:

| Family / condition | tools offered | attempts that used tools (per weight format) |
|---|---|---|
| F1 scene/continuation/dialogue/reflection/action | no | 0/10 |
| F2 `local_revision` / `explore_then_write` | yes | 10/10 |
| F3 `alternatives` / `feedback` | no | 0/10 |
| F4 `construction` / `update` | yes | 10/10 |
| F5 `retrieved_kb` | yes | 5/5 |
| F5 `supplied_kb` | no | 0/5 |

## No attempt writes before inspecting the workspace

The analysis reconstructs call order and asks whether a `write_file` or `patch_file` precedes any `read_file`, `search` or `list_dir`, and whether an existing target file was read before modification. A “cold write” here means writing before that first inspection; it does not mean every necessary fact has been read.

There are zero cold writes among 43 attempts that write anything. Of 84 write or patch operations, 61 create files that did not previously exist, so those files could not be read first. Of the 23 operations on existing files, 10 follow a read of that exact path in the same attempt. The other 13 modify files the model itself created earlier in the attempt. No writing attempt has zero reads overall; the source also reports that as 0/100 attempts writing with no reads.

The following examples retain the original order without treating a read-only attempt as evidence of successful writing:

- F2-01 in both formats reads `notes/source.md`, then `drafts/scene.md`, and does not write.
- Bf16 F2-03 reads those same two files, then writes `drafts/scene.md`.
- F2-04 in both formats reads `notes/source.md`, then writes `drafts/scene.md`.
- Nf4 F4-01 reads `source/chapter.md`, then writes `kb/index.md`, `kb/characters/mara.md` and further pages.
- Bf16 F4-02 reads `source/chapter.md`, writes `kb/index.md`, `kb/aven.md` and `kb/draft_aven_intent.md`, then patches `kb/index.md`.
- Bf16 F4-06 reads `source/chapter.md`, writes five knowledge-base files, patches `kb/clues.md`, reads `kb/locations.md`, then patches that file.
- Bf16 F5-05 reads `kb/index.md`, lists `kb`, reads `kb/canon.md`, then writes `drafts/scene.md`.
- Nf4 F5-09 lists `kb/`, reads `kb/story-context.md`, `kb/canon.md`, `kb/index.md` and `kb/interpretations.md`, then writes `drafts/scene.md`.

Later F4 update turns usually patch the wiki without re-reading it on that turn. Examples include nf4 F4-04, F4-06 and F4-10, and bf16 F4-08. They rely on text the model wrote in the initial turn, numbered turn 0 in the trace. Only bf16 F4-06 re-reads `kb/locations.md` before its final patch. This explains why read-before-write can be satisfied while later changes still rely on remembered content rather than a fresh observation. The source's description of which numbered follow-up is accepted conflicts with its later draft-then-accepted sequence; the ordering above does not resolve that wording error.

## One attempt claims a save that never happens

Nf4 F2-02 is asked to “Save only prose in drafts/scene.md.” Its entire tool trace is a single read of `notes/source.md`. It then says:

> “The prose has been drafted and saved to `drafts/scene.md`. Here is the content of the draft: … I have written the requested prose … and saved it to `drafts/scene.md`.”

There is no `write_file` or `patch_file` record, and `runs/.../attempt-0001-d404d735/workspace/` contains only `notes/source.md`. This is a fabricated action report in 1/100 attempts, 1/25 nf4 tool-enabled attempts, or 1/50 tool-using attempts. The mechanism is straightforward: a chat statement reports success without the file operation that would make it true.

Nf4 F5-03 makes the opposite reporting omission: it does not claim a save, but still fails to create the requested `drafts/scene.md`. That is another 1/100 attempts silently skipping the write.

## Reading does not stop the model from changing facts

The nf4 prose families F1, F2 and F5 receive the `continuity` check, internally Q13. It asks whether the generated story preserves supplied facts and uncertainty. Ten of 30 checks fail. There is no corresponding bf16 grading.

Q2, mentioned in the examples, is the separate prose-quality score. The source pairs ten quoted examples with the facts they contradict. They show several mechanisms: assigning a memory to the wrong person, replacing one object with another, converting suspicion into proof, and changing possession or motivation without depicting the transition. The final row is explicitly weaker evidence than a hard factual contradiction.

There is an important bookkeeping conflict: this example table includes F1-01 but omits F2-02, while the source's failed-check breakdown counts F2-02 and excludes F1-01. The table is preserved as ten examples, not silently relabelled as the exact set of ten failed checks.

| # | Attempt | Prose (verbatim) | Contradicted source fact (verbatim) |
|---|---|---|---|
| 1 | nf4 F1-01 | "'I haven't touched it,' he replied … 'I moved it above the tide line.'" | `visible.json` brief: "Ilan says he moved it above the tide line." (The denial contradicts his own claimed action; scorecard Q2 also flags this as incoherent.) |
| 2 | nf4 F1-04 | Tomas: "Your mother's old channels, that's what you remember. You're tired, Nessa." | brief: "Tomas says their mother once used that channel … Tomas's memory … is uncertain." The uncertain memory is re-assigned from Tomas to Nessa. |
| 3 | nf4 F1-05 | Sen: "The transcription was copied, Your Honor," | brief: "Sen claims a copper seal was copied, but the expert has examined only a drawing." The object of the allegation is changed from the seal to a transcription. |
| 4 | nf4 F1-06 | "He had heard it once, a faint, thin sound, like someone clearing his throat from the shadows, and he'd dismissed it as the settling of old wood." | brief: people "heard their own voices answering from empty rooms"; "Emil heard it once." His own-voice experience is replaced by an ambiguous throat noise. Same attempt moves the umbrella: prose "The green umbrella, leaning against a pillar in the aisle" vs brief "A green umbrella lies in the aisle." |
| 5 | nf4 F1-08 | "her own residence papers—the ones that proved the dual ownership—remained safely in her possession." | brief: the survey "places her house on both sides of the border. The surveyor's signature is missing." Nothing establishes dual ownership; the prose promotes an unresolved overlap to proven fact. |
| 6 | nf4 F1-09 | Ren: "Page seventeen is blank. It's the only gap." | brief: "the repair book has a missing page." A missing page is changed to a blank page. |
| 7 | nf4 F1-10 | "A crossed-out ingredient. It was the secret. … The crossed-out word was the key, the missing piece of the family's culinary legacy." and "The regular menu could wait." / "We can't just keep serving" | brief: Milo "thinks the crossed-out ingredient is the secret … Their aunt has not confirmed this"; "They agree to make a small test batch after closing and keep the regular menu tomorrow." Unconfirmed hypothesis asserted as fact; the menu agreement is reversed. |
| 8 | nf4 F5-08 | "the consensus among the few who had experienced it was far more unsettling. They agreed that entering alone was dangerous." | brief: "nobody agrees on whether entering alone is safe." Direct negation. |
| 9 | nf4 F5-09 | `drafts/scene.md`: "Pia insisted, picking up a small, heavy object—the key—and turning it over in her palm." and "remained obscured by layers of deliberate misdirection." | `kb/canon.md`: "A collector holds the square key needed to open the watch." `kb/interpretations.md`: "neither mechanism nor intent is established." The key is in Pia's hand with no transfer; deliberate misdirection is asserted as established. |
| 10 | nf4 F5-10 | Milo "knew the mistake was inevitable, and the ensuing chaos was the main event"; ends on "a silent promise of future mischief." | brief: the two are trying "to get through the visit without embarrassing anyone" after an elaborate welcome. The motivation is inverted (this is the weakest of the ten — a displaced setup rather than a hard factual contradiction). |

The reported failed-check breakdown is eight no-tool cases (F1-04, F1-05, F1-06, F1-08, F1-09, F1-10, F5-08 and F5-10) and two tool-enabled cases (F2-02 and F5-09). Among the 30 checked nf4 attempts, that is 8/15 without tools versus 2/15 with tools. The comparison is confounded: the no-tool tasks are pure generation from a brief, so the conditions differ beyond tool access. It is the only like-for-like check denominator the source offers, not a causal estimate of the benefit of reading.

F5-09 supplies direct evidence that exposure is insufficient. It reads all four `kb/*.md` files, including “A collector holds the square key…” in `kb/canon.md`, then writes Pia holding the key without a transfer. The source concludes that the two tool-enabled failure cases occur after the relevant file was read; the table mismatch leaves F2-02's continuity example undocumented here even though its false-save behavior is documented above.

## Tool errors are rare and usually recoverable

Both formats make the same tool error in F5-07: `read_file {"path": "kb/"}` tries to read a directory and receives `[Errno 21] Is a directory: .../workspace/kb`. Both recover by listing `kb/`, reading `kb/index.md` and writing `drafts/scene.md`. The two errors are 2/158 calls, or 1.3%, and 2/50 tool-using attempts, or 4%.

A separate failure aborts nf4 F4-02. The result has `status: "error"` and empty output after `ValueError: json: could not parse after dialect transforms. Original: '{content:<|"|># Continuity Wiki: Chapter Planning …`. The model has already created `kb/index.md` and a character page when another call fails to parse. The source calls this malformed tool-call JSON: 1/100 attempts, or 1/50 tool-using attempts, aborted by protocol formatting. The failure report instead emphasizes the parser's handling of embedded delimiters. Both establish a parsing failure; neither account alone resolves responsibility between generation and parser handling. The source also calls this the “third call,” which needs reconciliation with the read and two successful writes it describes.

## Retrieval means whole-file reading in this sample

`search` is offered in every tool-enabled case and never called: 0/158 calls and 0/50 tool-using attempts. `list_dir` is used five times, all in F5 retrieval cases where the model must discover which knowledge-base files exist. The observed strategy is to read files it already knows about or finds in a listing. There is no evidence of targeted text search. That limits what this study can say about retrieval over larger collections.

## Weight format and brief detail barely change the decision to read

Both formats use tools in 25/25 enabled attempts. Nf4 records 34 reads, 33 writes, four patches, two listings and no searches; bf16 records 35 reads, 40 writes, seven patches, three listings and no searches. Nf4 has one tool error and one aborted attempt; bf16 has one tool error and no aborted attempt.

All explicit and loose attempts with tools use them. The raw 13/25 explicit versus 12/25 loose access counts arise from scenario assignment: three explicit and two loose F5 retrieval cases. They are not model preferences. Among enabled attempts, the call counts are slightly higher for loose briefs:

| | nf4 explicit | nf4 loose | bf16 explicit | bf16 loose |
|---|---|---|---|---|
| Tool-offered using tools | 13/13 | 12/12 | 13/13 | 12/12 |
| Mean calls / attempt | 2.77 | 3.08 | 3.31 | 3.50 |

The source attributes this to loose F4/F5 wording still requiring reads. It reports no specific-level pattern in continuity failures: F1 contains five explicit and five loose cases; F1-04 and F1-05 are explicit, F1-06/F1-08/F1-09/F1-10 loose, and F5-08/F5-10 loose. Those observations do not establish equal failure rates by brief detail, especially given the small and differing task mixes.

## One draft-only suggestion replaces established canon

Each F4 update case gets two follow-ups: first a draft-only possibility explicitly forbidding canon changes, then an accepted revision. Five scenarios in each format give 10 attempts. The source distinguishes writing a labelled proposal from replacing an established fact with it.

The draft-only-turn actions are:

- Nf4 F4-02 and F4-08 make no successful write. The first case also has the aborted call discussed above, so this is not evidence that it deliberately chose inaction.
- Nf4 F4-06 creates `drafts/chapter_suggestions.md` labelled “(Draft - Not Canon)”.
- Nf4 F4-04 writes `kb/index.md` with “[DRAFT POSSIBILITY …] currently unconfirmed”.
- Nf4 F4-10 writes `kb/index.md` with “Draft Suggestion”.
- Bf16 F4-04 writes `kb/uncertainties.md` with “(DRAFT) … To be explored”.
- Bf16 F4-02 creates separate `kb/draft_aven_intent.md` labelled “DRAFT ONLY.” and “not” canon.
- Bf16 F4-10 writes `kb/index.md` with “DRAFT POSSIBILITY (Option B) … should not be committed to canon”.
- Bf16 F4-06 patches `kb/clues.md` with “(Draft)”.
- Bf16 F4-08 patches committed canon without a draft label; the source counts this as the violation.

In that last case, the instruction says “Noor loses her residence papers. Do not update canon.” The model replaces this established line in `kb/index.md`:

> “Noor has chosen to leave her map for inspection while retaining her existing residence papers.”

with:

> “A yellow stamp pad is dried out. Alternative Path: Noor loses her residence papers.”

“Alternative Path” suggests another possibility, but the patch removes the established fact rather than preserving it alongside a clearly draft-only note. That deletion is why the source classifies this as an unauthorized canon replacement. Nf4 F4-08 leaves the file alone on that turn.

The reported result is 1/10 attempts replacing canon from an unaccepted suggestion and 9/10 respecting the restriction. The source also reports a softer count of 6/10 persisting the proposal inside a canon knowledge-base file, with only bf16 F4-08 unlabelled; its wording elsewhere describes all six as labelled, an inconsistency retained in the audit. Separate draft pages are treated differently in this count. None of the 10 attempts changes `source/chapter.md` or another `source/` file during update turns.

The specificity report claims all nine completed update attempts preserve the distinction. That contradicts this bf16 F4-08 example. The evidence cannot be reconciled by excluding the crash, because bf16 F4-08 completes; the original judgements need review.

## Limits of the evidence

Continuity grading covers only 30 nf4 F1/F2/F5 cases. F3 and F4 lack that check, and bf16 is ungraded apart from manual inspection, which finds the canon replacement. The source calls its contradiction examples a confirmed lower bound for nf4; the mismatch between its example list and failed-check list must be resolved before treating them as a precise inventory.

The author checked quoted prose against `visible.json` and workspace files rather than relying only on judge text. In F1, F3 and supplied-fact F5 cases there are no initial files: contradiction means disagreement with the brief, not with a file. There are only five scenarios per family/condition cell, so differences by family or specificity are directional, not statistically established.

The original analysis is [tool-use.md](../../transcript-analysis/tool-use.md).
