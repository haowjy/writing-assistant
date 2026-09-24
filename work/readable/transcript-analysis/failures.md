# What the model fails at, and what our tests get wrong

This report separates failures that require improving the model from failures caused by our evaluation system. The model exceeds length limits, changes story facts, misses required knowledge and sometimes delivers prose to the wrong place; some tests also penalize permitted omissions or pass unchanged files, so training against every reported failure would teach the wrong behavior.

The evidence covers 100 attempts by Google's `google/gemma-4-E2B-it`, abbreviated E2B: the same 50-case test set, called `custom50`, run in two weight formats. The `nf4` run uses compressed 4-bit weights and has language-model judgements; the `bf16` run uses bfloat16 weights, the full-precision comparison, and has no semantic judgements. A semantic check requires interpreting meaning. A deterministic check tests something directly, such as file existence or word count. Required checks must pass; optional checks need not.

The five task families are F1, a scene in the reply; F2, revision or drafting saved to a file; F3, alternative directions; F4, building or updating a Markdown knowledge base (KB); and F5, using project facts supplied in a brief or retrieved from files. In F2-01, for example, F2 names the family and 01 names the case. Explicit briefs specify more decisions, such as length and viewpoint; loose briefs leave more open. A selected `scene` is the prose extracted from the designated reply or file for scoring, not necessarily everything the model wrote.

The source reads `card.json`, ASTRA scorecards (results from the language-model grading pipeline), `result.json`, messages, tool traces and workspace files. Its runs are `runs/custom50-e2b-it-2026-09-14` for nf4 and `runs/custom50-e2b-it-2026-09-21-bf16` for bf16. In a scorecard path, `<ID>` denotes the scenario identifier. All 50 nf4 attempts have `astra-graded/<ID>/scorecard.json` files, including the one that crashed. No source, tests or run files were changed by the analysis.

## Most attempts produce something usable, but completion hides missing prose

Ninety-three of 100 attempts produced a usable deliverable. This includes the single crashed attempt, F4-02 in nf4, because it wrote `kb/index.md` and one character page before crashing. Usability here is therefore weaker than successful completion.

| Weight format | completed | error | total |
|---|---|---|---|
| nf4 | 49 | 1 (F4-02) | 50 |
| bf16 | 50 | 0 | 50 |
| all | 99 | 1 | 100 |

Seven attempts have no extractable scene: nf4 F2-01, F2-02, F2-03, F2-05 and F5-03; bf16 F2-01 and F2-05. The coverage table shows that the missing outputs all belong to prose tasks.

| Family, per weight format | Attempts | Deliverable | nf4 | bf16 |
|---|---|---|---|---|
| F1, F2, F5 (prose) | 30 | selected `scene` extractable | 25/30 | 28/30 |
| F4 (wiki) | 10 | `kb/` pages present | 10/10 | 10/10 |
| F3 (directions) | 10 | non-empty answer | 10/10 | 10/10 |

The bf16 run has 115 pending semantic checks: continuity 30, viewpoint/style 15, genre fit 30, consequential state 10, qualified belief 10, optional context 10, accepted update 5 and three directions 5. It has no ASTRA scorecards. Comparisons between formats below are limited to deterministic checks; an ungraded semantic check is not a pass.

## What each check measures

The internal metric labels are Q1, instruction adherence; Q3, task completion; Q8, coverage of important information; Q11, correctness of updates; Q12, retrieval success; and Q13, continuity with established story facts. Point of view (POV) means whose experience the narration follows. “Close third” uses third-person narration centered on one character's perspective.

In the nf4 results, the largest scored problems concern length, requested style and continuity. The table retains unscored cases separately because missing selected prose cannot supply a length measurement. “Required” identifies a check that must pass. `allowed_changes` is the rule defining which paths may change.

| check | asserts | total | pass | fail | fail families / specificity |
|---|---|---|---|---|---|
| `word-budget` | selected prose 120–220 words (Q1, not required) | 15 | 1 | 9 (+5 unscored) | F1 explicit, F2 explicit, F5 explicit |
| `pov-style` | close third on the named character and requested style, combined (Q1) | 15 | 6 | 9 | F1/F2/F5 explicit |
| `continuity` | preserve required facts, keep belief uncertain, no forbidden assertion (Q13, required) | 30 | 20 | 10 | F1 (2 explicit + 4 loose), F2 (1 explicit), F5 (3 loose) |
| `wiki-links` | other KB pages reachable through links from `kb/index.md` (required) | 5 | 0 | 5 | F4 explicit only |
| `optional-context` | optional KB fact (Q8, weight 1) | 10 | 5 | 5 | F4 explicit 3, loose 2 |
| `consequential-state` | required current state in KB (Q8, required) | 10 | 6 | 4 | F4 explicit 3, loose 1 |
| `accepted-update` | incorporate accepted update + keep history, drop draft-only (Q11, required) | 5 | 2 | 3 | F4 explicit 2, loose 1 |
| `retrieve-state` | required fact exposed via read/search (Q12) | 5 | 3 | 2 | F5 explicit |
| `retrieve-belief` | uncertain belief exposed via read/search (Q12) | 5 | 3 | 2 | F5 explicit |
| `saved-scene` | selected scene file non-empty (Q3, required) | 10 | 9 | 1 | F2 explicit (F2-02) |
| `three-directions` | exactly three developed viable directions (Q1, required) | 5 | 4 | 1 | F3 explicit (F3-02) |
| `wiki-only` | only `kb/*.md` changed (allowed_changes, required) | 10 | 9 | 1 | F4 loose (F4-06) |
| `qualified-belief` | required uncertain belief in KB (Q8, required) | 10 | 9 | 1 | F4 explicit (F4-05) |
| `genre-fit` | fits the requested genre (Q1) | 30 | 30 | 0 | — |
| `preserve-ending` | retains the required final sentence | 3 | 3 | 0 | — |
| `edit-scope` | changes only the permitted material | 10 | 10 | 0 | — |
| `wiki-words` | keeps KB text within its word ceiling | 5 | 5 | 0 | — |
| `wiki-index` | provides the KB starting page | 10 | 10 | 0 | — |
| `preserve-kb` | leaves supplied KB files unchanged | 5 | 5 | 0 | — |

The source labels the nf4 retrieval failures as explicit cases. Retrieval tasks themselves include both explicit and loose briefs, and the bf16 failure list below includes loose F5-09. The rates and original nf4 labels are retained; interpreting differences by specificity requires checking the individual scenario assignments.

In bf16, length, cross-page linking and retrieval still fail deterministic tests. Semantic comparisons remain unavailable.

| check | pass | fail | unscored | pending | fail scenarios |
|---|---|---|---|---|---|
| `word-budget` | 6 | 7 | 2 | 0 | F1-03, F1-04, F2-03, F2-04, F5-01, F5-03, F5-05 |
| `wiki-links` | 0 | 5 | 0 | 0 | F4-01…F4-05 |
| `retrieve-belief` | 2 | 3 | 0 | 0 | F5-03, F5-05, F5-09 |
| `retrieve-state` | 3 | 2 | 0 | 0 | F5-03, F5-09 |
| `saved-scene`, `edit-scope`, `preserve-ending`, `preserve-kb`, `wiki-index`, `wiki-words`, `wiki-only` | all | 0 | 0 | 0 | — |
| semantic (`continuity`, `pov-style`, `genre-fit`, F4 semantic, `three-directions`) | — | — | — | 115 | not gradeable |

The selected prose counts are 25 present and five missing in nf4, compared with 28 present and two missing in bf16.

## The model exceeds explicit length limits

Of 15 nf4 length checks, nine fail, one passes and five are unscored. Of 15 bf16 checks, seven fail, six pass and two are unscored. Combined, 16 of 23 scored checks fail. Every scored failure is too long, never too short. This is an instruction-following failure: the scorer counts whitespace-separated words with `len(text.split())` and compares them with the stated 120–220 range (`src/writing_agent/scoring.py:185`; `development.py:500`).

The nf4 failing lengths are F1-02 266, F1-03 293, F1-04 262, F1-05 230, F2-04 281, F5-01 275, F5-02 254, F5-04 263 and F5-05 257. F1-01 passes at 216. The bf16 failing lengths are F1-03 226, F1-04 237, F2-03 233, F2-04 237, F5-01 240, F5-03 227 and F5-05 224. Other reports use different counts; this rewrite retains this report's counting method and values.

The 275-word nf4 F5-01 illustrates how extra description consumes the budget: “The air in the archive was thick, heavy with the scent of brine and decaying paper. Mara felt the cold seep not just through the fabric of her tunic, but into the bone, a constant, physical reminder of the encroaching tide.” The source describes a recurrent approximately 230–293-word output despite the cap. It proposes length-aware generation or reinforcement learning (training from rewards), or a shorter target. It identifies no contract error in the word-limit check.

## The model changes facts and turns guesses into certainty

Ten of 30 nf4 continuity checks fail: F1-04, F1-05, F1-06, F1-08, F1-09, F1-10, F2-02, F5-08, F5-09 and F5-10. This required Q13 check permits plausible new events, revelations, dialogue, viewpoint and endings. The failures concern overwriting the supplied starting state: treating a hypothesis as certain, assigning a fact to the wrong character, changing a detail or giving someone an object they did not have. Bf16 is ungraded.

In F1-05 the brief says “Sen alleges copying” of a copper seal, but the model writes “The transcription was copied, Your Honor.” The judge says this changes the allegation's object and therefore the evidentiary issue. In F5-08 the brief says “nobody agrees on whether entering alone is safe”; the model supplies “the consensus among the few who had experienced it” and “They agreed that entering alone was dangerous.” It invents an existing agreement rather than depicting a new event that changes people's minds.

The source classifies most cases as genuine failures to distinguish belief from established fact, and recommends training on that distinction and fact preservation. It regards two judgements as strict but defensible: F1-09 changes a missing page to a blank page, and F1-06 moves an umbrella a few feet. Those could be reviewed for strictness without changing the check generally.

## Requested style is missed, but the combined viewpoint/style check obscures why

Nine of 15 nf4 `pov-style` checks fail. The source says all nine judge rationales pass the viewpoint component and fail style: the model uses abstract, literary explanation where the brief requests spare, restrained or concrete prose. A combined result makes a style failure look like a failure of both requirements.

F1-01, requested to be restrained and concrete, says “The air between them thickened, tasting of salt and unsaid futures. … the red scarf on the chair seemed to mock the barrier.” The judge identifies abstract commentary and symbolic explanation. The mechanism is visible: the narration interprets the scene's meaning rather than staying with actions and observable details.

There is an unresolved contradiction in the source's all-nine claim. Its F5-02 example, requested to use spare prose, says “He had seen the watchers; he knew the risk was tangible … a flaw mirrored in the precariousness of their current situation.” The quoted judge says it “directly enters Aven's knowledge without framing it as Suri's inference” as well as departing from spare prose. The source also says the model sometimes slips viewpoint in F5-02. Both statements are preserved; nine style-only failures cannot be established from this account without reconciling that example.

The proposed responses remain distinct: train style control, and split `pov-style` into separate `pov` and `style` checks so each failure can be interpreted. Bf16 provides no semantic confirmation.

## The model creates wiki pages that the starting page cannot reach

All five applicable `wiki-links` checks fail in each format: 10/10 combined, covering F4-01 through F4-05. Only explicit F4 cases receive this check; loose F4-06 through F4-10 do not.

The model creates two to five pages but often links to headings within the current page instead of linking to another file. A reader starting at `kb/index.md` therefore cannot reach the other pages. In nf4 F4-01, four pages yield a valid-link fraction of 0.0 and reachability of 0.25: only one quarter of the pages is reachable. The orphaned pages are `kb/characters/ilan.md`, `kb/characters/mara.md` and `kb/plot_points/the_ledger.md`. One recorded link goes from `kb/characters/ilan.md` back to itself with anchor `plot_points-the-brass-ledger` and `valid: false`.

F4-03 has valid-link fraction 0.286, reachability 0.2 and four orphan pages. Its starting page uses `[Characters](#characters)` and `[Setting & Context](#setting--context)`, both links within the same page. Links to headings are legitimate Markdown, but they do not satisfy the request for a navigable wiki with at least one linked topic page. A link such as `[Mara](characters/mara.md)` would point to another file. The source recommends training cross-page links and optionally showing an example in the brief.

## The judge penalizes some omissions that the check permits

Five of 10 nf4 `optional-context` checks fail: F4-01, F4-03, F4-05, F4-07 and F4-09. F4-02, F4-04, F4-06, F4-08 and F4-10 pass. These checks have weight 1 and `required=False`. Their instruction says: “Assess whether recording this information would help a writer continue the chapter. Accept a defensible omission or alternative placement; do not treat this candidate fact as a mandatory checklist item.” (`development.py:383`).

The failed rationales nevertheless emphasize absence. F4-01 says “The optional fact that Ilan repairs flooded shelves is absent.” F4-03 says “The optional first-stop context is not retained.” F4-09 says “This omission materially weakens the continuation reference … Selectivity is allowed, but retaining speculative symbolic guidance instead does not adequately compensate for losing this actionable state.”

The source interprets all five as an evaluation-contract problem: the model selected salient facts and omitted lower-priority ones, as allowed, but was failed. It recommends either making the judge accept defensible omissions or declaring genuinely necessary facts required and explicitly asking for them. Whether every omission was defensible is not settled by the quotations alone—F4-09 supplies a reason for rejecting the selection. That limitation is recorded in the audit; the source's classification is retained here rather than silently corrected.

## Some prose goes to chat when it must go to a file

Seven of 60 prose-family attempts have no extractable scene: five of 30 nf4 and two of 30 bf16. Two mechanisms account for them.

In nf4 F2-01, F2-03 and F2-05, and bf16 F2-01 and F2-05, the model puts prose between `<prose>` and `</prose>` in its chat reply. The instruction requires those delimiters in the file. For example, nf4 F2-01 replies `<prose>Mara’s hand settled on the cold iron of the latch…` but leaves `drafts/scene.md` byte-identical to the initial `"Mara opened the door.\nThe harbor bell rang twice.\n"`. Extraction searches the file, finds no tags and records `missing_prose`; the word-budget check is unscored.

The other two nf4 cases create no scene file. F2-02 says “The prose has been drafted and saved to `drafts/scene.md`,” but the final workspace contains only `notes/source.md`; `saved-scene` correctly fails. F5-03 silently supplies prose in chat despite “Save prose in drafts/scene.md”.

These are model failures to execute the requested write, with a false claim of success in F2-02. They also reveal a scoring gap. An unchanged, already non-empty draft passes `saved-scene`, which checks only `bool(text.strip())` (`scoring.py:171`); preserving the ending and edit scope can also pass because nothing changed. Five nf4 attempts, or 10%, lose length/style signal through missing selected prose. The source's later summary groups all seven missing outputs under the unchanged-file issue, although two had no scene file and F2-02 explicitly failed `saved-scene`. That broader grouping should not be read as seven identical false passes.

The proposed fixes are to train file delivery; require a newly created or changed scene rather than mere non-emptiness; and either extract misplaced reply prose as a fallback or mark inapplicable structural checks `not_applicable` instead of passing them. A fallback could recover prose-quality evidence while the delivery failure remains separately recorded.

## Retrieval sometimes stops at the index

The `evidence_exposed` rule requires the specified fact string to occur in a `read_file` or `search` observation (`scoring.py:198`). In nf4, `retrieve-state` fails 2/5 and `retrieve-belief` fails 2/5. In bf16, they fail 2/5 and 3/5 respectively: combined state failures 4/10, belief failures 5/10.

In nf4 F5-03 the only call reads `kb/index.md`, returning 104 characters of links. The required state, “The captain alone knows the revised sailing time,” is in `kb/canon.md`; the belief, “Oren's creditor explanation is a guess…”, is in `kb/interpretations.md`. Neither file is opened. The model writes from the brief without retrieving those facts. In contrast, nf4 F5-09 reads `kb/story-context.md`, `kb/canon.md`, `kb/index.md` and `kb/interpretations.md` and passes retrieval.

This is a model skill gap: the available `list_dir`, `search` and `read_file` tools could expose the information, but the model stops before following the links. The proposed training target is to follow an index to the relevant files before answering; the source proposes no contract change.

## Knowledge-base summaries omit required state and history

In nf4, `consequential-state` fails 4/10 (F4-02, F4-03, F4-05, F4-07); `accepted-update` fails 3/5 (F4-02, F4-04, F4-06); and `qualified-belief` fails 1/10 (F4-05). These checks ask the knowledge base to retain actionable current state, incorporate accepted changes while preserving necessary history and excluding draft-only claims, and preserve uncertainty around a belief. Bf16 is ungraded.

F4-05 must retain that “the original seal remains in a locked evidence chest.” It records only “a physical object (a copper seal),” losing the location and secured state needed to continue the story. It also drops “the expert has examined only a drawing,” losing the reason the belief is uncertain. F4-04 loses the prior restriction “the antenna cannot be turned until the ice thaws,” which the accepted-update check expects to remain as history.

The source classifies required omissions as model failures and recommends summaries that retain consequential constraints. It makes two exceptions: F4-07's consequential-state check is optional and contains the same defensible-omission allowance discussed above, so it is classified as a contract problem; and downstream F4-02 failures are attributed to its crash below.

## The allowed-file rule is stricter than the written request

One of 10 nf4 `wiki-only` checks fails, F4-06. The check allows changes only under `kb/*.md` (`development.py:401`), but the brief says “Leave the chapter itself alone” and the follow-up says “Update the wiki.”

After writing `kb/index.md`, the model creates `drafts/chapter_suggestions.md` containing “# Chapter Draft Suggestions … The film definitely shows the festival. (Draft - Not Canon)”. The final file list is `drafts/chapter_suggestions.md`, `kb/index.md` and `source/chapter.md`; the source chapter is untouched. The source regards the extra draft file as defensible work adjacent to the request, and the failure as an underspecified contract. It proposes allowing `drafts/*` or explicitly saying “modify only `kb/`”. Bf16 passes 10/10 because it creates no extra file.

## One alternatives response treats a suspicion as established

F3-02 fails one of five required `three-directions` checks. The judge says “Direction 2 treats unestablished guilty concealment as fact, materially narrowing the central uncertainty.” The brief says only that Suri “wonders whether Aven damaged its song.” This is the same belief-to-fact error described under continuity. The remaining isolated failures—`saved-scene` F2-02 at 1/10 and `qualified-belief` F4-05 at 1/10—are already explained above.

## A tool-call parsing crash prevents one update

Nf4 F4-02 is the only run-level crash in 100 attempts. Its result is at `runs/custom50-e2b-it-2026-09-14/attempts/562592f7ff0981a0391a015aeca227b3cf37c82878e9e1e75a7aadbb1e4d388a/attempt-0001-cb39da41/result.json`, with `status: "error"`.

The model reads `source/chapter.md`, then successfully writes `kb/index.md` and `kb/characters/Suri_Aven_Dynamic.md`. On the follow-up “Draft-only possibility, not accepted canon: Aven deliberately broke the tree…” it attempts another `write_file`. The failing `model_output-010.txt` embeds the tool protocol's quote placeholder in the content:

````text
…<channel|><|tool_call>call:write_file{content:<|"|># Continuity Wiki: Chapter Planning … ```,path:<|"|>kb/index.md<|"|>}<tool_call|><eos>
````

The harness raises `ValueError: json: could not parse after dialect transforms. … Expecting ',' delimiter: line 1 column 2183`. Here “dialect transforms” means conversion from Gemma's tool-call representation into JSON the harness can parse. The source attributes the crash to embedded code fences and delimiters confusing that conversion, rather than refusal or creative judgement. It calls the emitted call well-formed-looking, not formally proven valid. The tool-use report describes it as malformed JSON; the precise boundary of model and parser responsibility remains unresolved.

The source assigns F4-02's downstream `wiki-links`, `consequential-state` and `accepted-update` misses to the aborted write and proposes handling embedded delimiters robustly or rejecting only the bad call and continuing.

## Limits on interpreting these results

The issues assigned to the harness are optional-context omissions (5/10 nf4 checks), F4-07's optional consequential-state omission (1/10), the combined viewpoint/style check (nine failures out of 15, subject to the contradiction above), F4-06's allowed-file rule (1/10), missing-prose coverage (five nf4 plus two bf16, approximately 7/60 prose attempts), and the parser crash (1/50 nf4). Their mechanisms differ, so their counts are not additive estimates of model failure.

Several checks have only five examples: accepted update, three directions and wiki links per format. The saved-scene result rests on one failure. These are directional findings rather than precise rates. Continuity and viewpoint/style depend on subjective language-model judgements; the source checked their quotations against the workspace, but that does not eliminate judgement calls.

All 115 bf16 semantic checks remain pending. The source excludes the four `distribution-*` runs, described there as 25 repetitions of F1-01, from its 100-attempt scope. Other lane reports describe two distribution runs; this inventory discrepancy does not alter the declared custom50 scope, but needs reconciliation before reproducing the broader study.

The original analysis is [failures.md](../../transcript-analysis/failures.md). Code locations above are the original report's references, retained as evidence pointers rather than a claim that the rewrite revalidated current code behavior.
