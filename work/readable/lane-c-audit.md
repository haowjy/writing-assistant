# Lane C: make the rationales and agent reports usable project memory

This audit covers four training-data documents and eleven background-agent reports. The separate rewrites explain each document's purpose, preserve its findings and limits, and give a returning reader enough context to understand the work without reconstructing the original sessions.

The audit applies `.pi/skills/llm-writing/SKILL.md`. Findings within each document are ranked from most obstructive to least: a missing meaning or unsupported conclusion comes before presentation defects. The source documents remain unchanged. Completion claims in the rewritten agent reports describe what those agents reported and checked at the time; this editing pass did not rerun their experiments or validators.

## Vocabulary rationale

Source: [vocabulary-rationale.md](../training-data/vocabulary-rationale.md). Rewrite: [why the vocabulary is being expanded](training-data/vocabulary-rationale.md).

1. “solves the 40/40 silent-default finding” and “solves the 9/10 overshoot” present proposed remedies as established results. The reader needs to know which behaviour was observed and whether the new rules were tested. The rewrite retains both counts and the intended fixes while identifying the absence of an outcome measurement.
2. “CONTINUITY — positive continuity pressure: reveal/withhold timing, learning, payoff, dormancy” names the concepts without explaining the writer's task. The missing mechanism is that a fact must become known at a particular moment, change a decision, or stay inactive until a later consequence. The rewrite explains these actions before listing the challenges.
3. “a genre-competent register” and “genre-pleasure prose” assume that the reader shares the author's critical vocabulary. These phrases obscure practical differences such as military orders and hardware, rapid dialogue, plain narration or familiar genre conventions. The rewrite explains what each addition asks the writer to do.
4. “relational/comic: intimacy rendered through idiom rather than declaration” exemplifies the repeated label-and-fragment construction. The reader must supply the subject, causal link and benefit. Every item now has a complete explanatory sentence while retaining its exact catalog wording.
5. “the explicit choices make a silent default impossible” overstates what a predicament guarantees. An agent can still silently choose how a council chair breaks a tie. The rewrite attributes this claim to the original rationale instead of presenting it as proven.

Content that editing cannot settle: the clarification composition rule says “the task must ask the choice”, without identifying whether the generated task or the answering agent asks. Nor does the list specify when to ask versus state a default. The two numerical motivations refer to different samples; the prose analysis supplies the 9/10 explicit-piece denominator and the specificity analysis supplies the 40/40 non-alternatives loose attempts, but this rationale does not establish the effectiveness of its remedies. The rewrite preserves the proposal and does not invent an implementation or measured improvement.

## Worlds rationale

Source: [worlds-rationale.md](../training-data/worlds-rationale.md). Rewrite: [why these settings were added](training-data/worlds-rationale.md).

1. “variational collapse seen in the bf16 arm” and “self-BLEU 0.71” put an unexplained model format and metric in the opening. The reader needs to know that bf16 denotes the 16-bit model-weight format and that higher self-BLEU means more overlap among outputs. The rewrite explains the 12/25 repeated openings as the concrete observation motivating broader settings.
2. “Collapse risk addressed” makes the table's most important column depend on a label rather than a question the reader can answer. A reader wants to know how a foundry, clinic or group chat changes the writing. The rewrite groups the 24 worlds into 12 setting families and explains each setting's voice and factual constraint in prose.
3. “the warped-sample pass is a specific tempting error that supports scoring” compresses the event and its significance. The missing action is passing a warped sample despite a test standard. That action supplies a concrete, checkable continuity error.
4. “Each world has 5 `required` facts plus `belief`, `incidental`, `forbidden`, `update`/`new_state`” assumes familiarity with the data schema. The rewrite explains which facts must survive, which beliefs remain unproved, which details may change and which assertions are forbidden before using the field names.

Content that editing cannot settle: this is a proposal rationale, not evidence that the additional worlds improve output diversity. The original provides the triggering observations but no post-extension measurement. Its claims about the individual worlds are retained as design rationales. The original does define F1–F5 in its coverage list; the problem is not that every label is undefined, but that the definitions compete with the table and do not explain the opening measurement.

## Steer-script companion

Source: [steer-scripts.md](../training-data/steer-scripts.md). Rewrite: [follow-up messages](training-data/steer-scripts.md).

1. “REQUIRES_ASK”, “PERMITS_DEFAULT” and “MUST_NOT_ASK” appear in a table with no definitions. Counts alone cannot tell the owner whether the agent is expected to clarify an open choice, declare a default or act on a clear instruction. The rewrite explains all three before comparing their counts.
2. “Depth x kind coverage” introduces a wide matrix whose axes are not explained. The reader needs to know that depth counts follow-up messages and kind classifies what the user does. The rewrite defines the nine kinds and transposes the table so that each row names one user action with its exact data label.
3. “reverting canon, re-deriving dependent objects, and holding the new state” hides the concrete repair. The rewrite says that the agent restores established facts, reconsiders objects affected by an invented death and preserves the correction.
4. “These are the exact behaviour the project has no data for” leaves the gap vague until after the tables. The rewrite opens with the missing interaction: an unstated assumption is noticed and corrected by the writer, and that correction must survive later turns.

Content that editing cannot settle: this companion reports 191/210 terse steers as 91%, while p47 calls the same count 90%. The fraction is about 90.95%; the discrepancy is rounding or truncation, not evidence of different underlying counts. The companion retains its reported 91%, and p47's rewrite records both. Definitions of response pressure also need to remain task-specific: the later training-wave report allows a provisional assumption in some REQUIRES_ASK cases, whereas these script labels describe a need to ask.

## Worked-example index

Source: [worked-examples.md](../training-data/worked-examples.md). Rewrite: [find a reference answer](training-data/worked-examples.md).

1. “All five families appear across nine reply and eleven file deliveries” assumes that the reader remembers the family codes and delivery channels. The rewrite defines F1–F5, reply versus file delivery, and KB before the table.
2. “These first-turn demonstrations use decisions, interrupted tasks, and dialogue to move 14 scenes” describes craft before explaining what the index is for. The new opening identifies 20 reference demonstrations, how to find them and their assistant authorship.
3. “Turn-scope exception: F4-04 intentionally leaves two required final-state checks unmet” contains an essential limit behind a bold label. The rewrite states plainly that a later accepted revision has not happened yet; it also retains F3-02's first-brief limit.

Content that editing cannot settle: there is no unresolved contradiction in this short index. The examples are explicitly assistant-authored demonstrations for human review, not independently human-authored reference data. The rewrite retains that distinction and every case, word count and link, directing links back to the existing supporting files rather than creating nonexistent copies.

## Failure-analysis agent

Source: [p40.md](../spawn-reports/p40.md). Rewrite: [separate model and evaluation failures](spawn-reports/p40.md).

1. “word-budget overlength 16/23 scored (nf4 9/15, bf16 7/13)” combines unlike denominators without warning. A reader cannot add the displayed subgroup denominators to obtain 23. The full failure analysis explains that nf4's 15 includes five unscored checks; the rewrite states this without replacing the reported counts.
2. “capability vs harness/contract” gives the classification before defining it. The distinction is whether the model failed to do the requested work or the task-running/scoring software applied a defective requirement. The rewrite explains that distinction through concrete failures.
3. “untouched files vacuously pass `saved-scene`/`preserve-ending`/`edit-scope`” hides the mechanism. A pre-existing file can remain non-empty with its original ending even when no revision occurred. The rewrite explains why those checks can pass absent the requested action.
4. “bf16 semantic failures are unknowable” is stronger than the actual limit. They were not graded in this analysis. The rewrite retains the 115 pending checks and missing scorecards and identifies semantic comparison as unavailable here.

Content that editing cannot settle: the short report alone does not explain why 93 delivered outputs and 99 completed attempts are different. The linked full analysis confirms that files written before the single crash count as delivered. The rewrite carries that clarification. It preserves the analyst's caveat that language-model judgements of style and continuity are subjective, including the strictness flags for F1-06 and F1-09.

## Prose-craft analysis agent

Source: [p41.md](../spawn-reports/p41.md). Rewrite: [recurring weaknesses in prose](spawn-reports/p41.md).

1. “the ‘language 2.20’ driver”, “‘characterization 2.24’ driver” and “direct cause of redundancy 2.00” give scores without a scale and turn interpretation into causation. The full craft analysis supplies a five-point scale. The rewrite defines it, explains the proposed mechanism and attributes causal claims to the analyst.
2. “H2 appositive/participial afterthought” and “H8 summary instead of scene” mix grammatical jargon with unexplained habit identifiers. The rewrite defines the constructions and introduces H1–H11 as this report's numbered habits. Counts, quoted phrase examples and every named habit remain.
3. “Goal check: ✅ file exists and is non-empty” leads with file existence rather than the research result. The owner needs to know what question the agent investigated and what the evidence supports. The rewrite leads with that question and preserves the later file-size and verification details.
4. “100 pieces available, ~35 read closely” appears beside statements about reading 25 pieces in full, their counterparts and 50 distribution outputs. The reader must distinguish material available from material examined in depth. The rewrite separates the 150-attempt corpus, 100 prose pieces available, 25 line-by-line readings and approximately 35 close readings.

Content that editing cannot settle: the precise depth of reading for every counterpart and repeated output is not recorded in the short report. “Memorised stock phrases” establishes repetition but does not prove that the phrases were memorised from training data. The rewrite preserves the analyst's characterisation with that limit. It also retains the rejected fast-resolution hypothesis, small-sample restriction on genre claims and exclusion of the two non-prose families.

## Tool-use analysis agent

Source: [p42.md](../spawn-reports/p42.md). Rewrite: [project-file use and continuity](spawn-reports/p42.md).

1. “identical tool-use behaviour across arms” conflicts with the displayed totals “nf4 73, bf16 85” if read as identical counts. It could mean a similar qualitative pattern, but the source does not define that scope. The rewrite quotes and limits the claim rather than silently deciding its meaning.
2. “0 cold writes” conceals the operational definition. The rewrite states that none of 43 writing attempts wrote before doing any reading, then preserves the separate same-path and newly-created-file counts. Reading any file and reading the file being overwritten are different checks.
3. “the raw 13 vs 12 split is a scenario-assignment artifact” leaves the quantity being split unidentified. The rewrite preserves both numbers and the source's attribution while flagging the missing quantity.
4. “Canon authorization” and “Q13” assume knowledge of the story-data contract and rubric. The rewrite explains established facts, proposed draft changes and the continuity check before giving the results.

Content that editing cannot settle: the meaning of “identical” and the exact measure behind 13 versus 12 remain unresolved in this short report. The 30 graded continuity checks belong only to nf4 in the three prose families. Separate manual inspection of bf16 cannot turn the failure table into a complete, comparable semantic evaluation; the reported lower-bound caveat remains.

## Brief-specificity analysis agent

Source: [p43.md](../spawn-reports/p43.md). Rewrite: [what changes when briefs leave decisions open](spawn-reports/p43.md).

1. “ASK 7, SAY 3, SILENT 89, IGNORE 0” supplies classification counts without definitions. The rewrite explains the four classes using the full analysis and identifies the 99 final-answer attempts as their denominator.
2. “Loose still names the deliverable in 12/25, so the axis is length/POV/scope, not files” is compressed methodological reasoning. The rewrite explains that point of view is narrative perspective and that named paths in loose tasks complicate a clean comparison of instruction specificity.
3. “Control: on F1-01 explicit, bf16 hit 25/25 in-range while nf4 hit 7/25” assumes the reader knows the repeated-prompt experiment. The rewrite says that one explicit prompt was repeated 25 times in each weight format and explains why it isolates a contribution beyond brief specificity.
4. “Report saved and verified” delays the important result: all 40 relevant loose attempts silently chose defaults. The rewrite puts that outcome in the opening and retains the file checks later.

Content that editing cannot settle: differing condition mixes within F2–F5 prevent a perfectly controlled interpretation of explicit versus loose briefs. The rewrite preserves that caveat and the distinction between repeated-prompt control results and the main comparison. No missing measurement is inferred.

## Task-balance specification agent

Source: [p45.md](../spawn-reports/p45.md). Rewrite: [specify a balanced task pool](spawn-reports/p45.md).

1. “13 HARD axes, 14 COVERAGE, 6 SAMPLED, 9 REPORTED” names a hierarchy without telling the reader what each category requires. The rewrite describes enforced balance/separation, minimum representation, sampled variation and measurement-only characteristics.
2. “SFT ≈ 200 accepted trajectories” and “GRPO needs ~256–300 distinct prompts (reuse ≤4)” depend on unexplained training methods and units. The rewrite defines supervised fine-tuning, Group Relative Policy Optimization, trajectories, prompts and rollouts before explaining why the two methods admit different material from the same pool.
3. “1/√rollouts variance argument” embeds a potentially incorrect statistical rationale inside shorthand. The source calls this variance; the usual independent-sample relationship is for standard error. The rewrite preserves the stated claim and flags it for technical review rather than repairing the research silently.
4. “H5/H6/H9/H10/C6/C7” and “H11/H13/C13” make prerequisites unintelligible without another document. The rewrite names each affected requirement using the full specification and explains why missing schema fields and too few source groups block implementation.
5. “the external anchor is the only falsifier” hides a strong methodological claim behind a metaphor. The rewrite identifies EQ-Bench and explains the claimed independence from the reward design, attributing that claim to the proposal.

Content that editing cannot settle: statistical justification needs review; the listed viewpoint/schema contradictions remain; and acquiring more than five works and two lineage groups remains a prerequisite for the stated source requirements. These are implementation limits, not blockers to the original agent's delivery of a written specification.

## Vocabulary-extension agent

Source: [p46.md](../spawn-reports/p46.md). Rewrite: [expand the generation vocabulary](spawn-reports/p46.md).

1. “Appended new atoms” uses an internal abstraction for ordinary catalog entries. The rewrite names the entries and explains their categories before comparing counts.
2. “positive-continuity operationalisation” compresses the proposed action into a noun stack. The rewrite says that rules must specify what becomes known, when it happens and what mistiming changes.
3. “batch-level anti-collapse, trope-as-pressure, ... load-bearing genre blends” hides three distinct requirements. The rewrite explains variation across a batch, tropes changing decisions and every genre changing how a blended scene works.
4. “Goal complete” precedes the assignment and makes file preservation compete with the purpose of the additions. The rewrite states the assignment and result first, then keeps the count table and each reported preservation check.

Content that editing cannot settle: “weighted toward PROSE-WRITING situations” was interpreted by the author as scenes suitable for dramatisation, not a measured sampling weight. The rewrite preserves that interpretation and the 24/24 situation split. Structural validation does not show that the additions improve later model behaviour.

## Follow-up-script agent

Source: [p47.md](../spawn-reports/p47.md). Rewrite: [write follow-up conversations](spawn-reports/p47.md).

1. “36 scripts / 210 steers” and “12 scripts per depth band” assume the units are obvious. The rewrite distinguishes a user message from a sequence of messages and defines the three length bands.
2. “PERMITS_DEFAULT 15, MUST_NOT_ASK 12, REQUIRES_ASK 9” gives unexplained expected-response categories. The rewrite defines each and preserves that every assignment is justified by visible text rather than private reasoning.
3. “Terse (<12 words): 191/210 (90%)” disagrees with the companion's 91% for the same fraction. The rewrite retains the exact count, both reported percentages and the approximate arithmetic value.
4. “Band counts are equal by script ... equal steer volume” leaves a crucial coverage choice until the notes. The rewrite explains it with the length-band totals: equal numbers of scripts produce unequal numbers of messages.

Content that editing cannot settle: equal script counts were the author's interpretation of equal coverage; the report itself says equal message volume would require rebalancing. No rewrite can resolve that original intent. The terse-percentage discrepancy is recorded above; no count was changed.

## World-authoring agent

Source: [p48.md](../spawn-reports/p48.md). Rewrite: [add settings and continuity constraints](spawn-reports/p48.md).

1. “a non-established `belief`, freely-changeable `incidental`, a specific `forbidden` assertion” treats field names as the explanation. The rewrite explains unproved belief, changeable detail and prohibited assertion before the schema inventory.
2. “family/register/suited scenario families (F1–F5 ... ) and the specific collapse risk” compresses setting category, prose voice, task type and repetition into one phrase. The rewrite separates those roles and defines all five task families.
3. “Mirrored shape exactly” assumes the reader knows why a bare list matters. The rewrite states that the proposal follows the existing file's top-level list and 18-key records, then preserves the exact keys and verification results.

Content that editing cannot settle: the short report does not explain the separate roles of `style` and `prose_style` enough to treat them as interchangeable; the rewrite retains both keys. The structural checks verify the proposal's format and coverage, not a reduction in repetitive generated prose. No original blocker was reported.

## Evaluation-case authoring agent

Source: [p49.md](../spawn-reports/p49.md). Rewrite: [create evaluation cases](spawn-reports/p49.md).

1. “Created 24 `final_eval` cases” begins with an undefined role and no assignment. The rewrite explains that the agent prepared the first evaluation batch and that `final_eval` marks its final-evaluation role.
2. “Verified 30 artifact selectors, 161 required checks, and all six retrieval paths” gives three kinds of verification without saying what they validate. The rewrite defines output selection and source-retrieval sequences, and distinguishes successful preparation from measured model performance.
3. “Every balance target matched” is impossible to inspect from the short report alone. The rewrite retains the claim and links the repository copy of the full report with its target-versus-actual counts, commands and outputs.

Content that editing cannot settle: this summary contains no model outcome or literary-quality measurement and does not enumerate the targets. The full report carries the evidence and corrected validator-fixture failure. The rewrite does not manufacture a model result from successful compilation.

## Training-case authoring agent

Source: [p50.md](../spawn-reports/p50.md). Rewrite: [create training cases](spawn-reports/p50.md).

1. “12/24/12 response pressure” omits the categories and their order. The full report identifies 12 REQUIRES_ASK, 24 PERMITS_DEFAULT and 12 MUST_NOT_ASK cases. The rewrite names and defines them, including the full report's allowance for a provisional assumption in the first category.
2. “16 reply-with-tools cases” assumes that output location and tool access are the same concept. The rewrite explains that these tasks require a chat reply while making file tools available.
3. “324 artifact-linked checks” replaces the relationship with a compound label. The rewrite explains that every check refers to the particular reply or file content being assessed.
4. “48 original sources using the validator-supported `synthetic` provenance” sounds contradictory without the distinction between new authorship and provenance category. The rewrite makes that distinction and avoids claiming human authorship.

Content that editing cannot settle: REQUIRES_ASK is not used as an absolute must-ask rule in the full training-wave report, so treating it as universally literal would misstate the cases. The rewrite records the local meaning. Numeric and loader checks do not establish writing quality or training improvement; no such result is invented.

## Worked-example authoring agent

Source: [p51.md](../spawn-reports/p51.md). Rewrite: [supply worked answers](spawn-reports/p51.md).

1. “All 41 deterministic checks pass” followed by “All 43 required checks accounted for” invites the mistaken conclusion that all requirements passed. The rewrite distinguishes passing mechanical checks from two unmet requirements that depend on a later follow-up.
2. “All five families; nine reply and eleven file examples” supplies counts without defining the task families or delivery locations. The rewrite defines F1–F5 and distinguishes reply from file delivery.
3. “Semantic checks remain unscored; quoted author assessments are provided” is a central evidence limit placed after success counts. The rewrite states the unscored status in the opening and explains that meaning is not independently assessed by the mechanical pass count.

Content that editing cannot settle: the two later-turn checks remain unmet, and semantic quality remains unscored. These are explicit limits of first-turn, assistant-authored demonstrations for human review, not defects that an editorial pass can fix.

## Verification of this editorial pass

All four document rewrites and all eleven report rewrites are present as separate, non-empty files. Vocabulary coverage was checked against every catalog addition, and world coverage against all 24 identifiers and task-family assignments. The 20 example rows, their word counts and their destinations were checked against the original index. Counts, measurements, verification details, limitations and historical output locations were reviewed against each original report; supporting local reports were read where necessary to define a term or identify a denominator. Original-file hashes were checked to confirm no original was changed. `meridian kg check` passed for all 16 new files and 78 links. Historical session paths remain as provenance; the links point to available repository files. The house style note below is 351 words.

## House style for future repository reports

Open with the assignment, result and practical limit. “Report saved and verified” in p43 tells the reader only that a file exists. Prefer: “The agent compared explicit and loose briefs; all 40 relevant loose attempts silently chose defaults.” Put file sizes and verification commands after the finding.

Define a label when it first matters. “self-BLEU 0.71” in the worlds rationale needs “higher means more repeated wording across outputs.” Keep case IDs and data keys so readers can find evidence, but introduce them after their plain meaning.

Explain an action instead of naming an abstraction. Replace p46's “positive-continuity operationalisation” with the requirement to name the fact, when it becomes known and what changes if the timing is wrong. Replace p50's “artifact-linked checks” with checks attached to a specified reply or file.

Give the conclusion before the table and name every unit and denominator. The steer companion's “Depth x kind coverage” should first say that every kind of feedback appears at every conversation length. Never mix scored and unscored denominators as p40's “16/23 ... 9/15 ... 7/13” does without explaining the difference.

Separate observations, interpretations and proposals. The vocabulary rationale's “solves the 40/40 silent-default finding” is a proposed remedy, not a measured result. The prose report's “direct cause” is an analyst's interpretation unless the method isolates that cause. Preserve strong claims by naming their evidence and author; do not turn them into vague reassurance.

Use complete sentences for findings. P41's “H1 atmospheric-opening template 21/25” needs an actor and meaning: “The model opened 21 of 25 pieces with atmosphere even though the briefs did not ask for it.” Bold labels, check marks and slash lists cannot supply that explanation.

State completion limits beside completion claims. P51's 41 passing mechanical checks coexist with two unmet later-turn requirements and unscored semantic checks. Name all three together. Distinguish successful file creation, successful validation and successful model behaviour.

Make reports self-contained enough to read a month later. Replace a bare session path with a named local report link while retaining the session identifier and historical path for traceability. Do not append a second summary that repeats the opening.
