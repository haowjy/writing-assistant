# Scorecard and grading

The [suite](index.md) records 13 quality measures, a
[prose metric catalog](prose-metrics.md), and two resource measures. These are reporting categories, not
separate grader calls. A rubric packet can score
several dimensions of one artifact. The operational checks are implemented;
our combined measures and Astra rubrics are not yet validated. Deterministic
checks and reference-based measurements run independently of Astra; semantic
judging supplements them.

## Quality measures

| ID | Measure | Definition and applicability |
|---|---|---|
| Q1 | Instruction adherence | Per-session passed applicable constraints / applicable constraints; macro-average across sessions. All families. Report hard violations alongside the fraction. |
| Q2 | Prose quality | Anchored 1–5 ratings for coherence, characterization, pacing, language, and redundancy when applicable. Report the dimension vector and median overall holistic rating; do not average missing dialogue criteria into every scene. F1/F2/F5. |
| Q3 | Task completion | Fraction of sessions meeting every predeclared required outcome, including artifact destination and edit scope. Separate successful completion from merely reaching a final assistant response. Tool/workspace families and multi-turn plans. |
| Q4 | Tool correctness | Schema-valid, permitted tool calls / attempted calls. Report execution failures and semantic misuse separately: a valid read of a missing file is not malformed syntax. Tool-enabled conditions. |
| Q5 | Planning usefulness | Anchored 1–5 judgment of whether ideas address the brief and can guide writing; assess relevance, feasibility within canon, and actionable development. F3. |
| Q6 | Alternative diversity | Distinct viable idea pairs / all requested idea pairs, using a fixed semantic-distinction rubric. Missing alternatives contribute no diverse pairs. F3 with at least two requested alternatives. Report the viable-alternative count too. |
| Q7 | KB faithfulness | Supported factual claims / checked factual claims. Separate contradicted and unsupported claims; assess qualified interpretations under Q9. F4. Audit a reproducible sample when exhaustive review is too expensive. |
| Q8 | Important-information coverage | Sum of reviewed importance weights correctly retained / sum of reference importance weights. F4 within a fixed storage budget. Required omissions are reported separately; optional omissions are not hard failures. |
| Q9 | Knowledge interpretation | Anchored 1–5 rating for evidence use, entity resolution, temporal/causal understanding, and preserving uncertainty. Accept defensible alternatives; unresolved evaluator disagreement remains visible. F4. |
| Q10 | Navigability | Correct, evidence-supported answers to hidden probes within the fixed read/tool budget / all probes. Report tokens read, tool calls, link failures, and reachable pages as diagnostics. F4 using a fixed fresh reader. |
| Q11 | Update correctness | Correctly satisfied update obligations / applicable obligations: new state, valid historical state, superseded claims, and preservation of unrelated material. F4 update sessions. |
| Q12 | Retrieval success | Required evidence units exposed by tool observations / required evidence units for the task. Semantic evidence matching permits different pages. F5 retrieval conditions; supplied-context controls are N/A. |
| Q13 | Continuity | Satisfied applicable canon, character-knowledge, temporal, and reveal constraints / all applicable continuity constraints. F5 and context-dependent F1/F2. Allowed invention is not an error. |

For Q6, define a viable alternative using the brief and minimum usefulness rubric
before scoring distinctness. Surface plot/action/viewpoint differences count;
renaming the same idea does not. Embedding distance alone is not this measure.
For Q7, an empty KB has undefined precision, zero coverage where information is
required, and failed task completion. Empty outputs must never earn perfect scores.

Q1 and Q13 intentionally overlap: one summarizes brief-following while the other
diagnoses continuity. Do not add them into a grand total. Likewise, Q3 should not
hide partial success shown by other measures.

## Scoring methods

Each check declares its scoring method before a run: `deterministic`,
`reference_match`, `learned_representation`, `model_likelihood`, `human`, or
`llm_judge`. Derived ratios using judge-matched checks are labeled `llm_assisted`.
Store the checker version, input artifact,
reference/label version, and denominator. Never silently replace a deterministic
check with an LLM judgment. Report mechanical and semantic components separately
when a metric combines them.

| Measures | Non-LLM scoring | Where judgment is still needed |
|---|---|---|
| Q1 instruction adherence | Word/count limits, required format, explicit strings, forbidden paths, output destination | Meaning, tone, indirect disclosure, and semantic constraints |
| Q2 prose quality | Repetition and length diagnostics below | Literary quality, coherence, and effective use of repetition |
| Q3 completion | Required files exist and are nonempty; declared state predicates pass; protected content is preserved | Open-ended outcomes such as whether the requested revision succeeds |
| Q4 tools | Parse/schema checks, allowed tool/path checks, execution status and budgets | Whether a syntactically valid action makes sense for the request |
| Q5–Q6 plans and alternatives | Requested alternative count, required sections, exact duplicate detection | Usefulness, viability, and meaningful differences |
| Q7–Q9 KB content | Explicit entity/relation fields matched against reviewed labels; source references resolve; weighted coverage once matches are established | Freeform claim support, importance labels, paraphrase matching, and interpretation |
| Q10 navigation | Link graph checks; exact/alias answer matching for suitable hidden probes; measured read/tool cost | Open-ended answers and whether the task set represents real needs |
| Q11 updates | Expected structured state, changed-file set, protected text, and reference/link consistency | Meaning-preserving updates in freeform prose |
| Q12 retrieval | Gold source-span coverage in actual tool observations; document hit/recall when document IDs are stable | Equivalent evidence in newly organized or paraphrased KB pages |
| Q13 continuity | Structured state/constraint predicates where the output exposes them | Narrative contradictions, character knowledge, and permitted invention |
| R1–R2 resources | Timers, trace counters, backend usage, and labeled tokenizer estimates | No literary judge required |

A model may perform a navigation task while code grades its answer. That is
non-LLM **grading**, although the experiment still depends on the reader model.
Likewise, code computing coverage from human labels does not make the importance
labels objective. Record those dependencies explicitly.

## Required deterministic checks and diagnostics

These are components of the existing scorecard, not additional top-level metrics.
Run applicable checks before semantic grading and retain their raw results.

| Check | Calculation / contract |
|---|---|
| Artifact and format validity | Pass/fail per declared file, format, required field, or heading. Do not require one arbitrary wiki layout. |
| Edit scope | Compare before/after file sets against allowed changes; compare protected spans byte-for-byte when exact preservation is requested. |
| Tool validity and execution | Valid permitted calls / attempted calls; separately successful executions / attempted calls. No attempted calls gives N/A, with required-tool omissions handled by completion checks. |
| Internal link validity | Resolvable local page/anchor references / local references, under the declared Markdown dialect. Exclude external URLs from this denominator. No links gives N/A, not perfect navigation. |
| Entry-point reachability | KB pages reachable from declared entry points / KB pages. Exclude source/manuscript files; for a flat KB, check its required section navigation separately. |
| Orphan pages | Count pages with no incoming internal link, excluding declared entry points. An orphan is a diagnostic, not automatically a failure. |
| Probe answer accuracy | Correct normalized answers / probes with exact or reviewed alias matching. Freeze normalization and aliases; use this only for questions with sufficiently determinate answers. |
| Evidence recall | Required gold evidence units fully exposed in tool observations / required units. Define spans and exposure criteria in labels; a truncated search hit may not expose the full evidence. |
| Structured fact precision/recall | Matched unique labeled facts / predicted facts, and matched required facts / required facts. Report only where schema and reference coverage support these denominators. |
| Prose repetition | For a fixed tokenization, repeated n-gram occurrences beyond the first / total n-gram occurrences; start with trigrams. Report exact duplicate paragraphs separately. Too-short texts give N/A. |
| Length and effort | Words/tokens in the designated prose artifact, KB storage size, pages read, read tokens, tool calls, and resource measures R1–R2. |

Deterministic checks must not overclaim. A resolving citation need not support its
claim. A connected link graph need not be easy to navigate. Name presence is not
proof that a relationship was understood. String absence is not proof that a
secret was not revealed in paraphrase. Repetition can be intentional. These
checks expose concrete behavior and complement the relevant semantic assessment.

For freeform KBs, avoid forcing rigid model output solely to simplify scoring.
Use exact source-span checks where available and explicit semantic grading where
needed. If an LLM extracts claims or maps paraphrases before a numerical formula,
label the resulting measure as LLM-assisted; the formula alone is not a non-LLM
measurement. Embedding/NLI-based checks are learned-model measurements and must
also be labeled separately from deterministic checks.

## Prose-distribution diagnostics

The [full prose metric catalog](prose-metrics.md) includes MAUVE, Self-BLEU,
Distinct-n, embedding dispersion, ROUGE, BERTScore, vocabulary/structure, overlap,
repetition, and conditional likelihood/author-retention measures. D1/D2 below
retain the detailed configuration for our initial distribution comparisons.

These are secondary improvement targets for reducing habitual phrasing and
broadening the range of generated prose. They are not the core quality metric.
Seek improvement alongside prose quality, instruction adherence, and continuity;
do not choose a checkpoint solely because its distribution distance is lower.

Record these alongside prose judgments for F1, F2, and F5. They describe the
**generated prose**, whether returned in a reply or saved to a manuscript file.
Use the scenario's [prose-selection contract](design.md#selecting-prose-from-replies-and-files).
A prose passage inside the conversation is eligible; the surrounding conversation
is not. Exclude prompts, plans, KB pages, tool traces, and assistant commentary.
Apply the same selected artifact to prose judgments and numerical prose features,
unless an explicitly labeled experiment requires different scopes. For local
revision tasks, use the declared edited passage and report its size; scoring an
entire mostly unchanged manuscript would hide the candidate's contribution.

| ID | Diagnostic | Recorded values |
|---|---|---|
| D1 | N-gram token-distribution L2 distance | Separate distances for token unigrams, bigrams, and trigrams against a fixed human-prose reference collection; retain normalized frequency vectors and largest frequency differences. |
| D2 | Embedding maximum mean discrepancy | RBF-kernel MMD squared between generated and reference prose embeddings; retain embeddings, sample IDs, estimator, bandwidth, and sample counts. |

D1 is deterministic after tokenization. D2 uses a fixed embedding model and a
numerical estimator, without an LLM judge. It is a learned representation, not a
purely model-free metric. Neither diagnostic is a per-task pass/fail requirement.

### D1: n-gram distribution distance

For each order n, count token n-grams within each prose sample, never across sample
boundaries. Pool counts within the comparison group and normalize by total n-gram
occurrences. With generated frequencies p and reference frequencies q, record:

`L2_n = sqrt(sum_g((p[g] - q[g]) ** 2))`

Use the union of observed n-grams and zero for absent counts. Pin one evaluation
tokenizer/revision across every candidate, preserve case and punctuation, exclude
special tokens, and record text normalization. This count-pooled estimate weights
longer passages more heavily; match length distributions and report sample/token
counts. No n-gram occurrences means N/A. Do not average the three orders into an
unexplained scalar or confuse distribution distance with repetition inside a scene.

The [Deft writing experiment](https://deftwriting.com/research/distribution-fine-tuning)
uses normalized token-frequency L2 and embedding MMD. It is precedent for this
measurement combination, not validation that proximity alone means better fiction.

### D2: embedding MMD

Pin the embedding model/revision, pooling, text length/chunking policy, and vector
normalization. The initial implementation uses 256-token chunks, token-count-weighted
mean pooling, then L2 normalization. Use `sentence-transformers/all-mpnet-base-v2`, revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`, for initial embeddings. Do not silently truncate
long passages. If chunking is needed, fix how chunk embeddings become one sample
representation and keep source-group identity for uncertainty estimates.

Use `k(x, y) = exp(-||x-y||^2 / (2 * sigma^2))`. For m generated embeddings x and
r reference embeddings y, record the unbiased estimate:

`MMD_u^2 = sum_{i!=j} k(x_i,x_j)/(m*(m-1))`
`          + sum_{i!=j} k(y_i,y_j)/(r*(r-1))`
`          - 2*sum_{i,j} k(x_i,y_j)/(m*r)`

Both groups need at least two samples. The unbiased estimate can be negative;
retain it and label it MMD squared rather than clipping it or taking its square
root. Select a positive bandwidth from development reference embeddings, document
the rule, and freeze it across comparisons. The statistical basis is
[Gretton et al., A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html).

### Reference data and saved representations

Use a versioned human-prose collection separate from training, matched as closely
as possible on genre, requested style, passage type, topic, and length. Keep modern
serial fiction and historical fiction distinguishable. Reference prose is a
comparison distribution, not the unique correct continuation. Record lineage and
rights for derived frequency/embedding artifacts as well as source text.

Save per-artifact token counts, sparse n-gram counts, embeddings, and existing
repetition/length diagnostics, keyed by prose hash and feature configuration.
Save group-level D1/D2 results with candidate/reference IDs, counts, configuration
hashes, and source-group bootstrap intervals where sample size supports them.
Run a human-versus-human reference split comparison to show ordinary variation.
Small development samples produce exploratory estimates, not reliable rankings.

Compute these in a separate resumable pass over saved prose. Cache features so
rescoring needs neither candidate regeneration nor Astra calls. Run embeddings
separately from generation on the 3090 and measure their resource cost. Missing
features or reference data produce an explicit unavailable status, not a zero.

Lower distance means closer to the chosen reference in this representation. It
does not establish creativity, instruction following, or literary merit; topics,
length, copying, and stylistic conformity can move the result. Retain the scores
as a quantitative prose profile alongside the other measures, not a combined
quality reward. Reference proximity and variability are different properties: lower L2 or MMD
does not by itself demonstrate more varied prose. Use a repeated-generation subset
with the same prompts, fixed sampling settings, and equal sample counts to inspect
variation without confusing it with a change in topic mix.

For that subset, record supporting diversity diagnostics using the cached features:
mean pairwise cosine distance between prose embeddings, exact duplicate-output
rate, and cross-output trigram overlap. For nonempty trigram sets A and B, use
Jaccard overlap `|A intersection B| / |A union B|`, averaged across pairs within
each prompt; report N/A where the sets are empty. Macro-average prompt-level
results. More embedding dispersion and less overlap can indicate variety, but
can also reflect incoherence or random wording. Inspect both all outputs and the
subset meeting predeclared quality/constraint criteria; report the retained count.

The intended gain is a wider range of successful realizations of the same brief,
not maximum distance or minimum overlap. Compare trajectories of these diagnostics
across checkpoints with human-prose variation where comparable references exist.
These are supporting measurements under the prose profile, not new core quality
scores. Additional distribution metrics remain optional experiments.

## Resource measures

- **R1 — Latency:** per-session elapsed candidate execution time, plus median and
  p95 across comparable conditions. Separate setup/model loading, source
  processing, grader time, and queue time. Report KB construction and subsequent
  use separately.
- **R2 — Token usage:** input, cached input, generated output, and reported reasoning
  tokens where available, separated by builder, reader, writer, and grader. Record
  provider counters verbatim and normalize without double-counting nested fields.
  Missing usage is unknown, not zero. Fixed-tokenizer estimates are labeled estimates.

## Evidence behind the measures

[Pyramid](https://doi.org/10.1145/1233912.1233913) weights semantic content units by
frequency across human reference summaries, accommodating different valid content
selections. Q8 is a KB adaptation inspired by that method, not its official score.
[QAPyramid](https://arxiv.org/abs/2412.07096) evaluates preservation through fine-grained
question–answer units; it informs content checks and hidden probes.

[BookWorm](https://aclanthology.org/2024.findings-emnlp.258/) distinguishes factual
character description from interpretation and uses complementary metrics.
[Re-DocRED](https://github.com/tonytan48/Re-DocRED) supplies relation-extraction
annotations; [ToMBench](https://github.com/zhchen18/ToMBench) tests beliefs and
knowledge; [LongMemEval](https://github.com/xiaowu0162/LongMemEval) tests updates,
temporal reasoning, and memory use. They motivate components, not an established
freeform wiki score.

[STORM/FreshWiki](https://arxiv.org/abs/2402.14207) supplies article-organization
assessment. [ALCE](https://github.com/princeton-nlp/ALCE) measures factual answering
and citation quality. Our continuity rubric must allow new fiction and cannot
apply ALCE's factual-grounding requirement to every creative sentence.

The [existing survey](../research-plan/dataset-and-metric-research.md) covers
IFEval, FollowBench, BFCL, writing rubrics, and human preference datasets. Run
external benchmarks with their own versions and scoring. An Astra substitution
or selected subset is an adapted evaluation and must be labeled accordingly.

## Subjectivity and grader validation

Importance labels depend on the KB brief. On a development sample, collect
independent judgments of which information is essential, useful, or incidental.
Use shared judgments to establish weights and preserve disputed labels. One
person's preferences can define a personalized target but not a population norm.
Membership in a reference is not the only evidence of validity.

Draft anchors with concrete examples before generation. For navigational
organization, for example, distinguish repeatedly inaccessible information from
findable information with avoidable searching and consistently clear routes.
Behavioral navigation scores complement these judgments; neither proves universal
usability outside the chosen reader and task distribution.

Use Astra (`gpt-6-astra`) as primary semantic/prose judge. Collect evidence-backed
structured results, model and rubric versions, raw outputs, and uncertainty flags.
Blind model identity; swap order in selected pairwise comparisons and allow ties.
Record wins/ties/losses; if summarizing, use `(wins + 0.5 * ties) / comparisons`.
Pairwise preference is a validation view of quality/usefulness, not an additional
headline measure.

After separate authorization, begin calibration with 20–30 development outputs spanning obvious failures,
plausible alternatives, long/short artifacts, and format variations. Obtain two
independent human ratings where feasible. Report raw agreement, ordinal agreement
(e.g. ordinal Krippendorff alpha), and pairwise agreement with Astra. Resample by
source group for confidence intervals. Repeated Astra calls measure stability,
not independent human agreement. With one human, label results as agreement with
that reviewer. No reliable human labels means semantic results stay exploratory.

Grader calibration data is separate from final evaluation. Have human spot checks
on final results without tuning the rubric to those cases. If final labels are
found defective, document a versioned correction and rescore every affected model.

## Aggregation and failure policy

Report by family, condition, model, and source group, with sample counts, missing
judgments, failures, and uncertainty. Macro-average per scenario so a heavily
annotated task does not dominate. Related variants from one book are correlated;
use source-group resampling rather than pretending every variant is independent.
Do not publish one combined wiki or suite ranking before validating aggregation.

Model truncation, refusal, malformed calls, and exhausted budgets remain in the
attempt denominator. Missing artifacts fail Q3; conditional artifact ratings
report how many attempts produced scorable prose. Grader errors remain pending,
not candidate zeros. Infrastructure failures receive a separate status and
predefined limited retries with all attempts retained. Report the original failure
rate alongside completed evaluations.
