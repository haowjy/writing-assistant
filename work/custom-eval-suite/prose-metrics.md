# Prose metric catalog

Record a quantitative profile of generated prose alongside the suite's semantic
quality judgments. These are secondary improvement targets and diagnostics, not
one core score to maximize. This catalog carries forward the measures identified
in the [research survey](../research-plan/dataset-and-metric-research.md) and
[writing experiments](../research-plan/evaluation-experiments.md).

All entries use the [prose-selection contract](design.md#selecting-prose-from-replies-and-files):
selected prose in assistant turns or manuscript files, excluding conversational
commentary, plans, tool traces, and KB text. Local edits use their declared passage
scope. Multiple versions/copies are not independent samples.

These measures have a second use beyond diagnostics. D1, D2 and D4 are the primary
metrics by which [distribution fine-tuning](../sft/distribution-finetuning.md) measures
how far generated text sits from human writing, so they are also the instrument for
judging whether a change to the data construction moved the output distribution. That
framing changes how to read them: they compare a *population* of outputs against a human
reference population, not one artifact against one expectation, and D1 and D2 need enough
samples to be meaningful. Their status as secondary targets is unchanged — matching a
metric is not the objective, and the author of the technique reports adding seven further
metrics specifically to check he was not doing that.

**Four of them are defined across outputs, not within one.** `prose.sample_distribution`
pooled over repeated attempts is what makes D2, D4 and D6 exist; a per-attempt profile can
only ever report them as `insufficient_samples`. Sizing the run matters more than choosing
the metric: `prose.MINIMUM_SAMPLES` withholds a measure below its floor, and
`RELIABLE_SAMPLES` marks where two models can be usefully compared. `prose.sampling_plan(n)`
resolves a proposed sample count against those floors before the GPU time is spent. The
current final suite runs four attempts per case and therefore cannot report D2 at all —
see [the sampling section](longform-suite.md#sampling-and-the-measurement-this-suite-cannot-make).

## Measurements to record

Availability below defines required inputs, not permission to drop a measure
silently. Register every entry in the reporting schema. Compute applicable measures
when their dependencies are available; otherwise retain `not_applicable`,
`insufficient_samples`, `missing_reference`, or `not_implemented` with a reason.
Implementation availability and verification are recorded in [delivery evidence](delivery.md).
Repeated-generation measurements consume saved outputs; adding a metric does not
authorize additional candidate generations.

| ID | Measurement | Inputs and calculation | Interpretation / availability |
|---|---|---|---|
| D1 | N-gram token-distribution L2 | Generated and matched human prose; separate normalized-frequency L2 for n=1,2,3. | Distribution similarity, not within-output repetition. Record in development release; [formula/configuration](metrics.md#d1-n-gram-distribution-distance). |
| D2 | Embedding MMD squared | Generated/reference embeddings and fixed RBF kernel; unbiased estimate, with sample counts and bandwidth. | Distribution difference in a learned representation. Record in development release; [formula/configuration](metrics.md#d2-embedding-mmd). |
| D3 | MAUVE | Two prose collections; pinned feature model, clustering configuration, seed, text policy, and library version. Retain score and divergence curve. | Higher means closer distributions under this method, not necessarily better fiction. Requires a larger collection for meaningful comparison; tiny pilot values are exploratory. |
| D4 | Self-BLEU | Repeated outputs for the same prompt; each output is the hypothesis and the others are references; average scores. Pin BLEU order, smoothing, and tokenization. | Higher means more lexical overlap across alternatives. Needs at least two outputs; hold number/length of references constant. |
| D5 | Distinct-n | Unique n-grams / total n-gram occurrences, for n=1,2,3; report per-artifact macro means and separately pooled counts. | Lexical variety; strongly length-dependent. Empty/too-short samples are N/A. Compute from cached counts. |
| D6 | Within-prompt embedding dispersion | Mean pairwise `1 - cosine_similarity` among repeated outputs for one prompt; macro-average across prompts. | Semantic variation can include topic drift or incoherence. Use the same fixed embeddings and generation counts across candidates. |
| D7 | ROUGE-1/2/L | Generated prose paired with a relevant reference; record precision, recall, and F1 with tokenizer and multi-reference aggregation fixed. | Reference overlap. Conditional on a meaningful paired task; low overlap is not failure for an alternate plot or original continuation. |
| D8 | BERTScore | Candidate/reference contextual token alignment; record precision, recall, F1, model/layer, IDF and rescaling configuration. | Learned semantic similarity without an LLM judge. Needs appropriate paired references; cannot substitute for creativity or continuity assessment. |
| D9 | Vocabulary and prose structure | Word type/token ratio; sentence/paragraph length distributions; duplicate-paragraph and repeated-sentence-opening rates. Pin word/sentence splitting and exact repetition rules. | Descriptive style profile. Report lengths and genre/style groups; there is no universal preferred value. |
| D10 | Source/prompt overlap | Candidate n-gram occurrences also present in the relevant prompt or supplied source / candidate occurrences; report prompt and source separately with n pinned. | Helps detect copying or context echo. Quotations and retained revision text may be intentional; measure only the declared generated scope. |
| D11 | Repetition and cross-output overlap | Within-prose repeated trigram occurrences beyond first / total occurrences; across alternatives, duplicate-output rate and pairwise trigram-set Jaccard. | Separate repetition within a scene from reuse across samples. Lower is not always better; intentional motifs should survive. |
| D12 | Held-out prose likelihood / perplexity | Candidate model log probabilities on fixed human prose under a specified context/windowing protocol; record total negative log likelihood and token/word counts. | A companion reference-corpus evaluation, not a feature of generated prose. Needs a log-probability backend; compare token perplexity only with compatible tokenization, or use a documented word-normalized protocol. |
| D13 | Author retention (USER) | Actual generated suggestion and the author's edited result; use the published retained-span algorithm and normalization. | Conditional on real editing records; not computable from a draft alone. Simulated edits must be labeled separately and cannot establish human acceptance. |

D3 and D8 use learned representations; D12 uses candidate model probabilities.
None requires a generative LLM grader, but none is a wholly model-free measurement.
D1/D4/D5/D7/D9/D10/D11 are deterministic given their text preprocessing and labels.
D13 requires genuine author interaction and a pinned matching implementation.

## Semantic measures retained alongside the profile

The existing Q2 prose rubric and pairwise judging retain coherence,
characterization, pacing, language, style adherence, redundancy/over-explanation,
and task-appropriate dialogue or imagery. Additional rubric views from the earlier
research include engagement, emotional credibility, subtext, originality, and
surprise with retrospective justification. They are subjective assessments, not
new deterministic formulas. Use only applicable dimensions and version their anchors.

For the surprise experiment, collect a prediction/plausibility judgment from prior
context before showing the development, then separately assess its causal support
afterward. Do not claim a post-hoc surprise rating measures reader prediction.

The earlier `JMQ` proposal maps to saved pairwise wins/ties/losses and
`(wins + 0.5*ties) / comparisons`. Do not relabel twice the win rate as a 0–1 score.
The earlier `D(n)` idea maps to the repeated-sample diversity profile: use D4/D5/D6
and D11 at declared sample counts. Mean pairwise dispersion need not increase with
sample count, so it is not automatically a diversity-growth curve.

## Interpretation and reporting

Compare matching prompts, genre/style strata, passage types, and length ranges.
Preserve per-artifact counts/embeddings and group-level outputs so the representation
can be inspected and reused. Numeric records include metric/configuration version,
prose/reference hashes, sample counts, status, and uncertainty where supportable.
Do not pool metric values from different embedding models, tokenizers, or references.

For variability, compare all outputs and those meeting predeclared instruction,
continuity, and quality criteria, reporting the retained fraction. Seek more varied
successful prose rather than random wording. Reference proximity, within-passage
repetition, and between-output diversity are separate properties and remain separate
columns. Do not treat every increase/decrease as an improvement.

D1/D2 and cheap text features belong in the development run. D4/D6 and cross-output
D11 require the planned repeated-generation subset. D7/D8 run on paired-reference
tasks. D3 requires a larger matched corpus; D12 requires log-probability support;
D13 requires authentic edits. Cache features and add these passes without repeating
candidate generation unnecessarily. No dependency justifies scoring the surrounding
conversation as prose.

## Research basis

- [Deft writing experiment](https://deftwriting.com/research/distribution-fine-tuning): n-gram L2, MMD, self-BLEU, repetition, and JMQ; author-reported evidence, not a universal prose-quality standard.
- [Gretton et al.](https://jmlr.org/papers/v13/gretton12a.html): statistical basis of MMD.
- [MAUVE implementation and papers](https://github.com/krishnap25/mauve): distribution comparison through quantized features; recommends thousands of samples for best practice. Our prose-only extraction is a declared task adaptation.
- [Texygen](https://arxiv.org/abs/1802.01886): text-generation evaluation including Self-BLEU.
- [ROUGE](https://aclanthology.org/W04-1013/) and [BERTScore](https://github.com/Tiiiger/bert_score): reference-based text similarity.
- [Agents' Room](https://arxiv.org/html/2410.02603): writing evaluation with reference similarity, lexical diversity, repetition, and human/LLM comparisons.
- [PG-19](https://github.com/google-deepmind/pg19): word-normalized perplexity on held-out books.
- [Storium paper](https://aclanthology.org/2020.emnlp-main.525.pdf): USER retention with real author edits; source access remains excluded, but the measurement is retained for future eligible interaction data.

The catalog is not a claim to implement every metric bundled in HANNA or every
possible style statistic. Preserve those as upstream research; add a new measure
when its inputs and interpretation are specified.
