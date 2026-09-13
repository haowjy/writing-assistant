# Existing writing datasets and evaluation methods

Research checked on 2026-09-11 for the [branching authorship plan](branching-authorship.md)
and [Gemma baseline comparison](baseline-comparison.md).

The [current data inventory and acquisition plan](../custom-eval-suite/data.md)
tracks the five-family suite, KB-specific sources, and webnovel candidates.
Storium is excluded from acquisition because access is unavailable for this project.

Storium supplies stories with author-written planning cards; CoAuthor supplies
recorded writing interactions. Tell Me a Story supplies human fiction paired with
detailed requests. These address different parts of our dataset. None of the
resources reviewed supplies our complete combination of branching fiction,
project files, author decisions, executable tools, and prose evaluation.

This is a source and method inventory. Release pages, documented schemas, and
evaluation protocols were inspected; bulk datasets were not downloaded or
benchmarks executed. Access notes describe published distribution routes, not
verified successful downloads. Recommendations below are project-specific judgments.

## Stories, notes, and authoring interactions

### Storium: the closest match for existing notes

The project describes roughly 6,000 collaborative stories and 125 million tokens,
with natural-language cards for characters, challenges, and strengths. These cards
condition scene continuations. This is genuine author-provided context, useful
for turning structured story state into workspace files. It differs from a
single novelist's outline or wiki. The release uses a research data agreement
that restricts redistribution; its terms need review before incorporating it into
a shareable dataset. The project page appeared in search, but direct opening
timed out during this review. [Project and access agreement](https://storium.cs.umass.edu/)

The evaluation puts generated continuations in front of actual authors, who edit
them and rate relevance, fluency, coherence, and likability. USER measures the
fraction of generated text retained using matching contiguous spans containing
non-stopwords. It requires a real edited result; an LLM simulator's edits would
measure a different population's preferences. [Paper, sections 4–5 and appendix C](https://aclanthology.org/2020.emnlp-main.525.pdf)

### CoAuthor: accepted suggestions and revisions

Contains 1,445 sessions from 63 writers, including 830 creative-writing sessions.
Timestamped editor events capture requests for suggestions, selection, dismissal,
insertion, and deletion. Downloads and replay tutorials are linked from the
project page. The page reports suggestion acceptance and human contribution
rates; this is an interaction dataset rather than a single standardized literary
quality score. Dataset reuse terms were not established from the page reviewed.
[Project, data, and replay tutorials](https://coauthor.stanford.edu/)

Use it to study realistic acceptance and editing behavior. Converting editor
events into conversational file-tool actions requires an explicit transformation;
the converted calls must be labeled as reconstructed.

### Tell Me a Story: detailed requests and human fiction

The released schema contains `example_id`, `inputs`, and `targets`, with train,
validation, and test files. It does not document intermediate workshop notes or
revision histories as released fields. The approximately 3 MB JSONL release is
encrypted to discourage scraping; the repository supplies the decryption keys
and procedure. The dataset is explicitly CC-BY-4.0, despite the repository's
Apache license badge. [Release and schema](https://github.com/google-deepmind/tell_me_a_story)

The paper reports 123 training, 52 validation, and 55 test examples. Expert
evaluation uses pairwise preferences for plot, creativity, development, language
use, and overall quality, including ties. It aggregates comparisons with a
Bradley–Terry model and reports inter-rater agreement. Automatic diagnostics
include ROUGE-L, BERTScore, length, vocabulary diversity, and trigram repetition;
LLM judging follows the human dimensions. This offers a compact prose baseline
and human reference material. [Agents' Room paper, sections 4 and 6](https://arxiv.org/html/2410.02603)

### WritingPrompts: prompt–story pairs at scale

The original release provides paired prompt/story files with train, validation,
and test splits. The paper's training recipe uses the first 1,000 words, while
the download retains full stories. It supplies prose and premises, with no
documented author notes or tool sessions. The official README links the archive;
the code license alone does not settle the underlying Reddit text's reuse terms.
[Original release instructions](https://github.com/facebookresearch/fairseq/blob/main/examples/stories/README.md)

For our work, use it as a possible source of contemporary prompt-conditioned
prose. Reconstructed notes must be labeled synthetic. HANNA and LitBench below
provide complementary evaluation annotations involving WritingPrompts material.

### PG-19: a prepared Gutenberg source collection

Provides full books published before 1919, title/date metadata, and splits of
28,602 training, 50 validation, and 100 test books. Its documented benchmark is
word-normalized perplexity: likelihood of existing text, not the quality of a
new continuation. The release strips boilerplate and replaces a specified set
of offensive words with placeholders, so it is not an untouched transcription.
The repository publishes download instructions and Apache-2.0 dataset metadata.
[PG-19 release](https://github.com/google-deepmind/pg19)

Filter for fiction and retain book identity before building checkpoints. Compare
its preprocessing with direct Gutenberg editions before choosing a source route.
Preserve the original split, and group related works/authors where needed.

### DOC: generated plans and controlled realization

DOC generates hierarchical plans before stories. It releases controller training
data and generated stories with human annotation results. Human comparisons
assess story properties such as coherence, relevance, and interest. It is useful
for studying plan-to-prose workflows; its generated outlines should not be
treated as historical author notes. The repository documents downloadable ZIPs
and links a newer implementation. Underlying asset terms need separate checking.
[Code, data, and outputs](https://github.com/yangkevin2/doc-story-generation)

## Prose benchmarks and judge-validation datasets

| Resource | Data and evaluation | Use and limitation |
|---|---|---|
| [WritingBench](https://github.com/X-PLUG/WritingBench) | Current documented release: 1,000 requests across six domains and 100 subdomains, with five request-specific criteria scored on a ten-point scale by an LLM or critic. | Useful for style, format, and material-dependent writing. Select literature/art tasks for a fiction subset. Original release had 1,239 requests; judge and criteria versions changed. Pin both. Public queries and scripts; Apache-2.0 repo. |
| [EQ-Bench Creative Writing v3](https://github.com/EQ-bench/creative-writing-bench) | 32 prompts, three generations each; rubric judging followed by pairwise comparisons and a modified Glicko-2 rating. Current README recommends Sonnet 4.6 for leaderboard parity. | Ready prose comparison through an API. Published scores depend on the judge, historical comparison pool, generation settings, and normalization. Pairwise comparison truncates to 4,000 characters and swaps order; it is not whole-session or roleplay evaluation. Public prompts/scripts; reuse license not confirmed here. |
| [HANNA](https://github.com/dig-team/hanna-benchmark-asg) | 1,056 stories across 96 prompts, three raters and six dimensions: relevance, coherence, empathy, surprise, engagement, complexity. Current release adds scores from 72 automatic metrics and four LLM judges. | Validate a judge or metric against human ratings, using correlations per dimension. CSVs are in the repository; MIT repo license. Older generator outputs may not represent current Gemma errors. Preserve the distinction between the original COLING branch and updated release. |
| [LitBench paper](https://arxiv.org/abs/2507.00769) and [release code](https://github.com/drfein/LitBench) | 43,827 training pairs and 2,480 held-out story comparisons. Evaluates preference-prediction accuracy; paper reports 73% for its best tested off-the-shelf judge and 78% for trained reward models. | Tests the judge's agreement with preferences, not a writer's generation ability directly. Repository documents Reddit rehydration requiring API credentials. Check recoverable records, label construction, and data terms before ingestion; availability is not guaranteed by the paper's counts. |
| [TTCW](https://huggingface.co/datasets/Salesforce/ttcw_creativity_eval) and [paper](https://arxiv.org/abs/2309.14556) | Stories with annotations for 14 binary tests in fluency, flexibility, originality, and elaboration. | Source of explicit creativity criteria and human labels. Report per-test results and agreement when replacing people with an LLM judge. A human-designed rubric does not validate an automatic judge by itself. Dataset card links the source repository. |
| [LiteraryTaste](https://github.com/mj-storytelling/LiteraryTaste) | 2,000 pairs of snippets; 60 participants each judge 100 pairs and supply stated reading preferences. CSVs link texts, annotators, and preferences, including uncertainty. | Useful for checking preference personalization instead of assuming one universal style ranking. Measure held-out preference prediction per reader. Repository documentation is incomplete and an explicit reuse license was not found on the reviewed page. |

WritingBench and EQ-Bench score new generations. HANNA, LitBench, TTCW annotations,
and LiteraryTaste let us test the scoring mechanism itself. Keep evaluation
annotations out of judge training unless using their designated training split.
Shared source material across these datasets also needs deduplication.

## Instruction following and tools

| Resource | Evaluation method | Relevance and limits |
|---|---|---|
| [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | Executable checks for verifiable instructions; reports instruction-level and whole-prompt accuracy under strict and loose checking. [Paper](https://arxiv.org/abs/2311.07911) describes 25 instruction types. | Good for count, keyword, and format constraints. Cheap scoring without a judge; does not establish semantic continuity or literary merit. Prompts and verifier code are released. |
| [IFBench](https://github.com/allenai/IFBench) | Adds 58 out-of-distribution constraint types and verifiers; paper generally reports prompt-level loose accuracy. Optional two-turn version separates the initial request from its added constraint. | Useful transfer check after instruction training. Current Python package includes these and classic IFEval verifiers. Separate IF-RLVR training data is linked; test constraints remain evaluation material. Apache-2.0 repo. |
| [FollowBench](https://github.com/YJiangcm/FollowBench) | Progressively adds content, situation, style, format, and example constraints. Hard Satisfaction Rate requires all constraints; Soft Satisfaction Rate averages constraint satisfaction within prompts. Strong LLMs judge open-ended constraints. | Closer to creative briefs than keyword checks alone. Judgment quality needs inspection; this is not a live file environment. English/Chinese data and evaluation scripts are released under an Apache-2.0 repo. |
| [Multi-IF](https://github.com/facebookresearch/Multi-IF) and [paper](https://arxiv.org/abs/2410.15553) | 4,501 three-turn conversations across eight languages, extending IFEval-style checks. Evaluate constraints across the sequence and inspect results by turn. | Tests whether instructions survive follow-ups. Supplied API and local-inference runners; Apache-2.0 repo. Does not directly test our accepted-canon semantics. |
| [AgentIF](https://github.com/THU-KEG/AgentIF) | Constraints from 50 agentic applications, averaging 11.9 per instruction. Constraint Success Rate and all-constraints Instruction Success Rate use code, LLM, or hybrid evaluation per label. | Useful schema for attaching a checker to each requirement. Agentic instructions alone do not establish correct live filesystem behavior. Data and evaluator configuration are documented; full data terms still need inspection. |
| [BFCL multi-turn](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) | Runs tools and checks both backend state and required execution results after each turn. A case passes only if all turns pass both checks; forced termination fails. | Strong precedent for checking state changes and read-only retrieval separately. Its required-path checks need care where several tool strategies are valid. Pin the dataset version; V3 multi-turn and later BFCL categories are distinct. |

Strict IFEval checks the original response; loose evaluation tolerates specified
formatting variants. Preserve the official distinction when reporting benchmark
results. For our custom constraints, define the denominator explicitly: pooling
all constraints gives long, heavily constrained prompts more weight than averaging
per-prompt satisfaction. Report whole-session success separately from partial
constraint success.

## Distribution and diversity metrics in our plan

The [Deft research post](https://deftwriting.com/research/distribution-fine-tuning)
provides a concrete recipe for the n-gram L2, embedding MMD, and JMQ combination:
normalized token n-gram frequencies; an embedding model with an RBF kernel;
and model-versus-human pairwise judging. It also uses self-BLEU across outputs.
Treat this as an author-reported experiment requiring replication.

The post defines JMQ as twice the model win rate, but also describes its range
as 0–1. Mathematically, twice a win rate ranges from 0–2. Our implementation should
report the raw win/tie/loss rates and define tie handling before adopting that name.
Its distribution fine-tuning also should not be equated with the separately
named Dynamic Fine-Tuning objective in [TRL](https://huggingface.co/docs/trl/sft_trainer). The source does not settle our proposed
same-prompt diversity function `D(n)`.

[MAUVE](https://github.com/krishnap25/mauve) offers a published distribution-comparison
metric and Python implementation. It embeds samples and compares distributions
through clustering and divergence calculations. It is a candidate diagnostic
alongside MMD, rather than another label for the same calculation.

For our pilot, specify the following before interpreting any distribution score:

- One fixed evaluation tokenizer for n-grams, the n-gram orders, and normalization.
- One embedding model/revision, text extraction rule, and length treatment.
- For MMD, kernel bandwidth and estimator, including whether reporting MMD or MMD².
- Matched genre, requested style, passage type, and sample counts across models.
- A human-versus-human split comparison to estimate ordinary corpus variation.
- Separate repetition within passages, similarity across unrelated prompts, and
  diversity among branches of the same prompt.

Low distribution distance can reflect topic or stylistic conformity. High
diversity can reflect incoherent output. Neither is a sufficient quality score
for our alternative-story tasks. These are interpretation limits, not reasons
to discard the diagnostics.

## Recommended first collection and evaluation

This is the proposed application of the evidence to our project:

1. Inspect **Tell Me a Story** examples and **PG-19/Gutenberg** book metadata for
   the first prose and checkpoint records. Inspect **CoAuthor** event samples
   for authentic authoring context. Storium is excluded from acquisition. Keep raw sources and
   reconstructed notes distinguishable.
2. Use **IFEval** for reproducible mechanical checks and **FollowBench/AgentIF**
   as references for annotated semantic constraints. Use **Multi-IF** to inform
   retention tests. Preserve official benchmark runs separately from adaptations.
3. Pilot **WritingBench-style request-specific rubrics** and blinded pairwise
   judging. Test judge agreement on **HANNA/LitBench** and our own reviewed
   branches. Consider **LiteraryTaste** for later style personalization.
4. Evaluate the actual writing artifact, whether in chat or a file. Add our
   project-specific checks for branch history, accepted decisions, edit scope,
   wiki links, and unauthorized updates. The reviewed datasets do not replace
   these live harness tests.
5. Add distribution metrics after the reference corpus and sampling protocol
   are fixed. Gather enough repeated samples to distinguish model differences
   from sampling noise; choose counts from the pilot rather than inventing a
   universal minimum.

On the 3090, generation, local judging, and embedding extraction can run in
separate passes. Reuse saved outputs when changing scorers. External judge calls
are optional and should be budgeted after a small pilot; no throughput or cost
estimate has been measured here.

The next deliverable is a small inspected sample pack with source identifiers,
split assignments, documented terms, and a mapping into our scenario format.
Remaining uncertainties are archive accessibility, exact reusable fields,
cross-dataset overlap, and judge reliability on our branching fiction.
