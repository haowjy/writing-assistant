# Data inventory and acquisition

Inventory checked against this checkout on 2026-09-11. This plan distinguishes
local artifacts, externally published data, and material we must create. Dataset
release links below establish acquisition routes; they are not successful download
receipts. Selected small releases have been downloaded and compiled; no scraper is implemented.

## What we have locally

| Asset | Contents | Permitted role in the pilot |
|---|---|---|
| [Task fixtures](../../data/fixtures/tasks.jsonl) | Five synthetic smoke cases: names, retrieval, local edit, stale state, draft versus canon | Runner checks only; not evidence of model quality |
| [Response scripts](../../data/fixtures/responses.json) | Scripted backend outputs | Deterministic execution checks, not model generations |
| [Trajectory fixture](../../data/fixtures/trajectories.jsonl) | One pending synthetic format example | Schema demonstration; not accepted training data |
| [Research survey](../research-plan/dataset-and-metric-research.md) | Source descriptions and methods | Acquisition planning, not a corpus |

The selected acquisition now includes Tell Me a Story (230 examples), HANNA
(1,056 annotated stories plus 576 additional generated stories), IFEval (541 tasks),
and three Gutenberg books. The shared catalog has 2,742 artifact records, including
10 authored synthetic worlds and separate prompt/reference records. Counts are not
independent sample counts: variants and references share source groups.

The 50 proposed scenarios and private labels are prepared, with human review pending.
Raw text and compiled catalog live in git-ignored data directories. The checked-in
[inventory](inventory.json) records acquisition receipts, hashes, counts, and fields.
No real-author KB collection, calibrated judgment set, or Gemma baseline exists.

## External sources to inspect and gather

The first acquisition pass samples small amounts and records actual availability,
terms, schema, source identity, and overlap before bulk collection. The broader
request to collect accessible datasets remains a backlog; running every downloaded
dataset as another benchmark is not part of the first suite.

| Source | What it supplies and intended role | Acquisition route / remaining work |
|---|---|---|
| [Tell Me a Story](https://github.com/google-deepmind/tell_me_a_story) | Detailed prompts and human fiction; F1 material | Acquired/decrypted all 230 examples with published keys; CC-BY-4.0. Original train/validation/test labels retained. |
| [PG-19](https://github.com/google-deepmind/pg19) / Gutenberg editions | Long fiction for checkpoints, KB construction, and alternate continuation | Acquired Gutenberg books 84, 1661, and 11 with original headers and raw hashes. PG-19 acquisition remains optional. |
| [WritingBench](https://github.com/X-PLUG/WritingBench) | Writing prompts and request-specific criteria | Sample literature/art tasks; pin release and judge criteria. Adaptations belong in the custom suite, not official leaderboard claims. |
| [HANNA](https://github.com/dig-team/hanna-benchmark-asg) | Human ratings of generated stories | Acquired annotation, additional-story, metric-score, and user-study CSVs. Judge validation remains separate from final custom evaluation. |
| [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | General instruction-following regression check | Acquired 541 prompts and main verifier files at a pinned revision. Official strict/loose benchmark execution remains deferred. |
| [BookWorm](https://github.com/apapoudakis/BookWorm) | Books paired with character descriptions and analyses | Inspect reconstruction scripts and archived annotation URLs; literature-site annotations are not an unrestricted redistributed corpus. Strong contamination concerns for familiar works. |
| [Re-DocRED](https://github.com/tonytan48/Re-DocRED), [ToMBench](https://github.com/zhchen18/ToMBench) | Relation and interpretation diagnostic examples | Public releases; preserve split and conditions. ToMBench explicitly requires evaluation-only use. Optional component diagnostics, not mandatory full runs. |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | Memory/update cases with evidence and answers | Published data; inspect a small sample before committing to long-context runs. Does not supply narrative wikis. |
| [STORM](https://github.com/stanford-oval/storm), [ALCE](https://github.com/princeton-nlp/ALCE) | Organization methods, retrieval corpora, grounding checks | Methods/data inspection; article answering differs from creative continuation. |
| [CoAuthor](https://coauthor.stanford.edu/) | Actual editor events and accepted/rejected suggestions | Published downloads; check terms and event schema. Tool trajectories reconstructed from events must be labeled reconstructed. |
| [WritingPrompts](https://github.com/facebookresearch/fairseq/blob/main/examples/stories/README.md), [DOC](https://github.com/yangkevin2/doc-story-generation) | Prompt/story pairs; generated plans and outputs | Published archives. Verify asset terms; avoid unnecessary model weights. Generated plans are not historical author notes. |
| [LitBench](https://github.com/drfein/LitBench) | Preference comparisons | Reddit rehydration route; credentials, deleted records, and terms may limit recovery. Optional judge-validation expansion. |
| [EQ-Bench](https://github.com/EQ-bench/creative-writing-bench), [TTCW](https://huggingface.co/datasets/Salesforce/ttcw_creativity_eval), [LiteraryTaste](https://github.com/mj-storytelling/LiteraryTaste) | Prose prompts, creativity tests, reader preferences | Published repositories/cards; inspect exact assets and reuse terms before compilation. Optional expansion. |
| HumanEval+ diagnostic subset | Proposed HumanEval/0–31 before/after regression set | Execution remains deferred; pin EvalPlus revision and isolated runner before acquisition or execution. |

Storium is excluded from the acquisition queue because access is unavailable for
this project. Its paper remains related work. No milestone depends on it.

## Webnovels and existing wikis

Serialized webnovels can supply chapter boundaries, evolving relationships,
long-distance dependencies, and prose closer to contemporary serial fiction.
Existing companion wikis can supply examples of human selection and navigation.
They are candidates to investigate, not already licensed or acquired datasets.

| Candidate source | Material to seek | Acquisition decision |
|---|---|---|
| Author-provided webnovel exports | Chapters, revision dates, optional outlines and private story bibles | Prefer EPUB/Markdown/HTML exports with explicit intended uses recorded. Author outreach is a separate task; no messages have been authorized or sent. |
| Independently hosted, explicitly licensed serials | Ordered chapters and author notes | Identify titles and verify both reuse terms and permitted automated access. Use feeds/downloads first; implement a small site-specific scraper only where appropriate. No eligible title is confirmed yet. |
| Royal Road | Contemporary serial fiction | Candidate requiring permission. Its [published terms](https://www.royalroad.com/tos), checked 2026-09-11, restrict automated scraping unless expressly permitted. Do not schedule a bulk scraper on the assumption that readable pages are reusable data. |
| Companion / fan wikis | Entity pages, timelines, indexes, links, revision IDs, source citations | Prefer exports or APIs where available. [MediaWiki export](https://www.mediawiki.org/wiki/Help:Export) supports page XML; each site's terms and content license must be checked separately. A wiki license does not grant rights to the underlying novel. |
| Private/unpublished writer contributions | Source scenes and actual author notes/KBs | Useful for a less exposed final evaluation. Record contributor permission, storage/sharing limits, and whether training is allowed. Not a dependency for the first synthetic development pilot. |

For an eligible website, preserve canonical URL, title, author, series, chapter
order, timestamps, source hash, acquisition method, and terms evidence. Cache raw
responses; identify the client, limit request rate, resume downloads, and log
missing/deleted chapters. Follow site access policies; do not bypass login,
paywalls, or technical restrictions. Separate story body, author notes, navigation,
and comments. Exclude reader profiles/comments unless specifically needed and
cleared. Strip presentation markup without flattening paragraph or dialogue breaks.

Retain wiki page IDs, redirects, link targets, section structure, revision IDs,
and attribution through Markdown conversion. Inspect conversion losses. Human
wikis are examples, not gold truth: check claims against source passages.
Full-series wikis often reveal later events; do not expose them to a chapter-cutoff
scenario without a reviewed cutoff-specific reconstruction. Publication date alone
does not establish the fictional time to which a statement applies.

## What we must create

| Artifact | Construction | Review needed |
|---|---|---|
| Task briefs | Write purpose-specific prompts, constraints, allowed invention, and output destinations | Ambiguity, feasibility, and comparable difficulty |
| Starting workspaces | Render source excerpts, notes, reviewed KBs, and controlled stale files | Correct files, source cutoff, no evaluator leakage |
| KB reference annotations | Label important information, evidence, possible interpretations, temporal/belief scope | Human salience and interpretation judgments; multiple acceptable placements |
| Hidden probes | Write retrieval/continuity questions independently of generated KBs | Answerability, importance for the brief, and correct evidence |
| Update sessions | Script accepted revisions, draft-only suggestions, changed beliefs, and author decisions | Exact update obligations and preservation requirements |
| Branching authorship examples | Generate alternate plots, styles, viewpoints, or continuations from a checkpoint | Canon, source provenance, diversity, and prose quality |
| Grader calibration set | Collect diverse development outputs and deliberate errors | Independent human labels and adjudication |
| Human-prose reference collection | Select held-out human passages matched by genre, style, topic, passage type, and length | Provenance, overlap, rights, and reference balance for D1/D2 |
| Format controls | Render the same reviewed content as flat/linked Markdown; later records/hybrid | Equal information content and valid links |

For reconstructed notes, label whether they use prior prose only or an explicit
future author plan derived from later text. Never conceal future-target access.
Synthetic author turns can make sessions reproducible; they are not evidence of
real author preferences. Final prose need not match one reference continuation.

## Compilation and contamination controls

1. Register sources and allowed uses before content enters a compiled release.
   Track downloading, transformation, training, evaluation, and redistribution
   separately; unresolved uses stay unapproved in the manifest.
2. Normalize stable identities and preserve raw bytes, source splits, and edition
   metadata. Detect exact and near duplicates across datasets, translations,
   serial mirrors, and related wikis.
3. Assign source groups before generating notes or variants. Keep a work/series
   and its derived wiki/branches together; group by author where possible.
   Cross-source tasks inherit the combined group. Conflicting upstream splits
   are quarantined or restricted to evaluation, not relabeled into training.
4. Separate training, development, and final evaluation. Keep public regression
   tests in a distinct collection. Grader calibration and prompt tuning use
   development only. Private labels never enter candidate-visible packages.
5. Validate and inspect the compiled sample: counts by family/type/source, missing
   evidence, duplicate groups, broken paths, and rejected records. Keep rejection
   reasons and transformation versions so compilation is repeatable.
6. Export accepted training trajectories only from training sources. Neither final
   evaluation tasks nor their judged outputs become training data. Pin final
   manifests before final runs and do not use final scores for checkpoint selection.

A useful common dataset is a catalog of typed, linked records, not a concatenation
that erases source meaning. Keep original task/split fields and transformation
lineage. Gutenberg, public webnovels, and freshly phrased questions about known
works may already be represented in pretraining. Mark exposure as unknown; add
fresh synthetic/private worlds for final evaluation and source-grounded changed-fact
controls. Such controls diagnose reliance on prior knowledge, not prove zero exposure.

## Provenance and research roles

One catalog holds artifact-level `human`, `synthetic`, `half_synthetic`,
`synthetic_fanfic`, or `unknown` labels. Half-synthetic means substantive human/model
coauthorship, not an exact proportion. Human review alone does not change synthetic
provenance; format conversion alone does not change human provenance. Preserve
parent sources and transformation history. Labels support metric slices and remain
independent of train/development/final-evaluation roles. Source grouping applies
across the whole catalog, including imports and derived branches.

Separately collected text, Markdown, HTML, and EPUB stories can enter through the
same import API. Optional webnovels need not be ideal prose references: use them
for narrative knowledge, updates, and continuity where appropriate. Synthetic
branches require quality review too.
