# SFT seed dataset

Created 24 draft trajectories: 19 assigned to training and five to validation.
This is a format and curation seed, not the substantive SFT training collection.
There are only eight independent source works/projects, and the validation examples
all come from one book. Repeating these records for more epochs would not fix that.

| Source | Direct prose | File revision | Planning | Build/update KB | Write from KB | Total |
|---|---:|---:|---:|---:|---:|---:|
| Tell Me a Story, upstream train | 4 | 0 | 0 | 0 | 0 | 4 |
| Gutenberg openings and derivatives | 2 | 2 | 2 | 2 | 2 | 10 |
| Newly authored synthetic projects | 2 | 2 | 2 | 2 | 2 | 10 |
| Total | 8 | 4 | 4 | 4 | 4 | 24 |

The collection has 18,830 native Gemma input tokens, including context, and 7,798
assistant loss tokens. The longest trajectory is 1,551 tokens; none were truncated.
All records remain `review_status: pending`. Tool replay, links, labels, and native
loss masks have been checked; no independent literary grader or training ran.

## Inspect and reproduce

- [Dataset JSONL](../../data/training/sft-v1.jsonl): complete messages, tools, and files.
- [Project seeds](../../data/training/starter-design.json): authored notes, alternatives,
  prose, accepted updates, and local revisions.
- [Selection and source inventory](../../data/training/starter-selection.json): source
  URLs, revisions/hashes, reuse terms, exclusions, and counts.
- [Readable examples](../../data/processed/sft-starter-v1/review.md): conversations and
  final files, generated locally.

The local `data/processed/sft-starter-v1/` directory also contains the source catalog,
three HANNA reference records, and `mask-audit.json` with native rendered conversations,
input IDs, labels, and loss-bearing text. Raw and processed downloads are Git-ignored;
the dataset, authored seeds, and source inventory are versioned.

Run `.venv/bin/python -m scripts.build_sft_starter` with the existing source catalog
at `data/processed/custom-eval/catalog.json`, the pinned Gemma tokenizer cached, and
Gutenberg downloads at `data/raw/sft-starter/{113,289}.txt`. Official download URLs are
`https://www.gutenberg.org/ebooks/113.txt.utf-8` and
`https://www.gutenberg.org/ebooks/289.txt.utf-8`; compare their byte hashes against the
recorded source inventory. Upstream files can change. Rebuilding must reproduce the
existing JSONL exactly; a changed dataset requires choosing a new `RECORDS` path.
The builder executes workspace tools and tokenizes locally, without model inference.

## Labels and source decisions

Each record carries its task family/type, source dataset, source group and parents,
upstream split, assigned split, genre/style, instruction specificity, review status,
and transformation history. Artifact metadata distinguishes human passages from
synthetic replies and files, with hashes. `author_id` identifies the originating
work's author; `generator` separately identifies the synthetic authoring session.
The session's exact model name was not recorded and is not inferred.

- **Tell Me a Story:** four complete upstream-train prompt/story pairs, retained
  verbatim under the repository's CC BY 4.0 attribution. Selection takes the shortest
  eligible pairs that fit the token limit; it does not select for literary score.
  Genre remains `upstream_unspecified` rather than inventing upstream annotations.
  Example 097 was excluded because its embedded song lyrics need separate reuse review.
  The source inventory retains the upstream repository URL and pinned revision.
- **Gutenberg:** Kenneth Grahame's *The Wind in the Willows* (#289, train) and Frances
  Hodgson Burnett's *The Secret Garden* (#113, validation). Both are listed as public
  domain in the USA by Project Gutenberg. The first two paragraphs define the visible
  authorship cutoff. F1 uses the original second paragraph; the other families use
  authored notes, alternatives, and new scenes. Later book text is not supplied to
  these tasks. Project Gutenberg terms and source spans/hashes are retained.
- **New synthetic projects:** an orbital hotel laundry and a village puppet theatre.
  They practice uncertainty, authorized decisions, local edits, and useful planning.
- **HANNA:** three locally cataloged stories with their existing human annotations,
  marked `evaluation_reference_only`. They are absent from SFT targets. Their evaluation
  role and unresolved underlying story reuse terms are preserved.
- **IFEval, Creative Writing v3, WritingBench, and coding benchmark cases:** excluded
  from this training dataset. Future instruction-following examples must be independently
  authored from skill categories, without copying held-out test prompts or targets.

Provenance counts are four `human`, six `half_synthetic`, four `synthetic_fanfic`, and
ten `synthetic`. F2 and F5 deliberately share an initial prose branch within each
project to exercise different delivery contexts; these are not independent stories.

All derivatives of a work stay in one split. Recorded custom-suite source IDs and
connected lineage groups, plus the previously cataloged Gutenberg prose-reference
works, are excluded. This is a recorded-lineage check, not proof of semantic
non-overlap or absence from a pretrained model. Anonymous TMAS authors prevent a
complete author-overlap audit. The single-book validation set cannot support broad
quality claims.

## Expand for the first substantive SFT experiment

Working target: **500–1,000 curated training trajectories**, plus a separate grouped
validation collection. This is an experiment-design starting point, not a proven
minimum or a guarantee of improvement. Freeze the count after measuring target-token
lengths, curation yield, and the bounded training test's throughput.

1. Broaden upstream-train human source selection and acquire more independent eligible
   works. Inventory source groups before assigning train/validation roles.
2. Build many distinct authorship checkpoints: supplied story passages, extracted notes,
   navigable KBs, alternate continuations, and accepted/rejected author decisions.
3. Cover all five families with varied genres, names, styles, lengths, prompt specificity,
   multi-turn revisions, and reply/file delivery. Avoid expanding by cosmetic renaming.
4. Execute tool traces, verify protected files and links, inspect native loss masks,
   and check source lineage and textual duplicates. Keep related task views grouped.
5. Grade target quality and instruction/continuity compliance with a fixed Astra rubric;
   revise or reject weak examples. Record that acceptance decision separately from
   mechanical verification. Synthetic authorship alone does not imply quality.
6. Freeze accepted records and token counts, then run the separately authorized GPU
   feasibility test before choosing full-run steps and checkpoint evaluation cadence.

The dedicated [next-experiment TODO](../research-plan/TODO.md) tracks this expansion.
