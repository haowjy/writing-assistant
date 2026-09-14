# Broad and task-selected reference comparisons

Both tracks are calculated for the saved thinking-enabled E2B-IT pilot. The
[side-by-side report](../../runs/pilot-e2b-it-thinking-2026-09-13/rescored-dual/review.md)
contains per-task results; [numeric summary](reference-comparisons.json) is versioned
with the code. No candidate generation or LLM grading was run.

## Reference selections

The broad track retains the original nine Gutenberg paragraphs from Frankenstein,
Sherlock Holmes, and Alice. Its values are unchanged.

The second track uses [nine frozen passages](../../data/references/pilot-matched-v1.json)
selected and read by the assistant. Source URLs, authors, development roles, text
hashes, character spans, scenario hashes, match dimensions, and limitations are
recorded in that manifest. Human acceptance remains pending. Selection happened
after inspecting outputs, so these are developmental diagnostics, not preregistered
final-evaluation targets.

| Task | Selected reference | Match and limits |
|---|---|---|
| F1-01: literary prose | Joyce, *Dubliners*, “Eveline”; 254/233/350 words | Literary fiction, concrete domestic detail, close third person, short-scene length band. Reflective exposition and historical register differ from the requested staged disagreement. |
| F2-06: comedy scene | Wilde, *The Importance of Being Earnest*; 369/379/384 words | Comedy, social embarrassment, character dialogue and stage directions, length near the produced script. Victorian setting differs; format selection follows the model's defensible interpretation of the loose request. |
| F5-05: fantasy scene | Carroll, *Alice*, trial passages; 187/176/203 words | Fantasy courtroom, narrative and dialogue, requested length. **Partial style match:** comic narrator differs from formal, precise close-third prose. |
| F3-03 / F4-08 | None | No designated fictional prose. Both tracks retain not-applicable prose metrics; their quality/resource scores remain present. |

Sources: [Dubliners](https://www.gutenberg.org/ebooks/2814),
[The Importance of Being Earnest](https://www.gutenberg.org/ebooks/844),
and [Alice](https://www.gutenberg.org/ebooks/11). The two newly selected works were
downloaded locally with their Gutenberg license text. Each selected passage was
checked against its parent source and stored span. They are reference material,
not additions to training data or paired gold answers for ROUGE/BERTScore.

## Results and interpretation

| Case | Broad unigram L2 | Task-selected unigram L2 |
|---|---:|---:|
| F1-01 | 0.106987 | 0.116983 |
| F2-06 | 0.094718 | 0.090593 |
| F5-05 | 0.108416 | 0.102627 |

Both bigram and trigram distances are saved alongside these values. Each prose
case has one independent output, so unbiased per-task MMD remains
`insufficient_samples` in both tracks. Paragraphs are not promoted to independent
candidate generations to manufacture a per-task MMD score.

Pooled exploratory MMD² is **0.115152** against the broad corpus and **0.075761**
against the task-selected corpus. Both use the same pinned features and RBF
bandwidth, **1.126177412218949**, determined from broad reference embeddings.
The latter pools the three outputs against the union of their selected references;
it is not a stratified, task-conditioned MMD estimate. The outputs mix genres and
formats, and each reference group shares a work/author. No significance or
confidence claims are supported. A smaller value does not establish better writing.

Markdown, including script labels and headings, is preserved exactly as in the
previous measurement view. The comparison does not silently change extraction or
resolve its known metadata limitations.

## Reproduce

```python
from pathlib import Path
from scripts.rescore_pilot import run

run(
    Path("runs/pilot-e2b-it-thinking-2026-09-13"),
    matched_manifest=Path("data/references/pilot-matched-v1.json"),
)
```

This reuses cached features and writes `rescored-dual/`, leaving the earlier
`rescored/` report intact. All 13 prose metrics are present in each task's two
`reference_tracks`; Q/R scores are retained at task level. The manifest is rejected
if source text hashes, scenario assignments, or development provenance are invalid.

Validation: 44 unit tests passed; lint and Markdown link checks passed. Runtime
rescoring completed with no feature errors. All nine exact source spans were
verified, and broad results match the previous run.
