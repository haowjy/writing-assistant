# Implement a Python-driven research evaluation suite

Implementation status and checks: [delivery evidence](delivery.md).

## Outcome and execution boundary

Build a reusable Python library with an editable research script as its main entry
point. Prepare 50 development scenarios (10 per family), data preparation, artifact
extraction, scoring, and reports without running the full experiment.

Authorized during implementation: basic tests, data downloads, and at most three
small Codex grading calls. Candidate generation remains unexecuted. The full
200-attempt comparison, repeated generations, navigation-reader experiments, and
full external benchmarks require explicit user approval. Do not automatically run
end-to-end evaluation after implementation. Human review is a delivered packet,
not an assumed completed calibration exercise.

## 1. Python interfaces and experiment script

Follow dev-principles: cohesive modules, shared behavior, explicit dependencies,
and minimal abstractions. Expose Python functions for source import/cataloging,
scenario compilation, selected execution, prose extraction, measurements,
judgments, and reports. Accept Python configuration and return structured results.
Library functions do not print, exit, change working directory, or download on import.

Use `scripts/evaluate.py` to select sources, provenance, models, conditions,
metrics, and output locations in readable Python. Preparation, generation, grading,
and reporting are independently callable. Default execution inspects expected
work. Explicit subsets and saved outputs support small experiments; resumability
and caching live in the library. Preserve existing CLI workflows; no new CLI is needed.

## 2. Data and scenario preparation

Download and inspect Tell Me a Story, HANNA, IFEval, and a small Gutenberg fiction
collection. Import separately gathered text, Markdown, HTML, and EPUB. Webnovels
and companion wikis are optional; scraping is not required.

Use one catalog with artifact-level `human`, `synthetic`, `half_synthetic`,
`synthetic_fanfic`, and `unknown` provenance. Preserve parents, transformations,
authors, works, revisions, hashes, upstream splits, and reuse terms. Provenance is
independent of train/development/final-evaluation roles. Group works and derivatives
before generation and compute metric slices from labels.

Prepare five initial cases, then 50 authored cases across direct prose, file
writing/revision, brainstorming, KB construction/maintenance, and KB-based writing.
Specify visible context, tools, budgets, follow-up turns, destinations, and private
labels. KB briefs specify purpose and detail; labels allow defensible selections
and interpretations. Deliver scenario and label review materials for the user.

## 3. Execution and prose extraction

Extend the runner with configurable tools, follow-ups, snapshots, budgets, and
resumable attempt records. Keep evaluator material outside candidate workspaces.
Extract designated prose-only replies, selected mixed-reply spans, manuscript
files, or requested revision spans. Save locations, hashes, versions, and
uncertainty; exclude commentary, plans, KBs, and tool traces. Count duplicate
reply/file prose once.

Support KB construction, reviewed-KB writing, and fresh-session generated-KB use.
Linked and flat Markdown controls share identical content for format-only tests.

## 4. Scoring and prose profiles

Independent passes cover mechanical constraints/files/edits/tools/links/evidence,
Astra judgments with evidence and uncertainty, and the [prose catalog](prose-metrics.md).
Cache counts and embeddings by prose/configuration hashes. Group results by
provenance, task, style, and length. Pin Gemma tokenization and
`sentence-transformers/all-mpnet-base-v2` revisions; keep preprocessing explicit.

Implement n-gram L2, MMD squared, lexical diversity, repetition, overlap, structure,
repeated-output diversity, and applicable ROUGE/BERTScore. Register MAUVE,
perplexity, and author retention with unavailable reasons until prerequisites exist.
Numeric profiles are secondary targets alongside quality and continuity.

Prepare evidence-backed human-review packets. Use at most three small Codex calls
for integration verification; do not start the calibration batch automatically.

## 5. Validation and handoff

Use scripted fixtures to check API/script composition, source grouping and
provenance, private-label separation, tools/follow-ups/budgets, fresh sessions,
prose extraction, known numerical inputs, resume/cache behavior, and rescoring
without regeneration. Mechanical grading works without Astra or optional models.
Run existing tests, lint, and documentation checks. Supplied test embeddings do
not require downloading a model.

Deliver the script/library, acquisition inventory, 50 scenarios, metric catalog,
review materials, and limitations. Provide a proposed manifest for four checkpoints,
200 core attempts, separately counted repeats/probes/external benchmarks, and
estimated cost. Cost estimates must distinguish arithmetic from measured throughput.
No training, publication, full evaluation, or model-quality claim is part of delivery.

## Local execution and training checkpoints

Use the [direct Transformers/PyTorch runtime](../../docs/local-inference.md) through
the existing harness. Evaluate saved full checkpoints or PEFT adapters with one
loaded model per serial batch. Future training can call the same backend with
resident weights between optimizer steps, or schedule evaluation from completed
saves. Define a fixed development subset and cadence before enabling that schedule;
final evaluation stays outside checkpoint selection. No full training loop or
automatic benchmark callback is part of this implementation.

Proposed training cadence, not enabled: run a fixed five-case development subset
(one per family) before training and every 500 optimizer steps; use the full
50-case development suite at selected milestones. Measure mechanical outcomes,
prose profiles, token usage, and time at each check. Grade selected milestones
with Astra under a separately set budget. Keep checkpoint selection based on
quality and task performance alongside prose variability, never one distribution
number. Choose the actual interval after measuring evaluation overhead.

## Coverage review before final data generation

Use the [current coverage inventory](coverage.md) as the starting audit. Review
genre, naming, narrative form, character relationships, instruction specificity,
knowledge complexity, and KB representation against the actual inputs. Set targets
and review the resulting coverage during final data generation. The inventory
records existing fixtures only; external-dataset integration and further generation
are deferred.
