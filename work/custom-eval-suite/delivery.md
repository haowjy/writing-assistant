# Implementation and verification

The Python research suite is implemented. The default
[research script](../../scripts/evaluate.py) inspects the proposed experiment;
the full benchmark comparison remains unexecuted. A bounded Gemma E4B-IT
inference check has exercised prose generation and a real file write.

## Delivered

- A reusable Python API for source preparation, scenario compilation, selected
  execution, prose extraction, mechanical scoring, numerical profiles, judgments,
  and reports. The script calls these stages independently.
- A shared catalog of 2,792 records, with original splits, hashes, source groups,
  provenance, and acquisition evidence. [Inventory](inventory.json) records what
  actually downloaded; raw and compiled corpus text remain git-ignored.
- [Fifty authored scenarios](../../data/scenarios/development.json), ten per family,
  across ten synthetic worlds, with five loose and five explicit initial requests
  per family. [Review materials](review/review.md) contain briefs
  and private labels. All remain pending human review.
- Configurable tools, scripted follow-ups, read/storage budgets, per-turn snapshots,
  saved traces, explicit retries, and interrupted-attempt accounting. Candidate
  workspaces never receive private labels or grader artifacts.
- Prose selectors for replies, delimiters, reviewed spans, and manuscript snapshots;
  explicit duplicate-delivery groups and unavailable/ambiguous extraction states.
- Mechanical scorecards and quantitative features, including n-gram L2, unbiased
  MMD squared, lexical/structural/repetition profiles, repeated-output comparisons,
  and optional paired ROUGE/BERTScore. Counts and embeddings are independently cached.
- Blinded Astra packets, structured judgment validation, evidence and uncertainty,
  per-dimension prose ratings, and review records that preserve human edits.
- A proposed [experiment manifest](experiment.json) with four pinned checkpoints,
  200 core attempts, 48 extra repeated generations, 80 navigation-reader attempts,
  and separately counted external checks. Its cost ranges are hypothetical decoding
  arithmetic, not measured 3090 throughput.

## Verification

All 42 offline tests pass. Ruff lint/format checks and Markdown link checks pass.
The tests cover existing workflows and the research API, including all
five families with scripted backends, source-role leakage, private-label separation,
tool restrictions, follow-ups, budgets, fresh readers, prose extraction, cached
features, resume/retry, interrupted attempts, and known numerical values. Tests
also protect missing-prose handling, preservation of human review notes, completion
after judgment merging, imported-content consumers, and builder attribution on resume.

A separate saved-output walkthrough executes five scripted cases through the
research API, then calls the script's scoring and reporting stages. Its short
fixture prose deliberately fails word-count checks; these are harness outcomes,
not model-quality scores. See [validation records](validation.json).

Two small live Astra responses were obtained within three Codex process attempts;
the first process failed during sandbox initialization. Saved responses passed
local validation. One response was revalidated after allowing judge-selected
subdimensions when a rubric does not prescribe names. Rubrics that prescribe names
still require the exact set. Raw responses and failure records remain under
`runs/grader-integration/`; [judgment review](review/judgment-review.json) preserves
both samples. No calibration batch was run.

## Experimental prerequisites and limits

- The local Transformers runtime loaded Gemma E4B-IT in NF4 on the 3090. A short
  prose reply completed. A file task wrote the correct file, then failed because
  its final reply violated the JSON protocol. The base variants, 12B, longer
  context budgets, and full research settings remain unverified. Script model
  configurations retain that status.
- Scenarios are short synthetic drafts, not a publication benchmark. Human review
  must establish salience, acceptable interpretations, and rubric usefulness.
- The catalog contains human reference candidates, but a matched reference selection
  and MMD bandwidth have not been accepted. No measured prose-distribution results
  for Gemma exist. Optional embedding and BERTScore execution has not been tested
  with model weights; known-input numerical tests do not establish model compatibility.
- The default read budget uses labeled whitespace-token estimates. Model comparisons
  should supply and record the chosen read tokenizer. Embedding extraction uses CPU;
  training/evaluation scheduling and representative throughput remain unmeasured.
- MAUVE, candidate perplexity, and authentic author-retention scoring remain explicit
  unavailable entries. Repeated-output functions consume supplied saved samples;
  the implementation does not authorize generating those samples.
- Near-duplicate detection is heuristic. The acquired catalog's audit identifies
  repeated reference text; original work/author/series grouping is still necessary.
  No claim of zero pretraining contamination is made.
- IFEval data and main verifier sources are downloaded. External benchmark execution,
  EvalPlus acquisition/runtime verification, training, and scraping remain separate work.

The full comparison, repeated-generation experiments, navigation-reader runs, and
full external benchmarks still require user approval, as specified in the
[delivery plan](plan.md).

## Local inference verification

[Local inference](../../docs/local-inference.md) documents the Python runtime and
checkpoint integration. Transformers, PyTorch, bitsandbytes, Accelerate, and PEFT
are installed. The project uses uv-managed Python 3.12 with the headers Triton
needs. The E4B-IT checkpoint is cached locally; other baseline weights have not
been provisioned by this step.

A tiny randomly initialized CPU model verified full-checkpoint and PEFT-adapter
loading plus actual generation, without downloading another candidate. The GPU
smoke used two tasks, a 2,048-token context budget and 256-token output limit.
It produced three responses total: prose, a valid tool call, and a plain-text
confirmation rejected by the declared JSON protocol. The saved file contains
`The rain stopped.` Earlier failed attempts caused by missing Python headers are
retained separately. See [inference validation](inference-validation.json); raw
outputs are under `runs/validation/local-gemma/`. These checks establish runtime
behavior, not benchmark quality or representative throughput.

The final-response wrapper requirement was removed in `writing-tools-v2`. Ordinary
conversation stays plain text with tools enabled or disabled; only calls use JSON.
The existing GPU smoke records above used v1 and are retained as historical evidence.
The v2 file-tool loop is covered by offline tests; a new GPU pilot remains pending.

The 50 scenarios now mix ten genres across families and instruction specificity.
Each world retains five derived genre-context source records with parent lineage;
this adds 50 catalog records. Briefs and KB fixtures include the relevant context.
The [coverage matrix](coverage.md) records assignments and remaining limitations.
