# Changelog

## [Unreleased]

- Freeze the approved 32-output Creative Writing baseline, enforce its $2 grading
  cap, and report subset scores separately from the full benchmark.

- Add resumable external generation, official IFEval scoring, and an isolated
  EvalPlus runner for the 32-task coding diagnostic. Pause paid prose grading
  while the baseline subset is reconsidered.

- Preserve omitted Creative Writing rubric criteria as unscored, matching the
  upstream instruction to skip inapplicable criteria.

- Add a Python-driven Creative Writing v3 runner, saved rubric judgments, and an
  Anthropic spending ledger with request reservations and cached responses.
- Prepare the authorized 50-case E2B-IT run and remove an unnecessary supernatural
  qualifier from historical-fiction context; preserve previous pilot artifacts.

- Add the Python research evaluation suite, local Transformers/PEFT inference,
  constrained agent workspace, artifact extraction, scoring, and saved-run reports.
- Prepare 50 development scenarios with ten genres and explicit/loose instructions;
  record source lineage, review materials, coverage, and current/deferred work.
- Keep conversational replies as ordinary text and use Gemma native tool calls for local instruction-tuned inference.
- Run and preserve a five-case Gemma E2B-IT pilot, with per-case review pages,
  mechanical scores, and a versioned results summary.
- Show trace-recorded tool schemas in pilot reviews and distinguish harness history
  from the fully rendered model prompt.
- Render native Gemma tool declarations and responses with the checkpoint template,
  parse its native output grammar, and capture actual model inputs and raw outputs.
- Verify an E2B-IT native read/write round trip; preserve its trailing-newline
  copying error separately from successful tool execution.
- Enable thinking explicitly in Gemma IT experiment configurations, preserve it
  between native tool calls, and show it separately in pilot reviews.
- Default native Gemma chat inference to thinking on; record the five-case
  thinking rerun and a qualitative review of writing, grounding, and wiki accuracy.
- Rescore saved pilot outputs with pinned token/embedding features, n-gram distance,
  source overlap, and exploratory pooled MMD; retain every metric status per task.
- Fix MPNet feature chunking for the installed Transformers tokenizer API.
- Complete broad and task-selected pilot comparisons with frozen source passages,
  explicit match limitations, per-task tracks, and saved pooled diagnostics.
