# Changelog

## [Unreleased]

- Replace the task-graph runtime with one verified rollout core: one pure derive per input,
  one event and one checkpoint per commit, the same derive for publication and replay.
  The old writer, scripted-author, checks, terminal, replay and runtime-log paths are gone.
  The evaluation CLI is unchanged.
- Stores require a verifier and refuse lineages not pinned to `task-graph-derive-v1`.
  Forged content raises `ProjectionError` at its field path; disk damage raises
  `CorruptRecordError`. The store no longer restores, materializes or diffs workspaces.
- Storage grows linearly with history: context is a chain of appended messages, writer
  turns bind their messages through the context revision, and checkpoints store counts.
- One `OutcomeV1` carries checks, transitions, status, reward and training eligibility.
  Groups start from verified views and credit only sampled content.
- Goldens pin the v1 wire shapes and a deterministic rollout; changing them requires a
  semantics bump.
- Native on-policy optimization remains future work.
- Admission rejects a node whose checks require multiple evaluator families.
- Token-limited entries require a usage-reporting sampler manifest and support total-token caps.
- Group collection is bound to each member's verified published head; real groups pin a
  runtime manifest and cannot void a valid terminal reward.
- Add three public Phase 8 probe graphs with scripted author/feedback paths and optional
  deterministic rewards that produce distinct scripted score levels.

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
