# Changelog

## [Unreleased]

- Restore context records as strict `WireRecord`s, derive their closure edges from one field declaration,
  and unify runtime, evaluator, and group payload codecs with their record classes. Classify malformed
  on-disk records as corruption.
- Restore direct tests for admission, scripted-author, budget, intake, quiescence, gate,
  identity, and on-disk transition-semantics guards; pin context and commit identities.
- Unify task-graph writer, scripted-author, check, context, and group execution behind one
  verified rollout core. The driver gathers typed, allowlisted inputs; one pure derive per
  input produces one event and complete successor state for both publication and replay.
- Require a semantic verifier when constructing a store, pin runtime lineages to
  `task-graph-derive-v1`, and resume published runs through verified heads.
- Remove the store's workspace restore/materialize/diff helpers; checkpoint persistence stays
  structural and publication or gate views perform semantic verification.
- Use `step_input` as the sole verified environment-to-port path.
- Store context as immutable content chains and source-linked revisions. Ordinary message
  appends belong to their source event; explicit carry, seed, drop, and compact operations
  derive directly from the active view and account for context budgets.
- Remove the discarded compaction evidence path and unused group worker-root surface; resolve
  group entry contracts from the verified view alone.
- Keep check batches, transitions, terminal status, reward, and training eligibility in one
  evolving `OutcomeV1`; groups start from verified views, derive receipts from ancestry, and
  collect without a redundant policy walk.
- Remove the separate runtime log, generic patch replay path, and legacy task-graph runtime
  modules. Existing evaluation CLI behavior is unchanged; native on-policy optimization
  remains future work.

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
