# Changelog

## [Unreleased]

- State each Phase 8 probe detail in its task brief and run the probe and CPU fixtures without
  author-tool calls or feedback turns.

- Sort native tool-outcome reports by writer-action ordinal and call index.

- Name the outside-packet canary for its actual scope: detecting a bulk private-store dump,
  not leakage of an admitted evaluator check spec.

- Require every planted privacy canary to be present in `training/private` as criterion 6's
  positive control, in addition to checking for leaks outside it.

- Report per-member terminal stop reasons and total native parse-failed turns in probe
  measurements.

- Separate pure Phase 8 probe-task loading and admission from its trainer recipe and
  tokenizer-bound run composition.

- Commit truncated or malformed native tool output as parse-failed turns, including incomplete
  call headers and reserved sentinels, while keeping parser configuration errors fail-closed.

- Share one adapter tensor hash, bounded/unbounded u32 decoder and durable atomic file writer
  across native training and trace evidence.

- Route verified member completion, admission and trainer-consumption receipts through the
  locked group coordinator; share one strict group/step sequence index for refusal and resume.

- Plant Phase 8 privacy canaries in private task records and fail criterion 6 if either reaches
  run artifacts outside the store's private area.

- Treat only recognized Gemma malformed-response errors as model parse failures; surface
  unexpected parser errors as protocol failures.
- Refuse false `native_parse_failed` claims at decode.

- Build rollout callbacks per trainer invocation so the lifecycle no longer depends on legacy
  rollout classes.

- Derive reward scaling from the runtime profile, preserving legacy plan identities.

- Preserve the checkpoint fork's W&B console privacy and resume bindings without forwarding
  unapproved environment keys.

- Bind task-graph experiment identities to the native sampler, audit, and core task-graph sources.

- Label criterion 4's byte-tampering checks as store-integrity controls, not audit proof.

- Use one fail-closed verdict rule in both the task-graph probe worker and parent.

- Add the native task-graph GRPO route: one admitted group per step, all checkpoints kept,
  and fail-closed admission evidence before DAPO updates.

- Apply the declared 4,096-token context cap when compiling the Phase 8 fixture task graphs.

- Add a one-shot generic stage supervisor with process-group timeouts, resource ceilings,
  and atomic per-stage status and resource records.

- Add the bounded Phase 8 task-graph probe phases, offline evidence inspection and verdict,
  plus a tiny-Gemma CPU dry-run profile.

- Require both offline inspection rounds to cover all three groups before criterion 4 can pass.

- Keep `inspect` read-only and persist its pre-run record from `prepare`.

- Set only validated W&B environment bindings, and only after runtime admission succeeds.

- Derive TRL reward scaling from the runtime profile, preserving legacy plan identities.

- Bind native Gemma tool calls to deterministic task-graph action IDs, pair replayed results
  exactly, and halt the CPU trace check on protocol-shaped tool-result errors while recording
  per-call outcomes and final-file changes.

- Derive tool-call rejection codes from exact committed error text and make P1 criterion 1
  fail on protocol-shaped native-intake rejections in any member.
- Bind scripted CPU-probe tool IDs exactly like native sampling so the dry run exercises the
  committed call/result identity contract.

- Treat Transformers 5's unset neutral generation controls as no-ops while continuing to
  refuse non-neutral defaults; persist failed trace timings as canonical nanoseconds.

- Add an offline, CPU-only native Gemma task-graph trace checker that records group admission,
  on-policy drift, prefill and generation timing, peak RSS, deterministic offline inspection,
  and protocol-shape classification for audit refusals.

- Audit native training batches against committed context and pinned tokenizer files before
  training; offline inspection re-derives the batch and admission without network access.
- Offline inspection now persists and re-derives tokenizer-file hash refusals through the
  admission audit instead of stopping at tokenizer load.

- Derive V2 total, prefill and cached-input usage from the committed token ledger before
  charging a native turn.

- Keep overrun lineages out of native training eligibility; coordinator finalization only
  derives token spans for eligible members, and eligible-layout contradictions are invariant
  breaches.

- Export V2 advantages with a high-precision exact-expression conversion, and expose stable
  reason codes for training-export refusals.

- Share native token layout between group coordination and training export, and refuse V2
  runtime sessions outside group sampling.

- Attribute native rendering-pin failures to their rendering field rather than the adapter
  trace projection.

- Refuse native group sealing when the entry declares an aggregate total-token limit that
  native rule 3 does not derive; remove that limit from the Phase 8 probe tasks.

- Skip tokenizer/model-backed native tests when their optional dependencies are absent, while
  keeping dependency-free import checks active.

- Native token history and the u32 ledger codec now have shared owners. Context ancestry
  checks require the sample or rollout-start boundary to be present; structural eligibility
  treats every intervening context event as a multi-segment member.

- Share Gemma native suffix and stop-set rules through `native_protocol`; accept task-graph
  author replies and check-driven continuations only when their context delta is
  template-prefix stable.

- Add a native Gemma renderer and V2 sampler that rebuilds each decision from committed token
  evidence, records sampled logprobs and usage, and keeps model KV caches local to each call.

- Classify filesystem path mistakes consistently in both dispatchers: missing, invalid,
  overlong, or file-ancestor paths are recoverable tool errors; host faults remain
  infrastructure failures.

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
- Native group seals require a V2 manifest, all three native sampling capabilities, and
  descriptor pins that match both group policy and entry rendering; existing V1 group
  identities and wire records remain unchanged.
- Admission rejects nodes mixing evaluator families, with `legacy-check-v1` treated as its own family.
- Token-limited entries require a sealed usage-reporting sampler before sampling and support
  cumulative generated/total-token caps and a high-water context-token cap. Native sampler
  inputs expose only remaining generated allocation and the context cap.
- The V2 writer derive validates byte-backed token counts, logprobs, sample pins and prior-turn
  prefixes; it derives termination from committed limits and maps valid token/context limits
  and unterminated stop classes to scored incomplete outcomes. V1 identities and goldens stay
  unchanged.
- V2 group traces bind present policy claims, zero-generation context-limit turns carry no
  output, and termination-mapped outcomes retain their stop reason and candidate checkpoint
  in the committed terminal `OutcomeV1`.
- Reward publication derives ordered structural eligibility from native manifest, group,
  context and sampled-message evidence; the reserved `eligible` status is never emitted.
- Real group member starts and commits require the sealed manifest. Collection binds terminal
  results to the verified head, resolves rewards published after collection, and records the
  judged head for infrastructure interruptions.
- Add three public Phase 8 probe graphs with scripted author/feedback paths and optional
  deterministic rewards that produce distinct scripted score levels.

- Disable W&B console capture for fork training: native TRL scalar/system metrics
  and package metadata remain available, while stdout/stderr cannot become a
  synced `output.log` carrying task content or failure traces.

- Pin the complete local Python training runtime for checkpoint-31 fork admission
  and reject an interrupted slot-002 suffix at its actual on-disk location before
  GPU admission, rather than relying only on the later no-resample guard.

- Tighten checkpoint-31 fork admission: preserve infrastructure continuation
  failures as unavailable/GroupPending, reject mismatched active W&B runs,
  validate ordered fork schedules and complete checkpoint seals, canonicalize
  model-config keys, pin implementation hashes, require frozen W&B before
  Trainer options, and allocate unique ownership records for every launch.
  Phase preflight remains CPU-only and no GPU or W&B run was started.

- Add the separately identified checkpoint-31 CPU fork boundary.  The fork pins
  the sealed v2 checkpoint and stopped group-32 attempt hashes, imports slots
  000/001/003 byte-for-byte, and preserves slot 002 under an immutable source
  prefix before its native follow-up continuation.  External public-Trainer
  resume accepts the pinned optimizer/scheduler/adapter/RNG checkpoint without
  copying or resealing it.  Native W&B configuration binds an explicit run while
  disabling model/checkpoint uploads; no GPU or online training run was started.
  Fork coverage reports the verified v2 prefix separately from fork-local
  checkpoint-32 evidence and refuses to claim readiness before that seal exists.
  W&B logging identity is now bound in the trainer manifest and fail-closed
  unless an explicit run name accompanies native reporting.
  The guarded trainer options stop the first fork pass at checkpoint 48;
  extending to 96 remains an explicit second invocation.
  Source trace events are checked against the five saved slot-002 boundaries,
  and committed-prefix seals plus fork checkpoint group seals are rehashed before
  coverage can report readiness.  Trainer identity includes the fork manifest.
  Continuation now resumes from the verified generation-event count (five),
  never the one completed-turn count.
- Separate immutable imported attempt trees from fork-derived rewards/results;
  continuations now emit a complete source-plus-suffix scorer trace while
  retaining the suffix audit stream.  Add source/fork contract compatibility,
  preflight and explicit 48-to-96 resume options, pending-group coverage, frozen
  W&B environment enforcement, and a leased checkpoint-31 fork CLI.  No launch
  or online W&B run was performed.
- Bind the stopped group invocation to the same original experiment as checkpoint
  31, reject source-tree output paths, and validate task/seed provenance before
  any continuation.
- Freeze W&B online mode alongside the approved run/entity/project binding;
  inherited offline or mismatched settings now fail closed before Trainer setup.
- Ruff import checks pass for the fork module, CLI, and CPU regression tests.

- Add a guarded agent continuation boundary and native sampled-token restoration
  for a final answer followed by a scheduled user turn. A read-only CPU replay of
  stopped v2 slot 002 verifies five original actions and the 25 masked follow-up
  tokens before entering only the missing sixth model call. This is not yet a
  checkpoint-31 fork, a new model response, or an optimizer update.

- Revise the final-answer EOS policy: keep `<eos>` in the sampled action ledger
  and append the scheduled user's native turn as a masked external suffix without
  inserting a replacement `<turn|>`. Verify the exact next input, action/mask
  ledger and complete-group continuation with the pinned tokenizer. A read-only
  audit of the stopped v2 attempt confirms the 25-token suffix; it does not
  synthesize or claim a missing subsequent model response.

- Preserve the second full48 production stop after checkpoint 31: group 32 sampled
  four attempts, one unavailable because a final answer ended in `<eos>` before a
  scheduled user follow-up. An initial correction classified it as a candidate
  failure; the revision above instead continues a scheduled follow-up while
  preserving sampled tokens. Neither change edits v2, resumes it, or replaces
  genuine host/protocol failures with a reward.

- Pass the single v6 headless 24576-token controlled fit with the corrected source:
  native generation, four DAPO microbatches, optimizer update and full checkpoint
  verification. Launch the separately preflighted fresh-base 96-group run, keeping
  the earlier checkpoint-14 failure separate; production outcomes remain pending.

- Retain every full48 optimizer-boundary checkpoint in the next fresh run (96 total
  when complete), leaving the two-checkpoint historical probe unchanged. A six-update
  tiny CPU schedule/resume run verified all checkpoints survive and exact state agrees
  with uninterrupted execution. Bind the changed source to a fresh v6 fit profile;
  the terminal first production run is not modified.

- Preserve the stopped first intact full48 run at checkpoint 14: group 15 sampled four
  attempts, two with unavailable native tool protocol evidence. Refuse in-place resume
  or resampling. A sampled tool call ending at `<eos>` now fails as a scored candidate
  before tool execution; a mixed content/tool call with a correct native boundary keeps
  its raw action tokens and appends only the masked external tool response. Infrastructure
  and corrupt-token failures still halt the group. A fresh identity and pinned-base
  restart are required.

- Record the passing single-attempt v5 24576-token RTX 3090 fit: headless admission,
  native generation, four public-DAPO microbatches, one optimizer update, and a
  hash-verified full checkpoint. Controlled memory qualification passed; original
  full48 sampling, writing quality, and production training remain unmeasured.

- Preserve the terminal v4 24576-token fit after its evidence observer rejected the
  correct 32768 group-active-token DAPO denominator. Native generation and the first
  training forward passed, but backward/update/checkpoint did not run. Correct the
  observer under a fresh v5 identity without changing the training recipe.

- Select a 24576-token v4 fit and production context after both 32768-token all-linear
  fits OOMed. Preserve 8192 tokens per decision, 16384 sampled actions, FP32 rank-8
  all-linear LoRA, original tasks and output requirements; reduce only observation/tool
  headroom, with overflow remaining an explicit failure.

- Record the terminal headless/expandable-segments 32768-token fit: ownership and
  native generation passed, but FP32 all-linear LoRA training OOMed in the MLP
  `down_proj` path before an optimizer update. Confirm the exact recipe does not fit
  the 24 GiB RTX 3090; keep full48 production at zero groups and attempts.

- Add a fresh headless 32768-token fit contract after the desktop-admitted profile
  OOMed: require an empty GPU process inventory, at least 24000 MiB free, and bind
  PyTorch expandable allocator segments before Torch import. Apply the same resource
  contract to full48 train/resume without changing model, tokens, precision, LoRA,
  objective, or task requirements.

- Add user-approved Xwayland and Ghostty GPU desktop allowlist entries while preserving
  the existing 256 MiB per-process, 768 MiB total and 22,000 MiB free-memory limits.

- Record the terminal one-attempt 32768-token Gemma fit result: native 32767+1
  generation passed, but controlled training OOMed during the first backward pass
  before an optimizer update or checkpoint. Preserve the failed profile without retry;
  full48 production remains at zero groups and attempts.

- Add a separate inspect-first 32768-token production Gemma controlled GPU fit gate,
  native long-prefix generation check, finite update/full-checkpoint evidence, and
  complete NVML ownership admission before fit and full48 train/resume model loads.

- Propagate host workspace failures as unavailable infrastructure instead of candidate
  rewards. Bind the intact runner to a practical 32,768-token context and 16,384-token
  sampled-action envelope; remove the known-impossible 131K dense-mask contract before
  GPU fit qualification.

- Admit opt-in source-pinned TRL `6c5f135` / Liger `0.8.3` streaming through public
  configuration, disabling unrelated Gemma4 kernel replacements. Bind verified
  package source trees before mutation and reject changed implementation on resume.
  Preserve legacy TRL 1.13 defaults. Qualify BF16 CPU train entry, masked DAPO
  group4 accumulation, tied continuation, token/vocabulary chunk boundaries,
  observation conditioning and exact adapter/optimizer/scheduler/RNG/ledger resume.
  Accept the maintained numerical variant without claiming native BF16 parity.
- Add a dedicated intact full48 runner: 96 ordered groups, 384 disjoint-seed attempts,
  pass-one checkpoint pause, exact boundary recovery, coverage accounting and an
  inherited single-writer lease. Bind it to the source-pinned streaming runtime while
  preserving historical probe limits and original data. Verify six real tiny CPU
  updates against uninterrupted state; native full48 memory fit remains a launch gate.

- Record approved isolated TRL/Liger installation and eight CPU softcap comparisons.
  Verify imports, dependencies and archive hashes; preserve the original environment.
  Report BF16 numerical differences without declaring parity. Remaining qualification
  and full48 GPU execution stay pending.

- Prepare the intact full48 mechanical reward with frozen release bindings, real
  sequence/write evidence and offline counterexamples. Reject unchanged or missing
  delivery; normalize valid tool-path aliases without mutating evidence. Preserve
  original tasks and explicitly unjudged semantics. Full-round GPU execution remains
  blocked on runtime and memory qualification.

- Distinguish saved Python advantage estimates from observed TRL tensors. Reproduce
  float32 residuals and fresh-Adam movement for fractional tied rewards without
  clipping, recentering or changing ordinary TRL behavior.

- Add explicit, identity-bound tied-group continuation without resampling. Keep
  default halt and unavailable-reward refusal; ordinary steps may move weights
  through Adam momentum or rounding residuals. Verify tied-group accounting, dense
  accumulation, all-tied completion and exact checkpoint recovery on CPU. Clarify
  that inference cache estimates do not establish long-trajectory training fit.

- Record full48 DAPO readiness: CPU integration passes, but intact-task output
  budgets, reward shortcuts, long-trajectory memory and tied-group coverage block
  GPU launch. Track two complete passes with advisory timing, not a wall-clock cap.

- Add identity-bound DAPO loss selection through public TRL configuration; keep
  GRPO as the default and preserve the fixed probe limits. Verify variable-length,
  observation-masked CPU accumulation against dense adapter updates and Adam
  moments, plus exact checkpoint resume across two passes over three tasks.

- Verify the microbatched Gemma probe on the RTX 3090: three optimizer updates,
  checkpoint resume, exact resident adapter reload and all 12 paired development
  attempts. Peak Torch allocation 18.249 GiB; total GPU-stage time 45.44 minutes.
  Preserve mixed mechanical outcomes (mean 0.229→0.313), generation-limit failures,
  and the original OOM run without claiming writing improvement.

- Separate GRPO reward groups from training microbatches. The Gemma probe now scores
  four attempts together and accumulates four single-attempt gradients per update;
  generic callers retain full-group training by default. Bind microbatch size to
  resume identity and verify CPU update/moment equivalence, task order, masks, and
  exact checkpoint resume without changing rewards, precision, or token budgets.

- Record the first real Gemma GPU probe: 12 baseline attempts, training rewards
  [1, 0, 1, 1], and CUDA OOM during loss-forward output conversion before any update.
  Preserve the failed run, 14.68 minutes of resource accounting, and the remaining
  checkpoint/adapter verification gap; no retry or profile change was made.

- Add the frozen Gemma probe runner: hash-bound preparation and token preflight,
  matched mechanical development scoring, step-1→3 resume, and resident adapter checks.
  Supervise GPU stages with a 60-minute aggregate cap, interruption accounting,
  competing-process refusal, and explicit missing resource measurements.
- Pass real rollout traces to training rewards. Keep trajectory generation limits
  opt-in so existing per-call-only inference configurations retain their behavior.

- Enable explicit non-reentrant activation checkpointing for GRPO. Record adapter
  state after checkpoint restoration and before updates. Extend the real CPU check
  through step 3, verifying exact resume after the step-1 checkpoint is pruned.

- Freeze a reproducible Gemma engineering probe: three training tasks and six
  development cases with separate source groups. Add completion-gated mechanical
  scoring and 91 valid/invalid fixture cases. Preserve original inputs and disclose
  shorter probe budgets; semantic and literary judgments remain unprepared.

- Connect file-tool task attempts and rewards to gated TRL GRPO updates. Preserve
  exact sampled tokens, mask environment text, and stop unavailable or tied groups.
  Separate inference adapters from hash-checked resumable trainer checkpoints.
  Add usage/recovery methodology and an offline CPU check proving adapter changes,
  reload, and exact resume. Gemma GPU execution remains unverified.

- Add a root `TODO.md` linked from the README as the active work order. Put the
  cached Gemma E2B short-context GRPO probe before longer sessions and any Qwen
  download; point older work checklists to it.
- Finalized native groups export `TrainingBatchV1` with V2-derived token masks and segment
  spans; trailing zero-generation context-limit turns remain audit-only.
- Route native batch token encoding and context-root checks through their shared owners; a
  `carry` context change now makes eligibility and export agree that a member is ineligible.

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
