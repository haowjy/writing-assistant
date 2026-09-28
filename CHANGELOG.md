# Changelog

## [Unreleased]

- Exclude intake placeholders from group segment credit using the derive's per-part decision.

- Derive idempotent member-start receipts from verified ancestry so retries survive progress.

- Move group member starts and result/credit collection onto verified rollout views.

- Port writer, tool, integrity, and downgrade forgeries through the rollout acceptance seam;
  retain the observation and error-classification findings as expected failures.

- Classify a runtime adapter-manifest change after binding as an adapter contract failure.

- Classify hash-correct malformed registered payloads as projection forgeries, not disk corruption.

- Document the driver, gatherer, replay, and cross-lane fixture extension contracts.

- Expose shared rollout fixture hooks for ports, alternatives, feedback, sessions, checkpoint
  observation, and resume-safe crash recovery; return named driver outcomes.

- Keep usage and group sampling checks at the derive, share logprob persistence and scripted
  reply generation, and pin the adapter, replay, privacy, and failure boundaries in tests.

- Make gate publication retries thread-independent, preserve store-corruption errors, and
  route branch semantics through the lineage-pinned selector.

- Centralize rollout head checks and derived-artifact persistence, remove unused environment
  workspaces, and dispatch gathers from typed ports with derive-owned rules.

- Add typed port gatherers and a synchronous rollout driver that resumes from published heads.

- Dispatch store semantics from the lineage's pinned transition version; only cache gate-owned
  views after publication, reject malformed typed payloads at write time, and pin each rollout's
  admission policy.

- Validate typed payloads in their canonical wire form so canonicalizable Python inputs still
  persist while malformed records fail at the write boundary.

- Route derive check selection through the controller, remove redundant author-request
  validation, and reuse the verified raw-call index when deriving writer turns.

- Add the single-step rollout environment for verified commits, typed port inputs, and
  post-publication view caching.

- Verify typed lineage folds through `LineageGate`, with a checkpoint-keyed view LRU and an
  optional store verifier port for publication and restore.

- Return author and writer transition logic to the derive modules, leaving scripted
  policy and sampling codecs in their runtime modules.

- Publish the strict wire-record API, derive registries from declarations, and require
  explicit value kinds for derived artifacts.

- Unify transition construction across derives, persist typed queue rejections, and bind
  derived artifact refs to their explicit value domains.

- Restore schema-bearing group spec wire compatibility; validate seed and group bindings at
  decode, resolve legacy group edges by pinned semantics, and halt when no required terminal
  evaluation is admitted.

- Derive context operations from sealed policies and ancestry, derive member starts from
  persisted group specs, and persist each spec at seal time.

- Derive checks, transitions, sealing and rewards through one `OutcomeV1` lifecycle, and
  resolve evaluator contracts for non-interactive writer nodes that declare them.

- Derive author requests and replies as single pure transitions, keeping requirement ledgers private.

- Derive writer turns and tool results once from typed inputs, with strict re-cut sampling
  evidence and sealed group-policy binding.

- Derive strict record field specs and `REFS` from each field declaration, keep binding
  rules on their records, and split the shared wire core from group and payload codecs.

- Return entry state and artifacts together, compare materialized contexts, and use typed or
  canonical-byte values for derived artifacts.

- Unify task-graph record validation and reference edges under declarative field specs;
  make sampled-message intake typed, and validate shared payload and execution-version
  records through store closure.

- Replace the deterministic controller view with pure lineage-view directives and one guarded edge selector.

- Add immutable task-graph transition views, stable nested diff paths, and pure entry derivation
  with explicit artifact outputs; centralize static tool schemas so derives do not import the
  workspace or a runtime port.

- Specify canonical changed-path effects and verify an identical-content write records an empty delta.

- Compute the tool-result workspace byte total once and rely on the pinned workspace/storage limit
  relation instead of checking an unreachable second bound.

- Scan the full `writing_agent` import graph and report cycles outside the task-graph module family.

- Centralize re-entrant operation scopes, give closure validation ownership of rejected-candidate
  cleanup, preserve operation exceptions during debug audits, and remove store pass-through methods.

- Centralize deterministic ask-author shape validation in the canonical call module and re-export
  it from the scripted-author boundary.

- Separate adapter-message intake from strict recorded-form validation, reject malformed typed
  markers, run the full canonical-call differential by default, and copy exported queue arguments.

- Restore fail-closed read-side coverage for record-domain artifacts stored outside the store API.

- Add strict transition-seam wire codecs and declared payload edges, plus chained context
  records in version-pinned store locations while preserving legacy runtime records.

- Add canonical sampled-message intake, declarative tool-call parsing, and pure file-effect
  contracts for the transition seam.

- Scope closure validation reuse to one re-entrant store operation, return copied artifact
  bodies, reject artifact writes using record domains, and avoid redundant canonical codec
  and frozen-record identity walks.

- Centralize task-graph runtime exceptions in a dependency-free leaf module, remove unused
  task-graph aliases, and guard the import seam with an AST-based SCC test.

- Deliver the complete canonical persisted sampling request to backends, persist
  typed binary logprobs without backend store access, and verify transcript-shaped
  evaluator evidence offline against frozen inputs and an authorized packet.

- Preserve prepublication and forced-recovery forgery coverage for sampled-budget
  stops by binding the locally constructed trace while replay checks the persisted ref.

- Reject unbound direct-submit writers on members of a manifest-sealed group,
  closing the legacy-constructor bypass while retaining unbound legacy group runs.

- Include the offline scripted backend's exact sample sequence in its immutable
  descriptor configuration, so distinct scripts cannot share a sealed manifest.

- Preserve remote execution infrastructure classification in an explicit error
  and interrupt before a tool observation or budget charge is published.

- Route action and terminal training-eligibility claims through one typed public
  sampling decoder, including exact outcome binding and native-ineligible policy.

- Exercise the separately admitted fixture evaluator through check replay, terminal
  outcome, and reward publication with no terminal branch for the new family.

- Bind started group members back to the exact sealed adapter manifest before
  any writer operation; a separately sealed A-vs-B session cannot act on A's branch.

- Close the runtime dependency boundaries: canonical sampling bind and native-ineligible
  decoding, a real prepared-input sampling backend and runner, sealed runtime sessions,
  remote-capable text execution ports, replay-verifiable evaluator families, and
  semantic replay finalization. Keep the legacy direct-submit path explicitly unbound.

- Install the declared judging extra in CI so the existing paid-call
  environment-validation test can import its dotenv dependency.

- Use per-instance mapping default factories for frozen task-graph records so
  supported Python 3.11 can import them; post-initialization still freezes the
  same values and preserves canonical identities.

- Scope CI Ruff checks to product source, scripts, and tests so research work
  artifacts and agent configuration are not treated as package Python code.

- Preserve the historical fail-closed error for Phase 5-only events attempted
  on a Phase 4 writer lineage after moving dispatch into semantic replay.

- Make sampling port methods return immutable typed V1 contracts; the writer
  serializes them only at the persistence boundary.

- Route group segment-credit trace admission through the typed sampling codec
  rather than locally interpreting native eligibility.

- Preserve the distinct stale verified-message error at the typed sampling
  boundary while retaining the generic malformed-preparation error for other
  request mismatches.

- Add injected sampling, environment, text-tool, and evaluator ports with current
  offline adapters and immutable content-addressed runtime manifests. Exercise
  alternate fake adapters through the same writer/check flow and seal a manifest
  in an existing group policy reference without changing group wire identity.

- Split semantic replay from visible-context projection: a typed cursor and closed
  event-handler registry validate admitted writer histories and emit authorized
  message/provenance contributions. Represent author reply progress with an
  internal enum; projection now only materializes and verifies the result.

- Move causal event/effect/log/context staging and atomic CAS publication into an
  environment-owned batch service. Writer, scripted author, deterministic checks,
  terminal routing, and compaction use the same publisher; remove their duplicate
  publisher plumbing and writer-private transaction API.

- Share pure sampled-usage, tool-attempt/result, context-append, and exhaustion
  accounting across writer/author production and semantic replay. Replay still
  compares independently loaded persisted claims to recomputed charges.

- Introduce immutable typed V1 codecs for prepared writer requests, normalized
  sampling and adapter evidence, and current native-ineligibility decisions.
  Centralize trace/request binding and group sampling claims without changing
  persisted wire identities or allowing arbitrary adapter payloads to qualify.

- Make group collection and offline finalization share complete result admission,
  and reject sampled budget-stop seed/model or declared policy drift before credit.

- Add an isolated deterministic Phase 7 GRPO group coordinator with full start-contract sealing, domain-separated member seeds, resumable member branches, immutable terminal/reward binding, exact symbolic group advantages, pending/tie/invalid handling, and writer-only segment credit without fabricated native traces or optimizer updates.

- Reject non-integer or noncontiguous writer runtime-log sequence witnesses and
  compare historical prefixes by canonical bytes on publication and recovery.

- Anchor context-operation validation in immutable writer entry ancestry across
  publication and recovery, bind event-envelope ownership, strictly type nested
  witness numbers, and add a verified-message request path while keeping arbitrary
  adapter payloads explicitly unverified.

- Add opt-in deterministic context carry, named ancestor seed, drop, and
  fixed visible-message compaction at drained writer boundaries. Immutable
  operation records bind exact source messages/events, summary bytes,
  old/new context revisions, and context charges; prepublication and offline
  recovery validate the same authority while preserving earlier action traces.

- Require a guaranteed Phase 5 completion edge at admission; optional or
  progress-only check guards cannot strand a passing terminal check batch.

- Close Phase 5 authority and recovery gaps: anchor strict event dispatch to
  the admitted entry, require complete atomic author replies, unify writer
  exhaustion with typed incomplete outcome/reward records, enforce tool/author
  budget precedence, and reject progress-only reward inputs and private bytes
  in public vocabulary at admission.

- Add an opt-in deterministic scripted-author graph slice: role-typed author and
  evaluator packets, structured `ask_author`, durable request/reply and disclosure
  updates, ordered mandatory feedback and preauthorized requirement supersession,
  frozen deterministic checks, environment terminal routing, exact reward records,
  and offline semantic restore/replay. The legacy runner remains the default.

- Validate complete writer histories against staged candidates before publication,
  require explicit writer-stop log attribution from the first event, and bind
  action/stop ordinals and adapter logprob references on production and recovery.

- Complete Phase 4 writer event ownership and trace attribution checks before
  publication and on restore/replay; classify parent-file conflicts as observations,
  preserve staging corruption as infrastructure interruption, and bound malformed
  call decoding and evidence.

- Close Phase 4 transactional writer review findings: preserve infrastructure
  failures, normalize malformed calls without dispatch, reject stale handles before
  tools, validate result provenance/execution/charges on restore and replay, and
  durably stop on token exhaustion while rejecting unenforceable context limits.

- Add an opt-in transactional task-graph writer/text-tool runtime with committed
  call queues, per-call observations and budget charges, exact event-derived writer
  context projection, restore/replay, and explicit non-native loss eligibility.

- Add fail-closed task-graph admission, immutable versioned node/check/guard contracts,
  deterministic environment-owned controller directives, and an opt-in adapter that
  preserves legacy scenario inputs while compiling them into a scripted writer node.

- Add private content-addressed task-graph persistence with full immutable checkpoints,
  atomic lineage-head compare-and-swap, safe fresh-workspace restore, isolated branches,
  state/file diffs, and offline reduction of recorded committed effects.

- Add a root `TODO.md` linked from the README as the active work order. Put the
  cached Gemma E2B short-context GRPO probe before longer sessions and any Qwen
  download; point older work checklists to it.

- Plan task-declared simulated-author use with cache-aware Astra calls rather than
  mandatory per-turn routing. Record Jev and self-play as optional ideas, distinguish
  incomplete sessions from environment errors, and correct the blanket exclusion of
  local 27B training in light of Unsloth's documented short-context 24GB QLoRA support.

- Clarify the training goal: long-form project memory and effective use of large
  writing projects come first, supported by wiki maintenance, with prose quality as
  a companion outcome. Record the full task mix, broad instruction variation, local
  3090 limits, and a cost-dependent 256K Qwen context target whose training/inference
  scope remains open. Keep code-to-documentation and code-to-story as optional ideas,
  not required coverage or authorization for additional datasets or execution.

- Set the active training plan to direct GRPO on instruction-tuned Qwen3.8-27B.
  Use E2B only for engineering verification; make SFT conditional on a demonstrated
  Qwen behavior gap. Align the work order and initial evaluation comparison, and
  list unresolved reward, task, system-prompt, compute, and success criteria.

- Make the across-output prose measures computable. `prose.sample_distribution` pools
  every attempt of one scenario, which is what MMD, self-BLEU and dispersion require; a
  per-attempt profile could only ever report them as insufficient samples. `MINIMUM_SAMPLES`
  withholds a measure below its floor with the required count in the reason,
  `RELIABLE_SAMPLES` marks where two models can be compared, and `sampling_plan` resolves a
  proposed attempt count against both. The held-out long-form suite now declares its
  attempts per case and the builder resolves it, so a metric cannot fail quietly at
  scoring time after the GPU time is spent.

- Add the `excludes_all` mechanical check, which passes only when none of a declared set
  of strings appears. Deleting a fact from a document fails in a second way beyond
  leaving the fact in: the text can keep referring to it by negation, and phrases like
  "no longer" or "rather than" are edit artifacts that should be caught mechanically
  rather than left to a judge.

- Raise the harness context limit to 65,536 tokens and the SFT acceptance bound to
  match, replacing arbitrary 8192/16384 settings that silently capped the long-form
  evaluation cases. Add `scripts/probe_context_budget.py`, which computes the inference
  KV cache from the checkpoint config and measures the attention step cost on the real
  layer shapes. Measured: 128K inference needs 1.9 GB of cache because 28 of 35 layers
  use a 512-token sliding window and every layer has a single KV head, while a 64K
  training step projects to 12.7 minutes. FlashAttention is unavailable because the
  full-attention layers use `global_head_dim=512`, above the FA kernel limit, so
  memory-efficient SDPA is the only O(n) path.

- Add the held-out long-form final-evaluation suite: six cases over three public-domain
  works that supply a multi-chapter manuscript and ask for more, so the axis is long-range
  continuity, plan adherence, revision propagation and arc closure rather than short
  single-scenario capability. `longform_suite.py` and
  `scripts/build_longform_suite.py` compile deterministic slices into the shared scenario
  contract, refuse to build when a check cites a fact its supplied text lacks, and refuse
  to build when a work is already in use by training or by the development suite.

- Bind the pinned EQ-Bench Longform Writing release as the external long-form anchor.
  `longform.py` renders the upstream 13-step plan-then-eight-chapters protocol,
  compiles twelve `final_eval` scenarios through the existing compile contract, and
  reimplements the upstream criteria, weights and arithmetic, including its
  chapter-degradation and staccato-penalty behaviour. Generation is reply-only in
  `faithful` mode for comparability and file-delivering in `workspace` mode, which is
  labelled an adaptation. `acquire_sources` gains the pinned `eqbench_longform`
  selection. Upstream's discontinuous staccato curve is documented and reproducible via
  `legacy_curve=True` rather than silently replicated.

- Add `reward.py`, the training-side RL scalar. Version 0 combines anchored 1-5 quality,
  intent and continuity judgments with deterministic mechanics, applies a declared
  critical-failure cap, composes stages with the final project state, and computes
  within-group advantages with a zero-variance report. A withheld judgment makes the
  reward unavailable rather than zero, and only pre-declared criteria can be critical.

- Restructure task preparation around one validated `Sampler` value: `Sampler.build`
  checks the selection and derives its index-addressable content, and
  `iter_requests` / `build_request` / `prepare_task_requests` each take it. This removes
  a seven-parameter contract duplicated across three functions, eager-validates instead
  of deferring errors to first iteration, and drops an unreachable branch.
- Give each specificity decision point one behavior and one introduction level and
  derive the per-family split from that table, replacing a nested ladder that restated
  every point up to five times and admitted non-monotonic splits. The table is
  validated at import, and an always-stated point can no longer be withheld silently.
- Resolve a request's source packet once per batch instead of once per validation and
  again per generation. Prepared batches are byte-identical to before the refactor.

- Make task preparation streaming and source-referencing: `iter_requests` and
  `build_request` yield one request at a time, each derivable from its index and seeded
  per-key permutations instead of shared RNG state, so a batch can resume mid-way.
  Requests carry a source id and hash rather than the passage; `resolve_packet` inlines
  it for model calls and validation. On the 100-request fixture this drops the prepared
  payload from 558 KB to 134 KB and embeds no source text. Batch and request
  `schema_version` is 3.

- Add the specificity ladder: a structured `instruction_specificity`
  (`level`, `stated`, `withheld`) with per-family decision points, ask-required vs
  default-safe behavior classes, and matched request ladders. Coverage reports the
  level and withheld points.

- Specify the training-distribution axes (task variables vs nuisance variables, reward
  invariance across nuisance axes) and treat underspecified requests as the primary case:
  clarify when a decision is consequential and undetermined, proceed when it is not, and
  reward the outcome against the author's hidden preference rather than the act of asking.
  Specify the simulated-author loop (deterministic environment gate, scripted controller,
  cached paid user model) that supplies author turns without a human.

- Record the first 100-task generation result: 54 of 100 admitted, 5 needs_revision, 41
  invalid on mechanical contract checks, $3.79 of the $10 cap spent.

- Pin the task-generator output schema (exact top-level keys, follow-up count,
  check `method`/`kind`, one prose selector per F1/F2/F5 stage, verbatim quotes),
  admit evidence quotes that match after whitespace collapsing, and ignore extra
  top-level keys. Enable thinking on author (`max_tokens=32768`) and reviewer
  (`max_tokens=16384`) calls.

- Allow explicit reconciliation of interrupted paid-call reservations to a terminal
  `abandoned` accounting key that retains the charge without blocking later calls.

- Switch the paid task author, reviewer, and output judge to DeepSeek V4.1 Flash
  (`deepseek-flash`) over its direct OpenAI-compatible API. Replace the GLM-5.3/Reka
  OpenRouter client with a provider-neutral `PaidClient` plus an isolated DeepSeek
  transport; update pricing to the direct route with cache-hit accounting. Verified by
  a live smoke call (valid JSON, `deepseek-flash`, $0.000024).

- Settle the RL optimizer as a critic-free group method (GRPO-family, not PPO) and
  reject a staged GRPO → critic-warmup → PPO pipeline; record the reward the algorithm
  consumes and flag pointwise-vs-pairwise judge output as the next reward decision.
  Add retrieval-coverage and reward-format validation tasks to the RL preparation plan.

- Add deferred final-base candidates (Gemma 4 12B, Qwen3.8-27B, Qwen3.5-9B,
  Qwen3-Coder-Next) with size, license, and local-fitness notes; base choice waits on
  the mini-eval subset.

- Add Python-driven GLM/Reka task generation and independent review with a shared
  $10 budget, resumable calls, source and schema checks, and visible/private export.
  Add a GLM output-grader adapter using existing rubrics; live calls await credentials.

- Specify purpose-dependent information coverage, relevance, density and structure
  measurements, including prose contribution and a proposed bounded nonfiction slice.

- Specify a proposed initial mixed RL reward, task-specific quality rubrics,
  critical-failure cap, unavailable-grading behavior and multi-stage aggregation.

- Identify released Skywork and RM-R1 reward-model candidates; record local memory,
  scoring-interface limits and criteria for deferring custom reward-model training.

- Review LitBench, HANNA, LiteraryTaste and WritingPreferenceBench as writing-reward
  references; distinguish preferences, dimensional ratings and unlabeled prose.

- Expand task-authoring options to 60 genres with explicit blends, 40 tropes,
  32 situations and continuity challenges; preserve source style in continuations
  and distinguish sampled options from verified narrative diversity.

- Compile training-system and synthetic-data research; prepare 100 source-backed
  task-generation assignments with coverage, hashes, lineage checks, and no API calls.
  Record GLM-5.3/Reka selection and keep generated-task admission pending.

- Record the 3090-first training decision, defer GPU rental, and outline optional
  W&B logging for written critiques, prose, metrics, and artifacts.

- Research composed multi-turn RL with simulated author feedback and context
  compaction; record reward/credit requirements and verified context limits.

- Specify on-demand branching-fiction RL tasks, source-backed judge context, and
  separate mechanical, semantic, and source-overlap rewards.

- Research SFT as an RL bootstrap, writing-specific rewards, local judge candidates,
  and provider restrictions; replace the fixed SFT expansion prerequisite with a
  proposed direct-RL versus short-SFT-plus-RL comparison.

- Create a labeled 24-trajectory SFT seed from upstream-train human stories, two
  Gutenberg works, and original synthetic projects; save tool traces, source lineage,
  token masks, and review materials. Plan expansion before substantive SFT.
- Fix SFT evaluation exclusions to use actual catalog source IDs and lineage groups.

- Prepare Python-driven Gemma SFT with native assistant/tool-call loss masks,
  grouped data exclusions, optional QLoRA execution, and pinned training dependencies.
  Verify five pending format probes and TRL label preservation without training.

- Record the next training-experiment TODO beside the full research plans, covering
  failure analysis, separate training data, QLoRA verification, and checkpoint evaluation.

- Record candidate harness, serving provider, and model on scorecards and report
  groups; keep scores from different execution systems separate.

- Record the completed Grok/OpenCode pilot: five task passes, mean Astra prose
  rating 4/5, and explicit comparisons with the same Gemma cases and harness limits.

- Add a five-case xAI Grok/OpenCode pilot with native tool traces, isolated scenario
  directories, saved artifacts, and shared Astra rubric scoring.

- Complete and publish all 50 custom-suite Astra judgments: 29 required-task passes,
  21 failures, archived pre-grading views, and a 50-session isolation audit.

- Distinguish missing required prose delimiters from ambiguous extraction, so
  missing file delivery fails completion instead of remaining pending.

- Add a dedicated Astra literary-grading instruction profile, isolated sessions,
  locked call accounting, and versioned custom50 semantic scorecard reports.

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
