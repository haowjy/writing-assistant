# Evaluation boundaries

The research entry point is [evaluate.py](../../scripts/evaluate.py). It selects
Python configuration and calls library stages independently. The existing CLI
retains its smoke-evaluation and training-format workflows.

## Responsibilities

- [suite.py](../writing_agent/suite.py) compiles scenarios and owns serial execution,
  isolated workspaces, resume identities, saved results, and fresh-reader conditions.
- [agent.py](../writing_agent/agent.py) owns the bounded conversation/tool loop;
  [workspace.py](../writing_agent/workspace.py) owns file operations and storage limits.
- [task_graph.py](../writing_agent/task_graph.py) owns core environment value records and
  their approved identities. [task_graph_store.py](../writing_agent/task_graph_store.py)
  owns private content-addressed persistence, typed reference closure, structural checkpoint
  saving, and the atomic mutable lineage head. A semantic verifier is required at construction;
  runtime commits publish through it. Runtime stepping does not materialize a workspace. Runtime lineages must
  pin `task-graph-derive-v1`; a lineage without that pin is refused at the versions
  reference, and the store has no patch-effect fallback.
  [task_graph_contracts.py](../writing_agent/task_graph_contracts.py) adds detailed
  execution contracts as immutable artifacts referenced by the frozen Phase 1 node shape;
  [task_graph_admission.py](../writing_agent/task_graph_admission.py) resolves their
  public/private closure and rejects an unsound graph before sampling.
- **Transition seam (new core).** [task_graph_wire.py](../writing_agent/task_graph_wire.py)
  is the field-spec vocabulary, strict decoder and `WireRecord` base;
  [task_graph_records.py](../writing_agent/task_graph_records.py) declares input, outcome, and
  context `WireRecord`s, plus runtime port-descriptor and manifest records. V1 records remain
  unchanged; V2 sampling, manifest-descriptor, and training-admission records are additive.
  Context content and revision records live here. Pure group record classes live in
  [task_graph_group_records.py](../writing_agent/task_graph_group_records.py).
  [task_graph_record_contracts.py](../writing_agent/task_graph_record_contracts.py) declares
  sealed wire contracts, while [task_graph_payloads.py](../writing_agent/task_graph_payloads.py)
  provides codecs for shared payload shapes without record classes. Reference edges derive
  from the field annotations, and the store follows them in shared closure.
  [task_graph_calls.py](../writing_agent/task_graph_calls.py) owns sampled-message intake,
  call parsing, exact rejection-message codes and the tool effect contract.
  [task_graph_tool_outcomes.py](../writing_agent/task_graph_tool_outcomes.py) reads verified
  member start/final views to pair every committed tool call with its result and derive the
  final file delta; trace checks and P1 evidence share this reader.
  [task_graph_transition.py](../writing_agent/task_graph_transition.py) owns the immutable
  view/transition types. [task_graph_controller.py](../writing_agent/task_graph_controller.py)
  is the pure directive boundary (`next_step`, `select_edge`, `applicable_checks`); it does
  not step a writer,
  execute checks, accept author transition/completion claims, or mutate runtime state.
  [task_graph_derive_entry.py](../writing_agent/task_graph_derive_entry.py) derives a node's
  entry state and root artifacts; the writer, author, outcome and context derive modules
  build each step through
  [task_graph_derive_common.py](../writing_agent/task_graph_derive_common.py).
  [task_graph_derive_context.py](../writing_agent/task_graph_derive_context.py) calls
  `require_quiescent`, `select_context`, and `charge_budget` directly, derives member starts
  from persisted group specs, and takes named seed contexts from view ancestry.
  [task_graph_gate.py](../writing_agent/task_graph_gate.py) assembles the derive
  registries into the one `derive_input` dispatch, re-admits from the versions-pinned
  policy, and folds typed events as the store's mandatory verifier. The gate's
  checkpoint-keyed view cache holds only published views and skips derives only; store
  closure still re-reads and hashes persisted bytes on every operation.
  [task_graph_environment.py](../writing_agent/task_graph_environment.py) is the producer:
  entry, cold open, head resume, per-step verification, derive-persist-publish commits,
  member starts, and the typed port inputs that are the privacy boundary.
  [task_graph_rollout.py](../writing_agent/task_graph_rollout.py) is the synchronous
  driver loop, and [task_graph_gatherers.py](../writing_agent/task_graph_gatherers.py)
  holds the thin port adapters that turn a typed port input into a recorded input. See
  [transition-seam.md](transition-seam.md) for how to add a record or a derive and for the
  layer order; [gate-and-rollout.md](gate-and-rollout.md) for gate, store and environment
  contracts; [rollout-execution.md](rollout-execution.md) for driver, gatherer, resume and
  acceptance-test contracts; and [group-coordination.md](group-coordination.md) for groups.
- **Task-graph runtime.** The transition-seam modules are the runtime. Each commit contains
  one typed input event; the same derive computes the producer transition and verifies it
  during publication. Cold opens and resumes use `RolloutEnvironment` and the gate's verified
  view. Context records carry the content chain and revisions;
  `OutcomeV1` carries outcome, reward and eligibility. `RolloutDriver` gets the verified
  directive and typed port input through `step_input`; there is no runtime log or workspace
  materialization. See [transition-seam.md](transition-seam.md),
  [gate-and-rollout.md](gate-and-rollout.md), and
  [rollout-execution.md](rollout-execution.md) for the contracts.
- [task_graph_sampling.py](../writing_agent/task_graph_sampling.py) owns the single
  V1/V2 writer-turn decoder and structural sampling evidence checks. The shared
  [task_graph_native_contracts.py](../writing_agent/task_graph_native_contracts.py) owns
  `NativeSamplingBudget`, `NativeSamplingHistory`, and the manifest-policy-rendering binding
  used by seal and derive paths. [task_graph_token_ledger.py](../writing_agent/task_graph_token_ledger.py)
  owns the little-endian u32 token codec shared by ledger readers/writers and training export;
  [task_graph_context_roots.py](../writing_agent/task_graph_context_roots.py) owns the fail-closed
  context ancestry walk used by V2 sampling, native history, eligibility and training export.
  Any intervening `context_changed` event, including `carry`, is a new root. V2 binds token
  bytes, prior-turn prefixes, derived prompt/completion/total/prefill/cache usage, sealed
  sampling pins, and termination
  derived from decoding and committed budgets. [task_graph_eligibility.py](../writing_agent/task_graph_eligibility.py)
  owns the ordered pure structural-eligibility decision, which `derive_reward` persists;
  it reads only the verified view and hash-addressed evidence through the artifact reader.
  [task_graph_accounting.py](../writing_agent/task_graph_accounting.py)
  supplies pure sampled, tool, context-append and exhaustion policy to production
  and replay; persisted budget/charge artifacts remain independently compared claims.
  [task_graph_ports.py](../writing_agent/task_graph_ports.py) defines immutable
  descriptors and typed sampling, execution-environment, tool-provider, and evaluator
  ports without importing concrete adapters. [task_graph_composition.py](../writing_agent/task_graph_composition.py)
  owns the persisted runtime manifest, sealed session, local transaction publisher,
  and runner that invokes a backend from verified current messages.
  [task_graph_local.py](../writing_agent/task_graph_local.py) supplies offline scripted
  sampling and a local workspace environment composed with the text provider. Its
  staging path never enters a public port. Direct `submit_action` is an explicitly
  unbound legacy/offline path; a bound session checks descriptor identity and any
  started group member receipt before effects.
  [task_graph_evaluation.py](../writing_agent/task_graph_evaluation.py) admits only
  versioned evidence families with replay-only verifiers. Deterministic and fixture
  families recompute exactly; transcript-review checks persisted transcript
  provenance and declared status against frozen inputs and an authorized private
  evaluator packet, without claiming subjective correctness. Injection does not
  override admitted schemas, semantic replay, or the native-ineligible decision. `prepare_request` pins caller-owned evidence;
  `prepare_verified_messages` checks typed messages against the active projection
  at preparation, publication, and recovery. The sampling decoder alone binds
  duplicated trace/action/request claims and dispatches V1/V2 ledger evidence. The bound sampling input carries the
  complete canonical persisted request/options value; composition stores typed
  binary logprob output and constructs its ref without backend CAS access. Only the
  composition runner invokes `SampleBackend`. The [author derive](../writing_agent/task_graph_derive_author.py)
  builds author requests and replies, disclosure and authorized requirement updates. The
  [scripted policy module](../writing_agent/task_graph_scripted.py) contains only pure
  script-policy helpers used by author derives and gatherers.
  [task_graph_compaction.py](../writing_agent/task_graph_compaction.py) owns the
  fixed visible-message summary, complete-exchange selection, immutable context
  operation evidence, and context byte accounting.
  [task_graph_group_contract.py](../writing_agent/task_graph_group_contract.py)
  resolves the sealed group environment and defines the member-result, decision,
  advantage and credit records and the typed scripted-terminal and execution-failure
  records; the sealed `GroupSpecV1` and `ContextPolicyV1` and their binding rules live in
  `task_graph_record_contracts.py`.
  [task_graph_group.py](../writing_agent/task_graph_group.py) starts each member as its own
  new-core lineage through `RolloutEnvironment.start_member` (a retry resumes through
  `open_head`). It admits collected results, and results finalized after a reopen, against
  gate-verified member views. It computes group advantages, and writer-only segment credit
  from the view's samples. It never samples models or emits native token masks. A
  coordinator requires a `RolloutEnvironment` and has no bare-store start path. It is
  not the trainer; Phase 8 connects it to the standalone DAPO trainer through
  `TaskGraphRollouts`. See
  [group-coordination.md](group-coordination.md), and
  [group coordination](../../docs/task-graph-groups.md) for the user-facing API.
  [task_graph_training_layout.py](../writing_agent/task_graph_training_layout.py) owns the
  pure native token layout shared by group segment-credit spans and export. The
  [training export](../writing_agent/task_graph_training_export.py) derives a
  `TrainingBatchV1` and byte artifacts from a settled native group; token masks are
  reconstructed from V2 ledgers, while tokenizer-backed admission remains adapter-side.
  [native_audit.py](../writing_agent/native_audit.py) re-renders committed context, audits
  exported token layouts and pinned tokenizer files, then returns `TrainingAdmissionV1`.
  `GroupCoordinatorV1` owns durable admission and trainer-consumption receipts. Only
  all-admitted batches may reach a trainer. `inspect_group_offline` repeats the batch
  and admission derivation from stored evidence and the pinned local tokenizer without
  network access; its canonical report contains no prompt, packet, context or token data.
  [task_graph_probe_experiment.py](../writing_agent/task_graph_probe_experiment.py) owns
  the Phase 8 probe recipe, admitted task loader and shared run composition; task configs
  are data inputs, while the CPU scripted sampler remains in the smoke script. Criterion 6
  uses [grpo_task_graph_probe_privacy.py](../writing_agent/grpo_task_graph_probe_privacy.py)
  to scan run artifacts for private author/check canaries outside `training/private`; its
  sibling scope reports per-member input reconstruction from that member's verified lineage,
  not general absence of unplanted shared or sibling-derived text.
- [legacy_graph.py](../writing_agent/legacy_graph.py) is an opt-in compiler from the
  existing visible brief/files/follow-ups/tools/budgets and private checks into one
  scripted writer node. Its projections match the unchanged `run_selected` call;
  compiling a graph never opts a scenario into a new runner.
- [backends.py](../writing_agent/backends.py) defines the model interface and HTTP transport.
  [inference.py](../writing_agent/inference.py) loads local Transformers/PEFT weights,
  renders native Gemma tools, parses responses with the pinned tokenizer, and adapts
  generation to that interface.
  Inputs contain messages and permitted tools, never evaluator labels.
- [catalog.py](../writing_agent/catalog.py) owns source identity, lineage, imports,
  JSON artifact formats, and overlap inspection. [atomic_io.py](../writing_agent/atomic_io.py)
  owns durable atomic byte/JSON replacement shared by catalog outputs, trainer reservations,
  trace reports and offline inspections. [acquisition.py](../writing_agent/acquisition.py)
  handles the selected upstream releases. [development.py](../writing_agent/development.py)
  compiles the authored world records into development cases.
- [artifacts.py](../writing_agent/artifacts.py) extracts designated prose and inspects
  supported Markdown links. [scoring.py](../writing_agent/scoring.py) consumes saved
  results and private checks; [prose.py](../writing_agent/prose.py) owns numerical
  features and distribution comparisons.
- [grading.py](../writing_agent/grading.py) owns blinded packets, bounded Codex calls,
  judgment validation, and human-review records.
- [evaluation.py](../writing_agent/evaluation.py) retains the older smoke protocol;
  it shares the agent loop. [data.py](../writing_agent/data.py) retains the reviewed
  training-trajectory format and export checks. Research scenarios are a distinct format.

Keep model execution, scoring, and grading independently callable. Library imports
perform no downloads, model loading, process exit, or working-directory changes.
Optional data, lexical-metric, and model dependencies live in separate extras.

## Records and replay

Task-graph immutable objects can exist before publication, but only canonical
`refs/<lineage>.json` head files are lineage authority. Their persisted body contains
only `head_commit`; expected-head is a compare-and-swap request. A transaction writes
and flushes immutable events, a full checkpoint, and its commit before atomically
replacing the head under the lineage lock. Unreachable files are harmless orphans. Checkpoint
opening and lineage resume go through `RolloutEnvironment.open` or `open_head`; the verified
`RuntimeHandle` contains state and context, not a workspace. The environment never rewinds an
event log.

Reference closure uses a thread-local, re-entrant operation scope with a typed traversal
and loaded, active, and completed sets. Closure-validated immutable objects are reused
only within the outermost store or role operation; each later operation re-reads and
re-hashes them. Artifact reads return deep JSON-shaped copies, so caller mutation cannot
alter a verified value. Checkpoint/commit/event depth is traversed iteratively; supplemental
and imported references cannot bypass the validators for their resolved record domain.
There is no process-global validation cache. Immutable-name and ref-name retries repeat
their containing-directory durability barrier even when equal bytes or the matching
head are already visible.
Every non-null authority head must target a checkpoint in its named lineage. A first
commit must start from a parentless checkpoint, which may belong to another lineage (a
group member starts from the shared entry); later commit ancestry may not cross lineages.
There is no mid-lineage branch operation in the runtime API.

Catalog records carry normalized content in `text`; local imports also preserve raw
bytes and a text file, with hashes for both representations. Content from the imported
file replaces any stale text supplied in metadata. Profile and overlap consumers use
the same canonical text.

Compiled releases separate visible packages from private labels. Runtime workspaces
contain only initial files supplied by the visible package. These path-constrained
file tools are not an OS sandbox; no arbitrary shell is exposed to candidates.
Store roots and workspace destinations reject lexical `..`, static symlink ancestry,
and any tree overlap before creation, then use the one checked absolute path. A store
root's parent must already be a real private directory; initialization creates only the
root and its fixed children and repeats each containing-directory fsync on retry.
Concurrent hostile path replacement remains outside this trusted-harness threat model.

Resume identity covers visible inputs, experimental observation metadata (including
builder identity and condition), model configuration, and package source hashes.
Identical KB content from different builders retains separate attempts and attribution. Label changes permit rescoring without generation. Completed failures
are reused unless the caller explicitly requests a retry. Interrupted attempts
remain on disk; retries start a clean workspace. The runner is serial and assumes
one writer per run directory. Reports select the latest attempt, including interruptions, per
identity and keep attempt history available. No automatic transport retries occur.

A final response ends one user turn; declared follow-ups continue within the same
step/tool budget. Each turn retains its workspace snapshot. The read budget counts
returned observation text using the recorded tokenizer (whitespace estimates by
default); rejected reads do not expose their content. Storage is measured in UTF-8
bytes. Provider usage retains nested detail fields: do not sum a detail into its
parent total a second time.

## Task-graph runtime

Each runtime commit stores one typed input event and its pure derive produces the successor
state and artifacts. The same derive verifies each commit during publication; opening a saved
checkpoint re-admits and folds it through the gate. Store construction requires a verifier, and
runtime lineages pin `task-graph-derive-v1`. `WriterTurnV1` pins `context_revision_ref`; optional
adapter-trace context claims are bound when present. There is no separate writer-request record.
`EnvironmentStateV1.history` stores nonnegative `action_count` and `tool_result_count` values.
See the rollout and transition-seam context above for context-record, outcome, driver, resume,
and group contracts.

## Scoring contracts

Explicit selectors locate prose in a final reply, a specified turn, a manuscript
snapshot, delimiters, or a reviewed hash-bound character span. Ambiguous extraction
needs review. Missing prose fails delivery and receives no literary score.
Reply/file duplicate suppression requires an explicit shared delivery group.
Independent identical alternatives remain samples for duplicate-rate measurement.

The scoring module owns check aggregation and task-completion invariants. Judgment
application supplies validated outcomes to that same aggregation function; it cannot
turn failed execution, missing delivery, or a required-check failure into success.
Mechanical checks include `excludes_all`, which passes only when none of a listed set of
strings appears. It exists because deleting a fact has a failure mode beyond leaving the
fact in: the text can keep referring to it by negation, and "no longer", "rather than"
and "unlike the earlier version" are edit artifacts that a judge should not be the first
to notice.

Mechanical scores remain available when Astra or optional models are unavailable.
Semantic outcomes remain pending on grader failure. The grader does not execute
artifact instructions; Codex runs from a temporary directory with a read-only tool
sandbox with shell execution, web search, and delegation disabled. Its first three attempts, including initialization failures, exhaust the
default persisted call allowance. There is no API-key fallback.

Features are keyed by prose hash and configuration. Model loading is explicit and
local-only by default. Comparisons reject incompatible feature configurations.
N-gram L2 uses pooled frequencies; MMD returns the unbiased squared estimate,
including negative values. No combined literary-quality score is computed.

Markdown navigation supports inline local links and ATX heading anchors outside
fenced code. Reference-style links are flagged unsupported. Graph integrity and
fresh-reader question success are separate measurements.

## Limits and verification

The 50 authored scenarios and rubric labels require human review. They are short
synthetic development cases, not a validated or contamination-free benchmark.
Near-duplicate discovery is heuristic; grouping works, authors, and derivatives
remains necessary. Checkpoint evaluation
uses saved weights or caller-owned single-device weights; the caller owns training
pauses and checkpoint identity. Generation restores module modes and PyTorch RNG
state. It never retains cross-scenario conversation or KV caches. Runtime imports
and downloads occur only during explicit loading. See
[local inference](../../docs/local-inference.md) for protocol and scheduling limits.

`inference.HARNESS_CONTEXT_TOKENS` is the harness context limit, a limit rather than an
allocation: the cache grows only with the conversation present. It is 65,536 because the
long-form cases need up to 28K before generating anything, and because the model's cache
is cheap at that length - 28 of 35 layers use a 512-token sliding window and every layer
has a single KV head, so 128K costs about 1.9 GB. `inference.kv_cache_bytes` computes
that from a checkpoint config, and `scripts/probe_context_budget.py` also measures the
attention step cost. FlashAttention is unavailable for this model because the
full-attention layers use `global_head_dim=512`, above the FA kernel limit, so
SDPA is the configured alternative. These inference cache/attention estimates do not
establish GRPO training fit: dense logits, activations and backward buffers also grow
with trajectory length.
`SFTSettings.max_length` is an acceptance bound that rejects overflow and never pads, so
widening it costs nothing until long trajectories exist to fill it; a long window is a
data problem before it is a compute problem.

Use [research-evaluation.md](../../docs/research-evaluation.md) for stage usage and
[delivery evidence](../../work/custom-eval-suite/delivery.md) for measured checks and
remaining experimental prerequisites. Tests cover the library without live candidate
models. The HTTP tests use mocked responses; optional embedding/BERTScore execution
requires separate verification with local weights.

Instruction specificity is scenario metadata carried into attempt identity,
scorecards, and report groups. Loose prompts leave style unspecified and remove
checks for unstated length, viewpoint, idea count, and ending requirements. The
current loose cases remain actionable; conditional clarification dialogues require
a separate runner/user-response contract.

Conversational replies remain ordinary text. `gemma-native-v1` passes schemas to
the chat template and parses raw generated tokens through the tokenizer's response
template. Tool results are associated with calls and rendered as native responses.
The backend receives the trace emitter and records actual model inputs before
generation and outputs before parsing. Base transcript conditions cannot use tools.
Earlier custom-protocol results retain their identities and remain historical.

Graph admission is a separate pre-execution gate. Every writer node names a public
`NodeContractV1` artifact; author packets, scripts, decision bindings, requirement
versions, and checks resolve through explicitly private references. Guards are limited
to the `GuardContractV1` vocabulary and competing exits require unique numeric
precedence. The graph has an immutable hop bound and every node an immutable visit
bound. Admission validates tools, writer families, interaction coverage, check scope
and controller/check versions without invoking a model. Admission supports `none` and
fixed `scripted_author` interaction. `simulated_author` fails closed until its
role-specific packet, policy and binding contracts exist, and mandatory feedback is
admitted only on `scripted_author` nodes. Any writer node that declares a typed
evaluator packet, interactive or not, has that packet and its reward contract resolved;
a `legacy-evaluation-package` body is not resolved as a typed packet.
Check admission uses exact evaluator-version-specific program schemas; semantic-v1 is
not an admitted evaluator. Mapping and store-backed admission share the same typed
closure and visibility rules. Author text is audit input and is never inspected for
routing (controller rules: [transition-seam.md](transition-seam.md)). Outcome records keep
task, execution, stop, reward, and training-eligibility state independent.

The scripted-author slice admits only a terminal edge guaranteed when required
completion checks pass: an unconditional guard, a fixed completion predicate, or
`check_status(pass)` for a required `each_turn`/`node_exit_candidate` check. Optional
and `before_feedback` checks cannot guarantee that edge because completion routing
reads only the current terminal check batch. Distinct precedence still decides which
of several matching edges wins; failed required checks take the typed incomplete
outcome and declared reward path instead of a completion edge.

Genre is separate from prose style and is carried into results and grouping.
Authored genre contexts create derivative sources linked to the original world;
all variants share its development role. Genre assertions must be visible in the
brief or supplied files, not introduced only through private grading labels.

Local Gemma thinking is an explicit `enable_thinking` model setting, enabled in
research/pilot IT configurations and by default for native chat. Non-thinking
chat is reserved for explicitly selected ablations. The parser returns `thinking`; history rendering
maps it to `reasoning` so the checkpoint template preserves it within tool turns.
Saved thinking remains separate from prose content. It consumes the same output
token budget as tool calls and final answers.

`prose.score_prose` is the shared saved-attempt measurement pass. It records D1–D13
for every task, uses only designated prose, and records feature errors and context
hashes. Tasks without designated prose have explicit not-applicable profiles.
A single draft cannot support unbiased MMD; aggregate comparisons must disclose
reference selection and grouping. MPNet features use tokenizer overflow chunks
(feature version 2); older cached features must not be pooled with them.

`prose.sample_distribution` pools every attempt of one scenario, which is what makes the
across-output measures exist: a per-attempt profile can only report MMD, self-BLEU and
dispersion as insufficient samples. Token features are always requested because the
n-gram measures are cheap and local; embeddings are separate, and leaving them off costs
D2 and D6 rather than the whole profile. `MINIMUM_SAMPLES` withholds a measure below its
floor with the required count in the reason, `RELIABLE_SAMPLES` marks where two models can
be compared, and `sampling_plan` resolves a proposed attempt count against both so a run
can be sized before it is paid for. Repeated identical outputs are retained, since they
are what the duplicate-rate measure exists to detect.

`references.load_matched_references` validates frozen development reference texts
and scenario-bound assignments. The pilot rescorer records broad and task-selected
profiles separately; partial matches remain explicit. Both tracks share feature
configuration and broad-reference bandwidth, without treating passages as repeated
candidate generations.

`anthropic_grading.AnthropicGrader` caches complete direct-API rubric responses and
serializes spending through a locked, persistent reservation ledger. Unresolved
requests require inspection before retrying. Credentials enter only at execution
through an explicit argument or `from_env`; they are excluded from saved requests.
Pricing is pinned to Sonnet 4.6. The external Creative Writing research script keeps
its prompt expansion and aggregation separate from the custom harness scorecards.

`external.generate_tasks` runs public single-turn benchmark prompts with the shared
Transformers backend. Its generation manifest binds task selection and model
configuration; completed and failed outputs are both resumable. External benchmark
graders retain their own protocols and denominators, rather than entering custom
suite scorecards. Coding execution belongs in the isolated EvalPlus container.

`longform.py` binds the pinned EQ-Bench Longform Writing release as the one benchmark
we did not design, which is what lets it separate "our reward moved the policy" from
"our reward moved the policy toward our own taste." It renders the upstream 13-step
plan-then-eight-chapters protocol, compiles twelve `final_eval` scenarios through the
unmodified `compile_scenarios` contract, and reimplements the upstream criteria,
weights and arithmetic. It is an adaptation, not a leaderboard entry: the staccato
penalty is the continuous curve upstream's own comment describes rather than its
discontinuous implementation, parsed metrics are filtered to the declared criteria, and
local sampling and context differ. `faithful` mode keeps generation reply-only so a
score stays comparable; `workspace` mode adds file delivery and is therefore not
comparable. Quote a score with its mode and chapter count, never alone.

`longform_suite.py` compiles the held-out long-form benchmark into that same scenario
contract. Supplied manuscripts are deterministic slices of three public-domain works,
and every case declares anchors that the build verifies occur in the text it supplies,
so a private check cannot cite a fact the candidate was never given. `holdout_audit`
refuses to build when a benchmark work is already claimed, by content hash, by training
or by a source a development case references. This suite is authored from the same
taxonomy as the training tasks and the reward, so it can show that a policy moved and
not that it improved; the external anchor is the half that can falsify. Do not run it
for checkpoint selection.

`CodexGrader` replaces built-in coding instructions with `grader_instructions.md`,
starts a fresh ephemeral session outside the repository, suppresses project/skill
instruction loading, and preserves launch evidence. Packet/cache identity includes
the instruction content. Its persisted call budget is locked across workers.
Semantic judgments remain uncalibrated until human review; original deterministic
and numerical measurements are retained when judgments are applied.

Prose extraction version 2 distinguishes absent required delimiters (missing
delivery) from multiple delimiters or invalid reviewed spans (needs review).
Missing delivery fails completion; ambiguous extraction stays pending. Successful
prose text/spans and numerical feature definitions are unchanged by this distinction.

Scorecards carry a shared `candidate` identity (`harness`, `provider`, `model`)
for all their measurements and prose profiles. Provider identifies the serving
route; locally loaded Google weights use provider `local`, not `google`. Keep the
full model configuration alongside this identity and judge identity separately.
Reports group by candidate identity as well as full model configuration, so
identical model names under different harnesses or providers are not pooled.
Legacy records infer only known routes; unspecified providers remain `unknown`.

`training.py` prepares accepted grouped trajectories and exposes explicit QLoRA
execution. Native template annotations must preserve rendered bytes, supervise
assistant text/tool calls/endings, and mask embedded observations. Preparation
rejects overlength data, reasoning fields, special-token input, and known evaluation
source groups. Saved token labels are preserved by the TRL collator. Training
requires explicit execution and matching prepared hashes; no benchmarks run from
the trainer. SFT GPU training and checkpoint restore remain unverified.

`grpo_runtime.py` owns explicit TRL implementation admission. Legacy TRL 1.13
remains the default; opt-in `trl-6c5f135-streaming` verifies the exact approved
TRL/Liger Python source trees before model loading or caller mutation and binds
them into experiment identity. It admits dense Gemma4 only and disables unrelated
Liger model replacements through public configuration. The maintained FP32
streaming softcap is an accepted numerical variant, not native BF16 parity.
See [GRPO usage](../../docs/grpo.md) for qualification scope and source pins.

`grpo.py` owns legacy experiment admission, frozen settings/identity and TRL config;
`grpo_trainer.py` owns the shared resume/quarantine, checkpoint callback, trainer
construction and adapter-export lifecycle exposed as `run_trainer(...)`. It does not
reuse SFT preparation or implement another RL loss. `grpo_rollout.py` owns append-only
sampled tokens and external suffix masks. Do not rebuild training actions by rendering
parsed messages: Gemma can reorder tool arguments and remove earlier thinking.
Training identity includes private scoring labels, unlike evaluation's rescorable
identity. `grpo_identity.py` checks catalog lineage and actual caller-owned base tensors
before resume can mutate the model, and owns the selected PEFT adapter tensor hash used by
trainer policy bindings and trace checks; engineered fixtures use separate, explicit admission.
Unavailable groups always stop before updates. Identity-bound `tie_policy="halt"`
also stops ties by default; explicit `"continue"` passes raw tied rewards through
ordinary TRL/Adam without resampling. Mathematically zero advantages can have
float32 residuals; these or momentum may move weights. This is not update skipping.
Saved `trl_advantages_estimate` values are Python-formula estimates, not observed
trainer tensors. Reward scaling is explicit and identity-bound (`group` by default;
task-graph training selects `none`). Groups record ties separately from checkpointed
optimizer progress. `GRPOSettings.microbatch_size=None`
trains the full group with accumulation 1. An explicit microbatch must be a positive
integer dividing `group_size`; `gradient_accumulation_steps` is derived as
`group_size // microbatch_size`. Reward-group size is distinct from training microbatch size: TRL scores the complete group, consumes its slices within one
accumulation window, and updates once. Checkpoints occur only at that boundary; no
partially consumed rollout buffer needs restoring. Microbatch settings are identity-bound.
`loss_type` is also identity-bound: `grpo` remains the default, while explicit `dapo`
uses public TRL's generation-group active-token denominator, excluding observations
and padding. Neither selection changes sampling, reward admission, or safety budgets.
Inference adapters and full trainer checkpoints are different artifacts. CPU optimizer/resume verification does
not establish Gemma GPU fit. See [GRPO methodology](../../docs/grpo.md) for the bounded
execution, recovery, and caller-owned reward contracts.

`grpo_probe.py` is a fixed engineering recipe over that trainer, not a general experiment
scheduler. `grpo_probe_data.py` owns its committed source packet, bounded derivatives,
and mechanical-only scorer; fixture successes are not sampled model successes or
literary judgments. Preparation and tokenizer evidence bind the package sources before
any model phase. Source changes therefore require fresh preparation, not rescoring an
old run. See the [probe guide](../../docs/grpo-probe.md) for phase admission and recovery.
The ordinary inference backend applies a trajectory token cap only when explicitly
configured; older callers retain their per-call budget.

`grpo_full48.py` admits the intact wave1 training release by frozen hashes and owns
its separately bound mechanical-only reward. Delivery requires completed sequence
and action evidence, all required artifacts at their lower word bounds, and actual
file changes. Other mechanical failures can retain partial reward. Semantic rubrics
and intermediate clarification faithfulness remain unjudged. Its fixture module
constructs offline counterexamples; it never establishes sampled success or memory
fit. See [full48 preparation](../../docs/grpo-full48.md); this is not a training runner.

`task_generation.Sampler.build` validates a selection once and derives its
index-addressable content; `iter_requests` streams from it, `build_request` returns the
one at an index, and `prepare_task_requests` materializes a batch with a coverage
summary. Requests reference their source by identity and hash instead of embedding the
passage, so a batch costs O(requests) rather than O(requests x source size);
`task_authoring.resolve_packet` inlines the passage for a model call or validation, and
a batch resolves each request once before spending. Each request is derived from its
index and seeded per-key permutations rather than shared RNG state, so a batch can
resume mid-way. Only selected human training sources are accepted; content hashes and
connected-lineage exclusions are checked, and source groups are reported separately
from work counts. `specificity.py` gives each decision point one behavior and one
introduction level, and derives each family's stated and withheld split from that
table; an inconsistent table is rejected at import. Coverage reports the level and
withheld points.
Prepared requests are not generated or accepted training tasks.

`reward.py` is the training-side scalar, and it is the only place a combined writing
score is computed; `scoring.py` must keep computing none. Version 0 weights quality,
intent and continuity as anchored 1-5 ratings and mechanics as the mean of the task's
applicable mechanical checks, so an inapplicable check earns no free credit and a
repeated check counts once. A withheld judgment leaves the reward unavailable rather
than zero, because a judge timeout and a bad draft are different events. A critical
criterion counts only when it was declared before sampling, so a rubric weakness cannot
be promoted after the answer is seen. Group advantages are the within-group
standardised rewards; they cancel a per-prompt judge offset but not rank flips or
length bias, and an all-tie group yields no signal and is reported rather than hidden.
The research script binds their hashes to source inventory and generator instructions.
Variation vocabulary is caller-supplied data.
Source-preserving continuations receive no genre blend and retain source style.
Tropes, situations and continuity challenges remain authoring suggestions until the
generator grounds them in a visible task. Assigned coverage is not realized coverage.

`task_authoring.resolve_packet` inlines the passage a request refers to and verifies
its identity and hash. Model calls and `validate_task` receive resolved requests;
stored outcomes and compiled scenarios keep the reference form and only the source id.

`task_authoring.author_tasks` turns frozen requests into the existing scenario format.
Structural admission and an independent task-review call precede compilation; model
review is not human acceptance or a successful writer trajectory. Outcomes retain
raw candidates and review evidence. Resume rejects changed inputs or cached outcomes.

`paid.PaidClient` owns the shared paid-call ledger for task authors, reviewers and
`OutputGrader`. Each call has fresh messages over the DeepSeek Flash direct route;
response identity includes instructions, payload, role and thinking. Task author
JSON calls disable thinking with a 16384-token budget; reviewer calls disable
thinking with 8192. Thinking stays off because reasoning shared the output budget
and truncated larger multi-stage tasks. Evidence quotes match after collapsing whitespace,
including newlines. Admission requires the seven named task fields; extra
top-level keys are ignored. Uncertain charges retain reservations and block new
requests. Interrupted reservations stay blocking until explicitly reconciled
to a terminal `abandoned` accounting key that keeps the charge and frees the
identity for a new reservation. Overrun still halts. Accounted spend never
decreases.
Keep the same ledger when revising a batch. Charges come from DeepSeek token usage at the
peak cache-miss ceiling, never from a provider cost field; missing cache
breakdown treats prompt tokens as misses, and incomplete usage keeps the
reservation. Transport failures write an inspectable error artifact beside the
reservation and halt. Cache-hit tokens are charged at the cache price when usage
reports them. Raw responses retain reasoning separately from candidate prose. `OutputGrader` uses
the existing blinded packet and judgment application; failed judgments leave
scores pending.

Grading packet version 3 includes each completed turn’s reply and file snapshot, so
planning and earlier revisions remain assessable after later stages replace them.
Prose quality still uses only designated prose selections.

`grpo_full48_runner.py` selects intact release/reward bindings and an explicit
`intact-full48-v1` admission profile. The default probe profile keeps its original
limits and 32-seed stride. Full48 reserves 48 seeds per slot, validates ordered
visits before sampling, pauses at pass one, and refuses recovery with uncommitted
sampled groups (generic trainer recovery may resample; this recipe may not).
Coverage counts optimizer progress only from complete hash-verified checkpoints.
`grpo_full48_supervisor.py` owns the inherited writer lease and advisory process
progress; quiet output never triggers termination. Preparation/preflight import no
model stack. See [full48 usage](../../docs/grpo-full48.md) for frozen allocations
and their limits; CPU schedule proof does not establish native training fit.

`grpo_gpu.py` owns the display allowance policy and desktop-consumer check as well as
complete graphics/compute NVML inventory admission for production fit and full48
train/resume. `grpo_probe.py` re-imports the display policy and check. Prepared identity
and pinned source admission precede ownership; ownership precedes model loading.
`grpo_gpu_fit.py` owns a separate single-attempt controlled token-ledger
profile and native prefill check. Generation and training use sequential fresh
processes so ownership never exempts an existing CUDA context. Controlled ledgers
are memory evidence only; production rollouts remain native sampling. See
[fit usage](../../docs/grpo-gpu-fit.md) for coverage and limits.
