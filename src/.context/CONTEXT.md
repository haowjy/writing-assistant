# Evaluation boundaries

The research entry point is [evaluate.py](../../scripts/evaluate.py). It selects
Python configuration and calls library stages independently. The existing CLI
retains its smoke-evaluation and training-format workflows.

## Responsibilities

- [suite.py](../writing_agent/suite.py) compiles scenarios and owns serial execution,
  isolated workspaces, resume identities, saved results, and fresh-reader conditions.
- [agent.py](../writing_agent/agent.py) owns the bounded conversation/tool loop;
  [workspace.py](../writing_agent/workspace.py) owns file operations and storage limits.
- [task_graph.py](../writing_agent/task_graph.py) owns immutable task-graph value records
  and their approved identities. [task_graph_store.py](../writing_agent/task_graph_store.py)
  owns private content-addressed persistence, reference closure, checkpoint
  materialization/restore/branch/diff, recorded Phase 2 effect replay, and the atomic
  mutable lineage head. [task_graph_contracts.py](../writing_agent/task_graph_contracts.py)
  adds detailed execution contracts as immutable artifacts referenced by the frozen
  Phase 1 node shape; [task_graph_admission.py](../writing_agent/task_graph_admission.py)
  resolves their public/private closure and rejects an unsound graph before sampling.
  [task_graph_controller.py](../writing_agent/task_graph_controller.py) is the pure,
  deterministic directive boundary. It does not step a writer, execute checks, accept
  author transition/completion claims, or mutate runtime state.
- [task_graph_writer.py](../writing_agent/task_graph_writer.py) is the opt-in Phase 4
  writer/text-tool stepper over an admitted ready-writer entry and trusted restored
  handle. [task_graph_projection.py](../writing_agent/task_graph_projection.py)
  reconstructs writer context from the entry checkpoint and authorized causal
  events; it never renders the private event log wholesale. A capable adapter calls
  `prepare_request` before sampling to pin exact request/context evidence; the caller
  then supplies parsed writer output and any raw-output/trace evidence. Neither module
  invokes a model. [task_graph_scripted.py](../writing_agent/task_graph_scripted.py)
  owns exact scripted author requests/replies, disclosure and authorized requirement
  updates; [task_graph_checks.py](../writing_agent/task_graph_checks.py) freezes and
  checks candidate checkpoints; [task_graph_terminal.py](../writing_agent/task_graph_terminal.py)
  applies terminal guards and publishes immutable outcome/reward evidence. All three
  use the writer projection's same prepublication and recovery semantic walk;
  [task_graph_author_validation.py](../writing_agent/task_graph_author_validation.py)
  holds the author-side replay rules. See
  [writer stepping](../../docs/task-graph-writer.md) for the entry budget artifact,
  restore API, and the Phase 4 boundary.
  [task_graph_compaction.py](../writing_agent/task_graph_compaction.py) owns the
  fixed visible-message summary, complete-exchange selection, immutable context
  operation evidence, and context byte accounting. The writer publishes a
  `context_changed` operation only at a drained `ready_writer` boundary; the
  projector validates its selection and budget before publication and on recovery.
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
  atomic JSON artifacts, and overlap inspection. [acquisition.py](../writing_agent/acquisition.py)
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
replacing the head under the lineage lock. Unreachable files are harmless orphans.
Restore always creates a new private directory and returns a trusted runtime handle;
it never rewinds an event log. The Phase 2 replay reducer accepts only persisted
`Phase2RecordedEffectV1` payloads and makes no model, network, or tool calls.
Publication reduces that supported event batch from the actual parent checkpoint and
requires typed reference closure after every intermediate effect as well as exact
complete-state equality before writing or moving authority. The effect's
`before_state_ref` is a state-identity assertion, not a persisted-object edge.
Unsupported transition envelopes fail at the reducer seam. Branch initialization is
the same recorded parent-to-child transition, not a pre-mutated state.

Phase 4 publishes each `writer_action` or `tool_result` with a following
`context_changed` event in **one** commit. The source event's recorded effect updates
the queue/files/budgets and metadata index; the second effect updates `context_ref`
after the revision can name the already-hashed source event. This preserves the
approved content/provenance split without changing the Phase 2 reducer or any frozen
identity. A tool-only exhausted turn may append `termination_recorded`; a final
reply enters `checking`, not automatic task completion. Writer-produced
`budget_charged` and `termination_recorded` use the explicit `writer_runtime` actor
and require their runtime-log entry even as a first event; generic Phase 2 events
retain their ordinary source. Native token loss masks remain unimplemented.

The scripted-author mode is opt-in and separate from the legacy fixed-followup
compiler. Admission resolves role-typed private author/evaluator packets, the public
decision policy, exact private script, check programs and reward weights before a
writer turn. An `ask_author` action is committed before its private request; a
replayed/restored outstanding request produces the same author reply without a
provider call. Request and reply boundaries are durable; only the paired tool
acknowledgement and explicit author utterance enter writer context. Final writer turns freeze an
immutable candidate before deterministic file checks. Check results name that
candidate, admitted check and evaluator packet, requirement version and recomputed
evidence. Environment transition and terminal outcome records remain separate from
reward availability and training eligibility. `task_graph_projection.py` validates
every new producer's exact field/actor authority and causal binding both against
staged events before publication and on restore/replay. The evaluator cannot edit
files or the terminal task/execution status; reward arithmetic is exact integer
normalization. Model-backed author, semantic judge, learned/model compaction,
and native on-policy eligibility are not implemented.

The immutable admitted entry contract, not a child state field or runtime log,
anchors Phase 5 semantic recovery. Its event dispatch is closed; other generic
event kinds reject before publication and on restore. A writer-request reply
commits its exact acknowledgement, disclosure, author turn and final context
together, with a drained cursor before the request clears. Admission screens
public decision IDs and labels as disclosure surfaces and allows only bounded
ASCII IDs. Exact-byte canaries are a guard, not semantic secrecy. Tool and
author eligibility is decided before dispatch: tool exhaustion wins, then
malformed syntax, then author exhaustion. Attempted calls charge once; an
exhausted tool counter never authorizes an acknowledgement. Terminal reward
components must have terminal check scope. Writer turn and measured token
exhaustion in this slice produce typed incomplete outcomes with a frozen
checkpoint and declared reward availability, not legacy untyped stops.

The graph writer uses a separate tool dispatch boundary so expected writer-operational
path/patch/policy failures can be observed without converting disk, permission or
corruption failures into writer mistakes. Raw call-array elements normalize to safe
queue syntax and escaped evidence; each declared malformed call remains paired and
charged without dispatch. Current lineage authority is checked before staging and
again by publication CAS. Producer, restore, projection, and replay use one complete
history-aware Phase 4 semantic walk (including exact per-event state/history ownership,
legal phases and stop statuses, independently derived action/result ordinals,
record/trace/prepared/log bindings, origin, execution fingerprints, file delta,
queue cursor, and budget charges). The producer stages immutable candidates and
validates them before publication; invalid candidates leave the head unchanged.
The generic Phase 2 reducer stays unchanged. Argument JSON and
raw call evidence have size/nesting bounds before decoding or serialization.
Known zero token capacity rejects request preparation and can be sealed as a classified
terminal stop. Already-sampled token overrun commits usage/output evidence and a
terminal budget stop without tool execution. Exact pre-sampling context-token capacity
and non-whitespace read counting are not supported and fail at runtime initialization.

Reference closure uses one operation-scoped typed traversal with loaded, active, and
completed sets. Checkpoint/commit/event depth is traversed iteratively; supplemental
and imported references cannot bypass the validators for their resolved record domain.
There is no process-global validation cache. Immutable-name and ref-name retries repeat
their containing-directory durability barrier even when equal bytes or the matching
head are already visible; identical `branch` retries use that same head-repair path.
Every non-null authority head must target a checkpoint in its named lineage. A first
commit may explicitly start from a checkpoint in another lineage, but later commit
ancestry may not cross lineages.

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
memory-efficient SDPA is the only O(n) path and the cost of length is time, not memory.
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
and controller/check versions without invoking a model. Phase 3 supports only `none`
and fixed `scripted` interaction: `simulated_author` and mandatory feedback fail closed
until their role-specific packet, policy, binding, and feedback-rule contracts exist.
Check admission uses exact evaluator-version-specific program schemas; semantic-v1 is
not an admitted evaluator. Mapping and store-backed admission share the same typed
closure and visibility rules. `DeterministicControllerV1`
returns only `request_author`, `continue_writer`, `propose_edge`, `stop_incomplete`, or
`wait_checks`; it checks writer exhaustion itself and permits a no-edge continuation
only when the admitted repair contract, runtime authorization, and remaining writer
budget all allow it. Author text is audit input and is never inspected for routing. Outcome
records keep task, execution, stop, reward, and training-eligibility state independent.

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
the trainer. GPU training and checkpoint restore remain unverified.

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
