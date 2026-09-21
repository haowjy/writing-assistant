# Evaluation boundaries

The research entry point is [evaluate.py](../../scripts/evaluate.py). It selects
Python configuration and calls library stages independently. The existing CLI
retains its smoke-evaluation and training-format workflows.

## Responsibilities

- [suite.py](../writing_agent/suite.py) compiles scenarios and owns serial execution,
  isolated workspaces, resume identities, saved results, and fresh-reader conditions.
- [agent.py](../writing_agent/agent.py) owns the bounded conversation/tool loop;
  [workspace.py](../writing_agent/workspace.py) owns file operations and storage limits.
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

Catalog records carry normalized content in `text`; local imports also preserve raw
bytes and a text file, with hashes for both representations. Content from the imported
file replaces any stale text supplied in metadata. Profile and overlap consumers use
the same canonical text.

Compiled releases separate visible packages from private labels. Runtime workspaces
contain only initial files supplied by the visible package. These path-constrained
file tools are not an OS sandbox; no arbitrary shell is exposed to candidates.

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
