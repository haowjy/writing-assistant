# Evaluation design

This document defines the [custom suite](index.md). Implementation and verification
status are recorded in [delivery evidence](delivery.md). The unit of evaluation is a scenario executed as a
session. A scenario defines the brief, available material, tools, budgets,
output destination, and private checks. A session records one candidate's attempt.

## Tasks and controlled variations

| Family | Initial task coverage | Variations to introduce after the first pilot |
|---|---|---|
| F1 | Draft a scene; continue supplied prose; follow a style/POV brief | Short/long context, dialogue/action/reflection, explicit and implicit constraints |
| F2 | Create a requested manuscript file; revise a bounded passage; explore before writing | File layouts, tool availability, stale notes, local/global revisions |
| F3 | Give three distinct directions; expand one; incorporate author feedback | Plot, viewpoint, tone, pacing, character choices; prose plan vs plan saved to files |
| F4 | Build a continuity KB; interpret a character; update after an accepted revision | KB purpose, detail, representation, uncertain claims, aliases, changing relationships |
| F5 | Write a scene using a reviewed KB; continue without premature reveals | Supplied facts vs tool retrieval; reply vs saved file; required vs optional knowledge |

Apply instruction adherence to every family when a brief includes constraints.
A no-tool task runs with an empty tool set. An optional-tool task
must permit a valid direct response. Do not prescribe one call sequence when
several strategies achieve the requested result.

Keep the model and harness distinguishable. Initially use the existing local
harness with a fixed prompt/protocol per model class. Record base completion and
IT chat formatting separately. Later harness ablations may vary prompts, tool
schemas, and workspace layouts one factor at a time. No AgentGym fork is required
for this first design.

## Separate KB construction from use

1. **Construction:** source passages and author decisions enter a builder session.
   Score the resulting KB for selection, interpretation, support, organization,
   and updates. Keep downstream evaluation questions hidden from the builder.
2. **Use:** a candidate receives a reviewed KB and a writing request. Compare
   supplied relevant context with retrieval from the full KB to diagnose retrieval
   versus writing failures.
3. **Combined:** a fresh session uses the candidate-built KB. Reset conversation
   history and copy only allowed KB artifacts. Score both intermediate KB and final
   prose. A fixed reader configuration provides a comparable navigation probe;
   later test another reader to check whether the result depends on one model.

A builder's omission should not automatically count as a writer's mistake. Report
whether required evidence was absent from the KB, present but not retrieved,
retrieved but misinterpreted, or understood but contradicted in prose. These are
failure annotations, not extra headline metrics.

For format-only comparisons, render the same reviewed content into each format.
For construction comparisons, provide the same source and format requirement,
then let the model select and organize content. Do not mix these experiments:
otherwise formatting and knowledge loss become inseparable.

Start with linked Markdown and a flat Markdown control on a small subset. Add
structured entity/event records and hybrid references later. Define supported
link/anchor syntax and entry points. Score facts across representations; run
format-specific integrity checks separately. Compare navigation-only access with
search-enabled access as distinct conditions, using fixed read-token and tool
budgets. Count repeated reads against the budget. A giant page is not free to read.

## Knowledge selection and interpretation

Each construction brief states its purpose, audience, scope, source cutoff, and
storage budget. A continuity KB should preserve consequential knowledge without
copying every sentence. A local detail may belong in a chapter note, a global
reference, or nowhere, depending on purpose.

Private labels distinguish required information, useful optional information,
incidental details, and unresolved interpretations. Support multiple defensible
interpretations and placements. Do not mark an unlisted but valid fact as false.
Store evidence spans, temporal scope, speaker/belief owner where relevant, and
acceptable qualifications. These are evaluator records; the model need not emit
a rigid schema inside its freeform wiki.

For example, an accusation may be incidental and safely omitted. If it drives a
plot thread, retain it with its attribution and uncertainty. If later accepted
prose establishes the truth, update the appropriate pages without confusing a
historical belief with current canon. A new draft alone does not authorize this
update. Interpretation scores assess support and qualification, not agreement
with one critic's reading.

New prose may invent permitted events, dialogue, and description. Grounding does
not mean every generated sentence must already exist in the KB. Labels separate
established constraints, requested changes, allowed invention, and deferred facts.

## Record contracts

Use versioned JSONL manifests and ordinary files. Keep raw sources, scenarios,
execution results, and judgments distinct; they change independently.

| Record | Required information |
|---|---|
| Source | Stable source/work/author/series IDs; URL and revision; original split; content hash; acquisition date; terms evidence; allowed uses; artifact provenance label, parents, and transformations |
| Scenario manifest | ID, schema version, family, source group, role (`train`, `development`, `final_eval`), condition, source cutoff, visible package hash, private-label hash, review status |
| Visible package | Brief, initial files, scripted user turns, available tools, budgets, output destination, permitted edits |
| Private labels | Constraints and severity, relevant evidence, required/optional information, acceptable interpretations, hidden probe tasks, scoring method per check and checker/rubric versions |
| Session result | Model/revision and quantization, protocol, generation settings/seed, harness hash, tool trace, per-turn usage/timing, before/after artifacts, status and errors |
| Judgment | Session/artifact hash, metric ID/version, score/status, evidence, rationale, uncertainty, judge identity/settings, raw response and human adjudication |

Destination is explicit: `reply`, `file`, or a declared set of artifacts. Score
F2/F5 saved prose from the file snapshot, not from the assistant's claim that it
wrote the file. Missing or empty required artifacts fail task completion; no prose
score is fabricated for missing content.

### Selecting prose from replies and files

Prose metrics consume an extracted artifact, never the whole conversation or
workspace. Each scenario declares which requested writing counts as prose:

- **Reply prose:** select the designated assistant turn and prose region. A
  prose-only response can use its whole body. A mixed response needs an explicit
  section/delimiter rule or reviewed span annotations to exclude explanations,
  planning, critique, and introductory or closing commentary.
- **File prose:** select the declared manuscript path and snapshot. For revisions,
  select the requested passage or final scene according to the scenario contract;
  preserve the distinction between new/rewritten prose and unchanged source text.
- **Multiple artifacts:** retain separate IDs for each requested scene or
  alternative. If prose is repeated in a reply and a file, designate one copy for
  distribution metrics so it is not counted twice. Score conflicting copies as
  separate delivery/completion evidence where the task requires them to agree.

Record extraction method/version, session and turn ID or file path/snapshot,
source artifact hash, selected character spans, and extracted prose hash. Keep
raw artifacts so extraction can be corrected and all affected metrics recomputed.
Intermediate drafts are included only in a declared draft-comparison condition;
otherwise score the designated final output, not whichever draft scores best.

Use deterministic selectors when the task format permits them. When a freeform
reply has ambiguous boundaries, flag it for reviewed segmentation; do not silently
score the entire reply or treat extraction uncertainty as poor prose. An optional
model-assisted segmenter must be labeled and audited separately from numerical
scoring. Code blocks, Markdown, or a file extension alone do not identify prose.

Keep separate statuses for missing prose, invalid/ambiguous extraction, and a
valid prose artifact. Reports show counts and reasons for unscored artifacts.


Examples used in tests are synthetic development fixtures, never final evaluation
items. Existing trajectory `train/validation/test` fields remain training-format
metadata; a compiler must explicitly map roles and preserve upstream splits.
Do not silently reinterpret arbitrary upstream `test` rows as training data.

## Runtime boundaries

| Concern | Implementation | Experimental prerequisite |
|---|---|---|
| Workspace and tools | Configurable tools, read/storage budgets, snapshots, Markdown link diagnostics | Choose and record a read tokenizer for model comparisons |
| Model execution | Scripted follow-ups, saved traces, explicit model configuration | Direct Transformers/PEFT checkpoint loading; verify each prompt/precision condition on the 3090 |
| Evaluation | Visible/private compilation, artifact selection, resumable attempts and independent rescoring | Review 50 proposed scenarios and private labels |
| Data | Pinned source acquisitions, one provenance catalog, source grouping and overlap audit | Review reference selection and source-specific reuse terms |
| Grading | Bounded Codex adapter, validated judgments and human-review packets | Calibrate semantic rubrics against human review |

Keep these concerns cohesive rather than creating a framework of adapters for hypothetical
backends. Separate deterministic scoring from remote judging so outputs can be
rescored without generating again. Introduce a new module only when it owns a real
independent responsibility. Keep terminal commands, JSONL results, and Markdown
reports sufficient; no dashboard or service is needed.

The candidate sees only its visible package. Private labels, hidden probes,
reference KBs, and grader outputs live outside its workspace. The current tools
provide path confinement, not an OS sandbox. Do not add arbitrary shell execution
to authoring tasks. External coding tests must use an isolated runner suitable
for executing generated code, with no credentials or evaluator files exposed.

Tool and manuscript text are untrusted data for the grader. Give the grader
source evidence and a fixed rubric, never permission to obey instructions embedded
in the artifact. Grade snapshots read-only. Record invalid judgments and retry
history rather than silently replacing them with successful scores.

Persist prose-only feature records (counts and embeddings) keyed by artifact and
configuration hashes, and group-level distribution comparisons with reference
manifest IDs. Compute them independently of semantic grading; see the
[prose-distribution diagnostics](metrics.md#prose-distribution-diagnostics).

The primary entry point is `scripts/evaluate.py`, backed by reusable Python functions.
Preparation, generation, scoring, and reporting are independent stages. Default
execution inspects the proposed experiment; generation requires explicit enabling.
Keep caching/resume in the library. Preserve the existing CLI without adding commands.

## Instruction specificity

The development collection contains five explicit and five loosely specified
initial requests per family, recorded as `instruction_specificity`. Loose requests
leave creative choices open while naming an observable deliverable. For example,
“Could you turn this into a scene? I'd like it to feel more alive” leaves length,
viewpoint, ending, and style to the agent. File destinations remain specified so
artifact extraction can identify the requested manuscript without guessing.

Loose requests must not retain hidden checks for the detailed prompt's word count,
viewpoint, exact idea count, protected ending, or wiki size. Continuity checks allow
plausible new events and revelations. KB labels offer candidate salient facts and
accept defensible omissions. Fixed follow-ups can still introduce explicit user
feedback or accepted changes. Report specificity separately; these initial groups
use different worlds and are not a controlled causal comparison of prompt wording.

Clarification-dependent tasks are a separate planned condition. They need a user
response policy for material missing information, alternative valid answers, and
scoring that distinguishes useful questions from unnecessary delay. The current
fixed-follow-up runner does not implement that policy. Do not add underspecified,
unanswerable tasks and then grade failure to guess a private preference as poor
instruction following. A later matched-prompt comparison should keep the underlying
world/task constant and count additional attempts explicitly.

## Genre distribution

The 50 cases assign ten genres once per family, five cases per genre overall.
Each genre spans both instruction-specificity groups. [Genre contexts](../../data/scenarios/genres.json)
add visible story information to passages and KBs, with derivative source records
linked to their original world. Prose-style descriptors are recorded separately.
Genre blends are acceptable when the original premise supports them; review still
needs to establish whether these short cases represent their genres convincingly.
See the [coverage matrix](coverage.md) for exact assignments and limitations.
