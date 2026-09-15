# Branching authorship dataset and harness

Build a dataset of writing sessions that branch from existing fiction. Reconstruct
notes at a point in a book, then ask for continuations, revisions, or changes in
style—the “make fanfiction” experiment. Collect sessions through a harness that
executes file operations and records the conversation for training and evaluation.

The [RL task and reward plan](../sft/rl-task-generation.md) extends this into an
on-demand training-task stream: source-grounded branch requests, initial files,
private grading evidence, and reproducible variations. The same five benchmark task
families guide training, while works and instances remain separate from evaluation.

## Sources and story checkpoints

Look for existing outlines, notes, drafts, revision histories, and collaborative
writing sessions. For each source, inspect samples and record its license,
author, work, and human or generated origin. For Gutenberg books, also preserve
the edition identifier and passage boundaries.

Choose a chapter or scene boundary. From the preceding text, reconstruct character
notes, established facts, chronology, unresolved threads, and chapter summaries.
Retain recent prose for voice and local context. Mark reconstructed notes as
synthetic.

Create two kinds of task:

- **Continue from prior context:** derive notes only from text before the checkpoint.
- **Write from a plan:** also supply a limited outline reconstructed from the
  original next passage. Record which future events the outline supplies.

Separate accepted history, author plans, character knowledge, and information
that should remain hidden from the reader. Keep future source prose and private
scoring criteria outside the writer's workspace.

## Branches and styles

Vary character decisions, plot direction, viewpoint, tone, narrative distance,
pacing, dialogue emphasis, and how much the prose explains. Include requests to
preserve the source voice and requests to change it.

An author might request three alternatives, choose one, ask for a scene, revise
it, and accept only part. Established history remains binding unless the author
requests a change. Once the branch diverges, later events in the source book
become optional.

Include paired examples that change plot direction while holding style constant,
and others that change style while holding the plot constant. Assess whether
prose achieves the requested effect. Choose the style categories and data mixture
after reviewing the pilot.

Give each branch its own workspace and decision history. Keep every checkpoint
and variant of a work in the same dataset split; group by author where feasible.
A Gutenberg holdout may already occur in a model's pretraining data, so include
fresh or private evaluation material too.

## Collecting training sessions

A scenario specifies starting files, available tools, author requests, output
locations, and private scoring criteria. An evaluation session is one model
running that scenario across conversation turns and tool calls.

For SFT demonstrations, run a permitted teacher model through real file tools. Record requests, assistant responses,
tool calls and results, revisions, approvals, and final files. Use scripted author
turns initially. Where feedback must reference generated content, use an author
simulator with a recorded model version and private brief. Replay sessions to
verify file changes, then review them before accepting training records.

For tasks following the original plot, the source passage can serve as the
human-written target. Construct and verify a compatible conversation and tool
sequence around it. Alternative branches need new prose; label human and synthetic
targets separately and compare their effects during training.

For RL, the current writer policy generates attempts in the real harness; score
those attempts with task-specific mechanical checks and semantic rewards where
validated. A teacher-written ideal continuation is not required for every RL task.

A short SFT bootstrap learns assistant responses and tool calls from accepted recordings.
The [supervision setup](training-experiments.md#supervision-setup) describes loss
masking. Preference or live-rollout training uses the environment once
its rewards are reliable. Vary file layouts and tool configurations to test whether
learned behavior transfers. The existing Python workspace harness is the starting point for RL integration.

## Evaluation and pilot

Run evaluation with frozen model weights. Score tool use, instruction following,
state changes, and prose separately; assess wiki organization when the task calls
for it. Each scenario identifies the chat response, file, or revised passage to
score. Use the original continuation as a reference only when the task calls for that
comparison. A requested alternate event or genre overrides incompatible original
canon; textual overlap and narrative departure are scored separately.

The pilot should contain source samples and a few reviewed checkpoint records,
then demonstrate a complete session: propose alternatives, choose, write, revise,
accept, update notes, replay, and export. Inspect the records and model behavior
before expanding collection.

See [baseline comparison](baseline-comparison.md) for the Gemma 4 evaluation and
[data experiments](data-experiments.md) for data mixtures and ablations.
