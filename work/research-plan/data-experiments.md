# Data experiments

## Objective

Construct examples that teach conversation, project-state use, planning, writing, and revision.

The [branching authorship pilot](branching-authorship.md) applies this plan to
existing fiction, with reconstructed notes, alternate continuations, and style variation.

Each record combines:

```text
multi-turn author conversation
+ external project state
+ available tool schemas
+ tool calls and observations
+ planning/reasoning
+ user constraints
→ requested writing/planning/editing artifact
```

Human prose should be used as the final target wherever possible, while synthetic models reconstruct the **authoring process around it**.

## Data Sources

See the [dataset inventory](dataset-and-metric-research.md) for existing stories,
authoring interactions, planning material, and access notes.

Prefer public-domain fiction, permissively licensed fiction, author-contributed work, commissioned/consenting contemporary fiction, the researcher's own writing, and curated high-quality prose with clear provenance.

Use caution with scraped web novels, contemporary copyrighted fiction, and datasets with unclear redistribution/training rights.

Especially valuable authoring-process data includes brainstorming chats, outline revisions, discarded ideas, chapter plans, character sheets, worldbuilding notes, wiki edits, consistency corrections, and decisions about what should remain hidden.

## Reconstructing Story State

For each source work, reconstruct a causal project state containing characters, world, plot, timeline, chapters, notes, unresolved threads, and local scene goals. Exclude information unavailable at the time of the request. Specify how author plans differ from character knowledge and reader-visible facts before generating examples.

## Representation Randomization

Generate multiple equivalent representations of the same semantic state. Vary:

- file format;
- file naming;
- directory hierarchy;
- normalization;
- redundancy;
- stale files;
- irrelevant files;
- duplicated information;
- conflicts requiring reconciliation.

Test generalization to unseen project representations.

## Tool-Schema Randomization

Canonical generic tools may include:

```text
list_dir(path)
read_file(path)
search(query, path?)
write_file(path, content)
patch_file(path, patch)
```

Moderately vary names and schemas, and vary which tools are available. Include no-tool-needed cases.

## Conversation Types

Train on:

- direct continuation;
- brainstorm → select → write;
- revision;
- plan maintenance;
- continuity repair;
- critique-only;
- state maintenance;
- ordinary discussion where no artifact should be produced.

## Conversation-Length Distribution

Do not train primarily on one-turn tasks. A starting mix might be approximately 45–55% short (1–4K tokens), 25–35% medium (4–8K), 10–20% long (8–16K), and a small 16K+ tail. Treat the exact mix as an ablation.

Long conversations should include changed decisions and corresponding state updates.

## Conversation Windows

Convert long sessions into overlapping training windows while keeping committed decisions in external project state. This teaches that persistent decisions should survive even after early turns fall out of context.

## Human-Prose Target Construction

Prefer:

```text
synthetic instruction/state/reasoning
→ real human prose
```

over synthetic instruction → synthetic prose.

One human chapter can generate multiple next-passage examples by using successive semantic beats as targets.

## Synthetic Supervision

Generate labels for:

- scene/chapter objective;
- required/optional/forbidden events;
- explicit/implicit/deferred information;
- character beliefs/goals;
- relevant prior events;
- causal explanation;
- alternate/rejected plans;
- continuity questions;
- relevant files;
- expected state changes.

Label reconstructed plans as synthetic supervision.

## Synthetic Negatives

### Writing corruptions
Add redundant emotional explanation, unnecessary backstory, repeated beats, generic atmosphere, premature exposition, explicit subtext, irrelevant worldbuilding, and unnecessary recap. Train the editor toward the original human passage.

### Agent failures
Generate over-retrieval, under-retrieval, wrong-tool use, malformed calls, premature canon updates, stale-state use, forgotten corrections, and unnecessary rewrites.

## Validation Pipeline

Validate structural correctness, story compatibility, conversation-state consistency, and interaction diversity. Reject examples with future leakage, impossible user instructions, incorrect tool results, or inconsistent state updates.

## Candidate dataset sizes

For a first 3090-scale experiment:

- 1,000–5,000 high-quality next-passage examples;
- 5,000–10,000 tool-use/operation trajectories;
- 2,000–5,000 revision/edit pairs;
- 2,000–5,000 planning examples, including alternatives.

Adjust these candidate sizes after reviewing the pilot dataset.

## Key Data Ablations

Test:

- human vs synthetic prose targets;
- one representation vs randomized representations;
- fixed tool names vs varied schemas;
- one-turn vs multi-turn;
- short-only vs mixed-length;
- full planner reasoning visible vs compressed interface;
- target chapter vs target semantic beat;
- clean project state vs messy/noisy state;
- exact tool-trajectory imitation vs outcome-based supervision.
