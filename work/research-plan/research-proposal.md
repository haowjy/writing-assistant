# Research proposal

Study the [authoring capabilities](../../wiki/project-goals.md) through the
experiments below. Model sizes, objective order, and specialist arrangements are
proposals to test.

## Core System Model

```text
user instruction
    ↓
inspect conversational context
    ↓
decide whether tool use is needed
    ↓
retrieve relevant project state
    ↓
reason / plan locally
    ↓
decide explicit vs implicit vs deferred information
    ↓
write the next semantic beat / passage
    ↓
observe what was actually written
    ↓
update story state / plan if needed
    ↓
repeat
```

The visible output may be a paragraph, scene, chapter section, or complete chapter, but internally the system should be capable of **interleaving reasoning and realization** rather than performing one long uninterrupted decode.

## Persistent Authoring Workspace

The model should not depend on one canonical story-state format. Training and evaluation should include variation across Markdown wikis, YAML state files, JSON records, CSV timelines, plain-text notes, mixed-format repositories, sparse or messy folders, stale or conflicting notes, and raw prior chapters.

Test whether the agent can use equivalent project facts across these representations.

## Specialist Capabilities

Initial experiments should test whether these capabilities should be one unified LoRA or separate adapters/models:

### Authoring Agent / Planner

- understand user intent;
- decide whether to brainstorm, plan, edit, write, or update state;
- inspect project files;
- reason causally;
- generate and compare alternatives;
- construct the next semantic writing beat.

### Retriever

- decide whether lookup is necessary;
- choose useful files/tools;
- avoid retrieving the entire project;
- reconcile duplicated or conflicting state.

### Writer / Realizer

- realize a known local plan as prose;
- follow constraints;
- avoid inventing unnecessary plot information;
- maintain subtext and restraint;
- expose only information that belongs on the page.

### Observer / State Updater

- inspect newly written prose;
- determine what became canonical;
- update character/world/timeline/plot state;
- detect deviations from the previous plan.

### Editor

- remove redundancy;
- remove unnecessary explanation;
- remove irrelevant lore/context;
- preserve meaningful atmosphere and pacing;
- make narrow edits locally.

## Experiment plans

- [Data](data-experiments.md): reconstruct authoring interactions around prose targets
  and vary conversation length, project representation, and tools.
- [Training](training-experiments.md): compare SFT adapters, preference objectives,
  targeted RL, and writing-loop architectures.
- [Evaluation](evaluation-experiments.md): define metrics and diagnose failures.
- [Execution checklist](execution-checklist.md): tasks for carrying out the experiments.

## Primary Research Questions

1. Can a small LoRA turn a capable general assistant into a materially better authoring collaborator without harming general abilities?
2. Does explicit external project state improve long-horizon coherence?
3. Does interleaving reasoning between writing chunks outperform single-pass chapter generation?
4. Should planner and writer behaviors be trained separately?
5. Can the model reason expansively while writing selectively?
6. Does distribution-focused fine-tuning reduce stereotyped LLM prose beyond SFT?
7. Which objective best improves meaningful creative diversity?
8. Can targeted RL improve agent behavior without degrading prose?
9. Does environment randomization improve generalization to unseen project layouts?
10. Does diffusion improve writing when isolated as a realization/refinement component?

## Success Criteria

Report agent performance and writing quality separately, following the
[evaluation principles](../../wiki/evaluation.md).
