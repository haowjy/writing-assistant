# Execution checklist

Proposed tasks for the research program. Checkboxes record planned work, not an
audit of implemented features; see [implementation milestones](implementation-milestones.md).

## Repository and Harness

- [ ] Create research repository.
- [ ] Save resolved experiment configurations.
- [ ] Add model-loading abstraction.
- [ ] Add LoRA loading/switching.
- [ ] Implement chat-template rendering.
- [ ] Implement tool-call parser.
- [ ] Implement sandboxed story-project filesystem.
- [ ] Implement `list_dir`, `read_file`, `search`, `write_file`, `patch_file`.
- [ ] Log all model/tool trajectories.
- [ ] Save token counts and latency.

## Baseline Benchmark

- [ ] Build 50–100 initial tasks.
- [ ] Cover direct continuation, plan following, revision, brainstorming, state update, retrieval, no-tool-needed, conflicting-state, and implicit/deferred information.
- [ ] Run untouched base model.
- [ ] Archive outputs.
- [ ] Implement initial agent metrics.
- [ ] Implement initial prose metrics.

## Human Prose Corpus

- [ ] Select 1,000–5,000 high-quality target passages.
- [ ] Track provenance/license.
- [ ] Segment by semantic beat/passage.
- [ ] Record genre/scene metadata.
- [ ] Deduplicate.
- [ ] Hold out complete works/authors for evaluation.

## Story-State Reconstruction

For each source work:

- [ ] reconstruct character state;
- [ ] reconstruct timeline;
- [ ] reconstruct plot state;
- [ ] reconstruct relevant world facts;
- [ ] identify unresolved threads;
- [ ] construct local plans;
- [ ] annotate required / optional / forbidden / explicit / implicit / deferred.

## Environment Randomization

- [ ] Markdown representation.
- [ ] YAML representation.
- [ ] JSON representation.
- [ ] Mixed representation.
- [ ] Messy folder structure.
- [ ] Irrelevant files.
- [ ] Stale files.
- [ ] Conflicting files.
- [ ] Varied filenames.
- [ ] Varied tool schemas.
- [ ] Varied tool availability.
- [ ] No-tool cases.

## Conversation Generation

Generate:

- [ ] direct writing conversations;
- [ ] brainstorming → selection → writing;
- [ ] revision conversations;
- [ ] plan-edit conversations;
- [ ] state-update conversations;
- [ ] critique-only conversations;
- [ ] continuity-repair conversations;
- [ ] medium-length sessions;
- [ ] long state-evolving sessions.

Prefer human prose targets.

## Negative Data

### Agent negatives

- [ ] unnecessary tool use;
- [ ] missing required retrieval;
- [ ] wrong file/tool;
- [ ] malformed call;
- [ ] premature canon update;
- [ ] stale-state use;
- [ ] forgotten user correction.

### Writing negatives

- [ ] redundant explanation;
- [ ] repeated emotion;
- [ ] unnecessary backstory;
- [ ] generic description;
- [ ] premature reveal;
- [ ] irrelevant lore;
- [ ] unnecessary recap.

## Dataset Validation

- [ ] syntax checks;
- [ ] file/tool consistency;
- [ ] target compatibility;
- [ ] no future leakage;
- [ ] conversation-state consistency;
- [ ] trajectory diversity;
- [ ] length-distribution checks;
- [ ] provenance audit.

## First SFT Experiment

### Unified LoRA

- [ ] QLoRA;
- [ ] small rank first;
- [ ] mixed conversational authoring data;
- [ ] general-assistant replay.

### Planner and writer adapters

- [ ] Planner/Agent LoRA;
- [ ] Writer LoRA.

Compare against the unified adapter.

## Decision Gate

Before advanced objectives, answer:

- Did instruction adherence improve?
- Did tool use improve?
- Did plan adherence improve?
- Did prose improve?
- Did general assistant ability degrade?
- Does split specialization outperform unified tuning?

If **no meaningful improvement**, do not start RL. Debug dataset, target quality, trajectory realism, adapter capacity, and interface design.

## Distribution / Preference Training

- [ ] DFT-style writer experiment.
- [ ] DPO/ORPO baseline.
- [ ] diversity-aware planner preference training.
- [ ] compare distribution metrics.
- [ ] compare held-out JMQ/human preference.

## Agent GRPO

Start with tool validity, retrieval, instruction compliance, plan adherence, state consistency, and forbidden-information leakage. Do not initially reward subjective prose quality directly.

## Interleaved Writing Loop

Implement:

```text
retrieve → reason → write beat → observe → repeat
```

Test whole chapter, ~1000-word interval, ~400-word interval, semantic beat, and learned stopping/replanning.

## Long Conversation Evaluation

- [ ] 10-turn sessions.
- [ ] 20-turn sessions.
- [ ] 40-turn sessions.
- [ ] old decision changed midway.
- [ ] stale plan remains on disk.
- [ ] external state updated.
- [ ] earliest turns dropped from context.

## Diffusion Study

Only after AR baseline is strong. Hold prompts, state, plans, and evaluation fixed while comparing AR realization, diffusion realization, and diffusion semantic-refinement hybrid.

## Consolidation

If specialists outperform unified model:

- [ ] keep runtime adapter switching as baseline;
- [ ] test adapter fusion;
- [ ] test compatible merges;
- [ ] generate teacher trajectories;
- [ ] distill into unified student;
- [ ] re-evaluate for interference.

Measure quality, latency, and interference against runtime switching before choosing consolidation.
