# Evaluation experiments

The [custom suite design](../custom-eval-suite/index.md) and its
[scorecard](../custom-eval-suite/metrics.md) own the current implementation scope.
This document retains optional experiments and diagnostics for later work.

Use the [evaluation principles](../../wiki/evaluation.md) to compare agent behavior
and writing quality. The metrics below are candidates; specify their formulas,
labels, judge rubrics, and thresholds before running an experiment.

The [dataset and metric research](dataset-and-metric-research.md) compares existing
evaluation protocols and identifies unresolved details in the proposed metrics.

## Grading method

Use Astra (`gpt-6-astra`) through Codex as the primary judge for prose and
semantic requirements. This is the selected judge, not a validated scoring
implementation. Mechanical constraints, tool schemas, file diffs, latency, and
token counts are checked by code. Embedding metrics use a separate fixed model.

The proposed scorecard combines published methods with project-specific checks:

| Measurement | Basis |
|---|---|
| Verifiable constraints and all-constraints success | [IFEval](https://arxiv.org/abs/2311.07911) |
| Semantic/style constraints and partial satisfaction | [FollowBench](https://aclanthology.org/2024.acl-long.257/) |
| State outcomes and evidence of tool execution | [BFCL multi-turn](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) |
| Pairwise prose preferences and dimension rubrics | [Agents' Room](https://arxiv.org/html/2410.02603), [WritingBench](https://arxiv.org/abs/2503.05244) |
| Checking judge agreement with human preferences | [LitBench](https://arxiv.org/abs/2507.00769), [HANNA](https://arxiv.org/abs/2208.11646) |
| Branch-specific leakage, canon updates, edit boundaries | Our task labels and acceptance rules; not established literary benchmarks |

The current scorecard has 13 quality measures, a linked
[prose metric catalog](../custom-eval-suite/prose-metrics.md), and two resource
measures, applied selectively across five task families. It replaces the earlier 12-primary-metric
proposal. N-gram L2 and embedding MMD are recorded for generated prose in the development
release; other distributional experiments remain optional. The combined suite and
its rubrics still require validation. Preserve official scoring when running an
existing benchmark; label adaptations.

For each judged item, provide only its brief, relevant story state, artifact,
and fixed rubric. Hide model identity and training condition. Treat artifact text
as evidence, not instructions to the grader. Return structured scores, cited
passages, brief explanations, and an uncertainty/needs-review flag. Use fresh
grading contexts and compare pairwise outputs in both orders. Record the requested
and reported judge model, reasoning setting, rubric version, CLI version, and
input/output hashes. Retain raw judgments for rescoring.

Calibrate against held-out human annotations and reviewed branch examples before
using aggregate scores to choose training methods. Measure preference agreement,
rubric agreement, order sensitivity, and repeated-judgment consistency. A model
used to generate synthetic targets should have those targets independently
reviewed; its own preference is not human validation.

The planned local integration uses `codex exec` with a JSON output schema and
saved ChatGPT authentication. The installed CLI exposes `--model`,
`--output-schema`, and `--output-last-message`. Grading has not been executed yet;
verify Astra access and measure usable throughput in a small pilot. Codex
subscription access and usage-based API access are distinct authentication routes.
[Non-interactive Codex](https://learn.chatgpt.com/docs/non-interactive-mode),
[authentication](https://learn.chatgpt.com/docs/auth).

## Authoring Agent / Instruction Metrics

### Operation Selection

Measure whether the model correctly identifies brainstorming, planning, critique, writing, revision, state update, tool lookup, or normal discussion. Use accuracy/F1.

### Instruction Adherence

Measure explicit constraints such as POV, length, tone, scene goal, "do not reveal X," "keep Y," local-edit scope, and whether canon should be updated. Use constraint-level precision/recall/F1 plus hard violation rate.

### Plan Adherence

Using required / optional / forbidden / deferred labels, measure required-beat coverage, forbidden-beat violation, deferred-information leakage, and unauthorized plot invention.

### Tool Use

Measure tool validity, retrieval precision/recall, relevant-fact recall, unnecessary calls, duplicate reads, and robustness across file formats/tool schemas.

### State Fidelity

Measure character knowledge, relationships, locations, inventory, chronology, unresolved threads, and canonical-state updates.

### Edit Locality

When asked to change a small region, measure requested-change success and semantic preservation outside the edited span.

### Multi-turn Retention

Construct conversations where early decisions are changed later and stale project files remain. Test whether the latest committed decision wins.

## Writing Artifact Metrics

### N-Gram Token Distribution L2 Distance

Compare generated prose against a human reference corpus across multiple n-gram orders, scene types, and genres.

### Maximum Mean Discrepancy (MMD)

Compute global and conditional MMD over prose embeddings. Keep it primarily evaluation-only at first.

### Model judging (JMQ placeholder)

Define the judge and scoring rubric for writing artifacts. Judge prose quality, coherence, characterization, pacing, dialogue, imagery, subtext, emotional credibility, and originality.

### Diversity across samples

For the same state/instruction, sample many outputs and measure how collective semantic diversity grows with sample count. Define the diversity function `D(n)` before comparing base, SFT, preference-trained, and RL models.

### Redundancy and Over-Explanation

Measure semantic redundancy, repeated emotional beats, repeated exposition, duplicate narrative functions, and explicit statements of inferable information.

### Surprise With Retrospective Justification

Rate predictability before the development and causal justification afterward. Desired behavior is low predictability + high retrospective justification.

## Diagnostic Ablations

### If Prose Quality Does Not Improve After SFT

Ablate:

- target-corpus quality;
- scaffold quality;
- synthetic-prose ratio;
- LoRA rank/targets;
- LR/epochs;
- base vs instruction checkpoint.

### If Instructions Improve but Writing Becomes Generic

Try:

- DFT-like distribution tuning;
- diversity-aware preference optimization;
- genre-balanced sampling;
- fewer SFT epochs;
- separate Writer LoRA;
- prose continued-pretraining before authoring SFT.

### If Writing Improves but Instruction Following Gets Worse

Try:

- general-assistant replay;
- separate Writer and Agent adapters;
- lower rank;
- fewer steps;
- mode/operation conditioning.

### If the Model Overuses Tools

Add no-tool examples, direct-response preference pairs, mild tool-cost rewards, and varied tool availability.

### If the Model Underuses Tools

Remove duplicated context, create tasks impossible without retrieval, reward relevant-fact recall, increase retrieval examples, and reduce tool penalties.

### If the Model Memorizes One Workflow

Randomize file layouts, tool availability, schemas/names, and valid trajectories. Include environments without search and environments requiring no tools.

### If Multi-turn Conversations Fail

Increase medium/long trajectories, state-changing conversations, conversation windows, stale-state conflict examples, and external-state commits.

### If Long Conversations Harm Writing Quality

Shorten visible history, summarize old conversation, retrieve compressed project state, separate Planner/Writer contexts, and feed Writer only a compact next-beat interface.

### If the Writer Dumps Planning Into Prose

Hide planner scratchpad, use compressed EXPLICIT/IMPLICIT/DEFER interfaces, train leakage negatives, use stronger editor data, and optionally target leakage with RL.

### If Planner/Writer Separation Produces Disjointed Prose

Give the writer more recent prose, enrich the interface, replan less often, write larger semantic beats, test unified LoRA, or let the writer request more context.

### If Reasoning Between Paragraphs Makes Prose Choppy

Compare reasoning every paragraph, every ~400 words, every ~1000 words, semantic-beat boundaries, and learned STOP_AND_REPLAN.

### If Planner Diversity Rises but Quality Falls

Use a quality threshold before diversity reward, lower diversity weight, branch first/rank later, or optimize event-level rather than lexical diversity.

### If DFT Improves Distribution Metrics but Hurts Writing

Condition distribution matching by scene type/genre/operation, reduce distribution-loss weight, use stronger embeddings, or keep SFT/reward objectives jointly.

### If GRPO Reward Rises but JMQ/Human Quality Falls

Investigate reward hacking and judge disagreement. Inspect high-reward outputs, remove suspect reward dimensions, strengthen held-out evaluation, and compare reward ensembles with structured checks.

### If Retrieval Improves but Writing Does Not

Run **oracle-context** tests. Improvement with gold context points to retrieval as a bottleneck. Failures that persist need checks of planning, realization, and task constraints. Also test selected-facts-only inputs and an explicit information-selection stage.

### If Continuity Still Fails

Separate retrieval failure, reasoning failure, realization failure, and state-update failure. Ablate oracle state, richer state extraction, more state-update training, explicit character-knowledge files, timeline validation, smaller write chunks, and more frequent reasoning.

### If New State Formats Fail

Increase format randomization, auxiliary conversion tasks, messy free-form notes, unseen tool schemas, and adversarial directory layouts.

### If Unified LoRA Underperforms Specialists

Compare unified LoRA, Planner+Writer, Planner+Writer+Editor, adapter switching, adapter fusion, and student distillation.

### If Specialist Weight Merging Fails

Use runtime switching, routing by operation, adapter fusion, or distillation instead of naïve merging.

### If Diffusion Does Not Improve Final Prose

Check whether it still improves semantic diversity, constraint reconciliation, or latent refinement. Test `AR planner → diffusion semantic refinement → AR writer` before discarding it.

## Experimental Discipline

### Fixed Core Evaluation Set

Never train on it. Include unseen synthetic worlds, private/unpublished stories, unseen tool layouts, long multi-turn sessions, explicit/implicit/deferred tests, plan-following tasks, revision tasks, and varied genres.

### Archive Every Checkpoint

Record dataset version, seed, base checkpoint, LoRA config, objective, LR, steps, context-length mix, generation settings, and benchmark outputs.

### Prefer Incremental Experiments

Vary the factor under study while holding the other conditions fixed.

### Recommended Initial Matrix

| ID | Model | Training |
|---|---|---|
| A0 | Base IT | none |
| A1 | Base IT | unified SFT LoRA |
| A2 | Base IT | Planner SFT + base writer |
| A3 | Base IT | Planner SFT + Writer SFT |
| A4 | Base IT | Writer SFT only |
| A5 | A3 | + Editor pass |

Then:

| ID | Extension |
|---|---|
| B1 | Writer SFT → DFT-like tuning |
| B2 | Planner SFT → diversity preference training |
| B3 | Agent SFT → GRPO |
| B4 | Planner SFT → GRPO diversity |
| B5 | Writer targeted RL |
| C1 | diffusion realization comparison |
