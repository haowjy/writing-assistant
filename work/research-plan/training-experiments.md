# Training experiments

## Starting point

The proposed starting point is an instruction-tuned 7B–12B model with QLoRA SFT.
Test whether adaptation improves authoring behavior before adding RL. Later
experiments depend on the remaining failures and the reliability of their rewards.
The [baseline comparison](baseline-comparison.md) covers Gemma 4 12B and E4B,
pretrained and instruction-tuned, on an RTX 3090. The training checkpoint and
software stack remain to be selected from those results.

The [branching authorship plan](branching-authorship.md) describes collecting
harness-backed sessions across plot directions and prose styles for this training.

## Untuned Baseline

Run the official checkpoint matrix and selected community derivatives described
in the baseline comparison, with separate agent and prose scores. Archive generations.

## SFT QLoRA

### Variant A — Unified Authoring-Agent LoRA
Train one adapter on planning, tool use, prose, revision, and state updates.

### Variant B — Split Planner + Writer

```text
Planner/Agent LoRA
    ↓
compressed next-beat interface
    ↓
Writer LoRA
```

### Variant C — Planner + Writer + Editor
Add a separate editing pass and measure whether it adds value beyond a better writer.

## Prose Distribution Training

Compare:

```text
SFT only
SFT → DFT-like distribution tuning
SFT → DPO/ORPO
SFT → diversity-aware preference tuning
```

Measure whether distribution matching reduces stereotyped prose while preserving plan adherence.
“DFT-like” remains a placeholder until its loss and reference method are specified.

## Planner Diversity Training

Test preference optimization over diverse/high-quality plans, explicit branching, group-relative rewards, and novelty-aware ranking. Do not reward novelty alone; use a quality threshold.

## Agent GRPO / RL

Use RL first where signals are objective or semi-objective:

- tool-call validity;
- retrieval quality;
- instruction compliance;
- plan adherence;
- state consistency;
- state-update quality;
- forbidden-information leakage;
- modest tool-efficiency penalties.

Task success should dominate so the model does not learn to avoid tools.

## Writer RL, Only If Necessary

Use writer RL only for specific residual failures that can be measured reliably, such as leakage, plan violations, repetition, semantic redundancy, recap, length errors, and continuity errors.

Avoid initially optimizing vague scalar rewards like creativity or literary quality.

## Interleaved Reasoning

Compare:

```text
A. plan once → whole chapter
B. reason every ~1000 words
C. reason every ~400 words
D. reason at semantic beat boundaries
E. model chooses CONTINUE vs STOP_AND_REPLAN
```

Measure coherence, plan adherence, pacing, repetition, stylistic fragmentation, continuity, and completion.

## Reasoning Visibility

Compare writer inputs:

- full planner scratchpad;
- compressed interface with NEXT BEAT / EXPLICIT / IMPLICIT / DEFER / AVOID;
- event only.

Hypothesis: compressed interfaces reduce analytical language leaking into prose.

## Tool-Policy Generalization

Vary tools available, tool names, schemas, file formats, directory structures, noise ratio, and stale/contradictory state. Reward outcomes rather than exact action sequences.

## Diffusion Ablation

Compare AR planner→AR writer, AR planner→diffusion writer, and AR planner→diffusion semantic refinement→AR writer while holding data/prompts/evaluation fixed.

## Distillation / Consolidation

Once specialist behaviors are understood, compare LoRA merging, adapter fusion,
and multitask student distillation against retaining specialists with runtime switching.

## General Capability Preservation

After each major stage, test ordinary instruction following, QA, summarization, coding, conversation, structured output, and non-fiction tool calling.

## Recommended First Experiment

Before RL:

```text
Base IT baseline
vs
Unified SFT LoRA
vs
Planner SFT + Writer SFT
```

If this does not produce measurable gains, debug data and architecture before introducing GRPO.

## Supervision setup

The proposed SFT loss covers assistant messages, including tool calls and visible
plans, while masking user, system, and tool-observation tokens. With TRL,
`assistant_only_loss=True` requires a compatible chat template that supplies
assistant masks. Inspect a rendered batch and verify the masks before training.
Use explicit plans and decisions as supervision; do not import private reasoning
traces automatically. The training runner and masking checks are not implemented.
