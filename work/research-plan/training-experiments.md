# Training experiments

## Starting point

The first feasibility checkpoint is Gemma 4 E2B-IT with QLoRA SFT; the final
training model size remains open. See [SFT preparation](../sft/plan.md).
The intended training direction is SFT bootstrapping followed by RL, with direct RL
from the IT checkpoint as a comparison. The [research review](../sft/rl-bootstrap-research.md)
separates bootstrap demonstrations, RL tasks, judge data, and held-out evaluation.
Training scope depends on rollout behavior and the reliability of rewards.
The [baseline comparison](baseline-comparison.md) covers Gemma 4 12B and E4B,
pretrained and instruction-tuned, on an RTX 3090. The broader checkpoint comparison remains separate from the initial training
feasibility test.

The [branching authorship plan](branching-authorship.md) describes collecting
harness-backed sessions across plot directions and prose styles for this training.

### Final-base candidates (deferred)

The E2B feasibility run does not choose the production base. Decide the base only after
the mini-eval subset exists and has been run on the E2B checkpoint. Candidates:

| Base | Params / type | License | Notes |
|---|---|---|---|
| Gemma 4 E2B-IT | ~2B dense | Gemma terms | Feasibility only; template/mask integration already verified. |
| Gemma 4 12B | 12B dense | Gemma terms | Same family; reuses verified integration. |
| Qwen3.8-27B | 27B dense VLM | Apache-2.0 | Current popular general model; ~14 GB at 4-bit, tight on 24 GB; full-context QLoRA likely needs a rented GPU. |
| Qwen3.5-9B | 9B dense VLM | Apache-2.0 | Comfortable local QLoRA size. |
| Qwen3-Coder-Next | 80B / 3B MoE | Apache-2.0 | Agentic-coding strength; ~40 GB at 4-bit, inference-only locally. |

Selection criteria: harness/tool reliability, long-context behavior, measured rollout
quality, QLoRA fit, and license. The Qwen Coder cards do not claim creative writing, and
the 3.5/3.6/3.8 line are VLMs whose vision bulk is unused here. Switching family requires
re-verifying tokenizer, chat template, loss masks, and rendering before training.

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

## Writer RL

Investigate writing-quality preference rewards alongside constraint and continuity
checks. Validate a training judge before optimizing its scores. Keep literary
assessment distinct from mechanical compliance and distribution diagnostics;
measure whether gains transfer to held-out projects and an independent evaluator.

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

After each major stage, test ordinary instruction following, QA, summarization, coding, conversation, structured output, and non-fiction tool calling. Preserving a capability is not the same as measuring it: a small rehearsed slice of basic coding tasks in the training mix is the anti-forgetting mechanism, and the checkpoint runs are its measurement. See the [training-distribution axes](../sft/training-distribution-axes.md) and the coding rehearsal entry in [deferred work](../FUTURE.md).

## Recommended First Experiment

Compare the unchanged IT checkpoint, SFT-only, direct RL from IT, and short SFT → RL.
Use matched RL task pools and rollout budgets for the two RL conditions. The SFT
warm-up should address demonstrated sampling or tool-use gaps; do not impose a fixed
500–1,000-example prerequisite. Separate planner/writer adapters remain later ablations.
Validate rewards, source permissions, and a bounded runtime test before scaling.

## Supervision setup

The proposed SFT loss covers assistant messages, including tool calls and visible
plans, while masking user, system, and tool-observation tokens. With TRL,
`assistant_only_loss=True` requires a compatible chat template that supplies
assistant masks. Inspect a rendered batch and verify the masks before training.
Use explicit plans and decisions as supervision; do not import private reasoning
traces automatically. The Python trainer and native masking checks are implemented; GPU training,
checkpoint resume, and trained-adapter inference remain to be verified.
