# Training experiments

## Current decision: direct GRPO on Qwen3.8-27B

Use the instruction-tuned Qwen3.8-27B checkpoint as the intended writing model and
train it directly with GRPO. GRPO compares several attempts at the same task and
updates the model toward the better-scoring attempts. There is no planned supervised
fine-tuning (SFT) stage. SFT trains by imitating demonstrations; it remains an optional
remedy for a demonstrated gap in Qwen, not a prerequisite for reward training.

Gemma 4 E2B-IT is only a cheap local test of the machinery: generating attempts,
judging them, applying updates, saving checkpoints, resuming, and loading the result.
Its failures do not establish that Qwen needs demonstrations. Its successes do not
establish that training will improve Qwen. Do not simplify the target task collection
merely to make E2B succeed.

Qwen's size is a working choice, not a measured minimum for usable writing. Pin its
exact checkpoint and training configuration before execution. Hosted Qwen inference
can support a baseline and judge calibration; training its weights requires separate
compute. The target is at least 64K training context, ideally 128K, subject to a
Qwen-specific feasibility measurement. E2B memory and timing estimates do not transfer.

The reward adapter exists, but an end-to-end GRPO training loop is not implemented.
The main unresolved readiness question is whether Qwen produces meaningfully different
attempts and whether the judge ranks them reliably. Reconsider targeted SFT only if a
needed behavior remains too rare after checking the tasks, instructions, and rewards.
A few stronger-model examples may help validate the judge without becoming SFT data.

## Decisions still needed before the first substantive run

- Define success: which writing and collaboration improvements matter, and which
  regressions would disqualify a checkpoint.
- Validate the reward, including its handling of instruction violations, missing
  evidence, and weak versus strong prose. The current weights are provisional.
- Correct the task-balance proposal's contradictory requirements before expanding the
  collection. The 300-task target is an estimate, not a proven minimum.
- Define system-prompt variation. Separate paraphrases, amount of guidance, and different
  behavioral requirements; keep the prompt identical within each GRPO comparison group.
  Record the exact prompt and give it to the judge. Variation is not implemented yet.
- Choose how the simulated writer responds to questions and revisions in multi-turn tasks.
- Freeze development comparisons, final-test separation, compute budget, and stopping rules.

The [next-experiment TODO](TODO.md) tracks execution. The [task and reward design](../sft/rl-task-generation.md)
and [simulated author](../sft/simulated-author.md) provide the supporting proposals.
No training run or additional paid work is authorized by this plan update.

## Other model options and later experiments

The remaining sections are possible follow-up experiments, not prerequisites for the
direct-GRPO run. Reopen model selection only with evidence; the earlier candidate list
is retained for comparison:

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

Compare the unchanged Qwen3.8-27B instruction-tuned checkpoint with the same checkpoint
trained directly using GRPO. Hold evaluation tasks, system prompts, tools, generation
settings, and precision fixed between the two. Keep final tests out of checkpoint
selection and report writing quality separately from instruction and tool correctness.

Before that comparison, use E2B for a bounded engineering test and Qwen inference for
judge validation on actual target-model attempts. No SFT-only or SFT-then-RL arm is
required. Add one only if a demonstrated Qwen behavior gap justifies that experiment.
Separate planner/writer adapters remain later options. Validate rewards, source
permissions, save/resume, and measured runtime before scaling.

## Supervision setup

The proposed SFT loss covers assistant messages, including tool calls and visible
plans, while masking user, system, and tool-observation tokens. With TRL,
`assistant_only_loss=True` requires a compatible chat template that supplies
assistant masks. Inspect a rendered batch and verify the masks before training.
Use explicit plans and decisions as supervision; do not import private reasoning
traces automatically. The Python trainer and native masking checks are implemented; GPU training,
checkpoint resume, and trained-adapter inference remain to be verified.
