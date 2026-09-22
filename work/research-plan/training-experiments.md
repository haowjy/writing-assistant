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
compute. For now, bound engineering experiments by measured RTX 3090 memory and
runtime. For Qwen, the desired context is at least 256K, subject to cost and verified
model support. Clarify whether this requires training on 256K sequences or reliable
256K use after training on a shorter length mix; these are different requirements.
Do not assume every rollout must fill the context window. E2B memory and timing
estimates do not transfer to Qwen. Obtain a compute quote before committing to long-
context Qwen training; the target is not authorization for rented GPUs.

The reward adapter exists, but an end-to-end GRPO training loop is not implemented.
The main unresolved readiness question is whether Qwen produces meaningfully different
attempts and whether the judge ranks them reliably. Reconsider targeted SFT only if a
needed behavior remains too rare after checking the tasks, instructions, and rewards.
A few stronger-model examples may help validate the judge without becoming SFT data.

## Success and training coverage

The primary goal is better long-form project memory and the ability to work with large
writing projects over time. The agent should preserve the author's decisions, retrieve
relevant facts from substantial project files, carry accepted revisions forward, and
maintain continuity across chapters and conversations. Project facts belong in external
files that the model uses through tools, not facts memorized into its weights.

A well-structured wiki is a means to that goal: navigable pages, useful links, accurate
content, clear distinctions between proposals and accepted story facts, and consistent
updates after revisions. Judge the wiki by whether it supports later retrieval and
writing, not just its appearance, page count, or number of tool calls. Better prose is
an important companion goal; prettier isolated scenes do not establish better project
memory.

Measure improvement over the unchanged target model on both pre-existing external
benchmarks and our own task-and-judging suite. External writing benchmarks assess prose;
our tasks must also exercise retaining, retrieving, updating, and using information
across a long project. Higher training reward alone does not qualify. Use development cases for iteration and checkpoint selection; keep final tests
held out. Report the separate scores and regressions rather than hiding tradeoffs in
one average. Exact minimum gains and acceptable regressions remain to be agreed.

Train on the full task mix: drafting, revision, planning and alternatives, project-file
and knowledge-base maintenance, retrieval-grounded writing, and multi-turn collaboration.
All are in scope; this does not imply equal counts.

Code-to-documentation and code-to-story are optional ideas, not required coverage or
approved collection work. If explored later, small code fixtures could test finding,
understanding, and using project information across formats. Documentation would need
to reflect implemented behavior; stories would distinguish source-defined rules from
permitted invention. No additional datasets, downloads, or code execution are needed
for the initial experiment. Decide whether these tasks add useful evidence before
spending storage or compute on them.

Correct the balance proposal's contradictory requirements before expanding it. The
300-task target remains an estimate, not a proven minimum.

Vary instructions broadly, including system prompts and user requests. Cover different
wording, amounts of guidance, and compatible behavioral requirements. Diversity must
still produce coherent, feasible tasks with clear judging criteria. Keep system-prompt
variation separate from task difficulty, preserve the exact prompt in run records, and
use identical prompts within a GRPO comparison group. The judge must see the instructions
that applied to that attempt. Reserve unfamiliar wording and combinations for evaluation.
This variation is a robustness objective, not a prerequisite of fine-tuning, and is not
yet implemented in the runner.

## What remains to validate or decide

The existing mixed reward design remains the starting point; do not restart reward
research by default. Validate its rankings on actual Qwen attempts, including better
versus weaker prose, instruction violations, and missing evidence. Unit tests verify
score calculation, not literary judgment. The current weights and pointwise-versus-
pairwise judging choice remain provisional; change them if the validation reveals a
problem. Keep training judgment separate from final evaluation.

Still resolve the simulated writer's responses to questions and revisions, exact
success thresholds, development comparison schedule, training budget, stopping rules,
and the scope and cost of the 256K context target.

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
