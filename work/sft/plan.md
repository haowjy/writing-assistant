# Optional SFT preparation

This is a fallback procedure, not the next scheduled training stage. The
[current training plan](../research-plan/training-experiments.md) starts directly
with GRPO on Qwen3.8-27B. E2B is an engineering test model; its failures do not establish
that Qwen needs SFT. Use this procedure only after a demonstrated target-model gap
justifies learning from demonstrations. The readiness notes and proposed run below
describe the earlier SFT preparation, not current GRPO readiness.

If that experiment is approved, prepare one supervised QLoRA adapter on `google/gemma-4-E2B-it`, revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`. E2B is the feasibility checkpoint;
this does not select the final model size. The [next-experiment TODO](../research-plan/TODO.md)
tracks the broader sequence.

## Current readiness

- Implemented Python preparation and explicit training entry points.
- Installed and locked TRL 1.13.0, datasets 5.0.1, PEFT 0.20.0, and bitsandbytes 0.50.2.
- Verified the cached native tokenizer on five synthetic format probes, without weights.
- Verified that TRL's dataloader preserves the labels and masks padding, using a tiny
  randomly initialized CPU model. No forward/backward training or optimizer step ran.
- Created a [24-record labeled seed dataset](dataset-starter.md), with 19 train and
  five validation records. All remain pending literary acceptance. This is a pipeline
  seed, not the substantive training collection.
- No Gemma training weights were loaded, adapter trained, or GPU throughput measured.

The [mask review](preparation-audit/review.md) shows complete native conversations
and the selected loss tokens. The accompanying masks.json preserves the token IDs
and labels. These probes remain pending fixtures and are not exported for training.

## Data priorities from the baseline

Across the saved custom50 Astra scorecards, failed checks included 10 required
continuity checks, five required wiki-link checks, three required accepted-update
checks, and three required consequential-state checks. Five prose artifacts were
missing. Optional word-budget and viewpoint/style checks each failed nine times.
Counts overlap and are not a partition of the 21 failed tasks.

Build independent training trajectories covering:

- Concrete prose with consistent staging, rather than repeated emotional explanation.
- Reading source files, delivering to the requested file, and preserving unrelated text.
- Navigable Markdown KBs with qualified beliefs, consequential facts, and useful links.
- Draft-only suggestions followed by accepted revisions, with appropriate canon updates.
- Distinct useful plans and prose grounded in retrieved KB information.

Vary genre, style, instruction specificity, names, and project structure across all
five families. Preserve source lineage and artifact provenance. Keep evaluation works
and derivatives out of training. Do not copy the custom50 or Grok pilot trajectories
into the production collection. Existing downloaded datasets need their own source
selection, terms review, and transformation decisions before use.

## Proposed feasibility run, not executed

The editable settings propose 20 optimizer steps, sequence cap 2048, batch size 1,
gradient accumulation 8, LoRA rank 16/alpha 32, all-linear targets, learning rate
0.0002, NF4 double quantization, bfloat16 compute, and gradient checkpointing. Packing
is disabled. Overlong trajectories fail preparation instead of being truncated.

These are starting settings, not measured VRAM or throughput claims. After approving
the data and bounded execution, inspect loss, trainable parameter count, peak VRAM,
checkpoint saving/resume, and adapter inference loading. Measure tokens per second
before estimating a larger run. Use a new output directory for a failed initialization;
resume accepts an existing trainer checkpoint from the same prepared experiment.

The loss covers assistant text, native tool calls, and assistant turn endings. User,
system, tool schemas, and tool observations are context only. Explicit reasoning fields
and literal special-token input are rejected in this first preparation path. Native
thinking mode remains enabled in rendering and evaluation; supervising reasoning is
a separate unresolved data decision. This path does not claim to teach reasoning.

## When to reconsider SFT

The [research review](rl-bootstrap-research.md) discusses SFT warm-up as an alternative,
not a required stage. Reconsider it if a needed Qwen behavior remains too rare after
checking task feasibility, instructions, and judging. SFT can make a rare behavior
reliable; it is not limited to teaching behavior absent from every sample. No SFT-only
or SFT-then-RL comparison is currently scheduled. Resolve training-source and
judge-output permissions before accepting any demonstrations.

The RL optimizer remains a critic-free group method, not PPO; see the
[algorithm decision](rl-algorithm-decision.md). Reward reliability and useful Qwen
attempts determine readiness, not completion of the pending SFT seed.

## Conditional SFT checklist (not the active work order)

- [ ] Design the training distribution and the simulated author: [axis set and reward
  rules](training-distribution-axes.md), [multi-turn loop](simulated-author.md).
- [ ] Measure how far current outputs sit from the human writing distribution before
  changing the data construction. Reuse `prose.score_prose` (D1, D2, D4) against the
  frozen references; local GPU time, no API budget. The result decides whether
  [distribution fine-tuning](distribution-finetuning.md) outranks the reward work.
- [ ] Establish permitted bootstrap demonstrations and an independent RL task pool.
  Test warm-up size from behavior; expand SFT only for observed gaps.
- [ ] Validate candidate training rewards/judges separately from held-out evaluation.
- [ ] Inspect their native masks and lengths, then freeze the preparation manifest.
- [ ] Approve and run the bounded GPU feasibility test; verify save/resume and adapter inference.
- [ ] Freeze checkpoint mini-evaluation selection and cadence after timing the test.

Astra is the primary subjective evaluator. Human calibration is optional. Evaluations
remain explicit: the trainer never launches candidate benchmarks or paid grading.

See [SFT usage and contracts](../../docs/sft.md). Technical references:
[TRL SFT](https://huggingface.co/docs/trl/sft_trainer) and
[PEFT quantized training](https://huggingface.co/docs/peft/developer_guides/quantization).
