# Next experiment: SFT bootstrap and RL preparation

The next priority is to prepare training tasks, validate rewards, and verify the
bounded SFT/RL pipeline, using completed baselines to identify learning objectives. The
[Grok comparison](../grok-pilot/results.md) provides examples of stronger outputs;
it compares different models and harnesses, so it does not isolate model capability.

This list sets the work order. It does not authorize training, new paid generation,
or larger evaluation runs. Agree on the bounded training test before executing it;
keep the existing [evaluation execution boundary](../custom-eval-suite/plan.md).

- [ ] Complete the approved first 100 training tasks using the
  [compiled research and generation workflow](../sft/training-data-research.md).
  Source packets and coverage assignments are prepared; actual generated requests,
  starting drafts/KBs, branch contracts, semantic review and admission remain.
  Use GLM-5.3 through Reka; training-use permission is settled. Credentials and a
  generation spending cap are pending. Generate SFT demonstrations separately.
- [ ] Follow the [3090-first compute plan](../sft/local-compute-and-tracking.md):
  measure bounded training and rollout memory/time locally; defer GPU rental until
  the bottleneck is known.
- [ ] Add optional W&B tracking for scores, written critiques, prose, and versioned
  artifacts while retaining local outputs. Logging is currently disabled.
- [ ] Summarize failure types from the [custom50 assessments](../custom-eval-suite/astra-grading.md):
  prose weaknesses, continuity errors, failed file delivery, and KB navigation.
  Aggregate counts are recorded in the [SFT plan](../sft/plan.md); retain representative
  cases and use the findings to select training examples.
- [ ] Establish permitted demonstrations and an independent RL task collection using
  the [bootstrap research](../sft/rl-bootstrap-research.md). Treat the
  [24-record seed](../sft/dataset-starter.md) as pending; determine SFT size by readiness, mixing
  vague and explicit requests, genres, and prose styles. Include successful tool
  trajectories, local revisions, and handling of proposals versus accepted canon.
  Preserve provenance and group related sources before splitting. Keep evaluation
  cases and related derivatives out of training; the five Grok outputs are comparison
  evidence, not an approved training dataset.
- [ ] Verify the [prepared supervised QLoRA pipeline](../sft/plan.md) for Gemma E2B-IT on the RTX 3090,
  using Transformers, PEFT, TRL, and bitsandbytes. Verify the native conversation/tool
  template and loss masking: train the intended assistant responses and tool calls,
  while excluding system/user messages and tool observations. Decide explicitly
  whether any reasoning data belongs in the training targets.
- [ ] Specify the [on-demand branching task generator](../sft/rl-task-generation.md):
  grounded source packets, permitted divergences, task-specific rewards, private judge
  evidence, coverage tracking, and reproducible per-group initial states.
- [ ] Design [composed multi-turn sessions](../sft/multi-turn-rl.md): grounded adaptive
  author feedback, shared project state, stage and final rewards, and reproducible
  compaction. Measure practical training context before expanding session length.
- [ ] Validate the selected GLM-5.3/Reka judge and informative
  rollout rewards. Propose IT → RL versus IT → short SFT → RL with matched budgets;
  preserve SFT-only and IT controls. A large SFT corpus is not a prerequisite.
- [ ] Define a short feasibility run, then obtain approval and execute it. Measure
  VRAM and processed tokens per second; verify checkpoint save, resume, and inference
  loading. Use measured throughput and the actual token count to estimate a full run.
- [ ] Freeze a mini-evaluation subset and run it at baseline and selected checkpoints
  once its execution scope is approved. Use the same Gemma harness, prompts, and
  generation settings. Record harness/provider/model, checkpoint identity, and step.
  Report completion, Astra rubric scores, and numerical prose profiles separately;
  include a small coding/instruction-following regression check. Choose cadence from
  measured training and evaluation time, rather than an arbitrary epoch interval.

Astra remains the primary subjective evaluator; human grading is optional. Fix its
model, rubric, and grading settings across checkpoints. Occasional repeated grading
can check consistency when separately scoped.

The broader alternatives remain in [training experiments](training-experiments.md),
[data experiments](data-experiments.md), and [evaluation experiments](evaluation-experiments.md).
