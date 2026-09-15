# Next experiment: first supervised fine-tuning run

The next priority is to prepare training data and verify a small QLoRA training
pipeline, using the completed baselines to choose what to teach. The
[Grok comparison](../grok-pilot/results.md) provides examples of stronger outputs;
it compares different models and harnesses, so it does not isolate model capability.

This list sets the work order. It does not authorize training, new paid generation,
or larger evaluation runs. Agree on the bounded training test before executing it;
keep the existing [evaluation execution boundary](../custom-eval-suite/plan.md).

- [ ] Summarize failure types from the [custom50 assessments](../custom-eval-suite/astra-grading.md):
  prose weaknesses, continuity errors, failed file delivery, and KB navigation.
  Aggregate counts are recorded in the [SFT plan](../sft/plan.md); retain representative
  cases and use the findings to select training examples.
- [ ] Expand the [24-record seed](../sft/dataset-starter.md) into a substantive collection
  (working target: 500–1,000 training trajectories plus grouped validation), mixing
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
