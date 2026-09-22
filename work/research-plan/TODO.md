# Next experiment: direct GRPO preparation

The target is direct GRPO on instruction-tuned Qwen3.8-27B, with no planned SFT stage.
Use E2B only to test the training machinery locally; use Qwen attempts to establish
reward reliability and target-model readiness. See the [current training decision](training-experiments.md).
The
[Grok comparison](../grok-pilot/results.md) provides examples of stronger outputs;
it compares different models and harnesses, so it does not isolate model capability.

The short, authoritative work order is **[the root TODO.md](../../TODO.md)**:
Gemma short-context GRPO first, then longer Gemma sessions, then the Qwen experiment.
This supporting list does not authorize training, new paid generation, or larger
evaluation runs. Agree on the bounded training test before executing it; keep the
existing [evaluation execution boundary](../custom-eval-suite/plan.md).

The checklist below is not a requirement to finish every item before the engineering
test. Demonstrations and SFT are conditional, not on the critical path.

## Supporting work

- [ ] Complete the approved first 100 training tasks using the
  [compiled research and generation workflow](../sft/training-data-research.md).
  Use DeepSeek V4.1 Flash (direct API, model `deepseek-flash`); its terms permit
  training-use. The user approved $10 total for generation and task review;
  `DEEPSEEK_API_KEY` is set. The Python generation pipeline and DeepSeek
  output-grader adapter are implemented and live-verified. The batch has run once:
  54 of 100 tasks admitted (5 needs_revision, 41 invalid), $3.79 of the cap spent.
  The invalid tasks failed mechanical contract checks rather than review, so tightening
  the generator instructions and re-running is the open step. Do not generate SFT
  demonstrations unless a target-model behavior gap justifies them.
- [ ] Specify the [training-distribution axes](../sft/training-distribution-axes.md) and
  check reward behavior across them. Separate task variables (content, starting point,
  instruction specificity, thinking level) from nuisance variables (phrasing, tool
  envelope, partner identity); confirm the reward is conditioned on the first set and
  invariant to the second. Treat underspecified requests as the primary case: clarify
  when a decision is consequential and undetermined, proceed when it is not.
- [ ] Specify the [simulated author](../sft/simulated-author.md) that supplies author
  turns without a human: a deterministic environment gate, a scripted/lookup controller
  for common turns, and a cached paid user model for unanticipated questions. Keep the
  partner grounded in the author's hidden decisions and never the rubric, and freeze its
  version per experiment.
- [ ] Probe a stronger task author (Claude Sonnet) against the DeepSeek baseline on a
  bounded batch, keeping DeepSeek as reviewer, and compare admitted yield and realized
  variation before changing the author role.
- [ ] Review realized genre blends, tropes, situations and KB constraints against the
  [variation catalog](../../data/training/variation-catalog-v1.json). Adapt incompatible
  combinations, record actual labels, and keep source fidelity and prose quality
  separate from label coverage. Acquire and split additional book candidates before use.
- [ ] Follow the [3090-first compute plan](../sft/local-compute-and-tracking.md):
  the context budget is measured, not assumed. 128K inference needs about 1.9 GB of KV
  cache, and a 64K training step projects to 12.7 minutes, so the binding constraints
  are attention compute and rollout throughput rather than memory. FlashAttention is
  unavailable for this model (`global_head_dim` 512 exceeds the FA limit), so
  memory-efficient SDPA is the only O(n) attention path. Re-run
  `scripts/probe_context_budget.py` before quoting any length.
- [ ] Add optional W&B tracking for scores, written critiques, prose, and versioned
  artifacts while retaining local outputs. Logging is currently disabled.
- [ ] Summarize failure types from the [custom50 assessments](../custom-eval-suite/astra-grading.md):
  prose weaknesses, continuity errors, failed file delivery, and KB navigation.
  Aggregate counts are recorded in the [SFT plan](../sft/plan.md); retain representative
  cases and use the findings to select training examples.
- [ ] Establish an independent RL task collection using the
  [bootstrap research](../sft/rl-bootstrap-research.md), mixing vague and explicit
  requests, genres, prose styles, local revisions, and proposals versus accepted canon.
  The [24-record SFT seed](../sft/dataset-starter.md) remains pending and is not a blocker
  for GRPO. Curate demonstrations only if a Qwen behavior gap warrants them.
  Preserve provenance and group related sources before splitting. Keep evaluation
  cases and related derivatives out of training; the five Grok outputs are comparison
  evidence, not an approved training dataset.
- [ ] Measure the [distribution gap](../sft/distribution-finetuning.md) between current outputs
  and human writing before rebuilding the data. Reuse `prose.score_prose` (D1 n-gram L2,
  D2 MMD, D4 self-BLEU) against the frozen references; this needs local GPU time and no
  API budget. A large gap makes human-target data construction the highest-leverage
  change available, ahead of reward work; a small gap retires the idea. Record the
  configuration with the number.
- [ ] Implement the direct-GRPO training connection and verify it with E2B on the RTX 3090.
  Validate the native conversation/tool template and which generated tokens receive
  training loss; exclude system/user messages and tool observations. Decide explicitly
  how reasoning tokens are handled. The [prepared SFT pipeline](../sft/plan.md) is an
  optional fallback, not this engineering test.
- [ ] Specify the [on-demand branching task generator](../sft/rl-task-generation.md):
  grounded source packets, permitted divergences, task-specific rewards, private judge
  evidence, coverage tracking, and reproducible per-group initial states.
- [ ] Design [composed multi-turn sessions](../sft/multi-turn-rl.md): grounded adaptive
  author feedback, shared project state, stage and final rewards, and reproducible
  compaction. Measure practical training context before expanding session length.
- [ ] Validate the selected DeepSeek V4.1 Flash judge on Qwen attempts, including
  meaningful quality differences and instruction violations. Compare unchanged Qwen
  with direct-GRPO Qwen; no SFT comparison is scheduled. E2B failures alone do not justify
  SFT for Qwen. The RL optimizer is fixed as a critic-free group method (not PPO) by the
  [algorithm decision](../sft/rl-algorithm-decision.md); the pointwise-vs-pairwise reward
  question remains open there.
- [ ] Validate retrieval coverage for the semantic judge on KB and long-form cases, since
  the judge sees retrieved evidence, not the whole wiki. Build cases with known continuity
  and knowledge errors plus their supporting source passages; measure whether the current
  retrieval surfaces those passages, and report each real error as caught, unscorable, or
  silently missed. Record the pass bar (e.g. no silent misses on the validation set) and
  produce a retrieval-coverage number used when interpreting judge scores.
- [ ] Decide the reward format by comparing pointwise 1–5 scalars against pairwise/rank
  judgments on matched good/weak and deliberately constraint-violating candidates. Check
  score spread across a rollout group (zero-variance rate), length/explanation inflation,
  and rank disagreement with the pointwise mode. Report which format avoids rewarding
  violations and gives usable within-group variance, and freeze the chosen format and
  prompt for RL.
- [ ] Validate the [information-value profile](../sft/information-value.md) with
  concise-but-incomplete, accurate-but-irrelevant, redundant and useful-detailed
  outputs. Keep statistical quality rewards inactive until checked; scope a small
  source-grounded nonfiction slice within the existing task families.
- [ ] Inspect and inventory [writing-reward reference data](../sft/writing-reward-data.md),
  starting with LitBench-Train. Preserve official test splits and the existing HANNA
  evaluation role; audit cross-dataset prompt/story overlap before judge calibration.
- [ ] Define a short feasibility run, then obtain approval and execute it. Measure
  VRAM and processed tokens per second; verify checkpoint save, resume, and inference
  loading. Use measured throughput and the actual token count to estimate a full run.
- [ ] Freeze a mini-evaluation subset and run it at baseline and selected checkpoints
  once its execution scope is approved. Compare each model with its own unchanged
  checkpoint using the same harness, prompts, precision, and generation settings.
  Keep E2B engineering results separate from Qwen quality results. Record
  harness/provider/model, checkpoint identity, and step.
  Report completion, Astra rubric scores, and numerical prose profiles separately;
  include a small coding/instruction-following regression check. Choose cadence from
  measured training and evaluation time, rather than an arbitrary epoch interval.
- [ ] Validate the [reward adapter](../../src/writing_agent/reward.py) on controlled
  failures and matched good/weak examples before its first policy update. It is
  implemented and unit-tested; version 0 weights are a hypothesis, and nothing has been
  optimized against them. Check the pointwise-vs-pairwise question alongside it.
- [ ] Keep the [long-form final suite](../custom-eval-suite/longform-suite.md) out of
  checkpoint selection. It is built and frozen, and running it to see how a change looks
  converts it into a development set. Report it beside the
  [external EQ-Bench anchor](../external-benchmarks/longform-eqbench.md); agreement
  between the two is the result, and disagreement is the more informative one.
- [ ] Build the distributional component of the final suite. The long-form cases run four
  attempts each and cannot report MMD, whose floor is twenty; reaching that inside them
  would cost roughly a million words. It needs short held-out prompts sampled many times,
  with human continuations as the paired reference, which would also make D7 and D8
  computable. Use `prose.sampling_plan` to size it. See
  [the sampling section](../custom-eval-suite/longform-suite.md#sampling-and-the-measurement-this-suite-cannot-make).

Astra remains the primary subjective evaluator; human grading is optional. Fix its
model, rubric, and grading settings across checkpoints. Occasional repeated grading
can check consistency when separately scoped.

The broader alternatives remain in [training experiments](training-experiments.md),
[data experiments](data-experiments.md), and [evaluation experiments](evaluation-experiments.md).
