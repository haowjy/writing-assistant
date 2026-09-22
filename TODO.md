# Current TODO

Start here for the active work order. Detailed designs live in the
[training plan](work/research-plan/training-experiments.md); the
[supporting checklist](work/research-plan/TODO.md) is not an additional prerequisite list.
Optional ideas live in [work/FUTURE.md](work/FUTURE.md).

The goal is better long-form project memory and effective use of large writing
projects, supported by a maintained wiki, with better prose alongside it.

## Next: DAPO over two complete training passes

- [x] Add identity-bound DAPO through public TRL. Verify unequal-length, masked CPU
  accumulation against dense updates and Adam moments, plus exact checkpoint resume
  across two passes. See [readiness and blockers](work/research-plan/dapo-readiness.md).
- [x] Audit all 48 wave1 training tasks without shortening them. All pass release
  identity checks, but seven cannot meet output requirements under the old token caps;
  the generic smoke reward also grants credit to unchanged drafts.
- [x] Implement approved ordinary TRL continuation through ties without resampling;
  keep default halt for the old probe. CPU proof verifies exact resume, momentum and
  float32 residual behavior. Tied groups are not skipped updates; unavailable rewards halt.
- [ ] Build and validate the faithful full48 runtime/reward contract, long-trajectory
  memory path, collision-free attempt seeds, and intermediate checkpoint recovery.
- [ ] Restart from the pinned base: 48 tasks × two passes × four fresh attempts =
  96 scheduled groups / 384 attempts. The user authorized overnight execution with
  **no elapsed-time cutoff**; timing estimates are advisory. Do not launch until the
  readiness blockers are resolved. No new GPU run has started.

## Completed: short-context GRPO engineering proof

- [x] Prepare the bounded [Gemma E2B GRPO probe](docs/grpo-probe.md): three training
  tasks, six development cases at two seeds, four attempts per training group, three
  optimizer steps, 4096-token context, and a 60-minute aggregate GPU-stage ceiling.
  Validate mechanical rewards on 91 fixture cases and native-token fits on all nine tasks.
  No SFT stage, semantic judge, download, or paid call is required for this engineering run.
- [x] Connect generation, workspace tools, rewards, and GRPO updates. Verify candidate
  token masks, real tiny-CPU adapter updates, save/reload, and exact checkpoint resume.
  See [GRPO usage and checkpoint methodology](docs/grpo.md).
- [x] Run the frozen probe on the **RTX 3090** and preserve the
  [measured result](work/research-plan/gemma-probe-result.md): 12 baseline attempts,
  then CUDA OOM before the first optimizer update. The first training group had
  rewards [1, 0, 1, 1]. Total GPU-stage time was 14.68 minutes; no retry was made.
- [x] Prepare a memory-reduced follow-up: retain four attempts per reward group,
  train one at a time, accumulate four gradients per update. CPU checks verify
  equivalent full-group updates and exact step-1→3 resume. The failed run stays frozen.
- [x] Execute the fresh [microbatch probe](work/research-plan/gemma-microbatch-result.md):
  three real Gemma updates, step-1→3 checkpoint resume, exact resident adapter reload,
  and 12 paired development attempts. Peak Torch allocation 18.249 GiB; total GPU-stage
  time 45.44 minutes. Mechanical mean 0.229→0.313, with mixed per-case changes and no
  semantic/literary improvement established.

Gemma readiness checked: `google/gemma-4-E2B-it` at revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7` has its weights (about 10.25GB), tokenizer,
and configuration cached locally. A tiny random CPU model passed exact step-1→3
resume even after checkpoint 1 was pruned. The original full-group training batch did
not fit, but microbatching completed real Gemma training, resume and adapter evaluation.
This proves the bounded engineering path, not writing or long-context effectiveness.
Semantic/literary judging still needs
calibration before substantive writing optimization. The SFT dataset is unprepared and
is not required for the next mechanical-only DAPO experiment.

## Then: establish the Qwen experiment

- [ ] Compare official Qwen3.8-27B and the DavidAU TURBO Fable/Cold-Fusion derivative
  on a small matched set of project-memory, wiki-maintenance, tool-use, and prose tasks.
  Treat the derivative as a candidate starting model, not a proven winner on our tasks.
- [ ] Validate the existing judge and reward on target-model attempts. Correct conflicting
  task-balance requirements; review the task pool and add broad, coherent instruction
  variation. Do not generate SFT demonstrations unless an observed gap warrants them.
- [ ] Probe the selected Qwen model at short context with Unsloth in an isolated
  environment. Approve the model download and storage budget first; Gemma success does
  not establish Qwen compatibility or memory fit.
- [ ] Define held-out success measurements, stopping rules, and the affordable training
  length mix. Price the desired **256K context** separately and distinguish training
  length from usable inference context. Preserve final tests for final evaluation.

Listing a task here does not launch training, download weights, or authorize paid
calls or rented GPUs. Update this short list as work completes; keep measurements and
implementation details in the linked plans rather than growing this into another report.
