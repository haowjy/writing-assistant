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
- [x] Prepare a separately bound [full48 mechanical reward](docs/grpo-full48.md),
  preserving originals and closing unchanged/missing-delivery shortcuts. Offline
  fixtures cover all 48; semantic quality and intermediate faithfulness remain unjudged.
- [x] Install approved pinned TRL `6c5f135` and Liger `0.8.3` in an isolated CPU
  qualification environment. Imports, declared dependencies and archive hashes pass;
  the original TRL 1.13 environment remains unchanged.
- [x] Accept the maintained upstream numerical variant explicitly; native BF16 parity
  is not claimed. Add source-pinned opt-in compatibility and pass live BF16 CPU
  Gemma4 train-entry, masked group4 accumulation, chunk boundaries, changed-observation
  conditioning and exact pause/resume checks. See [runtime usage](docs/grpo.md).
- [x] Build the separate [full48 runtime](docs/grpo-full48.md) with intact-task budgets,
  collision-free seeds, finite-work supervision and fail-closed checkpoint recovery.
  Bind production preparation/execution to the qualified streaming implementation;
  CPU proof verifies pass-one pause/resume and exact optimizer/token state.
- [x] Add the inspect-first [production GPU fit gate](docs/grpo-gpu-fit.md), with
  complete-process ownership admission shared by fit and full48 train/resume.
- [x] Execute the one-attempt native Gemma 32,768-token controlled fit after approved
  desktop admission. Native 32,767+1 generation passed; training OOMed during the
  first backward pass before an optimizer update or checkpoint. Preserve the terminal
  [fit result](work/research-plan/gemma-full48-fit-result.md); do not retry or alter it.
- [x] Select a fresh headless qualification contract without changing training math:
  require no listed GPU consumers, at least 24,000 MiB free, and PyTorch expandable
  allocator segments set before import. Bind the same contract to full48 train/resume.
- [x] Execute v3 once from mosh with no GPU consumers and 24,085 MiB free. Generation
  passed; training still OOMed in the FP32 MLP LoRA path before an update. Preserve the
  [v3 result](work/research-plan/gemma-full48-fit-v3-result.md); do not retry it.
- [x] Select a 24,576-token v4 contract on the RTX 3090. Keep the 8,192 per-decision /
  16,384 sampled-action limits and exact FP32 all-linear recipe; reduce only complete
  trajectory headroom. Original tasks and output requirements remain unchanged.
- [x] Execute v4 once headless. Ownership and native 24,575+1 generation passed, but
  a faulty evidence assertion stopped after the first training forward: public DAPO's
  group-active-token denominator was 32,768, not one trajectory's 24,576 tokens.
  Preserve the terminal [v4 result](work/research-plan/gemma-full48-fit-v4-result.md);
  it is neither an OOM nor a pass.
- [ ] Execute the corrected fresh v5 fit exactly once after separate authorization.
  It changes only the observer assertion. Constructed complete paths fit, but sampled
  attempts above 24,576 must fail explicitly without truncation or resampling.
- [ ] Restart from the pinned base: 48 tasks × two passes × four fresh attempts =
  96 scheduled groups / 384 attempts. The user authorized overnight execution with
  **no elapsed-time cutoff**; timing estimates are advisory. Do not launch until a new
  qualification contract passes. Production remains at 0 groups / 0 attempts.

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
