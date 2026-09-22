# Current TODO

Start here for the active work order. Detailed designs live in the
[training plan](work/research-plan/training-experiments.md); the
[supporting checklist](work/research-plan/TODO.md) is not an additional prerequisite list.
Optional ideas live in [work/FUTURE.md](work/FUTURE.md).

The goal is better long-form project memory and effective use of large writing
projects, supported by a maintained wiki, with better prose alongside it.

## Next: prove GRPO works with the Gemma already on disk

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
- [ ] Scope a memory-reduced follow-up for the training output-conversion bottleneck.
  Keep the failed run frozen. Actual Gemma adapter updates, checkpoint save/reload,
  resume, and matched adapter evaluation remain unverified; a load is not a passing probe.
- [ ] Extend the passing probe to longer, multi-turn wiki tasks. Add declared Astra
  author simulation only where needed, measure cache reuse, and test retention and
  use of earlier decisions. Increase context gradually rather than jumping to 256K.

Gemma readiness checked: `google/gemma-4-E2B-it` at revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7` has its weights (about 10.25GB), tokenizer,
and configuration cached locally. A tiny random CPU model passed exact step-1→3
resume even after checkpoint 1 was pruned. Real Gemma inference and grouped rollouts
ran, but the frozen training configuration did not fit; no trained adapter exists.
Semantic/literary judging still needs
calibration before substantive writing optimization. The SFT dataset is unprepared and
is not required for this GRPO probe.

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
