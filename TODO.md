# Current TODO

Start here for the active work order. Detailed designs live in the
[training plan](work/research-plan/training-experiments.md); the
[supporting checklist](work/research-plan/TODO.md) is not an additional prerequisite list.
Optional ideas live in [work/FUTURE.md](work/FUTURE.md).

The goal is better long-form project memory and effective use of large writing
projects, supported by a maintained wiki, with better prose alongside it.

## Next: prove GRPO works with the Gemma already on disk

- [ ] Prepare a bounded **Gemma E2B short-context GRPO probe**, initially around
  2–4K total tokens per attempt. Use a few mechanically scored tasks, several attempts
  per task, and an explicit step/time budget. No SFT stage or Qwen download is needed.
- [ ] Connect generation, workspace tools, reward calculation, and GRPO updates.
  Check that learning applies to candidate-generated actions, not user or tool text.
- [ ] After the probe scope is approved, run it on the **RTX 3090**. Verify actual
  adapter updates, checkpoint save/reload and resume; record peak GPU/RAM use, runtime,
  and disk growth. A successful load alone is not a passing training probe.
- [ ] Extend the passing probe to longer, multi-turn wiki tasks. Add declared Astra
  author simulation only where needed, measure cache reuse, and test retention and
  use of earlier decisions. Increase context gradually rather than jumping to 256K.

Gemma readiness checked: `google/gemma-4-E2B-it` at revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7` has its weights (about 10.25GB), tokenizer,
and configuration cached locally. No trained adapter was found under `runs/` in the
bounded inventory. The SFT dataset is unprepared; it is not required for this GRPO probe.

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
