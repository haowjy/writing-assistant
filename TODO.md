# Current TODO

Start here for the active work order. Detailed designs live in the
[training plan](work/research-plan/training-experiments.md); the
[supporting checklist](work/research-plan/TODO.md) is not an additional prerequisite list.
Optional ideas live in [work/FUTURE.md](work/FUTURE.md).

The goal is better long-form project memory and effective use of large writing
projects, supported by a maintained wiki, with better prose alongside it.

## Status: `main` is behind the real work

Training work continued off `main` through 2026-10-03 and is not merged. The newest
branch is `phase9/resume-openai`, 452 commits ahead of `main`; `feat/dapo-full-rounds`
is 59 ahead. Evidence and the decision log live outside the repo in
`~/.meridian/context/orange-juniper-leaf/work/`, mainly `task-graph-training/decisions.md`
and `task-graph-training/probes/deepseek-primary-check/report.md`.

- The [three-step Gemma probe](work/sft/gemma-grpo-probe.md) passed, and a DAPO run on
  Gemma E2B then reached checkpoint 48. Neither run established a prose improvement.
- Task-graph training is stopped before its B0 and E1 runs. The DeepSeek judge failed its
  pre-registered ranking gate, tau-b 0.527 against a required 0.60.
- Sonnet 5.5 scored 0.697 and Opus 5.5 scored 0.600 in an exploratory comparison on the
  same stories. That comparison does not qualify either model as the judge.
- DeepSeek author-v3 conformance also failed, 3 of 33 cases against a 5% threshold.

## Next: decide how task-graph training resumes

- [ ] Decide whether to stop this experiment or authorize a new pre-registered judge
  qualification. Do not retune the rubric or threshold against the failed test.
- [ ] Integrate the reviewed lanes R2c, R2e, and AV4-A, then finish AV4-B/C and R2d.
- [ ] Run the post-repair torch-enabled test suite and the cross-lane gates.
- [ ] Run J4/H1 human validation and integrated simulated-author validation.
- [ ] Decide how to bring `phase9/resume-openai` and `feat/dapo-full-rounds` into `main`.

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
