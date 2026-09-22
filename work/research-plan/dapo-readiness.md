# DAPO full-round readiness

**The DAPO configuration and CPU proof pass; the intact 48-task GPU run has not
started.** Tied-group continuation and the separate full48 mechanical reward are
implemented. Full-task runtime budgets and long-trajectory memory remain unresolved.
Raising the old probe's step count alone
would not produce the requested experiment.

## Intended workload

Restart from the original pinned Gemma base, not the trained GRPO adapter. Visit
`wave1-train-001` through `wave1-train-048` twice, with four fresh attempts per visit:
**96 scheduled groups, 384 scheduled attempts**. Exclude the older 54 tasks,
development cases and final evaluations. Preserve the original task requirements.

The user authorized overnight execution **without an elapsed-time cutoff**. Runtime
estimates are for planning, not termination. Errors and corrupt/unavailable evidence
still stop execution; quiet logs alone do not establish a stall. The previous
60-minute probe policy and sealed run artifacts stay unchanged.

Checkpoint full training state at completed update boundaries, retain recoverable
intermediate checkpoints, and deliberately test a pause/resume within the first
pass. Final evaluation is separate. A completed schedule has 96 optimizer steps,
including ordinary TRL steps for tied groups. Those steps are not evidence of
new relative reward information; rounding residuals and momentum can still move weights. Failures can still stop the schedule before full coverage.

The earlier approximately ten-hour estimate extrapolated three shortened training
tasks. It is **not a measured ETA for these intact tasks**. Fourteen have nine
follow-ups; no full-task timing or training-memory measurements exist yet.

## Completed and verified

Commit `0c6dfba` adds identity-bound `GRPOSettings.loss_type`, selecting public TRL
`grpo` or `dapo`. The default remains `grpo`; the old probe and admission limits are
unchanged. No custom RL loss, trainer subclass or upstream patch was introduced.
This selects DAPO normalization, not the full DAPO paper recipe.

The tiny random CPU model samples unequal action lengths, with external user
observations and padding masked out of loss. The proof checks the installed TRL
loss inputs against saved token ledgers, dense versus accumulated updates, Adam
moments, two passes over three task IDs, and exact step-1→6 resume across the pass
boundary. Those IDs share one toy brief; this is an engineering fixture.

| Check | Result |
| --- | --- |
| Dense versus microbatch-1 / accumulation-4 adapter difference | maximum `1.49e-08` |
| Adam first / second moment differences | maximum `1.40e-09` / `1.82e-12` |
| Interrupted versus uninterrupted | exact adapter, optimizer, scheduler, RNG and sampled token ledgers |
| Changed loss on resume | rejected before caller-state or output mutation |
| Primary integrated suite with full48 data tests enabled | 333 tests run, one skipped, success |
| Primary DAPO CPU rerun, Ruff lint/format | passed |

The implementation agent also reran the original GRPO CPU proof successfully.
These results do not establish native Gemma tool-trajectory fit, BF16 equivalence,
long-context performance, or writing improvement.

## Full-task audit

All 48 pass release/hash consistency checks and remain generated synthetic tasks,
not independently accepted successful trajectories. The bounded exclusion audit
found no checked held-out source conflict; that is not a global contamination
claim. Their initial native prompts fit at 402–850 tokens. Full paths remain
unmeasured.

| Blocker | Evidence and consequence |
| --- | --- |
| Task admission | All 48 exceed the generic 16-step ceiling; 14 exceed 32 tools and 38 exceed 8192 whitespace read units. Declared budgets are not measured consumption. |
| Per-call output | Tasks 005, 010, 040 and 045 require 900–1200-word replies; the cached-tokenizer audit rules out delivery within the old 768-token call cap. |
| Total output | Tasks 025, 029 and 030 require initially absent files totaling 2700, 1800 and 2700 words. Literal file tools cannot deliver them within the old 1536 sampled-token cap. |
| Generic reward shortcuts | Unchanged F2 drafts plus “Done.” earn 0.8 under the generic smoke callback; F5 “Done.” without retrieval earns 4/7. The separate full48 contract now scores these counterexamples zero. Do not use the generic callback for this experiment. |
| Probe derivatives are not originals | The three shortened probe goldens fail original word requirements. Use the new intact-release contract and fixtures, not the derivative scorer. |
| Schedule and seeds | The generic ceiling is 20 updates. Its 32-decision seed stride can overlap adjacent attempts under the originals' 48-step envelope. |

The [full48 preparation interface](../../docs/grpo-full48.md) now binds all original
task/release hashes, delivery/change evidence, required reads, reciprocal links and
retrieval/quote checks. It checks the declared turn sequence without pretending to
judge intermediate clarification. All required final artifacts must meet their
original lower word bounds before partial mechanical credit is available. This is
an explicit reward choice, not a tuned optimum. Existing nonempty files are not writes.

The revised offline validator passes 592 fixture expectations across all 48 tasks;
96 original missing/no-op counterexamples score zero. Primary review found and
corrected equivalent-path handling, preserving valid `./` and repeated-separator
read/write/patch operations without altering saved evidence. Protected text is
preserved exactly, but the originals do not unambiguously require its final position;
no suffix rule was invented. Semantic/literary quality and faithful use of intermediate
decisions remain unjudged. Repetitive fixture prose can pass, an explicit limitation.

All 48 complete constructed native paths were measured at 598–4066 total trajectory
tokens and 135–3014 action tokens. They use compressible filler, fixed short thinking
and short intermediate replies. These are not sampled successes, realistic length
forecasts, worst-case bounds, or evidence that a chosen context cap fits.

## Memory and tied groups

Microbatch 1 solved the short-run OOM, not arbitrary trajectory length. Ordinary
TRL computes dense logits for the padded completion, including loss-masked
observations. With Gemma's 262144-word vocabulary, one FP32 logits tensor costs
8 GiB at 8192 positions or 16 GiB at 16384 positions, before other tensors. These
are allocation calculations, not measured peak memory or fit.

The original TRL 1.13 environment's fused route is gated on absent Liger and does
not pass/apply Gemma's configured logit softcap. Installing Liger into that environment
and switching the flag would not establish correctness. A supported memory-efficient
path needs model-specific probability/gradient verification and measured fit.

There is now a concrete upstream option: [TRL PR #7077](https://github.com/huggingface/trl/pull/7077)
streams log probabilities with softcap support while retaining the native loss.
The user approved isolated installation and CPU qualification of TRL
`6c5f1350488e9bba9a71242c47db45f2869796fa` (`1.14.0.dev0`) plus Liger `0.8.3`.
**Installation is complete:** both packages import, their declared requirements are
satisfied, and all 140 TRL / 278 Liger Python files match the approved archives.
The fresh environment inherits existing dependencies through a `.pth` entry;
installation targeted only its own site-packages. This is not a self-contained copy
or a filesystem sandbox. Original package inventory and checked
source hashes remain unchanged. No dependency upgrades or model downloads occurred.

The user accepted the maintained upstream numerical variant: native Gemma softcaps
in BF16 before promotion; upstream streams that operation in FP32. Native BF16 parity
is not claimed, and no acceptance tolerance was invented. The repository now admits
legacy TRL 1.13.0 by default and the exact source-pinned stack through explicit
`implementation="trl-6c5f135-streaming"`. It verifies every TRL/Liger Python source
against the approved archives before model loading or caller mutation, then binds
the source identity and public configuration into the experiment manifest. Changing
implementation rejects resume. The legacy settings/plan shape and sealed probe
artifacts remain unchanged; repository source changes still invalidate older resumes.

Public `liger_kernel_config` explicitly disables every unrelated Gemma4 replacement.
The new live CPU train/resume proof observes unchanged module/class methods at trainer
initialization and all three train entries. No upstream package was edited, no
trainer was subclassed, and no custom loss or metadata spoofing was used. No
production model weights or GPU were used.

The first finite numerical probe used public `GRPOTrainer.compute_loss` on identical
small random Gemma4 text models with PLE/shared KV, LoRA rank 8 / alpha 16, synthetic tokens,
softcap 30 and masked observations. It used evaluation mode with autograd, not a
training loop. BF16 action-token differences were:

| Head regime | Temperature | Maximum logprob difference | LoRA gradient norm-relative difference |
|---|---:|---:|---:|
| Ordinary random head | 1 | 5.72e-6 | 1.02% |
| Ordinary random head | 0.7 | 7.63e-6 | 1.58% |
| Artificially saturated head (weights ×300) | 1 | 0.08098 | 0.79% |
| Artificially saturated head (weights ×300) | 0.7 | 0.11961 | 1.55% |

These are synthetic stress measurements, not cached E2B estimates. FP32 controls
had gradient norm-relative differences below 5.57e-7, although saturated cases
exceeded some absolute 1e-6 smoke diagnostics. That old FP32 smoke threshold is not
an established BF16 acceptance tolerance. No threshold was relaxed or precision
policy silently accepted; these results do not establish a formula bug.

Primary recomputation of all eight saved tensor pairs confirms identical parameters,
backbone states and direct loss-mask gradients. Streaming bypassed `lm_head.forward`;
no completion-wide vocabulary activation was saved. Trainer initialization preserved
model methods with kernel-replacement flags disabled. Those direct-loss comparisons
remain numerical evidence, separate from the later public train-loop qualification.

The user approved ordinary TRL continuation without resampling. Explicit,
identity-bound `tie_policy="continue"` is implemented; the default remains `"halt"`
for the old probe. Raw tied rewards pass to TRL, while Adam and the scheduler still
step. Mathematically zero advantages can have float32 residuals; those or momentum
can move weights. These are not skipped updates or new relative reward information.
Unavailable rewards still halt. Saved advantage estimates are labeled separately
from observed trainer values.

A separate tiny-model CPU proof covers six visits over two passes: four tied groups,
two nonzero-advantage groups, six optimizer steps and no resampling. It verifies
leading/middle/trailing ties, zero tied loss, moment decay and momentum-only movement,
dense/accumulated agreement, and exact step-2→6 checkpoint resume. An all-tied run
at reward 1.0 completes with unchanged parameters and advanced optimizer counters.
Review also reproduced eight rewards of 0.7 yielding actual advantages about
5.96e-4 and fresh-Adam movement despite zero estimated advantages. The CPU proof
now records this fractional case without altering ordinary TRL behavior. Saved group
reports distinguish ties from optimizer progress; full48 coverage accounting still
belongs to the unimplemented full-round recipe.

## Source-pinned CPU train/resume qualification

The live fixture uses random BF16 `Gemma4ForConditionalGeneration` with four text
layers, PLE, shared KV, softcap 30, LoRA rank 8 / alpha 16 and ordinary FP32 adapter storage.
It starts from empty output, takes six DAPO beta 0 optimizer steps over two passes,
and compares uninterrupted training with an explicit step-2 pause/resume.
Each visit samples four variable-length attempts, trains microbatch 1 / accumulation 4 and
consumes every action exactly once. Four visits tie at reward 1; two have heterogeneous
rewards. Adam and the constant scheduler advance through all six visits, without
resampling. Adapter, optimizer, scheduler, RNG and all 24 sampled token ledgers match
exactly after resume.

The 2,050-token external observation stays in attention and has zero direct loss
gradient. Actual streaming calls cross token 2048 and vocabulary 8192, including the
17-column vocabulary tail, with softcap 30. Changing one masked observation changes
a later action log-probability by `0.1126260757446289`, confirming conditioning in
this finite fixture. Group action counts and DAPO denominators are 20 despite the
long observation. Every trainer initialization and train-entry Liger dispatch
preserves observed model/module/class methods and functional cross entropy.

These results establish this CPU compatibility seam. They do not establish native
BF16 parity, native tool-tokenizer semantics, dense-versus-streaming update equality,
production GPU fit or writing quality. Existing native-token and legacy dense
accumulation evidence remains separate. The new script is
[scripts/smoke_grpo_streaming_cpu.py](../../scripts/smoke_grpo_streaming_cpu.py).

## Next gate and evidence

Implement the full48 runtime with faithful token budgets, collision-free seeds, finite-work
supervision and coverage-aware recovery. Prove native long-trajectory memory fit
before committing to the full run. Do not silently truncate, shorten, omit or
resample tasks to make the schedule finish.

Detailed evidence is in the local work item:
`/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/`.
Key files are `data-readiness.md`, `readiness-matrix.{md,json,csv}`,
`reward-counterexamples.json`, `runtime-study.md`, `core-report.md`,
`primary-tests.log`, `primary-dapo/smoke.json`, `ties-corrected-{cpu,grpo,dapo}/smoke.json`,
`ties-rereview.md`, `primary-contract-audit.json`, `primary-full48-fixtures/`,
`integrated-suite.log`, `integrated-ties-cpu/smoke.json`, `memory-path-options.md`
and `memory-primary-notes.md`. Installation evidence is in `memory-install.log` and
`memory-qualification-v1/{installation.md,environment.json,installed-source-hashes.json}`.
Numerical evidence: `memory-qualification-v1/cpu-gate/{report.md,results.json,*.pt}`
and `memory-qualification-v1/primary-verification.json`. Source-pinned compatibility:
`runtime-compat/report.md`, `runtime-compat/raw-summary.json`,
`runtime-compat/cpu/{summary.json,observations-full.json,observations-resumed.json}`
and the adjacent test/live-command logs. A failed postprocessing
hook assumption is preserved; correction required no package changes or model reruns.
Original data and previous GPU runs were not modified. Only the two approved package
archives were downloaded; no new GPU training, model download or paid judge call occurred.

See [current work order](../../TODO.md), [GRPO usage](../../docs/grpo.md), and the
[previous short-run result](gemma-microbatch-result.md).
