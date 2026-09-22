# DAPO full-round readiness

**The DAPO configuration and CPU proof pass; the intact 48-task GPU run has not
started.** Tied-group continuation is approved and implemented. Full-task token
budgets, reward gates and long-trajectory memory remain unresolved. Raising the old probe's step count alone
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
| Primary integration suite after tie-policy addition | 325 tests run, one skipped, success |
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
| Reward shortcuts | Fabricated unchanged F2 drafts plus “Done.” earn 0.8 under the generic smoke callback despite completion=0. “Done.” without retrieval earns 4/7 on F5. These are scorer counterexamples, not model results. |
| Missing full-task contract | The three shortened probe goldens fail original word requirements. Their dedicated scorer requires contracts absent from the originals; the generic scalar ignores intermediate turns and semantic rubrics. |
| Schedule and seeds | The generic ceiling is 20 updates. Its 32-decision seed stride can overlap adjacent attempts under the originals' 48-step envelope. |

A faithful mechanical-only reward needs explicit delivery/change evidence,
retrieval checks and intermediate-turn treatment. Existing nonempty files must not
count as new writes. Semantic and literary judging remain uncalibrated; mechanical
success cannot establish prose or nuanced continuity quality.

## Memory and tied groups

Microbatch 1 solved the short-run OOM, not arbitrary trajectory length. Ordinary
TRL computes dense logits for the padded completion, including loss-masked
observations. With Gemma's 262144-word vocabulary, one FP32 logits tensor costs
8 GiB at 8192 positions or 16 GiB at 16384 positions, before other tensors. These
are allocation calculations, not measured peak memory or fit.

The installed fused route is gated on absent Liger and does not pass/apply Gemma's
configured logit softcap. Installing a package and switching that flag is therefore
not an established correct solution. A supported memory-efficient path needs
model-specific probability/gradient verification and measured fit. No installation,
download, precision change or model replacement has been authorized by this report.

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

## Next gate and evidence

Implement and review a separately bound full48 runtime/reward recipe with faithful token budgets, collision-free seeds, finite-work
supervision and coverage-aware recovery. Prove native long-trajectory memory fit
before committing to the full run. Do not silently truncate, shorten, omit or
resample tasks to make the schedule finish.

Detailed evidence is in the local work item:
`/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/`.
Key files are `data-readiness.md`, `readiness-matrix.{md,json,csv}`,
`reward-counterexamples.json`, `runtime-study.md`, `core-report.md`,
`primary-tests.log`, `primary-dapo/smoke.json`, `ties-suite.log`,
`ties-cpu-verified/smoke.json` and `ties-regression-{grpo,dapo}/smoke.json`.
Original data and previous GPU
runs were not modified. No new GPU training, package install, download or paid call
occurred during this readiness work.

See [current work order](../../TODO.md), [GRPO usage](../../docs/grpo.md), and the
[previous short-run result](gemma-microbatch-result.md).
