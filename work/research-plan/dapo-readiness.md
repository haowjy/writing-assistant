# DAPO full-round readiness

**Two fresh-base attempts stopped before the planned 48-group pause.** The
[first run](gemma-full48-first-run-result.md) stopped after checkpoint 14 with two
unavailable tool-protocol attempts in group 15. The v6-qualified
[second run](gemma-full48-second-run-result.md) stopped after checkpoint 31:
group 32 sampled four attempts, one ending a final answer with `<eos>` before a
scheduled follow-up. Both remain separate and terminal under the present no-resample
runner. A first correction treated the second boundary as a candidate failure;
the current policy keeps the sampled EOS and appends a masked user-turn suffix.
Focused CPU tests pass, but no new GPU qualification or production outcome exists. Both earlier
32768-token profiles OOMed before an update. V4 reduced
only context but its observer rejected the correct DAPO denominator after the first
forward. V5 changed only that assertion and passed native generation, four accumulated
microbatches, one optimizer update and a complete checkpoint. This qualifies controlled
memory, not sampled task trajectories, reward quality, or production completion.
Tied-group continuation, reward, scheduling, recovery, and source-pinned CPU
qualification also passed. Those proofs do not authorize replaying group 15.

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
intermediate checkpoints, and deliberately stop at pass one/update 48 and explicitly resume to update 96. Final evaluation is separate. A completed schedule has 96 optimizer steps,
including ordinary TRL steps for tied groups. Those steps are not evidence of
new relative reward information; rounding residuals and momentum can still move weights. Failures can still stop the schedule before full coverage.

The earlier approximately ten-hour estimate extrapolated three shortened training
tasks. It is **not a measured ETA for these intact tasks**. Fourteen have nine
follow-ups; no full-task timing exists yet. V5's 219.636-second controlled training
stage excludes sampled tool conversations, reward execution and production checkpoint
frequency; it cannot supply a full48 ETA.

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
is implemented in the separate full-round recipe and CPU proof.

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

Integration review found that unexpected workspace and host filesystem exceptions
could be returned as candidate rewards. The shared agent/rollout path now preserves
an explicit infrastructure classification: EIO and unexpected harness failures make
the whole reward group unavailable, while candidate path/schema/missing-file errors
remain ordinary tool observations. Focused regressions reproduce both host failures.

The same review rejected the 131072-token production cap before GPU execution. The
installed Gemma/SDPA mask path constructs quadratic dense boolean masks; one 131K mask
alone is 16 GiB. The enforced full48 envelope is now 8192 tokens per decision, 16384
sampled action tokens and 24576 total trajectory tokens. Cached-tokenizer accounting
measured maxima of 2699 tokens for every initial file, 210 for contract-required reads,
141 for followups and 850 for the rendered initial prompt. Reserving the full sampled
total and those known inputs leaves 4502 tokens for tool framing, repeated reads and
other observations. Original step/tool/read/storage budgets and every task/output
remain unchanged. The context cap does not
promise that an attempt can spend the entire optional read allowance alongside the
maximum output; overflow remains explicit failure, never truncation.

## Production fit command

The [inspect-first GPU fit command](../../docs/grpo-gpu-fit.md) binds one fixed
24576-token controlled training profile and a separate native 24575+1 generation
check. V5 and full48 train/resume set PyTorch expandable allocator segments before
Torch import, require an empty complete GPU process inventory, and require at least
24000 MiB free. Source/profile and prepared-data checks precede model loading. A
passing CPU suite or prepared profile is not a passing GPU fit; production remains
blocked until live evidence passes. The fit command never starts production.

The first v2 invocation stopped at ownership before either production model load.
The RTX 3090 had 22676 MiB free, but total listed process allocation was 1079 MiB
against the unchanged 768 MiB cap. The user then explicitly requested termination
of Steam and approved Xwayland/Ghostty as ordinary desktop consumers. Steam and its
web helpers exited. The allowlist expansion did not change the 256 MiB per-process,
768 MiB total or 22000 MiB free-memory limits. The original rejection remains in
`gpu-fit-v2/runtime/ownership-before/`.

The one permitted fresh attempt then passed ownership with 681 MiB listed and 23203
MiB free. Native 32767+1 generation passed in 42.906 seconds, peaking at 16.767 GiB
Torch allocated and 20.072 GiB reserved. Controlled training reached a finite first
microbatch loss with the expected 32768 total / 8192 active tokens, then OOMed during
activation-checkpoint recomputation in backward while requesting another 768 MiB.
Peak Torch allocation was 21.379 GiB and reservation 22.223 GiB; PyTorch reported
114 MiB free at failure. No optimizer update or full checkpoint completed. The
profile was not retried or changed. The failed 768 MiB allocation exactly matches a
32768 × 6144 FP32 all-linear LoRA MLP projection. See the
[measured fit result](gemma-full48-fit-result.md).

The v3 contract changed only resource admission and allocator behavior: no listed GPU
consumer, at least 24000 MiB free, and `expandable_segments:True` bound before Torch
import. Model, 32768-token ledgers, BF16 base, FP32 rank-8 all-linear LoRA, DAPO,
checkpointing, and source pins remained unchanged. Ownership passed with zero consumers
and 24085 MiB free; native generation passed. Training reached 22.785 GiB allocated
and 23.043 GiB reserved, then the MLP `down_proj` LoRA path requested another 768 MiB
with 336.62 MiB free. Reserved-but-unallocated memory was only 83.78 MiB, ruling out
fragmentation as a sufficient fix. No optimizer update or checkpoint completed. See
the [v3 result](gemma-full48-fit-v3-result.md).

## Next gate and evidence

The v4 24576-token attempt passed headless ownership and native 24575+1 generation.
Its first training forward also completed at 17.097 GiB allocated / 17.711 GiB
reserved, but the observer then rejected the correct DAPO denominator: four ledgers
with 8192 active tokens produce a group denominator of 32768, not one ledger's 24576
total trajectory tokens. Backward, the remaining microbatches, optimizer step, and
checkpoint did not run. V4 is terminal, is not an OOM, and is not a fit pass. See the
[v4 result](gemma-full48-fit-v4-result.md).

The corrected v5 fit passed all three headless GPU ownership checks with zero
consumers and 24085 MiB free. Native 24575+1 generation passed. Four exact
24576-token microbatches with 8192 active tokens each showed the 32768 group DAPO
denominator and zero masked gradient; training peaked at 21.096 GiB Torch allocated /
21.414 GiB reserved. Optimizer step 1 changed the adapter, passed exact equality
against all 1050 saved adapter tensors, and sealed a full checkpoint with optimizer,
scheduler and RNG. See the [v5 result](gemma-full48-fit-v5-result.md). This is a
controlled memory pass, not a sampled writing result.

The source-pinned path is integrated with the [full48 runtime](../../docs/grpo-full48.md),
whose finite schedule and exact pass-one recovery pass a tiny CPU proof. The first
production attempt reached 14 committed groups and 56 committed attempts; group 15's
four sampled attempts remain uncommitted and unavailable, so this run cannot resume.
A different corrected run must start from the pinned base under a fresh identity.
The new protocol handling and all-checkpoint policy have CPU regression evidence.
The single [v6 controlled fit](gemma-full48-fit-v6-result.md) passed native generation,
all four accumulated microbatches, optimizer step 1 and a complete checkpoint,
with zero GPU consumers and 24085 MiB free at all three admission points. The fresh
pinned-base release was prepared and launched under a new identity, then stopped
at 31 updates on a different EOS/follow-up boundary. The revised rollout keeps
the sampled EOS and appends the next user turn as a masked external suffix;
its CPU tests were red before the revision and pass now. A documented exploratory
fork from checkpoint 31 would have to preserve group-32 sampled prefixes, generate
the missing response after slot 002's EOS, and verify RNG/optimizer behavior;
ordinary resume remains blocked.
No changed-source fit or new production launch has occurred. Constructed complete
paths measured only 598–4066 tokens, not sampled upper bounds; a sampled trajectory
over 24576 fails explicitly without truncation or resampling.

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
and the adjacent test/live-command logs. The terminal fit evidence is in
`gpu-fit-v2/approved-desktop-v1/`, `gpu-fit-v3/headless-expandable-v1/`, and
`gpu-fit-v4/context-24576-v1/`, and `gpu-fit-v5/context-24576-v1/`, with raw logs in
their adjacent runtime files. Durable summaries are
[gemma-full48-fit-result.md](gemma-full48-fit-result.md),
[gemma-full48-fit-v3-result.md](gemma-full48-fit-v3-result.md),
[gemma-full48-fit-v4-result.md](gemma-full48-fit-v4-result.md), and
[gemma-full48-fit-v5-result.md](gemma-full48-fit-v5-result.md). A failed postprocessing
hook assumption is preserved; correction required no package changes or model reruns.
Original data and previous GPU runs were not modified. The controlled v6 GPU fit passed and two separate production training attempts
ran, but neither completed all 96 groups. No paid judge calls occurred; mechanical
scores do not establish writing quality.

See [current work order](../../TODO.md), [GRPO usage](../../docs/grpo.md), and the
[previous short-run result](gemma-microbatch-result.md).
