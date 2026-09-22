# Gemma microbatch probe: training and resume passed

**The same BF16 model now completes the three-step GRPO engineering probe on the
RTX 3090.** Processing one attempt at a time, with four gradient accumulations,
avoids the allocation that stopped the [full-batch run](gemma-probe-result.md).
The reward group still contains four attempts. No task, reward, token budget,
precision, model revision, or learning-rate change was made.

This proves short-context training machinery and memory fit for this bounded run.
It does not establish better prose, semantic continuity, or long-form project memory.
A configured 4,096-token cap also does not prove every trajectory at that limit will fit.

## Evidence

Executed on 2026-09-22 from source commit `f0b0009`. A fresh release, 91 scorer
fixtures, all nine cached-tokenizer paths, and a fresh baseline were validated.
The original failed run was not overwritten or reused as this run's baseline.
Independent source review approved accumulation, normalization, masking and resume.

Before GPU execution, 323 tests ran with one skip. A real CPU run verified three
ordered task groups, twelve single-attempt loss forwards, and exactly matching
uninterrupted/resumed adapter, optimizer, scheduler, RNG and sampled-token state.
Full-batch versus accumulated CPU adapter tensors differed by at most `1.49e-8`;
Adam moments also agreed, checking gradient scaling rather than just update direction.
Changing microbatch size on resume was rejected before model/RNG mutation.

The initial CPU observer was attached to a model call that PEFT bypasses; those
failed assertions are preserved. The corrected nested-decoder observer and the
final all-groups token comparison passed. No training logic was changed for the observer.

### Actual GPU training

| Task | Four mechanical rewards |
|---|---|
| wave1-train-023: revision | 1, 0, 1, 1 |
| wave1-train-035: wiki | 0, 0, 0, 0.846154 |
| wave1-train-043: retrieval | 0.916667, 0.75, 0.5, 0 |

Every group had nonzero variance. The existing unavailable/all-tied guards remained
active. All groups and failed attempts are preserved; there was no resampling.

- Step 1 changed adapter parameters and saved a complete trainer checkpoint.
  LoRA-B L1 changed from 0 to `75.73054428212345`.
- Resume loaded step 1; the restored parameter hash exactly matched the saved
  post-update hash, `2e7e900e5c61a6ceaaaa9454438faf9336acaedcdc07c66b346bfb8f0927912c`.
- Training then reached step 3, with LoRA-B L1 `132.94712238758802` and a different
  parameter hash, `a7077981dd0bd805b62d42e78880acd22e5c1fc647805ad256752992dcab07a0`.
  Checkpoints 2 and 3 remain; retention pruned checkpoint 1 as designed.
- Adapter evaluation loaded the exact resumed export and verified all **1,050
  resident adapter tensors** against the saved tensors before sampling.

Parameter change is engineering evidence, not a quality metric. Exact uninterrupted
versus resumed trajectory equivalence was tested on CPU; a second uninterrupted
Gemma GPU training run was not performed.

## Resource measurements

| Stage | Wall time | Torch allocated peak | Torch reserved peak | Worker RSS peak | Run disk growth |
|---|---:|---:|---:|---:|---:|
| Base evaluation | 606.56 s | 9.807 GiB | 10.244 GiB | 10.553 GiB | 2.485 MiB |
| Train to step 1 | 347.65 s | 16.810 GiB | 18.359 GiB | 10.704 GiB | 308.823 MiB |
| Resume to step 3 | 752.36 s | 18.249 GiB | 19.916 GiB | 10.713 GiB | 310.766 MiB |
| Adapter evaluation | 1,019.60 s | 10.052 GiB | 10.814 GiB | 10.564 GiB | 2.559 MiB |

All four stages completed within **2,726.17 seconds (45.44 minutes)** of the new
3,600-second allowance. The earlier failed run's 880.65 seconds remain separately
recorded; they were not erased. There were no retries, downloads, installations,
paid calls, or final-test accesses. No process needed stopping for this run.

GPU peaks describe the PyTorch allocator, not all device memory. Disk growth uses
parent whole-run snapshots, excluding console logs in the work-artifact directory.
The original run failed before reaching a full training peak, so its recorded peak
is not a valid denominator for a percentage memory-saving claim. The stronger result
is that the unchanged four-attempt group now completes an optimizer update.
After workers exited, the device showed 22,679 MiB free and 0% utilization.

## Matched development results

Each cell is **base → adapter** mechanical reward, with the same task and seed.
Both conditions contain all twelve planned slots and no unavailable rewards.

| Case | Seed 104729 | Seed 130363 |
|---|---:|---:|
| F2-01 revision | 0 → 1 | 0 → 1 |
| F2-03 revision | 0 → 0 | 0 → 0 |
| F4-01 wiki | 0 → 0 | 0 → 0 |
| F4-03 wiki | 0 → 0 | 0 → 0 |
| F5-01 retrieval/use | 0.75 → 0.75 | 1 → 0.75 |
| F5-03 retrieval/use | 0.5 → 0.25 | 0.5 → 0 |

Mean reward moved from **0.229167 to 0.3125**: two attempts improved, three worsened,
and seven were unchanged. Completed executions rose from six to seven; generation-limit
failures fell from six to five. All wiki attempts still hit generation limits. The
higher mean comes from one revision case at both seeds, while three retrieval/use
attempts lost points. Do not describe this as a general improvement.

The fresh baseline also differs from the earlier run's 0.1875 mean; these seeded GPU
runs did not reproduce identical outcomes. Comparisons here use only the fresh,
matched baseline, not the lower historical value. Twelve attempts and literal
mechanical checks are not enough to establish a reliable capability change. No
semantic or literary judging was performed.

## Reproduce or inspect

Use the [probe guide](../../docs/grpo-probe.md) for a new, separately scoped run.
Do not rerun completed phases. The local evidence directory in the main checkout is
`runs/grpo-gemma-probe-microbatch-v1`, binding
`d5cbd1680f87fed7d04a561102599d2ed7c4f98a9f8887eb7dd813d432493b4f`.

- `resource-ledger.json`: four completed model phases and resource accounting.
- `training/groups/`: exact sampled IDs, masks, traces, files and rewards.
- `training/invocations/*/complete.json`: parameter evidence and saved exports.
- `training/checkpoint-2/`, `training/checkpoint-3/`: complete trainer state.
- `adapter-eval/adapter-reload.json`: exact resident adapter verification.
- `base-eval/scores.json`, `adapter-eval/scores.json`, `adapter-eval/paired.json`:
  all twelve task/seed comparisons, including execution failures.

The final adapter is under
`training/invocations/a7a6b41d5cb94f71b82c1a376bfc58ab/adapter` within that directory.
Meridian work item `grpo-microbatch` retains review, CPU evidence and console logs.
The [root TODO](../../TODO.md) tracks the next experiment; longer context and
semantic/literary effectiveness remain open.
