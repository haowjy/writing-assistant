# Gemma GRPO probe result

The training path works end to end on the RTX 3090, including resume from a checkpoint
and exact adapter reload. The run was too short to show learning.

Run: `runs/grpo-gemma-probe-microbatch-v1/` (2026-09-22, not committed). `plan.json`
holds the frozen settings; `resource-ledger.json` and `adapter-eval/paired.json` hold the
numbers below.

## Setup

`google/gemma-4-E2B-it` at revision `3e22461f`, bf16, no quantization. LoRA rank 8,
learning rate 1e-5, 4096 context tokens, 1536 generated tokens, four attempts per group,
microbatch size 1. Three optimizer steps over three frozen tasks: a fresh run to step 1,
then a resume from `checkpoint-1` to step 3. Rewards were mechanical only.

## What passed

- Training and resume both completed, and trainer state reached step 3.
- Reloading the saved adapter matched exactly across all 1050 tensors.
- Every stage finished inside the 3600-second GPU budget, about 45 minutes in total.

## Resource use

| Stage | Seconds | Peak RAM | Peak reserved VRAM |
|---|--:|--:|--:|
| Base evaluation | 607 | 11.3 GB | 11.0 GB |
| Training | 348 | 11.5 GB | 19.7 GB |
| Resume | 752 | 11.5 GB | 21.4 GB |
| Adapter evaluation | 1020 | 11.3 GB | 11.6 GB |

The run directory is 631 MB, of which checkpoints are most. Resume reserved 21.4 of the
card's 24 GB, so longer context will need its memory measured before it is scheduled.

## Paired evaluation

Twelve pairs: six tasks at two seeds. Mean reward moved from 0.23 to 0.31, which three
steps cannot support as a learning claim.

- F2-01 rose from 0 to 1 at both seeds.
- F5-01 fell from 1 to 0.75 at one seed and held at the other.
- F5-03 fell at both seeds, from 0.5 to 0.25 and from 0.5 to 0.
- The other three tasks did not change.

Attempts completed were 6 of 12 for the base model and 7 of 12 for the adapter. Semantic
quality was not judged.

## Later work

This run came before the DAPO and task-graph work; see the status in the
[root TODO](../../TODO.md). The DAPO run reached checkpoint 48. Its notes report tests of
action and mask fidelity, which this run's outputs do not show.

The trainer logged `completions/clipped_ratio` as 1.0 at all three steps here, and it was
still 1.0, 1.0, 1.0, and 0.75 at the last four steps of the checkpoint-48 run. Either
most completions hit their length limit, or the metric misreads this setup. This is
unresolved in what has been checked so far.
