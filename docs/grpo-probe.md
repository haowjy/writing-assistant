# First Gemma GRPO probe

This is the prepared three-task engineering experiment, not a full writing-training run.
Its scores measure mechanical compliance, not writing quality or semantic continuity.
CPU and tokenizer checks do not establish Gemma GPU fit. See the [root TODO](../TODO.md)
for execution status and the general [GRPO guide](grpo.md) for checkpoint contracts.
The [first GPU result](../work/research-plan/gemma-probe-result.md) is a completed baseline
followed by CUDA OOM before the first optimizer update. The failed run remains frozen;
the commands below describe the protocol, not a retry instruction.

## The task set

Three training cases are adapted from the existing wave1 release:

| Original ID | Assignment | Checks that matter here |
|---|---|---|
| `wave1-train-023` | Revise the second-violin scene | Changed draft, length, protected ending, unchanged unrelated files |
| `wave1-train-035` | Build the bakers' three-page wiki | Delivered pages, word limits, required links, unchanged source notes |
| `wave1-train-043` | Find the current studio permission | Retrieved evidence, current quotation, no withdrawn quotation or file edits |

The development set is `F2-01`, `F2-03`, `F4-01`, `F4-03`, `F5-01`, and `F5-03`
from custom-eval: two cases each for revision, wiki construction, and wiki use. Each
runs at two fixed seeds before and after training. These are development diagnostics,
not final tests. No final-evaluation or external benchmark cases enter this probe.

The [committed packet](../data/grpo-probe-v1/packet.json) retains the original scenarios,
source records, hashes, and explicit probe modifications. Scenes are shortened to
80–140 words; wiki limits and tool budgets are also narrowed. Originals remain unchanged.
New IDs start with `grpo-probe-v1-`. The three training cases share an author-connected
source group; the development cases use two separate worlds. This is a deliberately
small engineering set, not broad training coverage.

## What must pass before weights load

The prepared release verifies source roles, connected lineage, held-out exclusions,
and visible/private hashes. Scripted reference solutions exercise the real workspace
and tool loop. Counterexamples check missing, unchanged, empty, and broken outcomes.
These assistant-authored fixtures test the scorer; they are neither sampled candidate
successes nor human literary judgments.

Cached-tokenizer preflight then renders the good solutions through Gemma's actual
native protocol and measures prompt, action, observation, and accumulated token lengths.
A good path fitting the budget does not guarantee every sampled attempt will fit.
Context overflow remains explicit; history is never silently dropped. All nine scripted
paths fit: the largest boundary plus its next-call allowance is 2,251/4,096 tokens;
the largest complete action is 183/768, and the largest action total is 411/1,536.

Preparation hashes the code, release, and 91 fixture cases. Passing tokenizer preflight
seals `plan.json`; inspection and every model stage recheck that same evidence. Missing
or changed files block execution. A bare `preflight.json` is not proof of readiness.

The reward is a mean of declared mechanical checks, gated by completed execution and
required delivery. An unchanged revision or absent deliverable earns zero. Literal
checks cannot detect every paraphrased contradiction, and a successful file read does
not establish understanding. Semantic judgments remain unjudged.

## Bounded execution

The frozen model is local BF16 `google/gemma-4-E2B-it`, revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`. Training uses LoRA rank 8, learning rate
`1e-5`, no KL penalty, and non-reentrant activation checkpointing. Each task supplies
four attempts and one optimizer step: three steps total. Thinking stays enabled.

The memory-reduced profile keeps four attempts per reward group but processes one
attempt per loss forward/backward, accumulating gradients four times before updating.
This replaces the first run's four-attempt training batch without changing its tasks,
rewards, precision or token budgets. Full-group advantages are computed before splitting.
Use a fresh run directory; the original OOM run cannot resume under this profile.

Limits are 4096 context tokens, 768 generated tokens per decision, and 1536 generated
tokens per attempt. Evaluation uses the same BF16 precision and the same task conditions
for base and adapter, with seeds `104729` and `130363`. Training uses seed `42`.

There is a 60-minute aggregate GPU-stage allowance. Each model phase runs in a supervised
child process; timeout stops only that process group. A competing model allocation
blocks execution rather than triggering an automatic process kill. Known desktop consumers
are allowed within 256 MiB each and 768 MiB total, with at least 22,000 MiB free.
Unavailable rewards, unsupported framing, and all-tied reward groups stop training;
the runner does not keep sampling until it finds a group that produces an update.

Timeout and interruption preserve failures and charge elapsed time. An unresolved running
stage blocks further execution and reserves its full allowance until manually reconciled.
Do not delete a ledger entry to recover the budget or retry a failed phase.

## Commands

Run from the repository root using the existing training environment. The default is
inspection only. `prepare` and `preflight` require explicit execution but do not load
model weights; model stages do. Nothing downloads weights, installs packages, or calls
a paid judge.

```bash
PYTHON=/path/to/existing/training-environment/bin/python
export PYTHONPATH=src
# Choose a stable absolute path if using a disposable worktree.
RUN=/absolute/stable/path/grpo-gemma-probe-microbatch-v1
"$PYTHON" scripts/run_grpo_probe.py "$RUN"
"$PYTHON" scripts/run_grpo_probe.py "$RUN" --phase prepare --execute
"$PYTHON" scripts/run_grpo_probe.py "$RUN" --phase preflight --execute
```

Use a new destination for preparation; existing results are never overwritten. A source
change after preparation requires a new run directory and new validation. Documentation
changes alone do not invalidate the binding. The older `scripts/train_grpo.py` is a generic
unconfigured example; this experiment uses `scripts/run_grpo_probe.py`.

After inspection reports `offline-ready` and an exclusive GPU window is available:

```bash
"$PYTHON" scripts/run_grpo_probe.py "$RUN" --phase base-eval --execute
"$PYTHON" scripts/run_grpo_probe.py "$RUN" --phase train --stop-after-step 1 --execute
"$PYTHON" scripts/run_grpo_probe.py "$RUN" --phase resume \
  --resume "$RUN/training/checkpoint-1" --execute
```

Use the sealed inference-adapter path reported by the successful resumed invocation:

```bash
"$PYTHON" scripts/run_grpo_probe.py "$RUN" --phase adapter-eval \
  --adapter /path/reported/by/resumed/invocation/adapter --execute
```

`train` pauses at step 1; `resume` advances the same schedule to step 3. The runner
requires a completed matched base evaluation before training, and admits only this run's
resumed step-3 export for adapter evaluation. Each phase defaults to 20 minutes, shortened
to the remaining aggregate allowance; `--timeout-seconds` can request a smaller limit.

Keep the run directory and code unchanged across pause/resume. Preserve failed runs;
do not edit their tasks, rewards, seeds, or completed artifacts to make them pass.

## Read the evidence, not just the exit status

The resource ledger records stage outcomes, elapsed time, memory measurements, disk
usage, and identities. GPU peaks are PyTorch allocator measurements, not full-device
sampling; RSS belongs to the worker process. Disk records show whole-run snapshots and
growth. Unavailable measurements stay null. Per-attempt traces retain failures.
Training evidence must show
an actual adapter change; resumed training must change parameters beyond the restored
checkpoint, not merely load already-nonzero weights. CPU verification exercises step 1→3
with retention of only two checkpoints: the comparison still works after checkpoint 1
is pruned. Adapter reload compares saved tensors against the actual resident PEFT adapter,
separately from full optimizer-state resume.

A complete run compares all planned base/adapter development attempts under the same
conditions. `base-eval/scores.json` and `adapter-eval/scores.json` retain all 12 planned
slots; `adapter-eval/paired.json` pairs them by task and seed. Missing or failed attempts
stay visible; unavailable scores never become invented zeros. Training preserves sampled
native tokens, while both evaluations use the same history re-rendering policy.
A higher mechanical score alone
does not demonstrate better prose or long-form memory. The [root TODO](../TODO.md)
tracks which execution evidence has actually been obtained.
