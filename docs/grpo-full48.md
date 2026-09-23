# Intact full48 preparation

**The dedicated full48 runner preserves all 48 original tasks and mechanical rewards.**
CPU scheduling, recovery and the source-pinned streaming integration are verified.
Both desktop-admitted and headless/expandable-segments training-memory profiles failed
on the RTX 3090. Production remains blocked. See
[current readiness](../work/research-plan/dapo-readiness.md).

## Inspect and validate

Run from this checkout. Set `PYTHON` to the existing environment's Python executable
and `RELEASE` to the original `data/processed/gen-wave1-train` directory, which may
live in the main checkout rather than a worktree. Validation requires a new evidence directory;
it preserves every fixture result and refuses to overwrite previous evidence.

```bash
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src
"$PYTHON" scripts/prepare_grpo_full48.py inspect "$RELEASE"
"$PYTHON" scripts/prepare_grpo_full48.py validate "$RELEASE" \
  --evidence /absolute/new/full48-fixtures
```

Optional cached-tokenizer measurement takes a local snapshot directory, never
model weights:

```bash
"$PYTHON" scripts/prepare_grpo_full48.py measure "$RELEASE" \
  --evidence /absolute/new/full48-fixtures --tokenizer "$TOKENIZER_SNAPSHOT"
```

The scripted paths use repetitive mechanical text, fixed short thinking, short
intermediate replies and final-pass delivery. Their token counts are **not**
sampled successes, realistic prose-length forecasts, upper bounds, or GPU fit
proofs. They do not select a training context limit.

## Bound reward contract

`data/grpo-full48-v1/contract.json` binds the original release files, task hashes,
catalog/manifest, frozen local exclusions and explicit mechanical requirements.
`load_full48_release()` rejects altered tasks or release bytes and returns the
original tasks, production admission metadata and reward specification.

```python
from pathlib import Path
from writing_agent.grpo_full48 import load_full48_release, mechanical_full48_reward

release = load_full48_release(Path("/path/to/gen-wave1-train"))
tasks = release["tasks"]
admission = release["admission"]
reward_spec = release["reward_spec"]
reward_callback = mechanical_full48_reward
```

These are inputs to the [trainer interface](grpo.md). The dedicated runner below
selects the explicit full48 admission profile; the default probe limits remain
unchanged. Do not shorten tasks or reuse the probe scorer.

Reward is zero unless execution and the complete declared turn sequence are
supported by saved evidence, and every required final artifact is delivered in its
proper channel at the original lower word bound. File delivery also requires a
successful write/patch, replay agreement, and change from the authoritative initial
text. Existing nonempty drafts and verbal claims are not writes. Successful tool
paths are interpreted consistently with the workspace's relative-path rules.

After delivery, the reward is the mean of the original mechanical checks and the
applicable additional read, reciprocal-link and retrieval/quote checks. Upper word
limits, scope violations, content errors and failed retrieval can retain partial signal. This asymmetric lower-bound
delivery gate is an explicit reward-design choice, not a tuned optimum. The spec
binds the scorer and shared scoring implementation; source changes require a fresh
experiment identity.

All semantic rubrics remain **unjudged**. Intermediate replies can prove that the
sequence occurred, not that clarification was faithful or earlier decisions were
preserved. Protected sentences must survive verbatim; the originals do not
unambiguously require their final position, so the scorer adds no suffix gate.
Fixtures deliberately expose these limitations. Neither passing fixtures nor a
higher mechanical scalar establishes better writing or project memory.

## Dedicated finite runner

`scripts/run_grpo_full48.py` composes the original release and this reward with
public TRL training. Inspection, preparation and preflight never load a model,
query a GPU, install packages, or download anything. Use a fresh run directory:

```bash
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src
"$PYTHON" scripts/run_grpo_full48.py "$RELEASE" /absolute/new/full48-run
"$PYTHON" scripts/run_grpo_full48.py "$RELEASE" /absolute/new/full48-run --phase prepare
"$PYTHON" scripts/run_grpo_full48.py "$RELEASE" /absolute/new/full48-run --phase preflight
```

Preparation binds the release, ordered schedule, recipe and source hashes.
Preflight proves data/settings/recovery admission and binds the exact source-pinned
streaming implementation; it does not certify native GPU memory fit. Execution
verifies TRL `1.14.0.dev0` commit `6c5f135` and Liger `0.8.3` source trees before
model loading. The default trainer seam and historical probe still require TRL 1.13.

Only after a separately reviewed memory strategy and new qualification contract pass,
set `PYTHON` to the qualified isolated environment, make the intended GPU visible, and
explicitly execute each invocation:

```bash
"$PYTHON" scripts/run_grpo_full48.py "$RELEASE" /absolute/new/full48-run --phase train --execute
"$PYTHON" scripts/run_grpo_full48.py "$RELEASE" /absolute/new/full48-run --phase resume --execute
"$PYTHON" scripts/run_grpo_full48.py "$RELEASE" /absolute/new/full48-run --phase coverage
```

The first invocation stops after verified checkpoint 48. Explicit resume selects
only the latest complete checkpoint and finishes at 96. Recovery from an earlier
clean boundary still stops at 48 first. A group sampled beyond the latest complete
checkpoint blocks recovery: this recipe never resamples or retries it. Preserve
that partial run for inspection. Missing, altered or unavailable evidence also
halts. Later incomplete checkpoint directories can be quarantined by the trainer
only when there is no uncommitted sampled work. Keep all attempt files and the
latest two full checkpoints. Up to eight invocation exports allow the two planned
invocations plus clean-boundary recovery, without unbounded adapter copies.

The supervisor holds an exclusive file lock inherited by its child. Ownership
survives supervisor death while the worker remains alive. It records process CPU,
RSS and IO snapshots plus group/result/checkpoint progress every 15 seconds, with
worker logs saved separately. Quiet logs have no termination meaning. There is no
elapsed-time deadline or automatic retry. An explicit interrupt signals only its
own child process group. No GPU ownership or memory-fit measurement is implied by
these CPU/process observations.

### Frozen recipe and budgets

| Setting | Value |
|---|---|
| Base | Fresh `google/gemma-4-E2B-it`, revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| Schedule | Original 001–048 in order, twice; 96 groups, 384 fresh attempts |
| Objective/ties | Public DAPO; ordinary TRL continuation through ties; unavailable halts |
| Batch | Group 4, microbatch 1, accumulation 4; one group per optimizer update |
| LoRA | BF16 base, rank 8, alpha 16, all-linear, dropout 0, bias none |
| Optimizer | AdamW, learning rate `1e-5`, constant scheduler, weight decay 0, beta 0 |
| Generation | Thinking on; temperature/top-p 1, top-k 0 |
| Checkpointing | Every update, latest two retained; nonreentrant activation checkpointing |
| Native token caps | 8192 per decision, 16384 sampled total, 32768 complete trajectory/context |
| Original task maxima admitted | 48 decisions, 64 tools, 12000 whitespace read units; original storage ≤262144 bytes |
| Seeds | `42 + (group * 4 + slot) * 48 + decision`, all indices zero-based |

Each slot reserves 48 seeds, covering the largest unchanged decision envelope.
The 384 intervals reserve seeds 42–18473 without overlap. Tasks with fewer allowed
decisions leave unused seeds; no seed is reassigned. Settings and this policy are
part of experiment identity.

The per-call allocation allows a 1200-word deliverable at an explicit planning
allowance of four tokens/word (4800) plus 3392 reasoning/framing tokens. A maximal
final three-file delivery uses 14400 tokens at that allowance and may be spread over
several decisions; the sampled total leaves 1984 additional action tokens. Cached
Gemma tokenizer accounting measured at most 2699 tokens for every initial file in a
task, 210 for all contract-required reads, 141 for followups, and 850 for the initial
rendered prompt. After reserving the full sampled total, every initial file, followups
and initial prompt, the 32768 context leaves 12694 tokens for tool framing, repeated
reads and other observations.

These are finite engineering allocations, **not token-per-word upper bounds**.
They admit every original task and a full final delivery without shortening any
brief, followup or task budget. The original 12000-unit read allowance remains, but
the context cap does not promise that an attempt can spend all of it alongside the
maximum output. Nor do the caps guarantee arbitrary repeated full drafts, unbounded
reasoning or adversarial tokenization. The earlier constructed
native paths measured at most 951 tokens per action, 3014 sampled tokens and 4066
trajectory tokens; compressible filler and short intermediate replies make those
existence checks, not realistic writing forecasts. Context overflow fails explicitly;
ordinary candidate exhaustion retains its failure evidence and mechanical reward
semantics. The prior 131072-token cap was removed before launch because the installed
SDPA mask path requires dense quadratic boolean masks. The frozen controlled training profile at the enforced
32768-token cap OOMed during its first backward pass; native sampled-task fit remains
unproven and cannot authorize production.

Coverage reports retain expected group/pass/task/slot identities in the prepared
schedule and validate them against observed evidence, including seeds and invocation
lineage. They distinguish started/scored groups, ties, relative-signal groups,
checkpointed optimizer boundaries, missing/duplicate slots and uncommitted work.
Physical attempt-directory counts remain visible even when identity evidence is
corrupt. A complete report requires all 384 identities and checkpoint 96. Mechanical
reward and tie counts establish no semantic or literary improvement.

### Live CPU schedule proof

This creates random tiny CPU weights, uses explicitly engineered task/reward
fixtures, and runs the same full48 scheduling and recovery lifecycle. It deliberately
uses the legacy TRL 1.13 environment; the separate streaming CPU proof is documented
in [GRPO usage](grpo.md), and the production composition is exercised by the GPU fit gate.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  /path/to/original-trl-1.13/python -m scripts.smoke_grpo_full48_cpu \
  --execute --output /absolute/new/full48-cpu
```

Three fixture tasks visited twice produce six updates and 24 attempts. The lifecycle
pauses after pass one/update 3, then resumes through update 6. The check compares
adapter, optimizer, scheduler, RNG and sampled token ledgers exactly with an
uninterrupted run; checks four tied and two signal groups; verifies all seeds,
ordering, accounting and two-checkpoint retention. This proves the schedule shape
without claiming 96 production updates or native Gemma/BF16 memory fit.

## Production GPU fit gate

The separate [fit command](grpo-gpu-fit.md) freezes one controlled memory profile
at 32768 tokens and exercises cached-base native generation before one DAPO update.
Its deterministic tokens and diagnostic rewards do not establish sampled success.
The v2 attempt passed native generation but OOMed during training backward before an
optimizer update; see the [measured result](../work/research-plan/gemma-full48-fit-result.md).
It must not be retried or treated as a pass. The separately bound v3 contract retained
the exact training recipe and 32768-token ledgers while requiring a headless GPU and
expandable allocator segments. V3 also OOMed during FP32 MLP LoRA backward before an
optimizer update; see the [v3 result](../work/research-plan/gemma-full48-fit-v3-result.md).
The fit command never launches production.

Both full48 `train` and `resume` verify the prepared identity and pinned runtime, bind
`expandable_segments:True` before Torch import, then admit the complete
`nvidia-smi -q -x` inventory before model loading. The inventory must have no graphics
or compute consumers and at least 24000 MiB free. Each admission retains raw XML and
a structured result under `ownership/`. No consumer is terminated. This is a current
inventory check, not a reservation against applications starting later.
