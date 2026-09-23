# Production Gemma controlled GPU fit

`scripts/run_grpo_gpu_fit.py` qualifies a single frozen memory profile before the
[full48 runner](grpo-full48.md). Inspection performs no GPU queries, model imports,
downloads or writes. Preparation creates fresh evidence; execution requires both
`--phase fit` and `--execute`. There is no retry, alternate profile, elapsed cutoff,
or production launch.

The desktop-admitted v2 attempt is terminal: native 32767+1 generation passed, while
controlled training OOMed during the first backward pass before an optimizer update
or checkpoint. See the [measured result](../work/research-plan/gemma-full48-fit-result.md).
The separately identity-bound v3 attempt is also terminal. It retained all training
and token settings while requiring a headless GPU and expandable allocator segments.
Ownership and native generation passed, but training OOMed in the FP32 MLP LoRA path
before an optimizer update. See the [v3 result](../work/research-plan/gemma-full48-fit-v3-result.md).
Neither failed profile may be rerun. The current v4 profile instead reduces only the
complete trajectory/context to 24576 tokens while preserving FP32 all-linear LoRA,
8192 active fit actions, production action limits, and every task/output requirement.
Production remains blocked until v4 passes.

Use the already qualified Python environment and cached model only:

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
PYTHON=/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/memory-qualification-v1/env/bin/python
FIT=/absolute/new/gpu-fit-evidence
"$PYTHON" scripts/run_grpo_gpu_fit.py "$FIT"
"$PYTHON" scripts/run_grpo_gpu_fit.py "$FIT" --phase prepare
"$PYTHON" scripts/run_grpo_gpu_fit.py "$FIT" --phase preflight
"$PYTHON" scripts/run_grpo_gpu_fit.py "$FIT" --phase fit --execute
```

The prepared profile binds repository sources, production settings, pinned TRL/Liger
source trees, ownership policy and exact ledger construction. A changed profile or
source refuses before runtime. An exclusive attempt marker prevents repeating any
execution, including an ownership rejection. After the user resolves a rejection,
prepare a fresh evidence directory; preserve the rejected directory.

V4 requires a headless RTX 3090: the complete NVML inventory must contain no graphics
or compute consumers and report at least 24000 MiB free. Unplugging a monitor does not
satisfy this policy while the graphical session remains active. The check runs before
the initial execution and again in each fresh generation/training process, before model
loading. Processes are never terminated. It is an admission snapshot, not a reservation.

Before Torch import, the runner sets both supported allocator environment names to
`expandable_segments:True` and refuses conflicting values or an already imported Torch
module. The allocator setting, headless policy, source, and recipe are part of fit and
production identity. Full48 train/resume enforce the same contract.

The only profile uses pinned `google/gemma-4-E2B-it` revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`, BF16 base, SDPA,
source-pinned `trl-6c5f135-streaming`, LoRA rank 8/alpha 16/all-linear/dropout zero,
DAPO beta zero, group four/microbatch one/accumulation four, ordinary AdamW at
`1e-5`, and nonreentrant activation checkpointing. It takes one optimizer step and
verifies a full checkpoint. Training and production share the public TRL config
builder; there is no copied loss or trainer subclass.

Each of four deterministic real-token ledgers contains exactly **24576 model
input tokens**, including its native initial prompt. A 16384-token prompt/observation
prefix fills the space before exactly **8192 active action tokens**. Diagnostic rewards
are `[0, 0.25, 0.75, 1]`; rotated token sequences avoid assigning different rewards
to identical action rows. These are controlled memory inputs, not sampled actions,
successful tasks, or native tool-rollout semantics. Production continues to sample
natively and retains its 8192 decision / 16384 total sampled / 24576 context caps.
The fit has 8192 active tokens per attempt; it is not an exhaustive stress test of
every mask arrangement or all 16384 active-token trajectories.

A separate cached-base process preserves native chat framing around a constructed
24575-token prefix and generates one token, reaching the 24576 context cap with
KV caching enabled. It verifies finite raw logits and retained prefill cache.
That process exits before training, releasing its complete CUDA context. Native
prefill evidence is distinct from the controlled training ledgers.

Evidence includes exact token IDs/masks, source/recipe/package identity, actual
per-microbatch loss/mask/attention/DAPO denominator observations, finite gradients,
changed adapter tensors, exact resident-versus-checkpoint adapter equality, optimizer
step counters, scheduler/RNG files and full checkpoint hashes. Each process records
Torch allocated/reserved peaks, process peak RSS, stage timing and full raw NVML
before/after. Failures preserve tracebacks and stop further stages; source mismatch,
ownership rejection, generation failure, OOM, nonfinite values or incomplete
checkpoint cannot produce a passing result. Inspect `result.json` and stage files.

A pass is finite memory evidence for this profile. It does not establish writing
quality, successful intact tasks, every possible trajectory shape, or ownership
throughout a later run. Production preparation must bind the final source checkout,
and production train/resume repeat ownership admission immediately before loading.
