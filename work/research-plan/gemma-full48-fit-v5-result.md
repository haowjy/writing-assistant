# Gemma full48 24K controlled-fit v5 result

**The single v5 fit passed the controlled memory qualification on the RTX 3090.**
Native 24575+1 generation passed; public DAPO consumed four exact 24576-token
ledgers, accumulated four microbatches, changed the adapter at optimizer step 1,
and saved and verified a full checkpoint. This authorizes preparation for the
intact full48 run under the same resource gate. Production has **not started**:
0/96 groups, 0/384 sampled attempts. The deterministic fit is neither a sampled
writing success nor proof that every production trajectory will fit.

The immutable `gemma-full48-controlled-fit-v5` identity was
`83c764532db5e3d4dc953f9de4b54d459f618b7dea7b7ab149353878de17e9ed`
from source commit `6c66ae8`. It retained the v4 recipe: pinned fresh Gemma base,
source-pinned TRL/Liger streaming implementation, BF16 base, FP32 rank-8/alpha-16
all-linear LoRA, public DAPO, group four, microbatch one, accumulation four,
nonreentrant activation checkpointing, SDPA, and the 8192 active / 24576 total
per-ledger budget. The only v5 change was to the evidence observer's denominator
assertion; v4's failed evidence was preserved. The environment reported Torch
2.14.0, Transformers 5.17.0, TRL 1.14.0.dev0 and Liger 0.8.3. Native BF16
numerical parity is not claimed.

All three admissions (parent, generation worker, training worker) saw zero NVML
consumers and 24085 MiB free; the allocator was configured for expandable segments
before Torch import. The post-run device reported 17 MiB used / 24085 MiB free.

| Stage | Wall time | Torch allocated peak | Torch reserved peak | Process RSS peak |
| --- | ---: | ---: | ---: | ---: |
| Native generation | 17.238 s | 14.017 GiB | 14.352 GiB | 10.575 GiB |
| Controlled training | 219.636 s | 21.096 GiB | 21.414 GiB | 10.718 GiB |
| Entire fit | 239.544 s | — | — | — |

Each loss observation reported exactly 24576 total tokens, 8192 active tokens,
and a 32768 group-active-token DAPO denominator. Each observation's masked
log-prob gradient check passed. The four observations covered all four distinct
slots; 2208 adapter-gradient hooks ran with finite values. `training.json` records
`global_step: 1`, changed adapter tensors, optimizer counters `{1.0}`, a complete
hash-verified `checkpoint-1` with adapter/optimizer/scheduler/RNG state, and exact
resident-versus-saved equality for all 1050 adapter tensors. These are update and
recovery-artifact checks, not prose or continuity evaluations.

The fit does not cover all production action/observation mask arrangements or
sampled trajectories with up to 16384 action tokens. Production still requires a
fresh-base identity, intact original tasks, empty GPU process inventory with at
least 24000 MiB free before each train/resume model load, explicit failure on token
overflow, and complete optimizer-boundary checkpoints. Do not extrapolate the
219.636-second controlled update to total production time: native multi-decision
sampling, tool observations, reward execution and per-update checkpoints are absent
from this fit. Stop after update 48 and explicitly resume through update 96.

Raw immutable evidence:
`/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v5/context-24576-v1/`
(`result.json`, stage results, `training.json`, `loss-observations.json`,
`trainer/checkpoint-1/`, exact token ledgers, admission inventories) and adjacent
`gpu-fit-v5/context-24576-runtime.log`.
