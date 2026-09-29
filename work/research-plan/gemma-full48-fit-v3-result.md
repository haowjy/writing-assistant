# Gemma full48 headless controlled-fit result

The single `gemma-full48-controlled-fit-v3` attempt **failed its training stage with
CUDA OOM**. Headless ownership and the separate native 32767+1 generation stage
passed. No optimizer update or complete checkpoint was produced, so the fit gate
remains closed and full48 production training did not start.

## Frozen attempt

The attempt used profile identity
`da8fcabd931d56136c68c2eab45b158e9f9e3d2f4427e378b053d422cf263ebc`.
It preserved the v2 model, 32768-token ledgers, BF16 base, FP32 rank-8 all-linear
LoRA, public streaming DAPO path, microbatch one, accumulation four, nonreentrant
checkpointing, SDPA, and source pins. V3 changed only resource admission and allocator
behavior: no listed GPU consumers, at least 24000 MiB free, and
`expandable_segments:True` before Torch import. There was no retry or profile change.

Ownership admission passed with no graphics or compute consumers, 17 MiB GPU memory
used, and 24085 MiB free. A headless agent-browser Chrome process used SwiftShader
software rendering and did not appear in NVML.

## Measured stages

| Stage | Result | Time | Peak Torch allocated | Peak Torch reserved | Peak process RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Native 32767+1 generation | passed | 25.161 s | 16.767 GiB | 17.342 GiB | 10.575 GiB |
| Controlled training | CUDA OOM | 32.315 s | 22.785 GiB | 23.043 GiB | 10.718 GiB |

The first training microbatch reached backward activation-checkpoint recomputation.
The MLP `down_proj` LoRA path then requested another 768 MiB while evaluating its
FP32 low-rank input projection. At failure, PyTorch reported 22.79 GiB allocated,
83.78 MiB reserved but unallocated, 23.18 GiB total process use, and 336.62 MiB free.
The remaining microbatches, optimizer step, and full-checkpoint checks did not run.

Compared with v2, headless execution and expandable segments reduced reserved-but-
unallocated memory from 863.97 MiB to 83.78 MiB and allowed allocation to rise from
21.38 GiB to 22.79 GiB. Training still lacked at least 431 MiB for the observed
allocation, with no evidence that satisfying that allocation would complete backward.
This rules out desktop memory and allocator fragmentation as sufficient explanations.

## Consequence

The exact 32768-token, FP32 all-linear LoRA recipe does not fit this 24 GiB RTX 3090
under the qualified software path, even headless. The production schedule remains at
**0 of 96 groups and 0 of 384 attempts**.

Preserving this exact 32768-token recipe requires a larger GPU. The subsequently
approved v4 contract instead reduces complete trajectory/context to 24576 tokens while
retaining FP32 all-linear LoRA and the 8192/16384 action limits. All original tasks and
output requirements remain unchanged, but sampled attempts have less observation/tool
headroom. V4 is a new qualification contract, not a reinterpretation of this result.

Raw evidence is local at:

- `/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v3/headless-expandable-v1/result.json`
- `/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v3/headless-expandable-v1/{generation-result.json,training-result.json}`
- `/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v3/headless-expandable-runtime.log`
