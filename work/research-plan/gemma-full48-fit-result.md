# Gemma full48 controlled-fit result

The single admitted `gemma-full48-controlled-fit-v2` attempt **failed its training
stage with CUDA OOM**. The separate native 32767-token prefill plus one-token
generation stage passed. No optimizer update or complete checkpoint was produced,
so the fit gate remains closed and full48 production training did not start.

## Frozen attempt

The attempt used profile identity
`93510c26b10fc685b1813be2617d678e93e4ddcb032f3d4cb4d039cd41998f8f` and the
source-pinned environment described in [the fit protocol](https://github.com/haowjy/writing-assistant/blob/9cb9944/docs/grpo-gpu-fit.md).
It retained the fixed 32768-token ledgers, BF16 base, FP32 rank-8 LoRA adapters,
public streaming DAPO path, microbatch one, accumulation four, nonreentrant
checkpointing, SDPA, and unchanged ownership thresholds. There was no retry or
profile change.

Ownership admission passed before execution: the complete NVML inventory contained
681 MiB of approved desktop allocations, with 23203 MiB free. Every process stayed
within 256 MiB, the total stayed within 768 MiB, and free memory exceeded 22000 MiB.
Xwayland and Ghostty were admitted under the user-approved desktop policy.

## Measured stages

| Stage | Result | Time | Peak Torch allocated | Peak Torch reserved | Peak process RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Native 32767+1 generation | passed | 42.906 s | 16.767 GiB | 20.072 GiB | 8.178 GiB |
| Controlled training | CUDA OOM | 36.273 s | 21.379 GiB | 22.223 GiB | 10.618 GiB |

The first training microbatch reached a finite loss of `0.2738012969493866` over
32768 total tokens and 8192 active tokens. Its masked-gradient check passed. During
activation-checkpoint recomputation in backward, the LoRA MLP up projection requested
another 768 MiB. At failure, PyTorch reported 21.38 GiB allocated, 863.97 MiB reserved
but unallocated, 22.54 GiB total process use, and 114 MiB free. The remaining three
microbatches, optimizer step, and full-checkpoint checks did not run.

After the worker exited, a live inventory showed no training process, 899 MiB GPU
memory used, and 23204 MiB free. These measurements establish that the fixed native
generation check fits and the fixed controlled training profile does not fit this
24 GiB RTX 3090 under the admitted desktop load. They do not establish sampled-task
success, writing quality, or the fit of a modified recipe.

## Consequence

The terminal fit result is preserved. The production schedule remains at **0 of 96
groups and 0 of 384 attempts**. A separately identity-bound v3 contract later retained the same model, 32768-token
ledgers, precision, all-linear LoRA, objective, and task requirements while requiring
an empty GPU process inventory, at least 24000 MiB free, and PyTorch expandable
allocator segments before import. It also OOMed during training; see the
[v3 result](gemma-full48-fit-v3-result.md). That attempt does not alter or retry this
v2 result; neither failed profile may be represented as passing.

Raw evidence is local at:

- `/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v2/approved-desktop-v1/result.json`
- `/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v2/approved-desktop-v1/{generation-result.json,training-result.json,loss-observations.json}`
- `/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v2/approved-runtime.log`
