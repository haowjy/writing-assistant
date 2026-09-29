# Gemma full48 24K controlled-fit v4 result

The single `gemma-full48-controlled-fit-v4` attempt **failed because its evidence
observer asserted the wrong DAPO denominator**. Headless ownership and native
24575+1 generation passed. Controlled training completed its first forward loss,
but the observer raised before backward. No optimizer update or complete checkpoint
was produced, so v4 did not pass and full48 production training did not start.

## Frozen attempt

The attempt used profile identity
`08bb2ec6ff9a5c5489c9569d2c3b524f923c42f203cc06397f29e8a38f660ed8`.
It used 24576-token ledgers, an 8192-token active suffix in each of four attempts,
BF16 base weights, FP32 rank-8 all-linear LoRA, public streaming DAPO, microbatch
one, accumulation four, nonreentrant checkpointing, SDPA, the pinned runtime, an
empty NVML process inventory, and expandable allocator segments. Ownership passed
with 24085 MiB free and 17 MiB used.

## Measured stages

| Stage | Result | Time | Peak Torch allocated | Peak Torch reserved | Peak process RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Native 24575+1 generation | passed | 19.729 s | 14.017 GiB | 14.352 GiB | 10.577 GiB |
| Controlled training | observer failure after first forward | 22.175 s | 17.097 GiB | 17.711 GiB | 10.718 GiB |

The four fit ledgers each contained 8192 active action tokens. Public TRL DAPO uses
the generation group's active-token count as its denominator, excluding masked
observations: `4 × 8192 = 32768`. The v4 observer incorrectly compared that value
to the complete per-attempt trajectory length, 24576. Those values happened to be
equal under the former 32768-token profiles, so the latent observer defect was not
visible until the context cap changed.

This is not a CUDA OOM and does not show that 24K backward fits. It also is not a
passing memory qualification: backward, the remaining microbatches, optimizer step,
and checkpoint verification never ran. The attempt is terminal and will not be
rerun or reinterpreted.

## Consequence

The observer now checks the actual generation-group active-token denominator and the
corrected contract has a fresh v5 identity. That correction changes evidence
validation only, not model, task, token, LoRA, optimizer, objective, or resource
settings. A separately authorized v5 attempt is required before production can start.
Production remains at **0 of 96 groups and 0 of 384 attempts**.

Raw evidence is local at:

- `/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v4/context-24576-v1/result.json`
- `/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v4/context-24576-v1/{generation-result.json,training-result.json}`
- `/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v4/context-24576-runtime.log`
