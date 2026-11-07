# Gemma probe: baseline ran; training ran out of memory

The frozen BF16 probe stopped before its first optimizer update on the RTX 3090.
It produced a complete development baseline and a non-tied training reward group,
but no checkpoint or trained adapter. Resume and adapter evaluation therefore did
not run. This is a memory-fit failure of this configuration, not evidence that
Gemma cannot learn the tasks or that Qwen requires SFT.

The separately prepared [microbatch follow-up](gemma-microbatch-result.md) subsequently
passed training and resume. This report preserves the original full-batch failure.

## Measured outcomes

Executed on 2026-09-22 from source commit `c47a18c`, using the
[probe protocol](https://github.com/haowjy/writing-assistant/blob/9cb9944/docs/grpo-probe.md), without changing its settings or rewards.
This run trained all four attempts together with accumulation 1; the current guide
instead describes the later microbatch-1, accumulation-4 profile.
The Qwen server exited before execution; no desktop process was stopped. GPU
admission passed with about 22.1 GiB free.

| Stage | Result | Wall time | Torch allocated peak | Torch reserved peak | Worker RSS peak | Run disk growth |
|---|---|---:|---:|---:|---:|---:|
| Base development evaluation | All 12 slots recorded | 577.41 s | 9.807 GiB | 10.244 GiB | 9.867 GiB | 2.410 MiB |
| Training through step 1 | CUDA OOM before update | 303.24 s | 17.175 GiB | 19.318 GiB | 10.686 GiB | 1.494 MiB |
| Resume / adapter evaluation | Not run: no checkpoint | — | — | — | — | — |

Aggregate GPU-stage wall time was **880.65 seconds (14.68 minutes)** of the
3,600-second limit. GPU peaks are PyTorch allocator measurements, not full-device
sampling. Disk growth uses the parent's whole-run snapshots; console logs outside
the run directory are not included. The worker exited and released its allocation;
the subsequent device check showed 22,679 MiB free and 0% utilization.

### Development baseline

Six attempts completed, and six stopped at the generation limit. All rewards were
available under the frozen mechanical scorer; failed executions scored zero.
The mean over all 12 planned attempts was **0.1875**. No semantic or literary
judgments were made, and there is no trained-adapter comparison.

| Case | Seed 104729 | Seed 130363 |
|---|---:|---:|
| F2-01 revision | 0, completed | 0, generation limit |
| F2-03 revision | 0, generation limit | 0, completed |
| F4-01 wiki | 0, generation limit | 0, generation limit |
| F4-03 wiki | 0, generation limit | 0, generation limit |
| F5-01 retrieval/use | 0.75, completed | 1, completed |
| F5-03 retrieval/use | 0, completed | 0.5, completed |

“Completed” describes execution, not task success. The saved checks and files explain
zero scores on completed attempts; the table does not infer semantic failure from
literal checks.

### Training failure

The first task, `grpo-probe-v1-wave1-train-023`, produced four completed attempts
with rewards **[1, 0, 1, 1]**. There was a usable within-group difference; the
all-tied/unavailable guards did not cause this stop. Sampled action counts were
1,260 / 1,301 / 1,390 / 936 tokens. The largest prompt plus completion, including
external observations, was 2,247 tokens, below the 4,096 context limit.

The traceback enters TRL's `_compute_loss` and `_get_per_token_logps_and_entropies`,
then Accelerate's `convert_to_fp32`, which fails at `tensor.float()`. CUDA requested
**6.04 GiB** when **4.32 GiB** was free. This occurred during the loss forward pass,
before backward or an optimizer step. The observed bottleneck is output conversion
in the training path, not loading the base model or sampling the four attempts.
The trace identifies the operation; no allocation-level profiler was run.

There is no successful invocation marker, checkpoint, or inference export. The
runner recorded `environment-oom` and charged the elapsed time. No retries,
resampling, smaller profile, quantization, installation, download, or paid call
followed. Remaining allowance does not authorize retrying this failed phase.

## Evidence and next decision

Local run: `runs/grpo-gemma-probe-v1` in the main checkout. Its binding is
`25ab9225ee9acb18c9f37be68fe9e45c9a0a0d42f1433e30fbce43cd61cf207c`.
Key evidence:

- `resource-ledger.json`: parent/worker outcomes and resource measurements.
- `base-eval/scores.json`: all task/seed slots, rewards, result hashes and paths.
- `training/groups/step-000000-07b63be19487444d8e1c82a784dde514/`:
  sampled native tokens, masks, traces, files, rewards and group statistics.
- `training/invocations/4ceacdb4bf284267abcb03fd6012446c/stopped.json`: exact OOM.
- Meridian work item `grpo-gemma-probe`, `gpu-base-eval.log` and `gpu-train.log`:
  console output and full traceback.

The missing optimizer/checkpoint proof was subsequently obtained in the separately
prepared [microbatch run](gemma-microbatch-result.md), using public TRL controls.
This failed run stays frozen. The [root TODO](../../TODO.md) tracks work beyond that
short-context gate. This memory failure did not establish an SFT need or a reason
to redesign the reward.
