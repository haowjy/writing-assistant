# Fresh-restart v6 controlled GPU fit

**The single source-bound v6 fit passed on the headless RTX 3090.** It completed
native 24575+1 generation, four accumulated public-DAPO microbatches, one optimizer
update, and a sealed full checkpoint. The corrected native rollout policy and
all-checkpoint retention are bound in the source identity; this fit's controlled
ledgers do not sample those tool-protocol cases or establish writing quality. A new
pinned-base full48 production invocation was started only after fit verification.

- Source at fit/launch: `7828a8f` on `feat/dapo-full-rounds`.
- Fit profile/identity: `gemma-full48-controlled-fit-v6`,
  `68668fa6199a5082c1dbf59ca2e900505301e7e4a2b35d4457e44f7f4bfb8714`.
- GPU admission: zero listed consumers, 24085 MiB free before the parent and both
  worker stages; `expandable_segments:True` was set before Torch import.
- Runtime: exact source-pinned TRL `1.14.0.dev0` / Liger `0.8.3`; BF16 base,
  ordinary FP32 all-linear rank-8/alpha-16 LoRA, SDPA and nonreentrant checkpointing.
  The maintained FP32 streaming-softcap variant is not native-BF16 parity.

| Stage | Wall time | Peak Torch allocated | Peak Torch reserved | Worker peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Native 24575+1 generation | 43.939 s | 14.017 GiB | 14.352 GiB | 8.585 GiB |
| Four-microbatch DAPO training | 228.159 s | 21.096 GiB | 21.414 GiB | 10.283 GiB |
| Whole fit | 275.522 s | — | — | — |

Each of four distinct slots consumed exactly 24576 tokens (8192 active) with the
observed group-active-token denominator 32768; every masked log-prob gradient was
zero. Adapter-gradient hooks recorded 2208 finite checks. The adapter changed at
optimizer step 1; optimizer counters were `{1.0}`; all 1050 resident adapter tensors
matched the saved checkpoint exactly. The complete checkpoint's adapter, optimizer,
scheduler, RNG, tokenizer and trainer state were independently hash-verified.

The resulting production preparation was separately preflighted as
`full48-production-v2/`, identity
`9c5f32959ecf5bccceead3783e67db8144defee6cba3b8c9aa748acb1ed63e80`:
96 scheduled groups, 384 attempts, all 96 optimizer-boundary checkpoints retained,
stop at 48, then an explicit resume to 96. Initial production admission again found
zero GPU consumers and 24085 MiB free. The first invocation was launched; no
production completion or prose-quality improvement is claimed here. The earlier
stopped run remains terminal and separate; see the
[first-run result](gemma-full48-first-run-result.md).

Raw fit evidence:
`/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/gpu-fit-v6/protocol-retention-24576-v1/`
(`result.json`, `training.json`, `loss-observations.json`, complete checkpoint and
NVML inventories). The parent log is adjacent at
`gpu-fit-v6/protocol-retention-24576-runtime.log`.
