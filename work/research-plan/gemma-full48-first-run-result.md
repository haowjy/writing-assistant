# First intact full48 run — stopped at sampled group 15

**The original production run stopped after 14 verified optimizer boundaries and
cannot resume under its no-resampling contract.** It was prepared at source commit
`cfd6e28` from the pinned fresh base and the unchanged 48-task release. Production
identity: `7d54045a78d195d5e329f2fac685b7394abab6cde096b86e8692be7ace7c0bc3`.
The trainer's own experiment identity was
`a26038d04adcb13f6a52318150ce88a5b4eb23758d05ae3379d379aeac3c44b4`.
Keep these artifacts intact. This run is not a completed 96-update result, and its
14 updates must not be pooled with any restarted run.

Headless GPU admission passed with zero listed consumers and 24085 MiB free. The
supervised first invocation reached 14 complete, hash-sealed checkpoint boundaries:
56 attempts across 14 scored groups, including 9 tied and 5 relative-signal groups.
Group index 14 (`wave1-train-015`) sampled four more attempts before TRL raised
`GroupPending: Unavailable reward/infrastructure: whole group pending`. This left
60 attempt result files, 15 groups started, 14 scored, and 14 completed updates.
The latest complete checkpoint is `checkpoint-14`; an uncommitted sampled group
exists after it, so normal resume is forbidden. The GPU returned to idle (17 MiB
used / 24085 MiB free) after worker exit 1. No elapsed cutoff or retry fired.

Two attempts in that uncommitted group exposed different native response forms:

- Slot 000 emitted a parsed `write_file` tool call whose **raw output ended in
  `<eos>`**, not `<|tool_response>`. The old loop executed the write, then could not
  append the tool reply to an exact sampled prefix. It marked this an infrastructure
  failure with unavailable reward.
- Slot 003 emitted ordinary assistant content followed by a `write_file` tool call
  ending in `<|tool_response>`. The old native-suffix guard rejected the mixed
  content/tool-call shape despite having the correct raw stop boundary. It marked
  this an infrastructure failure with unavailable reward.

Slots 001 and 002 had completed attempt evidence and mechanical reward 0. Do not
reinterpret/reseal any of these old results or use their sampled outputs as
successful trajectories. The new implementation distinguishes a sampled tool call
with the wrong stop boundary (candidate invalid, scoreable as failure) from genuine
infrastructure loss, and derives only the external tool suffix after an already
sampled valid boundary even when the action includes content. The corrected behavior
requires a **new identity and fresh-base run**, not recovery from checkpoint 14.

Raw local evidence:
`/home/jimyao/.meridian/context/orange-juniper-leaf/work/dapo-full-rounds/full48-production-v1/`
(`coverage.json`, `trainer/groups/step-000014-*/`, `trainer/checkpoint-14/`,
`supervision/a68f89198a9b443b9e5d2f1d67342b2b/worker.log`, saved NVML inventory).
At the stop, two retained trainer checkpoints were approximately 205 MiB each;
the entire run directory was 479 MiB, with 270 GiB free on its filesystem.
Mechanical rewards and optimizer progress do not establish prose or continuity
improvement; no semantic or literary evaluation was performed.
