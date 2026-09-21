# Local compute and experiment tracking

Decision, 2026-09-15: use the existing RTX 3090 first; defer GPU rental until measured
memory or throughput limits justify it. Keep the workflow driven by Python scripts
over the terminal so the same experiment can move to a rented machine later.

A read-only GPU query returned NVIDIA GeForce RTX 3090, 24,576 MiB total and 927 MiB
used. This is an idle-memory observation, not a training capacity measurement.

## Establish the usable budget

The cached Gemma E2B configuration supports 131,072 positions, matching the
[official 128K context specification](https://ai.google.dev/gemma/docs/core/model_card_4).
Its sliding attention and shared KV states make a generic all-layers full-cache
estimate inappropriate, and the measured answer is far below that estimate. On
2026-09-21 `scripts/probe_context_budget.py` computed the cache from the checkpoint
config and measured the attention step cost on the real layer shapes:

| Context | Inference KV (bf16) | One full-attention layer, fwd+bwd | Projected optimizer step |
|---:|---:|---:|---:|
| 8K | 132 MB | 0.26 s | 0.2 min |
| 32K | 484 MB | 3.35 s | 3.1 min |
| 64K | 954 MB | 13.61 s | 12.7 min |
| 128K | 1.9 GB | not measured | not measured |

Projected steps assume gradient accumulation 8 and multiply the measured per-layer
figure by the seven full-attention layers. They are lower bounds: the sliding layers,
MLP, embeddings, optimizer step and activation-checkpoint recomputation are excluded.
An optimizer step is 8 sequences, so a 20-step run at 64K is roughly 4.2 hours, and
longer in practice.

Two consequences. **128K inference is not the problem** - at 1.9 GB the cache is a
rounding error against 24 GB, and an earlier 8192/16384 harness limit was an arbitrary
setting rather than a hardware limit. **FlashAttention is not available** for this
model: the full-attention layers use `global_head_dim=512`, above the 256 limit for the
FA kernels reachable through SDPA on this stack, so memory-efficient SDPA is the only
O(n) option. That makes cost time, not memory.

The remaining prerequisite is data, not hardware. `max_length` is an acceptance bound
that rejects overflow and never pads, so widening it costs nothing while trajectories
are short. A 64K window only means something once long trajectories exist to fill it.
Use the same probe before claiming any length, and record peak VRAM, step time,
attention backend, quantization and gradient accumulation together.

For RL, separately measure rollout generation, workspace execution, feedback generation,
reward evaluation, and policy updates. Avoid simultaneous policy/judge/simulator
residency until measured. Use bounded connected task sessions and test compaction to
support longer projects without requiring a 128K active training window. See
[multi-turn RL](multi-turn-rl.md).

Rent only after identifying a concrete bottleneck: sequences that cannot fit, poor
rollout throughput, or a reward/simulator model that makes serial execution impractical.
Move the same locked code, data/task manifests, checkpoint identity, and logging schema;
compare cost per completed useful experiment, not GPU hourly rate alone. No cloud
resource has been requested or provisioned.

## W&B recommendation, not yet integrated

Use Weights & Biases as an optional view over the existing local experiment artifacts.
Current `training.py` sets `report_to="none"`; no W&B logging or uploads were enabled.
The local records remain usable without W&B.

[W&B Tables](https://docs.wandb.ai/models/tables) support text alongside numerical and
structured fields. Proposed per-attempt rows contain session/stage IDs, task families,
source/branch IDs, candidate harness/provider/model, checkpoint/step, prose artifact,
mechanical results, semantic scores, written critique, evidence spans, uncertainty,
and judge/rubric versions. Link each critique to the exact output it evaluated.

Use [W&B Artifacts](https://docs.wandb.ai/models/artifacts) for versioned selected
transcripts, Markdown manuscripts/KB snapshots, task manifests, reports, and adapters.
They can preserve written analyses as files; a dashboard does not have to reduce them
to scalar scores. Store large full traces as artifacts rather than repeating them in
every metrics event. Select uploaded material explicitly; do not recursively upload
the repository, credentials, raw corpora, or unrelated user files.

Track loss and each reward component separately, plus completion, reward variation,
rollout length, token usage, memory, timing, and compaction counts. Use checkpoint and
scenario IDs to compare the same evaluation examples over time. A logging integration
must not trigger grading, generation, training, or model downloads on import.
