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

## Which model can be trained here at all

The second question is scale. `scripts/probe_context_budget.py --ladder` computes the
QLoRA memory floor for every candidate: quantised weights plus one checkpointed activation
boundary per layer, which is what gradient checkpointing retains and what does not shrink
with batch size. The remaining headroom has to cover the recomputed attention block.

| Model | Parameters | 8K | 32K | 64K |
|---|---:|---|---|---|
| Gemma-4-E2B-IT | 5.1B | fits (18.7 GiB free) | fits (16.2) | **fits (12.9)** |
| Gemma-4-E4B-IT | 8.0B | fits (16.4) | fits (11.5) | marginal (4.9) |
| Gemma-4-12B | ~12B | marginal | exceeds | exceeds |
| Gemma-4-31B | 31B | marginal (0.8) | **exceeds** | **exceeds** |
| Qwen3.8-27B | 27B | marginal (3.7) | **exceeds** | **exceeds** |

E2B and E4B figures use the cached checkpoints. 12B, 31B and 27B are estimates from
published configurations and are labelled as such in the artifact; the 12B weights are not
cached. The 64K column for E4B and the 8K column for the two large models show a floor
that fits while leaving too little room for the attention transient and runtime overhead,
so they should be read as unusable rather than as candidates.

These estimates identify pressure in the modeled configuration, not a proof that a
27B model cannot be trained on a 3090. Activation offload and different training kernels
can change the resident-memory requirement. Unsloth now documents Qwen3.8 QLoRA with
24GB; its example uses a 2,048-token sequence cap. That warrants a short-context test,
not a claim that long-context GRPO fits. Quantizing weights alone does not remove
activation, rollout-cache, or training overhead.

The same arithmetic carries an architectural argument in the other direction. Gemma-4's
sliding-window design makes long context cheap for a small model: 28 of 35 E2B layers cap
their cache at 512 tokens, so a 5B model gets a 128K window at 1.9 GB and a 64K training
step. That is why E2B is a legitimate long-form feasibility target rather than a toy.

The active plan uses E2B for engineering verification and Qwen3.8-27B as the training
target. An E4B intermediate run is optional, not a prerequisite. Measure Qwen with the
chosen stack before deciding that rental is necessary. Short-context local feasibility
would not settle the desired 256K context budget.

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

## Unsloth is a training-stack candidate

The [Qwen3.8 training guide](https://unsloth.ai/docs/models/qwen3.8/train) documents
4-bit QLoRA with 24GB, embedding offload, and gradient checkpointing. It also documents
an RL path with `fast_inference=False`, rather than the fast vLLM path. Its advertised
speed/memory savings are vendor measurements, not results from this checkout. Unsloth
supports reward training; adopting it would not imply adding SFT.

Before adopting it, use an isolated environment and a bounded probe of loading, one
update, save/resume, and native tool-message rendering. Measure peak GPU/RAM/disk use
and end-to-end rollout plus update time at short lengths before increasing context.
Do not overwrite the existing locked environment or download weights without a scoped
storage and execution budget. The official [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B)
states a native 262,144-token window; this is not evidence that 256K GRPO fits on 24GB.

Use the official model or a traceable training-format quantization as the initial base.
GGUF is primarily an inference/export format; the standard Unsloth QLoRA path loads
Transformers weights, not the linked community GGUF. Community merges remain optional
comparison models, separate from the training-framework decision. Save adapters and
bounded checkpoints rather than every merged full-weight copy when managing storage.

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
