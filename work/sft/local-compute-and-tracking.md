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
estimate inappropriate. Nevertheless, fitting an inference cache alone does not
establish that prompt processing or a training step fits: model weights, activations,
attention implementation, loss calculation, optimizer state, and other resident models
also matter. No 128K inference or training memory claim has been verified here.

Use the current 2K SFT cap for the initial bounded training measurement. Probe 4K,
then 8K only as memory and speed permit; these are proposed targets, not guarantees.
For each configuration record allocated/reserved peak VRAM, complete step time,
processed and generated tokens, attention backend, quantization, gradient accumulation,
and checkpoint behavior. Test optimizer steps when measuring training; inference-only
measurements miss training memory. Data acceptance and the scoped execution boundary
remain prerequisites to the actual experiment.

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
