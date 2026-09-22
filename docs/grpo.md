# GRPO training and checkpoints

The harness now connects task attempts, file tools, rewards, and TRL optimizer updates.
A tiny CPU model has passed adapter save/reload and interrupted-training resume checks.
**Gemma GPU training and writing improvement are not yet verified.** The
[root TODO](../TODO.md) sets the next execution step; no SFT demonstrations are required.

## Inspect first

For the frozen three-task experiment, use the [prepared Gemma probe](grpo-probe.md).
The generic entrypoint below remains unconfigured for other experiments.

Run from the repository root using the existing training environment. These commands
do not install dependencies; execution requires the pinned optional training packages,
including TRL 1.13.0.

```bash
PYTHONPATH=src uv run --no-sync python scripts/train_grpo.py
```

The default reports `No training tasks selected`. It does not load weights or create a
run. In [train_grpo.py](../scripts/train_grpo.py), explicitly select `RELEASE`, `TASK_IDS`,
`OUTPUT`, `SETTINGS`, `EXCLUDED_SOURCE_GROUPS`, and the reward definition before execution.
The entrypoint verifies the compiled release's catalog and selected task hashes, recomputes
connected source groups, and rejects held-out roles, exclusions, and cross-role near
duplicates. This checks consistency with the supplied inventory, not the truth of its
provenance or the absence of unknown contamination. Substantive task review is still needed.

Python callers must supply explicit `admission`: either the production catalog, release
manifest, hashes and exclusions, or a labeled `engineered-fixture` mode. See the two scripts
for complete examples. Fixture admission is never a substitute for production data review.

The built-in loader uses the cached pinned Gemma checkpoint, BF16 LoRA, one GPU, and
local files only. It does not download models, quantize weights, install Unsloth, call a
paid judge, or run evaluations. Defaults bound each attempt to 4096 total context tokens,
1024 generated tokens, and 256 tokens per decision; group size is two and the experiment
has a total budget of two optimizer steps. Task-specific turn/tool/storage limits also
apply. Context overflow fails explicitly rather than truncating history.

Only after approving the task selection, time/resource limits, and execution scope:

```bash
PYTHONPATH=src uv run --no-sync python scripts/train_grpo.py --execute
# Same settings and output directory; checkpoint-1 must be complete and latest.
PYTHONPATH=src uv run --no-sync python scripts/train_grpo.py \
  --execute --resume runs/grpo-probe/checkpoint-1
```

This generic entrypoint does not enforce a wall-clock watchdog or measure GPU/RAM peaks.
The [prepared probe runner](grpo-probe.md) adds supervised stages, an aggregate time limit,
resource records, and matched development evaluation.

## Rewards and unavailable groups

GRPO samples several attempts at the same task and trains toward its better attempts.
Each starts from the same instructions and files in a fresh bounded workspace; sampling
seeds differ. Candidate tools remain the five file operations, never shell execution.

The example callback measures **mechanical compliance only**, not prose quality. It
refuses semantic checks rather than inventing their scores. A custom callback can use
`reward.rollout_reward` or `reward.session_reward` and must return a `Reward`. It must be
a plain function without captured closure state; its source and declared configuration are
frozen in the manifest. Module globals, function defaults, external judge versions, private
settings, and service behavior must match that declaration. The caller owns this guarantee;
source hashing alone cannot prove an external grader stayed unchanged.

Unavailable judgments, callback errors, infrastructure failures—including observations
that cannot fit the context—or attempts with no sampled actions stop the whole group
before an update. They are not zero rewards. All
attempts and failure evidence remain on disk. Exact ties also stop the invocation before
optimizer momentum can move weights without new relative reward information. There is
no automatic resampling until a favorable group appears.

The saved group report distinguishes the repository helper's population-standard-deviation
statistics from TRL's sample-standard-deviation-plus-epsilon advantages. Training uses
TRL's GRPO loss, one update per fresh group, no KL penalty (`beta=0`), no weight decay,
and no dropout. This is not a capability-preservation guarantee.

## Exact tokens during tool use

Gemma's ordinary chat rendering sorts tool arguments and can remove earlier thinking
after a new user turn. Rebuilding parsed transcripts would change the actions or the
context used to sample them.

The training backend renders the initial prompt once, then appends exact sampled token
IDs and template-derived external suffixes. User follow-ups and tool observations remain
visible but are masked out of the loss. Sampled reasoning, tool calls, prose, and stopping
tokens are candidate actions. Every generation input must match its saved training prefix.
Unsupported framing stops the group as an infrastructure failure.

This append-only context policy deliberately differs from ordinary evaluation rendering.
It is versioned in the manifest. Compare a trained adapter with its base under matched
evaluation conditions; an evaluation run is not a replay of training. Use saved generation
boundaries to inspect termination: TRL's generic clipped-completion metric does not reliably
recognize Gemma's multiple native stop tokens.

## Adapter versus resumable checkpoint

An **adapter** contains learned changes and requires the same base checkpoint for
inference. A **trainer checkpoint** also contains optimizer, scheduler, random-number,
and trainer state. Loading an adapter alone is not resuming training.

A run records:

```text
experiment.json                       Frozen configuration and identity
 groups/step-000000-<id>/              Every attempt, tokens/masks, trace and workspace
 checkpoint-1/                        Adapter plus optimizer/scheduler/RNG/trainer state
   complete.json                      Hashes marking a completed checkpoint
 invocations/<id>/started.json        Resume origin and starting step
 invocations/<id>/adapter/            Separate inference export after a successful invocation
 invocations/<id>/complete.json        Results, paths and trainer metrics
 invocations/<id>/stopped.json         Failure record when an invocation stops
 quarantine/checkpoint-N-<id>/         Preserved partial saves displaced during recovery
```

The identity binds the base revision, ordered tasks including private checks and source
groups and source admission, reward specification, system prompt, tokenizer/templates,
model/LoRA/generation settings, budgets, package versions, and package Python sources.
Caller-owned fresh base models also have every parameter and buffer hashed before LoRA
attachment; changing a tensor cannot resume under the same descriptive label. The built-in
loader instead trusts the immutable local Hugging Face revision. Changing experiment
inputs starts a new experiment. Resume checks precede caller model/tokenizer/RNG mutation. A new run refuses a nonempty output directory. Choose a stable run path: the current
manifest also binds the resolved output directory. The single-writer rule applies:
do not launch concurrent invocations into the same directory.

Each trainer save receives a final hash marker covering its files and existing rollout
evidence. Resume rejects missing state, altered files, changed identity, and adapter-only
exports. Treat checkpoints as trusted local artifacts: hashes detect accidental changes,
not malicious replacement of both files and their markers; optimizer/RNG loading uses
Python serialization.

The scope uses one process/device, fixed task order, and saves at optimizer boundaries.
`microbatch_size=None` trains the complete reward group together. A positive integer
that divides `group_size` instead splits that same scored group into smaller training
batches; gradient accumulation is derived as `group_size / microbatch_size`. TRL buffers
the group only until its one optimizer update, never across updates. Rewards and
advantages are computed for the full group before splitting. Changing the microbatch
invalidates checkpoint identity, just like other training settings.

`max_steps` is the total optimizer-step budget, not an extra budget granted on resume
or a count of microbatches. The Python API's `stop_after_steps` permits a deliberate early
stop without changing that total schedule.

Keep the latest two trainer checkpoints and all attempt evidence. Separate inference
exports are bounded by `max_invocations` (three by default); exceeding it refuses execution
rather than deleting evidence. Do not repeatedly export merged copies of the full base.

### Recovery

Resume only the latest complete checkpoint. After preflight, the runner moves later
incomplete saves atomically into unique `quarantine/` paths, preserving their bytes.
It refuses to branch from an older save when a later complete checkpoint exists.
A pending group is sampled again from the last checkpoint with the same step seeds and
new attempt paths—not silently regraded or replayed. If no complete checkpoint exists,
preserve the failed run and start in a new directory. Changing a rubric to resolve a
pending judgment also requires a new experiment.

For inference, load the manifest's exact base revision, then use
`PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)`. Verify the
export's `complete.json` file hashes before use. The existing
[checkpoint evaluator](local-inference.md) accepts the adapter path plus `training_run`
and `global_step` for provenance. Keep precision, prompts, tasks, and budgets matched
against the base. Use development results for checkpoint selection and reserve final
tests for the chosen adapter. Do not pass an inference export as `--resume`.

## Reproduce the CPU engineering check

This explicit command samples a newly initialized tiny model without downloading weights
or exposing a GPU. Choose an output directory that does not already exist.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  uv run --no-sync python scripts/smoke_grpo_cpu.py --execute \
  --output /tmp/grpo-cpu-check
```

The check uses a clearly labeled artificial word-index reward, not writing-quality data.
Its `smoke.json` reports a nonzero LoRA update, unchanged base weights, exact adapter
reload, preserved partial-checkpoint recovery, and exact uninterrupted-versus-resumed
adapter/optimizer/scheduler/RNG state through step 3. Checkpoint retention removes step 1;
the restored pre-update state must still match the preserved step-1 export evidence.
It also checks equal sampled histories after resume and masks the inserted user turn.
The check uses group size 4, microbatch size 1, and four accumulation steps. It verifies
three tasks in order, one sampled group per update, and 12 single-attempt loss forwards.
Against full-group training, adapter tensors agree within absolute tolerance `1e-6`;
Adam moments agree within `rtol=1e-5, atol=1e-8`, which also checks gradient scaling.
Resume within the microbatched configuration remains exactly equal, not approximate.
Native Gemma tokenizer tests separately exercise file tools and multi-turn suffixes with
scripted outputs; those tests are not evidence of Gemma optimization or GPU fit.

The next real-model check must measure runtime, peak GPU/RAM, and disk growth; verify
adapter change, save/reload/resume, and task outcomes; and report tied or unavailable
groups. Loading alone is not success. Longer contexts, adaptive authors, Qwen, and
literary gains remain later experiments. See the [training plan](../work/research-plan/training-experiments.md)
and the separate [optional SFT path](sft.md).
