# Supervised fine-tuning

The editable research entry point is [scripts/train_sft.py](../scripts/train_sft.py).
It defaults to reporting readiness; it does not load weights or train:

```bash
uv sync --all-extras
.venv/bin/python -m scripts.train_sft
```

Run the tokenizer-only format audit from Python:

```python
from scripts.train_sft import audit_fixtures

audit_fixtures()
```

The five pending fixtures test prose, tools, revisions, planning, and KB use. They
are not a production training dataset. Read the
[preparation status](../work/sft/plan.md) before selecting a training run.

## Prepare data

Place version-1 trajectory records at `data/training/sft-v1.jsonl`, or change `SOURCE`
in the research script. The existing [trajectory validator](../src/writing_agent/data.py)
requires provenance, messages, tool schemas, initial/expected files, split, and review
status. SFT additionally requires a nonempty `source_groups` list for each accepted
record. Acceptance denotes an explicit data-quality decision; it need not be human grading.

Only accepted train/validation records are prepared. Author/work grouping and explicit
source groups must not cross splits. The script excludes source groups from the authored
evaluation scenarios. This is an exact lineage-ID check, not a semantic contamination
detector; curators must record shared origins and derivatives correctly.

```python
from scripts.train_sft import prepare

manifest = prepare()
```

Preparation writes trajectories, token IDs, labels, tokenizer files, lengths, and
hashes into a new directory. Inspect `encoded.json`: `supervised_text` decodes the
loss-bearing tokens and `rendered` shows the full conversation. Annotation must leave
the native template output unchanged. No truncation, packing, reasoning-trace import,
or silent fallback to full-sequence loss is allowed.

## Train and resume

After choosing and authorizing a bounded run:

```python
from scripts.train_sft import train

plan = train()                # inspect prepared experiment
# train(execute=True)         # explicitly starts QLoRA training
```

`train(execute=True, resume_from_checkpoint=...)` requires a trainer checkpoint inside
the same output directory and matching preparation/settings. Fresh training never
overwrites an existing output directory. Model loading is local-only.

The trainer saves PEFT adapters, tokenizer files, trainer checkpoints, runtime versions,
trainable parameter count, and final metrics including peak allocated GPU memory.
No checkpoint evaluation or paid grader runs automatically.

For checkpoint evaluation, use the existing
[local inference adapter interface](local-inference.md): select the same base checkpoint,
set its `adapter` to a saved `checkpoint-N` or final `adapter` directory, and evaluate
an explicitly selected subset. Directory contents determine adapter identity. Keep
the baseline's Gemma harness, prompts, and generation settings fixed. Full training
and GPU save/resume/inference have not yet been runtime-verified.
