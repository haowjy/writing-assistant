# Intact full48 preparation

**This prepares data and mechanical rewards; it does not launch training.**
The 48 original wave1 training tasks remain unchanged. The full-round runner,
larger runtime budgets and native training-memory fit are not yet verified.
See [current readiness](../work/research-plan/dapo-readiness.md).

## Inspect and validate

Run from this checkout. Set `PYTHON` to the existing environment's Python executable
and `RELEASE` to the original `data/processed/gen-wave1-train` directory, which may
live in the main checkout rather than a worktree. Validation requires a new evidence directory;
it preserves every fixture result and refuses to overwrite previous evidence.

```bash
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src
"$PYTHON" scripts/prepare_grpo_full48.py inspect "$RELEASE"
"$PYTHON" scripts/prepare_grpo_full48.py validate "$RELEASE" \
  --evidence /absolute/new/full48-fixtures
```

Optional cached-tokenizer measurement takes a local snapshot directory, never
model weights:

```bash
"$PYTHON" scripts/prepare_grpo_full48.py measure "$RELEASE" \
  --evidence /absolute/new/full48-fixtures --tokenizer "$TOKENIZER_SNAPSHOT"
```

The scripted paths use repetitive mechanical text, fixed short thinking, short
intermediate replies and final-pass delivery. Their token counts are **not**
sampled successes, realistic prose-length forecasts, upper bounds, or GPU fit
proofs. They do not select a training context limit.

## Bound reward contract

`data/grpo-full48-v1/contract.json` binds the original release files, task hashes,
catalog/manifest, frozen local exclusions and explicit mechanical requirements.
`load_full48_release()` rejects altered tasks or release bytes and returns the
original tasks, production admission metadata and reward specification.

```python
from pathlib import Path
from writing_agent.grpo_full48 import load_full48_release, mechanical_full48_reward

release = load_full48_release(Path("/path/to/gen-wave1-train"))
tasks = release["tasks"]
admission = release["admission"]
reward_spec = release["reward_spec"]
reward_callback = mechanical_full48_reward
```

These are inputs to the [trainer interface](grpo.md), not an executable full48
recipe: its existing admission limits still reject the original task envelopes.
Do not work around that rejection by shortening tasks or reusing the probe scorer.

Reward is zero unless execution and the complete declared turn sequence are
supported by saved evidence, and every required final artifact is delivered in its
proper channel at the original lower word bound. File delivery also requires a
successful write/patch, replay agreement, and change from the authoritative initial
text. Existing nonempty drafts and verbal claims are not writes. Successful tool
paths are interpreted consistently with the workspace's relative-path rules.

After delivery, the reward is the mean of the original mechanical checks and the
applicable additional read, reciprocal-link and retrieval/quote checks. Upper word
limits, scope violations, content errors and failed retrieval can retain partial signal. This asymmetric lower-bound
delivery gate is an explicit reward-design choice, not a tuned optimum. The spec
binds the scorer and shared scoring implementation; source changes require a fresh
experiment identity.

All semantic rubrics remain **unjudged**. Intermediate replies can prove that the
sequence occurred, not that clarification was faithful or earlier decisions were
preserved. Protected sentences must survive verbatim; the originals do not
unambiguously require their final position, so the scorer adds no suffix gate.
Fixtures deliberately expose these limitations. Neither passing fixtures nor a
higher mechanical scalar establishes better writing or project memory.
