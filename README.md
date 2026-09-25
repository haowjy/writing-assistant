# Creative Writing Agent

The public artifact is the generated task list in [data/tasks](data/tasks), with links
to the source texts. This repository does not vendor model weights or book corpora.
`scripts/fetch_sources.py` downloads a cited source when you need it.
`scripts/organize_sources.py` turns those downloads into the catalog.
`src/writing_agent/text_clean.py` strips Gutenberg wrappers, title pages, and web
residue. Supervised training is `scripts/train_sft.py`; the reward function is
`src/writing_agent/reward.py`. There is no GRPO trainer yet.

This research harness also runs tool-use evaluations and prepares conversational
training data for [creative-writing agents](wiki/project-goals.md).

**What to do next: [TODO.md](TODO.md)** — the short, active checklist.

## Quick start

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are recommended. The core has no third-party runtime dependencies.

```bash
uv sync --locked
uv run cwa eval --config configs/smoke.toml
uv run cwa validate-data data/fixtures/trajectories.jsonl
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format --check .
```

Without installing dependencies, use `PYTHONPATH=src python3 -m writing_agent eval --config configs/smoke.toml`.

The smoke run replays five scripted examples covering retrieval, local revision,
stale decisions, no-tool responses, and draft/canon separation. It checks the harness;
it does not evaluate a model.
The CLI exits nonzero if any task fails.

The checkpointed task-graph runtime is opt-in and does not replace this CLI path.
See [transactional writer stepping](docs/task-graph-writer.md) and the
[deterministic scripted-author slice](docs/task-graph-scripted.md) for its direct
Python API, offline replay, and current limits.

## Layout

```text
src/writing_agent/       Tool workspace, inference clients, agent loop, scoring, data, CLI
configs/                Committed experiment settings; paths relative to config files
data/fixtures/          Tiny synthetic development tasks and trajectory format example
tests/                  Behavioral tests and an end-to-end offline evaluation
docs/                   Usage documentation and data contract
wiki/                   Research goals, shared concepts, and source material
work/research-plan/     Proposed experiments and execution checklists
runs/                   Ignored run manifests, traces, scores, and private task workspaces
```

## Evaluate a model

Start a separately managed chat-completions server, such as vLLM, configured with
the chosen model's chat template and tool-call parser. Set `model` and `base_url`
in `configs/local_server.toml`, then run:

```bash
uv run cwa eval --config configs/local_server.toml
```

The default `127.0.0.1` address connects to a model server on the machine running
the evaluation command, including when that machine is a remote GPU host.

If the server requires authentication, supply `CWA_API_KEY` through the environment.
Token usage comes from the server; an empty usage
object means unavailable, not zero tokens. Temperature and seed are recorded but
do not guarantee deterministic GPU inference. Record the exact served checkpoint,
adapter, server version, and launch settings alongside each real experiment.

Each run records a config, task snapshot and hash, package/source fingerprint,
per-task JSONL traces, final workspace, results, and summary. Failed generations
remain in the denominator.

## Data preparation

See [the trajectory contract](docs/data-contract.md). The example is
`pending` review. The exporter includes only `accepted` records in the `train` split:

```bash
uv run cwa validate-data path/to/trajectories.jsonl
uv run cwa export-sft path/to/trajectories.jsonl data/processed/train.jsonl
```

Export produces TRL-style `messages` and `tools` columns and refuses to overwrite
an existing file.

## Limits

The filesystem tools block absolute paths, traversal, symlinks, oversized files,
and ambiguous patches. They do not provide OS isolation or protect against
concurrent untrusted filesystem writers.

Current scorers check literal output constraints, file outcomes, tool errors, and
edit scope. Literary quality is unscored. There is no automatic canon
commit policy: the prompt teaches the distinction, and tests check file outcomes.

GPU training and a held-out benchmark are not implemented.
See the [research wiki](wiki/index.md) for project concepts and the
[research work plan](work/research-plan/index.md) for proposed experiments.
