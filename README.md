# Creative Writing Agent

Initial research harness for the [conversational authoring roadmap](creative_writing_agent_research/00_RESEARCH_ROADMAP.md).
The first milestone is reproducible tool-use evaluation and trajectory preparation.

## Remote terminal workflow

This project is operated remotely through a terminal. Run setup, data preparation,
training, and evaluation through CLI commands on the remote machine. Core workflows
must not require notebooks, a browser, desktop UI, or an external tracking dashboard.
Use committed configs and environment variables for noninteractive execution, with
logs and results saved to disk for inspection from the terminal.

The current evaluation CLI supports this workflow. Long-running training commands
must additionally support checkpoint resume and persistent progress logs when the
training runner is implemented. Run long jobs under a persistent terminal session
or the remote machine's job scheduler, so their lifetime is independent of SSH.

In `configs/local_server.toml`, `127.0.0.1` means the machine running the evaluation
command. If evaluation and model serving both run on the remote GPU host, the default
address connects them there; it does not refer to your laptop.

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

The smoke run replays **scripted fixtures**, exercising retrieval, narrow revision,
stale decisions, no-tool responses, and draft/canon separation. Its five tasks are
public development examples, **not a held-out benchmark or a model-quality result**.
The CLI exits nonzero if any task fails.

## Layout

```text
src/writing_agent/       Tool workspace, inference clients, agent loop, scoring, data, CLI
configs/                Committed experiment settings; paths relative to config files
data/fixtures/          Tiny synthetic development tasks and trajectory format example
tests/                  Behavioral tests and an end-to-end offline evaluation
docs/                   Research sources, design decisions, data contract, next milestones
creative_writing_agent_research/  Original research documents
runs/                   Ignored run manifests, traces, scores, and private task workspaces
```

## Evaluate a model

Start a separately managed chat-completions server, such as vLLM, configured with
the chosen model's chat template and tool-call parser. Set `model` and `base_url`
in `configs/local_server.toml`, then run:

```bash
uv run cwa eval --config configs/local_server.toml
```

If the server requires authentication, supply `CWA_API_KEY` through the environment.
No API key belongs in a config. Token usage comes from the server; an empty usage
object means unavailable, not zero tokens. Temperature and seed are recorded but
do not guarantee deterministic GPU inference. Record the exact served checkpoint,
adapter, server version, and launch settings alongside each real experiment.

Each run records a config, task snapshot and hash, package/source fingerprint,
per-task JSONL traces, final workspace, results, and summary. Failed generations
remain in the denominator. Baseline and future tuned variants use the same loop.

## Data preparation

See [the trajectory contract](docs/data-contract.md). The example is intentionally
`pending` review. The exporter includes only `accepted` records in the `train` split:

```bash
uv run cwa validate-data path/to/trajectories.jsonl
uv run cwa export-sft path/to/trajectories.jsonl data/processed/train.jsonl
```

Export produces TRL-style `messages` and `tools` columns and refuses to overwrite
an existing file. It does not train a model or automatically certify data quality.

## Scope and next steps

The filesystem tools block absolute paths, traversal, symlinks, oversized files,
and ambiguous patches. They are a constrained text interface, **not an OS sandbox**;
do not expose the workspace to concurrent untrusted filesystem writers.

Current scorers check literal output constraints, file outcomes, tool errors, and
edit scope. Literary quality is explicitly unscored. There is no automatic canon
commit policy: the prompt teaches the distinction, and tests check file outcomes.

GPU training, adapter switching, assistant-loss masking checks, human judging,
50–100 held-out tasks, and long-running conversation/compaction experiments remain
future work. See [research and architecture](docs/setup-research.md) for the rationale
and [implementation milestones](docs/milestones.md) for the next concrete steps.
