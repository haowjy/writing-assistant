# Creative Writing Agent

A co-writer for long-running fiction: an agent that works over a project folder (the
manuscript plus a knowledge base of character, world, timeline and decision notes), keeps
committed canon apart from tentative ideas, tracks what each character knows, writes in the
author's style and proposes genuinely different directions.

The project restarted in November 2026. Work proceeds in this order:

1. A creative-writing benchmark and an RL environment, built and debugged on open-weight models
   (free to run); once the environment works, frontier models join as subjects and judges.
2. Training, once both work.
3. Comparing model-development ideas against them, on a held-out split never used for
   selection.

The benchmark is the `writing_assistant` package under `src/` (Python 3.12, `uv`). Its
packages are layered and `tests/test_layering.py` enforces the direction; the layer table and
the four protocols are in the work item's `plan/benchmark-code.md` §2. Run tests with
`CUDA_VISIBLE_DEVICES= uv run pytest` (no test may touch the network or a GPU); the fakes in
`tests/conftest.py` stand in for models, search providers and the model client.

## Running the benchmark

`models.toml` names every endpoint and judge; keys come from `.env` in the working directory
(`DEEPINFRA_API_KEY`), which `wa-bench` loads and never writes anywhere. Data lives outside
the repo under `/data/jimyao/research/writing-assistant/` (`WA_DATA` overrides): the corpus
text cache in `corpus-cache/`, task rows in `tasks/{1a,3}.jsonl`, personal manuscripts in
`local_dev/` (never synced, exported or sent to an endpoint without `local = true`).

```
uv run python scripts/build_seed_tasks.py collect    # more Gutenberg authors (network)
uv run python scripts/build_seed_tasks.py build      # task rows; judge answers cached in tasks/
uv run wa-bench run --suite 1a --endpoint gemma-4-26b [--where kind=pov_write] [--limit N]
uv run wa-bench score --run runs/<id> --judges eval|train   # re-score saved episodes
uv run wa-bench report --run runs/<id> [--group-by kind,condition] [--baseline runs/<base>]
uv run wa-bench guardrails --endpoint gemma-4-26b [--run runs/<id>]  # IFEval, HumanEval+, BFCL; docker, ~36 min
```

`--where` takes `key=value` tests joined by commas; a comma not followed by `key=` belongs to
the value, so `--where "author=Austen, Jane,mode=instruction"` works.
`report` on a directory holding only `guardrails.json` prints those scores.

`run` defaults to the `dev` split; `test` is held out and never used to choose anything, so
`--split test` also needs `--held-out-test` and every such run is logged in
`tasks/test-split-uses.jsonl`. A run directory is resumable (same `--out`); a resume with
another `--max-tokens` is refused, and an HTTP 401/402/403 from a host stops the run with
exit 1. Scoring reads each task kind's artefact, never `Episode.status`: an episode with
no reply or file to score gets no outcome metric, and `report` counts it under "no
artefact". Twins are baselines inside their parent's `net_*`, and `report` never averages
across conditions. Every metric that depends on a model judge reads "ungated" in `report`
until that judge's human-label gate passes (`[gates] passed` in `models.toml`), and is left
out of `report --baseline`. Each run stores the judge items behind its scores in
`judgments-<profile>.jsonl`, in `label_ui.py`'s format. Runs, task rows and model outputs
are not committed.

Project knowledge lives in the Meridian KB, not in this repo: run `meridian context --name kb`
and start at its `index.md` (goals, vocabulary, evaluation principles, training lessons). Work
items and their decisions are under `meridian context --name work`.

Compute: an RTX 3090 on this machine (read `~/.context/MACHINE/gpu.md` before GPU work),
Google Colab credits for training, and a 48 GB M4 Pro Mac mini on the tailnet (`j-mini48`)
for inference and judge models.

The code and history before the restart are in the private
`haowjy/writing-assistant-archive` repository, and a bundle of every ref is in
`/data/jimyao/research/writing-assistant/archive/`. Do not commit model weights or book
corpora.
