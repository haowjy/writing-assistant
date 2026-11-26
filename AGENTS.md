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
the four protocols are in `plan/benchmark-code.md` §2 of the work item
`project-reorientation` (`meridian context --name work` gives its folder). Three of the four
stand as written there; the suite protocol changed. A suite no longer has a `play`: it keeps no state outside the episode, and its `next_delivery(ep,
last_reply)` is one step (do what the last reply calls for, name the next delivery) that
`suites.play()`, the only loop, drives. A driver likewise rebuilds its conversation from the
episode at `open()`, so an episode saved when a delivery returns continues as the original
would have (`suites/__init__.py`). Run tests with
`CUDA_VISIBLE_DEVICES= uv run pytest` (no test may touch the network or a GPU); the fakes in
`tests/conftest.py` stand in for models, search providers and the model client.

## Running the benchmark

`models.toml` names every endpoint and judge. The subjects (`inkling-2-mini`, `qwen3.5-9b`,
`gemma-4-26b`) are served on this machine's GPU by `llmpick`, one at a time: start the
subject's profile (named beside its entry; check `llmpick status` first, the card is shared)
before `run`. DeepInfra entries (`*-deepinfra`, judges) are fallbacks; their key comes from
`.env` in the working directory (`DEEPINFRA_API_KEY`), which `wa-bench` loads and never
writes anywhere. Data lives outside
the repo under `/data/jimyao/research/writing-assistant/` (`WA_DATA` overrides): the corpus
text cache in `corpus-cache/`, task rows in `tasks/{1a,3}.jsonl`, personal manuscripts in
`local_dev/` (never synced, exported or sent to an endpoint without `local = true`).

```
uv run python scripts/build_seed_tasks.py collect    # more Gutenberg authors (network)
uv run python scripts/build_seed_tasks.py build      # task rows; judge answers cached in judge-cache.jsonl
llmpick start gemma-4-26b-qat                        # serve the subject first
uv run wa-bench run --suite 1a --endpoint gemma-4-26b [--where kind=pov_write] [--limit N] [--judges none]
uv run wa-bench run --suite 1a --endpoint gemma-4-26b --from '<ep_id>@<n>' --from-run runs/<id> --rollouts N --out runs/<new>   # N rollouts of a stored turn; omit --from-run only when --out already holds the episode; no --where or --limit
uv run wa-bench score --run runs/<id> --judges eval|train   # re-score saved episodes
uv run wa-bench report --run runs/<id> [--group-by kind,condition] [--baseline runs/<base>]
uv run wa-bench guardrails --endpoint gemma-4-26b [--run runs/<id>]  # IFEval, HumanEval+, BFCL; docker; hours locally (scripts/guardrails/.context)
uv run wa-bench view [--run runs/<id> runs/<id2> ...] [--port 8766]  # read-only page: task rows, scores, episodes, judges
```

`view`'s page is a React app in `viewer/` (Bun, Vite, TypeScript, shadcn/ui on Tailwind).
Build it once, and after changing it, before `wa-bench view`:

```
cd viewer && bun install && bun run build   # writes src/writing_assistant/cli/view_dist/ (not committed)
cd viewer && bun run dev                    # live reload on :5173; API from `wa-bench view --port 8767`
```

The Python tests do not need the page built; `view` refuses to start without it.

`view` serves a page on loopback (`--host` binds another address, refused while a run holds
`local_dev` episodes) with a JSON API under `/api/`. Every screen has its own URL under
`/<suite>/<split>/`, filters and the metric included: the Overview (`report`'s own
`build_report` per task kind, the loaded runs side by side), a kind's Tasks
(`kind/<kind>/tasks`: one row per task, one column per run), and what one model wrote
(`task/<task>/<arm>?run=`: the prose, the judge's questions and replies, the twin, what
differs in its prompt from the other arm, reasoning and tool calls; `&compare=1` puts the runs
side by side). The words that say what a task kind tests and what a metric measures live in
the suite (`ABOUT` beside its tasks and its scorer) and reach the page through `/api/kinds`.
A task row is shown by running the suite's own `setup` and `suites.play()` with a turn that answers
nothing, so its episodes, twin swaps, delivered messages and the payload fields kept from the
model are what a run would make; held-out `test` rows appear only behind the page's switch,
`local_dev` rows never. Adding a shadcn component with `bunx shadcn add` imports `cn` from an
npm package named `cn`; point it back at `@/lib/utils` and `bun remove cn`.
It writes nothing, fetches one task at a time and re-reads a run whose files change. An
open page keeps what it fetched for 30 seconds and does not poll, and between a task's
episodes closing and their scores being written its report shows them as unscored.

`--judges none` names no judge: with no judge served, `run` plays, saves and scores only what
needs no model judge (the judged metrics print as unscored "unavailable"), and `score
--judges eval` judges the saved run once a judge is up. run.json `served` records the weights
file and engine build that answered, so a number names the build it holds for.
`GUARDRAILS_LIMIT=N` runs N items per guardrail tool and writes no `guardrails.json`: check
a new endpoint with it before a full run.

`--where` takes `key=value` tests joined by commas; a comma not followed by `key=` belongs to
the value, so `--where "author=Austen, Jane,mode=instruction"` works.
`report` on a directory holding only `guardrails.json` prints those scores.

`run` defaults to the `dev` split; `test` is held out and never used to choose anything, so
`--split test` also needs `--held-out-test` and every such run is logged in
`test-split-uses.jsonl` (in the data directory, beside `tasks/`). A run directory is resumable (same `--out`); a resume with
another `--max-tokens` is refused, and an HTTP 401/402/403 from a host stops the run with
exit 1.

## How episodes are stored

A run's episodes are in `store.db` (SQLite, standard library; `episode/segments.py` has the
model and the rules, `design/branch-store.md` in the work item `project-reorientation` the reasons). There is no
other episode format and no reader for an older one.

- An episode is a **segment**: its own part of a log, stored after a parent segment or none.
  `Episode.parent` means only that; `Episode` itself keeps no history.
- `run` saves each episode before play and again every time a delivery returns (**turn
  checkpoints**), and closes a task's episodes together when its play ends (the **closing
  checkpoint**). `score`, `report`, `view`, `sync` and `build_calibration.py` read closed
  episodes only.
- A killed run loses the deliveries in flight and nothing else: the next `run` on the
  directory continues every open episode from its last turn checkpoint through the same
  loop, and sets up nothing twice. Runs that choose different tasks may share a directory
  at the same time; `run.lock` tells a starting run whether the open episodes are another
  live run's. It does not stop two runs choosing the same tasks: each task is still played
  once, and the run that loses a task prints a setup error for it and exits 1.
- Play continues, or is forked, from a turn checkpoint only. `--from '<ep_id>@<n>'` forks
  the named episode's turn checkpoint `n` `--rollouts` times; with `--from-run` the
  episode's chain is first copied into the new run's store, so every store is
  self-contained. `n` counts the named episode's own checkpoints, 0 being its start: for
  an episode `run` set up, `n` is the checkpoint taken when its n-th delivery returned; a
  rollout starts where it was forked, so `@1` of a rollout of `<ep_id>@3` is the source's
  fourth delivery.
- A **rollout** records what it replays in `Episode.rollout_of` (source `ep_id`, checkpoint)
  and is named `<ep_id>@<n>.<i>`, so each samples with its own seed. It has no twin: it
  carries its own metrics only and gets no `net_*` (its source's twin had another
  conversation, so subtracting it would not be a matched counterfactual). A twin can be
  rolled out too: its rollout carries its own metrics and is the baseline of nothing
  (pairing rollouts into a counterfactual is not built). `report`, `view` and `build_calibration.py` keep rollouts apart from
  the run's task episodes: no group, metric, episode list or sample includes one, and `report`
  prints them after the run's groups, per checkpoint. The lines at the head of `report`
  (spend, served, played by code, task errors) are per directory and do include them, and
  the page has no screen for rollouts yet (its API lists them under `rollouts`). A rollout stores no task, so a
  directory holding only rollouts of a task has not played it.
- Twins are paired by `Episode.baseline_of` (set by `fork(..., baseline=True)` in `setup`),
  never by who was forked from whom.
- A `test` or `local_dev` segment never shares a store with another split's. `sync --runs`
  takes the folder that holds run directories, not one run directory (given one, it finds no
  runs and stages nothing). It uploads
  no store file: it writes a new one holding the closed episodes that may leave, and holds
  an episode back when any segment in its chain is `local_dev` or draws, at any of its
  checkpoints, on a work whose licence does not allow release. It may run while a `run` is
  writing: each episode's chain is tested and copied inside one read transaction
  (`Store.reading()`), so a checkpoint an open ancestor gains after the test is not copied.
- A run directory written before the store (`episodes.jsonl`) cannot be read: `run`,
  `score`, `report`, `view` and `sync` refuse it by name (`segments.has_store`) and write
  nothing; `scripts/build_calibration.py` fails on one with a traceback carrying the same
  sentence. A refusal the user should read as a sentence is raised as `segments.Unreadable`,
  which `wa-bench` turns into its exit message. The ten
  `base-*` baseline runs, converted by replaying their saved calls through the suites so
  they hold real turn checkpoints and can be forked, are in
  `/data/jimyao/research/writing-assistant/baseline-stores/` with the one-off converter;
  no run directory is committed.

## Scoring

Scoring reads each task kind's artefact, never `Episode.status`: an episode with
no reply or file to score gets no outcome metric, and `report` counts it under "no
artefact". A refusal is the host's stop reason only; a reply under half the length its task
asked for is `too_short`, not scored. `collab_edit` has one artefact per round: a round the
writer skips loses its metrics, the others keep theirs. Twins are baselines inside their parent's `net_*`, and `report` never averages
across conditions. Every metric that depends on a model judge reads "ungated" in `report`
until that judge's human-label gate passes (`[gates] passed` in `models.toml`), and is left
out of `report --baseline`. Each run stores the judge items behind its scores in
`judgments-<profile>.jsonl`, in `label_ui.py`'s format. Runs, task rows and model outputs
are not committed.

A judge's calibration file is drawn from those stored judgments by
`scripts/build_calibration.py sample --use <use> --runs <run dirs> --n <N> --out
$WA_DATA/calibration/<file>` (balanced across the judge's answers, metrics, subjects and
templates; `--build-checks` adds the `1a` build's later-event `fact` checks). `add-judge
--endpoint <name>` adds another judge's answers to the same items, and `label_ui.py
--agreement --judge <model>` scores them against the same labels. Label files are not
committed.

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
