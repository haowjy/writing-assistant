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
`project-reorientation` (`meridian context --name work` gives its folder). The suite protocol
changed since: a suite keeps no state outside the episode, and its `next_delivery(ep,
last_reply)` is one step that `suites.play()`, the only loop, drives. A driver rebuilds its
conversation from the episode at `open()`, so an episode saved when a delivery returns
continues as the original would have.

`run/` sits between `suites`/`drivers` and `cli`. `Runner.play` saves a group of episodes
before play and each time a delivery returns, and closes them together; `run/directory.py`
is the one way a run directory is claimed (`open_run`); `run/control.py` is the controller.
Anything that needs episodes played (a generation loop, a trainer) calls `run`, never `cli`.

**Judging is a separate pass that nothing growing a world imports.** `Runner` plays and
closes; scoring is `run/scoring.py`, which the module that holds `Runner`, `run/control.py`
and `master/` do not import, nor `judges` (`tests/test_layering.py`). `wa-bench run` scores
what it played for suites 1a and 3 as a step of the command; a rollout of a situation is
scored only by `wa-bench score`. A run killed between a close and its scoring leaves a closed
episode with no row, which the same command scores first when run again.

**The controller** (`run/control.py`): `play_groups(lib, [Work(world, situation, rollouts,
round)], Settings(...))` returns each group's id once every group is played in the library,
and raises `Unplayed` otherwise. It keeps a run directory per world, suite and run identity
under `Settings.runs`, so the same work asked for again after a kill finds the run's id, the
same groups and what is open, and plays only what is missing. It pushes each rollout as it
closes, never mints a round, and holds `run.lock` exclusively, so a second call for the same
world and identity is refused while one is alive. Its pushes carry no score row. Losing a
controller run directory loses the run's id: the same work then plays as new groups.

**The run log.** `run` and `score` append one JSON line per event (`run.Log`: `started`,
`opened`, `resumed`, `closed`, `failed`, `pushed`, `judged`, `finished`, each with `at`,
`pid` and the names that find the thing in the store) to `log.jsonl` in the run directory;
`run --world` and `score --world` also to `<library>/worlds/<world>.log.jsonl`. A line holds
no episode text, and a failure of a `local_dev` episode logs the error's type only.
`wa-bench status --run DIR | --world NAME` reads the log and the store: what is running or
last ran, how far it got, what failed, what the same command would do next. Log a new code
path through the same writer.

Two rules live in `run/`:

- A **seat** is an entry in `Episode.actors`. `setup` gives it `scripted` (the suite's step
  speaks for it) or a **cast name**: `subject`, the model under test, or any other name the
  task uses (`counterpart`, ...). Nothing writes an endpoint there. The run's
  **cast** maps names to endpoints: `subject` is `--endpoint`, the rest come from the
  `[cast.<name>]` table `--cast` picks in `models.toml` (a table defining `subject` or
  `scripted` is refused). `run.Binding` looks a seat up when it is first called, so one
  stored episode is played by any cast and a rollout onto another subject changes no other
  seat. `Runner.bind` refuses a seat whose name the cast lacks, before the episode is saved
  or any model called. The resolved cast (name, endpoint, model) is in run.json's identity
  (without `subject`, which `endpoint` and `model` already are) and on every episode a run
  binds (`meta["cast"]`, `subject` included); who really answered a call is
  `CallRecord.endpoint`. An endpoint's entry there carries `thinking` (`trace` or `summary`)
  when models.toml gives one, which is where a judge that reads thinking learns whose it is.
  Thinking is never replayed to a model and never counted as the reply. `report --baseline`
  refuses two runs whose episodes used a cast name, other than `subject`, played by
  different models; the page's side-by-side screen does not check.
- The ten `base-*` stores were written when a run overwrote its subject's seats with its
  endpoint; `Runner.rollouts` puts `subject` back on those seats, and that line goes when
  the stores do.
- A seat that is not the subject's is driven through `direct` whatever `--driver` is. What
  a seat samples with is the episode's when `setup` gave it (`state["sampling"][seat]`), so
  the other seats keep their limits whatever `--max-tokens` a run has; otherwise the
  temperature is the task kind's or 0.7, and the limits are the run's.

A conversation among any number of seats is one shared step, `suites/converse.py`: seats
speak in table order, every line is a `say` event visible to the seats that hear it
(`data["to"]` addresses it to some), a model seat is delivered what it was shown since its
last delivery, and a scripted seat speaks from `state["script"]`. It ends at `max_turns`,
on `conversation_end`, on an empty script or when every seat was passed over
(`state["converse"]["ended"]`). A delivery to a seat that is not the subject's that leaves
the episode truncated fails it, so its row is `skipped`. Only round robin is built. Suite 1a
keeps its own two schedules until it goes. A module file directly under `suites/` is shared
by every suite package and imports none.

`suites/situations/` is a placeholder suite with one kind, `converse`, built on that step
(seats, each with a cast name, a brief, who hears it, the restricted files it may read; a
request; a turn budget; scored with no judge on turns, the subject's lines and words, and how
it ended). It is what later phases play until the benchmark's new kinds exist; its rows
hold no book text and it declares no `text_fields`. It sets a task up at a position
(`situate`) or from a row alone.

Run tests with
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
uv run wa-bench run --suite 1a --endpoint gemma-4-26b --from '<ep_id>@<n>' --from-run runs/<id> --rollouts N [--round R] [--cast bench] --out runs/<new>   # a group of N rollouts of a stored turn; omit --from-run only when --out already holds the episode; no --where or --limit
uv run wa-bench situate --run runs/<id> --at '<closed ep_id>' --row task.json [--from-run runs/<other>]   # store a new task (suite, kind, payload) on that episode's final files; play it with run --from '<name>@0'
uv run wa-bench score --run runs/<id> --judges eval|train [--all|--pending]   # score what no row stands for; --all: everything again; --pending: only where the scorer raised
uv run wa-bench report --run runs/<id> [--group-by kind,condition] [--baseline runs/<base>]
uv run wa-bench status --run runs/<id> | --world NAME   # what ran, how far it got, what failed, what is next
uv run wa-bench guardrails --endpoint gemma-4-26b [--run runs/<id>]  # IFEval, HumanEval+, BFCL; docker; hours locally (scripts/guardrails/.context)
uv run wa-bench view [--run runs/<id> runs/<id2> ...] [--world NAME ...] [--port 8766]  # read-only page: task rows, scores, episodes, judges; a world's situations, rollouts and findings
```

`run --suite` on task rows and `run --from` on a run directory are the older flow, and
still the way suites 1a and 3 are run. A task set up on a world's files (the practice run,
the situations suite) is played in a library world: "The library" below walks through it.

`view`'s page is a React app in `viewer/` (Bun, Vite, TypeScript, shadcn/ui on Tailwind).
Build it once, and after changing it, before `wa-bench view`:

```
cd viewer && bun install && bun run build   # writes src/writing_assistant/cli/view_dist/ (not committed)
cd viewer && bun run dev                    # live reload on :5173; API from `wa-bench view --port 8767`
```

The Python tests do not need the page built; `view` refuses to start without it.

`view` serves a read-only page on loopback (`--host` binds another address, refused while a
run holds `local_dev` episodes) with a JSON API under `/api/`. A task row is shown by running
the suite's own `setup` and `suites.play()` with a turn that answers nothing, so what the
page shows is what a run would make; `test` rows appear only behind the page's switch,
`local_dev` rows never. The words that say what a kind tests and a metric measures live in
the suite (`ABOUT`). `view --world NAME` adds `/world/<name>`; it loads every rollout of the
world, so it is for a world a person reads. Adding a shadcn component with `bunx shadcn add`
imports `cn` from an npm package named `cn`; point it back at `@/lib/utils` and `bun remove cn`.

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
`test-split-uses.jsonl` (in the data directory). A run directory is resumable (same
`--out`); a resume with another identity (suite, driver, endpoint, model, split,
`--max-tokens`, `--cast`) is refused, and an HTTP 401/402/403 from a host stops the run
with exit 1, what was saved staying resumable.

## How episodes are stored

A run directory is `run.json` (the run's identity, its random `id`, one entry per
invocation, the USD hosts reported, and `served`: the weights file and engine build that
answered), `store.db`, `log.jsonl`, `errors.jsonl` and `run.lock`. Episodes and their score
rows are in `store.db` (SQLite; `episode/segments.py` has the model and the rules,
`design/branch-store.md` and `design/buildout.md` in the work item `project-reorientation`
the reasons).

- The store's format is a count of steps that only add (`PRAGMA user_version`, now 3). A
  writer that opens an older store brings it up to date in place, in one transaction; a
  reader changes no file and refuses an older store, saying to `score` or `run` it once. A
  change that cannot be made by adding is a new store with no reader for the old one.
  Format 1 kept score rows in JSONL files beside the store: nothing reads those now.
- An episode is a **segment**: its own part of a log, stored after a parent segment or none.
  `Episode.parent` means only that; `Episode` itself keeps no history.
- `run` saves each episode before play and every time a delivery returns (**turn
  checkpoints**), and closes a task's episodes together (the **closing checkpoint**).
  Readers (`score`, `report`, `view`, `sync`, `build_calibration.py`) see closed episodes
  only.
- A killed run loses the deliveries in flight and nothing else: the next `run` on the
  directory continues every open episode from its last turn checkpoint through the same
  loop. Runs that choose different tasks may share a directory; `run.lock` tells a starting
  run whether the open episodes are another live run's. Two runs choosing the same tasks
  are not stopped: each task is played once and the loser prints a setup error.
- Play continues, or is forked, from a turn checkpoint only. `--from '<ep_id>@<n>'` forks
  checkpoint `n` `--rollouts` times; with `--from-run` the episode's chain is first copied
  in, so every store is self-contained. `n` counts the named episode's own checkpoints, 0
  being its start, so `@1` of a rollout of `<ep_id>@3` is the source's fourth delivery.
- A **rollout** records what it replays in `Episode.rollout_of`. The rollouts one run makes
  of one checkpoint in one `--round` are a **group** (`Episode.group`, 12 hex digits derived
  from run.json's `id`, the round, the source's name and the checkpoint); member `i` is
  `<ep_id>@<n>.<group>.<i>`, so each samples with its own seed, two runs never make the
  same name, and a resumed run makes only the members still missing. A twin's group is its
  base's, so rollout `i` of each is the matched pair. A group's size is fixed by the first
  command that asks for it (`meta["group_size"]`, per source episode); more rollouts are
  another round. A group is **played** (`Store.groups()`, which reads no score) when every
  source episode in it has that many closed members: that is all a runner, the controller
  or the master asks. It is **settled** under one judging (`Store.settled(judging)`) when
  it is played, no member's row is an `error`, and no two rows were scored by different
  judges for one use. A rollout carries its own metrics and no `net_*` (its source's twin
  had another conversation, so subtracting it is no matched counterfactual). `report`,
  `view` and calibration sampling keep rollouts out of the run's own numbers; `report`
  prints them per checkpoint and group after the run's groups.
- A **start** is an episode stored once and never played in place (`Store.add_starts`). A
  **situation** is a start with a parent: a new task set up at a **position**, a closed
  episode (`wa-bench situate`, `run.situate`). The task takes the position's files and
  revisions and nothing of its session (`Episode.begin`). `begin` keeps
  `state["file_acl"]` with every restricted path readable by nobody, and `setup` must
  grant the new seats what they may read: a path the list does not cover is open to
  everyone. Play of a situation is always a group of rollouts. `Store.names` and `episodes`
  leave starts out, so no reader of played episodes meets one. A situation's name is
  `<its root episode's name>/<12 hex>` unless the row gives `task_id`; its row is in
  `state["task"]`.
- A store takes an episode's `meta` once, when it first saves it: bind and stamp an episode
  before its first `Store.append`. `Runner.bind` is the one place a run's stamps are put.
- Twins are paired by `Episode.baseline_of`, never by who was forked from whom.
- A `test` or `local_dev` segment never shares a store with another split's. `sync --runs`
  takes the folder that holds run directories. It uploads no store file: it writes a new
  one holding the closed episodes that may leave, and holds an episode back when any
  segment in its chain, or in the chain of an episode its judge items quote, is `local_dev`
  or draws on a work whose licence does not allow release. It may run beside a `run`: each
  episode is tested and copied inside one read transaction (`Store.reading()`).
- A refusal the user should read as a sentence is raised as `segments.Unreadable`, which
  `wa-bench` turns into its exit message. A run directory from before the store
  (`episodes.jsonl`) is refused that way.
- The ten `base-*` baseline runs are in
  `/data/jimyao/research/writing-assistant/baseline-stores/` (format 1, rows in JSONL) with
  the one-off converter; this build reads copies brought up to date by
  `reports/buildout-phase1-import-base-rows.py` in the work item, which reproduce
  `report`, the page's API and calibration sampling exactly. No run directory is committed.

## The library

A **world** is one store file, `<library>/worlds/<name>.db` (`episode/library.py`; the
library is the folder `WA_LIBRARY` names, default `/cold/writing-assistant/library`, and
tests and scratch checks always set it). Its root is a start with no parent holding the
world's files; everything in it descends from that root and has its split. There is no
registry: `Library.worlds()` reads the files.

**From nothing to a read result**, as the practice run is done. Every command reads
`WA_LIBRARY`; unset, it is the real library. `--out` is where `run --world` keeps its run
directories (default `runs/` in the working directory): keep it, since a killed run resumes
from it and a restore pushes from it again.

```
export WA_LIBRARY=/cold/<scratch>/library
uv run wa-bench library create alder-quay --files pack/world/base --closed kb/private-gate.md   # the world: its root holds the folder's files
uv run wa-bench situate --world alder-quay --at alder-quay --row pack/tasks/advance-scene.json   # a task on the root's files; prints its name
uv run wa-bench situate --world alder-quay --at alder-quay --row pack/tasks/repair.json --files pack/world/sprawling   # a start that adds or replaces files
uv run wa-bench library list                                                   # name, split, situations, played episodes, groups
llmpick start gemma-4-26b-qat                                                  # serve the subject
uv run wa-bench run --world alder-quay --endpoint gemma-4-26b --rollouts 3 --out runs/practice [--where task_id=alder-quay/advance-scene]
uv run wa-bench score --world alder-quay --judges practice                     # each rollout under the rubrics its row lists
uv run wa-bench report --world alder-quay --judges practice [--rubric world_coherence]
uv run wa-bench view --world alder-quay                                        # the page: situations, rollouts, findings
```

- `--at` is a closed episode of the world: its own name (the root), a situation, or a
  played rollout. A row that depends on added files should list them in
  `payload.requires`; situating it where one is missing is refused.
- A row lists its rubrics in `payload.rubrics`. `counts` needs no judge;
  `world_coherence`, `character_voice_consistency` and `thinking_about_prose` each need the
  `--judges` profile to name an endpoint for the use of the same name, or `score` refuses
  before writing a row. No profile in the checkout's `models.toml` names them yet.
- Asked again, `run --world` plays only what is missing of the same groups (`--round R`
  makes new ones) and `score --world` scores only what no row stands for.
- `report --run` on the run directory under `--out` shows those rollouts unscored: the
  rows are in the world.
- **In and out.** `checkout` copies chains into a worker's store unlisted and without score
  rows, so a position is never read there as an episode that run played. `push` adds closed
  episodes with their chains and listings in one transaction; a checkpoint the world holds
  with another body (two stores continued one episode) is refused with nothing copied. A
  push repeated, or cut short and repeated, changes nothing or completes. Everything is
  added; nothing in a world is ever continued or rewritten.
- **World files use SQLite's write-ahead log**, run directories the rollback journal.
  Measured on a 100,000-rollout world (4.7 GB): in rollback mode `verify` held a push for
  44 s and `snapshot` for 55 s of the 60 s busy timeout; with the log a push started during
  either lands in about 20 ms. So a world has `-wal` and `-shm` files beside it while open;
  a snapshot is stored in rollback mode as one file, and `restore` puts the log back.
- **Keeping it whole.** `verify` runs SQLite's integrity check and loads every listed
  episode (the check alone, 3 s on that world, cannot see an altered checkpoint body; the
  whole takes 43 s). `snapshot` verifies, copies, verifies the copy and keeps the two
  newest. `restore` puts the newest snapshot that verifies in the world's place and keeps
  the file that was there beside it (`.replaced-<time>`). What was added after that
  snapshot is gone: push the rollouts again from their run directories, then `score
  --world`. `<world>.lock` is held shared by every reader and writer for as long as its
  connection is open and exclusively by a restore, so keep a `with lib.open(...)` block as
  short as the reading.
- **Costs at depth** (50 generations, on `/cold`): a checkout of the deepest situation is
  25 ms and 2.6 MB whether the world holds 1,000 or 100,000 rollouts; a push of one rollout
  0.1 to 0.3 s, of a group of 50 about 0.2 s. `Store.groups(one)` 6 ms, `settled(.., one)`
  19 ms; asked of every group they walk the file (1.3 s and 2.9 s at 100,000), as does
  `names()` (1.2 s), so per-group callers name the group.
- **The judging pass** finds what to score with one query (`Store.judged(profile)`) and one
  loaded rollout per situation (1.6 s and 0.9 s on the 100,000-rollout world with 2,000
  situations: what a pass with nothing to do costs); rollouts to be scored are loaded a
  batch at a time. It may run beside a `run --world`: it scores what is closed when it
  looks. One pass per world and judge profile at a time (`Library.judging`), since two
  would each pay the judge for the same rollouts.
- The library has no rule for splits or personal text: a `local_dev` world is a file on this
  machine, a hosted subject is refused for it by `run.directory.identity`, and a hosted
  judge leaves its findings `local_only` without being called. Nothing copies a world off
  this machine (`sync` stages run directories only).

## Scoring

Scoring reads each task kind's artefact, never `Episode.status`: an episode with
no reply or file to score gets no outcome metric, and `report` counts it under "no
artefact". A refusal is the host's stop reason only; a reply under half the length its task
asked for is `too_short`, not scored. `collab_edit` has one artefact per round: a round the
writer skips loses its metrics, the others keep theirs. Twins are baselines inside their parent's `net_*`, and `report` never averages
across conditions. Every metric that depends on a model judge reads "ungated" in `report`
until that judge's human-label gate passes (`[gates] passed` in `models.toml`), and is left
out of `report --baseline`.

A suite's scoring is split into **rubrics** (`Suite.rubrics`: rubric name -> the task kinds
it is run on by default; a task row may list others in `payload["rubrics"]`), and an
episode has one score row per **judging**: a judge profile and a rubric, stored as
`<profile>/<rubric>`. Suites 1a and 3 have one unnamed rubric whose judging is the bare
profile. `episode.segments.judging` and `split_judging` are the only code that joins or
splits that name; every command takes `--judges PROFILE [--rubric NAME]`. With no
`--rubric`, `score` covers each episode's listed rubrics, `score --pending` every rubric
the profile has rows for, and `report` every such rubric as a block. `Suite.score(ep,
services, tallies)` is handed one `Tally` per rubric wanted, so two rubrics may share one
judge call. A gate stays per judge use (`<profile>.<use>`). The page's front end shows the
profile's first rubric only.

**A rubric a judge scores by reading the rollout** (`judges/findings.py`): one judge use
per prompt file in `judges/rubrics/` (the file name is the use; the prompt version is a
hash of the text, so editing a prompt starts a new version). The judge is given the stored
rollout rendered for one seat (`episode.rollout_evidence`, `judges.findings.render`) and
replies with a JSON `findings` array; a reply in any other shape is unscored, never read
leniently. A finding is stored as `covers`, `seat`, `dimension` (`output` or `thinking`),
`criterion`, `value` 0/1/2 or `unscored` with the reason, and references: no text of the
rollout, since the record can be rendered again. The situations suite registers three
(`READS`): `world_coherence` and `character_voice_consistency` cover the task,
`thinking_about_prose` each turn. Each is asked once per seat the subject held; metrics are
`<dimension>.<criterion>` on 0 to 1. A criterion about thinking is not asked where none was
recorded and is stored as `no_thinking`. Scoring one of these under a profile that names
no endpoint for it is refused before any row is written (`Suite.needs`,
`run.scoring.refuse`): such a row would read as the rollout having been judged.

**What `score` scores** is decided in one place for a run directory and a world
(`run.scoring.to_score`). An episode's rubric is scored when its row does not **stand**:
there is none, it says the scorer raised, or a judge stamped on it (use, endpoint, model,
prompt version) is not the one the profile names now. So a second `score` adds nothing,
and an edited prompt or another endpoint under the same profile name is scored again,
once. `--all` scores every listed rubric again, `--pending` only rows that say the scorer
raised. Known limit: the stamp covers the prompt text, and neither the code that renders
the record a judge reads nor the scoring code; a row's `code` is the only trace, so after
a scorer fix run `score --all`.

Score rows live in the store, per judging, each with the judge items behind it in
`label_ui.py`'s format; a row says which code scored it (`code`), when (`scored_at`) and by
which judges (`judges`). Rows are only ever added. All readers take one row per episode
and judging by one rule (`segments._current`): the newest row that is not an `error`, or
the newest of all when every row is one. So a re-score that fails hides no good row and
unsettles no group, and the row shown may come from older scoring code than the latest
attempt: read its `code`. Each re-score adds about as much as its rows took the first time
(0.4 MB on the 1.4 MB `base-1a-gemma-4-26b` store); nothing removes superseded rows. A
profile is a name in one machine's `models.toml`, so `report` says so under a group whose
rows were scored by two judges for one use; it does not yet keep their metrics apart.
Runs, task rows and model outputs are not committed.

A judge's calibration file is drawn from those stored judgments by
`scripts/build_calibration.py sample --use <use> --runs <run dirs> --n <N> --out
$WA_DATA/calibration/<file>` (balanced across the judge's answers, metrics, subjects and
templates; `--build-checks` adds the `1a` build's later-event `fact` checks). `add-judge
--endpoint <name>` adds another judge's answers to the same items, and `label_ui.py
--agreement --judge <model>` scores them against the same labels. For a judge that reads a
rollout, `sample --use <rubric> --worlds NAME [--runs DIR]` turns each scored finding into
a yes/no item: the question states the judge's score and what it cited, the evidence is
the record the judge read, rendered again from the stored rollout, and "yes" agrees with
the score. So label files hold the text and stored findings do not. Label files are not
committed.

Project knowledge lives in the Meridian KB, not in this repo: run `meridian context --name kb`
and start at its `index.md` (goals, vocabulary, evaluation principles, training lessons). Work
items and their decisions are under `meridian context --name work`.

Compute: an RTX 3090 on this machine (read `~/.context/MACHINE/gpu.md` before GPU work),
Google Colab credits for training, and a 48 GB M4 Pro Mac mini on the tailnet (`j-mini48`)
for inference and judge models.

## Where things run

Run locally first (the 3090, `j-mini48`). Plan every cloud run in the work directory
before it starts: what runs, expected cost, what stops it. One that costs money needs
Jimmy's approval; free usage does not.

## What the code handles

Do not write code for rare edge cases. Make every run recoverable instead (a killed or
failed run restarts from saved state) and log what it does, so a failure can be traced.

## Messages and prompts

Start every message and prompt minimal: a command's output and errors, a judge or seat
prompt, a brief. Add text only when a real failure shows it is needed. Delete text that
does not help its reader.

## Google services

Use Google's CLIs through their skills, or a Google-hosted MCP server if connected:
`colab` (`colab-operator` skill) for Colab, `gwscli` (`gwscli` skill) for Drive and other
Workspace data. No direct REST calls: Google requires agents to use these from
February 2027.

The code and history before the restart are in the private
`haowjy/writing-assistant-archive` repository, and a bundle of every ref is in
`/data/jimyao/research/writing-assistant/archive/`. Do not commit model weights or book
corpora.
