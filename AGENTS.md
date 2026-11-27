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
would have (`suites/__init__.py`).

`run/` sits between `suites`/`drivers` and `cli` (it is not in that plan's table). It holds
the play itself: `Runner.play` takes a group of episodes, saves them before play and each
time a delivery returns, and closes them together. `cli` parses arguments, opens
the run directory and chooses what is played; anything else that needs episodes played (a
generation loop, a trainer) calls `run`, never `cli`.

**Judging is a separate pass that nothing growing a world imports.** `Runner` plays and
closes and does nothing else; scoring is `run/scoring.py` (`score_episode`, `score_into`),
and the module that holds `Runner`, `run/control.py` and `master/` import neither it nor
`judges` (`tests/test_layering.py`). `wa-bench run` scores what it played for suites 1a and
3, after the play, as a step of the command; a rollout of a situation (`run.situated`: its
task row is in `state["task"]`) is never scored by `run`, only by `wa-bench score`. So a
run killed between a close and its scoring leaves a closed episode with no row, which the
same command scores first when it is run again.

Two rules live in `run/`:

- A **seat** is an entry in `Episode.actors`. `setup` gives it `scripted` (the suite's step
  speaks for it) or a **cast name**: `subject`, the model under test, or any other name the
  task uses (`counterpart`, `master`, ...). Nothing writes an endpoint there. The run's
  **cast** maps names to endpoints: `subject` is `--endpoint`, the rest come from the
  `[cast.<name>]` table `--cast` picks in `models.toml` (a table defining `subject` or
  `scripted` is refused). `run.Binding` looks a seat up when it is first called, so one
  stored episode is played by any cast and a rollout onto another subject changes no other
  seat. `Runner.bind` refuses a seat whose name the cast lacks, before the episode is saved
  or any model called. The resolved cast (name, endpoint, model) is in run.json's identity
  (without `subject`, which `endpoint` and `model` already are) and on every episode a run
  binds (`meta["cast"]`, `subject` included); who really answered a call is
  `CallRecord.endpoint`. `report --baseline` matches the baseline's rollouts of a checkpoint
  whatever group they are in (a group is one run's), and refuses two runs whose episodes used a cast
  name, other than `subject`, played by different models; the page's side-by-side screen
  does not check.
- The ten `base-*` stores were written when a run overwrote its subject's seats with its
  endpoint and listed them in `meta["subject_actors"]` (added by the import). Readers do
  not care; `Runner.rollouts` puts `subject` back on those seats, and that line goes when
  the stores do. An episode left open by a build before the cast cannot be resumed.
- A seat that is not the subject's is driven through `direct` whatever `--driver` is. What
  a seat samples with is the episode's when `setup` gave it
  (`state["sampling"][seat]`: `temperature`, `max_tokens`, `max_steps`), so the other seats
  keep their limits whatever `--max-tokens` a run has; otherwise the temperature is the
  task kind's (`meta["temperature"]`) or 0.7, and the limits are the run's.

A conversation among any number of seats is one shared step, `suites/converse.py`
(`converse(ep, last_reply, hears=)`, its place in `state["converse"]`): seats speak in table
order; every line is a `say` event by its seat (the opening too, and a model seat's reply;
an empty reply logs none), visible to the seats that hear it; a model seat is delivered the
others' lines it was shown since its last delivery, and one that heard nothing is passed
over; a scripted seat speaks inside the step from `state["script"]` (`suites/script.py`,
one table per episode, taken in speaking order). A line may be addressed: `data["to"]`
names its listeners and only they and the speaker see it, whatever `hears` says (a model
seat's `say_to(seats, text)` tool during its turn, or `to` on a script entry or the
opening); a listener is delivered it as `<speaker> (to <names>): <text>`. The reply the
step is given belongs to the seat the latest `deliver` went to. It ends at `max_turns`
(turns by all seats: the opening, each scripted line, each delivery answered), when a
seat calls `conversation_end`, when a scripted seat finds the script empty, or when every
seat was passed over; `state["converse"]["ended"]` says which. A delivery to a seat that
is not the subject's that leaves the episode truncated fails it with `<seat>: <reason>`, so
its row is `skipped`. A kind built on the step logs no other `say` event. Only round robin
is built; a `hears` that changes mid-conversation and an `apply` hook for what a line does
besides being said are not. Suite 1a keeps its own two schedules (`tasks_1a.next_line`,
`next_dm`) until it goes. A module file directly under `suites/` is shared by every suite
package and imports none (`tests/test_layering.py`).

`suites/situations/` is a placeholder suite with one kind, `converse`, built on that step
(seats, each with a cast name, a brief, who hears it, the restricted files it may read; a
request; a turn budget; scored with no judge on turns, the subject's lines and words, and how
it ended). It is what later phases play until the benchmark's new kinds exist; its rows
hold no book text and it declares no `text_fields`. It sets a task up at a position
(`situate`) or from a row alone.

A store takes an episode's `meta` once, when it first saves it: bind and stamp an episode
before its first `Store.append`. `Runner.bind` is the one place a run's stamps are put
(`endpoint`, `model`, `cast`, `max_tokens`, `code`, `run_id`, `round`), and `Runner.rollouts` adds a
member's `group` and `group_size`.

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
uv run wa-bench score --run runs/<id> --judges eval|train [--pending]   # re-score saved episodes; --pending: only those whose scorer raised
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
another `--max-tokens` or `--cast` is refused, and an HTTP 401/402/403 from a host stops the run with
exit 1.

## How episodes are stored

A run directory is `run.json`, `store.db`, `errors.jsonl` and `run.lock`. The run's episodes
and their score rows are in `store.db` (SQLite, standard library; `episode/segments.py` has
the model and the rules, `design/branch-store.md` and `design/buildout.md` in the work item
`project-reorientation` the reasons).

- The store's format is a count of steps that only add (`PRAGMA user_version`, now 3). A
  writer that opens an older store brings it up to date in place, inside one transaction; a
  reader changes no file, so `report`, `view`, `sync` and `build_calibration.py` refuse an
  older store with a sentence that says to `score` or `run` it once. A change that cannot be
  made by adding is a new store with no reader for the old one. Format 1 kept score rows in
  `scores-<profile>.jsonl` and `judgments-<profile>.jsonl` beside the store: nothing reads
  those files now, and a format 1 directory gets its rows by being scored again.
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
- A **rollout** records what it replays in `Episode.rollout_of` (source `ep_id`, checkpoint).
  The rollouts one run makes of one checkpoint in one `--round` are a **group**
  (`Episode.group`, 12 hex digits derived from run.json's `id`, the round, the source's name
  and the checkpoint), and member `i` is named `<ep_id>@<n>.<group>.<i>`: each samples with
  its own seed, two runs never make the same name, and a resumed run works the id out again
  and makes only the members still missing. A twin's group is its base's, so rollout `i` of
  each is the matched pair. A group's size is fixed by the first command that asks for it,
  per source episode (`meta["group_size"]`; a base and its twin rolled out N times each are
  2N members saying N); another size under the same round is refused, and more rollouts are
  another round. A group is **played** (`Store.groups()`, which reads no score) when every
  source episode in it has that many closed members: that is all a runner, the controller
  or the master asks. It is **settled** under one judging (`Store.settled(judging)`, the
  only one of the two that reads `scores`) when it is played, every member's row is not an
  `error`, and no two of those rows were scored by different judges for one use. Both read
  the store alone, so a base whose twin's rollouts never arrived at all reads as a played
  group of the base. A rollout has no twin: it
  carries its own metrics only and gets no `net_*` (its source's twin had another
  conversation, so subtracting it would not be a matched counterfactual). A twin can be
  rolled out too: its rollout carries its own metrics and is the baseline of nothing
  (pairing rollouts into a counterfactual is not built). `report`, `view` and `build_calibration.py` keep rollouts apart from
  the run's task episodes: no group, metric, episode list or sample includes one, and `report`
  prints them after the run's groups, per checkpoint and group, saying which groups are settled. The lines at the head of `report`
  (spend, served, played by code, task errors) are per directory and do include them, and
  the page has no screen for rollouts yet (its API lists them under `rollouts`, each with its `group`). A rollout stores no task, so a
  directory holding only rollouts of a task has not played it.
- A **start** is an episode stored once and never played in place (`Store.add_starts`: saved,
  closed and marked `episodes.start` in one transaction). A **situation** is a start with a
  parent: a new task set up at a **position**, which is a closed episode (`wa-bench
  situate`, `run.situate`, `Suite.setup(task, services, at)`). The task takes the
  position's files and revisions and nothing of its session (`Episode.begin`: empty log,
  actors, meta and state), so its seats start new conversations and its first step is
  given `""`; the earlier conversation is one link away, in the parent segment. `begin`
  keeps `state["file_acl"]` with every restricted path readable by nobody, and `setup`
  must grant the new seats what they may read: a path the list does not cover is open to
  everyone. Play of a situation is always `run --from '<situation>@0'`. `Store.names` and
  `episodes` leave starts out, so no reader of played episodes (a run's resume and its
  scoring of what has no row, `score`, `report`, `view`, calibration sampling) meets one;
  `Store.starts()` and `situations()` list them, `Store.copy` carries the mark, and `sync`
  stages them like any closed episode. A situation's name is `<its root episode's
  name>/<12 hex>` unless the row gives `task_id`; its row is in `state["task"]`.
- Twins are paired by `Episode.baseline_of` (set by `fork(..., baseline=True)` in `setup`),
  never by who was forked from whom.
- A `test` or `local_dev` segment never shares a store with another split's. `sync --runs`
  takes the folder that holds run directories, not one run directory (given one, it finds no
  runs and stages nothing). It uploads
  no store file: it writes a new one holding the closed episodes that may leave, each with
  its score rows and judge items (`Store.copy`), and holds an episode back when any segment
  in its chain, or in the chain of an episode its judge items quote, is `local_dev` or draws,
  at any of its checkpoints, on a work whose licence does not allow release. It may run while
  a `run` is writing: each episode is tested and copied inside one read transaction
  (`Store.reading()`), so a checkpoint an open ancestor gains after the test is not copied.
- A run directory written before the store (`episodes.jsonl`) cannot be read: `run`,
  `score`, `report`, `view` and `sync` refuse it by name (`segments.has_store`) and write
  nothing; `scripts/build_calibration.py` fails on one with a traceback carrying the same
  sentence. A refusal the user should read as a sentence is raised as `segments.Unreadable`,
  which `wa-bench` turns into its exit message. The ten
  `base-*` baseline runs, converted by replaying their saved calls through the suites so
  they hold real turn checkpoints and can be forked, are in
  `/data/jimyao/research/writing-assistant/baseline-stores/` with the one-off converter.
  Those files are format 1 with their rows in JSONL: this build reads copies that a second
  one-off script brought to format 2 with the rows imported (`reports/buildout-phase1-import-base-rows.py`
  in the work item; it reproduces `report`, the page's API and calibration sampling exactly).
  No run directory is committed.

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
profile, so their stored rows are as they were. `episode.segments.judging` and
`split_judging` are the only code that joins or splits that name (`/` is refused in either
part); every command takes `--judges PROFILE [--rubric NAME]`, and with no `--rubric`
`score` runs each episode's listed rubrics, `score --pending` covers every rubric the
profile has rows for, and `report` and the page's API cover every such rubric (a block
each; the API reads the first and lists the rest under `others`). A listed name the suite
has no rubric for is reported by `score` and stops nothing. `Suite.score(ep, services,
tallies)` is handed one `Tally` per rubric wanted and returns each one's metrics, so two
rubrics may share one judge call (`Tally.found` records several findings from one call, or
findings no judge was asked for). A gate stays per judge use (`<profile>.<use>`). The
page's front end has not been rebuilt for rubrics: it shows the profile's first.

Score rows live in the run's `store.db`, per judging, each with the judge items behind
it in `label_ui.py`'s format; a row says which code scored it (`code`), when (`scored_at`)
and by which judges (`judges`: use, endpoint, model, prompt version). Rows are only ever
added: every scoring of an episode adds a row (`run`, `score`; one at a time, so a `score`
killed part-way keeps what it finished), and copying between stores adds the rows the other
store lacks. All readers take one row per episode and judging by one rule
(`segments._current`, behind `Store.scores`, `judgments` and `settled`): the newest row that
is not an `error`, or the newest of all when every row is one. A row with `error` means the
scorer raised, which is what a judge that was down leaves; `score --pending` scores those
episodes again. So a re-score that fails hides no good row and unsettles no group, and the
row shown may come from older scoring code than the latest attempt: read its `code`. Each
re-score of a run adds about as much as its rows took the first time (measured: 0.4 MB on
the 1.4 MB `base-1a-gemma-4-26b` store, 1.6 MB on the 3.2 MB `base-3-gemma-4-26b`; a row's
judge items are stored packed together); nothing removes superseded rows. A profile is a
name in one machine's `models.toml`, so `report` says so under a group whose rows were
scored by two judges for one use; it does not yet keep their metrics apart. Runs, task rows
and model outputs are not committed.

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

## Where things run

Run locally first (the 3090, `j-mini48`). Plan every cloud run in the work directory
before it starts: what runs, expected cost, what stops it. One that costs money needs
Jimmy's approval; free usage does not.

## Google services

Use Google's CLIs through their skills, or a Google-hosted MCP server if connected:
`colab` (`colab-operator` skill) for Colab, `gwscli` (`gwscli` skill) for Drive and other
Workspace data. No direct REST calls: Google requires agents to use these from
February 2027.

The code and history before the restart are in the private
`haowjy/writing-assistant-archive` repository, and a bundle of every ref is in
`/data/jimyao/research/writing-assistant/archive/`. Do not commit model weights or book
corpora.
