# Custom writing evaluation suite

Design and delivery plan, 2026-09-11. The [delivery plan](plan.md) defines the
Python-first implementation and execution approval boundary. This work item specifies one custom suite
with five task families and two external regression checks. The implementation
provides a Python research script, source catalog, 50 proposed development cases,
artifact scoring, and bounded Astra grading. Labels are not human-calibrated, and
only a small E2B-IT pilot has run. See [delivery evidence](delivery.md).

Task queues: [TODO — now](../TODO.md) and [FUTURE](../FUTURE.md).

Latest rerun: [three native-tool cases](native-pilot.md).

Native harness validation: [read/write smoke and limitations](native-inference.md).

Five-case run: [E2B-IT five-case pilot](pilot-e2b.md), with saved artifacts and observed failures.

## Read this work item

- [Inference and checkpoint evaluation](../../docs/local-inference.md): direct Python runtime, harness protocol, and future training integration.
- [Current coverage](coverage.md): existing genre/style, naming, narrative, instruction, and KB variation; final-generation review checklist.
- [Suite design](design.md): task families, experiments, runner boundaries, and record contracts.
- [Scorecard](metrics.md): quality measures, prose-distribution diagnostics (n-gram L2 and MMD), resource measures, and grading methods.
- [Prose metric catalog](prose-metrics.md): distribution, diversity, reference similarity, repetition, and conditional measures carried forward from research.
- [Data inventory and acquisition](data.md): what exists locally, what to download, what to generate, and webnovel/wiki scraping candidates.
- [Delivery plan](plan.md): milestones, acceptance evidence, pilot size, costs, and deferred work.

These documents own the current evaluation scope. They replace the earlier
12-metric proposal; the [older evaluation experiments](../research-plan/evaluation-experiments.md)
remain a backlog of optional research. The [source survey](../research-plan/dataset-and-metric-research.md)
contains broader dataset research. A surveyed dataset is not a commitment to run
another benchmark.

## Scope

| Family | Capability |
|---|---|
| F1 — Direct prose | Write from a prompt and supplied context. |
| F2 — File authoring | Explore, write, or revise through tools; evaluate the saved prose. |
| F3 — Brainstorming and planning | Produce useful, meaningfully different ideas and revise plans. |
| F4 — KB construction and maintenance | Select, interpret, organize, and update knowledge in a usable reference. |
| F5 — Writing using a KB | Retrieve and apply knowledge while producing new prose. |

Markdown wikis are the first KB representation. Flat Markdown, linked pages,
structured records, and hybrid representations are experimental conditions,
not separate benchmarks. KB quality depends on purpose and intended detail:
continuity references, scene notes, and literary analysis need different content.

IFEval checks general instruction following. A proposed 32-case HumanEval+ diagnostic subset checks retention of coding ability
after training. Coding is not a
specialization objective. Astra is the selected primary semantic/prose grader;
Its basic grading transport is verified; agreement with human judgments remains
unvalidated.
Prose-distribution measurements are secondary improvement targets for less
formulaic, more varied writing, evaluated alongside prose quality and continuity.
They are not the core metric or a standalone checkpoint-selection rule.

## Boundaries

The pilot is development data. Final evaluation is held out from training,
harness tuning, model selection, and grader calibration. Source works, series,
authors, and derived branches must not leak across these roles. Public benchmarks
cannot be assumed absent from model pretraining.

Build a useful small suite before generating a large training corpus. Plans for
alternate plots, styles, and fanfiction-like branches remain in
[branching authorship](../research-plan/branching-authorship.md). No AGENTS.md or
README changes are needed for this work plan.
