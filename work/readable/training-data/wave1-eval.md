# What the first evaluation-task generation produced

This report records 24 newly authored evaluation tasks and the offline checks of whether their requirements can be satisfied. All 24 compile and load, and the run met its assigned balance targets, but it did not deliver the 60-task internal test pool or the controlled coverage required by the balance specification developed at the same time. No model execution, evaluation, training, or paid API calls occurred, so these results do not establish model performance.

## What this batch does not provide

The [balance specification](balance-spec.md) calls for 60 internal test tasks, three in each combination of five task families and four instruction-detail levels. This run produced 24 tasks, uses the two categories explicit and loose, and prescribes reply or file delivery in every case. It therefore does not provide the specification's four-level comparison or its 30 test cases where the model may choose its delivery channel (`channel_free`).

All 24 sources are newly authored, cataloged as synthetic, and assigned to one final-evaluation lineage group. A lineage group collects related source material to keep it out of other pools. This is not the specification's source mix of 30 synthetic, 20 partly synthetic, and 10 human-derived test cases. The report does not establish compliance with every other specification requirement or a completed cross-pool lineage audit.

The reported authoring work has no unresolved compiler blocker. That narrower result should not be read as a completed test suite or proof against semantic contamination: unique IDs, distinct briefs, and an overlap heuristic cannot establish that two tasks contain no shared ideas or source knowledge. Nor do metadata style labels establish diverse model output. Mechanical examples used during validation show that the constraints can coexist; they do not measure literary quality, rubric scores, or successful model behavior.

## The targets this run met

A task, also called a scenario, supplies a brief, starting files, tools, and acceptance checks. The family codes identify its main activity: F1 is direct prose; F2 is file writing or revision; F3 is planning alternatives; F4 is building a knowledge base (KB), meaning linked project-reference pages; and F5 is writing from that knowledge base. F1, F2, and F5 are prose-bearing families because they require passages or scenes.

The channel is reply or file delivery. A followup is a later user instruction. Explicit and loose distinguish how much creative direction the brief gives; loose cases leave choices open and request a `Defaults:` line declaring assumptions. These two categories do not certify the repository's L0–L3 progression, in which L0 states goal and deliverable, L1 adds setting or navigation, L2 adds applicable length/style/selection/option-count decisions, and L3 adds applicable continuity, story-fact authorization, and branch choice.

Provenance records source origin: `original` describes newly authored task material, `half_synthetic` means partly synthetic material, and the source catalog uses `synthetic` for these newly generated sources. The role `final_eval` means reserved for final evaluation.

The run exceeded its minimum genre and style counts and met every assigned task-count target. The comparison below concerns those original targets, rather than the separate specification.

| Axis | Target | Actual |
|---|---|---|
| Total | 24 | 24 |
| Family F1 | 6 | 6 |
| Family F2 | 5 | 5 |
| Family F3 | 3 | 3 |
| Family F4 | 4 | 4 |
| Family F5 | 6 | 6 |
| Prose-bearing F1/F2/F5 | at least 70% | 17/24 = 70.83% |
| Specificity explicit | 12 | 12 |
| Specificity loose | 12 | 12 |
| Reply, no tools | 6 | 6 |
| Reply, tools | 8 | 8 |
| File, tools | 10 | 10 |
| Zero followups | 12 | 12 |
| 1–2 followups | 6 | 6 |
| 8–12 followups | 6 | 6 |
| Distinct genres | at least 8 | 24 |
| Distinct styles | at least 10 | 20 |
| Scenario provenance | prefer original/half_synthetic | 24 original |
| Existing source dependence | avoid existing works | 0 existing sources; 24 newly authored sources |
| Scenario role, corrected | final_eval or development | 24 final_eval |


The six long cases are `gen-wave1-eval-002` with 8 followups, 008 with 9, 013 with 10, 016 with 11, 020 with 8, and 024 with 12. Respectively, they concern a hospital family encounter, a seed-bed dispute, first-contact alternatives at a pool, a ballot-workshop KB, an investigation at a bus depot, and a supply decision at a mountain tram station. Each includes pauses, pushback, authorized additions, and a final request for the complete artifact. A selector is the rule identifying the reply passage or file to score; these selectors resolve the final reply or final file state rather than an obsolete intermediate turn.

## What the authoring checks establish

All eight reply cases with tools expose all five file tools: `list_dir`, `read_file`, `search`, `write_file`, and `patch_file`. Their briefs require a reply and unchanged project files. All ten file cases name concrete required files and provide write tools. This breaks the earlier perfect association between reply delivery and having no tools, although it does not test whether a model chooses its own channel.

Sources use markets, hospitals, mobile courts, kitchens, salvage vessels, classrooms, factories, farms, depots, stadiums, laboratories, stations, pools, auctions, bathhouses, and mills. None uses an archive, attic, manor, or lighthouse. On loose tasks, styles in metadata guide editorial sampling; they are not hidden response requirements. The literal `Defaults:` check establishes that the line exists, not that its assumptions are thoughtful. Realized output diversity requires future model runs.

Each stated word budget has a `word_range` check. Planning case F3, case 012, bounds each separately delimited alternative at 55–85 words. F4 cases also have a `kb_word_budget` check for combined page length and require links in both directions. All 30 selectors have a required `nonempty` check, so missing delivery fails.

Continuity checks pair a requested established sentence with a tempting rejected claim. Literal exclusions detect the specified wording; semantic rubrics judge meaning and can address contradictory paraphrases. Every required mechanical check is grounded in the visible brief, followups, or accessible source files. Qualitative rubrics cover prose, alternatives, faithfulness, and the meaning of updates. Required checks use only supported deterministic kinds—checks that the code can compute without a language-model judge.

F5 cases contain 19 shift reports, a project index, a superseded proposal, and a searchable signed decision. Reading the index and then searching exposes the required answer well within budget. The `evidence_exposed` check accepts evidence from either reading or searching, so the tasks encourage search without enforcing its exclusive use. The recorded six probes each found 4 matches and used 109–116 of 3500 read tokens; the exact per-case totals remain in the command record below.

The role/source correction was applied before final validation. The actual `compile_scenarios` compiler produced the manifest, catalog, and overlap audit; no fallback compiler was needed. Every scenario has the role `final_eval`, meaning held for final evaluation, and references a real local source. `validate_catalog` computes source groups.

The catalog does not accept `original` as provenance. Following the shipped `original-orbital-laundry` and `original-puppet-theatre` records, all source records use `provenance: synthetic`. Scenario metadata uses `original` to describe new task material. Each catalog record has exactly 18 requested fields, an on-disk UTF-8 raw source, matching byte hashes and character span, local evidence of use terms, no parents, and no transformations. The shared session-author identity puts all 24 sources in one lineage group, not 24 independent author groups. The overlap heuristic reports zero pairs.

## Delivered files

Under `data/processed/gen-wave1-eval/`, there are 24 visible packages named `visible/gen-wave1-eval-001.json` through `024.json`, 24 corresponding private label packages, `manifest.json`, `catalog.json`, `overlap-audit.json`, and 24 sources matching `sources/original-wave1-eval-*.md`. That is 51 JSON files and 24 nonempty source files. The manifest holds metadata; visible/private payloads match the compiled examples exactly, and loading reconstructs full records.

The work directory contains the following supporting evidence:

- `build-wave1-eval.py` authors records reproducibly and invokes the real compiler.
- `validate-wave1-eval.py` independently checks structure, balance, obligations, extraction, the catalog, and local tools.
- `wave1-eval-balance.json` records counts, genre/style distributions, and long-case IDs.
- `wave1-eval-authoring-audit.json` records visible support for each check and constructed examples that satisfy the mechanical requirements.
- `wave1-eval-search-probes.json` summarizes real tool results; `retrieval-probes/` holds isolated copies used for read/search validation.
- `write-wave1-report.py` generates the original `wave1-eval.md` report.

No source files in `src/`, `tests/`, `data/scenarios/`, `data/training/`, or `runs/` were edited. All authored deliverables and scratch artifacts remained in the two authorized directories.

## Recorded commands and results

These commands and outputs are evidence from the original run, not new executions for this rewrite. The first validation failed because its constructed Markdown example had an unlabeled link. Correcting that test fixture, without weakening any task requirement, made the final checks pass: 24 loaded records, 30 extractable outputs, and 161 required checks with visible support.

All commands ran from `/home/jimyao/gitrepos/research/writing-assistant`. The default uv cache was read-only during initial source inspection:

```text
error: failed to open file `/home/jimyao/.cache/uv/sdists-v9/.git`: Read-only file system (os error 30)
```

A work-local cache and the existing virtual environment resolved this without dependency installation. Later commands used these exact exports:

```bash
export UV_CACHE_DIR=/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/uv-cache
export UV_NO_SYNC=1
export PYTHONDONTWRITEBYTECODE=1
```

### 1. Repository compilation

```bash
UV_CACHE_DIR=/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/uv-cache PYTHONPATH=src uv run --no-sync python3 /home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/build-wave1-eval.py
```

Exit code 0; actual output:

```text
Compiled 24 final_eval scenarios with 24 original sources
```

The script invokes `compile_scenarios(records, catalog, OUT)` directly, which runs `validate_catalog` and `overlap_audit` before writing. It does not invoke a model or evaluation runner.

### 2. Full validation, first attempt

After the exports above:

```bash
PYTHONPATH=src uv run python3 /home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/validate-wave1-eval.py
```

Exit code 1; actual output:

```text
load_scenarios: 24 records loaded; all visible/private hashes verified
JSON/nonempty: 51 JSON files parsed; 75 delivered files nonempty
Catalog: 24 sources valid; 1 author-lineage group; 0 overlap pairs; raw hashes/spans verified
Novelty: 24 unique task IDs and briefs; 0 exact collisions across 904 existing JSON files
Traceback (most recent call last):
  File "/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/validate-wave1-eval.py", line 178, in <module>
    assert graph['pages']==2 and graph['valid_fraction']==graph['reachability']==1
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

The validator constructed a witness containing the checked suffix `](state.md)` without a Markdown label. This was a fixture error, not a task-record error. The witness builder was corrected to use `[State](state.md)` and `[Home](index.md)`. No task constraint was weakened.

### 3. Full validation after fixing the fixture

```bash
PYTHONPATH=src uv run python3 /home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/validate-wave1-eval.py
```

Exit code 0; actual output:

```text
load_scenarios: 24 records loaded; all visible/private hashes verified
JSON/nonempty: 51 JSON files parsed; 75 delivered files nonempty
Catalog: 24 sources valid; 1 author-lineage group; 0 overlap pairs; raw hashes/spans verified
Novelty: 24 unique task IDs and briefs; 0 exact collisions across 904 existing JSON files
Contract: 24 records; 30 linked, requested selectors; 161 required checks with visible support
Constraint witnesses: 30 successful extractions; all required mechanical constraints jointly satisfiable
KB witnesses: 4 reachable two-page graphs; valid forward/back links; all individual and combined word caps satisfied
Retrieval gen-wave1-eval-019: read_file + search, 4 matches, 111/3500 read tokens, required fact exposed
Retrieval gen-wave1-eval-020: read_file + search, 4 matches, 109/3500 read tokens, required fact exposed
Retrieval gen-wave1-eval-021: read_file + search, 4 matches, 112/3500 read tokens, required fact exposed
Retrieval gen-wave1-eval-022: read_file + search, 4 matches, 116/3500 read tokens, required fact exposed
Retrieval gen-wave1-eval-023: read_file + search, 4 matches, 112/3500 read tokens, required fact exposed
Retrieval gen-wave1-eval-024: read_file + search, 4 matches, 111/3500 read tokens, required fact exposed
Family: {'F1': 6, 'F2': 5, 'F3': 3, 'F4': 4, 'F5': 6}
Specificity: {'explicit': 12, 'loose': 12}
Channel/tools: {'reply/no_tools': 6, 'file/tools': 10, 'reply/tools': 8}
Followups: {'0': 12, '8-12': 6, '1-2': 6}
Diversity: 24 genres; 20 styles; 17/24 prose-bearing (70.83%); 24 original scenarios
Deep cases: {'gen-wave1-eval-002': 8, 'gen-wave1-eval-008': 9, 'gen-wave1-eval-013': 10, 'gen-wave1-eval-016': 11, 'gen-wave1-eval-020': 8, 'gen-wave1-eval-024': 12}
VALIDATION COMPLETE: no model, training, evaluation, or paid API calls
```

### 4. Requested recursive JSON parse

```bash
PYTHONPATH=src uv run python3 - <<'PY'
import json, glob
for f in sorted(glob.glob('data/processed/gen-wave1-eval/**/*.json', recursive=True)):
    json.load(open(f))
PY
```

Exit code 0; no stdout or stderr.

### 5. Missing-delivery guards

```bash
PYTHONPATH=src uv run python3 - <<'PY'
from pathlib import Path
from writing_agent.artifacts import extract_prose
from writing_agent.suite import load_scenarios
records = load_scenarios(Path('data/processed/gen-wave1-eval'))
selectors = 0
for record in records:
    for selector in record['visible']['prose']:
        assert any(c['required'] and c['kind']=='nonempty' and c['artifact']==selector['id'] for c in record['labels']['checks'])
        missing = extract_prose({'output':'','after':{}}, [selector])
        assert missing[0]['status']=='missing_prose'
        selectors += 1
print(f'Delivery guards: {selectors} selectors have required nonempty checks and reject missing delivery')
PY
```

Exit code 0; actual output:

```text
Delivery guards: 30 selectors have required nonempty checks and reject missing delivery
```

The first failed validation is retained above. All final validations completed successfully. No compiler rule remains unsatisfied.
