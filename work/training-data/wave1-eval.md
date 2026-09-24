# Wave 1 evaluation task samples

24 scenarios compile with the repository compiler and load with `load_scenarios`. All requested balance targets are satisfied. There are no outstanding blockers. No model execution, evaluation, training, or paid API calls were made.

## Balance: target versus actual

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

Deep cases are 002 (8 followups), 008 (9), 013 (10), 016 (11), 020 (8), and 024 (12). The cases develop, respectively, a hospital family encounter, a seed-bed dispute, first-contact alternatives at a pool, a ballot-workshop knowledge base, an investigation at a bus depot, and a supply decision at a mountain tram station. Each includes pauses, pushback, authorized additions, and a final request for the complete artifact. Selectors resolve the final reply or final file state, rather than an obsolete intermediate turn.

## Authoring decisions and limits

- All eight tool-enabled reply cases expose all five tools, including write tools. Their briefs explicitly request reply delivery and unchanged project files. The ten file cases require concrete named files and have write tools. This breaks the channel/tool availability confound.
- Sources span markets, hospitals, mobile courts, kitchens, salvage vessels, classrooms, factories, farms, depots, stadiums, laboratories, stations, pools, auctions, bathhouses and mills. No archive, attic, manor or lighthouse setting occurs.
- Loose cases leave creative choices open and explicitly request a `Defaults:` line. The literal check detects whether that line exists; it does not prove the defaults are thoughtful. Metadata styles on loose tasks are editorial sampling labels, not hidden response requirements. The wave makes no claim about realized output diversity without future model runs.
- Stated word budgets have `word_range` checks. F3 case 012 has separate 55–85-word bounds for each delimited alternative. F4 cases also have combined `kb_word_budget` checks and required forward/back links. All 30 selectors have a required nonempty guard.
- Continuity checks pair an explicitly requested established sentence with a tempting rejected claim. Semantic continuity rubrics cover contradictory paraphrases; literal exclusions alone cannot do that. Every required mechanical check is grounded in the visible brief, followups or accessible source files.
- F5 cases contain 19 shift reports, a project index, a superseded proposal, and a searchable signed decision. A read of the index followed by search exposes the answer well within budget. `evidence_exposed` accepts either read or search; these cases encourage search but do not falsely claim to enforce its exclusive use.
- Qualitative rubrics cover prose, alternatives, faithfulness and update meaning. Only supported deterministic check kinds appear in required checks. Offline constraint witnesses establish mechanical satisfiability, not literary quality, rubric scores or successful candidate behavior.

The correction to role and source identity was applied before final validation. `compile_scenarios` produced the final manifest, catalog and overlap audit; no fallback compiler was necessary. Scenario role is `final_eval`, every scenario references a real local source, and source groups are computed by `validate_catalog`.

The catalog provenance vocabulary does not include `original`. Mirroring the shipped `original-orbital-laundry` and `original-puppet-theatre` records, newly authored source records use `provenance: synthetic`. Scenario metadata uses the requested `original` label for novel task material. Each catalog record has exactly the 18 requested fields, an on-disk UTF-8 raw source, matching byte hashes and character span, local terms evidence, no parents, and no transformations. A shared session-author identity truthfully places all 24 sources in one final-evaluation lineage group; these are not 24 independent author groups. The overlap heuristic reports no pairs. Exact ID/brief reuse checks are not proof against semantic contamination.

## Files created

Under `data/processed/gen-wave1-eval/`:

- `visible/gen-wave1-eval-001.json` through `024.json`: 24 model-visible packages.
- `private/gen-wave1-eval-001.json` through `024.json`: 24 label packages.
- `manifest.json`, `catalog.json`, `overlap-audit.json`: repository-compiled metadata and lineage.
- `sources/original-wave1-eval-*.md`: 24 original source documents.

That is 51 JSON files and 24 nonempty source files. Metadata is in the manifest; visible/private payloads mirror the compiled examples exactly. Loading reconstructs the complete records.

In this report's work directory:

- `build-wave1-eval.py`: reproducible authoring script invoking the real compiler.
- `validate-wave1-eval.py`: independent structural, balance, obligation, extraction, catalog and local-tool validation.
- `wave1-eval-balance.json`: actual counts, genre/style distributions and deep-case IDs.
- `wave1-eval-authoring-audit.json`: per-check grounding and constructive mechanical witnesses.
- `wave1-eval-search-probes.json`: real local tool results summarized by case.
- `retrieval-probes/`: isolated copies used for local read/search validation.
- `write-wave1-report.py` and `wave1-eval.md`: report generator and this report.

No `src/`, `tests/`, `data/scenarios/`, `data/training/` or `runs/` source files were edited. All authored deliverables and scratch artifacts are within the two authorized directories.

## Validation commands and actual results

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
