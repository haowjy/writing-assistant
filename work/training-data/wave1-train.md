# Wave 1 training task generation

48 training scenarios compile and load without error from `data/processed/gen-wave1-train/`. All numeric balance targets are met. No paid calls, model execution, training, or evaluation ran.

## Target versus actual

| Axis | Target | Actual |
|---|---|---|
| Records | 48 | 48 |
| Family | F1 14 / F2 10 / F3 6 / F4 8 / F5 10 | 14 / 10 / 6 / 8 / 10 |
| Prose-bearing families F1/F2/F5 | At least 70% | 34/48 = 70.83% |
| Specificity | explicit 24 / loose 24 | 24 / 24 |
| Reply, no tools | 8 | 8 |
| Reply, with tools | 16 | 16, including 6 offering write tools |
| File, with tools | 24 | 24 |
| Read tools only | At least 6 | 10 |
| Zero followups | 20 | 20 |
| 1–2 followups | 14 | 14 |
| 8–12 followups | At least 14 | 14, each with 9 |
| REQUIRES_ASK | 12 | 12 |
| PERMITS_DEFAULT | 24 | 24 |
| MUST_NOT_ASK | 12 | 12 |
| Steer kinds | All nine, roughly even | All nine, 16–18 occurrences each |
| Genres | At least 12 catalog values | 40 |
| Styles | At least 14 catalog values | 20 |
| Tropes | Catalog values; no trope in more than 4 cases | 40 distinct; maximum frequency 2 |
| Situations | Catalog values | All 32 |
| Word budgets | Vary across all five requested ranges | All five used; artifact counts below |
| Source originality | Prefer original material; avoid dependence on existing works | 48 newly authored sources; no existing source IDs reused |
| Provenance preference | Prefer original / half_synthetic | `synthetic` for all 48, matching the actual catalog enum and shipped original-source precedent; see decision below |
| Role | train | train for all scenarios and sources |

Steer counts: ADD_CONSTRAINT 17; CHANGE_DIRECTION 16; PUSH_BACK 16; CLARIFY_ANSWER 16; REQUEST_REVISION 18; NEW_MATERIAL 17; SCOPE_CUT 16; APPROVE 16; INTERRUPT 16. The tags are metadata only; model-visible followups are plain strings.

Word-range counts are per selected artifact: 80–150: 26; 120–220: 18; 250–400: 10; 600–900: 10; 900–1200: 12. There are 76 artifacts because F3 selects three alternative files per case and F4 selects three KB pages per case. Every declared word range has a required `word_range` check. F4 additionally caps the whole KB at 520 words, compatible with the three page maxima of 150 + 220 + 150.

## Decisions and quality controls

- Preserved the actual compiled layout: `visible/<id>.json` contains exactly the six model-visible fields; `private/<id>.json` contains labels. The manifest carries metadata, source groups, and hashes. `scenarios.json` retains the complete records for recompilation.
- Applied the corrected lineage requirement. Each task references one original source in the wave's own catalog. Every source contains all 18 requested catalog fields, a real raw file, verified raw/text hashes, and a recoverable character span. The compiler generated the source groups. All sources truthfully share the session author's identity, so they form one train-only lineage group; different work IDs are not presented as different authors.
- Used `synthetic` provenance because `writing_agent.catalog.PROVENANCE` rejects `original`. The shipped `original-orbital-laundry` record also uses `synthetic`. “Original” here describes new authorship, not a legal provenance enum. No claim of human or half-synthetic authorship is made.
- F1 keeps the passage-to-reply task shape. Six F1 cases intentionally offer all five file tools and explicitly prohibit workspace edits, as needed for the requested reply/tools cell. The other ten reply/tools cases are F5 retrieval tasks with read tools only. This necessarily broadens the shared-context shorthand that all F1 tasks have no tools; the requested numeric targets and the real compiler permit this design.
- Each REQUIRES_ASK case presents a consequential unresolved branch, and its first followup resolves it. The author can ask or explicitly mark a provisional assumption. PERMITS_DEFAULT cases explicitly invite a stated reversible default. MUST_NOT_ASK cases delegate remaining creative choices and request immediate work.
- Every deep chain has nine authored, case-specific steers. They add material, redirect action, limit scope, interrupt expansion, and approve a final pass. They do not assume an unseen assistant error. All declared artifacts refer to the final reply or final snapshot, so an initial clarification does not cause missing-prose failure. Earlier turns remain available for rubric assessment of collaboration behavior.
- Every artifact has a required positive delivery check. All 324 checks carry valid artifact IDs. Required quoted or protected strings come from accessible files, not from leaked literals in briefs. Continuity prohibitions name specific false assertions; rubrics also assess their meaning and accepted later changes.
- F2 uses genuinely unfinished drafts and requires preserving an existing sentence. F3 requires three developed alternatives with different causal consequences. F4 has substantial source material, three explicitly required pages, cross-links, word caps, and a file allowlist.
- F5 includes an active instruction, a withdrawn version, and twelve irrelevant documents. The brief explicitly requests search and a context read. Both relevant versions can be found and read with the real tools for 183–197 of the 1200 available whitespace-v1 read tokens; reading everything exceeds that budget. The scene must quote the current instruction and exclude the withdrawn one. Reply-only cases cannot satisfy delivery by saving a file.
- Binary specificity is the requested explicit/loose axis. Explicit briefs fix narration and ending guidance; loose briefs delegate those creative decisions. These are not claimed to be a balanced L0–L3 experiment: even loose tasks necessarily state delivery, length, and source boundaries. The `specificity.py` distinction between consequential branch choices and reversible defaults informed the pressure assignments.

## Files created

Under `data/processed/gen-wave1-train/`:

- `visible/wave1-train-001.json` through `wave1-train-048.json`.
- Matching 48 files in `private/`.
- `scenarios.json`, `catalog.json`, `manifest.json`, and `overlap-audit.json`.
- 48 source text files under `sources/`.

The directory contains **148 nonempty files, including 100 parseable JSON files**. It is ignored by the repository's existing Git rules; the deliverables exist on disk and were not committed.

Work artifacts alongside this report: `seeds.json`, `turns.json`, `slots.json`, `kb_material.json`, `build_wave.py`, `validate_wave.py`, `authoring-audit.json`, `reachability-audit.json`, and `validation-summary.json`. Vocabulary stays in JSON, combination logic in Python, and authored prose in JSON slots. No changes were made to `src/`, `tests/`, `data/scenarios/`, `data/training/`, or `runs/`.

## Exact commands and observed results

Commands ran from `/home/jimyao/gitrepos/research/writing-assistant`. `UV_CACHE_DIR` points into the permitted work directory because the default uv cache is read-only in this sandbox.

### Build using the actual compiler

```bash
UV_CACHE_DIR=/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/.uv-cache PYTHONPATH=src uv run python3 /home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/build_wave.py
```

Output, exit 0:

```text
Compiled 48 scenarios and 48 original sources into data/processed/gen-wave1-train
```

This command was run twice, with the same output; the second build incorporated expanded KB source material.

### Compiler, loader, contract, extraction, workspace, retrieval, and balance validation

```bash
UV_CACHE_DIR=/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/.uv-cache PYTHONPATH=src uv run python3 /home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/validate_wave.py
```

Output, exit 0:

```text
Catalog: 48 valid original sources; raw/text hashes and character spans verified; 0 near-duplicate pairs
Compiler/load round trip: 48 scenarios; 48 visible/private package pairs; hashes match
Identity: 48 unique IDs and briefs; no collision with existing compiled scenario IDs or briefs
Contract: 324 checks have legal kinds/parameters and matching artifact IDs; required keys, safe paths, positive delivery checks and literal non-leakage verified
Extraction/feasibility: 76 final artifacts extract; all are missing on empty output; all requested file paths are writable; fixed strings fit word budgets
KB feasibility: 8 three-page graphs have valid links, complete reachability, and compatible page/total word budgets
Retrieval feasibility: 10 real search/read workflows find active and withdrawn records; cost 183-197 of 1200 whitespace-v1 read tokens; exhaustive reads exceed budget
Balance: {"channel_tools": {"file/tools": 24, "reply/no_tools": 8, "reply/tools": 16}, "family": {"F1": 14, "F2": 10, "F3": 6, "F4": 8, "F5": 10}, "followup_depth": {"0": 20, "1-2": 14, "8-12": 14}, "genres": 40, "max_trope_frequency": 2, "pressure": {"MUST_NOT_ASK": 12, "PERMITS_DEFAULT": 24, "REQUIRES_ASK": 12}, "prose_bearing_cases": 34, "read_only_cases": 10, "reply_with_write_tools": 6, "scenario_provenance": {"synthetic": 48}, "situations": 32, "source_provenance": {"synthetic": 48}, "specificity": {"explicit": 24, "loose": 24}, "steer_kinds": {"ADD_CONSTRAINT": 17, "APPROVE": 16, "CHANGE_DIRECTION": 16, "CLARIFY_ANSWER": 16, "INTERRUPT": 16, "NEW_MATERIAL": 17, "PUSH_BACK": 16, "REQUEST_REVISION": 18, "SCOPE_CUT": 16}, "styles": 20, "tropes": 40, "word_ranges_per_artifact": {"120-220": 18, "250-400": 10, "600-900": 10, "80-150": 26, "900-1200": 12}}
Offline validation complete. No model calls, training, or evaluation executed.
```

The validator source is retained so the asserted contract and reachability checks are inspectable. It invokes `compile_scenarios`, `load_scenarios`, `validate_catalog`, `overlap_audit`, `extract_prose`, `markdown_graph`, and actual workspace tool dispatch. Temporary workspaces stay under the allowed work path and are removed. Structural placeholder text checks extraction and budget compatibility; it is neither literary validation nor a training target. No agent backend or scorer evaluation was run.

### Required JSON parsing command

```bash
export UV_CACHE_DIR=/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/.uv-cache
PYTHONPATH=src uv run python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("data/processed/gen-wave1-train/**/*.json", recursive=True)):
    json.load(open(f))
PY
```

Output: no stdout or stderr; exit 0.

### Nonempty-file and final-load verification

```bash
export UV_CACHE_DIR=/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/.uv-cache
PYTHONPATH=src uv run python3 - <<'PY'
import json
from pathlib import Path
from writing_agent.suite import load_scenarios
root=Path('data/processed/gen-wave1-train')
files=sorted(p for p in root.rglob('*') if p.is_file())
assert files and all(p.stat().st_size > 0 for p in files)
json_files=[p for p in files if p.suffix == '.json']
for p in json_files:
    json.loads(p.read_text())
records=load_scenarios(root)
assert len(records) == 48
print(f'{len(files)} nonempty files; {len(json_files)} JSON files parsed; {len(records)} records loaded without error')
PY
git check-ignore data/processed/gen-wave1-train/manifest.json
```

Output, exit 0:

```text
148 nonempty files; 100 JSON files parsed; 48 records loaded without error
data/processed/gen-wave1-train/manifest.json
```

## Limits and blockers

No compiler rule remains blocked, and no numeric target was missed. The preferred `original` provenance spelling is not legal in the real source validator; newly authored sources therefore use the existing `synthetic` convention.

Satisfiability was reviewed against the authored source, brief, and followup sequence; the validator records per-check enabling evidence in `reachability-audit.json`. Mechanical feasibility does not prove that a model will write good prose, preserve all semantic distinctions, or choose to search. Literal exclusion checks cannot catch every paraphrase of a forbidden assertion, so continuity and collaboration rubrics remain necessary. Records retain `review_status: generated`; no independent model review or measured policy improvement is claimed.

`task_authoring.validate_task()` was inspected, but it validates a different author-generation packet containing evidence, branch contracts, stage families, and restricted generated-check kinds. It is not the loader for these scenario records. The actual scenario compiler, catalog validator, and loader were run successfully instead.

### Work-artifact integrity

The work directory is shared with other lanes. An initial directory-wide inventory reported 37 nonempty files; its fixed JSON/script-count labels were not a reliable count of that shared directory. The final check below explicitly scopes the ten files authored for this wave and computes all counts.

```bash
export UV_CACHE_DIR=/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation/.uv-cache
PYTHONPATH=src uv run python3 - <<'PY'
import ast, json
from pathlib import Path
work=Path('/home/jimyao/.meridian/projects/orange-juniper-leaf/spawns/p34/work/data-generation')
names=['seeds.json','turns.json','slots.json','kb_material.json','authoring-audit.json','reachability-audit.json','validation-summary.json','build_wave.py','validate_wave.py','wave1-train.md']
files=[work/name for name in names]
assert all(p.is_file() and p.stat().st_size > 0 for p in files)
json_files=[p for p in files if p.suffix == '.json']
python_files=[p for p in files if p.suffix == '.py']
for p in json_files:
    json.loads(p.read_text())
for p in python_files:
    ast.parse(p.read_text())
print(f'Wave work artifacts: {len(files)} nonempty files; {len(json_files)} JSON files parsed; {len(python_files)} Python scripts parsed; report present')
PY
```

Output, exit 0:

```text
Wave work artifacts: 10 nonempty files; 7 JSON files parsed; 2 Python scripts parsed; report present
```
