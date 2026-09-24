# What the first training-task generation produced

This report records the 48 training tasks produced in the first generation run and compares them with the targets that run was given. All 48 compile and load, and all of the run's numeric targets were met; the run did not meet the broader balance specification, which was developed concurrently and derived additional requirements. It also did not establish that a model can solve these tasks or improve through training: no model execution, paid calls, training, or evaluation ran.

## What this batch does not provide

The [balance specification](balance-spec.md) proposes 300 training tasks, four controlled levels of instruction detail, tasks where the model chooses between replying and saving a file, and a mix of human, partly synthetic, and fully synthetic sources. This batch contains 48 tasks, uses only the two categories explicit and loose, and always specifies reply or file delivery. All 48 sources are newly authored synthetic material in one training-only lineage group, rather than the specification's minimum of six training lineage groups. A lineage group collects related source material so that it cannot leak across training and evaluation through derivatives or shared authorship.

The batch uses 40 genres, while the specification names 60 and requires repeated coverage. Meeting this run's requirement of at least 12 genres therefore does not establish the specification's genre coverage. Likewise, meeting the run's five word-range targets does not establish the specification's different four budget categories. These are differences between two sets of requirements, not failed compiler checks. Other specification requirements were not comprehensively assessed by this report.

Mechanical validation shows that the requested files, word limits, links, and retrieval operations can work together. It does not show good prose, correct interpretation of every story fact, or a model's willingness to search. The records remain `review_status: generated`; no independent model review or measured policy improvement is claimed.

## The targets this run met

A task, also called a scenario, gives the model a brief, starting files, tools, and checks for acceptable work. The five task families are F1, direct prose in the reply; F2, writing or revising files; F3, planning alternatives; F4, building a knowledge base (KB), meaning linked project-reference pages; and F5, writing from facts in that knowledge base. F1, F2, and F5 require scenes or passages, so they count as prose-bearing tasks.

The delivery channel is where the answer must appear: in the reply or in files. The five file tools are `list_dir`, `read_file`, `search`, `write_file`, and `patch_file`; read-only cases omit the writing tools. A followup is a later user instruction, also called a steer. An artifact is one selected output, such as a reply passage or a required file.

Explicit briefs fix narration and ending guidance; loose briefs delegate those creative decisions. Both still state delivery, length, and source boundaries. They are not a balanced experiment across the repository's L0–L3 levels, which progressively state more decision points: L0 states the goal and deliverable; L1 adds setting or navigation; L2 adds applicable length, style, selection, or option-count decisions; L3 adds applicable continuity, story-fact authorization, and branch choice.

The pressure labels describe when to ask: `REQUIRES_ASK` presents an unresolved consequential branch, `PERMITS_DEFAULT` invites a stated reversible assumption, and `MUST_NOT_ASK` delegates remaining choices and requests immediate work. The authoring rule for `REQUIRES_ASK` also permits an explicitly provisional assumption; that allowance is preserved here, although it makes the label less strict than its name suggests.

Provenance records where source material came from. `synthetic` means newly generated material here; a catalog enum is the set of spellings the source validator accepts.

The table compares the run's actual counts with its assigned targets. It shows complete numeric compliance with those targets, including 34/48 prose-bearing tasks (70.83%) and 14 long conversations; it does not claim compliance with the separate specification.

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


All nine kinds of steering appear 16–18 times. `ADD_CONSTRAINT` adds a requirement (17); `CHANGE_DIRECTION` redirects the work (16); `PUSH_BACK` challenges it (16); `CLARIFY_ANSWER` answers an uncertainty (16); `REQUEST_REVISION` asks for changes (18); `NEW_MATERIAL` supplies material (17); `SCOPE_CUT` reduces the work (16); `APPROVE` approves it (16); and `INTERRUPT` interrupts it (16). These tags are metadata; the model sees ordinary followup strings.

Word ranges are counted per selected artifact, not per task: 80–150 words occurs 26 times; 120–220 occurs 18 times; 250–400 occurs 10 times; 600–900 occurs 10 times; and 900–1200 occurs 12 times. The 48 tasks request 76 artifacts because each F3 case selects three alternative files and each F4 case selects three KB pages. Every stated range has a required `word_range` check, which counts words against the range. F4 also caps the entire KB at 520 words, equal to the three page maxima of 150 + 220 + 150.

## How the tasks and checks were constructed

The compiled format was preserved. Each `visible/<id>.json` has exactly the six model-visible fields, each `private/<id>.json` holds scoring labels, and the manifest records metadata, source groups, and hashes. `scenarios.json` retains complete records for recompilation.

Each task references one original source in this wave's own catalog. All sources have the 18 requested catalog fields, a real raw file, verified raw/text hashes, and a recoverable character span. The compiler generated their source groups. All share the session author's identity and therefore form one train-only lineage group; different work IDs do not establish different authors. The validator's provenance vocabulary, `writing_agent.catalog.PROVENANCE`, rejects `original`, so the sources use `synthetic`, as the shipped `original-orbital-laundry` source does. Here, “original” means newly authored. It is not a legal catalog value, and no human or half-synthetic authorship is claimed.

F1 retains passage delivery in the reply. Six F1 cases offer all five file tools but explicitly prohibit workspace edits. The other ten reply-with-tools cases are F5 retrieval tasks with read tools only. Thus a reply requirement no longer always means that tools are absent. This broadens the earlier shorthand that all F1 cases have no tools; the requested counts and actual compiler permit it. Every channel is still prescribed, so this is not a test of free channel choice.

For each consequential unresolved branch, the first followup supplies the answer. Default-permitted cases explicitly invite a stated reversible choice, while no-question cases request immediate work. The distinction in `specificity.py` between consequential branch choices and reversible defaults informed these pressure assignments.

Every long conversation has nine authored, case-specific steers. They add material, redirect action, limit scope, interrupt expansion, and approve a final pass without assuming an unseen assistant error. Output selectors—the rules identifying which passage or file is scored—point to the final reply or final workspace snapshot. An initial clarification therefore does not fail the missing-prose check; earlier turns remain available for judging collaboration.

Every artifact has a required check that something was delivered. All 324 checks refer to valid artifact IDs. Strings that must be quoted or preserved occur in accessible files, rather than being leaked into the brief. Continuity prohibitions identify specific false assertions, while rubrics also assess their meaning and later authorized changes. Literal exclusions cannot catch every paraphrase, which is why continuity and collaboration rubrics remain necessary. The source, brief, and followup sequence were reviewed together for satisfiability; `reachability-audit.json` records evidence that enables each check.

The family-specific constructions exercise different skills:

- F2 starts with genuinely unfinished drafts and requires preserving an existing sentence.
- F3 requires three developed alternatives with different causal consequences.
- F4 provides substantial source material and requires three pages, cross-links, word caps, and edits restricted to an explicit file allowlist.
- F5 provides an active instruction, a withdrawn version, and twelve irrelevant documents. The brief requests search and a context read. Real tools find and read both relevant versions for 183–197 of the available 1200 `whitespace-v1` read tokens, the harness's whitespace-based read accounting. Reading everything exceeds that budget. The scene must quote the current instruction and exclude the withdrawn one; saving a file cannot satisfy reply-only delivery.

`task_authoring.validate_task()` was inspected but not used as the scenario loader. It validates a different author-generation packet, with evidence, branch contracts, stage families, and restricted generated-check kinds. These records passed the actual scenario compiler, catalog validator, and loader; no compiler rule remains blocked.

## Delivered files

`data/processed/gen-wave1-train/` contains `visible/wave1-train-001.json` through `wave1-train-048.json`, 48 matching private files, `scenarios.json`, `catalog.json`, `manifest.json`, `overlap-audit.json`, and 48 source texts under `sources/`. There are 148 nonempty files, including 100 parseable JSON files. Existing Git rules ignore this directory; the deliverables existed on disk and were not committed.

The work artifacts are `seeds.json`, `turns.json`, `slots.json`, `kb_material.json`, `build_wave.py`, `validate_wave.py`, `authoring-audit.json`, `reachability-audit.json`, `validation-summary.json`, and the original report. Vocabulary is in JSON, combination logic in Python, and authored prose in JSON slots. No changes were made to `src/`, `tests/`, `data/scenarios/`, `data/training/`, or `runs/`.

## Recorded commands and results

These are the original run's commands and outputs, retained as evidence; the editorial rewrite did not rerun them. A compiler assembles scenario packages, a loader reads them back, extraction selects the requested output, and the workspace checks exercise real file tools. Placeholder outputs establish that extraction and word budgets are mechanically possible; they are neither literary validation nor training examples.

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

## Recorded work-file integrity check

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
