# Research evaluation

[evaluate.py](../scripts/evaluate.py) is the editable experiment entry point. Its
default invocation inspects the proposed workload and prints a JSON manifest:

```bash
uv run python scripts/evaluate.py
```

It does not download data, load models, or generate candidate output by default.
Source selections, provenance filters, model configurations, feature settings, and
stages are ordinary Python values near the top of the script.

## Preparation and saved-output analysis

Install the small data and lexical-metric dependencies:

```bash
uv sync --extra data --extra metrics
```

Stages can also be called from Python:

```python
from scripts.evaluate import prepare, score_saved, report

inventory = prepare(download=True)
cards = score_saved()  # Mechanical checks and cheap prose features on saved results.
summary = report()
```

Raw downloads and receipts live under `data/raw/research/`. The shared catalog,
compiled visible packages, private labels, and overlap audit are under
`data/processed/custom-eval/`. These derived directories are git-ignored.
The authored scenarios and worlds remain in [data/scenarios/](../data/scenarios/worlds.json).
Preparation preserves human review fields when scenario content has not changed;
changed content retains the previous review and returns to pending status.

The API is exposed by concern rather than a second facade:

| Operation | Python function |
|---|---|
| Import a local story | `writing_agent.catalog.import_story` |
| Validate/filter provenance and roles | `validate_catalog`, `select_sources` in the same module |
| Compile/load scenarios | `writing_agent.suite.compile_scenarios`, `load_scenarios` |
| Run an explicit selection | `writing_agent.suite.run_selected` |
| Copy a generated KB / run fresh-reader probes | `kb_use_scenario`, `run_navigation` in the same module |
| Extract designated prose | `writing_agent.artifacts.extract_prose` |
| Score saved artifacts | `writing_agent.scoring.mechanical_score` |
| Cache features / compare prose | `ProseFeatures`, `prose_profile`, `compare_groups` in `writing_agent.prose` |
| Prepare/obtain/apply judgments | `grading_packet`, `CodexGrader`, `apply_judgment` in `writing_agent.grading` |
| Build reports | `writing_agent.scoring.build_report` |

Library functions return records without printing or exiting. Model backends are
supplied by the caller. `run_selected` defaults to planning only; execution requires
`execute=True`. Failed attempts remain saved; `retry_failed=True` creates another
attempt in a clean workspace. The runner assumes one serial writer per destination.

## Metrics and review

The [scorecard](../work/custom-eval-suite/metrics.md) and
[prose catalog](../work/custom-eval-suite/prose-metrics.md) define the measurements.
Numerical prose features consume explicit reply/file selections. A mixed reply
needs delimiters or reviewed character spans; a local revision scores its declared
passage rather than the unchanged manuscript.

Select human reference IDs from the shared catalog in `REFERENCE_IDS`. Set a frozen
`MMD_BANDWIDTH` after inspecting development reference embeddings. `compare_groups`
can group by family, condition, provenance, style, and length. It records reference
IDs and sample counts. A missing reference or model feature produces an unavailable
status. The initial collection has no accepted matched reference selection yet.

Optional model dependencies are available with `uv sync --extra models`; this may
install substantial packages but does not download checkpoint weights. Feature
methods load local weights only unless explicitly passed `allow_download=True`.
The evaluation tokenizer and embedding revisions are pinned in `FeatureConfig`.
Embeddings use 256-token chunks, token-count-weighted mean pooling, and normalization.
The default embedding implementation uses CPU separately from candidate generation.

`paired_similarity` computes ROUGE for an explicitly paired task. BERTScore requires
a locally provisioned compatible model and an explicit revision, and remains unavailable otherwise. MAUVE,
perplexity, and author retention have explicit prerequisite statuses. None silently
substitutes for a core prose-quality measure.

Astra grading uses the installed `codex exec` with existing authentication and a
saved JSON schema. The default grader allows three persisted attempts, including
failures. It does not fall back to an API key. Packets omit candidate model identity,
and results retain evidence, explanations, uncertainty, and human-review status.
The [review packet](../work/custom-eval-suite/review/review.md) contains scenario
briefs and private draft labels. Semantic scores are unvalidated until reviewed.

## Candidate execution

The script requires both explicit run authorization and a model configuration
marked runtime-verified. The default backend runs Transformers/PyTorch locally.
See [local inference](local-inference.md) for installation, checkpoint and PEFT
loading, the native tool protocol, and evaluation during future training. Before
marking a configuration verified, check its exact revision, quantization, context
budget, prompt formatting, and tool behavior on the remote machine.
No Gemma benchmark comparison has been run.

The proposed full comparison and separately budgeted probes, repeated generations,
and external checks are in [experiment.json](../work/custom-eval-suite/experiment.json).
The execution approval boundary is recorded in the [delivery plan](../work/custom-eval-suite/plan.md).

## Checks

```bash
uv run --extra data --extra metrics python -m unittest discover -s tests -v
uv run ruff check src scripts tests
```

These tests use synthetic fixtures and scripted backends. They do not establish
model quality, serving speed, or agreement between Astra and human reviewers.

Saved pilots can be rescored without candidate generation using
`scripts.rescore_pilot.run(Path("runs/<pilot-directory>"))`. This writes a separate
`rescored/` directory with all Q/R/D statuses per task, feature caches, selected
references, and an exploratory pooled comparison. Pass `allow_download=True`
explicitly if pinned metric weights are not cached. The pilot rescoring script
uses unreviewed Gutenberg paragraphs; it does not claim genre/style matching.
Semantic judgments remain pending unless the separate grading stage is run.

For both broad and task-selected comparisons, pass
`matched_manifest=Path("data/references/pilot-matched-v1.json")` to the pilot
rescoring function. It writes `rescored-dual/` and records both tracks per task.
See the [reference comparison report](../work/custom-eval-suite/reference-comparisons.md)
for selection limits and reproducible settings.
