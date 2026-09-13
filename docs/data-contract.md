# Trajectory contract, version 1

One JSON object per line:

| Field | Meaning |
|---|---|
| `schema_version` | Integer `1` |
| `id` | Unique example identifier |
| `split` | `train`, `validation`, or `test` |
| `review_status` | `pending`, `accepted`, or `rejected` |
| `provenance` | Nonempty `source`, `license`, `author_id`, and `work_id` strings |
| `initial_files` | Relative path to UTF-8 text before the conversation |
| `messages` | Ordered system/user/assistant/tool messages |
| `tools` | Function tool schemas available to the model |
| `expected_files` | Expected final files for replay validation |

The format example lives in `data/fixtures/trajectories.jsonl`. It is synthetic and
pending review, and is distinct from the evaluation task fixtures.

Tool calls use an `id`, `type: function`, and a `function` containing `name` and
object-valued `arguments`. Tool observations carry the matching `tool_call_id` and
string `content`. Every call must have an observation before conversation resumes.
HTTP traces may contain string-encoded arguments; normalize them before importing
them as training records. Trace logs are not automatically approved training data.

Validation checks IDs, required metadata, basic message/tool sequencing, relative
file paths, and author/work disjointness across splits **within the supplied file**.
Validate the combined manifest before splitting it into separate files. Format
validation does not verify licensing claims, historical accuracy, target leakage,
tool-schema semantics, or that `expected_files` actually result from replay.
Those are required review/replay checks before marking a record accepted.

For story-state terminology and acceptance concerns, see the
[authoring workspace](../wiki/authoring-workspace.md).

## Export

The exporter selects records with `split: train` and `review_status: accepted`,
then writes only `messages` and `tools`. File snapshots and provenance stay in the
source dataset. Tool observations in `messages` supply the model with file contents.

The exporter does not implement training or loss masking. The proposed supervision
setup is described in [training experiments](../work/research-plan/training-experiments.md).
