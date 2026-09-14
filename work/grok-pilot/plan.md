# Grok / OpenCode comparison pilot

[Completed five-case results](results.md).

Run `xai/grok-4.6` through the existing OpenCode xAI login on five frozen custom-suite
cases: F1-01, F2-06, F3-03, F4-08, and F5-05. This compares the Grok/OpenCode system
with the saved Gemma/custom-harness outputs. It does not isolate model capability
from harness behavior. No OpenCode Go or direct paid xAI API fallback is used.

[Research script](../../scripts/pilot_grok.py) defaults to inspection. Explicit
`run(execute=True, grade=True)` runs generation and up to five isolated Astra grading
sessions. Existing results and judgments are reused. An interrupted generation
requires inspection rather than an automatic retry.

## Conditions and artifacts

Use the frozen custom50 selection, including the KB task's two scripted follow-ups.
Each case starts in a temporary workspace outside the repository, containing only
its visible source files. Follow-ups resume that case's OpenCode session. Private
labels and grading artifacts remain outside candidate workspaces.

A custom primary writer uses the research harness's writing instructions.
OpenCode supplies native tool schemas and handles Grok's protocol and reasoning
continuation. File cases allow read, edit/write, list/glob, and grep operations;
other cases deny tools. Shell, delegation, web tools, and access outside the
workspace are denied. Project configuration, external skills, Claude instruction
loading, and external plugins are disabled. Reads and edits default to deny with explicit path patterns for notes/, source/,
kb/, and drafts/. This uses application permissions, not an operating-system sandbox.

Save configuration, launch commands, native events, stderr, per-turn file snapshots,
final files, readable conversations, separate reasoning, and scorecards under
`runs/pilot-grok46-opencode-2026-09-14/`. Credentials remain in OpenCode's login store
and are not copied into research artifacts.

## Scoring and comparison limits

Reuse prose extraction, deterministic artifact checks, basic numerical profiles,
and the same Astra rubric. Astra is the primary subjective evaluator; human
calibration is optional, not a prerequisite for this experiment. The existing
`unvalidated` metadata means agreement with human ratings has not been established.

Native read/grep observations are mapped into the existing evidence-exposure check;
original native tool names and events remain saved. Tool success describes
OpenCode execution, not validation by the custom tool dispatcher.

OpenCode uses a 12-step limit per user turn and a 300-second subprocess timeout.
The custom loop's cumulative tool, read-token, and storage budgets are not enforced
by OpenCode. Provider generation limits and reasoning defaults differ from Gemma's
2048-token allowance. Distribution metrics requiring reference features or repeated
samples may be unavailable in this pilot; basic prose profiles are still recorded.
Do not report these five cases as a full benchmark or a controlled model-only gain.

Configuration references: [OpenCode agents](https://opencode.ai/docs/agents/),
[permissions](https://opencode.ai/docs/permissions/), and
[configuration](https://opencode.ai/docs/config/). Local version: 1.18.29.

## Setup verification

An initial attempt inherited the repository directory despite a temporary subprocess
working directory. Read restrictions blocked repository file contents, but glob
exposed repository filenames. Its artifacts are preserved under the run directory
with an `-invalid-directory` suffix and excluded from comparisons. Every invocation
now passes `--dir` explicitly, including resumed turns.

OpenCode read/edit patterns are relative to its worktree, which is `/` for these
non-git temporary projects. The allowlist covers only the exact temporary path
and the four scenario directories in the supported path representations. Two
permission-setup retries are archived separately and excluded.
