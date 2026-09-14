# Astra grading of the custom50 outputs

On 2026-09-14 the user authorized Astra to grade all 50 saved custom attempts,
using fresh evaluation sessions with the coding context removed. No Gemma
regeneration or direct paid-API fallback is involved. The previous three-call
implementation limit does not apply to this authorized pass.

The [Python runner](../../scripts/grade_custom50.py) reads the frozen run selection
and saved numerical scorecards, builds blinded packets, and applies validated
judgments. Three independent Codex executions can run at a time. The persisted
call budget is locked across workers and allows 50 calls; successful judgments are
cached by packet, model, schema, and instruction content.

## Grader design

[Grader instructions](../../src/writing_agent/grader_instructions.md) replace Codex's
built-in model instructions through `model_instructions_file`. The CLI runs in a
new temporary working directory with no conversation resume or fork. Launch
settings ignore user configuration, suppress project instruction loading and skill
instructions, disable memory and coding tools, and use no coding personality.
Existing subscription authentication is retained; API-key fallback is removed.

Each packet contains the visible brief, follow-ups, initial files, resulting files,
recorded tool actions, selected prose, and private grading criteria. It omits model
identity and previous judgments. Candidate thinking is not used as evidence of
successful action. The grader cites observable artifacts, distinguishes beliefs
from canon, accepts defensible KB selections, and does not penalize Markdown itself.

Astra returns the requested semantic metric scores, dimension assessments,
uncertainty, and pass/fail checks. It does not replace numerical prose measurements,
run fresh-reader probes, or mark its own judgments as human-validated. Required
check outcomes update task completion independently of prose-quality scores.

The first F1-01 judgment validated: Q2 prose quality 2/5, continuity pass, and an
optional style check failed. Its CLI events contain no tool calls. The two startup
notices concern a deprecated memory-flag alias and the skill-discovery feature;
neither prevented grading.

## Artifacts and status

The batch is running under `runs/custom50-e2b-it-2026-09-14/astra-graded/`:

- `status.json` and `review.md`: progress and per-case assessment links.
- `<case>/scorecard.json` and `<case>/review.md`: combined metrics and readable evidence.
- `judgments/<hash>/`: exact packet, custom instructions, launch arguments, CLI events,
  raw response, and validated judgment or failure.

After all 50 judgments validate, `publish()` updates the original review entry
points and scorecards. It preserves their prior versions as `*.pre-astra.*`.
Human review and grader calibration remain pending after this automatic pass.

The instruction override and instruction-loading settings are documented in the
[official Codex configuration reference](https://developers.openai.com/codex/config-reference/).

The grading pass exposed a delivery-status bug in F2-01: the candidate put prose
in its reply, left the requested file unchanged, and omitted the required file
markers. Extraction version 2 distinguishes that missing delivery from ambiguous
multiple spans. Publishing reapplies cached judgments to the corrected extraction
metadata; it does not change prose text or request another judgment. The original
scorecards remain archived for comparison.
