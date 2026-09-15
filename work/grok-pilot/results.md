# Grok / OpenCode pilot results

`xai/grok-4.6` through OpenCode 1.18.29 completed all five selected cases. Astra
rated the three prose artifacts 4/5 each, compared with 2/5 each for Gemma E2B-IT
on the same cases. Both systems passed all five required-task checks in this subset.

| Measurement | Grok / OpenCode | Gemma / custom harness | Cases |
|---|---:|---:|---|
| Required-task completion | 5/5 | 5/5 | All five |
| Mean prose quality | 4.0/5 | 2.0/5 | F1-01, F2-06, F5-05 |
| Planning usefulness | 4/5 | 3/5 | F3-03 |
| Alternative diversity | 1.0 | 1.0 | F3-03 |
| KB faithfulness | 96.6% | 81.3% | F4-08 |
| KB interpretation | 4/5 | 3/5 | F4-08 |

These are Astra rubric scores, with evidence and explanations retained. KB
faithfulness denominators differ because the systems produced different claims.
The results describe five development cases and do not establish a general ranking.

The valid run used five distinct candidate sessions and five isolated Astra grading
calls. Native tools reported zero execution errors. Successful candidate generation
totaled approximately 324 seconds, excluding grading and discarded setup checks.
Reasoning traces are saved separately from delivered prose.

The KB case used 26 tools across three user turns, exceeding the custom harness's
24-call limit. OpenCode's per-turn step cap and provider generation limits also differ.
Interpret the results as a model-plus-harness comparison, not a controlled model-only
improvement. See the [pilot design](plan.md) for setup failures and limitations.

Artifacts are under `runs/pilot-grok46-opencode-2026-09-14/`:

- `review.md`: index of all five readable outputs and scorecards.
- `comparison.json`: paired scores, session counts, timing, and tool-budget audit.
- Each case directory: native events, launch configuration, snapshots, final files,
  conversation with reasoning, result record, and scored rubrics.
- `judgments/`: blinded Astra packets, launch records, and cached judgments.

Three earlier setup attempts are preserved in directories with `-invalid-*` suffixes
and excluded. Regenerating the report reused all five saved attempts and judgments;
no additional candidate or grader calls were needed.

Every scorecard records `candidate.harness = opencode`, `candidate.provider = xai`,
and `candidate.model = grok-4.6`. These fields apply to all measurements in the
card and appear in report groups. The full routed ID remains `xai/grok-4.6` in
the model configuration; Astra stays separate under judgment metadata.
