# Evaluation follow-ups

The active work order is now **[TODO.md at the repository root](../TODO.md)**.
Start there; the Gemma short-context GRPO probe comes first.

The older evaluation checklist below is retained for reconciliation against saved
results, not as the next work order. Unchecked historical run entries do not establish
that those runs remain unexecuted. Keep current priorities in the root checklist and
optional ideas in [FUTURE](FUTURE.md).

- [ ] Review the [five-case E2B-IT pilot](custom-eval-suite/pilot-e2b.md) and its saved artifacts with the user. Check task realism, hidden requirements, prose selection, and expected outcomes.
- [ ] Review the [five-case thinking rerun](custom-eval-suite/thinking-pilot.md): all workspace cases used tools. Inspect prose quality, single-page wiki interpretations, and the wiki-to-prose retrieval/word-budget failures.
- [ ] Complete and inspect the authorized 50-case E2B-IT run; retain failures and pending semantic judgments in the results.

Listing a run here does not start it. The full comparison and larger experiments
retain the execution boundary in the [suite plan](custom-eval-suite/plan.md).

- [ ] Address the [output review](custom-eval-suite/output-review.md): semantic grounding checks, metadata/script extraction, KB retrieval coverage, and conditional genre wording. Preserve current case versions and saved artifacts.

- [ ] Prepare and run the primary external prose benchmark: [Creative Writing v3 scope and cost](creative-writing-v3/cost-plan.md). Approved: 32 outputs, first variant per prompt, $2 grading cap; queued after custom50. WritingBench remains deferred.

- [ ] Complete the [automatic external checks](external-benchmarks/plan.md): IFEval and the 32-task HumanEval+ diagnostic are queued on E2B-IT. These follow the approved 32-output prose baseline.

- [ ] Review the [completed Astra custom50 judgments](custom-eval-suite/astra-grading.md), especially ambiguous interpretations and optional checks. Human calibration remains unvalidated.
