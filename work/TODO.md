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

## Task-graph environment: before Phase 8 training

Carried from the transition seam (`src/.context/transition-seam.md`). Each item must land
before group training or the first real model backend relies on it.

- [x] **Group head binding (HIGH-5):** require the bound manifest on real-member starts and commits; bind fresh collection to the verified head, record judged heads for interruptions, and refuse voiding a valid terminal reward.
- [x] **Token-budget admission:** require a usage-reporting sampler capability for token-limited entries and charge total-token limits.
- [x] **Mixed evaluator families (MEDIUM-3):** admission refuses a node whose checks span evaluator families.

Landed mixed-family and token-usage admission, sealed real-member execution, strict fresh-head
collection with reward-only resolution, and judged-head interruption records.
- [ ] **Compaction trigger:** make compaction controller-owned before group training uses it; today the caller triggers it. `task_graph_controller.py`.
- [ ] **Receipt journal:** record backend receipts before the first paid or remote backend.
- [ ] **Legacy graph evaluation adapter:** run legacy evaluation scenarios through the rollout core.
- [ ] **Non-derivable requests:** a backend whose exact request is not derivable from the pinned context and trace pins adds its own writer-turn field (the seam removed `request_ref`). `task_graph_records.py`, `task_graph_derive_writer.py`.

### Rollout cost: KV-cache reuse

- [ ] **Measure prefix rework first:** `work/sft/multi-turn-rl.md` — probe the share of each
  rollout's writer input already processed as an earlier-turn prefix.
- [ ] **Retain writer KV sessions:** `src/writing_agent/inference.py`,
  `src/writing_agent/task_graph_sampling.py`, `src/writing_agent/task_graph_group.py` — keep
  each rollout's cache across turns and share the byte-identical sealed start across members.
- [ ] **Verify exact token prefixes:** `src/writing_agent/inference.py` — check that turn N is
  an exact token prefix of N+1 under the pinned chat template; disable reuse if history changes.
- [ ] **Invalidate on policy updates:** `src/writing_agent/inference.py` — key writer state by
  policy version and flush on every weight update; frozen author, judge and reference caches may persist.
- [ ] **Record cache reuse:** `src/writing_agent/task_graph_records.py`,
  `src/writing_agent/task_graph_gatherers.py` — include cached-input token counts in writer usage.
- [ ] **Keep dynamic IDs out of prefixes:** `src/writing_agent/task_graph_sampling.py`,
  `src/writing_agent/inference.py` — exclude run IDs, timestamps and per-member values.
- [ ] **Reuse judge prefixes within groups:** `work/sft/multi-turn-rl.md` — put rubric and
  sources before the candidate, then grade group members back to back.
