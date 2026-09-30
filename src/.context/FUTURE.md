# Source FUTURE

Nice-to-have follow-ups scoped to `src/`. None blocks correctness. Must-do items are in
[TODO.md](TODO.md).

## Task-graph training (Phase 8)

- [ ] **Split the probe evidence module.** `grpo_task_graph_probe_evidence.py` is 964
  lines. Split it before adding code, for example by moving the ledger measurements or the
  group collection into their own module.
- [ ] **Report terminal status per member.** In `grpo_task_graph_probe_evidence.py`, add
  each member's `task_status` next to `stop_reason_counts.per_member`. A completed member
  has no stop reason, so its entry is `{}`, and that cannot be told apart from missing data.
- [ ] **Rename the store-dump canary literal.** `task_graph_probe_tasks.PRIVATE_STORE_DUMP_CANARY`
  still reads `P8R3C_PRIVATE_EVALUATOR_SPEC_CANARY_8365`, but the canary detects a bulk
  private-store dump, not leakage of an evaluator spec. The literal feeds the probe task
  graphs' hashes, so rename it only alongside another deliberate change to those hashes.
- [ ] **Stop parsing the rendering-pin message.** `task_graph_derive_writer._bind_writer_turn`
  recovers the differing field from the text of
  `task_graph_native_contracts`' `"native renderer <field> differs …"` error, and reports
  `input.context.rendering.<field>`. If that text is reworded, the rejection still happens
  but the reported path falls back to `input.adapter_trace`. Carry the field in a
  structured error instead.
- [ ] **Bind the classifier's inference messages.** `native_protocol.is_native_output_parse_error`
  matches the two messages `inference.parse_response` raises: `"Native tool-call output was
  not completely parsed"` and `"Native tool arguments must be an object"`. `inference.py`
  is not among the sources the task-graph experiment identity hashes (`grpo*`, `native_*`,
  `task_graph*`). A reword would therefore not change the identity. It fails closed, so the
  next malformed output halts instead of scoring. Move these messages into `native_protocol`
  as constants, or bind `inference.py`.
- [ ] **Calibrate GPU drift.** `on_policy_drift` does not gate on the GPU, because the FP32
  streaming softcap rules out bit equality and no bound is calibrated. Choose a bound from
  P1 and later GPU runs before using drift as a gate (`grpo_task_graph_observer.py`).
- [ ] **Train across compaction.** `task_graph_eligibility` refuses any lineage with more
  than one context root (`multi_segment_context`), and `task_graph_training_export` exports
  one root. Training compacted lineages needs segment likelihoods reconstructed against
  their original contexts, and the controller-owned compaction trigger first.

## Tests

- [ ] **Split the oversize test modules.** `tests/test_task_graph_rollout_env.py` (1,238
  lines) and `tests/test_grpo.py` (1,036) are over the 1,000-line cap. S14 shrinks
  `test_grpo.py`; split the rest by concern.
