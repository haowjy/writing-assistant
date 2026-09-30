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
- [ ] **Give the `_work_dir` refusals codes.** `grpo_task_graph_probe._work_dir` raises
  `ProbeError` both when it cannot locate the work item and when `MERIDIAN_ACTIVE_WORK_DIR`
  disagrees with the run directory. `tests/test_grpo_task_graph_probe.py` tells the two
  apart with `assertRaisesRegex` on the message, and test rules forbid asserting messages
  (review L2). Add a code or a subclass for the disagreement, and assert that instead.
  Also test the symlinked-path and empty-variable cases, which the review checked by hand.
- [ ] **Watch empty turns that stop on `<|tool_response>`.** In P1 attempt 2, one member
  followed a successful `write_file` with an empty turn that stopped on the tool-response
  token. `task_graph_sampling.termination_stop_reason` classified it
  `unterminated_final_answer`, and the member scored 0 as a trainable negative example.
  That is correct, not a bug. Rollout and termination owners should track how often it
  happens in longer runs before they change prompts, stop tokens or completion rules.
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

- [ ] **Reward meaning, not literal phrases, in real training tasks.** The probe's
  `stated_detail` check (`configs/phase8/probe-tasks/`) is a `contains` match on a phrase
  such as "amber lantern". A scene that honors "The lantern is amber." in other words misses
  it, and several P1 members did (review L5). That is harmless spread for a plumbing probe.
  Real training tasks should not reward a literal phrase when they mean "follow the brief".

## Tests

- [ ] **Split the oversize test modules.** `tests/test_task_graph_rollout_env.py` (1,238
  lines) and `tests/test_grpo.py` (1,036) are over the 1,000-line cap. S14 shrinks
  `test_grpo.py`; split the rest by concern.
