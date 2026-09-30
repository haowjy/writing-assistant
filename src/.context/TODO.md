# Source TODO

Must-do follow-ups scoped to `src/`. Each entry names the affected path and the follow-up.
Nice-to-have items are in [FUTURE.md](FUTURE.md).

## Task-graph training (Phase 8)

- [ ] **Delete the core `ask_author` machinery.** This is a designed change, coordinated
  with the `task-graph-environment` work item. It covers decision authority, the disclosure
  ledger, author-call budgets and scripted answer routing, across about 21 files. The probe
  tasks already run without the tool (interaction `none`). The change removes this root
  cause: `task_graph_contracts.ASK_AUTHOR_SCHEMA` declares `proposals` items as a bare
  `{"type": "object"}`, but `task_graph_calls.validate_ask_shape` demands exact
  `{id, text}` pairs. In P1 attempt 1, Gemma sent `{"name": …}` proposals twice, was refused
  twice, and then emitted a thought that made its member ineligible. If any author tool
  survives, its schema must state every field its validator checks. Fold in these review
  notes, which change probe graph hashes or experiment identity and need new pins:
  - **L1, the canary names.** `grpo_task_graph_probe_privacy` still scans for
    `unused_author_preference` (`task_graph_probe_tasks.AUTHOR_PACKET_CANARY`, spec key
    `unused_author_preference_canary`), but no author packet exists. Both canaries now sit
    in one private check record, so the positive control is effectively one bit. Drop the
    second canary or rename it. Rename `PRIVATE_STORE_DUMP_CANARY`'s literal
    (`P8R3C_PRIVATE_EVALUATOR_SPEC_CANARY_8365`) in the same change, because it detects a
    bulk store dump, not an evaluator-spec leak.
  - **L3, the simulator label.** `grpo_task_graph._native_policy` records
    `{"implementation": "scripted-author-v1", "script_ref": null}` for `none`-interaction
    nodes. Record a simulator that says there is no author. Update the S11 trace check's
    group-policy check (`scripts/task_graph_trace_check.py`) to match.
- [ ] **Decide the strict eligibility policy before any long training run** (design-lead).
  Today one ineligible member halts the whole run (`task_graph_training_export`,
  `grpo_task_graph.TaskGraphRollouts`). One spontaneously thinking member is enough, and
  that is model behavior, not a harness fault. P1 attempt 1 halted this way. In attempt 2 no
  member thought, so the rule went untested. Decide whether one ineligible member should
  still end a long run, and record the decision in the design.
- [ ] **S14: retire the DAPO legacy training path** (user decision N1; P1 has passed). It
  stays reproducible from `feat/dapo-full-rounds@9cb9944`. Delete:
  - `grpo_rollout.py`'s training-only parts (`RolloutGroups`, `NativeRolloutBackend`,
    `verify_tokens`);
  - `grpo_probe.py`, `grpo_probe_data.py`, `grpo_full48.py`, `grpo_full48_fixtures.py`,
    `grpo_full48_runner.py`, `grpo_full48_supervisor.py`, `grpo_gpu_fit.py` and
    `grpo_checkpoint31_fork.py`, with their scripts and tests;
  - the legacy `trl-1.13` route in `grpo_runtime.py`, and `train_grpo`'s legacy task path in
    `grpo.py` if nothing else uses it;
  - `reward.group_advantages` once it has no caller;
  - `docs/grpo-probe.md`, `docs/grpo-full48.md` and `docs/grpo-gpu-fit.md`, replaced with
    pointers to the DAPO branch.

  Also:
  - Delete `grpo_gpu._display_consumers`. It re-implements `ownership_report`'s rule over
    CSV lines, and only `grpo_probe` imports it, as a private name.
  - Recheck `agent.run_agent`'s failure classification (R1). It classifies by exception
    type, so a harness bug could be scored as a candidate failure. Fix it if `cwa eval`
    still depends on it.

  Accept only when no concept in design §1 has a live duplicate, no Phase 8 file exceeds
  1,000 lines, and the full suite, smoke and ruff pass. If N1 is reversed, rewire
  `grpo_probe.py` onto `training_stages.py` instead, and label the legacy route.
