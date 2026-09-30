# Source TODO

Must-do follow-ups scoped to `src/`. Each entry names the affected path and the follow-up.
Nice-to-have items are in [FUTURE.md](FUTURE.md).

## Task-graph training (Phase 8)

- [ ] **Record the P1 result.** `docs/task-graph-training.md` ("P1 result" section): replace
  "pending" with the verdict, the criteria and the non-gating measurements from
  `$RUN/result.json`, and link the run's result note. Until then, no doc may claim a GPU
  outcome.
- [ ] **S14: retire the DAPO legacy training path after P1 passes** (user decision N1). It
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
