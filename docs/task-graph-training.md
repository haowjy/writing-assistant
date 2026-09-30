# Task-graph training: the Phase 8 probe

Phase 8 trains Gemma E2B on task-graph rollouts with the existing DAPO trainer. The task
graph samples, scores and seals each group. The trainer applies the group's exact
advantages to the sampled tokens only. The probe is one bounded run on an RTX 3090 that
tests this path end to end:

- three public task graphs (t1–t3), each one writer node with file tools only and no
  author interaction;
- group size 4, three optimizer steps;
- stop after step 2, then resume from `checkpoint-2` to step 3;
- a deterministic offline inspection that writes `result.json`.

A pass proves the plumbing: native on-policy evidence, then task-graph reward, a masked DAPO
update, a checkpoint and a resume. It says nothing about writing quality or whether the
reward is valid.

The contracts behind the run live in the source context:
[records and admission](../src/.context/transition-seam.md),
[native sampling and tool outcomes](../src/.context/rollout-execution.md) and
[native training groups](../src/.context/group-coordination.md).

## The probe tasks

The three tasks are in [`configs/phase8/probe-tasks/`](../configs/phase8/probe-tasks/README.md).
`task_graph_probe_tasks.build_admitted_entry` admits them. Each one asks for a short scene in
`scene.txt` and states its task-specific detail in the brief, for example "The lantern is
amber." The writer has five tools: `list_dir`, `read_file`, `search`, `write_file` and
`patch_file`. The interaction mode is `none`, so there is no `ask_author` tool, no author
packet and no feedback turn. The core accepts mandatory feedback only on `scripted_author`
nodes, which must expose `ask_author`
(`task_graph_admission._validate_interaction`), so dropping the tool also dropped the
feedback turn.

Only `scene_nonempty` is required. Four optional checks give partial credit:

| Check | Weight (basis points) |
|---|---:|
| `scene_nonempty` (required) | 1,000 |
| `stated_detail`: `scene.txt` contains the literal phrase, such as "amber lantern" | 2,500 |
| `public_detail`: a second detail from the brief, such as "low tide" | 2,000 |
| `scene_word_range`: 24–70 words | 2,500 |
| `field_notes_nonempty`: the optional `field-notes.txt` | 2,000 |

`stated_detail` matches the literal adjective–noun phrase. A scene that follows "The lantern
is amber." in other words ("the lantern glowed amber") misses that component. This adds
reward spread, and it is not a failure to follow the brief.

Budgets per member: six writer turns, eight tool calls, a 4,096-token context cap, 512
tokens per decision and 1,536 generated tokens. There is no author-call budget.

Both privacy canaries sit in one private, non-admitted check record. Nothing public contains
them: not the brief, the initial files or the tool manifest. The id
`unused_author_preference` is a legacy name from the retired author packet. See criterion 6.

## P1 result

**P1 passed on its second attempt.** Attempt 2 ran `phase8/integration@95d4377` (tree
`c53bf23`) into `runs/phase8-probe-v2/`, and `result.json` reads `pass`. Attempt 1, at
`53b69ab`, failed at the train stage. The run records and the full write-up
(`probe-result.md`) live in the `task-graph-training` work item, outside this repository.

**What the pass shows.** Three optimizer steps on three tasks show that the integrated loop
is mechanically sound on real hardware. The loop covers native sampling, task-graph reward,
the masked DAPO update, checkpointing and resume. The run says nothing about whether
training improves writing, or whether the reward is valid.

### Attempt 1: `fail` at `train`

At that commit, each task exposed one author decision through `ask_author` and had one
scripted feedback turn. The train stage failed after 193 s with
`TrainingExportError: training member is not structurally eligible`. No ceiling was hit, and
peak reserved memory was 10.7 GiB.

In the first group (t1), one member called `ask_author` with `proposals` items shaped like
`{"name": "white"}`. The core refused the call twice with
`ask_author proposals need exact id/text pairs`. The model, with `enable_thinking=false`,
then emitted a thought channel on its own and chose the color itself. Derive recorded the
thought, so the member was `ineligible` (`reasoning_content_present`). The export refused
the group, and the run halted as designed: V1 training is non-thinking, and criterion 1
needs 12 of 12 members eligible.

It had two causes:

- **The tool schema said less than its validator.** `ASK_AUTHOR_SCHEMA`
  (`task_graph_contracts.py`) declares `proposals` items as a bare `{"type": "object"}`.
  `task_graph_calls.validate_ask_shape` demands exact `{id, text}` pairs. The model had no
  way to learn the shape. The S11 trace check had already recorded the same refusal as model
  behavior.
- **One ineligible member halts the whole run.** That policy is unchanged and still open;
  see `src/.context/TODO.md`.

### What changed before attempt 2

- **The probe tasks dropped `ask_author`** (the user's decision). Each brief states its
  detail, and `stated_detail` checks it. The feedback turn is gone, and both canaries moved
  into the private check record. Removing the `ask_author` machinery from the core is a
  separate, later change.
- **`_work_dir` stopped trusting the ambient variable.** Attempt 1's first `prepare` refused,
  because `MERIDIAN_ACTIVE_WORK_DIR` pointed at another work item. The runner now derives the
  work directory from the run directory and refuses a variable that disagrees.
- **The S11 trace check was brought up to date.** Its first run after the task change halted
  at startup (see [the S11 trace check](#the-s11-trace-check)). After the fix, a real-weight
  trace check on the new t1 shape passed. That check ran before the one-shot GPU attempt.

### Attempt 2: `pass`

| Phase | Result |
|---|---|
| `inspect`, `prepare`, `preflight` | ok. The task-graph hashes matched the pinned values |
| `train` (`--stop-after-step 2`) | ok in 158 s. Peak reserved 10.8 GiB, RSS 11.5 GiB |
| `resume` from `checkpoint-2` | ok in 105 s, reaching step 3. Peak reserved 10.9 GiB |
| `inspect-run` (CPU) | ok in 88 s. Verdict `pass` |

| Criterion | Evidence |
|---|---|
| 1. Groups, admission, tool outcomes | 3 groups, all `ready`, 0 ties. 12/12 members `structurally_eligible`, 3 all-admitted records, 0 protocol-shaped rejections |
| 2. Update | 3 optimizer steps, finite gradients, the adapter changed |
| 3. Checkpoints and resume | Checkpoints 1–3 verified, resume 2 → 3. Export = checkpoint 3 = resident adapter = PEFT reload |
| 4. Offline re-derivation | 0 mismatches in two byte-identical inspections. All six store-integrity controls refused |
| 5. Ceilings | 351 s of GPU time against 2,700. 10.9 of 21.0 GiB reserved, 11.5 of 24 GiB RSS. The run directory used 0.86 of 3 GiB |
| 6. Determinism and privacy | 0 canary hits in 1,316 files outside `training/private`. The positive control found both canaries inside it |

**What the model did.** The lead read this from the store, not from the admission records:

- **Every member wrote a scene.** All 14 tool calls (12 `write_file`, 2 `read_file`) returned
  `ok`, and no turn failed to parse. All 26 decisions ended `native_stop`, and no member
  emitted a thought.
- **Rewards had a spread in every group:** t1 0, 0.35, 0.55, 0.55; t2 0.6, 0.6, 0.8, 0.8;
  t3 0.55, 0.55, 0.55, 0.8. No member wrote `field-notes.txt`. Several paraphrased the
  stated detail and missed `stated_detail`.
- **One member ended incomplete, and that was scored honestly.** After a successful
  `write_file`, its next turn was empty and stopped on `<|tool_response>`.
  `task_graph_sampling.termination_stop_reason` classifies that turn as
  `unterminated_final_answer`. The member scored 0, stayed structurally eligible, and
  trained as a negative example.

**Measurements that do not gate:**

- on-policy drift: mean 0.0183 and max 0.459 over 1,123 masked tokens;
- re-prefill: 14,279 prefill tokens over 7,427 unique ledger tokens, a ratio of 1.92;
- 3.5 s mean generation time per decision;
- checkpoint 1 took 214 MB, and each later checkpoint added about 3 KB.

The strict eligibility rule was not tested this time, because no member thought.

## Before you run

The runner checks each of these and refuses to start when one fails.

- **A committed tree.** `prepare` refuses a dirty checkout, including untracked files. It
  records the commit and tree hash, and every later phase refuses if they change.
- **Offline settings.** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and
  `PYTHONDONTWRITEBYTECODE=1` are required. `CUDA_VISIBLE_DEVICES` must be `0` for the GPU
  run and empty for a CPU dry run.
- **The Phase 8 environment.** An overlay of TRL `6c5f135` and Liger `0.8.3` over the main
  virtual environment, built from archived, hash-verified files (`env-phase8/`).
  - `prepare` checks the installed source trees against the `trl-6c5f135-streaming`
    runtime pins in `grpo_runtime`, and the versions against
    `env-phase8/environment.json`: torch 2.14.0, transformers 5.17.0, peft 0.20.0.
  - The runner derives the work directory from the run directory
    (`grpo_task_graph_probe._work_dir`). It is the parent of the `runs/` directory that
    holds the run, or else the nearest ancestor that contains
    `env-phase8/environment.json`. If `MERIDIAN_ACTIVE_WORK_DIR` is set and resolves to a
    different directory, every phase refuses before it writes anything. Export it as the
    work directory, or unset it.
- **Cached weights and tokenizer.** `google/gemma-4-E2B-it` at revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, in the Hugging Face cache. Nothing is
  downloaded.
- **A new run directory.** `prepare` refuses an existing `$RUN`.
- **The local-model gate.** The [S11 trace check](#the-s11-trace-check) must have passed on
  real weights before the GPU run starts.

### GPU state (N3)

The probe uses the desktop GPU policy that the user chose (N3):

- at least **22,000 MiB free**;
- only the approved display consumers on the GPU: the COSMIC desktop processes, Xwayland,
  Ghostty, Chrome and Cursor, at most 256 MiB each and 768 MiB in total.

The ownership gate (`grpo_gpu.admit_gpu`) checks the full NVML inventory:

- in `preflight`;
- again before each GPU stage, writing its report to `ownership/<stage>/`.

The CUDA allocator is set to `expandable_segments:True` before Torch is imported. The
runner prepares the desktop policy only. A headless run with its own 23.0 GiB ceiling is
not implemented.

## Running the phases

Run from the approved commit, with the Phase 8 environment's Python. `$W` is the work
directory that holds `env-phase8/`. Each attempt gets its own `RUN`, named
`phase8-probe-vN`; P1 used `v1` and `v2`, so the next attempt is `v3`.

```bash
git rev-parse HEAD                       # must equal the approved commit
export MERIDIAN_ACTIVE_WORK_DIR="$W"     # or unset it; a different value is refused
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
PYTHON=$W/env-phase8/bin/python
RUN=$W/runs/phase8-probe-vN              # a new N per attempt; must not exist
"$PYTHON" scripts/run_task_graph_probe.py "$RUN"                      # inspect: no GPU, no writes
"$PYTHON" scripts/run_task_graph_probe.py "$RUN" --phase prepare
"$PYTHON" scripts/run_task_graph_probe.py "$RUN" --phase preflight
"$PYTHON" scripts/run_task_graph_probe.py "$RUN" --phase train --stop-after-step 2 --execute
"$PYTHON" scripts/run_task_graph_probe.py "$RUN" --phase resume \
  --resume "$RUN/training/checkpoint-2" --execute
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/run_task_graph_probe.py "$RUN" --phase inspect-run
```

| Phase | What it does |
|---|---|
| `inspect` (default) | Lists the phases. It reads no GPU and writes nothing |
| `prepare` | Checks the environment digests. Freezes the source commit and tree, the task-graph hashes, the tokenizer file hashes, the recipe, the N3 policy and the ceilings into `prepare.json` |
| `preflight` | Checks the GPU ownership and the environment digests, and loads no model. Writes `preflight.json` |
| `train` | Loads the base model in BF16 with SDPA attention, trains two steps and stops at `checkpoint-2`. `--stop-after-step` accepts only `2` |
| `resume` | Resumes from the numerically latest complete checkpoint, which must be `checkpoint-2`, and must reach step 3 |
| `inspect-run` | Offline. Runs the inspector twice, computes every criterion and writes `result.json` |

`train`, `resume` and `inspect-run` each run as a supervised subprocess
(`training_stages.py`). The supervisor enforces a wall-time kill, samples memory and disk,
and writes `stages/<stage>/status.json` and `resources.json`.

### One execution, no retries

- **Each stage runs once.** `attempt.lock/` holds an exclusive lock and a `<stage>.started`
  marker. A second `train`, a concurrent run or a leftover incomplete attempt is refused.
- **A failure is terminal.** A stage error, halt, OOM, ceiling breach, audit refusal or
  ownership refusal ends the run. Keep `$RUN` intact and record the failing stage. Do not
  retry in the same directory, and do not change the profile. A new attempt needs a new
  `$RUN`, a new `prepare` and a new go-ahead.
- **Resume uses only the latest checkpoint.** `--resume` must name the numerically latest
  `checkpoint-N` under `$RUN/training` that has its `complete.json`. Before the model
  loads, resume is also refused when group evidence exists at or after that checkpoint's
  step. A crash in the middle of a step therefore cannot be resumed; only the clean
  `--stop-after-step 2` boundary can.
- **Each step has one group.** Ties train with zero advantages. A `pending` or `invalid`
  group halts the run, and nothing is resampled.

### Frozen ceilings

`prepare.json` freezes these values and the supervisor enforces them:

| Limit | Value | `prepare.json` key |
|---|---|---|
| GPU stages in total | 45 min | `aggregate_gpu_seconds` = 2700 |
| Train stage (two steps plus load) | 25 min | `train_seconds` = 1500 |
| Resume stage (one step plus load) | 20 min | `resume_seconds` = 1200 |
| Peak reserved GPU memory | 21.0 GiB | `peak_reserved_gpu_bytes` |
| Peak process RSS | 24 GiB | `peak_rss_bytes` |
| Run-directory growth | 3 GiB | `run_directory_growth_bytes` |

The `inspect-run` stage has its own 45-minute wall limit and does not count toward the GPU
total.

### Recipe and budgets

- **Model:** `google/gemma-4-E2B-it` in BF16 with SDPA attention.
- **Adapter:** LoRA r8, α16, all linear layers, dropout 0.
- **Loss:** DAPO, beta 0, `scale_rewards="none"`, so TRL receives the task-graph
  advantage unchanged.
- **Optimizer:** AdamW at 1e-5, constant schedule.
- **Batching:** microbatch 1 with gradient accumulation 4, and non-reentrant activation
  checkpointing.
- **Rollout budgets:** a 4,096-token context cap, 512 tokens per decision, 1,536 generated
  tokens per member, up to 6 writer turns and 8 tool calls.

## Where the evidence is

Everything the run writes is under `$RUN`:

| Path | Contents |
|---|---|
| `inspect.json`, `prepare.json`, `preflight.json`, `preflight/`, `ownership/<stage>/` | Pre-run records and GPU ownership evidence |
| `attempt.lock/` | The exclusive attempt marker and the per-stage `.started` markers |
| `training/` | The task-graph store and the trainer's output root |
| `training/{artifacts,events,commits,context_content,context_revisions,instances,operations,refs,checkpoints}/` | Task-graph lineages and store checkpoints |
| `training/groups/<group_id>/` | `spec.json`, `start-<n>.json`, `result-<n>.json`, `training-batch.json`, `training-admission.json`, `trainer-consumed.json`, and `inspection.json` after `inspect-run` |
| `training/groups/step-NNNNNN` | Step reservations: one per optimizer step, with its final state (`consumed`, `audit-refused`, `adapter-drift`, `halted`, …) |
| `training/private/` | Private records. The privacy canaries live here, and the scan excludes only this directory |
| `training/experiment.json` | The experiment identity, which binds every `grpo*`, `native_*` and `task_graph*` source file |
| `training/checkpoint-{1,2,3}/` | Every trainer checkpoint, each with its `complete.json` |
| `training/invocations/<id>/` | Records for each invocation, and the exported adapter |
| `training/batches/`, `training/observer/` | Each step's batch refs and adapter hashes; the observer's masks, advantages, denominators and recomputed logprobs |
| `stages/<stage>/` | `status.json`, `resources.json`, `stdout.log`, and the stage's outputs, such as `training-result.json`, `generation-times.json` and `gradient-observer.json` |
| `inspection/inspect-{1,2}.json` | The two offline inspections |
| `result.json` | The verdict, the criteria and the measurements |

## Reading `result.json`

Top-level keys: `schema`, `run_id`, `execution_mode` (`gpu`, `cpu-dry-run` or
`cpu-all-tie`), `prepared_source` (the commit and tree), `verdict`, `criteria`,
`measurements`, `stages`, `failures` and `inspect_run_elapsed_seconds`. If the inspection
itself fails, the result keeps only the schema, the run, the mode, the verdict, the criteria
and `failures`.

### Verdicts

- **`pass`:** all six criteria are computed and true.
- **`inconclusive_no_signal`:** all three groups tie. Criteria 3–6 hold, and so does all of
  criterion 1 except "at least one group `ready`". Criterion 2's three steps and finite
  gradients hold. The plumbing works, but there is no update evidence. This verdict is
  registered in advance, is reported as inconclusive rather than as a pass, and is not
  retried.
- **`fail`:** anything else, including a criterion that could not be computed.

### Criteria

Each entry under `criteria` has `computed`, `passed`, `evidence`, `missing_inputs` and
`description`. A missing input fails its criterion.

1. **Groups, admission and tool outcomes.**
   - 3 groups, each `ready` or `tie`, and at least one `ready`.
   - 12 of 12 members `structurally_eligible`.
   - 3 `TrainingAdmissionV1` records that admit every member.
   - Complete per-member tool outcomes, with **0 protocol-shaped tool rejections**.

   A protocol-shaped rejection means our code refused the model's call: a duplicate or
   missing call ID, an invalid envelope, non-array calls, non-object arguments, or
   arguments that are not valid JSON. The native parser makes each of these impossible. A
   model's own error is recorded, but it does not fail the criterion. Examples are bad
   arguments, an unsafe path, a failed patch or a budget refusal. The counts come from
   `task_graph_tool_outcomes.read_member_tool_outcomes`. This rule was added after the S11
   trace check showed that admission alone can hide refused calls.
2. **The update happened.** The LoRA hash changed, the optimizer step counter is 3, and
   every gradient was finite.
3. **Checkpoints and resume.** Checkpoints 1–3 are kept and verified, and resume from 2
   reaches 3. The resident adapter equals the checkpoint-3 files, and the exported adapter
   reloads through `PeftModel` with the same tensors.
4. **Offline re-derivation.** Both inspections re-derive every ledger and admission with 0
   mismatches, and six **store-integrity controls** are refused on a copy of the store. The
   controls tamper with bytes (a generated token, an external token, a logprob, a policy
   pin, a termination claim, a V1 manifest swap), and content addressing catches them. They
   do not test the audit's checks against hash-valid lies; unit tests prove those, not P1.
5. **Ceilings.** Every applicable wall-time, GPU-memory, RSS, aggregate-GPU-time and
   run-directory ceiling held.
6. **Determinism and privacy.**
   - The two inspections are byte-identical, and the run used no network.
   - Two planted canaries, `unused_author_preference` and `private_store_dump`, must be
     present under `training/private` (the positive control) and absent from every other
     file in the run directory. Both sit in one private, non-admitted check record, so
     they detect a bulk dump or a copied private record. They do not test whether an
     admitted check spec leaked. The first id is a legacy name, because the tasks have no
     author packet.
   - Each member's sampler inputs are rebuilt from that member's own verified lineage.

   `sibling_input_scope.limits` states the limit of this check: it proves per-member
   lineage and input reconstruction. It does not prove the absence of arbitrary shared
   public text or of sibling-derived content that nobody planted.

### Measurements that do not gate

These are reported for inspection and never change the verdict:

| Key | Meaning |
|---|---|
| `on_policy_drift` | Mean and max `\|sampled − recomputed\|` logprob over masked tokens, and the token count. On the GPU the FP32 streaming softcap rules out bit equality, and no bound is calibrated yet |
| `re_prefill_ratio`, `prefill_tokens`, `unique_ledger_tokens`, `per_decision_generate_time_seconds`, `mean_generate_time_seconds` | The KV-cache measurement: every decision re-prefills its whole ledger |
| `termination_classes` | Counts of `native_stop`, `token_limit` and `context_limit` turns |
| `stop_reason_counts` | `{per_member: {member_id: {reason: count}}, totals: {…}}`. A member that completed has no stop reason, so its entry is `{}`, and this key alone cannot tell a completed member from missing data |
| `native_parse_failed_count` | Turns whose tool-call text did not parse. Those turns end their lineage `incomplete` (`unparsed_tool_call`, or the limit's reason) |
| `tool_call_count`, `tool_call_outcomes_by_member` | Per-member call counts by outcome code, and whether the files changed |
| `reward_min`, `reward_max`, `reward_spread`, `tie_count`, `group_count` | Reward spread and ties |
| `stage_runtime_seconds`, `stage_disk_growth_bytes`, `disk_growth_per_checkpoint` | Time and disk use |

## CPU dry runs and what they cannot show

The same runner has a CPU mode for checking the plumbing without a GPU:

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
"$PYTHON" scripts/run_task_graph_probe.py "$RUN" --cpu-dry-run --phase prepare
"$PYTHON" scripts/run_task_graph_probe.py "$RUN" --cpu-dry-run --phase preflight
"$PYTHON" scripts/run_task_graph_probe.py "$RUN" --cpu-dry-run --phase train --stop-after-step 2 --execute
"$PYTHON" scripts/run_task_graph_probe.py "$RUN" --cpu-dry-run --phase resume \
  --resume "$RUN/training/checkpoint-2" --execute
"$PYTHON" scripts/run_task_graph_probe.py "$RUN" --cpu-dry-run --phase inspect-run
```

Add `--cpu-all-tie` (with `--cpu-dry-run`) to force every group to tie. That run must end
`inconclusive_no_signal`.

A dry run is not a small version of the GPU run. It uses a tiny random Gemma in FP32 with
eager attention. Its sampler is `ScriptedNativeBackend` (`scripts/smoke_task_graph_grpo_cpu.py`),
which emits **fixture** tool-call text. The tiny model only supplies logprobs; its tool calls
were never sampled. The fixture text passes through the same `parse_native_response` and
call-ID binding as the native backend. GPU ownership is stubbed (`stubbed_for_cpu`).

So a CPU dry run can show that the phases, receipts, export, audit, checkpoints, resume and
criteria fit together. It cannot show:

- the tool-call shapes, EOS placement, parse failures or token-limit cuts that a real model
  produces;
- whether the real model's calls land and change files;
- GPU memory fit, the ceilings under load, or ownership on a live desktop;
- drift under BF16 with the FP32 streaming softcap on CUDA.

A dry run can also pass while hiding a protocol bug. The S12b dry run passed while 24 of its
calls had been refused as duplicate IDs. Criterion 1 now counts these rejections.

## The S11 trace check

Before the GPU run, `scripts/task_graph_trace_check.py` samples one group of two members on
task t1 with the **real** cached Gemma weights on CPU. It uses the P1 budgets and does no
training. It then finalizes, exports, audits and admits the group, and runs the offline
inspector twice.

Before loading the model, `load_trace_task_entry`
(`scripts/task_graph_trace_check_support.py`) checks t1's budgets against
`TRACE_TASK_BUDGETS` and refuses any `max_author_calls`. After sampling, the script
requires the trainer's group policy to be context operation `carry`, interaction `none` and
the simulator `{"implementation": "scripted-author-v1", "script_ref": null}`. The report
stores that under `group_policy` as `context_policy`, `simulator_ref` and `simulator`.
The field was `scripted_author_ref` before S10c. The trainer
(`grpo_task_graph._native_policy`) still records the `scripted-author-v1` label with no
script for these author-free tasks. The label is misleading but harmless.

```bash
timeout 3600 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH=src "$W/env-phase8/bin/python" scripts/task_graph_trace_check.py \
  "$W/runs/trace-check-vN"
```

- `--dry-run` prints the plan without importing the model stack, reading weights or writing
  anything. It stops before the startup checks, so it does not show that a real run would
  start. `tests/test_task_graph_trace_check.py` tests the startup path.
- The script refuses an existing output directory and has a 60-minute ceiling.
- It writes `report.json`, and `summary.json` when it completes. The report covers each
  decision's tool calls and results, tokens, termination and file changes, plus drift,
  re-prefill, timing and peak RSS.
- It halts with `protocol_shape=tool_result` on any protocol-shaped tool rejection.
- It runs once per output path. After a halt, fix the cause in a new commit, use a new path
  and record the attempt.

**Judge it by outcomes, not admission.** A run where every member is admitted can still be
wrong. Check each call's result and whether the files changed.

The attempts so far:

- **v1** halted before sampling on a generation-config check.
- **v2** was admitted and reported `pass`, but four of six tool calls had been refused as
  duplicate IDs. No write landed, and the reward scored the starter file.
- **v3** passed after the call-ID fix: 6 of 7 calls `ok` (the seventh was the model's own
  malformed `ask_author` arguments), both members changed `scene.txt`, rewards were 0.55
  and 0.30, and the group was `ready`. The tasks still used `ask_author` then.
- **v4**, the first run on the author-free tasks, halted at startup in 0.09 s. It still
  expected `max_author_calls: 2`, a check the S10b lane's `--dry-run` had never reached.
- **v5** passed after that fix. Each member made one `write_file` call (`ok`), stopped
  natively and did not think. The rewards were 0.8 and 0.55, and the group was `ready`.
  This was two members on the CPU, not a forecast for twelve on the GPU.
