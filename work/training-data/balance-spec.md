# Balance specification for the training and evaluation task pools

Status: proposed, pre-generation. This document governs both pools. It is a
specification, not an applied change: it does not edit `src/`, `tests/`, `data/`,
or `runs/`. Names of code symbols below are references to the current schema, so a
future implementation can assert against them.

Architecture rule this document obeys: atoms live in JSON vocabulary files
(`data/training/variation-catalog-v1.json`), combinations are assembled by Python
(`src/writing_agent/task_generation.py`), and prose/content are LLM-authored into
slots (`work/sft/task-generator-instructions.md`). No finished task JSON may be
hand-written. The balance targets below are therefore expressed over *assignment
fields*, not over literal task text.

---

## 0. Evidence base

Everything in this section was measured or read from the checkout, not assumed.

### 0.1 Current inventory

| Pool | Role | Count | Where |
|---|---|---:|---|
| SFT seed | train | 19 | `data/training/sft-v1.jsonl` (`split=train`) |
| SFT seed | validation | 5 | same (`split=validation`) |
| SFT seed | test | 0 | none exists |
| Prepared training requests | train candidate | 100 | `data/processed/training-tasks-v1/requests.json` |
| Generated training tasks | accepted train candidate | 54 of 100 | `data/processed/training-tasks-v1/generated/manifest.json` (`accepted: 54`, `compiled: 54`) |
| Authored development scenarios | development | 50 | `data/scenarios/development.json` |
| Held-out long-form cases | final_eval | 6 | `data/scenarios/longform-v1.json` |
| EQ-Bench Longform anchor | final_eval, external | 12 | `data/processed/longform-benchmark-v1-faithful/manifest.json` |

Note on the "roughly 100 eval items" figure: the measured eval-shaped inventory is
50 development + 6 held-out long-form + 12 external, while the 100 figure is the
*prepared training request* batch. Those must not be conflated. At no point is there
"more test than train" in a useful sense: there is almost no frozen test at all
(0 test SFT records, 18 frozen final-eval cases), an oversized advisory development
set (50), and only 19–54 usable training records. The spend priority is therefore
train ≫ val ≫ new test, and test must be *re-scoped and frozen*, not grown freely.

### 0.2 Defects the design must correct

| Defect | Measurement (source) | Design consequence |
|---|---|---|
| Prose collapse | 12/25 bf16 pieces open with "The damp air of the archive"; self-BLEU 0.71 | Genre/style/trope/situation must become COVERAGE floors, not one-shot assignments |
| Channel confound | dev `tools` count is `{0: 25, 5: 25}`; all reply cases have 0 tools, all file cases have 5. `longform_suite._prose`/`visible.tools` sets `[] if replies else WRITE_TOOLS` | Add a **channel_free** condition and decouple tool availability from channel |
| Shallow interaction | dev followups `{0: 40, 1: 5, 2: 5}`; generator caps at `stage_families` of length 1–3 (`_body`), so ≤2 follow-ups | Add a deep-interaction stratum (≥4 stages) and balance steer count |
| Silent defaults | 40/40 loose-brief attempts chose a default and never surfaced it | L-level must assert ask-required vs default-safe behavior, and default-safe tasks must surface a reversible default |
| Length overshoot | 9/10 explicit-budget pieces overshoot; median 268 words against 120–220 asked | Word budget becomes a COVERAGE axis and a HARD failure-mode stratum |
| Source poverty | accepted 54 tasks from 5 works in 2 groups; each non-preserve genre appears once | Source work/group and genre breadth get explicit floors |

### 0.3 Current accepted-task imbalance (the pool we already have)

From `data/processed/training-tasks-v1/generated/manifest.json` intersected with
`requests.json`: family `{F1:14, F2:12, F3:14, F4:8, F5:6}`; blend size
`{0:19, 1:19, 2:16}`; stages `{1:34, 2:12, 3:8}`; 35 non-preserve genres each
appearing exactly once; 5 source works each ~10–12. This is not close to a balanced
pool under any of the axes below, which is the point of the hierarchy.

---

## 1. Why a hierarchy, and what "no task uses the entire catalog" means here

Twenty-plus axes cannot be balanced as a flat list. A pool of 300 tasks has
300/60 = 5 per genre but only 300/(5·4) = 15 per (family × level) cell and
300/(60·20) = 0.25 per (genre × style) pair. A Cartesian target would demand
fractional cells and mutually contradictory requirements: a "close continuation"
must preserve source genre, so it cannot also satisfy an arbitrary genre value; a
no-tool reply task cannot satisfy a "multi-file deliverable" cell. Equality is only
achievable for a small number of axes whose values are cheap, orthogonal, and
reward-relevant.

`composition_rules[0]` — "Select a small number of compatible ingredients; no task
must use the entire catalog" — is the license for this. It applies **at the task
level**: each task carries exactly one genre (plus at most one blend alternative),
one style, one sampled trope, one situation, and one continuity challenge. It does
**not** apply at the pool level; pool-level breadth is exactly what COVERAGE and
HARD floors enforce. Rules that must hold:

- A task never consumes the whole catalog; the *pool* collectively covers it.
- Every axis is assigned by a seeded per-axis permutation in
  `task_generation.Sampler.orders`, so marginals are independent by construction
  rather than by post-hoc rejection sampling.
- The generator may replace an incompatible ingredient (trope/situation) and must
  record the replacement in `realized_variation`; assigned coverage is therefore
  checked on the request, realized coverage is REPORTED after generation.

Four rings:

| Ring | Meaning | Gating |
|---|---|---|
| **HARD** | Exactly balanced; generation fails if the target is missed | Asserted in code |
| **COVERAGE** | Every value appears at least N times; not equal | Asserted floor |
| **SAMPLED** | Drawn from the vocabulary; checked only in aggregate entropy; values may go unused | Non-gating |
| **REPORTED** | Measured after generation; never constrains generation | Dashboard only |

---

## 2. The axis list

Every axis carries exactly one ring. "Values" is the vocabulary or bin set.

### HARD axes

| # | Axis | Values | Justification |
|---|---|---|---|
| H1 | `family` | F1 direct prose, F2 file authoring/revision, F3 planning, F4 KB construction, F5 KB-grounded writing | Family selects the applicable rubric metrics (`Q2/Q5/Q7–Q13`) and the delivery channel. An unbalanced family silently shifts the reward's component mix, so equality is required for any cross-family comparison. |
| H2 | `instruction_specificity` level | L0, L1, L2, L3 | The primary declared axis (`specificity.py`); it determines withheld decision points. Imbalance confounds every other behavioural claim and is the documented "declared variation goes unrealized" failure. Equality over four levels is affordable at 300 tasks. |
| H3 | `withheld_decision_point` set | Derived from `DECISION_POINTS` per family/level | `split_spec` must be exact: a brief at a level states exactly the points whose level ≤ cutoff and withholds the rest. A leaked or missing point is the silent-default defect. |
| H4 | `ask_modality` | `fully_stated`, `ask_required_withheld`, `default_safe_withheld` | Decides whether the correct behavior is to ask or to proceed with a surfaced reversible default. Without an explicit fully-stated control the policy drifts to always-ask; without ask-required tasks it never learns discriminating clarification. |
| H5 | `deliverable_channel` (requirement) | `reply_required`, `file_required`, `channel_free` | The measured channel confound. `channel_free` is new and is the only condition in which a channel *choice* is observable. |
| H6 | `tool_availability` | `tools`, `no_tools` (+ tool surface: read-only vs read+write) | Tools were perfectly collinear with channel. Decoupling is required before any channel or tool-use claim is valid. |
| H7 | `channel × tool` joint | 6 cells, one structurally impossible | The decoupling invariant itself: every feasible cell must be populated and no channel may imply a tool set. |
| H8 | `genre_blend_size` | 0, 1, 2 | Blend size changes difficulty, the generator's task, and the reach of the continuity check. Already the intended generator axis; equality is cheap. |
| H9 | `stages` | 1, 2, 3, 4+ | Stage count sets the reward aggregation (`0.5·mean(stage) + 0.5·final_state`) and the follow-up count. The shallow-interaction defect is a stage-count ceiling, so the deep bin is a HARD requirement, not a floor left to chance. |
| H10 | `steer_count` | 0, 1, 2, 3+ (= `len(followups)`) | The fix for shallow interaction. Each corrective/steering turn must exist in balanced numbers so the model learns revision under feedback, not one-shot generation. |
| H11 | `provenance` | `human`, `half_synthetic`, `fully_synthetic` | Source poverty is measured (5 works, 2 groups). Grounding needs human sources; contamination control and generalization need synthetic/private worlds. Exact proportions make the claim auditable. |
| H12 | `failure_mode_coverage` | The scored check kinds: `word_range`, `wiki_links`, `evidence_exposed`/retrieval, `protected`, `excludes`, `nonempty`, continuity | These are the verifiable levers the reward actually reads. If a check is absent from the pool the reward has no gradient along it; if it is unbalanced the policy can pass by a strategy that never encounters it. |
| H13 | `source_lineage_group` separation | Group identity per work + derivatives | Random splitting leaks a work across pools through shared characters/facts, inflating held-out scores. Separation is a correctness property, archived by content hash (`longform_suite.holdout_audit`). |

### COVERAGE axes (every value ≥ N)

| # | Axis | Values | Justification |
|---|---|---|---|
| C1 | `genre` | 60 (`variation-catalog-v1.json`) | Breadth directly attacks the prose collapse (all tasks come from few genres). Equality over 60 values at 300 tasks needs 5 each and forbids continuation tasks from carrying a non-source genre, so a floor (≥3) is the honest target. |
| C2 | `style` | 20 | The dominant prose-collapse lever. 20 values × 300 tasks makes ≥10 each tractable; equality is possible but not worth constraining because "close continuation preserves source style". |
| C3 | `trope` | 40 | Breadth of narrative situation reduces templated plots. The generator may replace an incompatible trope, so a floor on the *assignment* is the enforceable form. |
| C4 | `situation` | 32 | Same rationale and replacement rule as tropes. |
| C5 | `continuity_challenge` | 12 | The skill the F4/F5 rewards measure. All 12 are groundable, so a high floor (≥15 of 300) is safe; exact equality is not required because a challenge must fit its source. |
| C6 | `emotional_register` | proposed 10–12 (e.g. warm, elegiac, comic, tense, dread, wonder, grief, wry, tender, hopeful) | The collapse is tonal as well as lexical. Register is not in the current schema, so this is a proposed new vocabulary key. Floor keeps each register from being a one-off. |
| C7 | `pov_person_tense` | 1st/past, 1st/present, close-3rd/past, close-3rd/present, omniscient/past, 2nd/present | Measured gap: only close third exists. POV is a prose-control capability, not a nuisance. Floor rather than equality because loose tasks legitimately leave viewpoint open (see the contradiction in §8). |
| C8 | `word_budget_envelope` | `open`, `120–220`, `400–800`, `800–1200` | The overshoot defect is a budget-following failure. Presence/absence of an explicit budget is fixed by the H2 level; the *envelope* is the free part and must span short and long so the model cannot specialize to one band. |
| C9 | `continuity_load` | facts that must be preserved: 0–2, 3–5, 6+ | Development fact sets are tiny; deeper loads are the capability under test. Floor per bin forces a real high-load arm. |
| C10 | `deferred_reveals` | count of unresolved questions that must stay unresolved: 0, 1, 2+ | Directly scores "retain an unresolved question without inventing its answer", one of the 12 continuity challenges and a documented reward check. |
| C11 | `revision_depth` | `create`, `single_edit`, `cumulative_multi_edit` | Revision is a distinct family skill. The current pool has almost no cumulative editing; a floor forces it. |
| C12 | `context_size` | `<2k`, `2–8k`, `8–16k`, `16–28k` tokens | Long-context retrieval and compaction are project goals; the frozen long-form suite only reaches the top bin. Floor per bin prevents all training from being short. |
| C13 | `source_work` (train only) | distinct works | The source-poverty lever: floor each work so no work dominates, and force ≥12 works. |
| C14 | `artifact_count` | `single`, `multi` (among file-capable tasks) | Multi-file delivery is a scored failure mode absent from the current pool. Floor on `multi` rather than equality because it is conditional on H5. |

### SAMPLED axes

| # | Axis | Values | Justification |
|---|---|---|---|
| S1 | `steer_position` | where in the stage sequence a steer lands | Meaningful variation for dialogue realism, but every position is equally valid and cannot be given a floor without multiplying H10. Aggregate entropy is sufficient. |
| S2 | blend pairing (second genre) | catalog genres | Pairing space is 60×59; covering it is impossible and unnecessary. The pair is sampled and only the *size* (H8) is gated. |
| S3 | `setting` / world | private worlds + source settings | Worlds are a vehicle for content; the already-authored `worlds.json` and new private worlds are sampled. Only the count of distinct worlds is reported. |
| S4 | `instruction_phrasing` | meaning-preserving rewrites | A nuisance axis: the reward must be invariant to it. Sampled for a robustness probe, never balanced. |
| S5 | `thinking`/reasoning budget | off / on | A conditioning axis explicitly deferred in `training-distribution-axes.md`; sampled for an ablation, not balanced. |
| S6 | `cast_size` / relationship type | e.g. 2-focal, ensemble | Useful breadth, no reward coupling; sampled. |

### REPORTED axes

| # | Axis | Measured by | Justification |
|---|---|---|---|
| R1 | realized vs assigned variation | `realized_variation` diffed against `assignment` | The generator may replace ingredients; only the realized value is ground truth. |
| R2 | prose distribution | `prose.score_prose` D1/D2/D4, opening-stem collision rate, self-BLEU | This is the metric the design is trying to move; it cannot be a generation constraint without gaming it. |
| R3 | continuity realization | semantic continuity score vs the `continuity_challenge` assignment | Whether a challenge was actually exercised is a quality outcome. |
| R4 | length delta | delivered words − declared budget | Diagnostic for the overshoot defect; reporting it keeps the cause visible without hard-coding a policy. |
| R5 | group reward variance | `frac_reward_zero_std`, per-group reward std | The GRPO health metric from `reward.py`/`rl-algorithm-decision.md`; a leading indicator of reward overfitting. |
| R6 | reference distance | MMD (only above the sample floor) | Needs ≥20 samples; reported when available, never gating. |
| R7 | tool/step/artifact counts | harness traces | Diagnoses whether a task silently needed more tools than budgeted. |
| R8 | lineage/overlap audit | `holdout_audit`, hash overlap | Confirms H13 after the fact; a build-time invariant, reported per release. |
| R9 | vocabulary utilization | entropy over each axis | Shows whether SAMPLED axes collapsed to a few values. |

---

## 3. Target distributions

Ratio: **TRAIN : VAL : TEST = 300 : 60 : 60 = 5 : 1 : 1.**

Defense of that ratio for this project specifically:

1. **RL, not supervised scaling, sets the train size.** GRPO samples a group of
   rollouts per prompt every step. The binding constraint is *distinct prompts*, not
   examples. With one RTX 3090, a long/multi-turn task admits roughly 2–4 prompts
   per optimizer step. At 200 steps that is 400–800 prompt-slots. Requiring each
   prompt be used at most ~4 times (beyond that the policy memorizes prompt-specific
   reward quirks and within-group advantage variance collapses) gives a floor of
   ~100–200 distinct prompts; a 300-task pool meets it with margin and 512 is the
   stretch target if batch or step count rises. 19–54 is far below the RL floor; it
   is recycled into a near-constant prompt distribution and yields low-entropy
   advantages.
2. **Val is deliberately small.** Validation is for checkpoint ranking, not
   capability claims. 60 tasks = 3 per (family × level) cell, which is enough to
   rank checkpoints without competing with train for source groups.
3. **Test is small because evaluation noise is reduced by rollouts, not tasks.**
   Variance of a per-task estimate falls as 1/√n_samples. Holding spend fixed,
   concentrating 12 rollouts on each of 60 frozen tasks gives a lower-variance
   comparison than 1 rollout on 720 tasks, and the long-form cases are far too
   expensive for the latter anyway (`longform-v1` declares 4 samples/case and 13
   turns at 14–28K context). The frozen suite can be as small as ~40 tasks if every
   reward-reading axis value appears ≥2 times and each gets ≥8 rollouts; 60 is the
   target with headroom.
4. **Test is a larger share than a typical ML split** (1/7) because the pool is
   small and the reward-overfitting risk is high; the external anchor (§5) supplies
   the independent half so the internal suite need not grow.

### TRAIN = 300

| Axis | Target |
|---|---|
| family | 60 each (F1–F5) |
| level | 75 each (L0–L3) |
| channel requirement | `file_required` 120, `reply_required` 90, `channel_free` 90 |
| tools | `tools` 180, `no_tools` 120 |
| channel×tool joint | file×tools 120; reply×tools 60; reply×none 30; free×tools 60; free×none 30; file×none forbidden (structurally impossible: writing a file needs a write tool) |
| genre_blend_size | 100 each (0/1/2) |
| stages | 1: 90, 2: 90, 3: 90, 4+: 30 |
| steer_count | 0: 90, 1: 90, 2: 90, 3+: 30 |
| provenance | human 120, half_synthetic 120, fully_synthetic 60 (40/40/20) |
| failure_mode_coverage | `word_range` ≥ 90; `wiki_links` ≥ 40; retrieval/`evidence_exposed` ≥ 60; `protected` ≥ 60; `excludes` ≥ 30; `nonempty` = 300; continuity ≥ 150 |
| genre | ≥ 3 each (avg ≈ 5); continuation tasks count as `preserve source genre` |
| style | ≥ 10 each |
| trope | ≥ 5 each |
| situation | ≥ 6 each |
| continuity_challenge | ≥ 15 each |
| emotional_register | ≥ 15 each |
| pov_person_tense | ≥ 30 each |
| word_budget_envelope | `open` 75, `120–220` 75, `400–800` 90, `800–1200` 60 |
| continuity_load | 0–2: 120, 3–5: 120, 6+: 60 |
| deferred_reveals | 0: 150, 1: 100, 2+: 50 |
| revision_depth | create 150, single_edit 90, cumulative 60 |
| artifact_count | single 180, multi 120 (among file-capable) |
| context_size | <2k: 120, 2–8k: 90, 8–16k: 60, 16–28k: 30 |
| source_work | ≥ 12 distinct, each ≥ 12 |
| source_lineage_group | ≥ 6 groups, each ≥ 20 |

### VALIDATION = 60

Family 12 each; level 15 each; provenance synthetic 30 / half 20 / human 10.
Mirror train marginals at 1/5 scale; coverage floors halved (genre ≥ 1, style ≥ 3).
Source works and lineage groups disjoint from both train and test. Purpose: rank
checkpoints and calibrate the grader; never used for capability claims.

### TEST = 60 internal (+ 6 long-form + 12 external)

| Axis | Test target | Note |
|---|---|---|
| family × level | 3 each = 60 | 20 cells, so a per-check reward gap is visible per cell |
| channel requirement | reply 15, file 15, channel_free 30 | `channel_free` is over-weighted because it is the diagnostic for the channel confound |
| tools | tools 40, no_tools 20 | file tasks require tools; the rest spread |
| provenance | synthetic 30, half 20, human 10 | test leans synthetic to reduce pretraining contamination |
| failure_mode_coverage | every scored check kind present, each ≥ 6, and ≥ 2 as `required` | HARD; this is the reward's read surface |
| word_budget_envelope | all 4 bins, ≥ 8 each | the overshoot defect must be faceable |
| continuity_load | 6+ present in ≥ 12 | deep continuity is a test-specific stratum |
| deferred_reveals | 2+ present in ≥ 12 | |
| steer_count | 3+ present in ≥ 12 | |
| context_size | 16–28k present in ≥ 6 | |
| genre / style / trope / situation | ≥ 2 / ≥ 2 / ≥ 1 / ≥ 1 | breadth is not the purpose of test |
| source_lineage_group | disjoint from train and val | |
| internal frozen suite | 60 short/medium + the 6 `longform-v1` cases | |
| external anchor | EQ-Bench Longform, 12 prompts, faithful mode | not ours, cannot be gamed by our reward |

Rollouts: internal 12 per task (720 samples); `longform-v1` as declared
(`samples_per_case: 4`); EQ-Bench per its upstream protocol in `faithful` mode.
Record the mode and chapter count with any external score.

---

## 4. SFT versus GRPO: one pool, two selection filters

They share the **generation pipeline and prompt pool** but not the admission
filter.

| | SFT | GRPO |
|---|---|---|
| Consumes | fixed corpus of accepted trajectories | prompts, with a fresh group of rollouts per step |
| Size driver | coverage of behavioural cells, not raw count | distinct prompts to keep advantages high-entropy |
| Corpus target | 200 accepted trajectories (usable range 150–300); ≥1 per family×level×check cell | 300 distinct train prompts (floor 256, stretch 512) |
| Admission | task accepted AND ≥1 high-quality trajectory, human-anchored cells preferred | task accepted; rejected/failed candidate outputs discarded but prompt identity retained |
| Failure mode | overfitting the few exemplars | overfitting the reward; flattening the prompt distribution |

**Does the same pool serve both?** Yes, with two views: an SFT *trajectory set*
(small, curated, quality-gated) and an RL *prompt set* (the whole accepted pool,
quality-gated only at the task level). Do not collapse them: an RL prompt whose best
trajectory is poor is still a useful GRPO prompt, but must not enter SFT. Conversely,
SFT must not re-use the frozen test prompts for demonstration.

**Why RL needs a large distinct-prompt pool.** GRPO normalizes rewards within a group
of rollouts on the *same* prompt (`reward.py`: within-group standardised rewards).
If the same handful of prompts is recycled every step, the group statistics become
correlated across steps, the policy memorizes the prompt, and the advantage signal
turns low-entropy — visible as a rising `frac_reward_zero_std` and as `all-tie`
groups. A 300-prompt pool caps reuse at ≈4 with a 4-prompt/step, 200-step run, which
keeps each step's groups drawn from different prompts. This is the concrete reason
the RL pool target is ~5× the SFT seed.

**GRPO overfits the reward, and which axes that makes load-bearing.** The optimizer
maximizes `0.40·quality + 0.30·intent + 0.20·continuity + 0.10·mechanics` plus
critical-failure caps. It will learn whatever the judge and the mechanical checks
reward — including a long-output bias (the observed overshoot is exactly the kind of
artifact a length-insensitive judge reinforces), style tics, and asking/not-asking
artifacts. The load-bearing held-out axes are therefore precisely the ones the reward
reads: **word budget, continuity/retrieval, wiki-links/navigation, channel/tool
behavior, and ask-modality.** These must be HARD-balanced in TEST so a reward-hacked
policy shows a gap. A held-out suite authored from our own taxonomy shares the
reward's assumptions and can show that the policy *moved*, not that it *improved*;
only the external anchor (EQ-Bench Longform) is independent enough to falsify. This
is why test separates the internal suite from EQ-Bench.

**Rollouts rather than tasks.** Because noise falls with 1/√rollouts, the cheap way
to make a held-out comparison stable is to raise samples per task. Implication: the
frozen suite can be small (≈40 tasks minimum, 60 target) provided every reward-reading
axis value appears ≥2× and each gets ≥8–12 rollouts. Spending on extra held-out
*tasks* instead buys coverage of axes the reward does not read, at the expense of
variance on the axes it does.

---

## 5. Train / test separation rules

Split by **source lineage group**, never randomly.

1. A *lineage group* is a work plus every derivative: genre adaptations, divergent
   branches, extracted KBs, and rephrasings. Derivatives inherit the parent's split.
2. Train, validation, and test are three disjoint sets of lineage groups. Validation
   and test must also be disjoint from each other, so validation cannot proxy for
   test in checkpoint selection.
3. No test prompt, its transforms, or its judged outputs may enter training, even
   reworded. Enforce by content hash, not filename: `longform_suite.holdout_audit`
   already refuses a build when a reserved work's hash is claimed by a training
   source or a development case.
4. The current 5 works / 2 groups populate **train only**. Test draws from newly
   acquired public-domain works plus fresh private synthetic worlds; validation
   uses a third disjoint set.
5. The 10 authored `world-*` development groups are already excluded from training
   (`task-generation-v1.json: excluded_source_groups`); keep them for validation and
   grader calibration, never for training.
6. The held-out `longform-v1` benchmark is reserved: no training source, no
   development case, and no checkpoint-selection decision may use it
   (`longform_suite` module docstring).

**How the external anchor differs.** `longform-v1` is *ours*: authored from the same
taxonomy and reward, compiled from our public-domain slices, with our checks. EQ-Bench
Longform is a pinned third-party release (`EQ_REVISION` in `longform.py`) with its own
12 prompts, 13-step plan-then-eight-chapters protocol, its own 14 weighted criteria and
negative criteria, and an adaptation of its own arithmetic. It shares neither our task
distribution nor our reward. That independence is the whole point: it separates "our
reward moved the policy" from "our reward moved the policy toward our own taste." It
only works in `faithful` mode (reply-only); `workspace` mode adds file delivery and is
not comparable, so any score must be quoted with its mode and chapter count.

---

## 6. What test must balance that train need not, and vice versa

A test set is not a miniature training set.

**TEST must HARD-balance (diagnostic surface):**
- `failure_mode_coverage` — the reward's read surface; without it reward hacking is
  invisible.
- `deliverable_channel` and `tool_availability` with the joint decoupling — the
  specific confound under test.
- `word_budget_envelope` — all bins, including the band where the overshoot lives.
- `continuity_load`, `deferred_reveals`, `steer_count` deep bins — the adversarial
  strata for continuity and revision.
- `provenance` leaning synthetic — to reduce pretraining contamination.
- source-lineage disjointness — a correctness property.

**TEST need not balance (and should not waste budget on):**
- genre/style/trope/situation breadth (≥2/≥2/≥1/≥1 is enough; a test set is not a
  breadth corpus).
- `genre_blend_size` or `stages` equality.
- emotional register or POV equally.
- performance-oriented distribution (`provenance` human share, source-work floors).

**TRAIN must HARD-balance (learning signal):**
- `family`, `instruction_specificity`/`ask_modality`, `genre_blend_size`, `stages`,
  `steer_count`, `provenance`, `failure_mode_coverage`.
- `channel`/`tool` independence, so the policy sees channel choices during training.
- COVERAGE floors on genre/style/trope/situation/continuity — the anti-collapse spend.

**TRAIN need not balance (and would waste budget):**
- the deep adversarial bins at test-level density (high continuity load, deferred
  reveals, 16–28K context can be COVERAGE floors rather than equal).
- `source_work` equality across the test works (test works are held out entirely).
- test-only diagnostic strata such as `channel_free` at 50% (train uses 30%).

In one line: **train balances what the policy must learn; test balances what the
reward can be gamed on, plus the adversarial strata that make a hack legible.**

---

## 7. Acceptance checks (one per HARD axis)

Assertions below are written against the assignment records that
`task_generation.coverage()` already summarizes. `N = len(pool)`; `counts[a][v]` is
the number of tasks with axis `a` = value `v`.

```
H1  family:           all(counts["family"][f] == N // 5 for f in FAMILIES)
H2  level:            all(counts["instruction_specificity_level"][l] == N // 4 for l in LEVELS)
H3  withheld points:  for every r: check_declared(split_spec(r.family, r.level).stated, r.declared_stated) == []
H4  ask modality:     every L0 task has no withheld point; every L3 task with branch_choice withheld has behavior ASK_REQUIRED;
                      count(ask_required_withheld) > 0 and count(fully_stated) > 0
H5  channel:          counts["channel"] == {"reply_required": 90, "file_required": 120, "channel_free": 90}   # train
                      and every task's visible.prose selectors agree with its declared channel
H6  tools:            counts["tools"] == {"tools": 180, "no_tools": 120}                                       # train
H7  channel × tools:  for every feasible cell c: count(c) >= N // 10; assert count({"file_required","no_tools"}) == 0
H8  blend size:       all(counts["genre_blend_size"][k] == N // 3 for k in (0, 1, 2))
H9  stages:           all(counts["stages"][k] >= N // 10 for k in (1, 2, 3)) and count(4+) >= N // 10
H10 steer count:      all(counts["steer_count"][k] >= N // 10 for k in (0, 1, 2)) and count(3+) >= N // 10
H11 provenance:       within +/- 1 of human 0.40N, half_synthetic 0.40N, fully_synthetic 0.20N
H12 failure modes:    each kind in {word_range, wiki_links, retrieval, protected, excludes, nonempty, continuity} appears >= floor listed in §3
H13 lineage split:    set(train_groups) & set(val_groups) == set() == set(train_groups) & set(test_groups) == set(val_groups) & set(test_groups)
```

For TEST the same predicates apply with the §3 test targets and the stronger
failure-mode floor (every kind ≥ 6, ≥ 2 required); for VAL with the 1/5 targets.

Two assertions worth stating explicitly because they are the measured defects:

```
channel confound:  P(tools | reply_required) != 1  and  P(tools | channel_free) in (0, 1)
                   (currently P(tools | reply) == 0 and P(tools | file) == 1; both must move)
safety net:        assert all(r.followups_expected == len(r.stage_families) - 1 for r in pool)
```

`H13` is not an LLM task-quality assertion; it is a build-time hash invariant and
should reuse `longform_suite.holdout_audit`, which raises rather than warns.

---

## 8. Contradictions, dependencies, and open items

These are flagged, not resolved by this spec.

1. **Viewpoint is scored but not a specificity decision point.** `src/.context/CONTEXT.md`
   states loose prompts "remove checks for unstated length, viewpoint, idea count,
   and ending requirements", and the baseline reports "optional word-budget and
   viewpoint/style checks each failed nine times". But `specificity.py: DECISION_POINTS`
   contains no `viewpoint`, `idea_count`, or `ending` point. Under the current schema
   C7 (`pov_person_tense`) cannot be withheld at any L-level, so a loose brief cannot
   make POV ask-required or default-safe. Either add a `viewpoint` point or drop the
   claim that loose tasks govern viewpoint.
2. **Deep interaction needs a schema/compiler change.** `_body` derives at most three
   stages, so `stages: 4+` and `steer_count: 3+` cannot be generated today. H9/H10
   depend on extending `stage_families` construction and the `followups = stages−1`
   contract in `task-generator-instructions.md`.
3. **`channel_free` is not expressible in the current schema.** `visible.prose` and
   `visible.tools` are set by family (`reply` for F1/F3, files otherwise), and the
   long-form compiler hard-codes `[] if replies else WRITE_TOOLS`. H5/H6/H7 require a
   channel-requirement field and a scoring rule for "a file write when reply was
   permitted".
4. **The "54 generated tasks" in the prompt is real but stale as a design input.**
   `generated/manifest.json` reports `requested: 100, accepted: 54`; `requests.json`
   reports `generated_tasks: 0, accepted_tasks: 0` (schema v2, pre-generation copy).
   Use the manifest as the inventory of record and regenerate requests under the
   current `SCHEMA_VERSION = 3`, which carries the L-level objects H2/H3 need.
5. **`genre`, `emotional_register`, and `pov_person_tense` do not all exist as
   vocabulary keys today.** `variation-catalog-v1.json` supplies genres/styles/tropes/
   situations/continuity_challenges only. C6/C7 need new vocabulary entries before
   the COVERAGE floors can be asserted.
6. **Source acquisition is a gating dependency.** H11/H13 and C13 need more than the
   5 current works / 2 groups. `work/custom-eval-suite/data.md` lists candidates but
   no webnovel or wiki has confirmed reuse terms. Until at least 12 train works and
   6 lineage groups exist, H11/H13 cannot be met and generation must fail closed.

Open decisions for the parent: (a) adopt 300/60/60 or the 512 stretch; (b) approve
the `channel_free` and `viewpoint` schema additions; (c) approve one new deep-
interaction stage rule; (d) confirm the external anchor's faithful-mode protocol and
method for the RL comparison.
