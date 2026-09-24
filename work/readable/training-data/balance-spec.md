# How many tasks we need, and what their mix must cover

This proposal sets the size and mix of the training, validation, and final-test task pools so that the model learns more than a few repeated task shapes and the tests can expose weaknesses in its reward. It proposes 300 training tasks, 60 validation tasks, and 60 internal test tasks, with six existing long-form tests and 12 external benchmark prompts kept separately. The counts and rules below preserve the original proposal, including unresolved numerical and schema conflicts; those conflicts must be decided before this can be an enforceable generation specification.

## Why training needs most of the tasks

A task gives the model a writing brief, starting project files, tools, and checks for acceptable work. Training tasks teach the model; validation tasks compare training checkpoints during development; test tasks are held back for the final measurement. Keeping a test suite “frozen” means fixing its cases and excluding them from training and checkpoint selection.

The proposed ratio is TRAIN : VAL : TEST = 300 : 60 : 60 = 5 : 1 : 1. The main reason for the larger training pool is repetition during reinforcement learning (RL), which trains by reward. In group-relative reinforcement learning (GRPO), the model answers the same prompt several times and receives a learning signal from how each answer scores relative to its other answers. With only a few prompts, repeated exposure lets the model learn prompt-specific scoring quirks. When all answers to a prompt earn the same reward, there is no within-group reward difference to learn from.

The proposal assumes one RTX 3090 and roughly 2–4 prompts per optimizer step for long or multi-turn tasks. Over 200 steps, that is 400–800 prompt slots. Its rule of thumb is to reuse each prompt at most about four times; that implies roughly 100–200 distinct prompts. A 300-task pool provides margin, and 512 is the stretch target if batch size or step count increases. The later admission table sets a stricter floor of 256 prompts. These are different thresholds in the source, not one measured cutoff. The original also claims that 300 prompts caps reuse at about four in a 4-prompt-per-step, 200-step run; the allocation rule that would enforce that cap is unspecified.

The current 19–54 usable training records fall below the proposed RL floor. The source's concern is that repeatedly recycling them makes the prompt distribution nearly constant and reduces variation in the relative rewards used for learning. It describes 300 as about five times the supervised-training seed, although the measured supervised seed is 19 training records; that comparison is unresolved. Spending should prioritize training, then validation, then new test cases. Existing test work needs a defined scope and a fixed held-out suite rather than unconstrained growth.

Validation is deliberately small: 60 tasks provide three for each combination of five task families and four instruction-detail levels. Its purpose is to rank checkpoints and calibrate the grader, not support capability claims. Keeping it small also avoids consuming source groups needed for training.

The proposal favors repeated answers to each fixed test task over many tasks answered once. It proposes 12 rollouts—independent sampled attempts—on each of 60 internal tasks, or 720 samples. It claims this gives a less noisy comparison than one rollout on each of 720 tasks at the same spend. It also says the suite could shrink to about 40 tasks if every property read by the reward appears at least twice and every task gets at least eight rollouts; elsewhere it specifies 8–12. The statistical claim needs qualification: the original calls the quantity falling as 1/√n_samples “variance,” and supplies no analysis establishing that fewer tasks reduce overall comparison error. Both the claim and the uncertainty are retained here rather than silently changing the design rationale.

Long-form cases are expensive: `longform-v1` declares four samples per case, 13 turns, and 14–28K tokens of context. The proposed internal test share is 1/7, larger than a typical machine-learning split according to the source, because this pool is small and reward overfitting is a concern. The external benchmark supplies the independently designed check, so the internal suite need not grow to do that job too.

## What the task labels mean

The five families are F1, direct prose; F2, file authoring or revision; F3, planning; F4, knowledge-base construction; and F5, writing grounded in a knowledge base. A knowledge base (KB) is the project's reference material, including linked pages about facts, characters, and accepted decisions.

Instruction specificity means which decisions a brief states, rather than merely calling the brief “explicit” or “loose.” In the referenced `specificity.py`, L0 states goal and deliverable; L1 adds setting or navigation; L2 adds applicable length, style, selection, and option-count decisions; L3 adds applicable continuity, authorization to establish story facts, and branch choice. Each level includes the earlier levels, and only decisions applicable to that family count. A withheld decision is one the model must resolve by asking or by stating a reversible default. The original acceptance rule reverses this ordering by saying L0 has nothing withheld; that contradiction is identified again below.

A delivery channel says where the work belongs. `reply_required` requires the reply, `file_required` requires saved files, and the proposed `channel_free` lets the model choose. `tools` and `no_tools` mean whether file tools are available; available tools may be read-only or allow both reading and writing. A file-writing requirement without a write tool is impossible.

A stage is one step of the task sequence. A steer is a later user instruction, counted by `len(followups)`; the proposed contract is followups = stages − 1. A source work and its derivatives belong to one source lineage group, which must stay in a single pool. Provenance categories are `human` for human source material, `half_synthetic` for partly synthetic material, and `fully_synthetic` for fully synthetic material. The proposal does not give operational criteria for distinguishing the partly synthetic category.

Genre blend size 0 means preserving the source genre in a close continuation; 1 selects one genre; 2 combines that genre with one alternative. A trope is a recurring narrative pattern, a situation is the immediate story circumstance, and a continuity challenge specifies what kind of story consistency the task must exercise. An artifact is a required output such as a passage or file; selectors identify the output the scorer should inspect. Context size is the amount of input material in tokens, with k or K meaning thousands.

## What the existing evidence says needs fixing

The inventory below was measured or read from the checkout. The role `final_eval` reserves a case for final evaluation; EQ-Bench Longform is the external writing benchmark described later. Supervised fine-tuning (SFT) trains by imitating accepted example responses; its seed records should not be confused with prepared generation requests or held-out evaluation cases.

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

The measured evaluation-shaped inventory is 50 development cases + 6 held-out long-form cases + 12 external prompts. The “roughly 100 eval items” figure actually refers to prepared training requests. The source rejects “more test than train” as a useful description: it counts no SFT test records, 18 fixed final-evaluation cases, an oversized advisory development set of 50, and only 19–54 usable training records.

Six observed problems drive the proposal:

1. Prose repeated itself: 12/25 pieces in the bf16 run began “The damp air of the archive,” and self-BLEU was 0.71. bf16 means the bfloat16 model-number format used in that run. Self-BLEU compares generated pieces with one another for overlapping word sequences; higher overlap means less varied phrasing. These observations motivate repeated coverage of genres, styles, tropes, and situations, rather than assigning each only once.
2. Tool availability was inseparable from delivery channel. Development cases had `{0: 25, 5: 25}` tools: every reply case had none, and every file case had five. `longform_suite._prose` sets `visible.tools` using `[] if replies else WRITE_TOOLS`. A comparison therefore cannot distinguish a channel effect from a tool-access effect. The proposal adds free channel choice and varies tool access separately.
3. Conversations were short. Development followups were `{0: 40, 1: 5, 2: 5}`, and generator `_body` builds `stage_families` of length 1–3, allowing at most two followups. A required group of tasks with at least four stages would force practice in longer revision sequences.
4. In 40/40 loose-brief attempts, the model chose a default without saying so. Tasks must distinguish missing decisions that require a question from decisions safe to resolve with an explicitly stated reversible default.
5. Nine of ten pieces (9/10) with explicit budgets overshot; the median was 268 words against a requested 120–220. Tasks therefore need multiple budget ranges and enough required word-count checks for the reward to detect overshoot.
6. The accepted 54 tasks came from five works in two groups, and each non-preserved genre appeared once. Source and genre floors would prevent this narrow set from standing in for broad coverage.

Joining `data/processed/training-tasks-v1/generated/manifest.json` to `requests.json` gives family counts `{F1:14, F2:12, F3:14, F4:8, F5:6}`, blend sizes `{0:19, 1:19, 2:16}`, and stages `{1:34, 2:12, 3:8}`. There are 35 non-preserved genres, each appearing exactly once, and five source works with about 10–12 tasks each. The source concludes that this pool is far from the proposed balance requirements.

## Why some properties get exact targets and others get minimums

The proposal controls the mix of tasks so that the model cannot learn only one form of work. But requiring every combination of more than twenty properties is infeasible. At 300 tasks, 60 genres would average five tasks each; five families crossed with four instruction levels would average 15 per combination; and 60 genres crossed with 20 styles would average only 0.25 per pair. Some combinations are contradictory: a close continuation must preserve source genre, and a no-tool reply task cannot require multiple output files.

The proposal therefore names four kinds of requirement. HARD means a mandatory generation gate, described in the original as exact balance, although several listed gates are minimums or integrity rules. COVERAGE means every vocabulary value must occur at least a stated number of times without equal counts. SAMPLED means values are drawn from the vocabulary and assessed only through aggregate entropy, a measure of how concentrated the distribution is; some values may go unused and the result does not block generation. REPORTED means a measurement taken after generation that does not constrain generation. The codes H, C, S, and R below identify requirements in those four groups.

The catalog's `composition_rules[0]` says to select a few compatible ingredients rather than use the whole catalog in a task. Each task has one genre plus at most one blend alternative, one style, one sampled trope, one situation, and one continuity challenge. The pool collectively supplies breadth. `task_generation.Sampler.orders` uses a seeded permutation for each vocabulary property; the source claims this makes the individual distributions independent without rejecting combinations after sampling. It does not provide evidence of joint statistical independence, so that claim remains a limitation.

A generator may replace an incompatible trope or situation but must record the change in `realized_variation`. Coverage gates inspect the requested assignments; later reporting compares them with what was actually authored. This distinction prevents an assigned label from being treated as evidence that the finished task exercises it.

## Proposed pool targets

These tables preserve the original numbers. They are targets, not counts already achieved. In particular, the training tools total says 180 with tools and 120 without, while the joint channel/tool rows sum to 240 with tools and 60 without. No generated pool can meet both versions.

### Training: 300 tasks

The training targets spread learning across the five activities, four levels of instruction detail, creative ingredients, and checked failure modes. `word_range` checks word bounds; `wiki_links` checks KB links; `evidence_exposed` checks that retrieval exposed needed evidence; `protected` guards material that must remain unchanged; `excludes` rejects forbidden content; and `nonempty` requires an output. Continuity checks concern consistency with established story facts.

The remaining table labels name concrete properties. `emotional_register` is tone; `pov_person_tense` is viewpoint and tense; `word_budget_envelope` is a length range or no stated limit; `continuity_load` is the number of facts to preserve; `deferred_reveals` counts questions that must remain unanswered; `revision_depth` distinguishes creation, one edit, and cumulative edits; and `artifact_count` counts single versus multiple requested outputs. Their full vocabulary and rationale follow later.

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

The `artifact_count` row assigns 180 single and 120 multiple artifacts “among file-capable” tasks but gives no consistent denominator for that phrase. Likewise, the numeric distribution for each clarification modality is not specified. These are unresolved requirements, not implied zero counts.

### Validation: 60 tasks

The proposal specifies 12 tasks per family, 15 per level, and provenance of 30 synthetic, 20 partly synthetic, and 10 human. It also says to mirror training distributions at one-fifth scale, with coverage floors halved, naming genre ≥1 and style ≥3. Those instructions are preserved, but they do not consistently follow either scaling rule or the training provenance ratio. Validation source works and lineage groups must be disjoint from both training and test. Validation ranks checkpoints and calibrates the grader; it cannot support capability claims.

### Internal test: 60 tasks, plus the existing long-form and external sets

The test targets emphasize properties that expose weaknesses in the reward, including free channel choice, longer revision, heavier continuity demands, and every scored check kind. The genre/style/trope/situation floors below are ambiguous: if the genre floor of two applies to all 60 catalog genres, it needs at least 120 tasks, exceeding this 60-task pool.

| Axis | Test target | Note |
|---|---|---|
| family × level | 3 each = 60 | 20 combinations, allowing reward-check differences to be examined within each |
| channel requirement | reply 15, file 15, channel_free 30 | `channel_free` is over-weighted because it is the diagnostic for the channel confound |
| tools | tools 40, no_tools 20 | file tasks require tools; the rest spread |
| provenance | synthetic 30, half 20, human 10 | test leans synthetic to reduce pretraining contamination |
| failure_mode_coverage | every scored check kind present, each ≥ 6, and ≥ 2 as `required` | Mandatory; these are the failures the reward checks |
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

### When training and test deliberately differ

Test must enforce coverage of every scored failure mode, the channel/tool combinations, all word-budget ranges, the deepest continuity/reveal/steer categories, a synthetic-leaning provenance mix, and disjoint source lineages. Otherwise a policy that exploits the reward may never face the conditions exposing that behavior.

Test need not enforce equal genre blends or stage counts, equal emotional registers or viewpoints, the training share of human sources, or training source-work floors. The original calls genre/style/trope/situation floors of ≥2/≥2/≥1/≥1 sufficient and argues against spending test budget on breadth; the vocabulary scope of those floors needs resolution as noted above.

Training must enforce family, instruction-detail and clarification behavior, blend size, stages, steer count, provenance, and scored failure modes. It must vary channel and tools separately and cover genres, styles, tropes, situations, and continuity challenges repeatedly to resist repetitive output. It need not match test's concentration of difficult cases: high continuity load, unresolved reveals, and 16–28K context can be minimum coverage groups. It must not include held-out source works merely to balance them. Free channel choice is proposed for 30% of training tasks and 50% of test tasks.

## One prompt pool, two ways to admit training material

SFT and GRPO share the generation pipeline and prompt pool but admit different material. SFT consumes a fixed set of accepted trajectories, meaning completed sequences of responses and tool use. Its target is 200 accepted trajectories, with a usable range of 150–300 and at least one for each applicable family × level × check combination. The source does not enumerate those combinations. A task must be accepted and have at least one high-quality trajectory before entering SFT; human-anchored examples are preferred. The main risk is overfitting a few examples.

GRPO consumes accepted prompts and generates fresh groups of attempts at every step. Its target is 300 distinct training prompts, with a floor of 256 and stretch of 512. Failed or rejected candidate outputs are discarded while prompt identity is retained. Even a prompt whose best current attempt is poor can remain useful for reward-based training, but that poor attempt must not become an SFT example. Frozen test prompts must never be reused as demonstrations.

In `reward.py`, GRPO standardizes rewards within the attempts on one prompt. Reusing a handful of prompts correlates group statistics across steps and may let the policy memorize prompt-specific quirks. The proposed warning measures are `frac_reward_zero_std`, the fraction of attempt groups with zero reward spread, and `all-tie` groups, where every attempt earns the same reward. The intended size driver is distinct prompts and varied relative rewards, rather than a raw count of accepted example responses.

The optimizer maximizes `0.40·quality + 0.30·intent + 0.20·continuity + 0.10·mechanics`, subject to critical-failure caps. It will pursue whatever judges and mechanical checks reward, including length bias, recurring style habits, and asking or avoiding questions inappropriately. A judge insensitive to length could reinforce the observed overshoot. The held-out checks that matter most therefore cover word budgets, continuity/retrieval, KB links/navigation, channel/tool behavior, and clarification behavior. The proposal calls these mandatory test requirements, even when it describes related training properties as coverage floors.

An internal suite uses the project's own taxonomy and reward assumptions. It can show that the policy changed under that reward without independently establishing that writing improved. The external EQ-Bench Longform benchmark supplies an independently designed test intended to challenge that interpretation.

## Keep source families in one pool and reserve the external comparison

Split by source lineage group, never by randomly assigning derived tasks. A group contains a work and every derivative: genre adaptations, divergent branches, extracted KBs, and rephrasings. Derivatives inherit the source's split. Train, validation, and test must be pairwise disjoint so that checkpoint selection on validation cannot indirectly expose test material.

No test prompt, transformation, or judged output may enter training, even reworded. Enforce the split through content hashes rather than filenames. `longform_suite.holdout_audit` already refuses a build if a reserved work's hash is claimed by a training source or development case. Hash checks enforce the recorded identity rule; the proposal does not explain how they alone detect arbitrary rewording.

The current five works in two groups are for training only. Test should draw from newly acquired public-domain works and fresh private synthetic worlds; validation needs a third disjoint set. The ten authored `world-*` development groups already appear in `task-generation-v1.json: excluded_source_groups`; keep them for validation and grader calibration, never training. The six `longform-v1` cases remain reserved from training, development cases, and checkpoint selection, as the module docstring requires.

`longform-v1` is the project's own benchmark, using its taxonomy, reward, public-domain slices, and checks. EQ-Bench Longform is a pinned third-party release, identified by `EQ_REVISION` in `longform.py`. It uses 12 prompts, a 13-step protocol that plans then writes eight chapters, 14 weighted criteria, negative criteria, and an adaptation of its own scoring arithmetic. It shares neither the internal task distribution nor the internal reward. The source describes it as an anchor that cannot be gamed by our reward; independence supplies the intended check, although that absolute claim is not demonstrated here.

Comparability requires EQ-Bench's reply-only `faithful` mode. `workspace` mode adds file delivery and is not comparable. Every external score must state mode and chapter count. Follow the upstream protocol for external rollouts; retain `samples_per_case: 4` for `longform-v1` and 12 per internal task, totaling 720 internal samples.

## The full requirement catalog and why each item matters

Each property is assigned one category in the original catalog. Later test-specific rules strengthen some coverage properties into mandatory gates; that exception needs to be made explicit in implementation.

### Mandatory requirements: H1–H13

H1, `family`, covers F1–F5 equally because different families activate different quality metrics and therefore change the reward mix. The referenced metric labels are Q2 (prose quality), Q5 (planning usefulness), Q7 (KB faithfulness), Q8 (important-information coverage), Q9 (knowledge interpretation), Q10 (navigability), Q11 (update correctness), Q12 (retrieval success), and Q13 (continuity). Equal family counts support comparisons across activities; the original schema also derives delivery channel from family.

H2, `instruction_specificity`, balances L0–L3. Which decisions are stated is the declared primary variation; imbalanced levels would confound other behavior comparisons, and assigned variation that never appears in the brief would not test it. Four equal levels are affordable at 300 tasks.

H3, `withheld_decision_point`, requires each brief to state exactly the family-specific `DECISION_POINTS` at or below its level cutoff and withhold the rest, as computed by `split_spec`. A leaked or omitted decision undermines the intended test of handling missing information.

H4, `ask_modality`, distinguishes `fully_stated` (nothing relevant left to clarify), `ask_required_withheld` (a missing decision requires a question), and `default_safe_withheld` (the writer may state a reversible default). Fully stated controls discourage asking reflexively; consequential missing decisions teach when asking matters. The proposal requires these distinctions but does not provide exact counts for all three.

H5, `deliverable_channel`, uses reply-required, file-required, and free-choice tasks. Only free-choice tasks make a channel decision observable. H6, `tool_availability`, varies no tools versus tools, including read-only and read/write access. H7 checks their joint distribution: six possible combinations, of which file-required without tools is impossible. Every feasible combination must appear so that channel does not determine tool access, subject to the unavoidable file-writing requirement.

H8, `genre_blend_size`, uses 0, 1, or 2. It changes writing difficulty, authoring demands, and the scope of continuity checks; equality is cheap and it is already an intended generator property.

H9, `stages`, uses 1, 2, 3, and 4+. Stage count affects both followups and reward aggregation, `0.5·mean(stage) + 0.5·final_state`. Requiring the deep category removes the existing ceiling rather than leaving longer interactions to chance. H10, `steer_count`, uses 0, 1, 2, and 3+, counted as `len(followups)`, to ensure practice revising under corrective feedback instead of only producing an initial answer.

H11, `provenance`, controls human, partly synthetic, and fully synthetic sources. Human sources provide grounding, while synthetic/private worlds support generalization and contamination control. Exact proportions make the intended mix auditable; the existing five works in two groups are too narrow.

H12, `failure_mode_coverage`, requires word-range, wiki-link, retrieval/evidence, protected-content, exclusion, nonempty-output, and continuity checks. If a check is absent, training cannot reward improvement on it; if it is rare, a policy may do well while mostly avoiding that difficulty.

H13 requires source-lineage separation. Randomly dividing derivatives exposes the same characters and facts in multiple pools and can inflate held-out scores. This is a build-integrity rule, recorded by content hash through `longform_suite.holdout_audit`, rather than an assessment of an authored task's quality.

### Minimum repeated coverage: C1–C14

C1, `genre`, covers 60 catalog values with a training floor of three each, compared with an average of about five at 300 tasks. A floor permits close continuations to retain the source genre; forced equality could demand an incompatible genre. C2, `style`, covers 20 values at least ten times each. Style is a principal intended remedy for repetitive prose, but close continuations may also need to preserve source style rather than obey equal counts.

C3, `trope`, covers 40 patterns, and C4, `situation`, covers 32 circumstances. Narrative breadth should reduce template plots. Because an author may replace incompatible ingredients, their enforceable floors apply to assignments, with realized changes reported later. C5, `continuity_challenge`, covers 12 kinds of consistency challenge at least 15 times each out of 300. These are the skills judged in F4/F5; all 12 can be grounded in source material, but fit to the source makes equality unnecessary.

C6 proposes `emotional_register`, a new vocabulary key with 10–12 values, including warm, elegiac, comic, tense, dread, wonder, grief, wry, tender, and hopeful. The concern is tonal as well as lexical repetition, and repeated coverage prevents each register from appearing just once.

C7, `pov_person_tense`, combines point of view and tense: first-person past, first-person present, close-third-person past, close-third-person present, omniscient past, and second-person present. Only close third exists in the measured material. Viewpoint is a writing-control skill, but loose tasks may legitimately leave it open; this motivates floors rather than equality and exposes the missing viewpoint decision point discussed below.

C8, `word_budget_envelope`, covers open, 120–220, 400–800, and 800–1200 words. The instruction level controls whether a budget is stated; the proposed envelope is the remaining variation. Both short and long ranges are needed so that the model cannot specialize in one band. How an unstated budget is counted under an envelope remains unspecified.

C9, `continuity_load`, counts facts that must survive: 0–2, 3–5, or 6+. Development fact sets are small; requiring the high-load group exercises deeper continuity. C10, `deferred_reveals`, counts questions that must stay unresolved: 0, 1, or 2+. It tests retaining uncertainty without inventing an answer, one of the 12 continuity challenges and a documented reward check.

C11, `revision_depth`, distinguishes creation, one edit, and cumulative multiple edits. The current pool has almost no cumulative editing, so minimum coverage forces that skill to appear. C12, `context_size`, covers <2k, 2–8k, 8–16k, and 16–28k tokens. Long-context retrieval and compaction are project goals; the frozen long-form suite reaches only the top category, and training must not remain entirely short.

C13, `source_work`, is training-only: use at least 12 works and repeat each enough to exercise it. The source says floors prevent a work from dominating, although it gives no upper bound that would actually prevent dominance. C14, `artifact_count`, distinguishes single from multiple outputs among file-capable tasks. Multi-file delivery is a scored failure mode absent from the current pool; its target is conditional on channel, so the proposal favors a minimum rather than equality.

### Sampled variation: S1–S6

S1, `steer_position`, varies where feedback lands in the stage sequence. Each position is valid and adds dialogue realism, but giving every position a minimum would multiply the steer-count requirements; aggregate entropy is the proposed check. S2 samples the second genre in blends: the 60×59 possible pairings are too numerous and unnecessary to cover exhaustively, so only blend size is gated.

S3 samples settings from source settings, already-authored `worlds.json`, and new private worlds; the count of distinct worlds is reported. S4 samples meaning-preserving `instruction_phrasing` rewrites as a robustness probe: wording should not change the reward. S5 samples `thinking` or reasoning budget off/on for an ablation, a comparison isolating that change; balancing it was explicitly deferred in `training-distribution-axes.md`. S6 samples cast size and relationship type, such as two focal characters or an ensemble, because they add breadth without a stated reward dependency.

### Measurements after generation: R1–R9

R1 compares `realized_variation` with the requested `assignment`; an ingredient actually used is stronger evidence than the assigned label. R2 measures prose distributions with `prose.score_prose`: D1 measures distance between generated and reference word-sequence distributions; D2 measures distance between distributions of text embeddings, numerical representations of meaning; and D4 is self-BLEU, the overlap measure described above. Lower D1/D2 means closer to the chosen reference, not automatically better writing; higher D4 means more repetition. R2 also records collisions in opening phrases. These are outcomes the design wants to improve, and turning them into generation constraints would invite optimizing the numbers themselves.

R3 compares semantic continuity scores with the assigned challenge to see whether the task actually exercises it. R4 reports delivered words minus declared budget, keeping overshoot visible without prescribing a model policy. R5 records within-group reward standard deviation and `frac_reward_zero_std`, the proportion of attempt groups with no reward spread, as GRPO health indicators and proposed early signs of reward overfitting; the references are `reward.py` and `rl-algorithm-decision.md`.

R6 reports maximum mean discrepancy (MMD), a distribution-distance measure used for comparison with reference texts, only at or above the 20-sample floor. R7 records tool, step, and artifact counts from harness traces to detect tasks that need more tool use than budgeted. R8 reports the lineage/hash-overlap audit per release to confirm H13. R9 reports vocabulary entropy for each property so that sampled variation concentrating on only a few values remains visible.

## Proposed implementation checks, preserved for review

The following pseudocode retains the source's assertions, including their unresolved conflicts. `N` is pool size, `counts[a][v]` counts tasks with property `a` equal to `v`, and `//` is integer division. The H codes refer to the requirements just defined. `ASK_REQUIRED` is the behavior of asking about a consequential missing decision. In the final conditional-probability check, `P(tools | reply_required)` means the fraction of reply-required tasks with tools.

These checks apply to the assignment records that
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

The original applies these predicates to TEST using its test targets and the stronger failure-mode floor (every kind ≥6, at least two required), and to VAL using the one-fifth (1/5) targets. The code block’s “§3” refers to the original target-distribution section, reproduced in the tables above.

Two assertions worth stating explicitly because they are the measured defects:

```
channel confound:  P(tools | reply_required) != 1  and  P(tools | channel_free) in (0, 1)
                   (currently P(tools | reply) == 0 and P(tools | file) == 1; both must move)
safety net:        assert all(r.followups_expected == len(r.stage_families) - 1 for r in pool)
```

`H13` is not an LLM task-quality assertion; it is a build-time hash invariant and
should reuse `longform_suite.holdout_audit`, which raises rather than warns.

## Dependencies and decisions this proposal leaves open

The original proposal flags six content and implementation dependencies:

1. Viewpoint is scored but has no specificity decision point. `src/.context/CONTEXT.md` says loose prompts remove checks for unstated length, viewpoint, idea count, and ending; the baseline says optional word-budget and viewpoint/style checks each failed nine times. But `DECISION_POINTS` contains no `viewpoint`, `idea_count`, or `ending`. C7 cannot make a withheld viewpoint ask-required or default-safe. Either add viewpoint or drop the claim that loose tasks govern it.
2. Deep interaction requires changing `_body`, which currently constructs at most three stages. H9/H10 depend on extending `stage_families` and the followups = stages − 1 contract in `work/sft/task-generator-instructions.md`.
3. Free channel choice is not representable in the current schema. `visible.prose` and `visible.tools` are derived from family, with reply for F1/F3 and files otherwise, while the long-form compiler hard-codes `[] if replies else WRITE_TOOLS`. H5/H6/H7 need a channel-requirement field and a scoring rule for a file write when reply was also permitted.
4. The 54 accepted tasks are real, but the request metadata is stale. `generated/manifest.json` says requested 100 and accepted 54; schema-v2 `requests.json` says generated_tasks 0 and accepted_tasks 0 because it is a pre-generation copy. Use the manifest for inventory and regenerate requests under `SCHEMA_VERSION = 3`, which carries the level objects needed by H2/H3.
5. The vocabulary is incomplete. The original names `genre`, `emotional_register`, and `pov_person_tense` as not all present; the actual catalog supplies the plural keys genres/styles/tropes/situations/continuity_challenges. C6/C7 need new entries before their minimums can be enforced.
6. Source acquisition blocks the proposed provenance, lineage, and work requirements. Five works in two groups are insufficient. `work/custom-eval-suite/data.md` lists candidates, but no webnovel or wiki has confirmed reuse terms. The proposal requires generation to fail closed until at least 12 training works and six lineage groups exist.

The editorial audit also found conflicts that wording cannot resolve: the incompatible training tool totals; L0/L3 reversed in H4; validation scaling rules and provenance that disagree; unclear test vocabulary floors; an unclear file-capable denominator; “exact” mandatory categories implemented as floors; a channel-probability assertion that does not exclude the current zero-tool reply condition; unspecified clarification counts; and unsupported statistical or reuse-cap claims. The [lane audit](../lane-b-audit.md) quotes each relevant passage and explains the decision needed. These have not been repaired by changing the numbers.

The original open decisions remain: adopt 300/60/60 or the 512 training stretch; approve the free-channel and viewpoint schema additions; approve a new deep-interaction stage rule; and confirm the external benchmark's faithful-mode protocol and method for the RL comparison.

## Status, generation architecture, and definition sources

The original status is “proposed, pre-generation.” This document governs both pools as a specification; it does not apply changes to `src/`, `tests/`, `data/`, or `runs/`. Code symbols reference the current schema for future assertions. The generation-wave reports explain that their runs and this proposal were developed concurrently, so the wave outputs are not evidence that this specification was followed.

Generation must keep vocabulary ingredients in `data/training/variation-catalog-v1.json`, combine them in `src/writing_agent/task_generation.py`, and have a language model author prose/content into slots under `work/sft/task-generator-instructions.md`. No finished task JSON may be hand-written. Balance targets therefore apply to assignment fields rather than literal task text.

Definitions added for readability were checked against `src/writing_agent/specificity.py`, `src/writing_agent/task_generation.py`, `src/writing_agent/scoring.py`, and `src/writing_agent/prose.py`. The [original specification](../../training-data/balance-spec.md) remains unchanged.
