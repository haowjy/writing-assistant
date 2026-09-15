# Branching fiction tasks and task-specific RL rewards

Selected direction, 2026-09-15. GLM-5.3 through Reka is the preferred training judge,
task generator, and simulated author, using separate role-specific calls. The user
has settled training-use permission; literary calibration remains unverified. Train on
the same five task families and harness capabilities as the benchmark, using separate
source works, task instances, and derivatives. Share scoring implementations where
appropriate; keep held-out benchmark examples and private evaluation labels out of
training. See the [SFT/RL research](rl-bootstrap-research.md).

The [multi-turn session design](multi-turn-rl.md) composes these families through
author feedback, revisions, shared project state, and eventual context compaction.
A session can exercise several families; the initial single-family cases are only
one sampling condition.

## What scores an attempt

### Proposed initial scalar reward: version 0

Start with `raw = 0.40 * quality + 0.30 * intent + 0.20 * continuity + 0.10 * mechanics`.
All four components range from zero to one. These weights are a starting hypothesis,
not validated experimental findings or an implemented RL reward adapter.

| Component | Initial scorer | Meaning |
|---|---|---|
| Quality | GLM-5.3/Reka | Literary effectiveness for prose; useful distinct alternatives for planning; useful selection and organization for KB tasks |
| Intent | GLM-5.3/Reka | Achieves the requested transformation, revision or other author goal; excludes mechanical delivery checks |
| Continuity | GLM-5.3/Reka with source evidence | Respects applicable source/KB facts, character knowledge and accepted decisions; allows authorized divergences |
| Mechanics | Existing deterministic checks | Correct delivery, protected content, paths, tools and navigation where applicable |

Use anchored 1–5 semantic ratings mapped by `(rating - 1) / 4`: failed, major issues,
usable with substantial revision, effective with minor issues, fully effective.
Require cited evidence and uncertainty alongside each semantic component. Quality
is task-dependent; do not score a KB as if it were a literary scene.

Mechanics is the mean of task-declared applicable checks, with delivery as a minimum
check on every initial task. Fix that check set before generating rollout groups;
an inapplicable link check does not earn a direct-prose task free points. A recovered
tool error is diagnostic rather than automatic failure. Tool-call count and file
count are not positive objectives.

For a preregistered critical failure, use `reward = min(raw, 0.25)`; otherwise use
`reward = raw`. Examples include a missing required final artifact, an unauthorized
protected edit, or violation of an explicitly critical canon constraint confirmed
against source evidence. Declare critical criteria before sampling; do not promote
an ordinary rubric weakness into a critical failure after seeing the answer. Assess
recoverable delivery/consistency failures at the required stage or final boundary.
Permitted fanfiction departures are not canon failures.

The cap retains partial learning signal while limiting compensation by polished
writing. It does not guarantee a failed attempt receives negative GRPO advantage:
relative ranking depends on the other group members. Monitor all-fail/all-equal
groups and adjust curriculum or demonstrations rather than interpreting higher
relative reward as absolute success.

If a required artifact is absent, its quality component is zero; a receipt or prose
written elsewhere does not substitute for it. A judge timeout or insufficient
evidence is different: mark reward unavailable and leave the rollout group pending
until resolved. Do not drop only inconvenient judgments or silently reweight them.

For initial two- or three-stage sessions, propose `0.5 * mean(stage_rewards) +
0.5 * final_state_reward`, applying the same critical-failure cap to unresolved
mandatory failures. The final-state assessment checks the current manuscript/KB
against active author requirements and persistence of earlier accepted decisions.
Superseded requirements are removed; repeated checks are intentionally tracking
persistence, not counted as independent evidence. Record this aggregation separately
from the later RL algorithm's per-action credit assignment.

Do not initially add MMD, n-gram distance, lexical diversity, a pretrained reward
model score, or a token-efficiency bonus to this scalar. Retain them as diagnostics
where applicable, then consider additions only after checking whether optimization
improves writing rather than gaming the measurement. Keep rubric and reward versions
fixed within a training experiment. Validate version 0 on controlled failures and
matched good/weak examples before its first policy update.

RL needs a reward function, not necessarily a language-model judge. Use mechanical
rewards for outcomes that code can verify and semantic judgments where meaning matters.
A mixed reward can include both. The existence of a requested file earns delivery
credit; it does not establish that its prose satisfies the request.

| Task | Mechanical evidence | Semantic evidence |
|---|---|---|
| Direct prose | Designated reply extracted; explicit length/format constraints; overlap diagnostics | Requested transformation, continuity, prose quality |
| File authoring/revision | Requested path and nonempty artifact; protected content preserved; patch scope | Requested revision achieved; file contains suitable prose |
| Planning | Requested number of proposals when explicitly specified | Distinct, useful alternatives compatible with author intent |
| KB construction/maintenance | Links resolve; pages reachable; required entries/files present; citations target real sources | Useful extraction, supported interpretation, qualified uncertainty, accepted updates |
| Writing from KB | Requested delivery; tool restrictions; source access when required | Uses relevant knowledge accurately; develops the requested branch |

Record components and applicability separately. Do not award positive scores for
inapplicable checks or silently redistribute their weight after a grading failure.
Keep missing semantic judgments explicit. For initial mechanical-only RL, choose tasks
whose intended learning objective is mechanically checkable; do not report that stage
as optimizing literary quality. Do not reward tool-call count, file count, or a receipt
claim as substitutes for the requested outcome.

Use partial credit without allowing easy delivery or formatting checks to dominate
content success. The scalar combination, gating, and within-group advantages still
need validation: an overall failure can receive positive relative advantage if all
other attempts are worse. Track all-fail/all-equal groups and adjust task difficulty
or add targeted demonstrations when useful behaviors are too rare.

[RLMR](https://arxiv.org/html/2508.18642v1) provides evidence for separating writing
quality and constraint compliance during RL. Its learned verification component is
not equivalent to our exact file checks. [Writing-Zero](https://arxiv.org/html/2506.00103v1)
uses a writing-specific pairwise judge; subjective judgments remain fallible even when
converted into a numerical reward.

## Generate a task and its evidence together

A reusable source preparation stage extracts chapter/scene boundaries, cited facts,
character knowledge, unresolved questions, and recent prose. Preserve edition hashes,
source spans, extraction provenance, and confidence. Verify reconstructed notes against
source passages before treating them as reference material.

An on-demand task generator chooses a training work, checkpoint, transformation, task
family, delivery mode, style, and instruction specificity. It emits:

- **Visible task:** the author's actual request, relevant source context, initial files,
  available tools, and budgets. A vague request must permit reasonable interpretations.
- **Branch contract:** source facts that remain binding, requested changes, explicitly
  permitted departures, accepted events versus proposals, and unresolved state. Private
  criteria may operationalize the visible request, not impose a secret chosen plot.
- **Private grading material:** cited source evidence, applicable mechanical checks,
  semantic questions, defensible alternative interpretations, and comparison passages
  for overlap analysis. Record the purpose and granularity of any requested KB.
- **Reproduction metadata:** parent work/branch, cutoff, source role, generator/model,
  seed, task version, hashes, transformations, and reward configuration.

The generator is another model role if transformations require invention; it is not
necessarily the writer or the reward judge. Model and provider permissions apply to
that generated training material too. It must not certify its own invented source facts
as true merely by placing them in private labels.

## Example: Moby-Dick transformations

These are proposed task types, not newly generated training records:

- **LitRPG adaptation:** specify which setting, relationships, and prior events remain;
  allow the genre transformation. Assess whether game-like progression or mechanics
  actually influence decisions and consequences. The mere presence of levels or a
  stat panel is weak evidence of success.
- **Major event divergence:** introduce a new event at an identified cutoff, mark it as
  accepted branch history, and ask for planning, prose, or a KB update. Score plausible
  consequences from that new state. Later original events are optional after divergence.
- **Close continuation:** preserve the established style/state with limited novelty.
  This needs a different originality expectation from a radical rewrite.

The requested transformation overrides incompatible original canon. Unknown future
facts are not errors just because the source book eventually supplies an answer.
Retain invariants only when supported by the context and compatible with the request.

## Judge context and departure from the source

Supply the semantic judge with the author request, branch contract, relevant cited
passages, and extracted candidate artifact. For KB tasks also provide source evidence
independent of the candidate's KB, so the output cannot become its own ground truth.
For longer histories, use layered summaries and retrieval, with access to underlying
passages. If necessary evidence is missing, retrieve it or record an unscorable judgment.
Do not substitute an assumed memory of a famous novel.

Measure two different things:

1. **Textual reuse:** long exact matches, n-gram overlap, and comparison spans. Exclude
   supplied quotations or names where appropriate; retain evidence rather than treating
   every shared phrase as a failure. Use private later-original passages only to analyze
   reuse, never to impose future canon on the writer.
2. **Requested narrative departure:** whether the changed event or genre has meaningful
   consequences. A semantic judge can flag a continuation that paraphrases the original
   plot while ignoring the requested intervention.

More distance is not universally better. Close imitation, faithful continuation, and
radical adaptation require different expectations. MMD remains a group/corpus prose
profile, not a test that one scene adequately transforms its source.

## Scale without losing control

Treat “infinite” as on-demand recombination, not unlimited independent information.
Generate a candidate task, validate source grounding and satisfiability, deduplicate,
then admit it to the training stream. Track coverage across works, interventions,
genres, task families, prompt specificity, and context lengths. Reuse prepared source
packets to amortize extraction and grounding work.

Freeze each admitted task and initial state across the members of an RL rollout group.
Generate varied attempts from that same task; generating a different task for every
member confounds relative rewards. Save every realized task so a streamed experiment
can be reproduced. The [TRL environment interface](https://huggingface.co/docs/trl/grpo_trainer)
supports stateful rollouts, but integration with our native Gemma harness remains work.

Split source works and derivatives before task generation. Keep final evaluation tasks
frozen and use separate unseen transformation combinations to test generalization.
Adapt the training-task curriculum from training-development evidence rather than
recycling failures from final evaluation into training examples.

## Next preparation work

1. Specify the source packet and branch contract, using independently grounded examples.
2. Create mechanically checkable training tasks for delivery, protected edits, and wiki
   navigation, plus semantic tasks that remain separate until a judge is validated.
3. Validate GLM-5.3 through Reka on source-backed comparisons, deliberate continuity
   errors, ineffective transformations, and near-copying; test order sensitivity and
   misleading instructions embedded in candidate text. Keep Astra as the held-out evaluator.
4. Verify task replay and group isolation, then propose a bounded RL feasibility run.
   Neither this document nor on-demand generation authorizes unbounded training or cost.

The [compiled research and generation workflow](training-data-research.md) defines
the first 100 task assignments and distinguishes prepared requests from generated,
reviewed tasks and successful SFT trajectories.
