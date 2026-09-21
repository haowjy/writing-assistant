# RL algorithm decision: critic-free group methods, not PPO

Decision, 2026-09-19. Status: settled for the current RL feasibility stage.

## Decision

Use a critic-free, group-relative optimizer for the RL stage. Target TRL 1.13's
`GRPOTrainer` (with `RLOOTrainer` as a fallback), not PPO. Do not build a staged
GRPO → freeze → train critic → PPO pipeline. Enable the DAPO/Dr.GRPO loss variants and
monitor per-group reward variance (`frac_reward_zero_std`) rather than adding a critic.

## Why

- **The reward is terminal and scalar.** One 1–5 semantic score plus mechanical checks
  arrive at the end of an artifact; there is no per-token signal. With a single terminal
  reward, GAE reduces to `R − V(s_t)` at λ=1, and its signal decays as `λ^(T−t)` at λ<1 —
  so a critic cannot invent the per-token credit the judge never produced. RLOO reports
  PPO's clip fires on <5% of tokens and that modeling partial completions is unnecessary
  ([RLOO](https://arxiv.org/abs/2402.14740)).
- **Hardware.** A PPO critic is typically policy-sized. It will not fit beside
  Gemma E2B-IT + reference + judge on one RTX 3090, and staging would pay both group
  sampling and critic memory ([DeepSeekMath](https://arxiv.org/abs/2402.03300),
  [Tülu 3](https://arxiv.org/abs/2411.15124)).
- **Tooling.** TRL 1.13 documents GRPO and RLOO trainers, not a PPO trainer.
- **Writing-RL practice.** Published writing-RL methods are critic-free: Writing-Zero
  (BRPO), RLMR (GRPO + mixed rewards), RLCS (GRPO-trained GenRM).
- **Group normalization suits a biased judge.** Within-group `(r − mean) / std` cancels
  a per-prompt additive offset or positive scale in the judge. It does not cancel rank
  flips, length bias, or all-tie groups.

## Rejected alternative: GRPO → freeze → critic → PPO

No published method stages GRPO then PPO. The closest recipe, VC-PPO
([Yuan et al. 2025](https://arxiv.org/abs/2503.01491)), warm-starts a critic on a frozen
**SFT** policy and then trains actor **and critic jointly** — the critic is never left
frozen while the policy moves, because a critic fit to a frozen policy is off-policy as
soon as PPO updates. The proposal also pays GRPO's group sampling, a critic pretrain, and
a policy-sized critic on one GPU. VC-PPO's own fix (λ_value = 1.0) is an admission that
the TD machinery is the failure mode on long sequences, not the feature.

The strongest pro-critic evidence is not on fiction: VRPO
([2508.03058](https://arxiv.org/abs/2508.03058)) shows a regularized value model absorbing
reward noise in dialogue, and VC-PPO improves long-CoT math. Neither transfers to a 7–8B
fiction policy with a 1–5 judge. No PPO-vs-GRPO ablation on fiction at this scale was found.

## Reward the algorithm consumes (for reference)

Mixed, not judge-only: deterministic mechanical checks (delivery path, protected content,
link reachability, requested counts, format) plus a semantic judge for quality, intent,
and continuity. Scalar
`raw = 0.40 * quality + 0.30 * intent + 0.20 * continuity + 0.10 * mechanics`, with a
critical-failure cap at 0.25. See [task rewards](rl-task-generation.md) and
[information value](information-value.md). The mechanical half scales to a large wiki
(navigation and reachability); the semantic half sees retrieved evidence, not the whole
wiki — see the open risk below.

## Open adjacent question (not decided here)

Pointwise 1–5 scalar vs pairwise/rank judge output. Writing-Zero shows a scalar RM + GRPO
hacks (long explanations, gibberish) while a pairwise GenRM resists; RLMR found a writing
RM ranking a constraint-violating slogan above a compliant one, fixed by a dynamic penalty
that drives violators negative. Our mixed reward has the same mechanical-vs-semantic
tension. Pairwise/rank reward is the lever that most reduces reward hacking, and it sits
in reward design, not the optimizer. Treat it as the next reward-design decision.

The semantic half depends on retrieval: if relevant source/KB passages are not retrieved,
the judge cannot flag continuity or knowledge errors. This is the same KB-retrieval
coverage gap recorded in the custom-eval output review.

## Revisit if

- Training shifts to verifiable or dense token-level rewards where a critic is identified.
- Multi-GPU training makes a policy-sized critic affordable, and a calibrated critic beats
  a group baseline on a held-out writing comparison.
- TRL ships a supported PPO trainer and the reward stops being terminal and scalar.

## Evidence

- [PPO vs GRPO survey](rl-bootstrap-research.md) and the algorithm-by-lab report from the
  prior session (`p8` work dir).
- Staged-pipeline and critic-value report (`p9` work dir): VC-PPO, RLOO, V0, VRPO,
  Writing-Zero, RLMR, RLCS.
