"""Turn a judged rollout into the scalar a group-relative optimizer consumes.

Training-side only. [scoring.py](scoring.py) owns the evaluation scorecard and
deliberately computes no combined literary score; this module exists because RL needs
one number per rollout, and it must not leak into the evaluation path.

The scalar is version 0 of a hypothesis about what "good collaboration" means, not a
validated measure of writing quality. Optimizing it can improve the measurement rather
than the writing; treat every change as a reward-design experiment and re-validate
against held-out human judgments. Rules and their rationale live in
`work/sft/rl-task-generation.md`.

Three design choices are load-bearing enough to state here:

- A rollout is *conditioned on* task variables and *invariant to* nuisance variables.
  Nothing in this module reads instruction phrasing, the tool envelope, or partner
  identity, because a reward that moved with those would be measuring the wrong thing.
- A critical criterion only counts if it was declared before sampling. An ordinary
  rubric weakness cannot be promoted after seeing the answer, because that would let a
  reviewer encode the outcome they wanted.
- Reward is convergence to the author's actual hidden preference, never the act of
  asking a clarifying question. Rewarding the act invites an always-ask policy.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from writing_agent.catalog import fingerprint

REWARD_VERSION = 0
COMPONENTS = ("quality", "intent", "continuity", "mechanics")
SEMANTIC_COMPONENTS = ("quality", "intent", "continuity")
WEIGHTS = {"quality": 0.40, "intent": 0.30, "continuity": 0.20, "mechanics": 0.10}
RATING_MIN, RATING_MAX = 1, 5
CRITICAL_CAP = 0.25
STAGE_WEIGHT = 0.5

# A component is scored from an anchored rating, scored as an explicit zero because the
# required artifact is absent, or withheld because the judge failed. The middle case is
# a low score; only the last one is a reason to leave the group pending.
SCORED, ABSENT, UNAVAILABLE = "ok", "absent", "unavailable"


@dataclass(frozen=True)
class Reward:
    """One rollout's scalar, or a named reason it has none yet."""

    status: str
    value: float | None = None
    components: dict = field(default_factory=dict)
    reason: str | None = None
    critical: bool = False

    def __post_init__(self):
        if self.status == SCORED:
            if self.value is None or not 0.0 <= self.value <= 1.0:
                raise ValueError("A scored reward needs a value in [0, 1]")
        elif self.status == UNAVAILABLE:
            if self.value is not None:
                raise ValueError("An unavailable reward carries no value")

    @property
    def available(self) -> bool:
        return self.status == SCORED


def rating_to_unit(rating) -> float:
    """Map an anchored 1-5 rating onto [0, 1]. Rejects anything off the anchor set."""
    if type(rating) is not int or not RATING_MIN <= rating <= RATING_MAX:
        raise ValueError(f"Rating must be an integer in [{RATING_MIN}, {RATING_MAX}]: {rating!r}")
    return (rating - RATING_MIN) / (RATING_MAX - RATING_MIN)


def mechanics_score(checks: list[dict]) -> float:
    """Mean of the task's applicable mechanical checks.

    An inapplicable check is excluded rather than counted as free credit, so a
    direct-prose task is not rewarded for a link check it never had. Repeating a check
    is persistence tracking, not a second chance to collect reward, so an outcome counts
    once; the same check disagreeing with itself is a bug, not a signal.
    """
    outcomes = {}
    for check in checks:
        if not check.get("applicable", True):
            continue
        identity, passed = check["id"], bool(check["passed"])
        if identity in outcomes and outcomes[identity] != passed:
            raise ValueError(f"Repeated check disagrees with itself: {identity}")
        outcomes[identity] = passed
    if not outcomes:
        raise ValueError("No applicable mechanical checks; the check set is declared up front")
    return sum(outcomes.values()) / len(outcomes)


def declared_check_set(checks: list[dict]) -> dict:
    """Freeze the mechanical check set so it cannot drift after rollouts are sampled."""
    ids = sorted({c["id"] for c in checks if c.get("applicable", True)})
    if not ids:
        raise ValueError("No applicable mechanical checks to declare")
    return {"ids": ids, "hash": fingerprint(ids)}


def critical_failures(declared, observed) -> list[str]:
    """Declared criteria that actually failed. Undeclared failures never count.

    Accepts the frozen set from `declared_check_set` directly. Passing that dict where an
    iterable of ids is expected iterates its keys, which matches nothing and silently
    reports no critical failure at all.
    """
    if isinstance(declared, Mapping):
        declared = declared.get("ids", ())
    return sorted(set(declared) & set(observed))


def _component_unit(judgment: dict, name: str) -> float | None:
    entry = judgment.get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"Missing judgment for {name}")
    status = entry.get("status", SCORED)
    if status == ABSENT:
        return 0.0
    if status == UNAVAILABLE:
        return None
    if "rating" not in entry:
        raise ValueError(f"{name} judgment needs a rating or an explicit status")
    return rating_to_unit(entry["rating"])


def rollout_reward(
    judgment: dict,
    checks: list[dict],
    *,
    declared_critical=(),
    observed_failures=(),
) -> Reward:
    """Score one attempt, or withhold the score until an incomplete judgment resolves.

    Withholding is deliberately not a zero. A judge timeout and a bad draft are
    different events, and averaging them would teach the policy to be unlucky.
    """
    units = {}
    for name in SEMANTIC_COMPONENTS:
        unit = _component_unit(judgment, name)
        if unit is None:
            return Reward(
                status=UNAVAILABLE,
                reason=f"{name} judgment unavailable; leave the group pending until resolved",
            )
        units[name] = unit
    units["mechanics"] = mechanics_score(checks)
    raw = sum(WEIGHTS[name] * units[name] for name in COMPONENTS)
    failed = critical_failures(declared_critical, observed_failures)
    if failed:
        return Reward(
            status=SCORED,
            value=min(raw, CRITICAL_CAP),
            components=units,
            critical=True,
            reason="Declared critical failure: " + ", ".join(failed),
        )
    return Reward(status=SCORED, value=raw, components=units)


def session_reward(
    stages: list[Reward],
    final_state: Reward,
    *,
    declared_critical=(),
    unresolved=(),
) -> Reward:
    """Combine stage scores with the final project state.

    A final average alone hides a failed crucial stage; an all-or-nothing score is too
    sparse to learn from. Stages that were never reached are simply absent, not zero.
    """
    if not stages:
        raise ValueError("A session needs at least one scored stage")
    pending = [r for r in stages + [final_state] if not r.available]
    if pending:
        return Reward(
            status=UNAVAILABLE,
            reason="Unresolved stage or final-state judgment; leave the group pending",
        )
    combined = (
        STAGE_WEIGHT * sum(r.value for r in stages) / len(stages)
        + (1 - STAGE_WEIGHT) * final_state.value
    )
    failed = critical_failures(declared_critical, unresolved)
    # A stage already capped for a critical failure must not be diluted by a clean final
    # state. Requiring the caller to restate the same failure through `unresolved` made
    # the cap depend on remembering to say it twice.
    capped = [f"stage {index}" for index, reward in enumerate(stages) if reward.critical]
    if final_state.critical:
        capped.append("final state")
    if failed or capped:
        return Reward(
            status=SCORED,
            value=min(combined, CRITICAL_CAP),
            components={"stages": [r.value for r in stages], "final": final_state.value},
            critical=True,
            reason="Unresolved mandatory failure: " + ", ".join([*capped, *failed]),
        )
    return Reward(
        status=SCORED,
        value=combined,
        components={"stages": [r.value for r in stages], "final": final_state.value},
    )


def group_advantages(rewards: list[Reward]) -> dict:
    """Within-group standardised rewards, which is what a critic-free method consumes.

    Group normalisation cancels a per-prompt offset or positive scale in the judge. It
    does not cancel rank flips or length bias, and it hides an all-tie group entirely —
    hence the zero-variance fraction, which is a curriculum signal rather than a
    statistic to tolerate.
    """
    if not rewards:
        raise ValueError("A rollout group needs at least one reward")
    if any(not r.available for r in rewards):
        return {
            "status": "pending",
            "advantages": [],
            "reason": "Group has an unresolved reward; do not take an update from it",
        }
    values = [r.value for r in rewards]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance**0.5
    # Variance is zero exactly when every reward is equal, so test the rewards rather than
    # the derived deviation. Three rewards of 0.1 average to 0.10000000000000002, leaving
    # a std near 1.4e-17; dividing that residue by itself yields a spurious advantage of
    # +-1, so a tied group looks like a strong uniform signal instead of no signal at all.
    zero_variance = max(values) == min(values)
    advantages = [0.0] * len(values) if zero_variance else [(v - mean) / std for v in values]
    return {
        "status": "ok",
        "advantages": [advantage + 0.0 for advantage in advantages],
        "mean": mean,
        "std": std,
        "zero_variance": zero_variance,
        "frac_zero_std": 1.0 if zero_variance else 0.0,
    }
