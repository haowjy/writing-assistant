"""Reward policy: component mapping, critical caps, group normalisation."""

import unittest

from writing_agent.reward import (
    COMPONENTS,
    CRITICAL_CAP,
    SCORED,
    UNAVAILABLE,
    WEIGHTS,
    Reward,
    critical_failures,
    declared_check_set,
    group_advantages,
    mechanics_score,
    rating_to_unit,
    rollout_reward,
    session_reward,
)


def check(identity, passed, *, applicable=True):
    return {"id": identity, "passed": passed, "applicable": applicable}


def judgment(**ratings):
    return {name: {"rating": value} for name, value in ratings.items()}


class RatingTests(unittest.TestCase):
    def test_anchors_map_onto_the_unit_interval(self):
        self.assertEqual(rating_to_unit(1), 0.0)
        self.assertEqual(rating_to_unit(3), 0.5)
        self.assertEqual(rating_to_unit(5), 1.0)

    def test_off_anchor_ratings_are_rejected(self):
        for rating in (0, 6, 2.5, -1, "4", None, True):
            with self.subTest(rating=rating), self.assertRaises(ValueError):
                rating_to_unit(rating)


class MechanicsTests(unittest.TestCase):
    def test_score_is_the_mean_of_applicable_checks(self):
        self.assertEqual(mechanics_score([check("a", True), check("b", False)]), 0.5)
        self.assertEqual(mechanics_score([check("a", True)]), 1.0)
        self.assertEqual(mechanics_score([check("a", False)]), 0.0)

    def test_inapplicable_checks_earn_no_credit(self):
        checks = [check("delivery", True), check("links", False, applicable=False)]
        self.assertEqual(mechanics_score(checks), 1.0)

    def test_a_repeated_check_counts_once(self):
        checks = [check("persistence", True), check("persistence", True)]
        self.assertEqual(mechanics_score(checks), 1.0)

    def test_a_self_contradicting_check_is_a_bug(self):
        checks = [check("links", True), check("links", False)]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            mechanics_score(checks)

    def test_an_empty_check_set_is_a_declaration_error(self):
        with self.assertRaisesRegex(ValueError, "No applicable"):
            mechanics_score([])
        with self.assertRaisesRegex(ValueError, "No applicable"):
            mechanics_score([check("links", True, applicable=False)])

    def test_declared_set_freezes_applicable_identities(self):
        declared = declared_check_set(
            [check("b", True), check("a", True), check("c", True, applicable=False)]
        )
        self.assertEqual(declared["ids"], ["a", "b"])
        self.assertEqual(declared, declared_check_set([check("a", False), check("b", False)]))


class CriticalCriteriaTests(unittest.TestCase):
    def test_only_declared_criteria_can_fail(self):
        self.assertEqual(critical_failures(["missing-artifact", "canon"], ["canon"]), ["canon"])

    def test_an_undeclared_failure_never_counts(self):
        self.assertEqual(critical_failures(["canon"], ["weak-dialogue"]), [])
        self.assertEqual(critical_failures([], ["missing-artifact"]), [])

    def test_a_frozen_check_set_can_be_passed_directly(self):
        # Passing the frozen dict where ids were expected used to iterate its keys, match
        # nothing, and report no critical failure at all.
        frozen = declared_check_set(
            [check("canon", True), check("delivery", False, applicable=False)]
        )
        self.assertEqual(critical_failures(frozen, ["canon"]), ["canon"])


class RolloutRewardTests(unittest.TestCase):
    def test_weighted_sum_of_the_four_components(self):
        reward = rollout_reward(
            judgment(quality=5, intent=4, continuity=3),
            [check("delivery", True), check("links", False)],
        )
        self.assertEqual(reward.status, SCORED)
        self.assertAlmostEqual(reward.value, 0.775)
        self.assertFalse(reward.critical)

    def test_weights_sum_to_one(self):
        self.assertEqual(set(WEIGHTS), set(COMPONENTS))
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)

    def test_an_absent_artifact_scores_zero_rather_than_pending(self):
        reward = rollout_reward(
            {
                "quality": {"status": "absent"},
                "intent": {"rating": 4},
                "continuity": {"rating": 4},
            },
            [check("delivery", False)],
        )
        self.assertEqual(reward.status, SCORED)
        self.assertEqual(reward.components["quality"], 0.0)

    def test_a_failed_judge_withholds_the_reward(self):
        reward = rollout_reward(
            {
                "quality": {"status": UNAVAILABLE},
                "intent": {"rating": 4},
                "continuity": {"rating": 4},
            },
            [check("delivery", True)],
        )
        self.assertEqual(reward.status, UNAVAILABLE)
        self.assertIsNone(reward.value)
        self.assertIn("pending", reward.reason)

    def test_declared_critical_failure_caps_the_reward(self):
        reward = rollout_reward(
            judgment(quality=5, intent=4, continuity=3),
            [check("delivery", True)],
            declared_critical=["missing-artifact", "protected-edit"],
            observed_failures=["protected-edit"],
        )
        self.assertEqual(reward.value, CRITICAL_CAP)
        self.assertTrue(reward.critical)
        self.assertIn("protected-edit", reward.reason)
        self.assertNotIn("missing-artifact", reward.reason)

    def test_a_missing_component_is_an_error_not_a_zero(self):
        with self.assertRaisesRegex(ValueError, "Missing judgment"):
            rollout_reward(judgment(quality=5, intent=4), [check("delivery", True)])

    def test_a_ratingless_component_needs_an_explicit_status(self):
        with self.assertRaisesRegex(ValueError, "needs a rating"):
            rollout_reward(
                {"quality": {}, "intent": {"rating": 4}, "continuity": {"rating": 4}},
                [check("delivery", True)],
            )


class SessionRewardTests(unittest.TestCase):
    def _scored(self, value):
        return Reward(status=SCORED, value=value)

    def test_stages_and_final_state_are_weighted_equally(self):
        reward = session_reward([self._scored(1.0), self._scored(0.0)], self._scored(1.0))
        self.assertAlmostEqual(reward.value, 0.75)

    def test_a_pending_stage_makes_the_session_pending(self):
        pending = Reward(status=UNAVAILABLE, reason="judge timeout")
        reward = session_reward([self._scored(1.0), pending], self._scored(1.0))
        self.assertEqual(reward.status, UNAVAILABLE)

    def test_an_unresolved_mandatory_failure_caps_the_session(self):
        reward = session_reward(
            [self._scored(1.0)],
            self._scored(1.0),
            declared_critical=["canon-update"],
            unresolved=["canon-update"],
        )
        self.assertEqual(reward.value, CRITICAL_CAP)
        self.assertTrue(reward.critical)

    def test_a_session_needs_a_stage(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            session_reward([], self._scored(1.0))

    def test_a_capped_stage_survives_a_perfect_final_state(self):
        capped = Reward(status=SCORED, value=CRITICAL_CAP, critical=True, reason="stage cap")
        reward = session_reward([capped], self._scored(1.0))
        self.assertEqual(reward.value, CRITICAL_CAP)
        self.assertTrue(reward.critical)
        self.assertIn("stage 0", reward.reason)

    def test_a_capped_final_state_survives_a_perfect_stage(self):
        capped = Reward(status=SCORED, value=0.0, critical=True, reason="state cap")
        reward = session_reward([self._scored(1.0)], capped)
        self.assertEqual(reward.value, CRITICAL_CAP)
        self.assertIn("final state", reward.reason)


class GroupAdvantageTests(unittest.TestCase):
    def _scored(self, value):
        return Reward(status=SCORED, value=value)

    def test_advantages_are_standardised_within_the_group(self):
        result = group_advantages([self._scored(0.0), self._scored(1.0)])
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["mean"], 0.5)
        self.assertAlmostEqual(result["std"], 0.5)
        self.assertEqual(result["advantages"], [-1.0, 1.0])
        self.assertFalse(result["zero_variance"])

    def test_identical_rewards_give_no_signal(self):
        result = group_advantages([self._scored(0.7)] * 4)
        self.assertEqual(result["advantages"], [0.0, 0.0, 0.0, 0.0])
        self.assertTrue(result["zero_variance"])
        self.assertEqual(result["frac_zero_std"], 1.0)

    def test_a_tie_is_found_whenever_the_mean_does_not_round_back(self):
        # The tie test must not depend on the arithmetic landing exactly. Groups of three
        # used to divide a residue of ~1e-17 by itself and report +-1 per member.
        for value in (0.1, 0.7, 0.3, 0.05, 1.0, 0.0):
            for count in (2, 3, 4, 5, 7):
                with self.subTest(value=value, count=count):
                    result = group_advantages([self._scored(value)] * count)
                    self.assertEqual(result["advantages"], [0.0] * count)
                    self.assertTrue(result["zero_variance"])

    def test_a_tied_group_is_never_advantaged_uniformly(self):
        for value in (0.1, 0.7, 0.05):
            with self.subTest(value=value):
                advantages = group_advantages([self._scored(value)] * 3)["advantages"]
                self.assertFalse(
                    any(advantage != 0.0 for advantage in advantages),
                    f"tied group produced a uniform signal: {advantages}",
                )

    def test_a_pending_member_blocks_the_group(self):
        result = group_advantages([self._scored(0.0), Reward(status=UNAVAILABLE, reason="x")])
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["advantages"], [])

    def test_a_per_prompt_judge_offset_does_not_change_advantages(self):
        plain = group_advantages([self._scored(0.2), self._scored(0.6)])
        shifted = group_advantages([self._scored(0.4), self._scored(0.8)])
        for left, right in zip(plain["advantages"], shifted["advantages"], strict=True):
            self.assertAlmostEqual(left, right, places=12)

    def test_advantages_never_contain_negative_zero(self):
        result = group_advantages([self._scored(0.5), self._scored(0.5), self._scored(0.5)])
        self.assertEqual([a for a in result["advantages"] if str(a).startswith("-")], [])

    def test_an_empty_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            group_advantages([])


class RewardRecordTests(unittest.TestCase):
    def test_a_scored_reward_requires_a_unit_value(self):
        with self.assertRaises(ValueError):
            Reward(status=SCORED)
        with self.assertRaises(ValueError):
            Reward(status=SCORED, value=1.5)

    def test_an_unavailable_reward_carries_no_value(self):
        with self.assertRaises(ValueError):
            Reward(status=UNAVAILABLE, value=0.0)


if __name__ == "__main__":
    unittest.main()
