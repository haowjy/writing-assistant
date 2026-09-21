"""Specificity ladder and declaration checks, without network or models."""

import unittest

from writing_agent.specificity import (
    ASK_REQUIRED,
    DECISION_POINTS,
    FAMILY_LADDER,
    LEVELS,
    check_declared,
    check_spec,
    decision_points,
    split_spec,
    withheld_behaviors,
)

PROSE_FAMILIES = ("F1", "F2", "F5")


class SpecificityLadderTests(unittest.TestCase):
    def test_every_ladder_point_is_known(self):
        for family, ladder in FAMILY_LADDER.items():
            for level in LEVELS:
                for name in ladder[level]:
                    self.assertIn(name, DECISION_POINTS, (family, level, name))

    def test_stated_points_grow_with_the_level(self):
        for family in FAMILY_LADDER:
            previous: set[str] = set()
            for level in LEVELS:
                current = set(split_spec(family, level)["stated"])
                self.assertLessEqual(previous, current, (family, level))
                previous = current

    def test_explicit_level_states_every_point(self):
        for family in FAMILY_LADDER:
            split = split_spec(family, "L3")
            self.assertEqual(split["withheld"], [])
            self.assertEqual(set(split["stated"]), set(decision_points(family)))

    def test_stated_and_withheld_partition_the_points(self):
        for family in FAMILY_LADDER:
            points = set(decision_points(family))
            for level in LEVELS:
                split = split_spec(family, level)
                stated, withheld = set(split["stated"]), set(split["withheld"])
                self.assertEqual(stated & withheld, set(), (family, level))
                self.assertEqual(stated | withheld, points, (family, level))

    def test_lowest_level_withholds_the_ask_required_point(self):
        for family in PROSE_FAMILIES:
            self.assertIn("branch_choice", split_spec(family, "L0")["withheld"])

    def test_every_family_withholds_something_below_explicit(self):
        for family in FAMILY_LADDER:
            self.assertTrue(split_spec(family, "L0")["withheld"], family)

    def test_applicable_prunes_inapplicable_points(self):
        split = split_spec("F1", "L0", applicable=("deliverable", "goal", "style"))
        self.assertEqual(set(split["stated"]), {"deliverable", "goal"})
        self.assertEqual(set(split["withheld"]), {"style"})

    def test_unknown_family_level_and_point_raise(self):
        with self.assertRaises(ValueError):
            split_spec("F9", "L0")
        with self.assertRaises(ValueError):
            split_spec("F1", "L9")
        with self.assertRaises(ValueError):
            split_spec("F1", "L0", applicable=("nonsense",))

    def test_withheld_behaviors_classify_by_required_behavior(self):
        behaviors = withheld_behaviors("F1", "L0")
        self.assertEqual(behaviors["ask_required"], ["branch_choice"])
        self.assertIn("style", behaviors["default_safe"])
        self.assertNotIn("deliverable", behaviors["ask_required"] + behaviors["default_safe"])

    def test_ask_required_points_are_never_default_safe(self):
        for family in FAMILY_LADDER:
            for level in LEVELS:
                behaviors = withheld_behaviors(family, level)
                self.assertEqual(
                    set(behaviors["ask_required"]) & set(behaviors["default_safe"]), set()
                )

    def test_ask_required_class_exists(self):
        self.assertTrue(any(kind == ASK_REQUIRED for kind in DECISION_POINTS.values()))


class SpecCheckTests(unittest.TestCase):
    def test_complete_spec_has_no_problems(self):
        points = decision_points("F3")
        spec = {name: "value" for name in points}
        self.assertEqual(check_spec(spec, "F3"), [])

    def test_missing_extra_and_empty_points_are_reported(self):
        points = decision_points("F3")
        spec = {name: "value" for name in points}
        self.assertTrue(any("omits" in p for p in check_spec({}, "F3")))
        incomplete = dict(spec, goal="")
        self.assertTrue(any("empty" in p and "goal" in p for p in check_spec(incomplete, "F3")))
        extra = dict(spec, bogus="value")
        self.assertTrue(any("bogus" in p for p in check_spec(extra, "F3")))

    def test_applicable_narrows_the_expected_spec(self):
        spec = {"deliverable": "reply", "goal": "plan"}
        self.assertEqual(check_spec(spec, "F1", applicable=("deliverable", "goal")), [])


class DeclaredSpecificityTests(unittest.TestCase):
    def test_matching_declaration_passes(self):
        points = ["deliverable", "goal", "setting"]
        self.assertEqual(check_declared(points, points), [])

    def test_extra_and_missing_declarations_are_reported(self):
        self.assertTrue(check_declared(["deliverable"], ["deliverable", "style"]))
        self.assertTrue(check_declared(["deliverable", "style"], ["deliverable"]))
        self.assertTrue(check_declared(["deliverable"], ["deliverable", "length", "style"]))

    def test_order_does_not_matter(self):
        self.assertEqual(check_declared(["goal", "deliverable"], ["deliverable", "goal"]), [])


if __name__ == "__main__":
    unittest.main()
