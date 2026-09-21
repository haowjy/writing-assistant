"""Specificity schema, ladder and declaration checks, without network or models."""

import unittest

from writing_agent.specificity import (
    ALWAYS_STATED,
    ASK_REQUIRED,
    BEHAVIORS,
    DECISION_POINTS,
    DEFAULT_SAFE,
    FAMILY_POINTS,
    LEVELS,
    check_declared,
    check_spec,
    decision_points,
    split_spec,
    withheld_behaviors,
)

PROSE_FAMILIES = ("F1", "F2", "F5")


class SchemaTests(unittest.TestCase):
    def test_every_family_point_is_declared(self):
        for family, points in FAMILY_POINTS.items():
            for point in points:
                self.assertIn(point, DECISION_POINTS, (family, point))

    def test_point_behaviors_and_levels_are_known(self):
        for name, point in DECISION_POINTS.items():
            self.assertIn(point.behavior, BEHAVIORS, name)
            self.assertIn(point.level, LEVELS, name)

    def test_every_family_reaches_the_explicit_level(self):
        for family in FAMILY_POINTS:
            self.assertEqual(
                set(split_spec(family, LEVELS[-1])["stated"]), set(decision_points(family))
            )


class SpecificityLadderTests(unittest.TestCase):
    def test_stated_points_grow_with_the_level(self):
        for family in FAMILY_POINTS:
            previous: set[str] = set()
            for level in LEVELS:
                current = set(split_spec(family, level)["stated"])
                self.assertLessEqual(previous, current, (family, level))
                previous = current

    def test_stated_and_withheld_partition_the_points(self):
        for family in FAMILY_POINTS:
            points = set(decision_points(family))
            for level in LEVELS:
                split = split_spec(family, level)
                stated, withheld = set(split["stated"]), set(split["withheld"])
                self.assertEqual(stated & withheld, set(), (family, level))
                self.assertEqual(stated | withheld, points, (family, level))

    def test_always_stated_points_are_stated_at_every_level(self):
        for family in FAMILY_POINTS:
            for level in LEVELS:
                stated = set(split_spec(family, level)["stated"])
                for point in decision_points(family):
                    if DECISION_POINTS[point].behavior == ALWAYS_STATED:
                        self.assertIn(point, stated, (family, level, point))

    def test_lowest_level_withholds_the_ask_required_point(self):
        for family in PROSE_FAMILIES:
            self.assertIn("branch_choice", split_spec(family, "L0")["withheld"])

    def test_every_family_withholds_something_below_explicit(self):
        for family in FAMILY_POINTS:
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

    def test_withheld_behaviors_never_drop_a_point(self):
        for family in FAMILY_POINTS:
            for level in LEVELS:
                split = split_spec(family, level)
                behaviors = withheld_behaviors(family, level)
                self.assertEqual(
                    set(split["withheld"]),
                    set(behaviors["ask_required"]) | set(behaviors["default_safe"]),
                )

    def test_ask_required_and_default_safe_are_disjoint(self):
        self.assertIn(ASK_REQUIRED, BEHAVIORS)
        self.assertIn(DEFAULT_SAFE, BEHAVIORS)
        for family in FAMILY_POINTS:
            for level in LEVELS:
                behaviors = withheld_behaviors(family, level)
                self.assertEqual(
                    set(behaviors["ask_required"]) & set(behaviors["default_safe"]), set()
                )


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
