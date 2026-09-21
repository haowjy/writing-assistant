"""Training source integrity at the task-preparation boundary."""

import copy
import unittest

from writing_agent.catalog import fingerprint
from writing_agent.task_generation import build_request, iter_requests, prepare_task_requests

VARIATIONS = {
    key: ["option one", "option two"]
    for key in ("genres", "styles", "tropes", "situations", "continuity_challenges")
}


def source(key, **changes):
    return {
        "id": key,
        "work_id": key,
        "role": "train",
        "provenance": "human",
        "text": key,
        "sha256": fingerprint(key),
        "parents": [],
        "terms": {"license": "test fixture"},
        **changes,
    }


class TaskGenerationTests(unittest.TestCase):
    def test_excluded_parent_and_shared_author_block_selected_child(self):
        parent = source("parent", author_id="author")
        child = source("child", parents=["parent"])
        sibling = source("sibling", author_id="author")
        for excluded in ({"parent"}, {"sibling"}, {"author"}):
            with self.subTest(excluded=excluded), self.assertRaisesRegex(ValueError, "Held-out"):
                prepare_task_requests(
                    [parent, child, sibling],
                    ["child"],
                    excluded_source_groups=excluded,
                    variation_catalog=VARIATIONS,
                )

    def test_rejects_held_out_changed_and_unaccepted_synthetic_sources(self):
        for changes in (
            {"role": "development"},
            {"upstream_split": "test"},
            {"text": "changed"},
            {"provenance": "synthetic"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                prepare_task_requests(
                    [source("book", **changes)],
                    ["book"],
                    excluded_source_groups=set(),
                    variation_catalog=VARIATIONS,
                )

    def test_preparation_is_reproducible_and_does_not_certify_or_mutate_sources(self):
        catalog = [source("book")]
        before = copy.deepcopy(catalog)
        batch = prepare_task_requests(
            catalog,
            ["book"],
            excluded_source_groups=set(),
            variation_catalog=VARIATIONS,
            count=8,
        )
        self.assertEqual(catalog, before)
        self.assertEqual(batch["generated_tasks"], 0)
        self.assertEqual(batch["accepted_tasks"], 0)
        self.assertEqual(
            batch,
            prepare_task_requests(
                catalog,
                ["book"],
                excluded_source_groups=set(),
                variation_catalog=VARIATIONS,
                count=8,
            ),
        )
        catalog[0]["text"] = "mutated after preparation"
        packet = batch["requests"][0]["packet"]
        self.assertEqual(packet["source_id"], "book")
        self.assertEqual(packet["source_sha256"], before[0]["sha256"])
        self.assertNotIn("source", packet)

    def test_genre_blends_respect_continuations_and_preserve_catalog(self):
        original = copy.deepcopy(VARIATIONS)
        batch = prepare_task_requests(
            [source("book")],
            ["book"],
            excluded_source_groups=set(),
            variation_catalog=VARIATIONS,
            count=20,
        )
        self.assertEqual(VARIATIONS, original)
        self.assertTrue(any(len(r["assignment"]["genre_blend"]) == 2 for r in batch["requests"]))
        for request in batch["requests"]:
            assignment = request["assignment"]
            blend = assignment["genre_blend"]
            self.assertEqual(len(blend), len(set(blend)))
            if assignment["transformation"] == "close_continuation":
                self.assertEqual(blend, [])
                self.assertEqual(assignment["style"], "preserve source style")


class SpecificityLadderRequestTests(unittest.TestCase):
    def _batch(self, count, levels=None):
        kwargs = {} if levels is None else {"levels": levels}
        return prepare_task_requests(
            [source("book")],
            ["book"],
            excluded_source_groups=set(),
            variation_catalog=VARIATIONS,
            count=count,
            **kwargs,
        )

    def test_count_must_divide_by_level_count(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            self._batch(3, ("L0", "L3"))

    def test_unknown_repeated_or_empty_level_raises(self):
        for levels in (("L0", "L9"), ("L0", "L0"), ()):
            with self.subTest(levels=levels), self.assertRaisesRegex(ValueError, "level"):
                self._batch(4, levels)

    def test_default_levels_state_every_point(self):
        batch = self._batch(4)
        for request in batch["requests"]:
            specificity = request["assignment"]["instruction_specificity"]
            self.assertEqual(specificity["level"], "L3")
            self.assertEqual(specificity["withheld"], [])

    def test_requests_form_matched_ladders(self):
        levels = ("L0", "L1", "L2", "L3")
        batch = self._batch(8, levels)
        requests = batch["requests"]
        self.assertEqual(len(requests), 8)
        self.assertEqual(batch["levels"], list(levels))
        for base in range(2):
            ladder = requests[base * 4 : base * 4 + 4]
            self.assertEqual(
                [r["assignment"]["instruction_specificity"]["level"] for r in ladder],
                list(levels),
            )
            shared = {
                key: value
                for key, value in ladder[0]["assignment"].items()
                if key != "instruction_specificity"
            }
            for request in ladder[1:]:
                other = {
                    key: value
                    for key, value in request["assignment"].items()
                    if key != "instruction_specificity"
                }
                self.assertEqual(shared, other)

    def test_lower_levels_withhold_points_but_not_deliverable(self):
        specificity = self._batch(4, ("L0",))["requests"][0]["assignment"][
            "instruction_specificity"
        ]
        self.assertIn("deliverable", specificity["stated"])
        self.assertTrue(specificity["withheld"])
        self.assertNotIn("deliverable", specificity["withheld"])

    def test_close_continuation_prunes_branch_choice(self):
        batch = self._batch(40, ("L0", "L3"))
        seen_branch = False
        for request in batch["requests"]:
            assignment = request["assignment"]
            specificity = assignment["instruction_specificity"]
            points = set(specificity["stated"]) | set(specificity["withheld"])
            if assignment["transformation"] == "close_continuation":
                self.assertNotIn("branch_choice", points)
            elif "branch_choice" in points:
                seen_branch = True
        self.assertTrue(seen_branch)

    def test_coverage_counts_levels_and_withheld_points(self):
        coverage = self._batch(8, ("L0", "L3"))["coverage"]
        self.assertEqual(coverage["instruction_specificity_level"], {"L0": 4, "L3": 4})
        self.assertGreater(coverage["withheld_points"].get("canon_authorization", 0), 0)
        self.assertNotIn("instruction_specificity", coverage)


class StreamingSamplerTests(unittest.TestCase):
    def _kwargs(self, **changes):
        base = {
            "excluded_source_groups": set(),
            "variation_catalog": VARIATIONS,
            "count": 8,
            "levels": ("L0", "L1", "L2", "L3"),
        }
        base.update(changes)
        return base

    def test_iterator_matches_the_materialized_batch(self):
        catalog = [source("book")]
        batch = prepare_task_requests(catalog, ["book"], **self._kwargs())
        self.assertEqual(
            list(iter_requests(catalog, ["book"], **self._kwargs())), batch["requests"]
        )

    def test_requests_are_addressable_without_materializing_the_batch(self):
        catalog = [source("book")]
        streamed = list(iter_requests(catalog, ["book"], **self._kwargs()))
        for index in (0, 3, 7):
            self.assertEqual(
                build_request(index, catalog, ["book"], **self._kwargs()), streamed[index]
            )

    def test_resuming_from_an_index_matches_the_full_stream(self):
        catalog = [source("book")]
        full = list(iter_requests(catalog, ["book"], **self._kwargs()))
        resumed = list(iter_requests(catalog, ["book"], start=5, **self._kwargs()))
        self.assertEqual(resumed, full[5:])

    def test_out_of_range_start_and_index_raise(self):
        catalog = [source("book")]
        with self.assertRaisesRegex(ValueError, "Start index"):
            list(iter_requests(catalog, ["book"], start=9, **self._kwargs()))
        with self.assertRaisesRegex(ValueError, "out of range"):
            build_request(99, catalog, ["book"], **self._kwargs())

    def test_packets_reference_sources_instead_of_embedding_them(self):
        packet = next(iter(iter_requests([source("book")], ["book"], **self._kwargs())))["packet"]
        self.assertEqual(packet["source_id"], "book")
        self.assertEqual(packet["source_work"], "book")
        self.assertNotIn("source", packet)
        self.assertNotIn("text", packet)
