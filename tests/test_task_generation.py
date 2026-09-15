"""Training source integrity at the task-preparation boundary."""

import copy
import unittest

from writing_agent.catalog import fingerprint
from writing_agent.task_generation import prepare_task_requests

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
            count=5,
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
                count=5,
            ),
        )
        catalog[0]["text"] = "mutated after preparation"
        self.assertEqual(batch["requests"][0]["packet"]["source"]["text"], "book")

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
