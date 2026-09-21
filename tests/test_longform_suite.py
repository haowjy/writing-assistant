"""Long-form benchmark compilation: anchoring, holdout proof, and freezing."""

import json
import tempfile
import unittest
from pathlib import Path

from writing_agent.longform_suite import (
    build_release,
    claimed_hashes,
    freeze,
    holdout_audit,
    section_slug,
    split_sections,
    verify_anchors,
)
from writing_agent.suite import compile_scenarios, load_scenarios

WORK = (
    "CHAPTER I. The Start\n\nAlpha travelled to Ingolstadt.\n\n"
    "CHAPTER II. The Middle\n\nBeta stayed in Geneva with Elizabeth.\n\n"
    "CHAPTER III. The End\n\nGamma left.\n"
)
HEADINGS = r"^CHAPTER\s+[IVXLCDM]+\.?[^\n]*$"


def base_case(**changes):
    case = {
        "id": "LF-01",
        "axis": "chaptered_continuation",
        "title": "Fixture",
        "work": "gutenberg-99",
        "supplied": [0, 1],
        "write": ["chapter-iii"],
        "family": "F2",
        "genre": "gothic",
        "specificity": "L3",
        "style": "unspecified",
        "context_target_tokens": 8000,
        "anchors": ["Ingolstadt", "Elizabeth"],
        "brief": "continue",
        "checks": [{"id": "c", "metric": "Q3", "kind": "nonempty", "method": "deterministic"}],
    }
    case.update(changes)
    return case


def fixture(root: Path, **case) -> Path:
    (root / "gutenberg").mkdir(parents=True, exist_ok=True)
    (root / "gutenberg" / "99.txt").write_text(WORK)
    spec = {
        "schema_version": 1,
        "benchmark": "longform-test",
        "role": "final_eval",
        "works": {
            "gutenberg-99": {
                "gutenberg": 99,
                "title": "Fixture",
                "author": "Anon",
                "headings": HEADINGS,
            }
        },
        "cases": [base_case(**case)],
    }
    path = root / "spec.json"
    path.write_text(json.dumps(spec))
    return path


class SectionTests(unittest.TestCase):
    def test_headings_become_stable_file_stems(self):
        self.assertEqual(section_slug("Chapter 1"), "chapter-1")
        self.assertEqual(section_slug("CHAPTER I."), "chapter-i")
        self.assertEqual(section_slug("I. A SCANDAL IN BOHEMIA"), "i-a-scandal-in-bohemia")

    def test_sections_keep_their_headings_with_their_bodies(self):
        sections = split_sections(WORK, HEADINGS)
        self.assertEqual(len(sections), 3)
        self.assertEqual(sections[0][0], "CHAPTER I. The Start")
        self.assertIn("Ingolstadt", sections[0][1])
        self.assertNotIn("Geneva", sections[0][1])

    def test_a_work_with_no_matching_headings_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No sections"):
            split_sections("plain text with no headings", HEADINGS)


class AnchorTests(unittest.TestCase):
    def test_a_grounded_anchor_passes(self):
        verify_anchors(
            base_case(), {"manuscript/a.md": "Alpha reached Ingolstadt with Elizabeth."}
        )

    def test_a_cited_fact_the_text_lacks_fails_the_build(self):
        case = base_case(anchors=["Ingolstadt", "the creature"])
        with self.assertRaisesRegex(ValueError, "the creature"):
            verify_anchors(case, {"manuscript/a.md": "Alpha reached Ingolstadt."})

    def test_anchors_are_matched_case_insensitively(self):
        verify_anchors(base_case(anchors=["ingolstadt"]), {"manuscript/a.md": "Ingolstadt."})


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_work_becomes_a_final_eval_source(self):
        catalog, _ = build_release(fixture(self.root), self.root)
        self.assertEqual(len(catalog), 1)
        source = catalog[0]
        self.assertEqual(source["role"], "final_eval")
        self.assertEqual(source["work_id"], "gutenberg-99")
        self.assertEqual(source["terms"]["training"], "excluded: final evaluation benchmark")
        self.assertIn("Alpha travelled", source["text"])

    def test_supplied_sections_become_manuscript_files(self):
        _, scenarios = build_release(fixture(self.root), self.root)
        files = scenarios[0]["visible"]["initial_files"]
        self.assertEqual(
            sorted(files),
            ["manuscript/chapter-i-the-start.md", "manuscript/chapter-ii-the-middle.md"],
        )
        self.assertTrue(
            files["manuscript/chapter-i-the-start.md"].startswith("CHAPTER I. The Start")
        )

    def test_a_named_section_overrides_its_stem(self):
        _, scenarios = build_release(fixture(self.root, names={"0": "opening"}), self.root)
        self.assertIn("manuscript/opening.md", scenarios[0]["visible"]["initial_files"])

    def test_deliverables_become_file_selectors(self):
        _, scenarios = build_release(fixture(self.root), self.root)
        self.assertEqual(
            scenarios[0]["visible"]["prose"],
            [
                {
                    "id": "chapter-iii",
                    "kind": "file",
                    "path": "manuscript/chapter-iii.md",
                    "selection": "whole",
                }
            ],
        )

    def test_a_reply_case_gets_no_tools_and_a_reply_selector(self):
        _, scenarios = build_release(fixture(self.root, replies=True, write=[]), self.root)
        visible = scenarios[0]["visible"]
        self.assertEqual(visible["tools"], [])
        self.assertEqual(visible["prose"], [{"id": "reply", "kind": "reply", "selection": "whole"}])

    def test_a_kb_case_also_scores_the_new_case_file(self):
        _, scenarios = build_release(
            fixture(self.root, axis="kb_bootstrapped_novel", write=[]), self.root
        )
        self.assertIn("new-case", [p["id"] for p in scenarios[0]["visible"]["prose"]])

    def test_budgets_scale_with_the_supplied_manuscript(self):
        _, scenarios = build_release(fixture(self.root), self.root)
        budgets = scenarios[0]["visible"]["budgets"]
        supplied = sum(len(t.encode()) for t in scenarios[0]["visible"]["initial_files"].values())
        self.assertGreaterEqual(budgets["max_total_bytes"], supplied * 6)
        self.assertGreater(budgets["max_steps"], 0)

    def test_extra_files_reach_the_workspace_but_do_not_ground_anchors(self):
        case = base_case(
            extra_files={"notes/outline.md": "Sam is new."},
            anchors=["Ingolstadt"],
        )
        _, scenarios = build_release(fixture(self.root, **case), self.root)
        self.assertIn("notes/outline.md", scenarios[0]["visible"]["initial_files"])
        with self.assertRaises(ValueError):
            build_release(fixture(self.root, anchors=["Sam"]), self.root)

    def test_the_release_compiles_and_reloads_through_the_shared_contract(self):
        catalog, scenarios = build_release(fixture(self.root), self.root)
        manifest = compile_scenarios(scenarios, catalog, self.root / "out")
        self.assertEqual(len(manifest["scenarios"]), 1)
        loaded = load_scenarios(self.root / "out")
        self.assertEqual(loaded[0]["role"], "final_eval")
        self.assertEqual(loaded[0]["longform_axis"], "chaptered_continuation")


class HoldoutTests(unittest.TestCase):
    def test_a_disjoint_benchmark_is_held_out(self):
        catalog = [{"id": "a", "sha256": "aaa"}]
        self.assertEqual(
            holdout_audit(catalog, claimed={"training": {"bbb"}})["status"], "held_out"
        )

    def test_sharing_a_claimed_source_fails_the_build(self):
        catalog = [{"id": "a", "sha256": "aaa"}]
        with self.assertRaisesRegex(ValueError, "not held out"):
            holdout_audit(catalog, claimed={"development_used": {"aaa"}})

    def test_claimed_hashes_can_be_taken_from_a_subset(self):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "catalog.json"
        path.write_text(json.dumps([{"id": "a", "sha256": "aaa"}, {"id": "b", "sha256": "bbb"}]))
        self.assertEqual(claimed_hashes(path), {"aaa", "bbb"})
        self.assertEqual(claimed_hashes(path, ids={"b", "missing"}), {"bbb"})
        tmp.cleanup()

    def test_a_missing_catalog_claims_nothing(self):
        self.assertEqual(claimed_hashes(Path("/nonexistent/catalog.json")), set())


class FreezeTests(unittest.TestCase):
    def test_freezing_is_deterministic_and_named_per_case(self):
        manifest = {
            "benchmark": "longform-test",
            "catalog_hash": "abc",
            "scenarios": [{"id": "LF-01", "visible_hash": "v", "labels_hash": "l"}],
        }
        first = freeze(manifest, Path("/tmp"))
        self.assertEqual(first, freeze(manifest, Path("/tmp")))
        self.assertEqual(first["cases"], 1)
        self.assertEqual(first["case_hashes"]["LF-01"], {"visible": "v", "labels": "l"})
        self.assertTrue(first["freeze_hash"])

    def test_changing_a_case_changes_the_freeze_hash(self):
        manifest = {
            "benchmark": "longform-test",
            "catalog_hash": "abc",
            "scenarios": [{"id": "LF-01", "visible_hash": "v", "labels_hash": "l"}],
        }
        changed = json.loads(json.dumps(manifest))
        changed["scenarios"][0]["labels_hash"] = "different"
        self.assertNotEqual(
            freeze(manifest, Path("/tmp"))["freeze_hash"],
            freeze(changed, Path("/tmp"))["freeze_hash"],
        )


if __name__ == "__main__":
    unittest.main()
