"""The pinned external anchor: protocol rendering, upstream scoring, adaptation modes."""

import json
import tempfile
import unittest
from pathlib import Path

from writing_agent.longform import (
    CRITERIA_WEIGHTS,
    DEFAULT_CHAPTERS,
    EQ_REVISION,
    PLANNING_STEPS,
    aggregate,
    benchmark_release,
    chapter_turns,
    degradation,
    judge_score,
    load_release,
    missing_criteria,
    parse_scores,
    render_steps,
    staccato_factor,
    task_score,
)

NAMES = (
    "prompt1.txt",
    "prompt2.txt",
    "prompt3.txt",
    "prompt4.txt",
    "prompt5.txt",
    "prompt_chapter_first.txt",
    "prompt_chapter_intermediate.txt",
    "prompt_chapter_last.txt",
)
PROMPTS = {
    "1": {"category": "Mythology", "title": "A", "writing_prompt": "Gods wore sneakers."},
    "2": {"category": "Sci-Fi", "title": "B", "writing_prompt": "They spoke."},
}


def fixture(root: Path) -> Path:
    """A minimal upstream-shaped release so the loader is tested without downloads."""
    data = root / "eqbench-longform" / "data"
    data.mkdir(parents=True)
    templates = {
        "prompt1.txt": "{writing_prompt}\n\nplan for {n_chapters} chapters",
        "prompt2.txt": "intention and chapter plan for {n_chapters}",
        "prompt3.txt": "critique the plan",
        "prompt4.txt": "final plan for {n_chapters}",
        "prompt5.txt": "character profiles",
        "prompt_chapter_first.txt": "write chapter 1",
        "prompt_chapter_intermediate.txt": "now chapter {chapter_number}",
        "prompt_chapter_last.txt": "finish with chapter {chapter_number}",
    }
    for name, text in templates.items():
        (data / name).write_text(text)
    (data / "longform_creative_writing_prompts_minimalist.json").write_text(json.dumps(PROMPTS))
    (data / "criteria_weights.json").write_text(json.dumps({"forced poetry or metaphor": 5}))
    (data / "longform_creative_writing_criteria_chapter.txt").write_text("Coherent\nPurple Prose\n")
    (data / "longform_creative_writing_criteria_final.txt").write_text("Coherent\nPurple Prose\n")
    (data / "longform_negative_criteria_chapter.txt").write_text("Purple Prose\n")
    (data / "longform_negative_criteria_final.txt").write_text("Purple Prose\n")
    (data / "longform_creative_writing_judging_prompt_chapter.txt").write_text(
        "judge {chapter_number}\n{writing_prompt}\n{final_plan}\n{character_profiles}\n"
        "{lower_is_better_criteria}\n{creative_writing_criteria}\n{chapter_text}"
    )
    (data / "longform_creative_writing_judging_prompt_final.txt").write_text(
        "judge final\n{writing_prompt}\n{full_story}\n{lower_is_better_criteria}\n"
        "{creative_writing_criteria}"
    )
    return root / "eqbench-longform"


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = fixture(Path(self.tmp.name))
        self.release = load_release(self.raw)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loading_reports_a_missing_fixture_rather_than_scoring_nothing(self):
        (self.raw / "data" / "prompt3.txt").unlink()
        with self.assertRaisesRegex(ValueError, "prompt3.txt"):
            load_release(self.raw)

    def test_prompts_keep_upstream_order(self):
        self.assertEqual([prompt["title"] for prompt in self.release.prompts], ["A", "B"])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.release = load_release(fixture(Path(self.tmp.name)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_protocol_is_planning_then_one_step_per_chapter(self):
        steps = render_steps(self.release, "Gods wore sneakers.", chapters=8)
        self.assertEqual(len(steps), PLANNING_STEPS + 8)
        self.assertIn("plan for 8 chapters", steps[0])
        self.assertIn("Gods wore sneakers.", steps[0])

    def test_only_the_first_and_last_chapter_get_their_own_wording(self):
        steps = render_steps(self.release, "P", chapters=8)
        self.assertEqual(steps[5], "write chapter 1")
        self.assertEqual(steps[6], "now chapter 2")
        self.assertEqual(steps[11], "now chapter 7")
        self.assertEqual(steps[12], "finish with chapter 8")

    def test_chapter_turns_follow_the_planning_steps(self):
        self.assertEqual(chapter_turns(8), (5, 6, 7, 8, 9, 10, 11, 12))
        self.assertEqual(len(chapter_turns(DEFAULT_CHAPTERS)), DEFAULT_CHAPTERS)

    def test_a_single_chapter_run_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "two chapters"):
            render_steps(self.release, "P", chapters=1)


class ScoreParsingTests(unittest.TestCase):
    CRITERIA = ("Coherent", "Purple Prose")

    def test_scores_are_read_from_metric_lines(self):
        self.assertEqual(
            parse_scores("Coherent: 15\nPurple Prose: [7]"), {"Coherent": 15.0, "Purple Prose": 7.0}
        )

    def test_an_explicit_scores_section_is_preferred(self):
        text = "Coherent: 3\n[Scores]\nCoherent: 18\nPurple Prose: 2\n"
        self.assertEqual(parse_scores(text), {"Coherent": 18.0, "Purple Prose": 2.0})

    def test_commentary_and_known_non_metrics_are_dropped(self):
        text = "Here is my analysis.\nReasoning: 5\nCoherent: 12\n"
        self.assertEqual(parse_scores(text), {"Coherent": 12.0})

    def test_out_of_range_scores_are_clamped(self):
        self.assertEqual(
            parse_scores("Coherent: 25\nPurple Prose: -4"), {"Coherent": 20.0, "Purple Prose": 0.0}
        )

    def test_declared_criteria_filter_invented_metrics(self):
        text = "Coherent: 12\nVibes: 20\n"
        self.assertEqual(parse_scores(text, criteria=self.CRITERIA), {"Coherent": 12.0})

    def test_a_judge_that_omits_a_criterion_is_reported(self):
        self.assertEqual(missing_criteria({"Coherent": 1.0}, self.CRITERIA), ["Purple Prose"])


class ScoringTests(unittest.TestCase):
    def test_negative_criteria_are_inverted_onto_the_shared_axis(self):
        self.assertEqual(judge_score({"Purple Prose": 0}, negative=["Purple Prose"]), 20.0)
        self.assertEqual(judge_score({"Purple Prose": 20}, negative=["Purple Prose"]), 0.0)

    def test_the_weighted_mean_is_normalized_by_total_weight(self):
        self.assertEqual(judge_score({"Coherent": 10}, negative=[]), 10.0)

    def test_forced_metaphor_carries_extra_weight(self):
        self.assertEqual(CRITERIA_WEIGHTS["forced poetry or metaphor"], 5.0)
        score = judge_score(
            {"Coherent": 10, "Forced Poetry or Metaphor": 10},
            negative=["Forced Poetry or Metaphor"],
        )
        # inverted to 10, convex-transformed to (10/20)**1.7*20, then weighted 5 against 1
        transformed = (10 / 20) ** 1.7 * 20
        self.assertAlmostEqual(score, (10 + transformed * 5) / 6)

    def test_forced_metaphor_is_convex_rather_than_linear(self):
        # Partial metaphor loses more than proportionality would suggest.
        penalized = judge_score(
            {"Forced Poetry or Metaphor": 5}, negative=["Forced Poetry or Metaphor"]
        )
        self.assertLess(penalized, 15.0)

    def test_an_absent_forced_metaphor_is_not_penalized(self):
        # Raw 0 means no forced metaphor at all, which inverts to a perfect score.
        self.assertEqual(
            judge_score({"Forced Poetry or Metaphor": 0}, negative=["Forced Poetry or Metaphor"]),
            20.0,
        )

    def test_heavy_forced_metaphor_is_penalized_hard(self):
        self.assertEqual(
            judge_score({"Forced Poetry or Metaphor": 20}, negative=["Forced Poetry or Metaphor"]),
            0.0,
        )

    def test_staccato_is_free_below_the_threshold(self):
        self.assertEqual(staccato_factor(0), 1.0)
        self.assertEqual(staccato_factor(4.9), 1.0)
        self.assertEqual(staccato_factor(5.0), 1.0)

    def test_staccato_reaches_its_floor_continuously(self):
        self.assertAlmostEqual(staccato_factor(18.0), 0.6)
        self.assertAlmostEqual(staccato_factor(18.0), staccato_factor(18.0001), places=6)
        self.assertEqual(staccato_factor(40.0), 0.6)

    def test_the_legacy_curve_reproduces_the_upstream_discontinuity(self):
        self.assertAlmostEqual(staccato_factor(18.0, legacy_curve=True), 0.4)
        self.assertAlmostEqual(staccato_factor(18.0001, legacy_curve=True), 0.6)

    def test_task_score_weights_chapters_against_the_whole_piece(self):
        self.assertAlmostEqual(task_score([10.0, 20.0], [20.0]), 50.0 / 3)

    def test_task_score_applies_the_staccato_factor(self):
        self.assertAlmostEqual(task_score([20.0], [], staccato=(40.0,)), 12.0)

    def test_an_unscored_task_has_no_score(self):
        self.assertIsNone(task_score([], []))

    def test_aggregate_scales_to_the_hundred_point_figure_and_keeps_gaps_visible(self):
        result = aggregate([10.0, 20.0, None])
        self.assertAlmostEqual(result["score_0_20"], 15.0)
        self.assertAlmostEqual(result["score_0_100"], 75.0)
        self.assertEqual(result["tasks_scored"], 2)
        self.assertEqual(result["tasks_total"], 3)

    def test_aggregate_with_nothing_scored_is_explicit(self):
        result = aggregate([None, None])
        self.assertIsNone(result["score_0_20"])
        self.assertEqual(result["tasks_scored"], 0)
        self.assertIn("No task", result["reason"])

    def test_degradation_is_the_fall_from_first_chapter_to_last(self):
        self.assertAlmostEqual(degradation([18.0, 15.0, 12.0]), 6.0)
        self.assertAlmostEqual(degradation([12.0, 18.0]), -6.0)
        self.assertIsNone(degradation([18.0]))


class BenchmarkReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = fixture(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_upstream_prompt_becomes_one_final_eval_case(self):
        catalog, scenarios = benchmark_release(self.raw)
        self.assertEqual(len(catalog), len(PROMPTS))
        self.assertEqual(len(scenarios), len(PROMPTS))
        for source in catalog:
            self.assertEqual(source["role"], "final_eval")
            self.assertEqual(source["terms"]["training"], "excluded: external benchmark")
        for scenario in scenarios:
            self.assertEqual((scenario["role"], scenario["family"]), ("final_eval", "F1"))

    def test_faithful_mode_stays_reply_only_like_upstream(self):
        _, scenarios = benchmark_release(self.raw, mode="faithful")
        visible = scenarios[0]["visible"]
        self.assertEqual(visible["tools"], [])
        self.assertEqual(visible["budgets"]["max_tool_calls"], 0)
        self.assertEqual(visible["prose"][0]["kind"], "reply")
        self.assertEqual(visible["prose"][0]["turn"], PLANNING_STEPS)

    def test_workspace_mode_saves_chapters_and_says_so(self):
        _, scenarios = benchmark_release(self.raw, mode="workspace")
        visible = scenarios[0]["visible"]
        self.assertIn("write_file", visible["tools"])
        # followups[0] is the second planning step; chapter 1 is followups[PLANNING_STEPS - 1].
        chapter_one = visible["followups"][PLANNING_STEPS - 1]
        self.assertIn("manuscript/chapter-1.md", chapter_one)
        self.assertEqual(visible["prose"][0]["kind"], "file")
        self.assertEqual(visible["prose"][0]["path"], "manuscript/chapter-1.md")

    def test_a_case_has_one_prose_selector_per_chapter(self):
        _, scenarios = benchmark_release(self.raw, chapters=3)
        self.assertEqual(
            [s["id"] for s in scenarios[0]["visible"]["prose"]],
            ["chapter-1", "chapter-2", "chapter-3"],
        )
        self.assertEqual(scenarios[0]["chapters"], 3)

    def test_the_pinned_revision_and_upstream_criteria_travel_with_the_case(self):
        _, scenarios = benchmark_release(self.raw)
        labels = scenarios[0]["labels"]
        self.assertEqual(scenarios[0]["upstream_revision"], EQ_REVISION)
        self.assertEqual(labels["upstream_revision"], EQ_REVISION)
        self.assertEqual(labels["criteria_chapter"], ["Coherent", "Purple Prose"])
        self.assertEqual(labels["negative_chapter"], ["Purple Prose"])
        self.assertEqual(labels["judge_model"], "claude-sonnet-4-6")
        self.assertEqual(labels["score_scale"], [0, 20])

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown mode"):
            benchmark_release(self.raw, mode="hybrid")


if __name__ == "__main__":
    unittest.main()
