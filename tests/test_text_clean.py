import unittest

from writing_agent.text_clean import clean_web_text, opening_prose, strip_gutenberg


class TextCleanTests(unittest.TestCase):
    def test_gutenberg_wrapper_is_removed(self):
        raw = (
            "license header\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
            "The story starts.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
            "license footer\n"
        )
        self.assertEqual(strip_gutenberg(raw), "The story starts.")

    def test_missing_gutenberg_marker_is_an_error(self):
        with self.assertRaises(ValueError):
            strip_gutenberg("no marker here")

    def test_opening_skips_title_page(self):
        text = (
            "Produced by Volunteers\n\n"
            "PRIDE AND PREJUDICE\n\n"
            "Chapter 1\n\n"
            + (
                "It is a truth universally acknowledged, that a single man in possession "
                "of a good fortune, must be in want of a wife. "
            )
            * 20
        )
        opening = opening_prose(text, min_chars=200, max_chars=500)
        self.assertTrue(opening.startswith("It is a truth"))
        self.assertNotIn("Produced by", opening)
        self.assertLessEqual(len(opening), 500)

    def test_web_residue_is_dropped_and_thin_pages_rejected(self):
        page = "\n".join(
            [
                "Home > News | Subscribe",
                "Copyright 2015 Example LLC",
                "A short note.",
            ]
        )
        self.assertIsNone(clean_web_text(page, min_words=40))
        prose = "The river kept the town awake. " * 30
        cleaned = clean_web_text("Share this\n" + prose, min_words=40)
        self.assertNotIn("Share this", cleaned)
        self.assertIn("river", cleaned)
