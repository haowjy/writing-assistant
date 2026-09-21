"""Check kinds that test a fact was removed rather than negated.

Deleting a fact from a document has a failure mode beyond leaving the fact in: the text
can keep referring to it by negation ("no longer", "rather than", "unlike the earlier
version"). That is a dead giveaway that something was edited, and it is mechanically
detectable, so it should not be left to a judge.
"""

import unittest

from writing_agent.scoring import mechanical_score

KEY = "kb/index.md"


def scenario(checks):
    return {
        "id": "T-1",
        "family": "F4",
        "condition": "kb",
        "provenance": "synthetic",
        "labels": {"checks": checks},
        "visible": {"prose": []},
    }


def result(text):
    return {"status": "completed", "output": "", "turns": [], "after": {KEY: text}, "before": {}}


def forbidden(identity, texts, **extra):
    return {
        "id": identity,
        "metric": "Q13",
        "kind": "excludes_all",
        "method": "deterministic",
        "required": False,
        "path": KEY,
        "texts": texts,
        **extra,
    }


class ExcludesAllTests(unittest.TestCase):
    def test_passes_when_no_forbidden_string_appears(self):
        checks = [forbidden("deleted", ["brass ledger", "no longer", "rather than"])]
        card = mechanical_score(scenario(checks), result("The archive holds tide charts."))
        self.assertEqual(card["scores"]["Q13"]["value"], 1.0)

    def test_fails_when_the_deleted_fact_remains(self):
        checks = [forbidden("deleted", ["brass ledger"])]
        card = mechanical_score(scenario(checks), result("The brass ledger sits upstairs."))
        self.assertEqual(card["scores"]["Q13"]["value"], 0.0)

    def test_fails_on_contrastive_negation_of_the_deleted_fact(self):
        checks = [forbidden("deleted", ["brass ledger", "no longer"])]
        card = mechanical_score(
            scenario(checks), result("The upper room no longer holds the tide records.")
        )
        self.assertEqual(card["scores"]["Q13"]["value"], 0.0)

    def test_matching_is_case_insensitive(self):
        checks = [forbidden("deleted", ["No Longer"])]
        card = mechanical_score(scenario(checks), result("the keeper no longer signs"))
        self.assertEqual(card["scores"]["Q13"]["value"], 0.0)

    def test_an_empty_forbidden_list_is_a_declaration_error(self):
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            mechanical_score(scenario([forbidden("deleted", [])]), result("anything"))

    def test_a_missing_list_is_a_declaration_error(self):
        check = forbidden("deleted", [])
        del check["texts"]
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            mechanical_score(scenario([check]), result("anything"))

    def test_the_single_string_kind_still_works(self):
        check = {
            "id": "one",
            "metric": "Q13",
            "kind": "excludes",
            "method": "deterministic",
            "required": False,
            "path": KEY,
            "text": "brass ledger",
        }
        card = mechanical_score(scenario([check]), result("no ledger here"))
        self.assertEqual(card["scores"]["Q13"]["value"], 1.0)

    def test_a_prohibition_check_cannot_pass_on_an_absent_path(self):
        # Empty text excludes every forbidden string, so a deliverable that was never
        # written used to earn full credit for not containing them.
        absent = {"status": "completed", "output": "", "turns": [], "after": {}, "before": {}}
        card = mechanical_score(scenario([forbidden("deleted", ["no longer"])]), absent)
        entry = card["checks"][0]
        self.assertIs(entry["passed"], False)
        self.assertNotEqual(entry["status"], "ok")
        self.assertNotEqual(card["scores"]["Q13"].get("value"), 1.0)

    def test_a_single_string_prohibition_is_guarded_too(self):
        check = {
            "id": "one",
            "metric": "Q13",
            "kind": "excludes",
            "method": "deterministic",
            "required": False,
            "path": KEY,
            "text": "brass ledger",
        }
        absent = {"status": "completed", "output": "", "turns": [], "after": {}, "before": {}}
        card = mechanical_score(scenario([check]), absent)
        self.assertIs(card["checks"][0]["passed"], False)

    def test_an_absent_required_deliverable_stops_task_completion(self):
        card = mechanical_score(
            scenario([{**forbidden("deleted", ["no longer"]), "required": True}]),
            {"status": "completed", "output": "", "turns": [], "after": {}, "before": {}},
        )
        self.assertEqual(card["scores"]["Q3"]["value"], 0)


if __name__ == "__main__":
    unittest.main()
