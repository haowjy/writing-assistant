import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from writing_agent.agent import run_agent
from writing_agent.artifacts import extract_prose, markdown_graph
from writing_agent.backends import ScriptedBackend
from writing_agent.catalog import fingerprint, import_story, overlap_audit, validate_catalog
from writing_agent.development import author_development
from writing_agent.grading import apply_judgment, grading_packet, validate_judgment, write_review
from writing_agent.prose import (
    FeatureConfig,
    ProseFeatures,
    compare_groups,
    dispersion,
    distribution_l2,
    lexical_features,
    mmd_squared,
    prose_profile,
    self_bleu,
)
from writing_agent.scoring import build_report, mechanical_score
from writing_agent.suite import (
    compile_scenarios,
    kb_use_scenario,
    load_scenarios,
    render_kb,
    run_navigation,
    run_selected,
    saved_results,
)
from writing_agent.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
MODEL = {"id": "scripted", "revision": "fixture-v1", "protocol": "fixture", "kind": "scripted"}


def tool(name, **arguments):
    return {
        "role": "assistant",
        "tool_calls": [{"id": "call", "function": {"name": name, "arguments": arguments}}],
    }


def answer(text):
    return {"role": "assistant", "content": text}


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog, self.scenarios = author_development(ROOT / "data/scenarios/worlds.json")

    def compile(self):
        compile_scenarios(self.scenarios, self.catalog, self.root / "release")
        return load_scenarios(self.root / "release")

    def test_genres_are_balanced_and_grounded_in_visible_material(self):
        from collections import Counter

        scenarios = self.compile()
        self.assertEqual(set(Counter(s["genre"] for s in scenarios).values()), {5})
        catalog = {r["id"]: r for r in self.catalog}
        genres = json.loads((ROOT / "data/scenarios/genres.json").read_text())
        labels = {g["id"]: g["label"] for g in genres}
        for family in range(1, 6):
            self.assertEqual(
                len({s["genre"] for s in scenarios if s["family"] == f"F{family}"}), 10
            )
        for genre in labels:
            self.assertEqual(
                {s["instruction_specificity"] for s in scenarios if s["genre"] == genre},
                {"explicit", "loose"},
            )
        for scenario in scenarios:
            source = catalog[scenario["source_ids"][0]]
            self.assertTrue(source["parents"])
            context = source["text"].split("\n\n", 1)[0]
            visible = scenario["visible"]
            self.assertIn(labels[scenario["genre"]], visible["brief"])
            self.assertIn(
                context, visible["brief"] + "\n" + "\n".join(visible["initial_files"].values())
            )

    def test_loose_requests_have_no_hidden_explicit_constraints(self):
        from collections import Counter

        scenarios = self.compile()
        self.assertEqual(
            Counter((s["family"], s["instruction_specificity"]) for s in scenarios),
            {(f"F{f}", kind): 5 for f in range(1, 6) for kind in ("explicit", "loose")},
        )
        for scenario in scenarios:
            if scenario["instruction_specificity"] != "loose":
                continue
            self.assertEqual(scenario["style"], "unspecified")
            checks = {c["id"] for c in scenario["labels"]["checks"]}
            self.assertFalse(
                checks
                & {
                    "word-budget",
                    "pov-style",
                    "preserve-ending",
                    "three-directions",
                    "wiki-words",
                    "wiki-links",
                }
            )
            self.assertNotIn("120–220", scenario["visible"]["brief"])
        scenario = next(s for s in scenarios if s["id"] == "F1-06")
        result = {
            "status": "completed",
            "output": "A scene.",
            "turns": [],
            "model": MODEL,
            "after": {},
        }
        card = mechanical_score(scenario, result)
        explicit = {**card, "instruction_specificity": "explicit"}
        report = build_report([card, explicit])
        self.assertEqual(len(report["groups"]), 2)
        self.assertEqual(
            {g["instruction_specificity"] for g in report["groups"]}, {"explicit", "loose"}
        )

    def test_judgment_merge_preserves_completion_invariants(self):
        scenario = next(s for s in self.compile() if s["id"] == "F2-01")
        for status, semantic_passed, missing, expected in [
            ("error", True, False, 0),
            ("completed", False, False, 0),
            ("completed", True, True, 0),
            ("completed", True, False, 1),
        ]:
            with self.subTest(status=status, semantic_passed=semantic_passed, missing=missing):
                result = {
                    "status": status,
                    "output": "Saved.",
                    "trace": [],
                    "usage": {},
                    "before": scenario["visible"]["initial_files"],
                    "after": {
                        **scenario["visible"]["initial_files"],
                        "drafts/scene.md": (
                            "<prose>Mara waited.</prose>\nThe harbor bell rang twice."
                        ),
                    },
                }
                if missing:
                    result["after"].pop("drafts/scene.md")
                card = mechanical_score(scenario, result)
                packet = grading_packet(scenario, result, card)
                judgment = {
                    "assessments": [
                        {
                            "metric": key,
                            "value": 3,
                            "evidence": ["Fixture prose"],
                            "rationale": "Fixture assessment",
                            "uncertainty": "Uncalibrated",
                            "dimensions": [
                                {"name": name, "value": 3, "reason": "Fixture"}
                                for name in rubric.get("dimensions", [])
                            ],
                        }
                        for key, rubric in packet["rubrics"].items()
                    ],
                    "checks": [
                        {
                            "id": check["id"],
                            "passed": semantic_passed,
                            "evidence": ["Fixture"],
                            "rationale": "Fixture",
                        }
                        for check in packet["checks"]
                    ],
                }
                merged = apply_judgment(
                    card,
                    packet,
                    {"status": "ok", "packet_hash": fingerprint(packet), "judgment": judgment},
                )
                self.assertEqual(merged["scores"]["Q3"]["value"], expected)
                self.assertEqual(merged["scores"]["Q3"]["denominator"], 1)

    def test_imported_content_reaches_overlap_and_profile_consumers(self):
        source = self.root / "story.html"
        source.write_text("<p>The old brass door stood open in the rain.</p>")
        for stale in (False, True):
            with self.subTest(stale_metadata=stale):
                metadata = {k: v for k, v in self.catalog[0].items() if k != "text"}
                metadata.update(provenance="human")
                if stale:
                    metadata["text"] = "Stale content must not survive importing."
                record = import_story(source, metadata, self.root / "imports")
                text = "The old brass door stood open in the rain."
                peer = {**record, "id": "peer", "text": text}
                self.assertEqual(len(overlap_audit([record, peer])), 1)
                self.assertEqual(record["text"], text)
                self.assertEqual(record["text_sha256"], fingerprint(record["text"]))
                validate_catalog([record])
                card = {
                    "model": MODEL,
                    "family": "F1",
                    "provenance": "synthetic",
                    "artifacts": [{"status": "ok", "text": text}],
                    "source_groups": [],
                }
                compared = compare_groups([card], [record], ProseFeatures(self.root / "features"))
                self.assertEqual(compared[0]["reference_samples"], 1)
                self.assertEqual(compared[0]["reference_hashes"], [record["text_sha256"]])

    def test_identical_kbs_keep_builder_attribution_on_resume(self):
        writer = next(s for s in self.compile() if s["id"] == "F5-01")
        results = []
        for builder_id in ("builder-one", "builder-two"):
            builder = {
                "identity": builder_id,
                "after": {"kb/index.md": "Shared fact."},
                "source_groups": writer["source_groups"],
            }
            scenario = kb_use_scenario(writer, builder, paths=["kb/index.md"])
            result = run_selected(
                [scenario],
                MODEL,
                lambda: ScriptedBackend([answer("Done.")]),
                self.root / "runs",
                execute=True,
            )[0]
            self.assertEqual(result["builder_identity"], builder_id)
            started = json.loads((Path(result["path"]) / "started.json").read_text())
            self.assertEqual(started["builder_identity"], builder_id)

            def unexpected_backend():
                raise AssertionError("Matching observation must resume")

            resumed = run_selected(
                [scenario], MODEL, unexpected_backend, self.root / "runs", execute=True
            )[0]
            self.assertEqual(resumed["identity"], result["identity"])
            results.append(result)
        self.assertNotEqual(results[0]["identity"], results[1]["identity"])

    def test_catalog_lineage_role_and_upstream_boundaries(self):
        parent = self.catalog[0]
        child = {
            **parent,
            "id": "child",
            "parents": [parent["id"]],
            "role": "final_eval",
            "sha256": "changed",
            "work_id": "changed",
        }
        with self.assertRaisesRegex(ValueError, "crosses research roles"):
            validate_catalog([parent, child])
        with self.assertRaisesRegex(ValueError, "Upstream held-out"):
            validate_catalog([{**parent, "role": "train", "upstream_split": "test"}])
        child.update(role="development", parents=["missing"])
        with self.assertRaisesRegex(ValueError, "Missing parent"):
            validate_catalog([parent, child])

    def test_compile_private_projection_and_changed_package(self):
        scenarios = self.compile()
        self.assertEqual(len(scenarios), 50)
        for family in range(1, 6):
            self.assertEqual(sum(s["family"] == f"F{family}" for s in scenarios), 10)
        visible = json.loads((self.root / "release/visible/F4-01.json").read_text())
        self.assertNotIn("labels", visible)
        self.assertNotIn("probes", visible)
        path = self.root / "release/visible/F4-01.json"
        path.write_text(json.dumps({**visible, "brief": "changed"}))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_scenarios(self.root / "release")
        self.scenarios[0]["visible"]["secret_labels"] = {}
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            compile_scenarios(self.scenarios, self.catalog, self.root / "bad")

    def test_initial_five_scripted_runs_and_rescoring_without_generation(self):
        scenarios = [s for s in self.compile() if s["id"].endswith("-01")]
        for scenario in scenarios:
            family = scenario["family"]
            scene = "Mara held the ledger strap and waited for Ilan to speak."
            scripts = [answer(scene)]
            if family == "F2":
                scripts = [
                    tool(
                        "write_file",
                        path="drafts/scene.md",
                        content="<prose>" + scene + "</prose>\nThe harbor bell rang twice.",
                    ),
                    answer("Saved."),
                ]
            elif family == "F3":
                scripts = [
                    answer("1. Ask the keeper.\n2. Wait with Ilan.\n3. Inspect the tide marks.")
                ]
            elif family == "F4":
                scripts = [
                    tool("write_file", path="kb/index.md", content="[Canon](canon.md)"),
                    tool(
                        "write_file",
                        path="kb/canon.md",
                        content=(
                            "Only the retired keeper has the upper-room key. "
                            "Mara suspects Ilan; the accusation is unproven."
                        ),
                    ),
                    answer("Saved."),
                ]
            elif family == "F5":
                scripts = [
                    tool("read_file", path="kb/index.md"),
                    tool("write_file", path="drafts/scene.md", content=scene),
                    answer("Saved."),
                ]
            records = run_selected(
                [scenario],
                MODEL,
                lambda scripts=scripts: ScriptedBackend(scripts),
                self.root / "runs",
                execute=True,
            )
            self.assertEqual(records[0]["status"], "completed")
            self.assertFalse(records[0]["is_model_evaluation"])
            card = mechanical_score(scenario, records[0])
            self.assertEqual(len(card["artifacts"]), int(family in {"F1", "F2", "F5"}))
            if family == "F4":
                self.assertEqual(card["wiki_graph"]["reachability"], 1)
            with patch("writing_agent.agent.run_agent", side_effect=AssertionError("regenerated")):
                repeated = run_selected(
                    [scenario], MODEL, lambda: None, self.root / "runs", execute=True
                )
                self.assertEqual(repeated[0]["identity"], records[0]["identity"])
                self.assertEqual(mechanical_score(scenario, repeated[0]), card)
            report = build_report([card])
            self.assertEqual(report["attempts"], 1)
            self.assertEqual(
                card["scores"]["Q2"]["status"],
                "pending" if family in {"F1", "F2", "F5"} else "not_applicable",
            )

    def test_interrupted_attempt_remains_in_report_denominator(self):
        scenario = self.compile()[0]
        result = run_selected(
            [scenario],
            MODEL,
            lambda: ScriptedBackend([answer("Done")]),
            self.root / "runs",
            execute=True,
        )[0]
        (Path(result["path"]) / "result.json").unlink()
        recovered = saved_results(self.root / "runs")
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["status"], "interrupted")
        self.assertEqual(mechanical_score(scenario, recovered[0])["scores"]["Q3"]["value"], 0)

    def test_no_execution_default_and_failure_resume(self):
        scenario = self.compile()[0]

        def forbidden():
            raise AssertionError("should not execute")

        planned = run_selected([scenario], MODEL, forbidden, self.root / "runs")
        self.assertEqual(planned[0]["status"], "planned")
        first = run_selected(
            [scenario], MODEL, lambda: ScriptedBackend([]), self.root / "runs", execute=True
        )[0]
        self.assertEqual(first["status"], "error")
        resumed = run_selected([scenario], MODEL, forbidden, self.root / "runs", execute=True)[0]
        self.assertEqual(resumed["identity"], first["identity"])
        retry = run_selected(
            [scenario],
            MODEL,
            lambda: ScriptedBackend([answer("Done")]),
            self.root / "runs",
            execute=True,
            retry_failed=True,
        )[0]
        self.assertEqual(retry["status"], "completed")
        self.assertNotEqual(retry["path"], first["path"])

    def test_followups_tool_restrictions_and_read_storage_budgets(self):
        workspace = Workspace(self.root / "workspace", max_total_bytes=30)
        workspace.write_file("note.md", "one two three four")
        script = [
            tool("write_file", path="bad", content="bad"),
            tool("read_file", path="note.md"),
            answer("First"),
            answer("Second"),
        ]
        trace = []
        result = run_agent(
            ScriptedBackend(script),
            workspace,
            [],
            tools=["read_file"],
            max_read_tokens=1,
            followups=["Now revise."],
            emit=trace.append,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual([t["output"] for t in result["turns"]], ["First", "Second"])
        self.assertNotIn("bad", workspace.snapshot())
        self.assertEqual(result["read_tokens"], 0)
        self.assertFalse(trace[2]["observation"]["valid"])
        with self.assertRaisesRegex(ValueError, "storage budget"):
            workspace.write_file("big", "x" * 20)

    def test_mixed_reply_files_ambiguity_and_duplicate_delivery(self):
        result = {
            "output": "Note. <prose>A scene.</prose> Done.",
            "after": {"draft.md": "A scene."},
        }
        selectors = [
            {
                "id": "reply",
                "kind": "reply",
                "selection": "delimited",
                "start": "<prose>",
                "end": "</prose>",
                "delivery_group": "scene",
            },
            {"id": "file", "kind": "file", "path": "draft.md", "delivery_group": "scene"},
        ]
        artifacts = extract_prose(result, selectors)
        self.assertEqual(artifacts[0]["text"], "A scene.")
        self.assertEqual(artifacts[1]["status"], "duplicate_delivery")
        result["output"] += "<prose>Another.</prose>"
        self.assertEqual(extract_prose(result, selectors)[0]["status"], "needs_review")
        self.assertEqual(
            extract_prose(result, [{"kind": "file", "path": "missing"}])[0]["status"],
            "missing_prose",
        )
        selected = extract_prose(
            result,
            [{"kind": "reply", "selection": "span", "start": 0, "end": 4, "source_hash": "wrong"}],
        )
        self.assertEqual(selected[0]["status"], "needs_review")

    def test_navigation_and_generated_kb_do_not_inherit_builder_conversation(self):
        scenarios = self.compile()
        builder = next(s for s in scenarios if s["id"] == "F4-01")
        writer = next(s for s in scenarios if s["id"] == "F5-01")
        result = {
            "identity": "builder",
            "after": {"kb/index.md": "Fact", "private.txt": "secret"},
            "source_groups": builder["source_groups"],
            "messages": [answer("HIDDEN HISTORY")],
        }
        derived = kb_use_scenario(writer, result, paths=["kb/index.md"])
        self.assertEqual(derived["visible"]["initial_files"], {"kb/index.md": "Fact"})
        observed = []

        class Reader:
            def complete(self, messages, tools, *, emit=lambda event: None):
                from writing_agent.backends import Completion

                observed.extend(messages)
                return Completion(answer("Fact"))

        probes = run_navigation(builder, result, MODEL, Reader, self.root / "probes", execute=True)
        self.assertEqual(len(probes), 2)
        self.assertNotIn("HIDDEN HISTORY", json.dumps(observed))
        self.assertNotIn("secret", json.dumps(observed))

    def test_html_import_and_markdown_controls(self):
        source = self.root / "story.html"
        source.write_text("<nav>menu</nav><p>One.</p><p>Two.</p><script>hidden</script>")
        imported = import_story(
            source, {**self.catalog[0], "provenance": "half_synthetic"}, self.root / "imports"
        )
        self.assertEqual(imported["provenance"], "half_synthetic")
        self.assertEqual(Path(imported["text_path"]).read_text(), "One.\n\nTwo.")
        sections = {"canon": "A fact.", "beliefs": "An uncertainty."}
        for linked in (False, True):
            files = render_kb(sections, linked=linked)
            graph = markdown_graph(files, ["kb/index.md"])
            self.assertEqual(graph["valid_fraction"], 1)
            self.assertEqual(graph["reachability"], 1)
        graph = markdown_graph({"kb/index.md": "[x](missing.md)\n[x](#absent)"}, ["kb/index.md"])
        self.assertEqual(graph["valid_fraction"], 0)

    def test_judge_failure_pending_and_identity_validation(self):
        scenario = self.compile()[0]
        result = {
            "status": "completed",
            "output": "Prose.",
            "before": {},
            "after": {},
            "model": MODEL,
            "usage": {},
            "identity": "sample",
            "trace": [],
        }
        card = mechanical_score(scenario, result)
        packet = grading_packet(scenario, result, card)
        self.assertNotIn("model", packet)
        self.assertEqual(
            apply_judgment(card, packet, {"status": "grader_error"})["scores"]["Q2"]["status"],
            "pending",
        )
        with self.assertRaises(ValueError):
            validate_judgment(packet, {"assessments": [], "checks": []})

    def test_preparation_preserves_human_review(self):
        destination = self.root / "review"
        write_review(self.scenarios[:1], destination)
        path = destination / "scenario-review.json"
        records = json.loads(path.read_text())
        records[0].update(review_status="accepted", reviewer="writer", comments="Keep this.")
        path.write_text(json.dumps(records))
        write_review(self.scenarios[:1], destination)
        saved = json.loads(path.read_text())
        self.assertEqual(saved[0]["review_status"], "accepted")
        self.assertEqual(saved[0]["comments"], "Keep this.")

    def test_missing_prose_never_gets_a_literary_judgment(self):
        scenario = self.compile()[0]
        result = {
            "status": "error",
            "output": "",
            "before": {},
            "after": {},
            "usage": {},
            "trace": [],
        }
        card = mechanical_score(scenario, result)
        self.assertEqual(card["scores"]["Q3"]["value"], 0)
        self.assertEqual(card["scores"]["Q2"]["status"], "not_applicable")
        self.assertNotIn("Q2", grading_packet(scenario, result, card)["rubrics"])

    def test_script_inspection_does_not_execute(self):
        spec = importlib.util.spec_from_file_location(
            "research_script", ROOT / "scripts/evaluate.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, "generate", side_effect=AssertionError("called")):
            manifest = module.inspect_experiment()
        self.assertEqual(manifest["core_attempts"], 200)
        self.assertEqual(module.STAGES, ("inspect",))
        self.assertFalse(module.APPROVED_CANDIDATE_RUN)


class ProseTests(unittest.TestCase):
    def test_known_distance_and_negative_unbiased_mmd(self):
        self.assertAlmostEqual(distribution_l2([["a"]], [["b"]], 1), math.sqrt(2))
        self.assertIsNone(distribution_l2([["a"]], [["b"]], 2))
        self.assertAlmostEqual(mmd_squared([[-1], [1]], [[-1], [1]], 1), math.exp(-2) - 1)
        self.assertIsNone(mmd_squared([[1]], [[1]], 1))
        self.assertAlmostEqual(dispersion([[1, 0], [0, 1]]), 1)
        self.assertEqual(self_bleu([list("abcdef"), list("abcdef")]), 1)

    def test_features_cache_and_configuration_boundaries(self):
        from dataclasses import asdict

        text = "Red birds fly. Red birds fly."
        with tempfile.TemporaryDirectory() as temp:
            features = ProseFeatures(Path(temp))
            first = features.extract(text)
            self.assertEqual(first, features.extract(text))
            profile = prose_profile([text], [first])
            self.assertEqual(profile["metrics"]["D2"]["status"], "unavailable")
            self.assertGreater(first["lexical"]["repeated_trigram_rate"], 0)
            other = copy.deepcopy(first)
            other["config"] = asdict(FeatureConfig(version=2))
            with self.assertRaisesRegex(ValueError, "cannot be pooled"):
                prose_profile([text], [first], references=[other])
            with self.assertRaisesRegex(ValueError, "do not match"):
                prose_profile(["Other"], [first])

    def test_empty_prose_is_not_perfect(self):
        self.assertIsNone(lexical_features("")["type_token_ratio"])
        self.assertIsNone(distribution_l2([], [], 1))
        with self.assertRaises(ValueError):
            mmd_squared([[0], [1]], [[0, 1], [1, 2]], 1)
        with self.assertRaises(ValueError):
            dispersion([[0, 0], [1, 0]])
