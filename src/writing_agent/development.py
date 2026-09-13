"""Compile the authored, synthetic development cases from inspectable world records."""

import copy
import json
from pathlib import Path

from writing_agent.catalog import fingerprint
from writing_agent.suite import render_kb

READ_TOOLS = ["list_dir", "read_file", "search"]
WRITE_TOOLS = [*READ_TOOLS, "write_file", "patch_file"]


def _check(identity, metric, kind, *, method="deterministic", required=False, **fields):
    return {
        "id": identity,
        "metric": metric,
        "kind": kind,
        "method": method,
        "required": required,
        **fields,
    }


def _rubric(description, *, fraction=False):
    return {
        "description": description,
        "range": [0, 1] if fraction else [1, 5],
        "anchors": (
            "0: none satisfied; 1: all satisfied. State numerator and denominator."
            if fraction
            else "1: fails the brief; 2: substantial weaknesses; "
            "3: usable with revision; 4: effective with minor weaknesses; "
            "5: consistently effective. Cite concrete evidence."
        ),
    }


PROSE_RUBRIC = _rubric(
    "Rate selected prose holistically for coherence, characterization, pacing, language, "
    "and unnecessary explanation. Discuss applicable dialogue, imagery, subtext and style. "
    "Do not penalize an allowed alternative plot for differing from a reference."
)

PROSE_RUBRIC["dimensions"] = [
    "coherence",
    "characterization",
    "pacing",
    "language",
    "redundancy",
    "style_adherence",
]


def _loosen_request(scenario: dict, world: dict) -> None:
    """Keep the deliverable observable while leaving creative choices to the writer."""
    visible, labels = scenario["visible"], scenario["labels"]
    family = scenario["family"]
    scenario["instruction_specificity"] = "loose"
    scenario["style"] = "unspecified"
    labels["rubric_version"] = 2
    labels["checks"] = [
        c
        for c in labels["checks"]
        if c["id"]
        not in {
            "word-budget",
            "pov-style",
            "preserve-ending",
            "three-directions",
            "wiki-words",
            "wiki-links",
        }
    ]
    if family == "F1":
        visible["brief"] = (
            world["source"]
            + "\n\nCould you turn this into a scene? "
            + ("I'd like it to feel more alive. Just the story, please.")
        )
    elif family == "F2":
        request = (
            "The scene in drafts/scene.md feels flat. Can you rework it?"
            if scenario["condition"] == "local_revision"
            else "Could you make a scene out of the notes and save it in drafts/scene.md?"
        )
        visible["brief"] = request + (
            " The background is in notes/source.md. Leave the notes alone."
        )
        visible["prose"] = [{"id": "scene", "kind": "file", "path": "drafts/scene.md"}]
    elif family == "F3":
        visible["brief"] = world["source"] + (
            "\n\nI'm not sure where to take this. What are some directions I could try?"
        )
        if visible["followups"]:
            visible["followups"] = [
                "These feel a bit too busy. Could we lean into what's going on between them?"
            ]
        labels["rubrics"]["Q6"] = _rubric(
            "Assess meaningful variety among the viable directions actually offered. "
            "Do not require a fixed number, numbered formatting, or a prescribed explanation. "
            "Count meaningfully different viable pairs divided by all pairs of offered directions. "
            "Fewer than two directions yields zero; superficial rephrasings are not diverse.",
            fraction=True,
        )
    elif family == "F4":
        visible["brief"] = (
            "Can you turn source/chapter.md into a small Markdown wiki I can use while "
            "writing the next chapter? Put the starting page at kb/index.md and make "
            "things easy to find. Leave the chapter itself alone."
        )
        for check in labels["checks"]:
            if check["metric"] == "Q8":
                check["required"] = False
                check["text"] = (
                    "Assess whether recording this information would help a writer continue "
                    "the chapter. Accept a defensible omission or alternative placement; "
                    "do not treat this candidate fact as a mandatory checklist item: "
                    + check["text"]
                )
        for fact in labels["knowledge"]:
            if fact["importance"] == "required":
                fact["importance"] = "optional"
                fact["omission"] = "Accept if defensible for next-chapter planning."
    else:
        if scenario["condition"] == "supplied_kb":
            visible["brief"] = (
                world["genre_context"]
                + "\n\n"
                + world["required"]
                + "\n\n"
                + world["optional"]
                + "\n\n"
                + (world["belief"] + "\n\nCan you write a scene from these notes? Just the story.")
            )
            visible["prose"] = [{"id": "scene", "kind": "reply", "selection": "whole"}]
        else:
            visible["brief"] = (
                "Could you write a scene using the notes in kb/? Save it in drafts/scene.md. "
                "Keep the notes as they are for now."
            )
    for check in labels["checks"]:
        if check["id"] == "continuity":
            check["text"] = (
                "Respect the supplied starting facts and distinguish character beliefs from "
                "established knowledge. Accept plausible new events, revelations, dialogue, "
                "viewpoint, and endings; no particular continuation is required."
            )
    for rubric in labels["rubrics"].values():
        rubric["description"] += (
            " The request leaves creative choices open. Do not invent unstated requirements "
            "for length, viewpoint, style, structure, ending, or number of ideas."
        )


def author_development(world_path: Path) -> tuple[list[dict], list[dict]]:
    """50 proposed cases across ten worlds; labels remain pending human review."""
    worlds = json.loads(world_path.read_text())
    genres = json.loads(world_path.with_name("genres.json").read_text())
    sources, scenarios = [], []
    for i, world in enumerate(worlds):
        source_id = "world-" + world["id"]
        sources.append(
            {
                "id": source_id,
                "work_id": source_id,
                "role": "development",
                "text": world["source"],
                "sha256": fingerprint(world["source"]),
                "provenance": "synthetic",
                "parents": [],
                "revision": fingerprint(world),
                "generator": "Codex authored development fixture",
                "terms": {
                    "origin": "project-generated",
                    "evaluation": "allowed",
                    "training": "excluded: development world",
                },
                "upstream_split": "development",
                "review_status": "pending",
            }
        )
        base_world = world
        base_source = sources[-1]
        for family in range(1, 6):
            genre = genres[(i + 2 * (family - 1)) % len(genres)]
            world = copy.deepcopy(base_world)
            world["genre_context"] = genre["context"].format(**world)
            world["source"] = world["genre_context"] + "\n\n" + world["source"]
            world["style"] = world.get("prose_style", world["style"])
            derived_id = f"{source_id}-{genre['id']}"
            sources.append(
                {
                    **base_source,
                    "id": derived_id,
                    "text": world["source"],
                    "sha256": fingerprint(world["source"]),
                    "parents": [source_id],
                    "revision": fingerprint(world),
                    "transformations": [{"kind": "authored_genre_context", "genre": genre["id"]}],
                }
            )
            kb_sections = {
                "canon": world["required"] + "\n\n" + world["optional"],
                "interpretations": world["belief"],
                "story-context": world["genre_context"],
            }
            key = f"F{family}-{i + 1:02d}"
            visible = {
                "brief": "",
                "initial_files": {},
                "followups": [],
                "tools": [],
                "budgets": {
                    "max_steps": 12,
                    "max_tool_calls": 24,
                    "max_read_tokens": 4000,
                    "max_total_bytes": 24_000,
                },
                "prose": [],
            }
            checks, rubrics, knowledge = [], {}, []
            condition = "direct"
            subject = f"{world['pov']} and {world['other']} in {world['place']}"
            writing = (
                f"Write 120–220 words about {subject}, in {world['style']}. "
                f"Use close third person centered on {world['pov']}. End before the "
                "central uncertainty is resolved. Invent dialogue and sensory details, "
                "but respect established facts and character knowledge."
            )
            if family == 1:
                condition = ["scene", "continuation", "dialogue", "reflection", "action"][i % 5]
                emphasis = {
                    "scene": "Stage a disagreement over the object.",
                    "continuation": "Continue immediately after the supplied passage.",
                    "dialogue": "Let most of the scene unfold through dialogue.",
                    "reflection": "Center a private hesitation before an outward decision.",
                    "action": "Show an interrupted practical task rather than summarizing.",
                }
                visible["brief"] = (
                    world["source"]
                    + "\n\n"
                    + writing
                    + " "
                    + emphasis[condition]
                    + " Return only the prose, without a title or commentary."
                )
                visible["prose"] = [{"id": "scene", "kind": "reply", "selection": "whole"}]
            elif family == 2:
                visible["tools"] = WRITE_TOOLS
                visible["initial_files"] = {"notes/source.md": world["source"]}
                if i % 2 == 0:
                    condition = "local_revision"
                    visible["initial_files"]["drafts/scene.md"] = world["revision"]
                    visible["brief"] = (
                        f"Read notes/source.md and drafts/scene.md. Expand only the "
                        f"first sentence into 120–220 words in {world['style']}. "
                        "Keep the final sentence exactly unchanged. Put only the "
                        "replacement prose between <prose> and </prose> in the "
                        "file, then retain the final sentence after </prose>. "
                        f"Use close third person on {world['pov']}. "
                        "Preserve facts and unresolved beliefs. Reply with a short "
                        "completion note; do not repeat the scene."
                    )
                    visible["prose"] = [
                        {
                            "id": "scene",
                            "kind": "file",
                            "path": "drafts/scene.md",
                            "selection": "delimited",
                            "start": "<prose>",
                            "end": "</prose>",
                        }
                    ]
                    checks.append(
                        _check(
                            "preserve-ending",
                            "Q1",
                            "protected",
                            required=True,
                            path="drafts/scene.md",
                            text=world["protected"],
                        )
                    )
                else:
                    condition = "explore_then_write"
                    visible["brief"] = (
                        "Read notes/source.md. "
                        + writing
                        + " Save only prose in drafts/scene.md. Reply briefly afterward."
                    )
                    visible["prose"] = [{"id": "scene", "kind": "file", "path": "drafts/scene.md"}]
                checks.extend(
                    [
                        _check(
                            "saved-scene", "Q3", "nonempty", required=True, path="drafts/scene.md"
                        ),
                        _check(
                            "edit-scope",
                            "Q1",
                            "allowed_changes",
                            required=True,
                            paths=["drafts/scene.md"],
                        ),
                    ]
                )
            elif family == 3:
                condition = "feedback" if i % 2 else "alternatives"
                visible["brief"] = (
                    world["source"] + "\n\nGive three numbered, meaningfully "
                    "different next-scene directions. For each, explain the dramatic "
                    "choice, its consequence, and why it fits known facts. Do not "
                    "draft the scene or resolve the central uncertainty."
                )
                if i % 2:
                    visible["followups"] = [
                        "Revise all three directions to reduce physical action and increase "
                        "interpersonal tension. Keep them distinct; explain what changed."
                    ]
                rubrics["Q5"] = _rubric(
                    "Assess usefulness, feasibility within canon, and actionable "
                    "development; honor the latest feedback."
                )
                rubrics["Q6"] = _rubric(
                    "Among all three requested pairs, count pairs whose two "
                    "alternatives are viable and meaningfully different in "
                    "dramatic choice or consequence. Missing alternatives earn "
                    "no diverse pairs. Return diverse viable pairs / 3.",
                    fraction=True,
                )
                checks.append(
                    _check(
                        "three-directions",
                        "Q1",
                        "semantic",
                        method="llm_judge",
                        required=True,
                        text="Exactly three developed viable directions.",
                    )
                )
            elif family == 4:
                condition = "update" if i % 2 else "construction"
                visible["tools"] = WRITE_TOOLS
                visible["initial_files"] = {"source/chapter.md": world["source"]}
                visible["brief"] = (
                    "Read source/chapter.md. Create a navigable Markdown continuity "
                    "wiki under kb/, starting at kb/index.md. Its audience is a "
                    "writer planning the next chapter. Prioritize consequential "
                    "facts, uncertainty, and character knowledge. At least one "
                    "linked topic page is needed. Omit incidental details where "
                    "appropriate; cite the source path. Keep source files unchanged. "
                    "Use no more than 500 words across KB pages."
                )
                if i % 2:
                    visible["followups"] = [
                        "Draft-only possibility, not accepted canon: "
                        + world["forbidden"]
                        + " Do not update canon from this suggestion.",
                        world["update"] + " Update the wiki for this accepted revision; preserve "
                        "relevant history and unrelated facts.",
                    ]
                knowledge = [
                    {
                        "id": "essential",
                        "statement": world["required"],
                        "importance": "required",
                        "weight": 3,
                        "evidence": world["source"],
                        "after_update": world["new_state"] if i % 2 else None,
                    },
                    {
                        "id": "belief",
                        "statement": world["belief"],
                        "importance": "required",
                        "weight": 3,
                        "evidence": world["source"],
                        "placement": "Any findable page; keep attribution and uncertainty.",
                    },
                    {
                        "id": "context",
                        "statement": world["optional"],
                        "importance": "optional",
                        "weight": 1,
                        "evidence": world["source"],
                    },
                    {
                        "id": "incidental",
                        "statement": world["incidental"],
                        "importance": "incidental",
                        "weight": 0,
                        "evidence": world["source"],
                        "omission": "allowed",
                    },
                ]
                checks.extend(
                    [
                        _check("wiki-words", "Q1", "kb_word_budget", max=500),
                        _check("wiki-links", "Q1", "wiki_links", required=True),
                        _check("wiki-index", "Q3", "nonempty", path="kb/index.md", required=True),
                        _check(
                            "wiki-only", "Q1", "allowed_changes", paths=["kb/*.md"], required=True
                        ),
                        _check(
                            "consequential-state",
                            "Q8",
                            "semantic",
                            method="llm_judge",
                            weight=3,
                            required=True,
                            text=world["new_state"] if i % 2 else world["required"],
                        ),
                        _check(
                            "qualified-belief",
                            "Q8",
                            "semantic",
                            method="llm_judge",
                            weight=3,
                            required=True,
                            text=world["belief"],
                        ),
                        _check(
                            "optional-context",
                            "Q8",
                            "semantic",
                            method="llm_judge",
                            weight=1,
                            text=world["optional"],
                        ),
                    ]
                )
                rubrics["Q7"] = _rubric(
                    "Identify factual KB claims and count source-supported "
                    "claims / checked claims. Distinguish uncertainty and "
                    "qualified interpretation; an empty KB cannot pass.",
                    fraction=True,
                )
                rubrics["Q9"] = _rubric(
                    "Assess evidence, entity and temporal understanding, "
                    "selection for the stated purpose, and uncertainty. "
                    "Accept defensible alternative interpretations and placement."
                )
                if i % 2:
                    checks.append(
                        _check(
                            "accepted-update",
                            "Q11",
                            "semantic",
                            method="llm_judge",
                            required=True,
                            text=world["new_state"] + " Preserve relevant "
                            "history and do not adopt the draft-only suggestion.",
                        )
                    )
            else:
                condition = "supplied_kb" if i % 2 else "retrieved_kb"
                kb = render_kb(kb_sections, linked=i % 3 != 0)
                if i % 2:
                    visible["brief"] = "\n\n".join(kb_sections.values()) + "\n\n" + writing
                    visible["brief"] += " Return a short note, then prose inside <prose></prose>."
                    visible["prose"] = [
                        {
                            "id": "scene",
                            "kind": "reply",
                            "selection": "delimited",
                            "start": "<prose>",
                            "end": "</prose>",
                        }
                    ]
                else:
                    visible["initial_files"] = kb
                    visible["tools"] = WRITE_TOOLS
                    visible["brief"] = (
                        "Use the continuity wiki starting at kb/index.md. "
                        + writing
                        + " Save prose in drafts/scene.md; keep the KB unchanged."
                    )
                    visible["prose"] = [{"id": "scene", "kind": "file", "path": "drafts/scene.md"}]
                    checks.extend(
                        [
                            _check(
                                "retrieve-state", "Q12", "evidence_exposed", text=world["required"]
                            ),
                            _check(
                                "retrieve-belief", "Q12", "evidence_exposed", text=world["belief"]
                            ),
                            _check(
                                "preserve-kb",
                                "Q1",
                                "allowed_changes",
                                paths=["drafts/scene.md"],
                                required=True,
                            ),
                        ]
                    )
            if family in {1, 2, 5}:
                rubrics["Q2"] = copy.deepcopy(PROSE_RUBRIC)
                checks.extend(
                    [
                        _check(
                            "word-budget", "Q1", "word_range", artifact="scene", min=120, max=220
                        ),
                        _check(
                            "continuity",
                            "Q13",
                            "semantic",
                            method="llm_judge",
                            required=True,
                            text=f"Preserve {world['required']} Keep {world['belief']} "
                            f"Do not assert: {world['forbidden']}",
                        ),
                        _check(
                            "pov-style",
                            "Q1",
                            "semantic",
                            method="llm_judge",
                            text=f"Close third person on {world['pov']}; {world['style']}.",
                        ),
                    ]
                )
            scenario = {
                "id": key,
                "family": f"F{family}",
                "role": "development",
                "condition": condition,
                "style": world["style"],
                "instruction_specificity": "explicit",
                "source_ids": [derived_id],
                "genre": genre["id"],
                "provenance": "synthetic",
                "review_status": "pending",
                "visible": visible,
                "labels": {
                    "rubric_version": 1,
                    "checks": checks,
                    "rubrics": rubrics,
                    "knowledge": knowledge,
                    "source_cutoff": "supplied passage only",
                },
            }
            if family == 4:
                scenario["labels"]["entrypoints"] = ["kb/index.md"]
                scenario["labels"]["probes"] = [
                    {
                        "id": "state",
                        "question": "What practical restriction or permission governs "
                        "the next action? Answer with supporting page paths.",
                        "expected": world["new_state"] if i % 2 else world["required"],
                    },
                    {
                        "id": "belief",
                        "question": "Which central belief is still uncertain? "
                        "Name its owner and supporting page paths.",
                        "expected": world["belief"],
                    },
                ]
            if i >= 5:
                _loosen_request(scenario, world)
            visible["brief"] += f"\n\nThis is for a {genre['label']} story."
            if family in {1, 2, 5}:
                scenario["labels"]["checks"].append(
                    _check(
                        "genre-fit",
                        "Q1",
                        "semantic",
                        method="llm_judge",
                        text=f"Fit the requested {genre['label']} genre in a defensible way; "
                        "do not require a fixed trope checklist. Genre blends are allowed "
                        "where the source premise supports them.",
                    )
                )
            scenarios.append(scenario)
    return sources, scenarios
