"""The external long-form anchor: EQ-Bench Longform Writing, pinned and adapted.

Our own long-form suite is authored from the same taxonomy that produces our training
tasks and our reward, so it can show that a policy *moved*, never that it moved in a
better direction. This module binds one measurement we did not design.

Upstream runs a 13-step protocol per prompt: five planning turns, then eight
approximately-1000-word chapters, and judges each chapter and the whole piece with
Claude Sonnet 4.6 against fourteen 0-20 criteria. The criteria, negative criteria,
weights and arithmetic below are reimplemented from the pinned upstream source.

Two divergences are deliberate and must be reported alongside any score:

- Upstream's staccato penalty interpolates with the wrong coefficient, so its factor
  falls to 0.4 at an index of 18 and then jumps back to the 0.6 cap. That is a
  discontinuity, not a policy. `staccato_factor` implements the continuous penalty its
  own comment describes. Set ``legacy_curve=True`` to reproduce the upstream numbers.
- Criteria are filtered to the declared set and a missing criterion is reported, so a
  judge that invents or drops a metric cannot silently move the score.

This is an adaptation of an official benchmark, not an official leaderboard entry. A
different harness, context window or sampling configuration changes the number; label
it as such wherever it is quoted.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from writing_agent.catalog import fingerprint

EQ_REVISION = "34f60a028c3f973c19cde98dc5a9e8f9875a87e3"
SOURCE = "EQ-Bench Longform Writing"
PLANNING_STEPS = 5
DEFAULT_CHAPTERS = 8
CHAPTER_WORDS = 1000
SCORE_MIN, SCORE_MAX = 0, 20
DEFAULT_WEIGHT = 1.0
FINAL_WEIGHT = 1.0
FORCED_METAPHOR = "forced poetry or metaphor"
FORCED_METAPHOR_EXPONENT = 1.7
CRITERIA_WEIGHTS = {FORCED_METAPHOR: 5.0, "purple prose": 1.0}
JUDGE_MODEL = "claude-sonnet-4-6"

# Upstream's staccato curve: free below 5, capped at a 40% penalty from 18 upward.
STACCATO_FREE, STACCATO_CEILING, STACCATO_FLOOR = 5.0, 18.0, 0.6
READ_TOOLS = ["list_dir", "read_file", "search"]
WRITE_TOOLS = [*READ_TOOLS, "write_file", "patch_file"]

PLANNING_TEMPLATES = (
    "prompt1.txt",
    "prompt2.txt",
    "prompt3.txt",
    "prompt4.txt",
    "prompt5.txt",
)
CHAPTER_TEMPLATES = (
    "prompt_chapter_first.txt",
    "prompt_chapter_intermediate.txt",
    "prompt_chapter_last.txt",
)
REQUIRED_FILES = (
    *PLANNING_TEMPLATES,
    *CHAPTER_TEMPLATES,
    "criteria_weights.json",
    "longform_creative_writing_criteria_chapter.txt",
    "longform_creative_writing_criteria_final.txt",
    "longform_negative_criteria_chapter.txt",
    "longform_negative_criteria_final.txt",
    "longform_creative_writing_judging_prompt_chapter.txt",
    "longform_creative_writing_judging_prompt_final.txt",
    "longform_creative_writing_prompts_minimalist.json",
)
NON_METRIC_LINES = frozenset(
    {"overall assessment", "summary", "reasoning", "critique", "feedback", "notes"}
)
_SCORE_LINE = re.compile(r"^\s*([^:]+?)\s*:\s*\[?(-?\d+(?:\.\d+)?)\]?\s*$")


@dataclass(frozen=True)
class Release:
    """The pinned upstream fixtures, loaded once and passed to the pure functions."""

    prompts: tuple[dict, ...]
    templates: dict
    criteria_chapter: tuple[str, ...]
    criteria_final: tuple[str, ...]
    negative_chapter: tuple[str, ...]
    negative_final: tuple[str, ...]
    judge_chapter: str
    judge_final: str


def _lines(path: Path) -> tuple[str, ...]:
    return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())


def load_release(raw: Path) -> Release:
    """Load the pinned fixtures. ``raw`` is the acquired ``eqbench-longform`` directory."""
    data = raw / "data"
    missing = [name for name in REQUIRED_FILES if not (data / name).exists()]
    if missing:
        raise ValueError(f"Missing upstream fixtures under {data}: {', '.join(missing)}")
    prompts = json.loads((data / "longform_creative_writing_prompts_minimalist.json").read_text())
    ordered = [prompts[key] for key in sorted(prompts, key=int)]
    return Release(
        prompts=tuple(ordered),
        templates={
            name: (data / name).read_text() for name in (*PLANNING_TEMPLATES, *CHAPTER_TEMPLATES)
        },
        criteria_chapter=_lines(data / "longform_creative_writing_criteria_chapter.txt"),
        criteria_final=_lines(data / "longform_creative_writing_criteria_final.txt"),
        negative_chapter=_lines(data / "longform_negative_criteria_chapter.txt"),
        negative_final=_lines(data / "longform_negative_criteria_final.txt"),
        judge_chapter=(data / "longform_creative_writing_judging_prompt_chapter.txt").read_text(),
        judge_final=(data / "longform_creative_writing_judging_prompt_final.txt").read_text(),
    )


def render_steps(release: Release, writing_prompt: str, *, chapters: int = DEFAULT_CHAPTERS):
    """The upstream user turns: five planning steps, then one per chapter."""
    if type(chapters) is not int or chapters < 2:
        raise ValueError("A long-form run needs at least two chapters")
    fields = {"writing_prompt": writing_prompt, "n_chapters": chapters}
    first, middle, last = CHAPTER_TEMPLATES
    steps = [release.templates[name].format(**fields) for name in PLANNING_TEMPLATES]
    steps.append(release.templates[first].format(**fields))
    steps.extend(
        release.templates[middle].format(chapter_number=number, **fields)
        for number in range(2, chapters)
    )
    steps.append(release.templates[last].format(chapter_number=chapters, **fields))
    return tuple(steps)


def chapter_turns(chapters: int = DEFAULT_CHAPTERS) -> tuple[int, ...]:
    """Turn indices holding chapters, given that turn 0 answers the brief."""
    return tuple(range(PLANNING_STEPS, PLANNING_STEPS + chapters))


def _case_steps(mode: str, steps: tuple[str, ...], chapters: int) -> tuple[str, ...]:
    """In workspace mode a chapter step also names where to save it."""
    if mode == "faithful":
        return steps
    head = steps[:PLANNING_STEPS]
    saved = tuple(
        f"{step}\n\nAlso save this chapter to manuscript/chapter-{number}.md."
        for number, step in enumerate(steps[PLANNING_STEPS:], start=1)
    )
    return head + saved


def benchmark_release(raw: Path, *, chapters: int = DEFAULT_CHAPTERS, mode: str = "faithful"):
    """Compile the upstream prompts into final-eval scenarios this harness can run.

    ``mode="faithful"`` keeps generation reply-only exactly as upstream does, which is
    what makes a score comparable to the leaderboard. ``mode="workspace"`` writes the
    chapters to files so the run also exercises project-file delivery; that changes the
    protocol, so its score is an adaptation and must be reported as one. Scoring is
    identical in both modes; only the prose selector and the delivery step differ.
    """
    if mode not in {"faithful", "workspace"}:
        raise ValueError(f"Unknown mode: {mode}")
    release = load_release(raw)
    catalog, scenarios = [], []
    for index, prompt in enumerate(release.prompts, start=1):
        source_id = f"eqbench-longform-{index:02d}"
        steps = _case_steps(
            mode, render_steps(release, prompt["writing_prompt"], chapters=chapters), chapters
        )
        catalog.append(
            {
                "id": source_id,
                "work_id": source_id,
                "role": "final_eval",
                "provenance": "human",
                "text": prompt["writing_prompt"],
                "sha256": fingerprint(prompt["writing_prompt"]),
                "parents": [],
                "revision": fingerprint(prompt),
                "review_status": "upstream",
                "transformations": [],
                "terms": {
                    "origin": f"EQ-Bench Longform Writing {EQ_REVISION}",
                    "evaluation": "allowed",
                    "training": "excluded: external benchmark",
                },
            }
        )
        workspace = mode == "workspace"
        scenarios.append(
            {
                "id": f"EQLF-{index:02d}",
                "family": "F1",
                "role": "final_eval",
                "condition": "novella",
                "style": "unspecified",
                "instruction_specificity": "upstream",
                "genre": prompt["category"],
                "provenance": "synthetic",
                "review_status": "upstream",
                "source_ids": [source_id],
                "benchmark": SOURCE,
                "upstream_revision": EQ_REVISION,
                "adaptation": mode,
                "chapters": chapters,
                "visible": {
                    "brief": steps[0],
                    "initial_files": {},
                    "followups": list(steps[1:]),
                    "tools": WRITE_TOOLS if workspace else [],
                    "budgets": {
                        "max_steps": 2 * len(steps),
                        "max_tool_calls": 2 * chapters if workspace else 0,
                        "max_read_tokens": 4000 if workspace else 0,
                        "max_total_bytes": 4_000_000 if workspace else 1_000,
                    },
                    "prose": [
                        {
                            "id": f"chapter-{number}",
                            **(
                                {"kind": "file", "path": f"manuscript/chapter-{number}.md"}
                                if workspace
                                else {"kind": "reply", "turn": turn}
                            ),
                            "selection": "whole",
                        }
                        for number, turn in zip(
                            range(1, chapters + 1), chapter_turns(chapters), strict=True
                        )
                    ],
                },
                "labels": {
                    "rubric_version": 1,
                    "benchmark": SOURCE,
                    "upstream_revision": EQ_REVISION,
                    "criteria_chapter": list(release.criteria_chapter),
                    "negative_chapter": list(release.negative_chapter),
                    "criteria_final": list(release.criteria_final),
                    "negative_final": list(release.negative_final),
                    "criteria_weights": dict(CRITERIA_WEIGHTS),
                    "judge_model": JUDGE_MODEL,
                    "score_scale": [SCORE_MIN, SCORE_MAX],
                    "checks": [],
                    "rubrics": {},
                },
            }
        )
    return catalog, scenarios


def _clamp(value: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, value))


def _normalized(values) -> set[str]:
    return {value.strip().lower() for value in values}


def parse_scores(text: str, *, criteria=()) -> dict:
    """Read ``Metric: score`` lines, preferring an explicit ``[Scores]`` section.

    A judge that writes commentary between metrics is the normal case; a judge that
    writes a metric name twice takes its last value, matching the upstream parser.
    """
    wanted = _normalized(criteria)
    lines, collecting, sectioned = [], False, "[Scores]" in text or "--- Scores ---" in text
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "[Scores]" in line or "--- Scores ---" in line:
            collecting = True
            continue
        if collecting and ("---" in line or "[End Scores]" in line):
            collecting = False
            continue
        if not sectioned or collecting:
            lines.append(line)
    scores = {}
    for line in lines:
        match = _SCORE_LINE.match(line)
        if not match:
            continue
        metric = match.group(1).strip()
        if metric.lower() in NON_METRIC_LINES:
            continue
        if wanted and metric.lower() not in wanted:
            continue
        scores[metric] = _clamp(float(match.group(2)))
    return scores


def _weighted(scores: dict, *, negative, weights=None) -> float | None:
    table = CRITERIA_WEIGHTS if weights is None else weights
    total = weight_total = 0.0
    for metric, value in scores.items():
        normalized = metric.strip().lower()
        weight = table.get(normalized, DEFAULT_WEIGHT)
        total += _processed(metric, value, negative) * weight
        weight_total += weight
    return total / weight_total if weight_total else None


def _processed(metric: str, value: float, negative) -> float:
    """Put one criterion on the shared axis before weighting it."""
    score = _clamp(value)
    if metric.strip().lower() in _normalized(negative):
        score = SCORE_MAX - score
    if metric.strip().lower() == FORCED_METAPHOR:
        score = (score / SCORE_MAX) ** FORCED_METAPHOR_EXPONENT * SCORE_MAX
    return score


def judge_score(scores: dict, *, negative, weights=None) -> float | None:
    """One judge's weighted mean over its criteria, or None if none were usable."""
    return _weighted(dict(scores), negative=negative, weights=weights)


def missing_criteria(scores: dict, criteria) -> list[str]:
    """Declared criteria the judge did not return, compared case-insensitively."""
    returned = _normalized(scores)
    return [name for name in criteria if name.strip().lower() not in returned]


def staccato_factor(index: float, *, legacy_curve: bool = False) -> float:
    """Scale a chapter score down when it degrades into short single-sentence paragraphs.

    Upstream jumps from 0.4 to 0.6 at an index of 18 because its interpolation
    coefficient disagrees with its own 40% ceiling. The default here is the continuous
    curve the comment describes; ``legacy_curve`` reproduces the upstream numbers.
    """
    if index < STACCATO_FREE:
        return 1.0
    if index > STACCATO_CEILING:
        return STACCATO_FLOOR
    span = STACCATO_CEILING - STACCATO_FREE
    strength = 0.6 if legacy_curve else 1.0 - STACCATO_FLOOR
    return 1.0 - ((index - STACCATO_FREE) / span) * strength


def task_score(
    chapter_scores: list[float],
    final_scores: list[float],
    *,
    staccato: tuple[float, ...] = (),
) -> float | None:
    """Combine scored chapters and the whole-piece judgment, on the upstream 0-20 scale."""
    scaled = [
        score * staccato_factor(staccato[index] if index < len(staccato) else 0.0)
        for index, score in enumerate(chapter_scores)
    ]
    total = weight_total = 0.0
    if scaled:
        total += sum(scaled) / len(scaled) * len(scaled)
        weight_total += len(scaled)
    if final_scores:
        total += sum(final_scores) / len(final_scores) * FINAL_WEIGHT
        weight_total += FINAL_WEIGHT
    return total / weight_total if weight_total else None


def aggregate(scores: list[float | None]) -> dict:
    """Average tasks and report the 0-100 figure, keeping unscored tasks visible."""
    scored = [score for score in scores if score is not None]
    if not scored:
        return {
            "score_0_20": None,
            "score_0_100": None,
            "tasks_scored": 0,
            "tasks_total": len(scores),
            "reason": "No task could be scored",
        }
    mean = sum(scored) / len(scored)
    return {
        "score_0_20": mean,
        "score_0_100": mean * (100.0 / SCORE_MAX),
        "tasks_scored": len(scored),
        "tasks_total": len(scores),
        "reason": None,
    }


def degradation(chapter_scores: list[float]) -> float | None:
    """How far the last chapter fell below the first. Negative means it improved."""
    if len(chapter_scores) < 2:
        return None
    return chapter_scores[0] - chapter_scores[-1]


def judge_prompts(
    release: Release,
    *,
    writing_prompt: str,
    final_plan: str,
    character_profiles: str,
    chapters: dict[int, str],
) -> dict:
    """Render the upstream chapter and whole-piece judgments, unmodified in substance."""
    if not chapters:
        raise ValueError("A judged run needs at least one chapter")
    fields = {
        "writing_prompt": writing_prompt,
        "final_plan": final_plan,
        "character_profiles": character_profiles,
        "lower_is_better_criteria": "\n".join(release.negative_chapter),
        "creative_writing_criteria": "\n".join(release.criteria_chapter),
    }
    return {
        "chapters": {
            number: release.judge_chapter.format(chapter_number=number, chapter_text=text, **fields)
            for number, text in sorted(chapters.items())
        },
        "final": release.judge_final.format(
            full_story="\n\n".join(chapters[number] for number in sorted(chapters)),
            lower_is_better_criteria="\n".join(release.negative_final),
            creative_writing_criteria="\n".join(release.criteria_final),
            writing_prompt=writing_prompt,
        ),
    }
