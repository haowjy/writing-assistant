"""Structural control of instruction specificity, the primary distribution axis.

Specificity is *which decision points a brief states*, not an adjective handed to the
author model. An author merely told "loose" decides for itself what to leave open, which
is how a declared variation goes unrealized. This module owns the schema: the decision
points of each task family, the level at which each becomes stated, and the checks that
a generated brief declared what it was asked to.

A withheld point is not an absence; it is a target. Ask-required points must be
clarified, and default-safe points must be resolved with a stated reversible default.
"""

from typing import NamedTuple

ALWAYS_STATED = "always_stated"
ASK_REQUIRED = "ask_required"
DEFAULT_SAFE = "default_safe"
BEHAVIORS = (ALWAYS_STATED, ASK_REQUIRED, DEFAULT_SAFE)

LEVELS = ("L0", "L1", "L2", "L3")


class Point(NamedTuple):
    """A decision a brief can state, and what a withheld one requires."""

    behavior: str
    level: str


# A decision point becomes stated at exactly one level, for the whole schema, so a
# per-family ladder is derived instead of restated. An always-stated point is stated at
# the lowest level and can never be withheld.
DECISION_POINTS = {
    "deliverable": Point(ALWAYS_STATED, "L0"),
    "goal": Point(ALWAYS_STATED, "L0"),
    "setting": Point(DEFAULT_SAFE, "L1"),
    "navigation": Point(DEFAULT_SAFE, "L1"),
    "length": Point(DEFAULT_SAFE, "L2"),
    "style": Point(DEFAULT_SAFE, "L2"),
    "selection": Point(DEFAULT_SAFE, "L2"),
    "option_count": Point(DEFAULT_SAFE, "L2"),
    "continuity": Point(DEFAULT_SAFE, "L3"),
    "canon_authorization": Point(DEFAULT_SAFE, "L3"),
    "branch_choice": Point(ASK_REQUIRED, "L3"),
}

# The points each family can have, in the order a split reports them. Direct prose and
# its KB-grounded variant share a point set; planning and KB building do not.
_PROSE = (
    "deliverable",
    "goal",
    "setting",
    "length",
    "style",
    "continuity",
    "branch_choice",
    "canon_authorization",
)
FAMILY_POINTS = {
    "F1": _PROSE,
    "F2": _PROSE,
    "F3": ("deliverable", "goal", "setting", "option_count", "continuity", "canon_authorization"),
    "F4": ("deliverable", "goal", "navigation", "selection", "continuity"),
    "F5": _PROSE,
}


def _validate_schema() -> None:
    """Reject an inconsistent point table at import rather than at request time."""
    if not DECISION_POINTS:
        raise ValueError("The decision point table is empty")
    for name, point in DECISION_POINTS.items():
        if point.behavior not in BEHAVIORS:
            raise ValueError(f"Decision point {name} has an unknown behavior: {point.behavior}")
        if point.level not in LEVELS:
            raise ValueError(f"Decision point {name} has an unknown level: {point.level}")
        if point.behavior == ALWAYS_STATED and point.level != LEVELS[0]:
            raise ValueError(f"Always-stated point {name} must be stated at {LEVELS[0]}")
    for family, points in FAMILY_POINTS.items():
        if len(set(points)) != len(points):
            raise ValueError(f"Family {family} repeats a decision point")
        unknown = set(points) - set(DECISION_POINTS)
        if unknown:
            raise ValueError(f"Family {family} names unknown decision points: {sorted(unknown)}")


_validate_schema()


def decision_points(family: str) -> tuple[str, ...]:
    """All decision points of a family, in the order a split reports them."""
    if family not in FAMILY_POINTS:
        raise ValueError(f"Unknown task family: {family}")
    return FAMILY_POINTS[family]


def split_spec(family: str, level: str, *, applicable=None) -> dict:
    """Split a family's decision points into what a brief states and what it withholds.

    ``applicable`` optionally prunes points irrelevant to a specific task, such as a
    branch choice on a close continuation. Pruned points appear in neither list.
    """
    points = decision_points(family)
    if level not in LEVELS:
        raise ValueError(f"Unknown specificity level: {level}")
    if applicable is not None:
        unknown = set(applicable) - set(points)
        if unknown:
            raise ValueError(f"Points not in family {family}: {sorted(unknown)}")
        points = tuple(point for point in points if point in set(applicable))
    cutoff = LEVELS.index(level)
    stated = [point for point in points if LEVELS.index(DECISION_POINTS[point].level) <= cutoff]
    withheld = [point for point in points if point not in set(stated)]
    return {"level": level, "stated": stated, "withheld": withheld}


def withheld_behaviors(family: str, level: str, *, applicable=None) -> dict:
    """Classify the withheld points by the behavior they require of the writer."""
    classed = {ASK_REQUIRED: [], DEFAULT_SAFE: []}
    for point in split_spec(family, level, applicable=applicable)["withheld"]:
        behavior = DECISION_POINTS[point].behavior
        if behavior == ALWAYS_STATED:
            raise ValueError(f"Withheld decision point is always stated: {point}")
        classed[behavior].append(point)
    return {"ask_required": classed[ASK_REQUIRED], "default_safe": classed[DEFAULT_SAFE]}


def check_spec(spec: dict, family: str, *, applicable=None) -> list[str]:
    """Return problems if a full spec omits, adds, or leaves empty a decision point."""
    expected = set(decision_points(family))
    if applicable is not None:
        expected &= set(applicable)
    problems = []
    for name in sorted(expected - set(spec)):
        problems.append(f"spec omits decision point: {name}")
    for name in sorted(set(spec) - expected):
        problems.append(f"spec adds unknown decision point: {name}")
    for name in sorted(expected & set(spec)):
        if not spec[name]:
            problems.append(f"spec leaves decision point empty: {name}")
    return problems


def check_declared(intended_stated, declared_stated) -> list[str]:
    """Return problems if a brief declared a different specificity than was intended."""
    intended = set(intended_stated)
    declared = set(declared_stated)
    problems = []
    if declared - intended:
        problems.append(f"brief declared unrequested points: {sorted(declared - intended)}")
    if intended - declared:
        problems.append(f"brief omitted intended points: {sorted(intended - declared)}")
    return problems
