"""Structural control of instruction specificity, the primary distribution axis.

Specificity is *which decision points a brief states*, not an adjective handed to the
author model. An author merely told "loose" decides for itself what to leave open, which
is how a declared variation goes unrealized. This module owns the schema: the decision
points of each task family, the level ladder that decides which are stated, and the
checks that a generated brief declared what it was asked to.

A withheld point is not an absence; it is a target. Ask-required points must be
clarified, and default-safe points must be resolved with a stated reversible default.
"""

ALWAYS_STATED = "always_stated"
ASK_REQUIRED = "ask_required"
DEFAULT_SAFE = "default_safe"

# Decision points and the behavior expected when one is withheld.
DECISION_POINTS = {
    "deliverable": ALWAYS_STATED,
    "goal": ALWAYS_STATED,
    "setting": DEFAULT_SAFE,
    "style": DEFAULT_SAFE,
    "length": DEFAULT_SAFE,
    "navigation": DEFAULT_SAFE,
    "selection": DEFAULT_SAFE,
    "option_count": DEFAULT_SAFE,
    "continuity": DEFAULT_SAFE,
    "canon_authorization": DEFAULT_SAFE,
    "branch_choice": ASK_REQUIRED,
}

LEVELS = ("L0", "L1", "L2", "L3")

# Points introduced at each level, cumulative from L0. The lowest level always states
# the deliverable and goal, because a request with no target is not a task.
FAMILY_LADDER = {
    "F1": {
        "L0": ("deliverable", "goal"),
        "L1": ("setting",),
        "L2": ("length", "style"),
        "L3": ("continuity", "branch_choice", "canon_authorization"),
    },
    "F2": {
        "L0": ("deliverable", "goal"),
        "L1": ("setting",),
        "L2": ("length", "style"),
        "L3": ("continuity", "branch_choice", "canon_authorization"),
    },
    "F3": {
        "L0": ("deliverable", "goal"),
        "L1": ("setting",),
        "L2": ("option_count",),
        "L3": ("continuity", "canon_authorization"),
    },
    "F4": {
        "L0": ("deliverable", "goal"),
        "L1": ("navigation",),
        "L2": ("selection",),
        "L3": ("continuity",),
    },
    "F5": {
        "L0": ("deliverable", "goal"),
        "L1": ("setting",),
        "L2": ("length", "style"),
        "L3": ("continuity", "branch_choice", "canon_authorization"),
    },
}


def _ladder(family: str) -> dict:
    if family not in FAMILY_LADDER:
        raise ValueError(f"Unknown task family: {family}")
    return FAMILY_LADDER[family]


def decision_points(family: str) -> tuple[str, ...]:
    """All decision points of a family, in the order the ladder introduces them."""
    ladder = _ladder(family)
    points: list[str] = []
    for level in LEVELS:
        for name in ladder[level]:
            if name not in DECISION_POINTS:
                raise ValueError(f"Unknown decision point in ladder: {name}")
            if name not in points:
                points.append(name)
    return tuple(points)


def split_spec(family: str, level: str, *, applicable=None) -> dict:
    """Split a family's decision points into what the brief states and what it withholds.

    ``applicable`` optionally prunes points irrelevant to a specific task, such as a
    branch choice on a close continuation. Pruned points appear in neither list.
    """
    ladder = _ladder(family)
    if level not in LEVELS:
        raise ValueError(f"Unknown specificity level: {level}")
    points = decision_points(family)
    if applicable is not None:
        unknown = set(applicable) - set(points)
        if unknown:
            raise ValueError(f"Points not in family {family}: {sorted(unknown)}")
    index = LEVELS.index(level)
    stated = {name for seen in LEVELS[: index + 1] for name in ladder[seen]}
    if applicable is not None:
        stated &= set(applicable)
    return {
        "level": level,
        "stated": [name for name in points if name in stated],
        "withheld": [
            name
            for name in points
            if name not in stated and (applicable is None or name in applicable)
        ],
    }


def withheld_behaviors(family: str, level: str, *, applicable=None) -> dict:
    """Classify the withheld points by the behavior they require of the writer."""
    classed = {"ask_required": [], "default_safe": []}
    for name in split_spec(family, level, applicable=applicable)["withheld"]:
        kind = DECISION_POINTS[name]
        if kind == ASK_REQUIRED:
            classed["ask_required"].append(name)
        elif kind == DEFAULT_SAFE:
            classed["default_safe"].append(name)
    return classed


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
