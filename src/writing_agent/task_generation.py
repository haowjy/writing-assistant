"""Prepare reproducible, source-backed task-authoring requests without inference."""

from collections import Counter
from copy import deepcopy

from writing_agent.catalog import fingerprint, validate_catalog

FAMILIES = ("F1", "F2", "F3", "F4", "F5")
GENRES = (
    "pastoral fantasy",
    "gothic mystery",
    "workplace science fiction",
    "literary realism",
    "adventure",
    "romantic comedy",
    "LitRPG",
    "historical fiction",
)
STYLES = (
    "spare and concrete",
    "expansive and sensory",
    "dialogue-led",
    "reflective and interior",
    "dryly comic",
)
TRANSFORMATIONS = ("close_continuation", "genre_adaptation", "major_event_divergence")


def prepare_task_requests(
    catalog: list[dict],
    source_ids: list[str],
    *,
    excluded_source_groups: set[str],
    count: int = 100,
) -> dict:
    """Create coverage assignments, not finished tasks or successful demonstrations.

    Only selected human training sources enter the requests. Exclusions apply to
    all identities in their connected lineage groups, including unselected parents.
    """
    if type(count) is not int or count < 1:
        raise ValueError("Task count must be a positive integer")
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("Select unique source IDs")
    groups = validate_catalog(catalog)
    indexed = {source["id"]: source for source in catalog}
    blocked = set()
    for source in catalog:
        identities = {source["id"], source["work_id"], groups[source["id"]]}
        identities.update(source.get(key) for key in ("author_id", "series_id"))
        if identities & excluded_source_groups:
            blocked.add(groups[source["id"]])
    packets = []
    for source_id in source_ids:
        if source_id not in indexed:
            raise ValueError(f"Unknown source: {source_id}")
        source = indexed[source_id]
        if source["role"] != "train" or groups[source_id] in blocked:
            raise ValueError(f"Held-out source or lineage: {source_id}")
        if source["provenance"] != "human":
            raise ValueError("This batch selects human source passages only")
        if not source.get("text", "").strip() or fingerprint(source["text"]) != source["sha256"]:
            raise ValueError(f"Missing or changed source text: {source_id}")
        packets.append(
            {
                "source": deepcopy(source),
                "source_group": groups[source_id],
                "scope": "Only this passage is binding; do not assume omitted book events.",
                "cutoff": len(source["text"]),
                "cutoff_basis": "characters in supplied source text",
            }
        )
    requests = []
    for index in range(count):
        family_index = index % len(FAMILIES)
        source_index = (index // len(FAMILIES)) % len(packets)
        variant = index // (len(FAMILIES) * len(packets))
        family = FAMILIES[family_index]
        stages = [family]
        if variant % 4 >= 2:
            stages.append("F2" if family != "F2" else "F4")
        if variant % 4 == 3:
            stages.append("F5" if "F4" in stages else "F4")
        transformation = TRANSFORMATIONS[(source_index + family_index + variant) % 3]
        assignment = {
            "family": family,
            "stage_families": stages,
            "instruction_specificity": "loose" if (index + variant) % 2 else "explicit",
            "transformation": transformation,
            "genre": (
                "preserve source genre"
                if transformation == "close_continuation"
                else GENRES[(index + variant * 3) % len(GENRES)]
            ),
            "style": STYLES[(source_index + family_index + variant) % len(STYLES)],
            "delivery": "reply" if family in {"F1", "F3"} else "files",
            "kb_format": "linked_markdown" if index % 2 else "flat_markdown",
        }
        body = {"schema_version": 1, "packet": packets[source_index], "assignment": assignment}
        requests.append(
            {
                "id": f"training-task-{index + 1:03d}",
                **body,
                "request_hash": fingerprint(body),
                "status": "awaiting_generation",
            }
        )
    coverage = {
        key: dict(sorted(Counter(r["assignment"][key] for r in requests).items()))
        for key in ("family", "instruction_specificity", "transformation", "genre", "style")
    }
    coverage["stages"] = dict(Counter(len(r["assignment"]["stage_families"]) for r in requests))
    coverage["source_groups"] = dict(Counter(r["packet"]["source_group"] for r in requests))
    coverage["source_works"] = dict(Counter(r["packet"]["source"]["work_id"] for r in requests))
    return {
        "schema_version": 1,
        "status": "prepared_requests_only",
        "requests": requests,
        "coverage": coverage,
        "catalog_hash": fingerprint(catalog),
        "excluded_source_groups": sorted(excluded_source_groups),
        "generated_tasks": 0,
        "accepted_tasks": 0,
    }
