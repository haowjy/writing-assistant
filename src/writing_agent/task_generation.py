"""Prepare reproducible, source-backed task-authoring requests without inference.

Requests reference their source passage by identity rather than embedding it, so a
batch costs O(requests) instead of O(requests x source size). Each request is derived
from its index and a per-key seeded permutation, so a batch can be streamed, resumed
mid-way, or sampled one request at a time.

Inputs are validated once when a ``Sampler`` is built; the operations over it are
lazy and addressable.
"""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from random import Random

from writing_agent.catalog import fingerprint, validate_catalog
from writing_agent.specificity import LEVELS, decision_points, split_spec

SCHEMA_VERSION = 3
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
TRANSFORMATIONS = ("close_continuation", "genre_adaptation", "major_event_divergence")
CONTENT_KEYS = ("genres", "styles", "tropes", "situations", "continuity_challenges")
SOURCE_SCOPE = "Only this passage is binding; do not assume omitted book events."
COVERAGE_KEYS = (
    "family",
    "transformation",
    "genre",
    "style",
    "trope",
    "situation",
    "continuity_challenge",
)


def _validate_options(values, key: str) -> None:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"Variation dimension needs unique nonempty strings: {key}")


def _permutation(values, *, seed: int, key: str) -> tuple[str, ...]:
    """A seeded permutation so a value can be addressed by index without shared state."""
    _validate_options(values, key)
    order = list(values)
    Random(f"{seed}:{key}").shuffle(order)
    return tuple(order)


def _value(order: tuple[str, ...], index: int) -> str:
    return order[index % len(order)]


def _source_packets(catalog: list[dict], source_ids: list[str], excluded: set[str]) -> list[dict]:
    """Validate the selection and return source references, not source passages."""
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("Select unique source IDs")
    groups = validate_catalog(catalog)
    indexed = {source["id"]: source for source in catalog}
    blocked = set()
    for source in catalog:
        identities = {source["id"], source["work_id"], groups[source["id"]]}
        identities.update(source.get(key) for key in ("author_id", "series_id"))
        if identities & excluded:
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
                "source_id": source["id"],
                "source_work": source["work_id"],
                "source_sha256": source["sha256"],
                "source_group": groups[source_id],
                "scope": SOURCE_SCOPE,
                "cutoff": len(source["text"]),
                "cutoff_basis": "characters in supplied source text",
            }
        )
    return packets


@dataclass(frozen=True)
class Sampler:
    """Validated request inputs, with content derived once and addressable by index."""

    packets: tuple[dict, ...]
    orders: dict
    variation_catalog_hash: str
    catalog_hash: str
    excluded_source_groups: tuple[str, ...]
    count: int
    seed: int
    levels: tuple[str, ...]

    def __len__(self) -> int:
        return self.count

    @classmethod
    def build(
        cls,
        catalog: list[dict],
        source_ids: list[str],
        *,
        excluded_source_groups: set[str],
        variation_catalog: dict,
        count: int = 100,
        seed: int = 42,
        levels: tuple[str, ...] = ("L3",),
    ) -> "Sampler":
        """Validate a selection once and derive its index-addressable content.

        Requests are emitted as matched ladders: each base assignment repeats once per
        specificity level, with the level cycling fastest so a base's ladder is adjacent.
        ``count`` is the total request count and must be divisible by the level count.
        The default level states every decision point, which is an explicit task.
        """
        if type(count) is not int or count < 1:
            raise ValueError("Task count must be a positive integer")
        levels = tuple(levels)
        if (
            not levels
            or len(set(levels)) != len(levels)
            or any(level not in LEVELS for level in levels)
        ):
            raise ValueError("Specificity levels must be unique members of LEVELS")
        if count % len(levels):
            raise ValueError("Request count must be divisible by the number of specificity levels")
        return cls(
            packets=tuple(_source_packets(catalog, source_ids, excluded_source_groups)),
            orders={
                key: _permutation(variation_catalog.get(key), seed=seed, key=key)
                for key in CONTENT_KEYS
            },
            variation_catalog_hash=fingerprint(variation_catalog),
            catalog_hash=fingerprint(catalog),
            excluded_source_groups=tuple(sorted(excluded_source_groups)),
            count=count,
            seed=seed,
            levels=levels,
        )


def _body(sampler: Sampler, slot: int, level: str) -> dict:
    """Every coverage field for one base at one level, independent of other requests."""
    family_index = slot % len(FAMILIES)
    source_index = (slot // len(FAMILIES)) % len(sampler.packets)
    variant = slot // (len(FAMILIES) * len(sampler.packets))
    family = FAMILIES[family_index]
    stages = [family]
    if variant % 4 >= 2:
        stages.append("F2" if family != "F2" else "F4")
    if variant % 4 == 3:
        stages.append("F5" if "F4" in stages else "F4")
    transformation = TRANSFORMATIONS[(source_index + family_index + variant) % 3]
    continuation = transformation == "close_continuation"
    if continuation:
        genre, blend = "preserve source genre", []
    else:
        genre = _value(sampler.orders["genres"], slot)
        blend = [genre]
        alternatives = [candidate for candidate in sampler.orders["genres"] if candidate != genre]
        if slot % 2 and alternatives:
            blend.append(alternatives[slot % len(alternatives)])
    applicable = tuple(
        point for point in decision_points(family) if point != "branch_choice" or not continuation
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "packet": sampler.packets[source_index],
        "assignment": {
            "family": family,
            "stage_families": stages,
            "transformation": transformation,
            "genre": genre,
            "genre_blend": blend,
            "style": (
                "preserve source style" if continuation else _value(sampler.orders["styles"], slot)
            ),
            "trope": _value(sampler.orders["tropes"], slot),
            "situation": _value(sampler.orders["situations"], slot),
            "continuity_challenge": _value(sampler.orders["continuity_challenges"], slot),
            "creative_options_status": "suggestions_until_grounded_in_visible_task",
            "delivery": "reply" if family in {"F1", "F3"} else "files",
            "kb_format": "linked_markdown" if slot % 2 else "flat_markdown",
            "instruction_specificity": split_spec(family, level, applicable=applicable),
        },
        "variation_catalog_hash": sampler.variation_catalog_hash,
        "seed": sampler.seed,
    }


def iter_requests(sampler: Sampler, *, start: int = 0) -> Iterator[dict]:
    """Yield one request at a time, resumable from ``start``."""
    if type(start) is not int or not 0 <= start <= sampler.count:
        raise ValueError("Start index must be within the batch")
    return _stream(sampler, start)


def _stream(sampler: Sampler, start: int) -> Iterator[dict]:
    for index in range(start, sampler.count):
        body = _body(
            sampler,
            index // len(sampler.levels),
            sampler.levels[index % len(sampler.levels)],
        )
        yield {
            "id": f"training-task-{index + 1:03d}",
            **body,
            "request_hash": fingerprint(body),
            "status": "awaiting_generation",
        }


def build_request(sampler: Sampler, index: int) -> dict:
    """Return one request without materializing the batch. Bounded by the sampler count."""
    if type(index) is not int or not 0 <= index < sampler.count:
        raise ValueError(f"Request index out of range: {index}")
    return next(iter_requests(sampler, start=index))


def coverage(requests: list[dict]) -> dict:
    """Marginal coverage per axis, plus the specificity level and withheld points."""
    summary = {
        key: dict(sorted(Counter(r["assignment"][key] for r in requests).items()))
        for key in COVERAGE_KEYS
    }
    summary["genre_blend_size"] = dict(
        Counter(len(r["assignment"]["genre_blend"]) for r in requests)
    )
    summary["stages"] = dict(Counter(len(r["assignment"]["stage_families"]) for r in requests))
    summary["source_groups"] = dict(Counter(r["packet"]["source_group"] for r in requests))
    summary["source_works"] = dict(Counter(r["packet"]["source_work"] for r in requests))
    summary["instruction_specificity_level"] = dict(
        sorted(
            Counter(r["assignment"]["instruction_specificity"]["level"] for r in requests).items()
        )
    )
    summary["withheld_points"] = dict(
        sorted(
            Counter(
                point
                for r in requests
                for point in r["assignment"]["instruction_specificity"]["withheld"]
            ).items()
        )
    )
    return summary


def prepare_task_requests(sampler: Sampler) -> dict:
    """Materialize a batch of requests and its coverage summary.

    For on-demand work iterate the sampler; this exists for the frozen prepared-batch
    artifact.
    """
    requests = list(iter_requests(sampler))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_requests_only",
        "requests": requests,
        "coverage": coverage(requests),
        "catalog_hash": sampler.catalog_hash,
        "variation_catalog_hash": sampler.variation_catalog_hash,
        "seed": sampler.seed,
        "levels": list(sampler.levels),
        "excluded_source_groups": list(sampler.excluded_source_groups),
        "generated_tasks": 0,
        "accepted_tasks": 0,
    }
