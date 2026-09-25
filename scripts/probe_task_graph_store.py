"""Exercise a deep branched store closure and report measured read complexity.

This is a real-filesystem runtime probe, not a benchmark.  It constructs valid
immutable records, then measures one cold operation-scoped validation traversal.

    uv run python scripts/probe_task_graph_store.py --depth 256
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from writing_agent.task_graph import (
    CheckpointV1,
    ContextRevisionV1,
    EnvironmentStateV1,
    EventV1,
    GraphInstanceV1,
    MessageV1,
    NodeSpecV1,
    tree_hash,
)
from writing_agent.task_graph_store import TaskGraphStore


def run(depth: int) -> tuple[int, float, int]:
    with tempfile.TemporaryDirectory() as temporary:
        store = TaskGraphStore(Path(temporary) / "store")
        refs = {
            name: store.put_artifact({"probe": name})
            for name in (
                "entry",
                "template",
                "tokenizer",
                "tools",
                "requirements",
                "decisions",
                "disclosures",
                "versions",
                "budgets",
                "rng",
                "external",
                "outcome",
                "provenance",
            )
        }
        private = store.put_artifact({"probe": "private"}, private=True)
        instance = GraphInstanceV1(
            template_ref=refs["template"],
            entry_node="write",
            nodes=(NodeSpecV1(id="write", entry_contract=refs["entry"]),),
        )
        store.persist(instance)
        context = ContextRevisionV1(
            messages=(MessageV1(content=("probe",), origin="probe:request"),),
            rendering={
                "projection_version": "v1",
                "prefix_id": "root",
                "template_ref": refs["template"],
                "tokenizer_ref": refs["tokenizer"],
                "tool_schema_ref": refs["tools"],
            },
        )
        store.persist(context)

        previous = None
        for sequence in range(1, depth + 1):
            event = EventV1(
                previous=previous,
                seq=sequence,
                lineage_id="probe",
                kind="tool_result",
                audience=("controller",),
                payload_ref=refs["entry"],
                versions_ref=refs["versions"],
                provenance_ref=refs["provenance"],
            )
            store.persist(event)
            previous = event.identity()

        files = {"draft.txt": "probe"}
        state = EnvironmentStateV1(
            instance_ref=instance.identity(),
            position={
                "node_id": "write",
                "visit_id": "visit-1",
                "phase": "ready_writer",
                "entry_contract": refs["entry"],
                "start_checkpoint": None,
                "loop_counts": {},
                "lineage_id": "probe",
            },
            files=files,
            tree_hash=tree_hash(files),
            history={
                "head": previous,
                "seq": depth,
                "branch_base": None,
                "imported_refs": (),
                "action_ids": (),
                "tool_result_ids": (),
            },
            context_ref=context.identity(),
            requirements_ref=refs["requirements"],
            decisions_ref=refs["decisions"],
            disclosures_ref=refs["disclosures"],
            author_packet_ref=private,
            versions_ref=refs["versions"],
            budgets_ref=refs["budgets"],
            rng_ref=refs["rng"],
            external_inputs_ref=refs["external"],
            outcome_ref=refs["outcome"],
            provenance_ref=refs["provenance"],
            continuation={
                "tool_queue": (),
                "next_call": 0,
                "author_request": None,
                "check_requests": (),
                "external_requests": (),
                "applied_responses": (),
                "feedback_cursor": 0,
            },
            in_flight_effects=(),
        )
        root = CheckpointV1(state=state, event_head=previous)
        store.persist(root)
        ancestry = [root.identity()]
        for _ in range(depth):
            checkpoint = CheckpointV1(parents=(ancestry[-1],), state=state, event_head=previous)
            store.persist(checkpoint)
            ancestry.append(checkpoint.identity())
        left = CheckpointV1(parents=(ancestry[-1],), state=state, event_head=previous)
        right = CheckpointV1(parents=(ancestry[depth // 2],), state=state, event_head=previous)
        store.persist(left)
        store.persist(right)
        joined = CheckpointV1(
            parents=(left.identity(),),
            state=state,
            event_head=previous,
            artifact_refs=(right.identity(),),
        )
        store.persist(joined)

        reads = 0
        read_bytes = store._read_bytes

        def counted(path: Path) -> bytes:
            nonlocal reads
            reads += 1
            return read_bytes(path)

        store._read_bytes = counted  # type: ignore[method-assign]
        started = time.perf_counter()
        store.load_checkpoint(joined.identity())
        elapsed = time.perf_counter() - started
        limit = 2 * depth + 32
        if reads > limit:
            raise RuntimeError(f"nonlinear reads: {reads} > {limit}")
        return reads, elapsed, limit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=256)
    arguments = parser.parse_args()
    if arguments.depth < 1:
        parser.error("--depth must be positive")
    reads, seconds, limit = run(arguments.depth)
    print(f"depth={arguments.depth} reads={reads} limit={limit} validation_seconds={seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
