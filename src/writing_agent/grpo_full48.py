"""Hash-bound mechanics for intact wave1 train tasks; no models or semantic judge.

Only this separate recipe adds delivery/action gates. Originals and shared scorecards
remain unchanged. See the bound contract for exact source identity and ambiguities.
"""

import json
from pathlib import Path

from writing_agent.artifacts import extract_prose, markdown_graph
from writing_agent.catalog import fingerprint
from writing_agent.grpo_identity import admission_identity
from writing_agent.reward import Reward
from writing_agent.scoring import mechanical_score
from writing_agent.suite import load_scenarios

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "data/grpo-full48-v1/contract.json"
CONTRACT_HASH = "696741fb6e8041a1ebfa99b3b29b3691c83560b45a816e0e06d2cffe044f8c98"
TASK_IDS = tuple(f"wave1-train-{i:03d}" for i in range(1, 49))
LIMITATIONS = [
    "Mechanical-only; all original semantic rubrics remain unjudged.",
    "Turn evidence proves sequence completion, not faithful clarification or accepted decisions.",
    "Intermediate Done replies are not judged semantically; no intermediate length gate.",
    "Literal checks do not judge prose quality, factual implications, or alternative diversity.",
    "Successful reads establish exposure, not comprehension or causal use.",
    "F2 protected text survives verbatim; final position is ambiguous and not enforced.",
    "All originals remain generated; fixtures are constructions, not model successes.",
    "Exclusions are the frozen local audit inventory, not a global contamination guarantee.",
]


def _binding():
    binding = json.loads(CONTRACT_PATH.read_text())
    if fingerprint(binding) != CONTRACT_HASH or tuple(binding["tasks"]) != TASK_IDS:
        raise ValueError("Full48 bound contract hash/selection mismatch")
    return binding


def reward_spec():
    """Pass alongside mechanical_full48_reward to the existing trainer interface.

    Bind transitive scoring implementation too: trainer callback inspection alone
    does not fingerprint helper functions or this recipe's external JSON.
    """
    root = Path(__file__).parent
    return {
        "id": "full48-mechanical-v1",
        "mode": "mechanical-only-smoke",
        "config": {
            "contract_hash": CONTRACT_HASH,
            "implementation_hashes": {
                name: fingerprint((root / name).read_bytes())
                for name in ("grpo_full48.py", "scoring.py", "artifacts.py", "reward.py")
            },
            "aggregation": "zero-unless-executed-sequence-and-all-final-delivery-then-check-mean",
            "delivery_minimum": "original declared lower word bounds; upper bounds remain checks",
            "semantic_status": "unjudged",
            "limitations": LIMITATIONS,
        },
    }


def load_full48_release(directory: Path) -> dict:
    """Verify the original release in place, returning unmodified tasks and admission."""
    directory = Path(directory)
    binding = _binding()
    actual = {
        str(p.relative_to(directory)): fingerprint(p.read_bytes())
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    if actual != binding["release_files"]:
        raise ValueError("Original full48 release bytes differ from binding")
    tasks = load_scenarios(directory)
    if tuple(t["id"] for t in tasks) != TASK_IDS:
        raise ValueError("Full48 requires exactly the original 48 ordered train tasks")
    for task in tasks:
        _contract(task, binding)
    catalog = json.loads((directory / "catalog.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        fingerprint(catalog) != binding["catalog_hash"]
        or fingerprint(manifest) != binding["manifest_hash"]
    ):
        raise ValueError("Original catalog/manifest mismatch")
    if any(c["role"] != "train" or c["provenance"] != "synthetic" for c in catalog):
        raise ValueError("Full48 catalog role/provenance mismatch")
    admission = {
        "mode": "production",
        "catalog": catalog,
        "catalog_hash": binding["catalog_hash"],
        "release_manifest": manifest,
        "excluded_source_groups": binding["excluded_source_identities"],
    }
    admission_identity(tasks, admission)
    return {
        "tasks": tasks,
        "admission": admission,
        "reward_spec": reward_spec(),
        "contract_hash": CONTRACT_HASH,
    }


def _contract(task, binding):
    contract = binding["tasks"].get(task.get("id"))
    if (
        not contract
        or fingerprint(task) != contract["task_hash"]
        or task["role"] != "train"
        or task["provenance"] != "synthetic"
    ):
        raise ValueError("Task differs from intact bound full48 original")
    return contract


def _sequence(task, result):
    """Cross-check saved turn events and messages; never grade intermediate prose."""
    turns = result.get("turns", [])
    messages = result.get("messages", [])
    events = result.get("trace", [])
    expected = [task["visible"]["brief"], *task["visible"]["followups"]]
    if (
        len(turns) != len(expected)
        or [m.get("content") for m in messages if m.get("role") == "user"] != expected
        or [e["turn"] for e in events if e.get("type") == "turn"] != turns
        or [e["content"] for e in events if e.get("type") == "followup"] != expected[1:]
    ):
        return False
    boundaries = [e["type"] for e in events if e.get("type") in {"turn", "followup"}]
    if boundaries != ["turn", "followup"] * (len(expected) - 1) + ["turn"]:
        return False
    indices = [t.get("message_index", -1) for t in turns]
    if indices != sorted(set(indices)) or not indices or indices[0] < 0:
        return False
    # Each user message must precede exactly its own final assistant message.
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    for n, (index, turn) in enumerate(zip(indices, turns, strict=True)):
        if index >= len(messages) or not user_indices[n] < index:
            return False
        if n + 1 < len(turns) and not index < user_indices[n + 1]:
            return False
        message = messages[index]
        if (
            message.get("role") != "assistant"
            or message.get("tool_calls")
            or message.get("content") != turn.get("output")
        ):
            return False
    return turns[-1].get("output") == result.get("output") and turns[-1].get(
        "snapshot"
    ) == result.get("after")


def _actions(result):
    actions = []
    for event in result.get("trace", []):
        if event.get("type") != "tool" or not event.get("observation", {}).get("ok"):
            continue
        function = event["call"]["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        actions.append((function["name"], arguments, event["observation"].get("result")))
    return actions


def _written_files(initial, actions):
    """Replay successful write/patch events, not file existence or verbal claims."""
    state = dict(initial)
    written = set()
    for name, args, _ in actions:
        if name == "write_file":
            state[args["path"]] = args["content"]
            written.add(args["path"])
        elif name == "patch_file":
            old = state.get(args["path"], "")
            if not args["old"] or old.count(args["old"]) != 1:
                raise ValueError("Successful patch trace cannot be replayed")
            state[args["path"]] = old.replace(args["old"], args["new"], 1)
            written.add(args["path"])
    return state, written


def mechanical_full48_reward(task: dict, result: dict) -> Reward:
    """Plain closure-free callback. Identity errors raise; malformed evidence scores zero."""
    contract = _contract(task, _binding())
    initial = task["visible"]["initial_files"]
    result = {**result, "before": initial}
    try:
        sequence = _sequence(task, result)
        actions = _actions(result)
        replayed, written = _written_files(initial, actions)
    except (KeyError, TypeError, ValueError):
        return Reward(
            status="ok",
            value=0.0,
            reason="Invalid saved action/turn evidence",
            components={"semantic_status": "unjudged", "limitations": LIMITATIONS},
        )
    artifacts = {a["id"]: a for a in extract_prose(result, task["visible"]["prose"])}
    after = result.get("after", {})
    delivery = []
    for requirement in contract["delivery"]:
        artifact = artifacts[requirement["id"]]
        passed = (
            artifact["status"] == "ok"
            and len(artifact["text"].split()) >= requirement["minimum_words"]
        )
        if requirement["kind"] == "file":
            path = requirement["path"]
            passed = (
                passed
                and path in written
                and replayed.get(path) == after.get(path)
                and after.get(path, "").strip() != initial.get(path, "").strip()
            )
        delivery.append({"id": requirement["id"], "passed": bool(passed)})
    card = mechanical_score(task, result)
    checks = [
        {"id": c["id"], "passed": c["status"] == "ok" and c["passed"] is True}
        for c in card["checks"]
        if c["method"] in {"deterministic", "reference_match"}
    ]
    for path in contract["reads"]:
        checks.append(
            {
                "id": "read-" + path,
                "passed": any(
                    name == "read_file" and args.get("path") == path and text == initial[path]
                    for name, args, text in actions
                ),
            }
        )
    if contract["wiki_edges"]:
        graph = markdown_graph(
            {p: t for p, t in after.items() if p.startswith("kb/")}, task["labels"]["entrypoints"]
        )
        edges = {(e["from"], e["to"]) for e in graph["links"] if e["valid"]}
        checks.append(
            {
                "id": "reciprocal-links",
                "passed": all(tuple(edge) in edges for edge in contract["wiki_edges"]),
            }
        )
    retrieval = contract.get("retrieval")
    if retrieval:
        path = retrieval["path"]
        found = False
        retrieved = False
        for name, args, text in actions:
            if name == "search" and isinstance(text, list):
                found |= any(match.get("path") == path for match in text)
            if found and name == "read_file" and args.get("path") == path and text == initial[path]:
                retrieved = True
        checks += [
            {"id": "search-then-read-active-context", "passed": retrieved},
            {
                "id": "verbatim-quote-once",
                "passed": result.get("output", "").count(retrieval["quote"]) == 1,
            },
        ]
    mechanics = sum(c["passed"] for c in checks) / len(checks)
    eligible = (
        result.get("status") == "completed" and sequence and all(d["passed"] for d in delivery)
    )
    return Reward(
        status="ok",
        value=mechanics if eligible else 0.0,
        components={
            "execution_completed": result.get("status") == "completed",
            "sequence_completed": sequence,
            "required_delivery": delivery,
            "mechanical_completion": eligible and all(c["passed"] for c in checks),
            "mechanics": mechanics,
            "checks": checks,
            "semantic_status": "unjudged",
            "unjudged_rubrics": sorted(task["labels"].get("rubrics", {})),
            "limitations": LIMITATIONS,
        },
        reason=None if eligible else "Failed execution, sequence evidence, or final delivery",
    )
