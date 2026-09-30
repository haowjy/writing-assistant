"""Frozen engineering-only probe inputs and mechanical reward; never load models."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

from writing_agent.agent import run_agent
from writing_agent.artifacts import extract_prose, markdown_graph
from writing_agent.backends import ScriptedBackend
from writing_agent.catalog import fingerprint, save_json, validate_catalog
from writing_agent.grpo_identity import admission_identity
from writing_agent.reward import Reward
from writing_agent.scoring import mechanical_score
from writing_agent.suite import compile_scenarios, load_scenarios
from writing_agent.workspace import Workspace

DEFAULT_PACKET = Path(__file__).resolve().parents[2] / "data/grpo-probe-v1/packet.json"
LIMITATIONS = [
    "Mechanical engineering compliance only; no literary-quality or semantic-continuity judgment.",
    "Literal exclusions catch named strings, not paraphrases, negation, or false implications.",
    "Tool evidence establishes exposure, not comprehension or causal use.",
    "Three training tasks share one author-connected source group; development uses two worlds.",
    "Catalog provenance is internally verified, not an independent authenticity audit.",
]


def _packet(path):
    packet = json.loads(Path(path).read_text())
    if packet["version"] != 1 or packet["name"] != "grpo-probe-v1":
        raise ValueError("Unknown probe packet version")
    validate_catalog(packet["catalog"])
    records = {r["id"]: r for r in packet["catalog"]}
    for origin in packet["origins"]:
        for key, digest in origin["selected_catalog_record_hashes"].items():
            if fingerprint(records[key]) != digest:
                raise ValueError("Frozen parent source record hash mismatch")
    for item in packet["scenarios"]:
        original = item["original"]
        if fingerprint(original) != item["original_hash"] or any(
            fingerprint(original[field]) != original[field + "_hash"]
            for field in ("visible", "labels")
        ):
            raise ValueError("Frozen parent scenario hash mismatch")
    return packet


def _scenarios(packet):
    scenarios = []
    for item in packet["scenarios"]:
        task = copy.deepcopy(item["original"])
        for key in ("source_groups", "visible_hash", "labels_hash", "schema_version"):
            task.pop(key, None)
        task["id"] = "grpo-probe-v1-" + task["id"]
        task["probe_origin"] = {k: copy.deepcopy(v) for k, v in item.items() if k != "original"}
        task["probe_origin"]["scenario_id"] = item["original"]["id"]
        task["review_status"] = "engineering-fixtures-validated-not-literary-reviewed"
        task["visible"].update({k: item["overrides"][k] for k in ("brief", "budgets")})
        task["labels"]["checks"] = copy.deepcopy(item["overrides"]["checks"])
        task["labels"]["probe_contract"] = copy.deepcopy(item["contract"])
        scenarios.append(task)
    return scenarios


def _empty_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output must be an absent or empty directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_probe_release(destination: Path, *, packet_path: Path = DEFAULT_PACKET) -> dict:
    """Rebuild only from versioned source packets; no ignored data dependencies."""
    packet = _packet(packet_path)
    scenarios = _scenarios(packet)
    destination = _empty_directory(destination)
    manifest = compile_scenarios(scenarios, packet["catalog"], destination)
    freeze = {
        "version": 1,
        "packet_hash": fingerprint(packet),
        "packet_bytes_sha256": fingerprint(Path(packet_path).read_bytes()),
        "goldens_hash": fingerprint(
            json.loads(Path(packet_path).with_name("goldens.json").read_text())
        ),
        "manifest_hash": fingerprint(manifest),
        "catalog_hash": fingerprint(packet["catalog"]),
        "origins": packet["origins"],
        "limitations": LIMITATIONS,
    }
    save_json(destination / "freeze.json", freeze)
    return load_probe_release(destination)


def load_probe_release(directory: Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    catalog = json.loads((directory / "catalog.json").read_text())
    freeze = json.loads((directory / "freeze.json").read_text())
    if (
        fingerprint(manifest) != freeze["manifest_hash"]
        or fingerprint(catalog) != freeze["catalog_hash"]
    ):
        raise ValueError("Frozen release hash mismatch")
    tasks = load_scenarios(directory)
    train = [t for t in tasks if t["role"] == "train"]
    development = [t for t in tasks if t["role"] == "development"]
    if len(train) != 3 or len(development) != 6 or len(tasks) != 9:
        raise ValueError("Probe must contain three train and six development tasks")
    groups = validate_catalog(catalog)
    admission = {
        "mode": "production",
        "catalog": catalog,
        "catalog_hash": fingerprint(catalog),
        "release_manifest": manifest,
        "excluded_source_groups": sorted(
            {groups[r["id"]] for r in catalog if r["role"] != "train"}
        ),
    }
    admission_identity(train, admission)
    return {
        "tasks": tasks,
        "train": train,
        "development": development,
        "manifest": manifest,
        "admission": admission,
        "freeze": freeze,
        "directory": str(directory),
    }


def inspect_probe_release(directory: Path) -> dict:
    release = load_probe_release(directory)
    identity = admission_identity(release["train"], release["admission"])
    return {
        "freeze": release["freeze"],
        "train_ids": [t["id"] for t in release["train"]],
        "development_ids": [t["id"] for t in release["development"]],
        "admission_mode": identity["mode"],
        "connected_groups": identity["connected_groups"],
        "excluded_source_groups": identity["excluded_source_groups"],
        "overlap_audit": identity["overlap_audit"],
        "tasks": [
            {
                "id": t["id"],
                "visible_hash": t["visible_hash"],
                "labels_hash": t["labels_hash"],
                "budgets": t["visible"]["budgets"],
                "initial_bytes": sum(
                    len(s.encode()) for s in t["visible"]["initial_files"].values()
                ),
                "modifications": t["probe_origin"]["modifications"],
            }
            for t in release["tasks"]
        ],
    }


def mechanical_probe_reward(task: dict, result: dict) -> Reward:
    """Completion-gated mean of declared mechanics; semantic checks stay pending."""
    # The supplied task, not a candidate's before snapshot, defines unchanged content.
    result = {**result, "before": task["visible"]["initial_files"]}
    contract = task["labels"]["probe_contract"]
    artifacts = extract_prose(result, task["visible"]["prose"])
    after = result.get("after", {})
    files = contract["required_files"]
    delivery = all(
        a["status"] == "ok" and len(a["text"].split()) >= contract["minimum_delivery_words"]
        for a in artifacts
    ) and all(
        len(after.get(p, "").split()) >= contract["minimum_delivery_words"]
        and after[p].strip() != result["before"].get(p, "").strip()
        for p in files
    )
    if contract.get("wiki_topic_required"):
        pages = {p: text for p, text in after.items() if p.startswith("kb/") and p.endswith(".md")}
        delivery = (
            delivery
            and len(pages) >= 2
            and all(
                len(text.split()) >= contract["minimum_delivery_words"] for text in pages.values()
            )
        )
    execution = result.get("status") == "completed" and bool(result.get("output", "").strip())
    card = mechanical_score(task, result)
    checks = [
        {"id": c["id"], "passed": c["status"] == "ok" and c["passed"] is True}
        for c in card["checks"]
        if c["method"] in {"deterministic", "reference_match"}
    ]
    for path, suffix in contract.get("protected_suffixes", {}).items():
        checks.append(
            {"id": "ending-" + path, "passed": after.get(path, "").rstrip().endswith(suffix)}
        )
    events = [
        e
        for e in result.get("trace", [])
        if e.get("type") == "tool" and e.get("observation", {}).get("ok")
    ]
    for requirement in contract.get("tool_requirements", []):
        exposed = False
        for event in events:
            function = event["call"]["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if function["name"] == requirement["name"] and all(
                arguments.get(k) == v for k, v in requirement["arguments"].items()
            ):
                exposed = True
        checks.append({"id": requirement["id"], "passed": exposed})
    if contract.get("exact_quote"):
        checks.append(
            {
                "id": "quote-once",
                "passed": result.get("output", "").count(contract["exact_quote"]) == 1,
            }
        )
    if contract.get("wiki_edges"):
        graph = markdown_graph(
            {p: t for p, t in after.items() if p.startswith("kb/")}, ["kb/index.md"]
        )
        edges = {(e["from"], e["to"]) for e in graph["links"] if e["valid"]}
        checks.append(
            {
                "id": "wiki-required-edges",
                "passed": all(tuple(e) in edges for e in contract["wiki_edges"]),
            }
        )
    mechanics = sum(c["passed"] for c in checks) / len(checks)
    complete = bool(execution and delivery and all(c["passed"] for c in checks))
    return Reward(
        status="ok",
        value=mechanics if execution and delivery else 0.0,
        components={
            "execution_completed": execution,
            "required_delivery": bool(delivery),
            "mechanical_completion": complete,
            "mechanics": mechanics,
            "checks": checks,
            "semantic_status": "unjudged",
            "limitations": LIMITATIONS,
        },
        reason=None if complete else "Failed execution, delivery, or declared mechanical check",
    )


def golden_messages(task: dict) -> list[dict]:
    fixtures = json.loads(DEFAULT_PACKET.with_name("goldens.json").read_text())["fixtures"]
    fixture = fixtures[task["probe_origin"]["scenario_id"]]
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"golden-{i}", "type": "function", "function": action}],
        }
        for i, action in enumerate(fixture["actions"])
    ] + [{"role": "assistant", "content": fixture["output"]}]


def validate_probe_fixtures(directory: Path, destination: Path) -> dict:
    """Real tool-loop goldens plus counterfactual invalid artifacts, never model calls."""
    release = load_probe_release(directory)
    destination = _empty_directory(destination)
    fixture_hash = fingerprint(json.loads(DEFAULT_PACKET.with_name("goldens.json").read_text()))
    if fixture_hash != release["freeze"]["goldens_hash"]:
        raise ValueError("Golden fixtures differ from frozen release")
    evidence = []
    for task in release["tasks"]:
        root = destination / task["id"]
        workspace = Workspace(
            root / "workspace", max_total_bytes=task["visible"]["budgets"]["max_total_bytes"]
        )
        for path, text in task["visible"]["initial_files"].items():
            workspace.write_file(path, text)
        trace = []
        result = run_agent(
            ScriptedBackend(golden_messages(task)),
            workspace,
            [{"role": "user", "content": task["visible"]["brief"]}],
            tools=task["visible"]["tools"],
            emit=trace.append,
            **{k: v for k, v in task["visible"]["budgets"].items() if k != "max_total_bytes"},
        )
        result.update(
            before=task["visible"]["initial_files"], after=workspace.snapshot(), trace=trace
        )
        candidates = {"golden": result}
        for name in ("untouched", "empty", "failed", "scope", "missing", "partial", "broken"):
            bad = copy.deepcopy(result)
            if name == "untouched":
                bad.update(after=copy.deepcopy(result["before"]), output="Done.", trace=[])
            elif name == "empty":
                bad.update(after={}, output="", trace=[])
            elif name == "failed":
                bad["status"] = "step_limit"
            elif name == "scope":
                bad["after"]["unrequested.md"] = "Unauthorized change."
            else:
                paths = task["labels"]["probe_contract"]["required_files"]
                if paths:
                    path = paths[0]
                    if name == "missing":
                        bad["after"].pop(path, None)
                    elif name == "partial":
                        bad["after"][path] = ""
                    else:
                        bad["after"][path] = "Broken delivery."
                else:
                    bad["output"] = "" if name in {"missing", "partial"} else "Broken delivery."
            candidates[name] = bad
        if task["family"] == "F4":
            bad = copy.deepcopy(result)
            bad["after"]["kb/index.md"] += "\n[Missing](absent.md)\n"
            candidates["bad-link"] = bad
            bad = copy.deepcopy(result)
            topic = next(p for p in bad["after"] if p.startswith("kb/") and p != "kb/index.md")
            bad["after"][topic] = "[Index](index.md)"
            candidates["partial-topic"] = bad
        for path, suffix in task["labels"]["probe_contract"].get("protected_suffixes", {}).items():
            bad = copy.deepcopy(result)
            bad["after"][path] = bad["after"][path].replace(
                suffix, "The protected sentence changed."
            )
            candidates["protected-text"] = bad
        if task["family"] == "F5":
            bad = copy.deepcopy(result)
            bad["trace"] = []
            candidates["no-tool-evidence"] = bad
        for check in task["labels"]["checks"]:
            if check["kind"] != "excludes":
                continue
            bad = copy.deepcopy(result)
            selector = next(s for s in task["visible"]["prose"] if s["id"] == check["artifact"])
            if selector["kind"] == "file":
                path = selector["path"]
                bad["after"][path] = check["text"] + "\n" + bad["after"][path]
            else:
                bad["output"] = check["text"] + "\n" + bad["output"]
            candidates["literal-forbidden-" + check["id"]] = bad
        if task["probe_origin"]["scenario_id"] == "wave1-train-043":
            bad = copy.deepcopy(result)
            bad["output"] = bad["output"].replace(
                task["labels"]["probe_contract"]["exact_quote"],
                "The blue bowl and private sketch may appear in the grant portfolio.",
            )
            candidates["withdrawn-instruction"] = bad
        rows = []
        for name, candidate in candidates.items():
            reward = mechanical_probe_reward(task, candidate)
            row = {
                "case": name,
                "result_hash": fingerprint(candidate),
                "reward": asdict(reward),
                "artifact_hashes": {
                    p: fingerprint(t.encode()) for p, t in candidate["after"].items()
                },
            }
            save_json(root / f"{name}.json", {"result": candidate, **row})
            if (name == "golden" and reward.value != 1) or (name != "golden" and reward.value == 1):
                raise ValueError(f"Unexpected fixture reward: {task['id']} {name}: {reward}")
            rows.append(row)
        evidence.append(
            {
                "id": task["id"],
                "cases": rows,
                "read_tokens": result["read_tokens"],
                "tool_calls": result["tool_calls"],
            }
        )
    report = {
        "freeze": release["freeze"],
        "fixtures_hash": fixture_hash,
        "tasks": evidence,
        "limitations": LIMITATIONS,
        "is_model_evaluation": False,
    }
    save_json(destination / "evidence.json", report)
    return report
