"""Editable, Python-driven IFEval and HumanEval+ development diagnostics."""

import json
import os
import shutil
import subprocess
import sys

from scripts.pilot_e2b import MODEL, ROOT
from writing_agent.catalog import fingerprint, save_json
from writing_agent.external import generate_tasks

OUTPUT = ROOT / "runs/external-e2b-it-2026-09-14"
RAW = ROOT / "data/raw/research"
CONFIG = {
    **MODEL,
    "temperature": 0,
    "max_tokens": 8192,
    "context_tokens": 16384,
    "purpose": "external instruction-following and coding diagnostics",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare():
    """Keep test answers outside public generation manifests."""
    instructions = read_jsonl(RAW / "ifeval/data/input_data.jsonl")
    coding = read_jsonl(RAW / "humanevalplus/HumanEvalPlus.jsonl")
    coding = [x for x in coding if x["task_id"] in {f"HumanEval/{i}" for i in range(32)}]
    assert len(instructions) == 541 and len(coding) == 32
    collections = {
        "ifeval": [{"id": str(x["key"]), "prompt": x["prompt"]} for x in instructions],
        "humanevalplus": [
            {
                "id": x["task_id"].replace("/", "-"),
                "task_id": x["task_id"],
                "prompt": "Complete the following Python function. Return the complete "
                "implementation in a Python code block.\n\n" + x["prompt"],
            }
            for x in coding
        ],
    }
    manifest = {
        "model": CONFIG,
        "cases": {k: len(v) for k, v in collections.items()},
        "dataset_hashes": {
            "ifeval": fingerprint(instructions),
            "humanevalplus_subset": fingerprint(coding),
        },
        "ifeval_revision": "26d8ccdab6fec61b5c83ad6327ea8bda9e580288",
        "humanevalplus_version": "v0.1.10",
        "evalplus_version": "0.3.1",
        "coding_scope": "HumanEval/0–31 diagnostic subset",
        "judge_api_cost": 0,
        "thinking": True,
        "generation_failures_count_as_empty_responses": True,
    }
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("External evaluation selection changed; use a new destination")
    save_json(path, manifest)
    return collections


def generate(name):
    generate_tasks(
        prepare()[name],
        CONFIG,
        OUTPUT / name,
        on_result=lambda r: print(name, r["task"]["id"], r["status"], flush=True),
    )


def responses(name):
    rows = []
    for task in prepare()[name]:
        path = OUTPUT / name / "items" / task["id"] / "generation.json"
        if not path.exists():
            raise ValueError(
                "Generation incomplete; do not score a partial run as the full selection"
            )
        result = json.loads(path.read_text())
        rows.append(
            (
                task,
                result.get("message", {}).get("content", "")
                if result["status"] == "completed"
                else "",
            )
        )
    return rows


def grade_ifeval():
    destination = OUTPUT / "ifeval"
    package = destination / "evaluator/instruction_following_eval"
    package.mkdir(parents=True, exist_ok=True)
    for name in (
        "instructions.py",
        "instructions_registry.py",
        "instructions_util.py",
        "evaluation_main.py",
    ):
        shutil.copyfile(RAW / "ifeval" / name, package / name)
    inputs = destination / "responses.jsonl"
    inputs.write_text(
        "".join(
            json.dumps({"prompt": t["prompt"], "response": text}) + "\n"
            for t, text in responses("ifeval")
        )
    )
    env = {**os.environ, "PYTHONPATH": str(package.parent), "NLTK_DATA": str(RAW / "nltk")}
    with (destination / "grading.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "instruction_following_eval.evaluation_main",
                f"--input_data={RAW / 'ifeval/data/input_data.jsonl'}",
                f"--input_response_data={inputs}",
                f"--output_dir={destination}",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    report = {}
    for mode in ("strict", "loose"):
        rows = read_jsonl(destination / f"eval_results_{mode}.jsonl")
        values = [v for row in rows for v in row["follow_instruction_list"]]
        report[mode] = {
            "prompt_accuracy": sum(x["follow_all_instructions"] for x in rows) / len(rows),
            "instruction_accuracy": sum(values) / len(values),
            "prompts": len(rows),
            "instructions": len(values),
        }
    save_json(destination / "report.json", report)
    return report


def grade_coding():
    destination = OUTPUT / "humanevalplus"
    staged = destination / "container-input"
    staged.mkdir(parents=True, exist_ok=True)
    problems = read_jsonl(RAW / "humanevalplus/HumanEvalPlus.jsonl")
    ids = {f"HumanEval/{i}" for i in range(32)}
    (staged / "problems.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in problems if x["task_id"] in ids)
    )
    (staged / "samples.jsonl").write_text(
        "".join(
            json.dumps({"task_id": t["task_id"], "solution": text}) + "\n"
            for t, text in responses("humanevalplus")
        )
    )
    results = destination / "evaluation"
    results.mkdir(exist_ok=True)
    image = subprocess.check_output(
        ["docker", "image", "inspect", "writing-assistant-evalplus:0.3.1", "--format", "{{.Id}}"],
        text=True,
    ).strip()
    save_json(
        destination / "container.json",
        {"image": image, "network": "none", "memory": "4g", "cpus": 2},
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=128",
        "--memory=4g",
        "--cpus=2",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,size=1g",
        "-v",
        f"{staged}:/input:ro",
        "-v",
        f"{results}:/output:rw",
        "-e",
        "HUMANEVAL_OVERRIDE_PATH=/input/problems.jsonl",
        image,
        "python",
        "/runner.py",
    ]
    with (destination / "grading.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
    report = json.loads((results / "samples-sanitized_eval_results.json").read_text())
    rows = [values[0] for values in report["eval"].values()]
    assert len(rows) == 32
    save_json(
        destination / "report.json",
        {
            "tasks": len(rows),
            "subset": "HumanEval/0–31",
            "samples_per_task": 1,
            "base_pass_at_1": sum(x["base_status"] == "pass" for x in rows) / len(rows),
            "plus_pass_at_1": sum(
                x["base_status"] == "pass" and x["plus_status"] == "pass" for x in rows
            )
            / len(rows),
        },
    )
    return report


def run():
    """Wait for the current custom run, then execute the two approved free checks."""
    import time

    state = OUTPUT / "pipeline.json"
    prepare()
    save_json(state, {"stage": "waiting_for_custom50"})
    while not (ROOT / "runs/custom50-e2b-it-2026-09-14/review.md").exists():
        time.sleep(10)
    try:
        for name, grader in (("humanevalplus", grade_coding), ("ifeval", grade_ifeval)):
            save_json(state, {"stage": "generation", "benchmark": name})
            generate(name)
            save_json(state, {"stage": "grading", "benchmark": name})
            grader()
        save_json(state, {"stage": "finished"})
    except Exception as exc:
        save_json(state, {"stage": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
