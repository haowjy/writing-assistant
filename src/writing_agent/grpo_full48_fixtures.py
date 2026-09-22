"""Offline mechanical constructions, never quality goldens or sampled successes."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

from writing_agent.agent import run_agent
from writing_agent.backends import ScriptedBackend
from writing_agent.catalog import fingerprint, save_json
from writing_agent.grpo_full48 import (
    LIMITATIONS,
    _binding,
    load_full48_release,
    mechanical_full48_reward,
)
from writing_agent.workspace import Workspace


def fixture_messages(task, case="positive"):
    """Construct known mechanics using private checks OFFLINE, never in candidate inputs."""
    contract = _binding()["tasks"][task["id"]]
    visible = task["visible"]
    actions = []
    for path in contract["reads"]:
        actions.append({"name": "read_file", "arguments": {"path": path}})
    retrieval = contract.get("retrieval")
    if retrieval:
        actions += [
            {"name": "search", "arguments": {"query": "STORY-" + task["id"][-3:]}},
            {"name": "read_file", "arguments": {"path": "kb/index.md"}},
            {"name": "read_file", "arguments": {"path": retrieval["path"]}},
        ]
    bodies = {}
    for delivery in contract["delivery"]:
        prefix, suffix = "", ""
        if retrieval:
            prefix = retrieval["quote"] + "\n"
        if task["family"] == "F2":
            suffix = next(c["text"] for c in task["labels"]["checks"] if c["kind"] == "protected")
        if task["family"] == "F4":
            prefix = (
                "\n".join(
                    f"[{Path(target).stem}]({Path(target).name})"
                    for origin, target in contract["wiki_edges"]
                    if origin == delivery["path"]
                )
                + "\n"
            )
        words = delivery["minimum_words"] + 10 - len((prefix + suffix).split())
        bodies[delivery["id"]] = prefix + " ".join(["fixture"] * words) + "\n" + suffix
    if case == "partial":
        bodies[contract["delivery"][0]["id"]] = " ".join(
            ["fixture"] * (contract["delivery"][0]["minimum_words"] - 1)
        )
    if case == "content-error":
        excluded = next(c for c in task["labels"]["checks"] if c["kind"] == "excludes")
        bodies[excluded["artifact"]] += "\n" + excluded["text"]
    if case == "duplicate-quote":
        bodies["scene"] += "\n" + retrieval["quote"]
    if case == "missing-sibling-link":
        bodies["state"] = bodies["state"].replace("[questions](questions.md)", "questions")
    if case == "moved-protected-line":
        body = bodies["scene"]
        line = next(c["text"] for c in task["labels"]["checks"] if c["kind"] == "protected")
        bodies["scene"] = line + "\n" + body.removesuffix(line)
    if case == "no-retrieval":
        actions = []
    if case == "read-without-search":
        actions = [a for a in actions if a["name"] != "search"]
    if case == "search-without-read":
        actions = [a for a in actions if a["name"] != "read_file"]
    if case == "read-before-search":
        actions = list(reversed(actions))
    output = "Done."
    file_channel = contract["delivery"][0]["kind"] == "file"
    for i, delivery in enumerate(contract["delivery"]):
        text = bodies[delivery["id"]]
        if file_channel:
            path = delivery["path"]
            if case == "missing" and i == 0:
                continue
            if case == "wrong-channel":
                output += "\n" + text
                continue
            if case == "no-op":
                if path not in visible["initial_files"]:
                    continue
                text = visible["initial_files"][path]
            actions.append({"name": "write_file", "arguments": {"path": path, "content": text}})
        elif case not in {"missing", "no-op", "wrong-channel"}:
            output = text
        elif case == "wrong-channel":
            actions.append(
                {"name": "write_file", "arguments": {"path": "unrequested.md", "content": text}}
            )
    if case == "equivalent-paths":
        if task["family"] == "F2":
            write = next(a for a in actions if a["name"] == "write_file")
            body = write["arguments"]["content"]
            actions.append(
                {
                    "name": "patch_file",
                    "arguments": {
                        "path": write["arguments"]["path"],
                        "old": body,
                        "new": body.replace("fixture", "revised", 1),
                    },
                }
            )
        for action in actions:
            args = action["arguments"]
            if action["name"] == "search":
                args["path"] = "./kb//."
            elif "path" in args:
                args["path"] = "./" + args["path"].replace("/", "//./")
    messages = [{"role": "assistant", "content": "Done."} for _ in visible["followups"]]
    messages += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"fixture-{i}", "type": "function", "function": action}],
        }
        for i, action in enumerate(actions)
    ]
    messages.append({"role": "assistant", "content": output})
    if case == "failed":
        messages = [{"role": "invalid-fixture-role", "content": "invalid"}]
    return messages


def run_fixture(task, messages, root):
    visible = task["visible"]
    workspace = Workspace(root / "workspace", max_total_bytes=visible["budgets"]["max_total_bytes"])
    for path, text in visible["initial_files"].items():
        workspace.write_file(path, text)
    trace = []
    result = run_agent(
        ScriptedBackend(messages),
        workspace,
        [{"role": "user", "content": visible["brief"]}],
        tools=visible["tools"],
        followups=visible["followups"],
        emit=trace.append,
        **{k: v for k, v in visible["budgets"].items() if k != "max_total_bytes"},
    )
    result.update(before=visible["initial_files"], after=workspace.snapshot(), trace=trace)
    return result


def validate_full48(release_directory, destination):
    """Persist every execution and discrepancy before reporting failure; never overwrite."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Use an absent evidence directory; failed runs must be preserved")
    destination.mkdir(parents=True)
    release = load_full48_release(release_directory)
    save_json(destination / "identity.json", {k: v for k, v in release.items() if k != "tasks"})
    rows, failures = [], []
    for task in release["tasks"]:
        cases = [
            "positive",
            "missing",
            "no-op",
            "partial",
            "failed",
            "wrong-channel",
            "content-error",
            "no-retrieval",
        ]
        if task["family"] == "F5":
            cases += [
                "read-without-search",
                "search-without-read",
                "read-before-search",
                "duplicate-quote",
            ]
        if task["family"] == "F4":
            cases += ["missing-sibling-link"]
        if task["family"] == "F2":
            cases += ["moved-protected-line"]
        if task["id"] in {"wave1-train-023", "wave1-train-043"}:
            cases += ["equivalent-paths"]
        candidates = {}
        for case in cases:
            root = destination / task["id"] / case
            messages = fixture_messages(task, case)
            candidates[case] = (
                run_fixture(task, messages, root),
                messages,
                "real-scripted-tool-loop",
            )
        positive = candidates["positive"][0]
        for case in ("missing-turn-evidence", "missing-tool-evidence", "forged-before"):
            result = copy.deepcopy(positive)
            if case == "missing-turn-evidence":
                result["turns"] = []
            elif case == "missing-tool-evidence":
                result["trace"] = [e for e in result["trace"] if e["type"] != "tool"]
            else:
                result["before"] = {"candidate-claimed.md": "Not authoritative"}
            candidates[case] = (result, None, "saved-evidence-counterfactual")
        if task["id"] in {"wave1-train-023", "wave1-train-043"}:
            for case, prefix in (("forged-absolute-path", "/"), ("forged-traversal-path", "./../")):
                result = copy.deepcopy(positive)
                event = next(e for e in reversed(result["trace"]) if e["type"] == "tool")
                args = event["call"]["function"]["arguments"]
                args["path"] = prefix + args["path"]
                candidates[case] = (result, None, "impossible-successful-trace-counterfactual")
        for case, (result, messages, evidence_kind) in candidates.items():
            before_scoring = fingerprint(result)
            reward = mechanical_full48_reward(task, result)
            if fingerprint(result) != before_scoring:
                raise ValueError("Reward mutated saved fixture evidence")
            if case in {"positive", "forged-before", "moved-protected-line", "equivalent-paths"}:
                expected = "one"
            elif case in {
                "missing",
                "no-op",
                "partial",
                "failed",
                "wrong-channel",
                "missing-turn-evidence",
                "forged-absolute-path",
                "forged-traversal-path",
            }:
                expected = "zero"
            elif case == "missing-tool-evidence" and task["family"] in {"F2", "F3", "F4"}:
                expected = "zero"
            elif case in {"no-retrieval", "missing-tool-evidence"} and (
                task["family"] == "F3" or not task["visible"]["tools"]
            ):
                expected = "one"
            else:
                expected = "partial"
            passed = {
                "one": reward.value == 1,
                "zero": reward.value == 0,
                "partial": 0 < reward.value < 1,
            }[expected]
            row = {
                "task": task["id"],
                "case": case,
                "expected": expected,
                "passed": passed,
                "evidence_kind": evidence_kind,
                "reward": asdict(reward),
                "result_hash": fingerprint(result),
            }
            save_json(
                destination / task["id"] / case / "result.json",
                {**row, "messages_script": messages, "result": result},
            )
            rows.append(row)
            if not passed:
                failures.append(
                    {"task": task["id"], "case": case, "expected": expected, "actual": reward.value}
                )
    report = {
        "contract_hash": release["contract_hash"],
        "is_model_evaluation": False,
        "fixture_description": "Repetitive mechanical text; Done on intermediate turns; "
        "not semantically validated, not sampled, not literary goldens.",
        "limitations": LIMITATIONS,
        "cases": rows,
        "failures": failures,
    }
    save_json(destination / "evidence.json", report)
    if failures:
        raise ValueError(f"{len(failures)} fixture discrepancies; see {destination}/evidence.json")
    return report


def measure_full48(release_directory, evidence_directory, tokenizer_directory):
    """Count complete constructed paths with native thinking/tool framing, no weights.

    Repeated fixture words and fixed reasoning are not realistic sampled lengths,
    upper bounds, or sufficient evidence for choosing a training context ceiling.
    """
    from transformers import AutoTokenizer

    from writing_agent.grpo_rollout import native_suffix
    from writing_agent.inference import render_messages
    from writing_agent.workspace import TOOL_SCHEMAS

    evidence_directory = Path(evidence_directory)
    destination = evidence_directory / "native-token-evidence.json"
    if destination.exists():
        raise ValueError("Native evidence already exists; preserve previous measurements")
    tokenizer_directory = Path(tokenizer_directory)
    if not tokenizer_directory.is_dir():
        raise ValueError("Tokenizer must be an existing local directory")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory, local_files_only=True, trust_remote_code=False
    )
    release = load_full48_release(release_directory)
    rows = []
    for task in release["tasks"]:
        saved = json.loads((evidence_directory / task["id"] / "positive/result.json").read_text())
        if mechanical_full48_reward(task, saved["result"]).value != 1:
            raise ValueError("Saved fixture no longer passes frozen mechanics")
        messages = copy.deepcopy(saved["result"]["messages"])
        schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in task["visible"]["tools"]]

        def encode(history, generation, schemas=schemas):
            text = tokenizer.apply_chat_template(
                render_messages(history),
                tools=schemas or None,
                tokenize=False,
                add_generation_prompt=generation,
                enable_thinking=True,
            )
            return tokenizer.encode(text, add_special_tokens=False)

        history = messages[:2]
        initial = encode(history, True)
        appended, boundaries = [], []
        i = 2
        while i < len(messages):
            assistant = messages[i]
            if assistant["role"] != "assistant":
                raise ValueError("Unexpected fixture message sequence")
            assistant["thinking"] = "Follow the visible instructions and use the workspace."
            prompt = encode(history, True)
            full = encode(history + [assistant], False)
            if full[: len(prompt)] != prompt:
                raise ValueError(f"Native fixture prefix unstable: {task['id']}")
            action = full[len(prompt) :]
            boundary = "<|tool_response>" if assistant.get("tool_calls") else "<turn|>"
            stop = tokenizer.encode(boundary, add_special_tokens=False)
            if len(stop) != 1 or stop[0] not in action:
                raise ValueError("Missing native action stop")
            action = action[: len(action) - action[::-1].index(stop[0])]
            i += 1
            external = []
            while i < len(messages) and messages[i]["role"] != "assistant":
                external.append(messages[i])
                i += 1
            suffix = (
                native_suffix(tokenizer, assistant, external, action, thinking=True)
                if external
                else []
            )
            boundaries.append(
                {
                    "training_input_tokens": len(initial) + len(appended),
                    "evaluation_prompt_tokens": len(prompt),
                    "constructed_action_ids": action,
                    "environment_ids": suffix,
                }
            )
            appended.extend(action + suffix)
            history.extend([assistant, *external])
        rows.append(
            {
                "task": task["id"],
                "result_hash": saved["result_hash"],
                "initial_ids": initial,
                "boundaries": boundaries,
                "constructed_action_tokens": sum(
                    len(b["constructed_action_ids"]) for b in boundaries
                ),
                "max_constructed_action_tokens": max(
                    len(b["constructed_action_ids"]) for b in boundaries
                ),
                "training_trajectory_tokens": len(initial) + len(appended),
            }
        )
    report = {
        "is_model_evaluation": False,
        "thinking": "fixed 8-word sentence per decision",
        "warning": "Constructed mechanical paths only; no sampled success, semantic "
        "validation, worst-case bound, budget recommendation, or GPU fit claim.",
        "contract_hash": release["contract_hash"],
        "tokenizer_files": {
            p.name: fingerprint(p.read_bytes())
            for p in tokenizer_directory.iterdir()
            if p.name in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
        },
        "tasks": rows,
    }
    save_json(destination, report)
    return report
