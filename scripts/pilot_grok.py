"""Five-case Grok/OpenCode research pilot; explicit run(execute=True) starts inference."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from scripts.pilot_e2b import ROOT, SCENARIO_IDS, _fenced
from writing_agent.agent import SYSTEM_PROMPT
from writing_agent.catalog import fingerprint, save_json
from writing_agent.grading import CodexGrader, apply_judgment, grading_packet
from writing_agent.prose import ProseFeatures, score_prose
from writing_agent.scoring import build_report, mechanical_score
from writing_agent.workspace import Workspace

OUTPUT = ROOT / "runs/pilot-grok46-opencode-2026-09-14"
SOURCE = ROOT / "runs/custom50-e2b-it-2026-09-14/selection.json"
MODEL = {"id": "xai/grok-4.6", "kind": "opencode", "harness": "opencode-native-v3"}
LIMITATIONS = (
    "Model-plus-harness pilot: OpenCode native tools, provider reasoning defaults, "
    "and provider generation limits. Custom read/storage/tool budgets are not enforced "
    "by OpenCode. Steps are capped per user turn, not across the conversation. "
    "This is not a controlled model-only comparison."
)


def configuration(workspace, has_tools):
    permission = {"*": "deny"}
    if has_tools:
        for name in ("read", "edit"):
            permission[name] = {"*": "deny"}
            for directory in ("notes", "source", "kb", "drafts"):
                permission[name][directory + "/*"] = "allow"
                permission[name][str(workspace / directory / "*")] = "allow"
                # Non-git OpenCode projects use / as the worktree for permission matching.
                permission[name][str(workspace / directory / "*").lstrip("/")] = "allow"
        permission.update(glob="allow", grep="allow", list="allow", external_directory="deny")
    return {
        "share": "disabled",
        "autoupdate": False,
        "snapshot": False,
        "instructions": [],
        "permission": permission,
        "agent": {
            "writer": {
                "description": "Creative writing collaborator",
                "mode": "primary",
                "prompt": SYSTEM_PROMPT,
                "permission": permission,
                "steps": 12,
            }
        },
    }


def parse_events(events):
    """Retain native calls and expose read observations to existing artifact scoring."""
    messages, trace, usage = [], [], {"input": 0, "output": 0, "reasoning": 0}
    final, thinking, session, reason = "", "", None, None
    errors = []
    for event in events:
        session = event.get("sessionID", session)
        part = event.get("part", {})
        if event["type"] == "reasoning":
            thinking += part.get("text", "")
        elif event["type"] == "text":
            final = part["text"]
            messages.append({"role": "assistant", "content": final, "thinking": thinking})
            thinking = ""
        elif event["type"] == "tool_use":
            state = part["state"]
            name = part["tool"]
            trace.append(
                {
                    "type": "tool",
                    "native_tool": name,
                    "call": {
                        "id": part["callID"],
                        "function": {
                            "name": {"read": "read_file", "grep": "search"}.get(name, name),
                            "arguments": state.get("input", {}),
                        },
                    },
                    "observation": {
                        "ok": state["status"] == "completed",
                        "valid": state["status"] == "completed",
                        "result": state.get("output", state.get("error", "")),
                    },
                }
            )
        elif event["type"] == "step_finish":
            reason = part.get("reason")
            for key in usage:
                usage[key] += part.get("tokens", {}).get(key, 0)
        elif event["type"] == "error":
            errors.append(event.get("error", event))
    return messages, trace, usage, final, session, reason, errors


def attempt(scenario, destination):
    existing = destination / "result.json"
    if existing.exists():
        saved = json.loads(existing.read_text())
        prompts = [m["content"] for m in saved["messages"] if m["role"] == "user"]
        visible = scenario["visible"]
        expected = [visible["brief"], *visible["followups"]]
        if (
            saved["model"] != MODEL
            or saved["before"] != visible["initial_files"]
            or (saved["status"] == "completed" and prompts != expected)
        ):
            raise ValueError("Saved attempt configuration changed")
        return saved
    if (destination / "started.json").exists():
        raise ValueError(f"Interrupted attempt needs inspection: {destination}")
    visible = scenario["visible"]
    if visible["tools"] and set(visible["tools"]) != {
        "read_file",
        "list_dir",
        "search",
        "write_file",
        "patch_file",
    }:
        raise ValueError("Pilot only supports no-tools or the complete file-tool set")
    save_json(destination / "started.json", {"scenario_id": scenario["id"], "model": MODEL})
    started = time.monotonic()
    messages, trace, turns, usage = [], [], [], {"input": 0, "output": 0, "reasoning": 0}
    output, error, session = "", None, None
    status = "completed"
    with tempfile.TemporaryDirectory(prefix="writing-grok-") as temporary:
        root = Path(temporary)
        workspace = Workspace(root)
        for name, content in visible["initial_files"].items():
            workspace.write_file(name, content)
        config = configuration(root, bool(visible["tools"]))
        save_json(destination / "opencode-config.json", config)
        env = {**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps(config)}
        for name in ("CLAUDE_CODE", "EXTERNAL_SKILLS", "PROJECT_CONFIG", "AUTOCOMPACT"):
            env["OPENCODE_DISABLE_" + name] = "true"
        try:
            for turn, prompt in enumerate([visible["brief"], *visible["followups"]]):
                messages.append({"role": "user", "content": prompt})
                command = [
                    "opencode",
                    "run",
                    "--dir",
                    str(root),
                    "--pure",
                    "--model",
                    MODEL["id"],
                    "--agent",
                    "writer",
                    "--format",
                    "json",
                    "--thinking",
                    "--title",
                    "Writing pilot " + scenario["id"],
                ]
                if session:
                    command.extend(["--session", session])
                command.append(prompt)
                save_json(
                    destination / f"launch-{turn}.json", {"command": command, "cwd": temporary}
                )
                with (destination / f"events-{turn}.jsonl").open("w") as out:
                    with (destination / f"stderr-{turn}.txt").open("w") as err:
                        proc = subprocess.run(
                            command, cwd=root, env=env, stdout=out, stderr=err, timeout=300
                        )
                events = [
                    json.loads(line)
                    for line in (destination / f"events-{turn}.jsonl").read_text().splitlines()
                    if line
                ]
                msgs, calls, tokens, output, session, reason, errors = parse_events(events)
                messages.extend(msgs)
                trace.extend(calls)
                for key in usage:
                    usage[key] += tokens[key]
                if (
                    proc.returncode
                    or errors
                    or reason != "stop"
                    or not output.strip()
                    or not session
                ):
                    raise ValueError(
                        f"Incomplete OpenCode turn: exit={proc.returncode}, "
                        f"reason={reason}, errors={errors}"
                    )
                snapshot = workspace.snapshot()
                turns.append(
                    {"output": output, "snapshot": snapshot, "message_index": len(messages) - 1}
                )
                save_json(destination / f"snapshot-{turn}.json", snapshot)
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            status, error = "error", str(exc)
        after = workspace.snapshot()
        saved = Workspace(destination / "workspace")
        for name, content in after.items():
            saved.write_file(name, content)
    result = {
        "schema_version": 1,
        "scenario_id": scenario["id"],
        "family": scenario["family"],
        "genre": scenario["genre"],
        "instruction_specificity": scenario["instruction_specificity"],
        "provenance": "synthetic",
        "condition": scenario.get("condition", "default"),
        "model": MODEL,
        "status": status,
        "error": error,
        "output": output,
        "messages": messages,
        "trace": trace,
        "turns": turns,
        "before": visible["initial_files"],
        "after": after,
        "tool_calls": len(trace),
        "attempted_tool_calls": len(trace),
        "tool_errors": sum(not e["observation"]["ok"] for e in trace),
        "usage": usage,
        "latency_seconds": time.monotonic() - started,
        "session_id": session,
        "path": str(destination),
        "limitations": LIMITATIONS,
    }
    save_json(existing, result)
    return result


def run(*, execute=False, grade=False):
    selection = json.loads(SOURCE.read_text())
    by_id = {s["id"]: s for s in selection["scenarios"]}
    scenarios = [by_id[key] for key in SCENARIO_IDS]
    manifest = {
        "model": MODEL,
        "scenarios": scenarios,
        "limitations": LIMITATIONS,
        "opencode_version": subprocess.check_output(["opencode", "--version"], text=True).strip(),
    }
    if not execute:
        return {"cases": SCENARIO_IDS, "model": MODEL, "limitations": LIMITATIONS}
    path = OUTPUT / "selection.json"
    if path.exists() and fingerprint(json.loads(path.read_text())) != fingerprint(manifest):
        raise ValueError("Saved experiment selection changed")
    save_json(path, manifest)
    grader = CodexGrader(OUTPUT / "judgments", max_calls=5, timeout=600) if grade else None
    features, cards = ProseFeatures(OUTPUT / "features"), []
    index = [
        "# Grok 4.6 / OpenCode five-case pilot",
        "",
        LIMITATIONS,
        "",
        "| Case | Execution | Task completion | Prose /5 | Review |",
        "|---|---|---|---|---|",
    ]
    for scenario in scenarios:
        destination = OUTPUT / scenario["id"]
        result = attempt(scenario, destination)
        card = mechanical_score(scenario, result)
        card["prose_profile"] = score_prose(card, scenario, result, features)
        if grader:
            packet = grading_packet(scenario, result, card)
            card = apply_judgment(card, packet, grader.grade(packet))
        save_json(destination / "scorecard.json", card)
        cards.append(card)
        review = [
            f"# {scenario['id']}",
            "",
            LIMITATIONS,
            "",
            "[Scorecard](scorecard.json) · [Result and tool trace](result.json)",
            "",
            "## Conversation",
            "",
        ]
        for message in result["messages"]:
            if message.get("thinking"):
                review += ["### Thinking", "", _fenced(message["thinking"])]
            review += [f"### {message['role']}", "", message["content"], ""]
        review += ["## Final files", ""]
        for name, content in result["after"].items():
            review += [f"### {name}", "", _fenced(content)]
        review += ["## Scores", "", "| Metric | Score | Status |", "|---|---|---|"]
        for key, score in card["scores"].items():
            if score["status"] != "not_applicable":
                value = score["value"]
                if isinstance(value, dict):
                    value = "See scorecard"
                review.append(f"| {key} | {value} | {score['status']} |")
        for key, score in card["scores"].items():
            if score.get("rationale"):
                review += ["", f"### {key}: {score['value']}", "", score["rationale"], ""]
                review.extend(
                    f"- {d['name']}: {d['value']} — {d.get('reason', '')}"
                    for d in score.get("dimensions", [])
                )
        (destination / "review.md").write_text("\n".join(review))
        task = card["scores"]["Q3"]
        prose = card["scores"]["Q2"]
        index.append(
            f"| {scenario['id']} | {result['status']} | "
            f"{task['value'] if task['status'] == 'ok' else task['status']} | "
            f"{prose['value'] if prose['status'] == 'ok' else prose['status']} | "
            f"[Read]({scenario['id']}/review.md) |"
        )
        (OUTPUT / "review.md").write_text("\n".join(index) + "\n")
        build_report(cards, OUTPUT / "reports")
        print(scenario["id"], result["status"], task["value"], flush=True)
    return {"review": str(OUTPUT / "review.md"), "cases": len(cards)}


if __name__ == "__main__":
    print(run())
