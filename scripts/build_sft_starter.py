"""Compile a small, labeled SFT collection from selected sources and authored projects."""

import copy
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from scripts.train_sft import ROOT, SETTINGS, evaluation_source_groups, tokenizer
from writing_agent.agent import SYSTEM_PROMPT
from writing_agent.catalog import fingerprint, save_json, validate_catalog
from writing_agent.data import validate_records
from writing_agent.training import encode_trajectory
from writing_agent.workspace import TOOL_SCHEMAS, Workspace, dispatch

DESTINATION = ROOT / "data/processed/sft-starter-v1"
RECORDS = ROOT / "data/training/sft-v1.jsonl"
GENERATOR = {"harness": "codex", "provider": "openai", "model": "session-model-not-recorded"}
TASKS = {
    "F1": "direct_prose",
    "F2": "file_authoring_revision",
    "F3": "planning",
    "F4": "kb_construction_maintenance",
    "F5": "writing_from_kb",
}


def source_record(seed):
    if "book" in seed:
        path = ROOT / f"data/raw/sft-starter/{seed['book']}.txt"
        text = path.read_text()
        start = text.index(seed["opening"])
        paragraphs = list(re.finditer(r"\S.*?(?=\n\s*\n|\Z)", text[start:], re.S))[:2]
        end = start + paragraphs[-1].end()
        excerpt = text[start:end]
        source = {
            "id": seed["id"] + "-opening",
            "text": excerpt,
            "raw_file": str(path.relative_to(ROOT)),
            "raw_sha256": fingerprint(path.read_bytes()),
            "character_span": [start, end],
            "span_basis": "UTF-8 decoded, universal newlines",
            "url": f"https://www.gutenberg.org/ebooks/{seed['book']}",
            "provenance": "human",
            "upstream_split": "unsplit",
            "terms": {
                "license": "Public domain in the USA; Project Gutenberg terms retained",
                "evidence": f"https://www.gutenberg.org/ebooks/{seed['book']}",
            },
        }
        seed = {
            **seed,
            "source": excerpt,
            "prefix": paragraphs[0].group(),
            "continuation": paragraphs[1].group(),
        }
    else:
        source = {
            "id": seed["id"],
            "text": seed["source"],
            "provenance": "synthetic",
            "generator": GENERATOR,
            "upstream_split": "newly_authored",
            "terms": {
                "license": "Repository-authored synthetic research material",
                "evidence": "data/training/starter-design.json",
            },
        }
    source.update(
        work_id=seed["id"],
        author_id=seed["author_id"],
        author=seed["author"],
        role="train" if seed["split"] == "train" else "development",
        sha256=fingerprint(source["text"]),
        parents=[],
        transformations=[],
    )
    return source, seed


class Trajectory:
    """Construct tool observations by executing the real local workspace functions."""

    def __init__(self, seed, source, family, initial):
        self.temporary = tempfile.TemporaryDirectory(prefix="sft-starter-")
        self.workspace = Workspace(Path(self.temporary.name))
        for path, text in initial.items():
            self.workspace.write_file(path, text)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.source, self.seed, self.family, self.initial = source, seed, family, initial
        self.tools = set()
        self.calls = 0

    def say(self, role, content):
        self.messages.append({"role": role, "content": content})

    def tool(self, name, **arguments):
        self.tools.add(name)
        self.calls += 1
        call_id = f"call-{self.calls}"
        self.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        )
        observation = dispatch(self.workspace, name, arguments)
        if not observation["ok"]:
            raise ValueError(f"Authored trajectory failed: {observation}")
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(observation, ensure_ascii=False),
            }
        )

    def finish(self):
        seed, family, source = self.seed, self.family, self.source
        final = self.workspace.snapshot()
        self.temporary.cleanup()
        label = "synthetic"
        if "book" in seed:
            label = "synthetic_fanfic" if family in {"F2", "F5"} else "half_synthetic"
        artifacts = []
        for i, message in enumerate(self.messages):
            if message["role"] == "assistant":
                origin = "human" if family == "F1" and "book" in seed else "synthetic"
                artifacts.append(
                    {
                        "location": {"message_index": i},
                        "provenance": origin,
                        "kind": "tool_call" if message.get("tool_calls") else "reply",
                        "parents": [source["id"]],
                        "sha256": fingerprint(message),
                    }
                )
        for path, text in self.initial.items():
            artifacts.append(
                {
                    "location": {"initial_file": path},
                    "parents": [source["id"]],
                    "provenance": source["provenance"] if text == source["text"] else "synthetic",
                    "kind": "context",
                    "sha256": fingerprint(text),
                }
            )
        for path, text in final.items():
            if self.initial.get(path) != text:
                artifacts.append(
                    {
                        "location": {"final_file": path},
                        "parents": [source["id"]],
                        "provenance": "synthetic",
                        "kind": "kb" if path.startswith("kb/") else "prose",
                        "sha256": fingerprint(text),
                    }
                )
        return {
            "schema_version": 1,
            "id": "starter-" + seed["id"] + "-" + family,
            "family": family,
            "task_type": TASKS[family],
            "split": seed["split"],
            "genre": seed["genre"],
            "style": seed["style"],
            "instruction_specificity": "explicit" if family in {"F1", "F2"} else "loose",
            "source_dataset": "gutenberg" if "book" in seed else "new_synthetic",
            "source_groups": [source["work_id"]],
            "parents": [source["id"]],
            "upstream_split": source["upstream_split"],
            "review_status": "pending",
            "provenance": {
                "label": label,
                "author_id": source["author_id"],
                "work_id": source["work_id"],
                "source": source.get("url", source["id"]),
                "license": source["terms"]["license"],
                "generator": GENERATOR,
            },
            "transformations": [
                {
                    "kind": "authorship_checkpoint" if "book" in seed else "authored_project",
                    "source_hash": source["sha256"],
                    "visible_cutoff": source.get("character_span"),
                    "task": TASKS[family],
                    "later_source_text_supplied": False,
                }
            ],
            "artifacts": artifacts,
            "initial_files": self.initial,
            "expected_files": final,
            "messages": self.messages,
            "tools": [t for t in TOOL_SCHEMAS if t["function"]["name"] in self.tools],
            "verification": {"tool_calls_replayed": self.calls, "tool_errors": 0},
        }


def project_examples(seed, source):
    rows = []
    t = Trajectory(seed, source, "F1", {})
    if "book" in seed:
        t.say(
            "user",
            "Continue this opening with its next paragraph. Return prose only.\n\n"
            + seed["prefix"],
        )
        t.say("assistant", seed["continuation"])
    else:
        t.say(
            "user",
            seed["source"]
            + "\n\nWrite a short scene in "
            + seed["style"]
            + ". Return prose only; leave the unanswered question open.",
        )
        t.say("assistant", seed["prose"])
    rows.append(t.finish())

    t = Trajectory(seed, source, "F2", {"notes/source.md": source["text"]})
    t.say(
        "user",
        "Use notes/source.md to write "
        + ("an alternate continuation" if "book" in seed else "a short scene")
        + ". Save it in drafts/scene.md and leave the notes unchanged.",
    )
    t.tool("read_file", path="notes/source.md")
    t.tool("write_file", path="drafts/scene.md", content=seed["branch"])
    t.say("assistant", "Saved the scene in drafts/scene.md; the source notes are unchanged.")
    edit = seed["revision"]
    t.say("user", f"In the draft, replace “{edit['old']}” with “{edit['new']}”. Keep the rest.")
    t.tool("patch_file", path="drafts/scene.md", **edit)
    t.say("assistant", "Made that local edit and kept the rest of the draft.")
    rows.append(t.finish())

    t = Trajectory(seed, source, "F3", {})
    t.say(
        "user",
        source["text"] + "\n\nWhere could the next scene go? Give me a few different directions, "
        "without choosing one yet.",
    )
    t.say("assistant", seed["alternatives"])
    rows.append(t.finish())

    index = "# Project index\n\n- [Facts at the cutoff](facts.md)\n- [Open questions](questions.md)"
    facts = (
        "# Facts at the supplied cutoff\n\n"
        + seed["notes"]
        + "\n\nSource: [passage](../source/passage.md)."
    )
    questions = "# Open questions\n\n" + seed["uncertainty"]
    t = Trajectory(seed, source, "F4", {"source/passage.md": source["text"]})
    t.say(
        "user",
        "Make a small Markdown wiki from source/passage.md so I can plan the next scene. "
        "Start at kb/index.md, make it navigable, and keep the source intact.",
    )
    t.tool("read_file", path="source/passage.md")
    for path, content in [
        ("kb/index.md", index),
        ("kb/facts.md", facts),
        ("kb/questions.md", questions),
    ]:
        t.tool("write_file", path=path, content=content)
    t.say(
        "assistant", "Created the index, source-backed facts, and open questions as linked pages."
    )
    t.say(
        "user",
        seed["accepted_update"] + " Update the wiki, preserving the earlier state as history.",
    )
    t.tool(
        "patch_file",
        path="kb/facts.md",
        old=facts,
        new=facts + "\n\n## Subsequent author update\n\n" + seed["accepted_update"],
    )
    t.say(
        "assistant",
        "Recorded the accepted update separately from the facts at the original cutoff.",
    )
    rows.append(t.finish())

    t = Trajectory(
        seed,
        source,
        "F5",
        {
            "kb/index.md": index,
            "kb/facts.md": facts.replace("\n\nSource: [passage](../source/passage.md).", ""),
            "kb/questions.md": questions,
        },
    )
    t.say(
        "user",
        "Use the wiki starting at kb/index.md to write "
        + ("a new alternate continuation" if "book" in seed else "a short scene")
        + ". Put the prose in your reply and leave the wiki alone.",
    )
    for path in ("kb/index.md", "kb/facts.md", "kb/questions.md"):
        t.tool("read_file", path=path)
    t.say("assistant", seed["branch"])
    rows.append(t.finish())
    return rows


def build():
    upstream = json.loads((ROOT / "data/processed/custom-eval/catalog.json").read_text())
    catalog, rows = [], []
    tok = tokenizer()
    seeds = json.loads((ROOT / "data/training/starter-design.json").read_text())
    for seed in seeds:
        source, seed = source_record(seed)
        catalog.append(source)
        rows.extend(project_examples(seed, source))
    candidates = sorted(
        [s for s in upstream if s["id"].startswith("tmas-train-") and s["id"].endswith("-story")],
        key=lambda s: (len(s["text"].split()), s["id"]),
    )
    selected_ids = []
    excluded_tmas = {
        "tmas-train-example_097-story": "Embedded song lyrics require separate reuse review."
    }
    for story in candidates:
        if story["id"] in excluded_tmas:
            continue
        prompt = next(s for s in upstream if s["id"] == story["id"].replace("-story", "-prompt"))
        record = {
            "schema_version": 1,
            "id": "starter-" + story["work_id"],
            "family": "F1",
            "task_type": TASKS["F1"],
            "split": "train",
            "review_status": "pending",
            "genre": "upstream_unspecified",
            "style": "upstream_prompt",
            "instruction_specificity": "explicit",
            "source_dataset": "tell_me_a_story",
            "source_groups": [story["work_id"]],
            "parents": [prompt["id"], story["id"]],
            "upstream_split": "train",
            "provenance": {
                "label": "human",
                "author_id": "tmas-unattributed-authors",
                "work_id": story["work_id"],
                "source": story["url"],
                "license": story["terms"]["license"],
                "generator": None,
            },
            "transformations": [
                {"kind": "verbatim_prompt_target_pair", "source_hash": story["sha256"]}
            ],
            "initial_files": {},
            "expected_files": {},
            "tools": [],
            "messages": [
                {"role": "user", "content": prompt["text"]},
                {"role": "assistant", "content": story["text"]},
            ],
            "artifacts": [
                {
                    "location": {"message_index": i},
                    "provenance": s["provenance"],
                    "parents": [s["id"]],
                    "sha256": s["sha256"],
                }
                for i, s in enumerate((prompt, story))
            ],
            "verification": {"tool_calls_replayed": 0, "tool_errors": 0},
        }
        try:
            encode_trajectory(record, tok, max_length=SETTINGS.max_length)
        except ValueError as exc:
            if "exceed" in str(exc):
                continue
            raise
        rows.append(record)
        for item in (prompt, story):
            entry = copy.deepcopy(item)
            entry.update(role="train", author_id="tmas-unattributed-authors")
            entry["terms"]["training"] = "selected_upstream_train"
            catalog.append(entry)
        selected_ids.append(story["id"])
        if len(selected_ids) == 4:
            break
    if len(selected_ids) != 4:
        raise ValueError("Could not select four complete TMAS train pairs within the token limit")
    hanna = [copy.deepcopy(s) for s in upstream if s["id"] in {"hanna-0", "hanna-1", "hanna-2"}]
    for source in hanna:
        source["use"] = "evaluation_reference_only"
    catalog.extend(hanna)
    excluded = evaluation_source_groups()
    # Previously acquired Gutenberg works were used as evaluation prose references.
    excluded.update(s["work_id"] for s in upstream if s["id"].startswith("gutenberg-"))
    used = {g for row in rows for g in row["source_groups"]}
    if used & excluded:
        raise ValueError(f"Evaluation source reuse: {sorted(used & excluded)}")
    separation = {
        "excluded_source_groups": sorted(excluded),
        "overlapping_source_groups": [],
        "upstream_tmas_split": "train",
        "training_validation_group_check": "passed",
        "limitation": "Exact recorded lineage only; anonymous author overlap is unknown.",
    }
    validate_catalog(catalog)
    validate_records(rows)
    encoded = [encode_trajectory(r, tok, max_length=SETTINGS.max_length) for r in rows]
    serialized = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if RECORDS.exists() and RECORDS.read_text() != serialized:
        raise ValueError("Existing dataset differs; choose a new RECORDS path for a revision")
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    RECORDS.write_text(serialized)
    save_json(DESTINATION / "catalog.json", catalog)
    save_json(DESTINATION / "hanna-reference.json", hanna)
    save_json(DESTINATION / "mask-audit.json", encoded)
    inventory = {
        "version": 1,
        "records": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "tasks": dict(Counter(r["family"] for r in rows)),
        "sources": dict(Counter(r["source_dataset"] for r in rows)),
        "provenance": dict(Counter(r["provenance"]["label"] for r in rows)),
        "tokens": sum(len(r["input_ids"]) for r in encoded),
        "max_tokens": max(len(r["input_ids"]) for r in encoded),
        "supervised_tokens": sum(r["supervised_tokens"] for r in encoded),
        "hanna_reference_only": len(hanna),
        "selected_tmas": selected_ids,
        "excluded_tmas": excluded_tmas,
        "dataset_hash": fingerprint(rows),
        "review_status": "pending",
        "selection_policy": (
            "four shortest complete upstream-train TMAS pairs that fit after exclusions; "
            "two independent Gutenberg works; two original synthetic projects"
        ),
        "source_inventory": [{k: v for k, v in s.items() if k != "text"} for s in catalog],
        "separation_audit": separation,
    }
    save_json(DESTINATION / "inventory.json", inventory)
    save_json(ROOT / "data/training/starter-selection.json", inventory)
    lines = [
        "# SFT starter dataset",
        "",
        "Pending literary review; structural checks and token masks verified.",
        "",
        "| ID | Task | Split | Source | Provenance | Tokens |",
        "|---|---|---|---|---|---:|",
    ]
    for r, e in zip(rows, encoded, strict=True):
        lines.append(
            f"| {r['id']} | {r['family']} | {r['split']} | {r['source_dataset']} | "
            f"{r['provenance']['label']} | {len(e['input_ids'])} |"
        )
    for r in rows:
        lines += ["", "## " + r["id"], "", "### Conversation", ""]
        for m in r["messages"]:
            if m["role"] in {"user", "assistant"} and m.get("content"):
                lines += ["**" + m["role"] + "**", "", m["content"], ""]
            if m.get("tool_calls") or m["role"] == "tool":
                lines += [
                    "**" + m["role"] + " (tool trace)**",
                    "",
                    "````json",
                    json.dumps(m, ensure_ascii=False, indent=2),
                    "````",
                    "",
                ]
        for path, text in r["expected_files"].items():
            lines += ["### Final file: " + path, "", "````markdown", text, "````", ""]
    (DESTINATION / "review.md").write_text("\n".join(lines))
    return inventory


if __name__ == "__main__":
    result = build()
    print({k: v for k, v in result.items() if k not in {"source_inventory", "separation_audit"}})
