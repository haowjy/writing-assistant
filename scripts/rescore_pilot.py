"""Rescore saved outputs only. Explicit run selection; no candidate or judge calls."""

import json
import re
from pathlib import Path

from writing_agent.catalog import fingerprint, save_json
from writing_agent.prose import ProseFeatures, bandwidth, prose_profile, score_prose

ROOT = Path(__file__).resolve().parents[1]


def run(source: Path, *, allow_download=False):
    destination = source / "rescored"
    selection = json.loads((source / "selection.json").read_text())
    scenarios = {s["id"]: s for s in selection["scenarios"]}
    extractor = ProseFeatures(destination / "features")
    catalog = json.loads((ROOT / "data/processed/custom-eval/catalog.json").read_text())
    references = []
    for book in catalog:
        if not book["id"].startswith("gutenberg-"):
            continue
        eligible = [
            p.strip() for p in re.split(r"\n\s*\n", book["text"]) if 100 <= len(p.split()) <= 300
        ]
        for i, text in enumerate(eligible[:3]):
            references.append(
                {
                    "id": f"{book['id']}-paragraph-{i}",
                    "text": text,
                    "parent_id": book["id"],
                    "parent_hash": book["sha256"],
                    "provenance": "human",
                    "role": "development",
                    "review_status": "unreviewed",
                    "sha256": fingerprint(text),
                }
            )
    save_json(
        destination / "reference-selection.json",
        {
            "policy": "First three 100–300 word paragraphs per Gutenberg book (whitespace count)",
            "limitation": "Unreviewed; not genre/style matched; paragraphs share works",
            "references": references,
        },
    )
    refs = [
        extractor.extract(r["text"], tokens=True, embeddings=True, allow_download=allow_download)
        for r in references
    ]
    vectors = [r["embedding"] for r in refs if "embedding" in r]
    sigma = bandwidth(vectors) if len(vectors) == len(refs) and len(vectors) >= 2 else None
    cards, texts, feats = [], [], []
    for path in sorted(source.glob("attempts/*/*/result.json")):
        result = json.loads(path.read_text())
        card = json.loads((path.parent / "scorecard.json").read_text())
        card["prose_profile"] = score_prose(
            card,
            scenarios[result["scenario_id"]],
            result,
            extractor,
            references=refs,
            sigma=sigma,
            allow_download=allow_download,
        )
        card["prose_profile"]["reference_policy"] = "exploratory Gutenberg; not matched"
        save_json(destination / (result["scenario_id"] + ".json"), card)
        cards.append(card)
        for artifact in card["artifacts"]:
            if artifact["status"] == "ok":
                texts.append(artifact["text"])
                feats.append(extractor.extract(artifact["text"], tokens=True, embeddings=True))
    pooled = prose_profile(texts, feats, references=refs, sigma=sigma)
    pooled["limitation"] = "Exploratory pooled prose across tasks/genres; not a matched benchmark"
    save_json(destination / "pooled-exploratory.json", pooled)
    lines = [
        "# Rescored pilot",
        "",
        "Saved outputs only; no new generation or LLM judging.",
        "",
        "D1/MMD references are unreviewed Gutenberg paragraphs, not task-matched targets.",
        "Original Markdown is preserved. Prose selectors and their limitations are unchanged.",
        "",
        "| Case | Metric | Status | Value / reason |",
        "|---|---|---|---|",
    ]
    for card in cards:
        for key, m in {**card["scores"], **card["prose_profile"]["metrics"]}.items():
            value = (
                json.dumps(m["value"])
                if m["value"] is not None
                else (m.get("reason") or "No numeric result")
            )
            if len(value) > 250:
                value = f"See [{card['scenario_id']}.json]({card['scenario_id']}.json)"
            lines.append(f"| {card['scenario_id']} | {key} | {m['status']} | {value} |")
    lines += [
        "",
        "[Pooled exploratory metrics](pooled-exploratory.json) · "
        "[Reference selection](reference-selection.json)",
        "",
    ]
    (destination / "review.md").write_text("\n".join(lines))
    return {
        "review": str(destination / "review.md"),
        "pooled_mmd": pooled["metrics"]["D2"],
        "feature_errors": [c["prose_profile"]["feature_errors"] for c in cards],
    }
