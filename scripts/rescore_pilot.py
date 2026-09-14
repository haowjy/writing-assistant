"""Rescore saved outputs only. Explicit run selection; no candidate or judge calls."""

import json
import re
from pathlib import Path

from writing_agent.catalog import fingerprint, save_json
from writing_agent.prose import ProseFeatures, bandwidth, prose_profile, score_prose
from writing_agent.references import load_matched_references

ROOT = Path(__file__).resolve().parents[1]


def run(source: Path, *, allow_download=False, matched_manifest: Path | None = None):
    destination = source / ("rescored-dual" if matched_manifest else "rescored")
    selection = json.loads((source / "selection.json").read_text())
    scenarios = {s["id"]: s for s in selection["scenarios"]}
    matched = load_matched_references(matched_manifest, scenarios) if matched_manifest else None
    if matched:
        save_json(destination / "matched-reference-selection.json", matched)
    extractor = ProseFeatures(source / "rescored" / "features")
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
    matched_features = (
        {
            r["id"]: extractor.extract(
                r["text"], tokens=True, embeddings=True, allow_download=allow_download
            )
            for r in matched["references"]
        }
        if matched
        else {}
    )
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
        card["reference_tracks"] = {"broad": card["prose_profile"]}
        if matched:
            assignment = matched["assignments"][result["scenario_id"]]
            profile = score_prose(
                card,
                scenarios[result["scenario_id"]],
                result,
                extractor,
                references=[matched_features[key] for key in assignment["reference_ids"]],
                sigma=sigma,
                allow_download=allow_download,
            )
            profile["reference_selection"] = assignment
            profile["reference_manifest_hash"] = fingerprint(matched)
            profile["bandwidth_policy"] = "Same frozen broad-reference bandwidth for both tracks"
            card["reference_tracks"]["matched"] = profile
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
    if matched:
        matched_pool = prose_profile(
            texts, feats, references=list(matched_features.values()), sigma=sigma
        )
        matched_pool["limitation"] = (
            "Pooled descriptive comparison across task-selected references; not stratified MMD. "
            "References share authors/works; fantasy style match is partial."
        )
        matched_pool["reference_manifest_hash"] = fingerprint(matched)
        save_json(destination / "pooled-matched-exploratory.json", matched_pool)
        lines += [
            "",
            "## Both reference tracks",
            "",
            "Same pinned features and broad-reference RBF bandwidth in both tracks.",
            "Task-selected references are exploratory; F5 style match is partial.",
            "",
            "| Case | Track | Unigram L2 | Bigram L2 | Trigram L2 | MMD² status |",
            "|---|---|---|---|---|---|",
        ]
        for card in cards:
            for track, profile in card["reference_tracks"].items():
                d1, d2 = profile["metrics"]["D1"], profile["metrics"]["D2"]
                values = d1["value"] or {str(i): d1["status"] for i in (1, 2, 3)}
                row = " | ".join(str(values[str(i)]) for i in (1, 2, 3))
                lines.append(f"| {card['scenario_id']} | {track} | {row} | {d2['status']} |")
        lines += [
            "",
            "[Matched references and limitations](matched-reference-selection.json) · "
            "[Pooled matched-reference diagnostic](pooled-matched-exploratory.json)",
        ]
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
