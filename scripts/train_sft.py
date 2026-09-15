"""Editable SFT experiment. The default invocation only inspects readiness."""

from dataclasses import asdict
from pathlib import Path

from writing_agent.catalog import save_json
from writing_agent.data import read_records
from writing_agent.development import author_development
from writing_agent.training import SFTSettings, encode_trajectory, prepare_sft, train_sft

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = SFTSettings()
SOURCE = ROOT / "data/training/sft-v1.jsonl"
PREPARED = ROOT / "data/processed/sft-v1"
OUTPUT = ROOT / "runs/sft-e2b-feasibility"
AUDIT = ROOT / "work/sft/preparation-audit"


def inspect():
    records = read_records(SOURCE) if SOURCE.exists() else []
    return {
        "settings": asdict(SETTINGS),
        "source": str(SOURCE),
        "accepted_training_examples": sum(
            r["split"] == "train" and r["review_status"] == "accepted" for r in records
        ),
        "prepared": (PREPARED / "manifest.json").exists(),
        "training_enabled": False,
        "next": "Curate independent training trajectories; inspect masks; approve bounded test.",
    }


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        SETTINGS.model_id,
        revision=SETTINGS.revision,
        local_files_only=True,
        trust_remote_code=False,
    )


def audit_fixtures():
    """Tokenize five pending synthetic probes, without weights, grading, or training."""
    tok = tokenizer()
    records = read_records(ROOT / "data/fixtures/sft-mask-probes.jsonl")
    rows = [encode_trajectory(r, tok, max_length=SETTINGS.max_length) for r in records]
    save_json(AUDIT / "masks.json", rows)
    lines = [
        "# Native SFT mask audit",
        "",
        "Synthetic format probes only; no training executed.",
        "",
    ]
    for row in rows:
        lines += [
            f"## {row['id']}",
            "",
            f"{len(row['input_ids'])} input tokens; {row['supervised_tokens']} loss tokens.",
            "",
            "Rendered conversation:",
            "",
            "```text",
            row["rendered"],
            "```",
            "",
            "Supervised text (all other tokens have label -100):",
            "",
            "```text",
            row["supervised_text"],
            "```",
            "",
        ]
    (AUDIT / "review.md").write_text("\n".join(lines))
    return {"examples": len(rows), "review": str(AUDIT / "review.md")}


def prepare():
    _, scenarios = author_development(ROOT / "data/scenarios/worlds.json")
    excluded = {g for s in scenarios for g in s.get("source_groups", [])}
    return prepare_sft(
        read_records(SOURCE),
        tokenizer(),
        PREPARED,
        settings=SETTINGS,
        excluded_source_groups=excluded,
    )


def train(*, execute=False, resume_from_checkpoint=None):
    return train_sft(
        PREPARED,
        OUTPUT,
        settings=SETTINGS,
        execute=execute,
        resume_from_checkpoint=resume_from_checkpoint,
    )


if __name__ == "__main__":
    print(inspect())
