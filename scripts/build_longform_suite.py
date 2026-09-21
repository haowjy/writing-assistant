"""Build and freeze the held-out long-form benchmark.

Run from the repository root:

    python scripts/build_longform_suite.py

The build refuses to complete if a case cites a fact its own supplied text lacks, or if
any benchmark work is already in use by training or by the development suite.
"""

import json
import sys
from pathlib import Path

from writing_agent.catalog import save_json
from writing_agent.longform_suite import (
    build_release,
    claimed_hashes,
    documented_sampling,
    freeze,
    holdout_audit,
)
from writing_agent.suite import compile_scenarios

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "data/scenarios/longform-v1.json"
RAW = ROOT / "data/raw/research"
DESTINATION = ROOT / "data/processed/longform-v1"
TRAINING = ROOT / "data/processed/sft-starter-v1/catalog.json"
DEVELOPMENT = ROOT / "data/processed/custom-eval/catalog.json"
DEVELOPMENT_MANIFEST = ROOT / "data/processed/custom-eval/manifest.json"


def development_ids() -> set[str]:
    """Only the sources the development cases actually reference are claimed."""
    if not DEVELOPMENT_MANIFEST.exists():
        return set()
    manifest = json.loads(DEVELOPMENT_MANIFEST.read_text())
    return {
        source
        for scenario in manifest.get("scenarios", [])
        for source in scenario.get("source_ids", [])
    }


def main() -> int:
    catalog, scenarios = build_release(SPEC, RAW)
    audit = holdout_audit(
        catalog,
        claimed={
            "training": claimed_hashes(TRAINING),
            "development_used": claimed_hashes(DEVELOPMENT, ids=development_ids()),
        },
    )
    manifest = compile_scenarios(scenarios, catalog, DESTINATION)
    frozen = freeze(manifest, DESTINATION)
    sampling = documented_sampling(SPEC)
    save_json(DESTINATION / "freeze.json", {"holdout": audit, "sampling": sampling, **frozen})
    print(f"{len(scenarios)} cases -> {DESTINATION.relative_to(ROOT)}")
    print(f"holdout: {audit['status']} | claimed {audit['claimed']}")
    print(f"sampling: {sampling['samples_per_case']}/case -> {sampling['plan']}")
    print(f"freeze: {frozen['freeze_hash'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
