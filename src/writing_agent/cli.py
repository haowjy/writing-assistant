import argparse
import json
from pathlib import Path

from writing_agent.data import export_sft, read_records
from writing_agent.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Creative-writing agent research tools")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("eval", help="Run a frozen task suite")
    evaluation.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))
    validation = commands.add_parser("validate-data", help="Validate trajectory JSONL")
    validation.add_argument("path", type=Path)
    export = commands.add_parser("export-sft", help="Export accepted train records for TRL")
    export.add_argument("source", type=Path)
    export.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "eval":
            directory = evaluate(args.config)
            summary = json.loads((directory / "summary.json").read_text())
            print(json.dumps({"run_dir": str(directory), **summary}, indent=2))
            if summary["passed"] != summary["tasks"]:
                raise SystemExit(1)
        elif args.command == "validate-data":
            print(f"Validated {len(read_records(args.path))} records")
        else:
            count = export_sft(args.source, args.destination)
            print(f"Exported {count} accepted training records to {args.destination}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
