"""`wa-bench report`: a run's metrics per group, with gates, refusals and baseline deltas."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from writing_assistant.cli import ERRORS_FILE, twins_of
from writing_assistant.episode.store import RUN_FILE, load_episodes, matches, read_jsonl
from writing_assistant.llm.client import LLMClient
from writing_assistant.suites import SUITES, mean

# The five metrics that get a familiarity gap column (benchmark.md, "The Corpus").
FAMILIARITY_GAP = ("discriminator_fool_rate", "verifier_match", "pairwise_craft_win", "score_100_endings", "restraint_score")


def _summarise(run_dir: Path, profile: str, keys: list[str], where: str | None) -> tuple[dict, dict]:
    """(run.json, groups): groups maps a key (the episode's values for `keys`) to
    {metric: [values]} plus counts. Twins are baselines inside their parent's net_* and are
    left out; `condition` is matched and grouped on like a row field."""
    info = json.loads((run_dir / RUN_FILE).read_text())
    loaded = load_episodes(run_dir)
    twins = {t.ep_id for t in twins_of(loaded, SUITES[info["suite"]]).values()}
    episodes = {ep.ep_id: ep for ep in loaded if ep.ep_id not in twins}
    scores = {r["ep_id"]: r for r in read_jsonl(run_dir / f"scores-{profile}.jsonl")}
    groups: dict[tuple, dict[str, Any]] = defaultdict(lambda: {"values": defaultdict(list), "unflagged": defaultdict(list),
                                                              "familiarity": defaultdict(list), "counts": Counter()})
    for ep in episodes.values():
        row = ep.meta.get("row", {}) | {"condition": ep.condition}
        if not matches(row, where):
            continue
        g = groups[tuple(str(row.get(k)) for k in keys)]
        g["counts"][f"status:{ep.status}"] += 1
        if ep.failure_reason:
            g["counts"][f"reason:{ep.failure_reason[:60]}"] += 1
        sc = scores.get(ep.ep_id)
        if sc is None:
            g["counts"]["unscored:missing"] += 1
            continue
        if "error" in sc:
            g["counts"]["unscored:scorer_error"] += 1
            continue
        metrics = sc.get("metrics", {})
        flagged = metrics.get("overlap_flagged", 0.0) > 0
        for name, value in metrics.items():
            if name.startswith("n_"):
                g["counts"][name] += value
            elif not (isinstance(value, float) and math.isnan(value)):
                g["values"][name].append(value)
                if not flagged:
                    g["unflagged"][name].append(value)
                if name in FAMILIARITY_GAP and row.get("familiarity") is not None:
                    g["familiarity"][name].append((float(row["familiarity"]), value))
    return info, groups


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def _guardrails_line(path: Path) -> str:
    g = json.loads(path.read_text())
    scores = ", ".join(f"{k} {g[k]}" for k in ("ifeval", "humaneval_plus", "bfcl"))
    return f"guardrails: {scores} ({g['endpoint']}, {g['model']}, {g['date'][:10]}, {g['image']})"


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    if not (run_dir / RUN_FILE).exists() and (run_dir / "guardrails.json").exists():
        print(_guardrails_line(run_dir / "guardrails.json"))  # a `wa-bench guardrails` directory
        return 0
    group_by = [g for g in (args.group_by or "").split(",") if g]
    # Conditions are arms (full_view and pov_view measure different things): never averaged.
    keys = group_by + ([] if "condition" in group_by else ["condition"])
    info, groups = _summarise(run_dir, args.judges, keys, args.where)
    suite = SUITES[info["suite"]]
    base = _summarise(Path(args.baseline), args.judges, keys, args.where)[1] if args.baseline else None
    passed = set(LLMClient(args.models).config.get("gates", {}).get("passed", []))
    codes = sorted({i["code"] for i in info.get("invocations", [])} or {info.get("code", "unknown")})
    print(f"run {run_dir.name}: suite {info['suite']}, endpoint {info['endpoint']} ({info['model']}), "
          f"driver {info['driver']}{' (direct: reported apart from harness runs)' if info.get('reported_apart') else ''}, "
          f"split {info['split']}, played by code {', '.join(codes)}, judges {args.judges}")
    scoring = info.get("scoring", {}).get(args.judges, {})
    if "code" not in scoring:
        print("scores written before the scoring code was stamped (before the artefact and twin rules): "
              f"re-score with `wa-bench score --run {run_dir} --judges {args.judges}`")
    if spent := info.get("spent_usd"):
        print(f"spent: ${sum(spent.values()):.4f} ({', '.join(f'{k} ${v:.4f}' for k, v in spent.items())})")
    if (run_dir / "guardrails.json").exists():
        print(_guardrails_line(run_dir / "guardrails.json"))
    errors = read_jsonl(run_dir / ERRORS_FILE)
    if errors:
        print(f"task errors: {len(errors)} (see {ERRORS_FILE})")
    for key in sorted(groups):
        g = groups[key]
        label = ", ".join(f"{k}={v}" for k, v in zip(keys, key, strict=True))
        print(f"\n== {label}: " + ", ".join(f"{k} {int(v)}" for k, v in sorted(g["counts"].items()) if not k.startswith("n_")))
        # Episodes whose outcome metrics are absent because the artefact they read is missing.
        for prefix, what in (("n_no_output.", "no artefact"), ("n_twin_no_output.", "twin had no artefact (no net_*)")):
            if missing := {k.removeprefix(prefix): int(v) for k, v in sorted(g["counts"].items()) if k.startswith(prefix)}:
                print(f"  {what}: " + ", ".join(f"{reason} {n}" for reason, n in missing.items()))
        print(f"  {'metric':40} {'mean':>8} {'n':>4} {'w/o flag':>8} {'refused':>8}  notes")
        for name in sorted(g["values"]):
            vals = g["values"][name]
            judged = g["counts"].get(f"n_judged.{name}", 0)
            unscored = {k.rsplit(".", 1)[1]: v for k, v in g["counts"].items() if k.startswith(f"n_unscored.{name}.")}
            total = judged + sum(unscored.values())
            refused = unscored.get("refusal", 0) / total if total else None
            notes = []
            use = suite.judged_by.get(name)
            if use and f"{args.judges}.{use}" not in passed:
                notes.append(f"ungated ({use})")
            if other := {k: int(v) for k, v in unscored.items() if k != "refusal"}:
                notes.append(f"unscored {other}")
            if base is not None:
                b = mean(base.get(key, {}).get("values", {}).get(name, []))
                notes.append(f"Δ {_fmt(mean(vals) - b) if b is not None else '-'} vs baseline")
            fam = sorted(g["familiarity"].get(name, []))
            if len(fam) >= 3:
                third = len(fam) // 3
                low, high = mean([v for _, v in fam[:third]]), mean([v for _, v in fam[-third:]])
                notes.append(f"low-fam {_fmt(low)}, gap {_fmt(high - low)}")
            print(f"  {name:40} {_fmt(mean(vals)):>8} {len(vals):>4} {_fmt(mean(g['unflagged'].get(name, []))):>8} "
                  f"{_fmt(refused) if total else '':>8}  {'; '.join(notes)}")
        # metrics every item of which went unscored: absent above, never 0.0
        for k in sorted(g["counts"]):
            if k.startswith("n_unscored.") and k.split(".")[1] not in g["values"]:
                print(f"  {k.split('.')[1]:40} {'absent':>8}    0 {'':>8} {'':>8}  unscored: {k.split('.')[2]} x{int(g['counts'][k])}")
    return 0
