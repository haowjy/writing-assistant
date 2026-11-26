"""`wa-bench report`: a run's metrics per group, with gates, refusals and baseline deltas.

A metric whose judge has not passed its human-label gate (`[gates] passed` in models.toml,
`Suite.judged_by`) prints marked "ungated" in a run's own report and is left out of a
`--baseline` comparison (DECISIONS #12): looking at it calibrates the gate, comparing
models on it would not be a result. `--baseline` is also refused when a cast name other than
`subject` that both runs' episodes used was played by different models: the two subjects
then met different partners, and the difference is not the subjects'."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from writing_assistant.cli import episode_row
from writing_assistant.episode.segments import closed_episodes, open_episodes, read_run
from writing_assistant.episode.store import RUN_FILE, matches, read_jsonl
from writing_assistant.llm.client import LLMClient
from writing_assistant.run import ERRORS_FILE
from writing_assistant.suites import SUBJECT, SUITES, mean

# The five metrics that get a familiarity gap column (benchmark.md, "The Corpus").
FAMILIARITY_GAP = ("discriminator_fool_rate", "verifier_match", "pairwise_craft_win", "score_100_endings", "restraint_score")


def _summarise(run_dir: Path, profile: str, keys: list[str], where: str | None,
               rollouts: bool = False) -> tuple[dict, dict, Counter]:
    """(run.json, groups, scoring codes): groups maps a key (the episode's values for `keys`)
    to {metric: [values]} plus counts; the codes count the score rows read per scoring code.
    The run's task episodes, or with
    `rollouts` its rollouts instead: the two are never summed. Twins are baselines inside
    their parent's net_* and are left out; `condition` is matched and grouped on like a row
    field, and so is a rollout's `rollout_of`."""
    info = json.loads((run_dir / RUN_FILE).read_text())
    episodes = [ep for ep in closed_episodes(run_dir) if ep.baseline_of is None and (ep.rollout_of is not None) == rollouts]
    scores = read_run(run_dir, lambda store: store.scores(profile), {})
    groups: dict[tuple, dict[str, Any]] = defaultdict(lambda: {"values": defaultdict(list), "unflagged": defaultdict(list),
                                                              "familiarity": defaultdict(list), "counts": Counter(),
                                                              "judges": defaultdict(set)})
    codes: Counter = Counter()
    for ep in episodes:
        row = episode_row(ep)
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
        codes[sc["code"]] += 1
        for stamp in sc.get("judges", []):
            g["judges"][stamp["use"]].add(f"{stamp['model']} ({stamp['endpoint']}, {stamp['prompt']})")
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
    return info, groups, codes


def other_cast(run_dir: Path) -> dict[str, set[str]]:
    """Cast name -> who played it (`endpoint (model)`, as each episode was stamped) over the
    run's closed episodes, for the names their seats use other than the subject's."""
    played: dict[str, set[str]] = defaultdict(set)
    for ep in closed_episodes(run_dir):
        for name in set(ep.actors.values()) - {SUBJECT}:
            if stamp := ep.meta.get("cast", {}).get(name):
                played[name].add(f"{stamp['endpoint']} ({stamp['model']})")
    return played


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def guardrails_line(path: Path) -> str:
    g = json.loads(path.read_text())
    scores = ", ".join(f"{k} {g[k]}" for k in ("ifeval", "humaneval_plus", "bfcl"))
    return f"guardrails: {scores} ({g['endpoint']}, {g['model']}, {g['date'][:10]}, {g['image']})"


def build_report(run_dir: Path, judges: str, keys: list[str], where: str | None, baseline: Path | None,
                 passed: set[str]) -> dict[str, Any]:
    """A run's report as data: `header` lines, then one entry per group with its counts, the
    episodes missing an artefact, the metric rows (numbers unformatted, `refused` None when no
    item was judged), the ungated metrics left out of a baseline comparison and the metrics
    every item of which went unscored. `groups` are the run's task episodes and its headline
    numbers; `rollouts` holds the same entries for the rollouts stored beside them, grouped
    first by the checkpoint they were forked from and the group they were made in, each
    with `group`: its size, how many of this checkpoint's members are closed with a final
    row, and whether the group is settled (`Store.groups`). An entry's `mixed` names the
    judge uses whose rows were scored by more than one judge: a profile is a name in one
    machine's models.toml, and the metrics such a use judges are then no one judge's.
    `cmd_report` prints it; `wa-bench view` renders it."""
    info, groups, scored_by = _summarise(run_dir, judges, keys, where)
    suite = SUITES[info["suite"]]
    codes = sorted({i["code"] for i in info.get("invocations", [])} or {info.get("code", "unknown")})
    header = [f"run {run_dir.name}: suite {info['suite']}, endpoint {info['endpoint']} ({info['model']}), "
              f"driver {info['driver']}{' (direct: reported apart from harness runs)' if info.get('reported_apart') else ''}, "
              f"split {info['split']}, played by code {', '.join(codes)}, judges {judges}"]
    if still_open := len(open_episodes(run_dir)):
        header.append(f"{'1 episode is' if still_open == 1 else f'{still_open} episodes are'} open and in no number below: "
                      "a run is still playing, or was killed. Running the same `wa-bench run` command again continues "
                      "each from its last saved turn")
    if scored_by:
        header.append(f"scored by code {', '.join(sorted(scored_by))}")
    if spent := info.get("spent_usd"):
        header.append(f"spent: ${sum(spent.values()):.4f} ({', '.join(f'{k} ${v:.4f}' for k, v in spent.items())}); "
                      "counts only the chat-call cost a host reports (DeepInfra's estimated_cost): embedding calls "
                      "and hosts that report no cost add nothing")
    if served := info.get("served"):
        header.append("served: " + "; ".join(f"{k} {' | '.join(v)}" for k, v in served.items()))
    if (run_dir / "guardrails.json").exists():
        header.append(guardrails_line(run_dir / "guardrails.json"))
    if errors := read_jsonl(run_dir / ERRORS_FILE):
        header.append(f"task errors: {len(errors)} (see {ERRORS_FILE})")
    by = ["rollout_of", "group", *keys]
    # A group is one run's, so the baseline's rollouts of a checkpoint are matched whatever group they are in.
    rollouts = _entries(suite, judges, by, _summarise(run_dir, judges, by, where, rollouts=True)[1], passed,
                        _summarise(baseline, judges, ["rollout_of", *keys], where, rollouts=True)[1] if baseline else None,
                        base_keys=["rollout_of", *keys])
    stored = read_run(run_dir, lambda store: store.groups(judges), {})
    for entry in rollouts:
        g = stored[entry["key"]["group"]]
        entry["group"] = {"size": g["size"], "final": g["sources"][entry["key"]["rollout_of"].rpartition("@")[0]],
                          "settled": g["settled"]}
    return {"info": info, "header": header, "rollouts": rollouts,
            "groups": _entries(suite, judges, keys, groups, passed,
                               _summarise(baseline, judges, keys, where)[1] if baseline else None)}


def _entries(suite: Any, judges: str, keys: list[str], groups: dict, passed: set[str], base: dict | None,
             base_keys: list[str] | None = None) -> list[dict[str, Any]]:
    """`build_report`'s entry for each of `_summarise`'s groups; `base` is the baseline run's
    groups, keyed by `base_keys` when those are fewer than `keys`."""
    out = []
    for key in sorted(groups):
        g = groups[key]
        # Episodes whose outcome metrics are absent because the artefact they read is missing.
        missing = {what: m for prefix, what in (("n_no_output.", "no artefact"),
                                                ("n_twin_no_output.", "twin had no artefact (no net_*)"))
                   if (m := {k.removeprefix(prefix): int(v) for k, v in sorted(g["counts"].items()) if k.startswith(prefix)})}
        rows, left_out = [], []
        for name in sorted(g["values"]):
            use = suite.judged_by.get(name)
            ungated = bool(use) and f"{judges}.{use}" not in passed
            if ungated and base is not None:  # an ungated number never enters a comparison
                left_out.append(name)
                continue
            vals = g["values"][name]
            judged = g["counts"].get(f"n_judged.{name}", 0)
            unscored = {k.rsplit(".", 1)[1]: v for k, v in g["counts"].items() if k.startswith(f"n_unscored.{name}.")}
            total = judged + sum(unscored.values())
            notes = [f"ungated ({use})"] if ungated else []
            if other := {k: int(v) for k, v in unscored.items() if k != "refusal"}:
                notes.append(f"unscored {other}")
            if base is not None:
                theirs = tuple(v for k, v in zip(keys, key, strict=True) if k in (base_keys or keys))
                b = mean(base.get(theirs, {}).get("values", {}).get(name, []))
                notes.append(f"Δ {_fmt(mean(vals) - b) if b is not None else '-'} vs baseline")
            fam = sorted(g["familiarity"].get(name, []))
            if len(fam) >= 3:
                third = len(fam) // 3
                low, high = mean([v for _, v in fam[:third]]), mean([v for _, v in fam[-third:]])
                notes.append(f"low-fam {_fmt(low)}, gap {_fmt(high - low)}")
            rows.append({"metric": name, "mean": mean(vals), "n": len(vals), "unflagged": mean(g["unflagged"].get(name, [])),
                         "refused": unscored.get("refusal", 0) / total if total else None, "ungated": ungated, "notes": notes})
        # metrics every item of which went unscored: absent from the rows, never 0.0
        absent = [{"metric": k.split(".")[1], "reason": k.split(".")[2], "n": int(g["counts"][k])}
                  for k in sorted(g["counts"]) if k.startswith("n_unscored.") and k.split(".")[1] not in g["values"]]
        out.append({"key": dict(zip(keys, key, strict=True)),
                    "counts": {k: int(v) for k, v in sorted(g["counts"].items()) if not k.startswith("n_")},
                    "missing": missing, "rows": rows, "left_out": left_out, "absent": absent,
                    "mixed": {use: sorted(by) for use, by in sorted(g["judges"].items()) if len(by) > 1}})
    return out


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    if not (run_dir / RUN_FILE).exists() and (run_dir / "guardrails.json").exists():
        print(guardrails_line(run_dir / "guardrails.json"))  # a `wa-bench guardrails` directory
        return 0
    group_by = [g for g in (args.group_by or "").split(",") if g]
    # Conditions are arms (full_view and pov_view measure different things): never averaged.
    keys = group_by + ([] if "condition" in group_by else ["condition"])
    passed = set(LLMClient(args.models).config.get("gates", {}).get("passed", []))
    if args.baseline:
        ours, theirs = other_cast(run_dir), other_cast(Path(args.baseline))
        if differ := {name: (ours[name], theirs[name]) for name in ours.keys() & theirs.keys() if ours[name] != theirs[name]}:
            sys.exit(f"{run_dir} and {args.baseline} were played with different casts: " + "; ".join(
                f"{name} is {', '.join(sorted(a))} here and {', '.join(sorted(b))} there" for name, (a, b) in sorted(differ.items()))
                + ". The other seats differ, so the runs do not compare their subjects")
    report = build_report(run_dir, args.judges, keys, args.where, Path(args.baseline) if args.baseline else None, passed)
    print("\n".join(report["header"]))
    for g in report["groups"]:
        _print_group(g)
    if report["rollouts"]:
        print("\nrollouts, by the checkpoint each was forked from and the group it was made in; none of them is in "
              "the groups above")
        for g in report["rollouts"]:
            _print_group(g)
    return 0


def _print_group(g: dict[str, Any]) -> None:
    label = ", ".join(f"{k}={v}" for k, v in g["key"].items())
    print(f"\n== {label}: " + ", ".join(f"{k} {v}" for k, v in g["counts"].items()))
    if group := g.get("group"):
        print(f"  group of {group['size']} per episode: {group['final']} closed with a final row here; "
              + ("settled" if group["settled"] else "not settled"))
    for use, by in g["mixed"].items():
        print(f"  rows scored by {len(by)} different {use} judges ({'; '.join(by)}): the metrics it judges mix them")
    for what, missing in g["missing"].items():
        print(f"  {what}: " + ", ".join(f"{reason} {n}" for reason, n in missing.items()))
    print(f"  {'metric':40} {'mean':>8} {'n':>4} {'w/o flag':>8} {'refused':>8}  notes")
    for r in g["rows"]:
        refused = "" if r["refused"] is None else _fmt(r["refused"])
        print(f"  {r['metric']:40} {_fmt(r['mean']):>8} {r['n']:>4} {_fmt(r['unflagged']):>8} {refused:>8}  "
              f"{'; '.join(r['notes'])}")
    if g["left_out"]:
        print(f"  left out of the comparison, judge not yet gated: {', '.join(g['left_out'])}")
    for a in g["absent"]:
        print(f"  {a['metric']:40} {'absent':>8}    0 {'':>8} {'':>8}  unscored: {a['reason']} x{a['n']}")
