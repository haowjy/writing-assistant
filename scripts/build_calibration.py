"""Judge-calibration files for `label_ui.py`, drawn from the judgments saved runs stored.

    uv run python scripts/build_calibration.py sample --use fact --runs runs/a runs/b --n 200 --out FILE [--build-checks]
    uv run python scripts/build_calibration.py sample --use author_verify --runs runs/c --n 60 --out FILE
    uv run python scripts/build_calibration.py add-judge --file FILE --endpoint <models.toml endpoint>

`sample` reads the judge items each run's store holds under the profile (only items the judge answered:
an unscored item says nothing about agreement) and drops repeats of the same input. It
splits `--n` evenly between the judge's answers (yes/no, or P(a) above/below 0.5), each
answer's part evenly between metrics, and takes each metric's items round-robin over
(subject, template or author); a group smaller than its share gives the spare to the
others. So no one answer, metric, model or template fills the file. `--build-checks` (fact only) adds the build-time checks
of `scripts/build_seed_tasks.py`: per SRD world, does any event after the hidden one state
the secret? They are replayed from the judge cache with no model call, as their own group.

Each line is a `label_ui.py` item plus `answers` ({judge model: answer}; `model` is the
first judge's), `judge` (its stamp) and `source` (where the item came from). `add-judge`
asks another endpoint the same items and adds its answer under its model name, so
`label_ui.py --agreement --judge <model>` scores it on the same labels. Label files hold
book and model text: they live under `$WA_DATA/calibration/`, never in git.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from writing_assistant.cli import load_dotenv
from writing_assistant.episode.segments import closed_episodes, read_run
from writing_assistant.episode.store import read_jsonl
from writing_assistant.judges import judges_from
from writing_assistant.judges.extract import FACT
from writing_assistant.llm.client import LLMClient
from writing_assistant.suites.world import tasks_1a, worlds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_seed_tasks import CACHE, _h  # noqa: E402  (the build's judge cache and its key)

SEED = 20261125


def _id(use: str, item: dict) -> str:
    keys = ("question", "evidence") if use == "fact" else ("question", "a", "b", "context")
    return f"{use}:" + hashlib.sha256(json.dumps([item.get(k) for k in keys]).encode()).hexdigest()[:16]


def _stratum(ep_id: str, row: dict) -> str:
    """The template (1a) or author (3) an episode's row came from."""
    return str(row.get("template") or row.get("author") or ep_id.split("~")[0].split(":")[2])


def run_items(run: Path, use: str, profile: str) -> list[dict]:
    info = json.loads((run / "run.json").read_text())
    episodes = closed_episodes(run)
    rows = {ep.ep_id: ep.meta.get("row", {}) for ep in episodes}
    # Rollouts are many plays of one checkpoint: sampled with a run's task episodes they would
    # fill the file with one task.
    rollouts = {ep.ep_id for ep in episodes if ep.rollout_of}
    out = []
    for j in read_run(run, lambda store: store.judgments(profile), []):
        if j["judge"]["use"] != use or j["model"] is None or j["scored"] in rollouts:
            continue
        item = {k: j[k] for k in ("question", "evidence", "a", "b", "context") if k in j}
        out.append(item | {
            "model": j["model"], "answers": {j["judge"]["model"]: j["model"]}, "judge": j["judge"],
            "source": {"run": run.name, "subject": info["endpoint"], "ep_id": j["ep_id"], "metric": j["metric"],
                       "kind": rows.get(j["ep_id"], {}).get("kind"),
                       "stratum": _stratum(j["ep_id"], rows.get(j["ep_id"], {}))}})
    return out


def build_checks(models: Path, profile: str) -> list[dict]:
    """The build's later-event checks, replayed from its judge cache: (probe, the events after
    the hidden one) and the cached answer. A check missing from the cache is left out."""
    cache = {r["key"]: r["value"] for r in read_jsonl(CACHE)}
    stamp = judges_from(LLMClient(models), profile).fact.stamp
    seen: list[dict] = []

    async def fact(question: str, evidence: str) -> bool:
        key = _h(stamp["model"], _h(FACT.version, question, evidence))
        if key in cache:
            seen.append({"question": question, "evidence": evidence, "model": cache[key]})
        return bool(cache.get(key, False))

    async def collect_all() -> None:
        for template in worlds.TEMPLATES:
            for i in range(2):
                w = worlds.make_world(template, i)
                log = await worlds.pre_played_log(w)
                hidden = next(n for n, e in enumerate(log) if e["data"].get("text") == w["secret"])
                later = tasks_1a.render_log([SimpleNamespace(kind=e["kind"], data=e["data"]) for e in log[hidden + 1:]], "the party")
                before = len(seen)
                for q in w["hidden_probes"]:
                    await fact(q, later)
                for s in seen[before:]:
                    s["source"] = {"run": "build", "subject": "build", "ep_id": f"{template['id']}:{i}",
                                   "metric": "build_later_event", "kind": "pov_write", "stratum": template["id"]}

    asyncio.run(collect_all())
    return [s | {"answers": {stamp["model"]: s["model"]}, "judge": stamp} for s in seen]


def answer_class(model: Any) -> str:
    return ("yes" if model else "no") if isinstance(model, bool) else ("a" if model > 0.5 else "b" if model < 0.5 else "tie")


def _share(sizes: dict[Any, int], n: int) -> dict[Any, int]:
    """`n` split as evenly as the groups' sizes allow, the spare of a small group shared out again."""
    share = dict.fromkeys(sizes, 0)
    left = n
    while left and any(share[g] < sizes[g] for g in sizes):
        open_ = [g for g in sorted(sizes, key=str) if share[g] < sizes[g]]
        each = max(1, left // len(open_))
        for g in open_:
            take = min(each, sizes[g] - share[g], left)
            share[g] += take
            left -= take
    return share


def sample(items: list[dict], n: int) -> list[dict]:
    """Half per answer, then an even share per metric inside each answer, then round-robin
    over (subject, template or author) inside each metric."""
    rng = random.Random(SEED)
    groups: dict[str, dict[str, dict[tuple, list]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for it in items:
        src = it["source"]
        groups[answer_class(it["model"])][src["metric"]][(src["subject"], src["stratum"])].append(it)
    queues: dict[str, dict[str, list]] = defaultdict(dict)
    for answer, metrics in sorted(groups.items()):
        for metric, buckets in sorted(metrics.items()):
            order = sorted(buckets)
            rng.shuffle(order)
            for b in order:
                rng.shuffle(buckets[b])
            queues[answer][metric] = [buckets[b][i] for i in range(max(map(len, buckets.values())))
                                      for b in order if i < len(buckets[b])]
    picked = []
    per_answer = _share({a: sum(map(len, ms.values())) for a, ms in queues.items()}, n)
    for answer, ms in queues.items():
        per_metric = _share({m: len(q) for m, q in ms.items()}, per_answer[answer])
        picked += [it for m, q in ms.items() for it in q[: per_metric[m]]]
    rng.shuffle(picked)
    return picked


def cmd_sample(args: argparse.Namespace) -> None:
    items = [it for run in args.runs for it in run_items(Path(run), args.use, args.profile)]
    if args.build_checks:
        items += build_checks(args.models, args.profile)
    unique: dict[str, dict] = {}
    for it in items:
        unique.setdefault(_id(args.use, it), it)
    picked = sample([{"id": i} | it for i, it in unique.items()], args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        sys.exit(f"{args.out} exists; it may hold labels. Pick another --out.")
    args.out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in picked), encoding="utf-8")
    print(f"{args.out}: {len(picked)} of {len(unique)} distinct judged items")
    for key in ("metric", "subject", "kind"):
        counts: dict[str, int] = defaultdict(int)
        for it in picked:
            counts[str(it["source"][key])] += 1
        print(f"  by {key}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    answers: dict[str, int] = defaultdict(int)
    for it in picked:
        answers[answer_class(it["model"])] += 1
    print("  judge answers: " + ", ".join(f"{k} {v}" for k, v in sorted(answers.items())))
    print(f"  strata (template/author): {len({it['source']['stratum'] for it in picked})}")


async def add_judge(path: Path, models: Path, endpoint: str, use: str) -> None:
    client = LLMClient(models)
    client.config["judges"]["_calibration"] = {use: endpoint}
    judge = getattr(judges_from(client, "_calibration"), use)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sem = asyncio.Semaphore(client.endpoints[endpoint].get("concurrency", 4))

    async def one(row: dict) -> None:
        args = (row["question"], row["evidence"]) if use == "fact" else (row["a"], row["b"], row["context"])
        async with sem:
            row.setdefault("answers", {})[judge.stamp["model"]] = (await judge.judge(*args)).value

    try:
        await asyncio.gather(*(one(r) for r in rows))
    finally:
        await client.aclose()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)  # labels made meanwhile in label_ui would be lost: do not run both at once
    print(f"{path}: answers from {judge.stamp['model']} added to {len(rows)} items")


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", type=Path, default=ROOT / "models.toml")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--use", required=True, choices=["fact", "author_verify", "edit_pair", "craft_pair", "secret_2afc"])
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--profile", default="eval")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--build-checks", action="store_true", help="fact: add the build's later-event checks")
    a = sub.add_parser("add-judge")
    a.add_argument("--file", type=Path, required=True)
    a.add_argument("--endpoint", required=True)
    a.add_argument("--use", required=True)
    args = p.parse_args()
    if args.cmd == "sample":
        cmd_sample(args)
    else:
        asyncio.run(add_judge(args.file, args.models, args.endpoint, args.use))


if __name__ == "__main__":
    main()
