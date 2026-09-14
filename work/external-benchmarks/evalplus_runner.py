"""Run only the staged HumanEval+ selection inside the isolated CPU container."""

import json
from pathlib import Path

from evalplus.data import get_human_eval_plus
from evalplus.evaluate import evaluate
from evalplus.sanitize import sanitize

problems = get_human_eval_plus()
samples = [json.loads(line) for line in Path("/input/samples.jsonl").read_text().splitlines()]
assert {row["task_id"] for row in samples} == set(problems)
for row in samples:
    row["solution"] = sanitize(row["solution"], entrypoint=problems[row["task_id"]]["entry_point"])
Path("/output/samples-sanitized.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in samples)
)
evaluate(dataset="humaneval", samples="/output/samples-sanitized.jsonl", parallel=2)
