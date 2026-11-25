"""local_dev text never reaches an endpoint without `local = true`: the client refuses it
before sending, and `wa-bench run` plays local_dev rows under that guard."""

import json
from pathlib import Path

import pytest

from writing_assistant import cli
from writing_assistant.episode import store
from writing_assistant.llm.client import LLMClient, LocalOnly, local_only

MODELS_TOML = Path(__file__).resolve().parents[1] / "models.toml"


async def test_a_hosted_endpoint_refuses_local_dev_text():
    client = LLMClient(MODELS_TOML)
    with local_only(), pytest.raises(LocalOnly):
        await client.chat([{"role": "user", "content": "Ambertwo, chapter 9"}], endpoint="gemma-4-26b",
                          temperature=0, max_tokens=8)
    with local_only(), pytest.raises(LocalOnly):
        await client.embed(["Ambertwo"], endpoint="qwen3-embed")
    assert client._http is None  # refused before a connection was opened


def test_wa_bench_run_plays_local_dev_rows_under_the_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "LOCAL_DEV_DIR", tmp_path / "local_dev")
    monkeypatch.setattr(store, "TASKS_DIR", tmp_path / "tasks")
    (tmp_path / "local_dev" / "tasks").mkdir(parents=True)
    row = {"task_id": "ld-1", "suite": "3", "kind": "critique_edit", "split": "dev", "work_ids": [], "payload": {
        "passage": {"work_id": "local:amber", "chapter": 1, "text": "One.\n\nTwo two.\n\nThree."},
        "objection": "Paragraph 2 is flat.", "scope": [2], "path": "passage.md"}}
    (tmp_path / "local_dev" / "tasks" / "3.jsonl").write_text(json.dumps(row) + "\n")
    out = tmp_path / "run"
    cli.main(["--models", str(MODELS_TOML), "run", "--suite", "3", "--split", "local_dev",
              "--endpoint", "gemma-4-26b", "--out", str(out), "--no-score"])
    [error] = [json.loads(line) for line in (out / "errors.jsonl").read_text().splitlines()]
    assert error["task_id"] == "ld-1" and error["error"].startswith("LocalOnly")
    assert not (out / "episodes.jsonl").exists()
