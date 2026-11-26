"""local_dev text is read only under the guard and never reaches an endpoint without
`local = true`: the client refuses it before sending, `wa-bench run` refuses a hosted
subject, and a hosted judge called while scoring leaves its item unscored without costing
the episode its other metrics."""

import json
from pathlib import Path

import httpx
import pytest

from writing_assistant import cli
from writing_assistant.episode import store
from writing_assistant.episode.store import LocalOnly, local_only
from writing_assistant.llm.client import LLMClient

MODELS_TOML = Path(__file__).resolve().parents[1] / "models.toml"
ROW = {"task_id": "ld-1", "suite": "3", "kind": "critique_edit", "split": "dev", "work_ids": [], "payload": {
    "passage": {"work_id": "local:amber", "chapter": 1, "text": "One.\n\nTwo two.\n\nThree."},
    "objection": "Paragraph 2 is flat.", "scope": [2], "path": "passage.md"}}


async def test_a_hosted_endpoint_refuses_local_dev_text():
    client = LLMClient(MODELS_TOML)
    with local_only(), pytest.raises(LocalOnly):
        await client.chat([{"role": "user", "content": "Ambertwo, chapter 9"}], endpoint="gemma-4-26b",
                          temperature=0, max_tokens=8)
    with local_only(), pytest.raises(LocalOnly):
        await client.embed(["Ambertwo"], endpoint="qwen3-embed")
    assert client._http is None  # refused before a connection was opened


@pytest.fixture
def local_dev_row(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "LOCAL_DEV_DIR", tmp_path / "local_dev")
    monkeypatch.setattr(store, "TASKS_DIR", tmp_path / "tasks")
    (tmp_path / "local_dev" / "tasks").mkdir(parents=True)
    (tmp_path / "local_dev" / "tasks" / "3.jsonl").write_text(json.dumps(ROW) + "\n")
    return tmp_path


def test_local_dev_rows_are_read_only_under_the_guard(local_dev_row):
    with pytest.raises(LocalOnly):
        store.load_local_dev("3")
    with local_only():
        assert [t.task_id for t in store.load_local_dev("3")] == ["ld-1"]


def test_wa_bench_run_refuses_a_hosted_subject_for_local_dev(local_dev_row):
    with pytest.raises(SystemExit, match="local = true"):
        cli.main(["--models", str(MODELS_TOML), "run", "--suite", "3", "--split", "local_dev",
                  "--endpoint", "gemma-4-26b", "--out", str(local_dev_row / "run")])


def test_a_hosted_judge_leaves_a_local_dev_item_unscored_and_the_rest_scored(local_dev_row, monkeypatch):
    sent = []

    def endpoint(request: httpx.Request) -> httpx.Response:  # the local subject, over the real client
        sent.append(request.url.host)
        if len(sent) == 1:
            call = {"id": "c1", "type": "function", "function": {"name": "episode_write", "arguments": json.dumps(
                {"path": "passage.md", "text": "One.\n\nTwo, and the engine coughed.\n\nThree.", "rev": 1})}}
            message, finish = {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": "Sharpened paragraph two."}, "stop"
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}], "usage": {}})

    client = LLMClient(MODELS_TOML)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(endpoint))
    monkeypatch.setattr(cli, "LLMClient", lambda models: client)
    out = local_dev_row / "run"
    cli.main(["--models", str(MODELS_TOML), "run", "--suite", "3", "--split", "local_dev",
              "--endpoint", "qwen3-8b-local", "--out", str(out)])
    [row] = [json.loads(line) for line in (out / "scores-eval.jsonl").read_text().splitlines()]
    assert row["metrics"]["edited"] == 1.0 and row["metrics"]["edit_scope"] == 1.0
    assert row["metrics"]["n_unscored.edit_pair_win.local_only"] == 1.0 and "edit_pair_win" not in row["metrics"]
    assert set(sent) == {"127.0.0.1"}  # the hosted judge was never called
