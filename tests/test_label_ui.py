"""label_ui round trip: serve a file, post labels and an undo, read the JSONL back."""

import json
import threading
import urllib.request

from writing_assistant import label_ui

ITEMS = [
    {"id": "f1", "question": "Does Zoë know?", "evidence": "Zoë read the letter — «ça va».", "model": True},
    {"id": "f2", "question": "Is the door locked?", "evidence": "Nothing says.", "model": None},
    {"id": "p1", "question": "Which is the author?", "context": ["held-out one", "held-out two"],
     "a": "passage a", "b": "passage b", "model": 0.75},
    {"id": "p2", "question": "Which is the author?", "a": "x", "b": "y", "model": 0.25},
]


def test_round_trip(tmp_path):
    path = tmp_path / "cal.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ITEMS), encoding="utf-8")
    server = label_ui.serve(path, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def post(route, body):
        req = urllib.request.Request(url + route, json.dumps(body).encode(), method="POST")
        return json.loads(urllib.request.urlopen(req).read())

    try:
        state = json.loads(urllib.request.urlopen(url + "/state").read())
        assert state["item"]["id"] == "f1" and "model" not in state["item"]
        assert state["item"]["evidence"] == ITEMS[0]["evidence"]
        post("/label", {"id": "f1", "key": "1"})  # yes: agrees with the judge
        state = post("/label", {"id": "f1", "key": "2"})  # double press: refused, f1 stays "yes"
        assert state["item"]["id"] == "f2"
        post("/label", {"id": "f2", "key": "skip"})
        p1 = post("/label", {"id": "f2", "key": "1"})["item"]  # already skipped: refused
        left_is_a = p1["left"] == "passage a"
        post("/label", {"id": "p1", "key": "1" if left_is_a else "2"})  # choose a
        p2 = post("/label", {"id": "p2", "key": "2"})  # then undo it
        assert p2["item"] is None
        assert post("/undo", {})["item"]["id"] == "p2"
    finally:
        server.shutdown()
        server.server_close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r.get("label") for r in rows] == ["yes", "skip", "a", None]
    assert rows[0]["evidence"] == ITEMS[0]["evidence"]
    stats = label_ui.agreement(rows)
    assert stats["agreement"] == (1.0 + 0.75) / 2 and stats["n"] == 2 and stats["skipped"] == 1
