"""Label judge-calibration items in a browser; labels are saved straight back to the JSONL.

    uv run python -m writing_assistant.label_ui --file calibration.jsonl [--port 8765]
    uv run python -m writing_assistant.label_ui --file calibration.jsonl --agreement [--threshold 0.85 --min-labels 200] [--judge MODEL]

Standalone: imports nothing from the package. One file holds one calibrated use. Each line
is one item, either shape:

    {"id": "f1", "question": "...", "evidence": "...", "model": true}                       # yes/no
    {"id": "p1", "question": "...", "context": "..." | ["..."], "a": "...", "b": "...", "model": 0.75}   # A/B

`model` is the judge's answer (bool, or P(a)); null where the judge left the item unscored.
An item may also carry `answers` ({judge model: answer}, the same shape as `model`), so
several judges are scored on one set of labels: `--agreement --judge MODEL` reads
`answers[MODEL]` in place of `model`. The page adds `label` ("yes"/"no"/"a"/"b"/"skip") and `seq` (labelling order, so `u` undoes
the latest label even after a reload). Keys: 1 / 2 answer (yes/no, or left/right), Space
skips, u undoes. A/B sides are shown in an order fixed per id, so a file whose `a` is always
the true answer still labels blind; the model's answer is never shown before labelling.

Agreement is the chance the judge agrees with the label: P(a) or 1 - P(a) for a pair, 1 or 0
for yes/no; skipped items and items the judge left unscored are counted, not averaged.

Safety: one server per file (a lock file refuses a second); requests are served one at a
time, each label is written to a temp file, fsynced and renamed over the JSONL before the
page advances; a label posted for an item that already has one (a second tab, a double
press) is refused and the page moves on.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ANSWERS = {"yes_no": ("yes", "no"), "ab": ("a", "b")}


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("id"), str) or not isinstance(row.get("question"), str):
                raise ValueError(f"{path}:{n}: an item needs a string id and question")
            if ("a" in row) != ("b" in row) or ("a" not in row and "evidence" not in row):
                raise ValueError(f"{path}:{n}: an item has evidence (yes/no) or both a and b (A/B)")
            m = row.get("model")
            if not (m is None or (isinstance(m, bool) if kind(row) == "yes_no" else
                                  isinstance(m, int | float) and not isinstance(m, bool) and 0 <= m <= 1)):
                raise ValueError(f"{path}:{n}: model is a bool for yes/no, P(a) in [0, 1] for A/B, or null")
            self.rows.append(row)
        ids = [r["id"] for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate ids")
        self.by_id = {r["id"]: r for r in self.rows}

    def label(self, item_id: str, key: str) -> bool:
        """Apply key "1", "2" or "skip" to an unlabelled item; False if it already has a label."""
        row = self.by_id[item_id]
        if row.get("label") is not None:
            return False
        if key == "skip":
            row["label"] = "skip"
        else:
            first, second = ANSWERS[kind(row)]
            left_is_first = not swapped(row)
            row["label"] = first if (key == "1") == left_is_first else second
        row["seq"] = 1 + max((r.get("seq") or 0 for r in self.rows), default=0)
        self._write()
        return True

    def undo(self) -> dict | None:
        done = [r for r in self.rows if r.get("seq")]
        if not done:
            return None
        row = max(done, key=lambda r: r["seq"])
        row.pop("label", None)
        row.pop("seq", None)
        self._write()
        return row

    def next_item(self) -> dict | None:
        return next((r for r in self.rows if r.get("label") is None), None)

    def _write(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


def kind(row: dict) -> str:
    return "ab" if "a" in row else "yes_no"


def swapped(row: dict) -> bool:
    return kind(row) == "ab" and hashlib.sha256(row["id"].encode()).digest()[0] % 2 == 1


def agreement(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("label") in ("yes", "no", "a", "b") and r.get("model") is not None]
    credit = [float(r["model"]) if r["label"] in ("yes", "a") else 1.0 - float(r["model"]) for r in scored]
    return {
        "agreement": sum(credit) / len(credit) if credit else None,
        "n": len(scored),
        "skipped": sum(r.get("label") == "skip" for r in rows),
        "model_unscored": sum(r.get("label") in ("yes", "no", "a", "b") and r.get("model") is None for r in rows),
        "labelled": sum(r.get("label") is not None for r in rows),
        "total": len(rows),
    }


def view(row: dict | None) -> dict | None:
    """What the page shows: never `model`, and A/B sides as left/right."""
    if row is None:
        return None
    out = {"id": row["id"], "kind": kind(row), "question": row["question"]}
    if out["kind"] == "yes_no":
        out["evidence"] = row["evidence"]
    else:
        ctx = row.get("context", "")
        out["context"] = "\n\n* * *\n\n".join(ctx) if isinstance(ctx, list) else ctx
        out["left"], out["right"] = (row["b"], row["a"]) if swapped(row) else (row["a"], row["b"])
    return out


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/state":
                self._state(store.next_item())
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/label":
                if body.get("id") not in store.by_id or body.get("key") not in ("1", "2", "skip"):
                    return self._send(400, b"bad label", "text/plain")
                store.label(body["id"], body["key"])
                self._state(store.next_item())
            elif self.path == "/undo":
                undone = store.undo()
                self._state(undone or store.next_item())
            else:
                self._send(404, b"not found", "text/plain")

        def _state(self, item: dict | None) -> None:
            state = {"item": view(item), "stats": agreement(store.rows)}
            self._send(200, json.dumps(state, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _send(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            pass

    return Handler


def serve(path: Path, port: int) -> HTTPServer:
    """A server bound to 127.0.0.1 (port 0 picks a free one), holding the file's lock."""
    lock = open(path.with_name(path.name + ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{path} is open in another label_ui server") from None
    server = HTTPServer(("127.0.0.1", port), make_handler(Store(path)))
    server.lock_file = lock  # released when the server's process exits
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--agreement", action="store_true", help="print agreement for the file and exit")
    ap.add_argument("--threshold", type=float, help="with --agreement: the gate's agreement threshold")
    ap.add_argument("--min-labels", type=int, default=0, help="with --agreement: labels the gate needs")
    ap.add_argument("--judge", help="with --agreement: score answers[JUDGE] instead of model")
    args = ap.parse_args(argv)
    if args.agreement:
        rows = Store(args.file).rows
        if args.judge:
            rows = [r | {"model": r.get("answers", {}).get(args.judge)} for r in rows]
        s = agreement(rows)
        a = "n/a" if s["agreement"] is None else f"{s['agreement']:.3f}"
        print(f"agreement {a} on {s['n']} labels ({s['skipped']} skipped, {s['model_unscored']} unscored by the "
              f"judge, {s['labelled']}/{s['total']} items labelled)")
        if args.threshold is None:
            return 0
        ok = s["agreement"] is not None and s["agreement"] >= args.threshold and s["n"] >= args.min_labels
        print(f"gate {'PASS' if ok else 'FAIL'}: agreement >= {args.threshold} on >= {args.min_labels} labels")
        return 0 if ok else 1
    server = serve(args.file, args.port)
    print(f"labelling {args.file} at http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>label</title>
<style>
  body { font: 16px/1.5 Georgia, serif; margin: 0; color: #1d1d1b; background: #faf8f3; }
  header { font: 13px system-ui, sans-serif; padding: 8px 20px; border-bottom: 1px solid #ddd8cc;
           display: flex; gap: 24px; position: sticky; top: 0; background: #faf8f3; }
  main { max-width: 1100px; margin: 0 auto; padding: 16px 20px 80px; }
  h1 { font-size: 20px; font-weight: normal; margin: 8px 0 16px; }
  .box { white-space: pre-wrap; background: #fff; border: 1px solid #ddd8cc; padding: 12px 14px;
         max-height: 45vh; overflow: auto; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
  .label { font: 12px system-ui, sans-serif; color: #6b675e; margin: 14px 0 4px; text-transform: uppercase; }
  footer { font: 13px system-ui, sans-serif; position: fixed; bottom: 0; left: 0; right: 0; padding: 10px 20px;
           background: #efeadf; border-top: 1px solid #ddd8cc; }
  kbd { border: 1px solid #b9b3a5; border-radius: 3px; padding: 0 5px; background: #fff; }
  #msg { color: #a3391f; margin-left: 16px; }
</style></head><body>
<header><span id="progress"></span><span id="agree"></span></header>
<main id="item"></main>
<footer><span id="keys"></span><span id="msg"></span></footer>
<script>
let item = null, busy = false;
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls;
                                  if (text !== undefined) e.textContent = text; return e; };
function render(state) {
  item = state.item; const s = state.stats;
  document.getElementById("progress").textContent = `${s.labelled}/${s.total} labelled, ${s.skipped} skipped`;
  document.getElementById("agree").textContent = s.agreement === null ? "agreement: no labels yet"
    : `agreement with judge ${s.agreement.toFixed(3)} on ${s.n} (${s.model_unscored} unscored by judge)`;
  const main = document.getElementById("item"); main.replaceChildren();
  const keys = document.getElementById("keys");
  if (!item) { main.append(el("h1", "", "All items labelled.")); keys.innerHTML = "<kbd>u</kbd> undo"; return; }
  main.append(el("h1", "", item.question));
  if (item.kind === "yes_no") {
    main.append(el("div", "label", "Evidence"), el("div", "box", item.evidence));
    keys.innerHTML = "<kbd>1</kbd> yes &nbsp; <kbd>2</kbd> no &nbsp; <kbd>Space</kbd> skip &nbsp; <kbd>u</kbd> undo";
  } else {
    if (item.context) main.append(el("div", "label", "Context"), el("div", "box", item.context));
    const pair = el("div", "pair");
    for (const [n, side] of [["1", "left"], ["2", "right"]]) {
      const col = el("div"); col.append(el("div", "label", n), el("div", "box", item[side])); pair.append(col);
    }
    main.append(pair);
    keys.innerHTML = "<kbd>1</kbd> left &nbsp; <kbd>2</kbd> right &nbsp; <kbd>Space</kbd> skip &nbsp; <kbd>u</kbd> undo";
  }
  window.scrollTo(0, 0);
}
async function call(path, body) {
  busy = true;
  try {
    const r = await fetch(path, body === undefined ? {} : {method: "POST", body: JSON.stringify(body)});
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    render(await r.json()); document.getElementById("msg").textContent = "";
  } catch (e) { document.getElementById("msg").textContent = `not saved: ${e.message}`; }
  finally { busy = false; }
}
document.addEventListener("keydown", (e) => {
  if (busy || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  const key = e.key === " " ? "skip" : e.key;
  if (item && ["1", "2", "skip"].includes(key)) { e.preventDefault(); call("/label", {id: item.id, key}); }
  else if (key === "u") call("/undo", {});
});
call("/state");
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main())
