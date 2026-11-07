# Phase 1 identity-contract gate — resolution evidence

The Phase 1 R2 reproductions are now covered by `tests/test_task_graph.py` and the
independent `tests/fixtures/task_graph_chain.json` fixture.

- Wire decoding accepts only plain JSON dictionaries/lists, typed message parts,
  complete node records, and complete reference arrays. `from_dict`/`from_json`
  reject constructor shorthands and null derived event/context identities; the
  canonical input bytes must equal the record's output bytes. Direct Python
  constructors remain the ergonomic (normalizing) path.
- `EVENT_KINDS` is exactly the 16 kinds in `work/sft/task-graph.md`; aliases are
  rejected.
- History action and tool-result IDs are logical IDs (not SHA-256 values), and
  duplicate continuation call IDs are rejected.
- Goldens cover file/tree/instance/context-content identities. The chain fixture
  independently specifies canonical bodies and hashes for content → event →
  provenance-bearing revision → checkpoint → commit, with nonempty history and
  tool queue.
- The task-graph specification now states the file-byte codec versus binary
  `:bytes` domains and requires Phase 2 refs to persist only `{head_commit: ...}`;
  `expected_head` remains a CAS request.

Verification:

```
PYTHONPATH=.:src uv run --extra judging --with pytest pytest -q
291 passed, 10 skipped, 104 subtests passed
uv run --with ruff ruff check src/writing_agent/task_graph.py tests/test_task_graph.py
All checks passed
uv run --with ruff ruff format --check src/writing_agent/task_graph.py tests/test_task_graph.py
2 files already formatted
```
