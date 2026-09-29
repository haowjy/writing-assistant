# Phase 1 gate — R4 resolution evidence

R4-1 is closed with a literal, independently hashed write-action fixture.

- `pre_action_input` is a typed `ContextContentV1` containing only the user
  request. `payloads.request` is a distinct exact-request artifact that points
  to that content. `assistant_output` is a separately hashed `MessageV1`.
- The trace points to the pre-action content/request only; the action payload's
  `message_ref` points to the assistant message. Post-action and post-result
  contexts remain separate typed content/revision artifacts.
- `budget_before` and `budget_after` are distinct payload artifacts. The latter
  records one consumed tool call, links to the former, and is referenced by S2;
  S1 references the pre-charge budget. Result charge and state references agree.
- Root P0 now carries the pre-action context and request/budget artifacts in its
  closure; descendant checkpoint and commit hashes were recomputed after that
  closure was tightened.
- Before/after `ExecutionValueProjectionV1` payload bodies are present and
  independently checked for file, message, queue, and budget transitions.
  Every payload and descendant record hash is recomputed with literal domain
  tags, then loaded through its typed record decoder where one exists.
- Obsolete duplicate fixture aliases were removed. The malformed graph matrix
  now includes the exact hash-key object, empty object, and empty-string cases
  for both reference arrays through constructor and wire loading. The binary
  helper docstring states that it appends `:bytes` to a base domain argument.

Verification:

```
PYTHONPATH=.:src uv run --with pytest pytest -q tests/test_task_graph.py
17 passed, 20 subtests passed
PYTHONPATH=.:src uv run --with pytest pytest -q
292 passed, 10 skipped, 124 subtests passed
uv run --with ruff ruff check src/writing_agent/task_graph.py tests/test_task_graph.py
All checks passed
uv run --with ruff ruff format --check src/writing_agent/task_graph.py tests/test_task_graph.py
2 files already formatted
```

The fixture is canonical JSON generated from the record constructors; its
embedded artifact bodies and hashes also pass an independent standard-library
SHA-256 recomputation.
