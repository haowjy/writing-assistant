# Phase 1 identity-contract gate — R3 resolution evidence

The two R3 blockers are covered by the task-graph contract tests and fixture.

- `GraphInstanceV1.source_refs` and `request_refs` validate the outer container
  as an array before iterating hash values. Constructors and strict wire loaders
  reject object, string, boolean, null, and numeric shapes; valid empty and
  singleton arrays are exercised for both fields.
- Standalone goldens now assert file bytes, tree, graph instance, context
  content, and the other retained record identities.
- `tests/fixtures/task_graph_chain.json` independently specifies canonical
  bodies and payload hashes for a design §5 write action/trace and tool result,
  then E1/P1/K1 and E2/P2/K2. It records logical action/call/result IDs, a
  queued call in P1, an empty queue in P2, content prepared without an E2
  backlink, and a provenance-bearing context revision after E2. No storage or
  tool implementation is included.
- The weaker chain test is named `test_acyclic_record_round_trip`; it no longer
  claims complete action/result coverage. `domain_hash("file", text)` remains an
  unambiguous compatibility spelling for exact-byte `file_hash`; structured
  values are rejected, and the binary helper documents its separate `:bytes`
  domain.

Verification:

```
PYTHONPATH=.:src uv run --with pytest pytest -q
292 passed, 10 skipped, 118 subtests passed
uv run --with ruff ruff check src/writing_agent/task_graph.py tests/test_task_graph.py
All checks passed
uv run --with ruff ruff format --check src/writing_agent/task_graph.py tests/test_task_graph.py
2 files already formatted
```
