"""The one shared dataset of works every suite draws on.

`Corpus` is built once in cli.py and reaches suites as `Services.corpus`. Works (`rows.Work`)
and task rows (`episode.store.Task`) are two schemas; a task names its works by `work_id`.

- `rows.py`: `Work`, `Corpus` (index, text cache, `select`, `text`, `chapters`, `overlap`,
  `flag`).
- `release.py`: the licence rule, the one owner of what text may leave the machine
  (`export_public` behind `wa-bench export --public`, `stage_sync` behind `wa-bench sync`).
- `text.py`: decoding, Gutenberg header/footer stripping, the chapter splitter, token counts.
- `overlap.py`: `context_copy` / `tool_copy` / `memorization` over word 8-grams.
- `collect.py`: `collect()` (behind `wa-bench collect`); Slice 1 has the Gutenberg importer.
- `authors.py`: the within-author partition and nested `k` exemplar draws for Suite 3.
"""

from writing_assistant.corpus.rows import Corpus, Work

__all__ = ["Corpus", "Work"]
