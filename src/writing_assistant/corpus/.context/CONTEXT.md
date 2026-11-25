# corpus/ — what the code doesn't show

## The splitter on real Gutenberg books

`text.split_chapters` was checked against the first live import (2026-11-25, 10 books in
`/data/jimyao/research/writing-assistant/corpus-cache/`). Re-run this comparison after
changing any rule in it; each line is the expected count.

| Work | Pieces | Notes |
|---|---|---|
| 1342 Pride and Prejudice | 61 | Chapter I's heading sits in an illustration caption (`Chapter I.]`); the list of illustrations is skipped |
| 158 Emma | 55 | Three volumes, with chapter numbers restarting in each |
| 161 Sense and Sensibility | 50 | |
| 84 Frankenstein | 28 | Letters 1–4 plus Chapters 1–24 |
| 18247 The Last Man | 30 | The printed edition repeats CHAPTER IV in Vol. I (11+9+10) |
| 244 A Study in Scarlet | 14 | Parts dropped as title pages, chapters kept |
| 2852 Hound of the Baskervilles | 15 | |
| 1661 Adventures of Sherlock Holmes | 14 | Bare roman numerals. Story I has sections I–III, so it yields 3 pieces; the other 11 stories yield one each |
| 1513 Romeo and Juliet | 25 | Scenes; the Act II prologue becomes its own short piece |
| 1524 Hamlet | 20 | Scenes; IV.1 (253 words) is kept because the contents-entry rule applies only before the first chapter |

Contents lists, `ACT` lines directly followed by `SCENE`, and part title pages are all dropped
by the 50-word minimum body. Front matter before the first heading is dropped too.

## Seams

- `overlap.FETCH_TOOLS` duplicates the search tool names until `search/gateway.py` lands.
  Take them from its `TOOLS` then.
- `overlap` counts every `deliver` message as benchmark text, following the convention in
  `episode/model.py`. If a suite ever relays a model actor's reply through `Turn` (the
  Slice 2 co-actors), the `deliver` event cannot say who wrote it. That actor's words would
  then count as context, and a co-actor's recitation would be reported as `context_copy`. The
  fix belongs in `episode/`: record the author on the `deliver` event.
- A suite tool that echoes the subject's words back in its result, other than as an exact
  argument string, leaks those words into `context_copy`.
