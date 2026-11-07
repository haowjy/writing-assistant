# data/raw moved to /data on 2026-11-06

`data/raw` is a symlink to `/data/jimyao/research/writing-assistant/raw`. Scripts that
use `data/raw` keep working, and `scripts/fetch_sources.py` downloads through the link.

**Why:** the raw sources are 16 GB and were on the root drive, which was 91% full. PG-19
is 11 GB and BookCorpus is 4.3 GB; the rest is about 100 MB of small benchmark sets.

**On a new clone or machine:** the symlink is not tracked by git. `data/raw` is ignored,
so either create the symlink or run `scripts/fetch_sources.py` to download into a plain
`data/raw` directory.

**BookCorpus:** the upload's card says MIT, but the books are from Smashwords and should
be treated as copyrighted. Do not commit or redistribute them. See
`data/raw/research/bookcorpus/revision.json`.
