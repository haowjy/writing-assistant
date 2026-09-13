# Wiki conventions

Write one concept per page and link to its owner rather than repeating it. Pages
describe project intent; they must not imply that a capability has been implemented
or an experiment has succeeded. Label open questions in the work plan until settled.

The wiki is code-agnostic. CLI and dataset-format instructions belong in
[docs](../../docs/data-contract.md) or the root [README](../../README.md);
implementation contracts belong in [src/.context/](../../src/.context/CONTEXT.md).

## Sources

[original-research.zip](../sources/original-research.zip) preserves the five supplied
research documents unchanged. Treat it as immutable source material. The
[research work plan](../../work/research-plan/index.md) holds their proposed methods
and checklists; the wiki summarizes the project intent shared across those proposals.

Update [index.md](../index.md) when adding or moving pages. Keep terminology in
[vocab.md](../vocab.md). Check relative links with `meridian kg check wiki` and
update instruction mirrors with `meridian qi claude-md-fix wiki`.
