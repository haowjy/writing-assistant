"""Suite 3 scorers: pure over a frozen episode, reaching models only through `services`.

k_exemplar_scene (the scene is the writer's last reply):
- `discriminator_fool_rate`: `author_verify` between the scene and the author's true passage
  y* for the same beats, with the held-out references in view: P(the scene is the
  author's). 0.5 means the verifier cannot tell them apart.
- `verifier_match`: `author_verify` between the scene and an impostor passage by another
  author of the same split (the general-impostors framing of authorship verification):
  P(the scene is the author's rather than the impostor's).
- `style_margin`: `embed_style` cosine of the scene to the mean of the references, minus its
  cosine to the impostor.
Both A/B comparisons cut the scene to the passage budget y* was cut to, so neither side
is judged on length; nothing here rewards or penalises length.

critique_edit (the scorer reads the edited file):
- `edited`: 1.0 when the file changed at all.
- `edit_scope`: share of the paragraphs outside the objection's scope left exactly as they
  were (whitespace collapsed); 1.0 is a perfectly scoped edit.
- `scope_changed`: share of in-scope paragraphs that changed.
- `edit_pair_win`: `edit_pair` P(the edited passage answers the objection better than the
  original); absent when nothing was edited.

Every prose output also gets `readability_*` and `overlap_*` (`corpus.overlap`, with the
pre-registered flag); for an edit, the prose is the paragraphs the writer wrote.
Unscored judge items are counted per metric by `Tally`; a metric with none judged is absent.
"""

from __future__ import annotations

import asyncio

import numpy as np

from writing_assistant.corpus.authors import PASSAGE_TOKENS
from writing_assistant.corpus.text import count_tokens, truncate_tokens
from writing_assistant.episode import Episode
from writing_assistant.suites import Tally, last_reply, norm, paragraphs, prose_metrics
from writing_assistant.suites.style.tasks import WRITER

JUDGED_BY = {
    "discriminator_fool_rate": "author_verify",
    "verifier_match": "author_verify",
    "style_margin": "embed_style",
    "edit_pair_win": "edit_pair",
}


async def _scene(ep: Episode, services, out: dict[str, float], tally: Tally) -> str:
    s = ep.state["scene"]
    text = last_reply(ep, WRITER)
    if not text:
        return text
    j = services.judges
    budget = max(count_tokens(s["y_star"]["text"]), 1)
    candidate = truncate_tokens(text, min(budget, PASSAGE_TOKENS))
    refs = [r["text"] for r in s["verifier_refs"]]
    fool, match, vectors = await asyncio.gather(
        tally.ask("discriminator_fool_rate", j.author_verify(candidate, s["y_star"]["text"], refs)),
        tally.ask("verifier_match", j.author_verify(candidate, s["impostor"]["text"], refs)),
        tally.ask("style_margin", j.embed_style([text, s["impostor"]["text"], *refs])),
    )
    for name, value in (("discriminator_fool_rate", fool), ("verifier_match", match)):
        if value is not None:
            out[name] = float(value)
    if vectors is not None:
        cand, impostor, refs_v = vectors[0], vectors[1], vectors[2:]
        centroid = refs_v.mean(axis=0)
        out["style_margin"] = float(_cos(cand, centroid) - _cos(cand, impostor))
    return text


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


async def _edit(ep: Episode, services, out: dict[str, float], tally: Tally) -> str:
    s = ep.state["edit"]
    original, after = s["original"], ep.files.get(s["path"], "")
    before_pars = [norm(p) for p in paragraphs(original)]
    after_pars = [norm(p) for p in paragraphs(after)]
    kept = set(after_pars)
    scope = {i - 1 for i in s["scope"]}
    outside = [p for i, p in enumerate(before_pars) if i not in scope]
    inside = [p for i, p in enumerate(before_pars) if i in scope]
    out["edited"] = float(norm(after) != norm(original))
    if outside:
        out["edit_scope"] = sum(p in kept for p in outside) / len(outside)
    out["scope_changed"] = sum(p not in kept for p in inside) / len(inside)
    if out["edited"]:
        win = await tally.ask("edit_pair_win", services.judges.edit_pair(after, original, s["objection"]))
        if win is not None:
            out["edit_pair_win"] = float(win)
    written = set(before_pars)
    return "\n\n".join(p for p in paragraphs(after) if norm(p) not in written)


SCORERS = {"k_exemplar_scene": _scene, "critique_edit": _edit}


async def score(ep: Episode, services, twin: Episode | None = None) -> dict[str, float]:
    if ep.status not in ("completed", "truncated"):
        raise ValueError(f"{ep.ep_id} is {ep.status}; only finished episodes are scored")
    out: dict[str, float] = {}
    tally = Tally()
    prose = await SCORERS[ep.state["kind"]](ep, services, out, tally)
    tally.into(out)
    prose_metrics(out, prose, ep, services)
    return out
