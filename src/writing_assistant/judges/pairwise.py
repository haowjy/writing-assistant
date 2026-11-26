"""Position-swapped A/B judge. Each use (craft, edit, author verification, secret 2AFC) is a
prompt template over the same `ab` call; `judges_from` binds each to its endpoint."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from writing_assistant.judges.reply import MAX_TOKENS, TEMPERATURE, Prompt, Unscored, pick

if TYPE_CHECKING:
    from writing_assistant.llm.client import LLMClient

_FINAL = (
    "Give at most three sentences of reasoning, then a final line that is exactly "
    "`ANSWER: A` or `ANSWER: B`."
)

CRAFT = Prompt(
    version="craft-v1",
    system=(
        "You compare two passages of fiction written for the same place in a story. A "
        "reference passage by the story's human author is shown so you can see what the "
        "story needs; neither passage has to resemble it. Prefer the passage that is better "
        "fiction for that place.\n"
        "Count in favour: concrete, specific detail; consistency with the characters and "
        "facts the reference establishes; subtext over stated emotion; dialogue that sounds "
        "like these people; pacing that suits the moment.\n"
        "Count against: purple or ornamental prose; cliché and stock phrasing; explaining "
        "what a scene already shows; generic detail that could sit in any story; padding.\n"
        "Length is not a criterion. Which passage appears first means nothing. " + _FINAL
    ),
    user="Reference:\n<<<\n{context}\n>>>\n\nPassage A:\n<<<\n{a}\n>>>\n\nPassage B:\n<<<\n{b}\n>>>",
)

EDIT = Prompt(
    version="edit-v1",
    system=(
        "An editor raised one objection to a passage. You see two versions of the passage. "
        "Pick the version that better answers the objection while leaving the rest of the "
        "passage as it was. A version that ignores the objection, or rewrites more than the "
        "objection asks for, answers it worse. Which version appears first means nothing. " + _FINAL
    ),
    user="Objection:\n<<<\n{context}\n>>>\n\nVersion A:\n<<<\n{a}\n>>>\n\nVersion B:\n<<<\n{b}\n>>>",
)

AUTHOR = Prompt(
    version="author-v2",
    system=(
        "You see several passages by one author, then two candidate passages A and B. Pick the "
        "candidate more likely written by that author. One, both or neither may be; when you "
        "cannot be sure, pick the one closer to the author's voice. Do not try to identify the "
        "works or authors involved. Judge by voice, not topic: sentence length and rhythm, "
        "diction, punctuation habits, how dialogue is tagged, how much is described and how. "
        "Which candidate appears first means nothing. " + _FINAL
    ),
    user="Passages by the author:\n<<<\n{context}\n>>>\n\nCandidate A:\n<<<\n{a}\n>>>\n\nCandidate B:\n<<<\n{b}\n>>>",
)

SECRET = Prompt(
    version="secret-v1",
    system=(
        "You read a transcript. One of two statements, A or B, is a secret that a character "
        "in it holds; the other is a decoy. Pick the statement the transcript points to: "
        "what is said, done or avoided that fits one statement and not the other. Which "
        "statement appears first means nothing. " + _FINAL
    ),
    user="Transcript:\n<<<\n{context}\n>>>\n\nStatement A:\n<<<\n{a}\n>>>\n\nStatement B:\n<<<\n{b}\n>>>",
)

# A reply naming neither side is an even split for that order, not a failure to answer.
_CHOICE = {"a": 1.0, "passage a": 1.0, "b": 0.0, "passage b": 0.0, "tie": 0.5, "neither": 0.5, "both": 0.5}


async def ab(client: LLMClient, endpoint: str, prompt: Prompt, a: str, b: str, context: str) -> float:
    """P(a), averaged over both orders. Both orders must parse: scoring the one that did
    would let the position bias the swap exists to cancel back in, so either failing raises
    `Unscored` (refusal if either order refused) carrying both calls."""

    async def call(x: str, y: str):
        return await client.chat(
            prompt.messages(context=context, a=x, b=y),
            endpoint=endpoint, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )

    fwd, rev = await asyncio.gather(call(a, b), call(b, a))
    scores, reasons = [], []
    for rec, flip in ((fwd, False), (rev, True)):
        try:
            p = pick(rec, _CHOICE)
            scores.append(1.0 - p if flip else p)
        except Unscored as e:
            reasons.append(e.reason)
    if reasons:
        worst = next(r for r in ("refusal", "truncated", "unparseable") if r in reasons)
        raise Unscored(worst, [fwd, rev])
    return sum(scores) / 2
