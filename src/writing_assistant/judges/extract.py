"""Narrow extraction with the evidence in view: one yes/no fact, or an IdeaLens outline."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from writing_assistant.judges.reply import MAX_TOKENS, TEMPERATURE, Prompt, Unscored, answer_text, pick, why_unanswered

if TYPE_CHECKING:
    from writing_assistant.episode import CallRecord
    from writing_assistant.llm.client import LLMClient

FACT = Prompt(
    version="fact-v1",
    system=(
        "You check one question against the evidence you are given. Use only the evidence, "
        "never outside knowledge of the story or its source. Give at most two sentences of "
        "reasoning, then a final line that is exactly `ANSWER: YES` or `ANSWER: NO`."
    ),
    user=(
        "Evidence:\n<<<\n{evidence}\n>>>\n\n"
        "Question: {question}\n\n"
        "Answer YES only if the evidence shows the answer is yes. If the evidence shows "
        "otherwise, or does not say, answer NO."
    ),
)

# IdeaLens (arXiv:2610.06778) induces a role set per format; ours for fiction is a starting
# set the model may extend, not the paper's induced set.
OUTLINE = Prompt(
    version="outline-v1",
    system=(
        "You outline the ideas of a piece of fiction, not its wording. An outline is an "
        "ordered list of items; each item names the discourse role a stretch of the text "
        "plays and paraphrases its content in one short sentence that reuses none of the "
        "text's distinctive phrases. Typical roles: setting, character introduction, "
        "backstory, inciting incident, goal, obstacle, conflict, dialogue exchange, "
        "revelation, decision, action, reflection, reversal, climax, resolution, hook. Use "
        "another short role name when none fits. Reply with only a JSON array of objects "
        'with keys "role" and "paraphrase", in the order the text presents them.'
    ),
    user="Text:\n<<<\n{text}\n>>>",
)


async def yes_no(client: LLMClient, endpoint: str, question: str, evidence: str) -> tuple[bool, list[CallRecord]]:
    """(answer, [the call]). Raises `Unscored` on a refusal or a reply with no YES/NO answer."""
    rec = await client.chat(
        FACT.messages(question=question, evidence=evidence),
        endpoint=endpoint, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
    )
    return pick(rec, {"yes": True, "no": False}), [rec]


async def outline(client: LLMClient, endpoint: str, text: str) -> tuple[list[dict], list[CallRecord]]:
    """(IdeaLens items `{role, paraphrase}` in text order, [the call]). Raises `Unscored` unless the reply
    holds a non-empty JSON array of such items."""
    rec = await client.chat(
        OUTLINE.messages(text=text), endpoint=endpoint, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
    )
    reply = answer_text(rec)
    # The array may sit inside a ```json fence or after a sentence of preamble.
    m = re.search(r"\[.*\]", reply, re.DOTALL)
    try:
        items = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        items = None
    ok = isinstance(items, list) and items and all(
        isinstance(it, dict)
        and isinstance(it.get("role"), str) and it["role"].strip()
        and isinstance(it.get("paraphrase"), str) and it["paraphrase"].strip()
        for it in items
    )
    if not ok:
        raise Unscored(why_unanswered(rec), [rec])
    return [{"role": it["role"].strip(), "paraphrase": it["paraphrase"].strip()} for it in items], [rec]
