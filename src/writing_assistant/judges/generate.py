"""Generation a scorer compares against: short forecasts of how a story ends."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from writing_assistant.judges.reply import MAX_TOKENS, Prompt, Unscored, answer_text, why_unanswered

if TYPE_CHECKING:
    from writing_assistant.episode import CallRecord
    from writing_assistant.llm.client import LLMClient

FORECAST = Prompt(
    version="forecast-v1",
    system=(
        "You read the opening of a story and forecast how it ends. Give {n} forecasts that "
        "differ from each other in what happens, not only in wording. Each is one sentence "
        "naming the outcome for the main characters. Reply with only a numbered list, one "
        "forecast per line: `1. ...`."
    ),
    user="Opening:\n<<<\n{prefix}\n>>>",
)
# Spread is the point of n forecasts; a deterministic judge would collapse them.
TEMPERATURE = 0.8


async def forecasts(client: LLMClient, endpoint: str, prefix: str, n: int) -> tuple[list[str], list[CallRecord]]:
    """(the first n numbered forecasts, [the call]). Raises `Unscored` when the reply has fewer than n."""
    rec = await client.chat(FORECAST.messages(prefix=prefix, n=n), endpoint=endpoint, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    reply = answer_text(rec)
    items = [m.strip() for m in re.findall(r"^\s*\d+\s*[.)]\s*(.+?)\s*$", reply, re.MULTILINE)]
    items = [i for i in items if i]
    if len(items) < n:
        raise Unscored(why_unanswered(rec), [rec])
    return items[:n], [rec]
