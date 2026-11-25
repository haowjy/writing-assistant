"""Each judge parses scripted replies from the fake client; refusals and garbage raise
`Unscored` instead of turning into a score."""

import math
from pathlib import Path

import numpy as np
import pytest

from writing_assistant.judges import ReferenceIndex, Unscored, judges_from
from writing_assistant.judges.readability import profile

MODELS = Path(__file__).resolve().parents[2] / "models.toml"


def judges(fake_llm, replies):
    return judges_from(MODELS, fake_llm(replies), "eval")


@pytest.mark.parametrize(
    "reply, want",
    [
        ("The letter is in the drawer.\nANSWER: YES", True),
        ("<think>No mention... wait, chapter two says so.</think>**Answer:** No", False),
        ("Now, the evidence never mentions the key.\n\nNo.", False),  # "Now" is reasoning, last line answers
        ("No doubt she knows: she read the letter.\nANSWER: YES", True),  # leading "No" is not the answer
        ("ANSWER: NO, on reflection.\nANSWER: YES", True),  # the last answer wins
    ],
)
async def test_fact_parses(fake_llm, reply, want):
    j = judges(fake_llm, [reply])
    assert await j.fact("Does Mara know where the key is?", "Mara read the letter.") is want


@pytest.mark.parametrize(
    "reply, reason",
    [
        ("I'm sorry, but I can't help with sexually explicit content.", "refusal"),
        ("The evidence is ambiguous on this point.", "unparseable"),
        ("<think>Let me consider whether the key", "unparseable"),  # cut off while thinking
        ("", "unparseable"),
    ],
)
async def test_fact_unscored(fake_llm, reply, reason):
    j = judges(fake_llm, [reply])
    with pytest.raises(Unscored) as e:
        await j.fact("q", "evidence")
    assert e.value.reason == reason and e.value.records[0].response == reply


async def test_pairwise_averages_both_orders(fake_llm):
    # Forward order shows a as A; the swapped order shows a as B.
    j = judges(fake_llm, ["ANSWER: A", "ANSWER: B", "ANSWER: A", "ANSWER: A", "Both read alike.\nTie", "ANSWER: B"])
    assert await j.craft_pair("a", "b", "reference") == 1.0  # consistent: a both times
    assert await j.author_verify("a", "b", ["p1", "p2"]) == 0.5  # position bias cancels
    assert await j.edit_pair("edited", "original", "objection") == 0.75  # tie, then a
    assert j.author_verify.stamp == {"use": "author_verify", "endpoint": "openrouter-judge",
                                     "model": "<cross-family judge model>", "prompt": "author-v1"}


async def test_pairwise_one_order_failing_is_unscored(fake_llm):
    llm = fake_llm(["ANSWER: A", "I cannot compare these passages."])
    j = judges_from(MODELS, llm, "eval")
    with pytest.raises(Unscored) as e:
        await j.secret_2afc("transcript", "secret", "decoy")
    assert e.value.reason == "refusal" and len(e.value.records) == 2
    # the transcript is the context; the secrets are the two options, in both orders
    shown = [r["messages"][1]["content"] for r in llm.requests]
    assert all("transcript" in s for s in shown)
    assert shown[0].index("secret") < shown[0].index("decoy") and shown[1].index("decoy") < shown[1].index("secret")
    with pytest.raises(Unscored, match="unparseable"):
        await judges(fake_llm, ["Passage A has better rhythm", "ANSWER: B"]).craft_pair("a", "b", "r")


async def test_outline_and_forecast(fake_llm):
    j = judges(fake_llm, [
        'Here is the outline:\n```json\n[{"role": "setting", "paraphrase": "A port town in winter."},'
        ' {"role": "inciting incident", "paraphrase": "A stranger brings a sealed letter."}]\n```',
        '[{"role": "setting"}]',
        "Sure!\n1. She burns the letter.\n2) The town learns the truth.\n3. He leaves alone.",
        "1. Only one idea.",
    ])
    assert await j.outline("text") == [
        {"role": "setting", "paraphrase": "A port town in winter."},
        {"role": "inciting incident", "paraphrase": "A stranger brings a sealed letter."},
    ]
    with pytest.raises(Unscored, match="unparseable"):
        await j.outline("text")
    assert await j.forecast("Once upon a time", 2) == ["She burns the letter.", "The town learns the truth."]
    with pytest.raises(Unscored):
        await j.forecast("Once upon a time", 2)


async def test_embed_and_reference_index(fake_llm, tmp_path):
    llm = fake_llm(vectors={"x": [3.0, 4.0], "y": [0.0, 2.0], "z": [0.0, 0.0]})
    j = judges_from(MODELS, llm, "eval")
    v = await j.embed_semantic(["x", "y"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    idx = ReferenceIndex(ids=["x", "y"], vectors=v, model="nomic-embed-text-v2")
    idx.save(tmp_path / "refs.npz")
    back = ReferenceIndex.load(tmp_path / "refs.npz")
    assert back.ids == ["x", "y"] and back.model == "nomic-embed-text-v2"
    assert np.allclose(back.nearest_distance(np.array([[0.0, 1.0], [1.0, 0.0]])), [0.0, 0.4])
    with pytest.raises(ValueError):
        await j.embed_style(["z"])


def test_readability_on_fixed_text():
    p = profile(
        "The report argues that the plan failed. The book I read was long, and the book that "
        "Zoë wrote was longer. The government policy review board met at dawn."
    )
    assert p["n_sentences"] == 3
    assert p["inanimate_animate_per_1k"] > 0  # "the report argues"
    assert p["object_relatives_per_1k"] == 2 * p["reduced_object_relatives_per_1k"] > 0  # "I read" / "that Zoë wrote"
    assert p["stacked_nouns_per_1k"] > 0  # "government policy review board"
    assert 0 < p["mean_dep_length"] and 0 <= p["rare_word_rate"] < 1
    empty, word = profile(""), profile("Run!")
    assert empty["n_tokens"] == 0 and all(math.isnan(empty[k]) for k in empty if k not in ("n_tokens", "n_sentences"))
    assert word["n_tokens"] == 1 and math.isnan(word["mean_dep_length"]) and word["stacked_nouns_per_1k"] == 0
