"""Suite 3: what the writer is shown. The author's true passage, the verifier's references and
the impostor stay in the episode for the scorer and never reach a prompt; a row whose
exemplars overlap any of them is refused at setup."""

import copy

import pytest

from writing_assistant.episode.store import Task
from writing_assistant.suites.style import suite


def _p(work, chapter, words):
    return {"work_id": work, "chapter": chapter, "text": " ".join(words)}


POOL = [_p("g:1", i, [f"exemplar{i}w{j}" for j in range(40)]) for i in range(3)]
Y_STAR = _p("g:2", 0, [f"target{j}" for j in range(40)])
REFS = [_p("g:3", i, [f"verifier{i}w{j}" for j in range(40)]) for i in range(2)]
IMPOSTOR = _p("g:9", 4, [f"impostor{j}" for j in range(40)]) | {"source_group": "Other, Author"}
ROW = Task("3:a:t0:instruction:k3", "3", "k_exemplar_scene", "dev", ["g:1", "g:2", "g:3"], {
    "mode": "instruction", "k": 3, "author": "Some, Author", "exemplars": POOL,
    "beat_sheet": [{"role": "setting", "paraphrase": "A ship leaves port at dawn."}],
    "y_star": Y_STAR, "verifier_refs": REFS, "impostor": IMPOSTOR,
})


@pytest.mark.parametrize("mode", ["instruction", "continuation", "style_retrieved"])
async def test_targets_and_verifier_passages_never_reach_the_writer(fake_turn, mode):
    row = copy.deepcopy(ROW)
    row.payload["mode"] = mode
    [ep] = suite.setup(row, None)
    turn = fake_turn(["The ship left at dawn."])
    await suite.play([ep], None, turn)
    [request] = turn.llm.requests
    shown = "\n".join(m["content"] for m in request["messages"])
    assert all(p["text"] in shown for p in POOL)
    assert not [w for w in ("target", "verifier", "impostor", "Some, Author") if w in shown]


@pytest.mark.parametrize("hidden", [Y_STAR, REFS[1], IMPOSTOR])
def test_an_exemplar_that_is_a_hidden_passage_is_refused(hidden):
    row = copy.deepcopy(ROW)
    row.payload["exemplars"] = [POOL[0], copy.deepcopy(hidden), POOL[2]]
    with pytest.raises(ValueError, match="overlaps"):
        suite.setup(row, None)
    # the same text from another chapter is caught by its words, not only its position
    row.payload["exemplars"][1] = hidden | {"work_id": "g:7", "chapter": 99}
    with pytest.raises(ValueError, match="overlaps"):
        suite.setup(row, None)
