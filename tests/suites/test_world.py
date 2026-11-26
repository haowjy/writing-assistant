"""Suite 1a: the SRD engine's determinism, legality and visibility, and the POV write task's
two guarantees (the projected prompt holds no hidden text; the swap twin differs only in the
designated event and its state)."""

import copy
from types import SimpleNamespace

import pytest

from writing_assistant.episode import Episode
from writing_assistant.episode.store import Task
from writing_assistant.episode.tools import dispatch_tool
from writing_assistant.judges import Judge
from writing_assistant.suites import Tally
from writing_assistant.suites.world import srd, suite_1a

SECRET = "Orin is the last heir of the drowned kings."
SWAPPED = "Orin is a smuggler who owes the harbour guild."


def _character(name, location, inventory=()):
    return {"name": name, "location": location, "hp": 9, "max_hp": 9, "ac": 12, "attack_bonus": 4,
            "abilities": dict.fromkeys(srd.ABILITIES, 1), "conditions": [], "inventory": list(inventory)}


WORLD = {
    "seed": 7, "roll_count": 0, "rules_rejections": 0, "party": ["mira", "tam"],
    "locations": {"tavern": {"name": "Tavern", "exits": ["street"], "items": []},
                  "street": {"name": "Street", "exits": ["tavern", "crypt"], "items": []},
                  "crypt": {"name": "Crypt", "exits": ["street"], "items": []}},
    "characters": {"mira": _character("Mira", "street", ["dagger"]), "tam": _character("Tam", "tavern"),
                   "orin": _character("Orin", "crypt")},
    "items": {"dagger": {"name": "dagger", "kind": "weapon", "damage": "1d4", "damage_bonus": 2}},
    "secrets": {"heir": {"text": SECRET, "owner": "orin", "known_by": ["orin"], "reveal": {"location": "crypt"}}},
    "facts": [{"text": SECRET, "visible_to": ["dm", "orin"]}],
}


def _dm_episode() -> Episode:
    return Episode(ep_id="e", task_id="t", suite="1a", split="dev", state={"world": copy.deepcopy(WORLD)},
                   actor_tools={"dm": list(srd.TOOLS)})


async def _call(ep, name, args):
    return await dispatch_tool(ep, "dm", name, args, srd.TOOLS)


async def test_a_fork_made_mid_play_rolls_the_parents_dice():
    ep = _dm_episode()
    await _call(ep, "srd_roll", {"expr": "1d20"})
    fork = ep.fork("mid")
    attack = {"action": "attack", "actor": "mira", "target": "tam", "item": "dagger"}
    await _call(ep, "srd_resolve", {"action": "move", "actor": "tam", "to": "street"})
    await _call(fork, "srd_resolve", {"action": "move", "actor": "tam", "to": "street"})
    for e in (ep, fork):
        await _call(e, "srd_roll", {"expr": "3d6+1"})
        await _call(e, "srd_resolve", attack)
    assert [ev.data for ev in ep.events[1:]] == [ev.data for ev in fork.events[1:]]
    assert ep.state["world"] == fork.state["world"]


async def test_an_illegal_action_is_rejected_and_counted():
    ep = _dm_episode()
    before = copy.deepcopy(ep.state["world"]["characters"])
    moves = [
        ("srd_resolve", {"action": "move", "actor": "tam", "to": "crypt"}),  # no exit tavern -> crypt
        ("srd_resolve", {"action": "attack", "actor": "mira", "target": "tam", "item": "dagger"}),  # not here
        ("srd_reveal", {"secret_id": "heir", "to": ["mira"]}),  # reveal only in the crypt
        ("srd_roll", {"expr": "1d0"}),
    ]
    results = [await _call(ep, name, args) for name, args in moves]
    assert [r["ok"] for r in results] == [False] * 4
    assert [r["rule"] for r in results] == ["legality", "legality", "visibility", "dice"]
    assert ep.state["world"]["rules_rejections"] == 4
    assert ep.state["world"]["characters"] == before
    assert [e.kind for e in ep.events] == ["tool"] * 4  # refusals put nothing into the world


async def test_a_hidden_roll_is_visible_to_the_dm_only():
    ep = _dm_episode()
    await _call(ep, "srd_roll", {"expr": "1d20", "hidden": True})
    await _call(ep, "srd_roll", {"expr": "1d20"})
    hidden, public = [e for e in ep.events if e.kind == "world"]
    assert hidden.visible_to == {"dm": "full"} and public.visible_to == {"*": "full"}
    assert [e.data["text"] for e in ep.view("mira").events if e.kind == "world"] == [public.data["text"]]


POV_LOG = [
    {"actor": "mira", "kind": "say", "data": {"text": "Mira: We should find Orin."},
     "visible_to": {"mira": "full", "tam": "full", "dm": "full"}},
    {"actor": "dm", "kind": "world", "data": {"tool": "propose_event", "text": "Mira finds a glove with a crown sigil."},
     "visible_to": {"mira": "full", "dm": "full"}},
    {"actor": "dm", "kind": "world", "data": {"tool": "record_fact", "text": SECRET},
     "visible_to": {"dm": "full", "orin": "full"}},
    {"actor": "dm", "kind": "world", "data": {"tool": "roll", "text": "Roll 1d20: [4] = 4"},
     "visible_to": {"dm": "full"}},
    {"actor": "dm", "kind": "say", "data": {"text": "The glove is still warm."},
     "visible_to": {"mira": "full", "dm": "full"}},
]
SWAP = {"data": {"tool": "record_fact", "text": SWAPPED},
        "state": {"world": {"secrets": {"heir": {"text": SWAPPED}}, "facts": [{"text": SWAPPED, "visible_to": ["dm", "orin"]}]}}}
POV_TASK = Task("pov-1", "1a", "pov_write", "dev", [], {
    "world": WORLD, "pov": "mira", "log": POV_LOG, "hidden_event_idx": 2, "swap": SWAP,
    "request": "Write the next chapter from Mira's point of view.",
    "hidden_probes": ["Does the chapter state or imply that Orin is an heir?"],
    "permitted_probes": ["Does the chapter mention the glove?"],
})


async def test_pov_write_projects_the_log_and_swaps_only_the_hidden_event(fake_turn):
    eps = suite_1a.setup(POV_TASK, None)
    by_id = {ep.ep_id: ep for ep in eps}
    full, full_twin = by_id["pov-1~full_view"], by_id["pov-1~full_view~twin_swap"]
    pov = by_id["pov-1~pov_view"]

    # The swap twin differs from its parent only in the designated event and its state.
    assert full_twin.parent_id == full.ep_id
    assert [i for i, (a, b) in enumerate(zip(full.events, full_twin.events, strict=True)) if a != b] == [2]
    assert full_twin.events[2].data["text"] == SWAPPED
    assert {k for k in full.state["world"] if full.state["world"][k] != full_twin.state["world"][k]} == {"secrets", "facts"}
    assert SECRET not in str(full_twin.state["world"]) and full_twin.files == full.files

    # A swap that leaves the secret in the world state is refused at setup.
    leaky = copy.deepcopy(POV_TASK)
    del leaky.payload["swap"]["state"]["world"]["facts"]
    with pytest.raises(ValueError, match="still contains"):
        suite_1a.setup(leaky, None)

    turn = fake_turn(["Mira turned the glove over. Orin was the heir; she knew it."] + ["Mira held the glove."] * 3)
    await suite_1a.play(eps, None, turn)
    prompts = {ep.ep_id: "\n".join(m["content"] for m in r["messages"]) for ep, r in zip(eps, turn.llm.requests, strict=True)}
    pov_prompt, pov_twin_prompt = prompts[pov.ep_id], prompts[f"{pov.ep_id}~twin_swap"]
    hidden = [ev["data"]["text"] for ev in POV_LOG if "mira" not in ev["visible_to"]]
    assert not [t for t in hidden if t in pov_prompt] and "heir" not in pov_prompt
    assert pov_prompt == pov_twin_prompt  # the pov_view twin is a second sample of identical input
    assert SECRET in prompts[full.ep_id] and SWAPPED in prompts[full_twin.ep_id]

    # Scored with a scripted fact judge: the leak nets out against the twin, recall is the floor.
    async def fact(question, evidence):
        return ("heir" in question and "heir" in evidence) or ("glove" in question and "glove" in evidence), []

    services = SimpleNamespace(judges=SimpleNamespace(fact=Judge(fact, {"use": "fact"}), readability=lambda text: {}),
                               corpus=SimpleNamespace(overlap=lambda text, ep: {}, flag=lambda o: False))
    for ep in eps:
        ep.complete()
    scores = await suite_1a.score(full, services, Tally(), full_twin)
    assert (scores["pov_leak"], scores["twin_pov_leak"], scores["net_pov_leak"], scores["permitted_fact_recall"]) == (1.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("secret", ["Orin burned the café.", 'Orin said "I did it".'])
def test_a_secret_quoted_in_a_later_visible_event_is_refused_whatever_its_characters(secret):
    log = [{"actor": "dm", "kind": "world", "data": {"text": secret}, "visible_to": {"dm": "full"}},
           {"actor": "dm", "kind": "say", "data": {"text": secret}, "visible_to": {"dm": "full", "mira": "full"}}]
    task = Task("pov-2", "1a", "pov_write", "dev", [], POV_TASK.payload | {
        "log": log, "hidden_event_idx": 0, "swap": {"data": {"text": "Orin is innocent."}}})
    with pytest.raises(ValueError, match="still contains"):
        suite_1a.setup(task, None)
