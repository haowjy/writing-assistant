"""The conversation step (`suites/converse.py`) through the placeholder suite that uses it:
who speaks, what each seat is delivered, how a conversation ends, and that one saved at any
delivery goes on as the original did."""

import shutil
from itertools import count

import pytest

from writing_assistant.episode.segments import Store
from writing_assistant.episode.store import Task
from writing_assistant.run import score_episode
from writing_assistant.suites import SUBJECT, converse, play
from writing_assistant.suites.situations import suite

CAST = {"counterpart": "other-endpoint"}


def talk(seats: dict[str, dict], max_turns: int, lines: int = 0, **payload) -> Task:
    """A `converse` row: `seats` maps seat -> its cast name, or a dict with more of its fields."""
    seats = {name: {"brief": f"You are {name}."} | (s if isinstance(s, dict) else {"cast": s}) for name, s in seats.items()}
    return Task("talk", "situations", "converse", "dev", [], {
        "request": "Plan the crossing.", "max_turns": max_turns, "seats": seats,
        "script": [{"say": f"scripted line {n}"} for n in range(lines)]} | payload)


def say_to(seats: list[str], text: str) -> dict:
    return {"name": "say_to", "arguments": {"seats": seats, "text": text}}


def numbered():
    """A model that never says the same thing twice: `<endpoint> <n>`."""
    n = count()
    return lambda messages, tools, endpoint: f"{endpoint} {next(n)}"


def lines(ep) -> list[tuple[str, str]]:
    return [(e.actor, e.data["text"]) for e in ep.events if e.kind == "say"]


def deliveries(ep) -> list[tuple[str, str]]:
    return [(e.actor, e.data["message"]) for e in ep.events if e.kind == "deliver"]


async def played(task, turn):
    [ep] = suite.setup(task, None)
    await play(suite, [ep], turn)
    ep.complete()
    return ep


TWO = {"author": "scripted", "writer": SUBJECT}
EIGHT = {"gm": "scripted", "ana": SUBJECT, "bo": "counterpart", "crow": "scripted", "dee": SUBJECT, "eli": "counterpart",
         "fen": SUBJECT, "guide": "scripted"}
# Whispers: ana is heard only by the gm and bo, the crow only by dee, and nobody hears eli.
WHISPERS = EIGHT | {"ana": {"cast": SUBJECT, "heard_by": ["gm", "bo"]}, "crow": {"cast": "scripted", "heard_by": ["dee"]},
                    "eli": {"cast": "counterpart", "heard_by": []}}


@pytest.mark.parametrize("seats, budget", [(TWO, 7), (EIGHT, 30), (WHISPERS, 30)], ids=["two", "eight", "eight-whispering"])
async def test_seats_speak_in_turn_to_the_budget_and_each_is_delivered_exactly_the_lines_it_heard(fake_turn, seats, budget):
    turn = fake_turn({"subject-endpoint": [numbered()] * 40, "other-endpoint": [numbered()] * 40}, converse.TOOLS, CAST)
    ep = await played(talk(seats, budget, lines=40), turn)
    conv, names = ep.state["converse"], list(seats)
    assert (ep.status, conv["ended"], conv["turns"], len(lines(ep))) == ("completed", "turn_budget", budget, budget)
    assert lines(ep)[0] == (names[0], "Plan the crossing.")
    if seats is not WHISPERS:  # everyone hears everyone, so nobody is passed over
        assert [seat for seat, _ in lines(ep)] == [names[n % len(names)] for n in range(budget)]
    hearers = {name: s.get("heard_by", names) if isinstance(s, dict) else names for name, s in seats.items()}
    last_sent: dict[str, int] = {}
    for sent in (e for e in ep.events if e.kind == "deliver"):
        heard = [f"{e.actor}: {e.data['text']}" for e in ep.events[last_sent.get(sent.actor, 0):sent.idx]
                 if e.kind == "say" and e.actor != sent.actor and sent.actor in hearers[e.actor]]
        assert heard and sent.data["message"] == "\n\n".join(heard)
        last_sent[sent.actor] = sent.idx
    assert all(ep.actors[seat] != "scripted" for seat in last_sent)
    assert {c.actor: c.endpoint for c in ep.calls} == {
        seat: "subject-endpoint" if ep.actors[seat] == SUBJECT else "other-endpoint" for seat in last_sent}
    replies = {(c.actor, c.response) for c in ep.calls}
    assert {line for line in lines(ep) if ep.actors[line[0]] != "scripted"} == replies


async def test_a_seat_that_has_heard_nothing_is_passed_over(fake_turn):
    """The gm speaks to ana alone and ana answers the gm alone, so bo is never called. And
    when nobody hears anybody the conversation ends instead of going round for ever."""
    seats = {"gm": {"cast": "scripted", "heard_by": ["ana"]}, "ana": {"cast": SUBJECT, "heard_by": ["gm"]}, "bo": "counterpart"}
    ep = await played(talk(seats, 6, lines=10), fake_turn({"subject-endpoint": [numbered()] * 10}, converse.TOOLS, CAST))
    assert [seat for seat, _ in lines(ep)] == ["gm", "ana", "gm", "ana", "gm", "ana"]
    assert {seat for seat, _ in deliveries(ep)} == {"ana"} and ep.state["converse"]["ended"] == "turn_budget"
    deaf = {"ana": {"cast": SUBJECT, "heard_by": []}, "bo": {"cast": "counterpart", "heard_by": []}}
    ep = await played(talk(deaf, 6), fake_turn([], converse.TOOLS, CAST))
    assert (deliveries(ep), ep.state["converse"]["ended"], ep.status) == ([], "silence", "completed")


async def test_an_addressed_line_reaches_its_listeners_and_its_speaker_and_nobody_else(fake_turn):
    """The opening, a model seat's `say_to` and a script entry, each addressed. bo hears nobody's
    public lines here (`heard_by`), and is still told what ana says to it."""
    seats = {"gm": "scripted", "ana": {"cast": SUBJECT, "heard_by": ["gm", "cy"]}, "bo": "counterpart", "cy": "counterpart"}
    task = talk(seats, 6, to=["ana"], script=[{"say": "A word, cy.", "to": ["cy", "ana"]}])
    refused = [say_to(["zed"], "x"), say_to(["ana"], "x"), say_to([], "x")]
    turn = fake_turn({"subject-endpoint": [{"tool_calls": [*refused, say_to(["bo", "bo"], "Hold the rope.")]}, "We cross at dawn.", "Agreed."],
                      "other-endpoint": ["bo answers", "cy answers"]}, converse.TOOLS, CAST)
    ep = await played(task, turn)
    said = [(e.actor, e.data["text"], e.data.get("to"), sorted(e.visible_to)) for e in ep.events if e.kind == "say"]
    assert said == [("gm", "Plan the crossing.", ["ana"], ["ana", "gm"]),
                    ("ana", "Hold the rope.", ["bo"], ["ana", "bo"]),
                    ("ana", "We cross at dawn.", None, ["ana", "cy", "gm"]),
                    ("bo", "bo answers", None, ["ana", "bo", "cy", "gm"]),
                    ("cy", "cy answers", None, ["ana", "bo", "cy", "gm"]),
                    ("gm", "A word, cy.", ["cy", "ana"], ["ana", "cy", "gm"]),
                    ("ana", "Agreed.", None, ["ana", "cy", "gm"])]
    assert deliveries(ep) == [("ana", "gm (to ana): Plan the crossing."),
                              ("bo", "ana (to bo): Hold the rope."),
                              ("cy", "ana: We cross at dawn.\n\nbo: bo answers"),
                              ("ana", "bo: bo answers\n\ncy: cy answers\n\ngm (to cy, ana): A word, cy.")]
    errors = [e.data["result"].get("error") for e in ep.events if e.kind == "tool"]
    assert errors == ["no seat named zed; the seats are gm, ana, bo, cy", "a seat cannot address itself", "it is addressed to nobody", None]
    assert (ep.state["converse"]["turns"], ep.state["converse"]["ended"]) == (6, "turn_budget")


async def test_two_seats_that_only_whisper_to_each_other_stop_at_the_turn_budget(fake_turn):
    """Each replies "" after its `say_to`, so no public line is ever logged after the opening;
    turns are still counted."""
    def whisper(messages, tools, endpoint):
        mine = "ana" if endpoint == "subject-endpoint" else "bo"
        return "" if messages[-1]["role"] == "tool" else {"tool_calls": [say_to(["bo" if mine == "ana" else "ana"], f"from {mine}")]}

    turn = fake_turn({"subject-endpoint": [whisper] * 20, "other-endpoint": [whisper] * 20}, converse.TOOLS, CAST)
    ep = await played(talk({"ana": SUBJECT, "bo": "counterpart"}, 7), turn)
    assert lines(ep) == [("ana", "Plan the crossing."), *[(("bo", "from bo"), ("ana", "from ana"))[n % 2] for n in range(6)]]
    assert deliveries(ep)[1:3] == [("ana", "bo (to ana): from bo"), ("bo", "ana (to bo): from ana")]
    assert (ep.state["converse"]["turns"], ep.state["converse"]["ended"], ep.status) == (7, "turn_budget", "completed")


async def test_one_endpoint_playing_three_seats_keeps_three_conversations(fake_turn):
    """Each call's prompt is its own seat's brief, the deliveries to that seat and that seat's
    replies: nothing another seat was sent or said outside what it was delivered."""
    turn = fake_turn([numbered()] * 20, converse.TOOLS)
    ep = await played(talk({"ana": SUBJECT, "bo": SUBJECT, "cy": SUBJECT}, 10), turn)
    assert {c.endpoint for c in ep.calls} == {"subject-endpoint"} and {c.actor for c in ep.calls} == {"bo", "cy", "ana"}
    sent: dict[str, list[dict]] = {seat: [{"role": "system", "content": f"You are {seat}."}] for seat in ep.actors}
    for call in ep.calls:
        delivered = next(e.data["message"] for e in ep.events if e.kind == "deliver" and e.data["turn_idx"] == call.turn_idx)
        sent[call.actor].append({"role": "user", "content": delivered})
        assert [(m["role"], m["content"]) for m in call.messages] == [(m["role"], m["content"]) for m in sent[call.actor]]
        sent[call.actor].append({"role": "assistant", "content": call.response})


async def test_a_conversation_of_scripted_seats_ends_when_the_script_does(fake_turn):
    ep = await played(talk({"gm": "scripted", "crow": "scripted"}, 50, lines=3), fake_turn([]))
    assert lines(ep) == [("gm", "Plan the crossing."), ("crow", "scripted line 0"), ("gm", "scripted line 1"), ("crow", "scripted line 2")]
    assert (ep.state["converse"]["ended"], ep.status, ep.calls) == ("script", "completed", [])


async def test_a_seat_ends_the_conversation_by_calling_conversation_end(fake_turn):
    replies = ["Let us wait for the tide.", {"tool_calls": [{"name": "conversation_end", "arguments": {"reason": "The plan is made."}}]},
               "Then we are agreed."]
    ep = await played(talk(TWO, 50, lines=10), fake_turn(replies, converse.TOOLS))
    assert lines(ep) == [("author", "Plan the crossing."), ("writer", "Let us wait for the tide."), ("author", "scripted line 0"),
                         ("writer", "Then we are agreed.")]
    assert (ep.state["converse"]["ended"], ep.status) == ("writer", "completed")


async def test_another_seat_cut_off_fails_the_episode_so_the_subject_is_not_scored_on_it(fake_turn):
    seats = {"author": "scripted", "writer": SUBJECT, "critic": "counterpart"}
    turn = fake_turn({"subject-endpoint": ["A first draft."], "other-endpoint": [{"text": "It could be", "finish": "length"}]},
                     converse.TOOLS, CAST, max_tokens=64)
    ep = await played(talk(seats, 50, lines=10), turn)
    assert (ep.status, ep.failure_reason) == ("failed", "critic: critic was cut off at max_tokens=64")
    row, _ = await score_episode(suite, ep, None, None)
    assert row["skipped"] == ep.failure_reason and "metrics" not in row


async def test_a_conversation_of_eight_saved_at_any_delivery_goes_on_as_the_original_did(fake_turn, tmp_path):
    """Killed after any delivery returned and continued from the store, the conversation asks
    the models the same things and ends as the same episode: the step's place is in the
    episode, and the scripted lines it spoke between two deliveries were saved with it. Every
    third turn says a line to the gm with `say_to` before it replies, which puts a line on the
    log in the middle of the turn."""
    replies = [r for n in range(40) for r in ([{"tool_calls": [say_to(["gm"], f"aside {n}")]}] if n % 3 == 0 else []) + [f"reply {n}"]]
    killed = []

    async def finish(store, ep, turn, copies=False):
        async def saved(ep, actor, message):
            await turn(ep, actor, message)
            store.append(ep)
            if copies:
                killed.append((shutil.copy(store.path, tmp_path / f"killed-{len(killed) + 1}.db"), len(turn.llm.requests)))

        await play(suite, [ep], saved)
        ep.complete()
        store.close(ep)

    [ep] = suite.setup(talk(WHISPERS, 30, lines=40), None)
    turn = fake_turn(replies, converse.TOOLS, CAST)
    with Store(tmp_path / "store.db") as store:
        store.append(ep)
        await finish(store, ep, turn, copies=True)
    assert len(killed) == len(deliveries(ep)) > 10
    for copy_path, answered in killed:
        with Store(copy_path) as store:
            [reopened] = store.episodes(closed=False)
            resumed = fake_turn(replies[answered:], converse.TOOLS, CAST)
            await finish(store, reopened, resumed)
            assert [r["messages"] for r in resumed.llm.requests] == [r["messages"] for r in turn.llm.requests[answered:]]
            assert reopened == ep and store.load(ep.ep_id) == ep
