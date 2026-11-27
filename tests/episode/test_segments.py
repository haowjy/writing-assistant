import threading
from dataclasses import MISSING, fields

import pytest

from writing_assistant.episode import CallRecord, Episode
from writing_assistant.episode.segments import FORMAT, Store, Unreadable


def test_a_saved_episode_loads_back_whole(tmp_path):
    """Re-scoring reads only the stored episode: a field left out of the save, or one that
    loads back as another type, silently changes what a re-score sees."""
    arm = Episode(ep_id="t~pov_view", task_id="t", suite="1a", split="test", files={"a.md": "one"}, file_revs={"a.md": 1})
    arm.log("dm", "world", {"text": "hidden", "rolls": [3, 17]}, {"dm": "full", "pc": "redacted"})
    ep = arm.fork("twin_swap", baseline=True)
    ep.meta = {"driver": "direct", "row": {"k": 3, "mode": "instruction"}}
    ep.rollout_of, ep.group = ("t~pov_view", 2), "0123456789ab"
    ep.files["a.md"], ep.file_revs["a.md"] = "Zoë «x»", 2
    ep.state = {"world": {"secrets": {"s": {"text": "x", "known_by": ["npc"]}}}, "pov": {"arm": "pov_view"}}
    ep.actors = {"writer": "gemma-4-26b", "author": "scripted"}
    ep.actor_tools = {"writer": ["episode_read"]}
    ep.status, ep.failure_reason = "truncated", "writer was cut off at max_tokens=10"
    ep.deliver("writer", "write it")
    ep.calls.append(CallRecord(
        actor="writer", turn_idx=0, at_event=2, endpoint="gemma-4-26b", messages=[{"role": "user", "content": "write it"}],
        response="It was dark.", tool_calls=[{"id": "c1", "name": "episode_read", "arguments": {"path": "a.md"}}],
        tools_advertised=["episode_read"], usage={"input_tokens": 9, "output_tokens": 4, "cost_usd": 1.5e-05},
        ext={"logprobs": [-0.1]}, finish_reason="length", reasoning="Thinking about the dark.",
    ))
    # Every field carries a non-default value, so a field added later must be added here too.
    def defaults(obj):
        return [f.name for f in fields(obj)
                if f.default_factory is MISSING and getattr(obj, f.name) == f.default
                or f.default_factory is not MISSING and f.name != "seg" and getattr(obj, f.name) == f.default_factory()]

    assert defaults(ep) == [] and defaults(ep.calls[0]) == []

    with Store(tmp_path / "store.db") as store:
        store.close(ep)
    with Store(tmp_path / "store.db", readonly=True) as store:
        [back] = store.episodes()
    assert back == ep
    assert back.events[0].data["rolls"] == [3, 17] and back.view("pc").events[0].data == {}


def test_a_fork_made_past_what_its_parent_has_stored_is_refused(tmp_path):
    """A fork is stored after a position in its parent's log. Past the parent's last
    checkpoint nothing is stored there yet, so the fork would load with whatever the parent
    put there later: a resumed parent that played on differently would rewrite its child."""
    def say(ep, text):
        ep.log("dm", "say", {"text": text}, {"*": "full"})

    with Store(tmp_path / "store.db") as store:
        parent = Episode(ep_id="p", task_id="t", suite="1a", split="dev")
        say(parent, "stored")
        store.append(parent)
        say(parent, "in flight")
        ahead = parent.fork("ahead")
        with pytest.raises(ValueError, match="past the 1 events and 0 calls stored"):
            store.append(ahead)
        assert store.names(closed=False) == ["p"]
        store.append(parent)
        store.append(ahead)
        assert store.load("p~ahead") == ahead


@pytest.mark.parametrize("held_apart", ["local_dev", "test"])
def test_a_local_dev_or_test_segment_shares_a_store_with_no_other_splits(tmp_path, held_apart):
    """What leaves the machine and what a reader may see are decided per store as well as per
    segment, so personal text and the held-out split never sit in a file with anything else:
    not by being saved there, and not by being copied in as an ancestor."""
    def ep(ep_id, split):
        return Episode(ep_id=ep_id, task_id="t", suite="1a", split=split)

    with Store(tmp_path / "apart.db") as apart, Store(tmp_path / "dev.db") as dev:
        apart.append(ep("mine", held_apart))
        dev.append(ep("d", "dev"), ep("tr", "train"))
        for mixing in (lambda: apart.append(ep("other", "dev")), lambda: dev.append(ep("other", held_apart)),
                       lambda: apart.copy_chain("mine", dev), lambda: dev.copy_chain("d", apart)):
            with pytest.raises(ValueError, match="share a store with no other split's"):
                mixing()
        splits = "SELECT DISTINCT split FROM segments ORDER BY split"
        assert apart.db.execute(splits).fetchall() == [(held_apart,)] and dev.db.execute(splits).fetchall() == [("dev",), ("train",)]


def _closed(store, *names):
    store.close(*(Episode(ep_id=name, task_id="t", suite="1a", split="dev") for name in names))


def _row(ep_id, at, **more):
    return {"ep_id": ep_id, "scored_at": at, "metrics": {}} | more


def _items(ep_id, n):
    return [{"id": f"{ep_id}#{i}", "scored": ep_id, "ep_id": ep_id} for i in range(n)]


def test_a_re_score_with_fewer_judged_items_leaves_no_judgment_of_the_old_row(tmp_path):
    with Store(tmp_path / "store.db") as store:
        _closed(store, "a", "b")
        store.put_scores("eval", [_row("a", "1"), _row("b", "1")], _items("a", 3) + _items("b", 2))
        store.put_scores("eval", [_row("a", "2")], _items("a", 1))
        assert [j["id"] for j in store.judgments("eval")] == ["b#0", "b#1", "a#0"]
        assert store.judgments("eval", scored="a") == _items("a", 1)


def test_which_row_of_an_episode_is_read_does_not_depend_on_the_order_rows_arrived_in(tmp_path):
    """Rows are only added, and two stores may each have scored an episode. Whatever order
    they are copied into a third in, readers take the same row with its own judge items: the
    later one, except that a row saying the scorer raised (a judge was down) never hides one
    that does not."""
    def failed(at):
        return {"ep_id": "a", "scored_at": at, "error": "judge down"}

    pairs = {"a failed re-score hides no good row": (_row("a", "1"), failed("2")),
             "the later of two good rows": (_row("a", "2", note="later"), _row("a", "1")),
             "the later of two failures": (failed("2"), failed("1"))}
    for case, (read, other) in pairs.items():
        seen = []
        for n, order in enumerate(((read, other), (other, read))):
            with Store(tmp_path / f"{case}-{n}-0.db") as one, Store(tmp_path / f"{case}-{n}-1.db") as two, \
                    Store(tmp_path / f"{case}-{n}.db") as third:
                _closed(one, "a")
                one.copy(["a"], two)
                for store, row in zip((one, two), order):
                    store.put_scores("eval", [row], _items("a", 2 if row is read else 3))
                    store.copy(["a"], third)
                seen.append((third.scores("eval"), third.judgments("eval"), len(third.judgments("eval", superseded=True))))
        assert seen[0] == seen[1] == ({"a": read}, _items("a", 2), 5), case


def test_rows_from_two_judges_under_one_profile_name_do_not_settle_a_group_together(tmp_path):
    """A profile is a name in one machine's models.toml: two machines' `eval` can be two
    judges, and each row says which answered."""
    def stamp(model):
        return {"use": "fact", "endpoint": "e", "model": model, "prompt": "fact-v1"}

    with Store(tmp_path / "store.db") as store:
        source = Episode(ep_id="s", task_id="t", suite="1a", split="dev")
        store.append(source)
        members = [source.rollout(0, i, "g") for i in (0, 1)]
        for ep in members:
            ep.meta = {"group_size": 2}
        store.close(*members)
        store.put_scores("eval", [_row("s@0.g.0", "1", judges=[stamp("here")]), _row("s@0.g.1", "1", judges=[stamp("there")])])
        assert store.groups() == {"g": {"size": 2, "sources": {"s": 2}, "played": True}}
        assert store.settled("none")["g"] == {"size": 2, "sources": {"s": 0}, "settled": False}
        assert store.settled("eval") == {"g": {"size": 2, "sources": {"s": 2}, "settled": False}}
        store.put_scores("eval", [_row("s@0.g.1", "2", judges=[stamp("here")])])
        assert store.settled("eval")["g"]["settled"]


def test_a_format_1_store_is_brought_up_to_date_by_a_writer_and_refused_by_a_reader(tmp_path):
    """A store file outlives formats, and the steps between them only add. Two writers that
    open an old file together must not both apply a step."""
    path = tmp_path / "store.db"
    with Store(path) as store:
        base = Episode(ep_id="a", task_id="t", suite="1a", split="dev", files={"a.md": "one"}, file_revs={"a.md": 1})
        base.log("dm", "say", {"text": "hello"}, {"*": "full"})
        store.close(base, base.fork("twin", baseline=True))
        before = list(store.episodes())
        for undo in ("ALTER TABLE episodes DROP COLUMN start", "DROP TABLE judgments", "DROP TABLE scores",
                     "DROP INDEX segments_rollout", "ALTER TABLE segments DROP COLUMN grp", "PRAGMA user_version = 1"):  # format 1, as it was written
            store.db.execute(undo)
    with pytest.raises(Unreadable, match="store format 1.*Opening it once for writing"):
        Store(path, readonly=True)

    together, failed = threading.Barrier(2), []

    def writer():
        together.wait()
        try:
            Store(path).db.close()
        except Exception as e:
            failed.append(e)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert failed == []
    with Store(path, readonly=True) as store:
        assert store.db.execute("PRAGMA user_version").fetchone()[0] == FORMAT == 3
        assert list(store.episodes()) == before and store.scores("eval") == {}


def test_settled_answers_from_one_state_of_a_store_a_group_is_arriving_in(tmp_path, monkeypatch):
    """`settled` asks what is played and then reads the rows: a group pushed and scored by
    another process between the two is not half there. (A world file, which keeps a log: in
    a run directory the push would wait for the read to end.)"""
    path = tmp_path / "store.db"
    with Store(path) as store:
        store.db.execute("PRAGMA journal_mode = WAL")
        source = Episode(ep_id="s", task_id="t", suite="1a", split="dev")
        store.append(source)

        def arrives(group):
            with Store(path) as other:  # another process: its own connection
                ep = source.rollout(0, 0, group)
                ep.meta = {"group_size": 1}
                other.close(ep)
                other.put_scores("eval", [_row(ep.ep_id, "1")])

        arrives("g")
        groups = Store.groups

        def groups_then_a_push_lands(self, group=None):
            found = groups(self, group)
            arrives("late")
            return found

        with Store(path, readonly=True) as reader:
            monkeypatch.setattr(Store, "groups", groups_then_a_push_lands)
            assert reader.settled("eval") == {"g": {"size": 1, "sources": {"s": 1}, "settled": True}}
            monkeypatch.setattr(Store, "groups", groups)
            assert set(reader.settled("eval")) == {"g", "late"}
