"""The library (`episode/library.py`): one test per failure case of its design, and each step
that reads and then acts run with a second writer in between."""

import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import closing

import pytest

from writing_assistant.episode import Episode
from writing_assistant.episode import library as library_module
from writing_assistant.episode.library import Damaged, Library
from writing_assistant.episode.segments import Store

TABLES = ("segments", "checkpoints", "episodes", "scores", "judgments")
NOTHING = Counter(episodes=0, segments=0, checkpoints=0, scores=0)


def root(name="sel", split="dev") -> Episode:
    return Episode(ep_id=name, task_id=name, suite="situations", split=split, files={"kb.md": "The harbour town of Sel. " * 200},
                   file_revs={"kb.md": 1})


@pytest.fixture
def lib(tmp_path) -> Library:
    lib = Library(tmp_path / "library")
    lib.create(root())
    return lib


def played(lib: Library, path, group: str, n: int = 2, world: str = "sel", close: bool = True) -> Store:
    """A worker's store at `path` holding `n` rollouts of the world's root, closed and scored
    under `eval` (or left open), as a run of group `group` would leave it."""
    store = Store(path)
    lib.checkout(world, [world], store)
    with lib.open(world) as home:
        source = home.load(world, at=0)
    for i in range(n):
        ep = source.rollout(0, i, group)
        ep.meta = {"group_size": n}
        ep.log("writer", "say", {"text": f"{group} {i} " + "words " * 50}, {"*": "full"})
        ep.write_file("writer", "draft.md", f"A draft by {group} {i}.")
        store.append(ep)
        if close:
            ep.complete()
            store.close(ep)
            store.put_scores("eval", [{"ep_id": ep.ep_id, "scored_at": "2026-11-26T10:00:00.000000Z", "metrics": {"m": float(i)}}],
                             [{"scored": ep.ep_id, "id": f"{ep.ep_id}#0", "metric": "m", "question": "q", "model": True}])
    return store


def rows(path) -> dict[str, list]:
    """Every row of a store file, in an order that does not depend on when it arrived."""
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        return {t: sorted(db.execute(f"SELECT * FROM {t}").fetchall(), key=repr) for t in TABLES}


def side_files(folder) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.endswith(("-wal", "-shm", "-journal")))


def after(seconds: float, act) -> threading.Thread:
    """`act` in another thread, started now; the caller checks it is still waiting after `seconds`."""
    thread = threading.Thread(target=act)
    thread.start()
    time.sleep(seconds)
    return thread


def overwrite(path, at: int, data: bytes) -> None:
    with open(path, "r+b") as f:
        f.seek(at)
        f.write(data)


def test_a_world_is_created_whole_or_not_at_all_and_two_creates_of_one_name_cannot_both_win(tmp_path, monkeypatch):
    lib = Library(tmp_path / "library")
    listed_half_way = []
    link = os.link

    def linked(src, dst):  # the file is written and not yet named: a listing must not meet it
        listed_half_way.append(lib.worlds())
        link(src, dst)

    monkeypatch.setattr(os, "link", linked)
    assert lib.create(root()) == "sel" and listed_half_way == [[]]
    first = rows(lib.path("sel"))
    with pytest.raises(ValueError, match="a world named 'sel' already exists"):
        lib.create(root())
    assert rows(lib.path("sel")) == first and {p.name for p in lib.worlds_dir.iterdir()} - set(side_files(lib.worlds_dir)) - {"sel.lock"} == {"sel.db"}
    lib.create(root("private", "local_dev"))
    assert [(w.name, w.split) for w in lib.worlds()] == [("private", "local_dev"), ("sel", "dev")]
    with pytest.raises(ValueError, match="a world's root has no parent"):
        lib.create(Episode(ep_id="child", task_id="child", suite="situations", split="dev", parent=("some-seg", 0, 0)))
    with pytest.raises(ValueError, match="is no world's name"):
        lib.create(root("../sel"))
    with lib.open("sel") as world:
        assert (world.starts(), world.names(), world.load("sel").files.keys()) == (["sel"], [], {"kb.md"})


def test_a_checked_out_position_is_read_by_that_store_as_nothing_it_played(lib, tmp_path):
    """The root arrives unlisted with no rows: scored, reported and resumed by nothing."""
    with played(lib, tmp_path / "a.db", "a") as a:
        lib.push(a)
        first = a.names()[0]
    with Store(tmp_path / "worker.db") as worker:
        lib.checkout("sel", ["sel", first], worker)
        assert (worker.names(), worker.names(closed=False), worker.starts(), worker.judgings(), worker.groups()) == ([], [], [], [], {})
        with lib.open("sel") as world:
            assert worker.load(world.chain(first)[-1]) == world.load(first)


def test_two_pushes_land_in_either_order_and_a_push_repeated_changes_nothing(tmp_path):
    one, two = Library(tmp_path / "one"), Library(tmp_path / "two")
    start = root()
    one.create(start)
    two.create(start)
    with played(one, tmp_path / "a.db", "a") as a, played(one, tmp_path / "b.db", "b", n=3) as b:
        assert one.push(a) == Counter(episodes=2, segments=2, checkpoints=4, scores=2)
        assert one.push(b) == Counter(episodes=3, segments=3, checkpoints=6, scores=3)
        two.push(b)
        two.push(a)
        assert rows(one.path("sel")) == rows(two.path("sel"))
        assert one.push(a) == one.push(b) == NOTHING and rows(one.path("sel")) == rows(two.path("sel"))
    with one.open("sel") as world:
        assert len(world.names()) == 5 and {g: v["played"] for g, v in world.groups().items()} == {"a": True, "b": True}
        assert len(world.scores("eval")) == 5 and len(world.judgments("eval")) == 5


def test_a_push_cut_short_leaves_nothing_of_it_and_repeated_makes_the_group_whole(lib, tmp_path, monkeypatch):
    before = rows(lib.path("sel"))
    with played(lib, tmp_path / "a.db", "a", n=3) as a:
        add_score, calls = Store._add_score, []

        def dies_at_the_second(self, *args):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return add_score(self, *args)

        monkeypatch.setattr(Store, "_add_score", dies_at_the_second)
        with pytest.raises(KeyboardInterrupt):
            lib.push(a)
        assert rows(lib.path("sel")) == before
        monkeypatch.undo()
        lib.push(a)
    with lib.open("sel") as world:
        assert world.groups() == {"a": {"size": 3, "sources": {"sel": 3}, "played": True}} and world.settled("eval")["a"]["settled"]


def test_an_open_episode_and_a_segment_of_another_world_are_each_refused_by_name(lib, tmp_path):
    lib.create(root("other"))
    before = rows(lib.path("sel")), rows(lib.path("other"))
    with played(lib, tmp_path / "a.db", "a", close=False) as a:
        [first, second] = a.names(closed=False)
        assert lib.push(a) == Counter()  # nothing is closed, so there is nothing to push
        with pytest.raises(ValueError, match=f"{first}, {second} cannot be pushed: only a closed episode"):
            lib.push(a, [first, second])
    with played(lib, tmp_path / "b.db", "b") as b, played(lib, tmp_path / "o.db", "o", world="other") as other:
        other.copy(other.names(), b)  # one store holding rollouts of two worlds
        foreign = other.names()
        with pytest.raises(ValueError, match=f"{foreign[0]}, {foreign[1]}: not of the world 'sel'"):
            lib.push(b)
    assert (rows(lib.path("sel")), rows(lib.path("other"))) == before


def test_two_stores_that_closed_one_episode_differently_cannot_both_be_pushed(lib, tmp_path):
    """A run directory copied while an episode was open, and both copies played on."""
    with played(lib, tmp_path / "a.db", "a", n=1, close=False) as a:
        [name] = a.names(closed=False)
    (tmp_path / "copy.db").write_bytes((tmp_path / "a.db").read_bytes())
    for path, line in ((tmp_path / "a.db", "one ending"), (tmp_path / "copy.db", "another ending")):
        with Store(path) as store:
            [ep] = store.episodes(closed=False)
            ep.log("writer", "say", {"text": line}, {"*": "full"})
            ep.complete()
            store.close(ep)
    with Store(tmp_path / "a.db") as a, Store(tmp_path / "copy.db") as copy:
        lib.push(a)
        held = rows(lib.path("sel"))
        with pytest.raises(ValueError, match=f"checkpoint 1 of {name} is stored there with another body"):
            lib.push(copy)
    assert rows(lib.path("sel")) == held


def test_a_stale_push_after_a_rescore_adds_a_row_no_reader_takes(lib, tmp_path):
    with played(lib, tmp_path / "a.db", "a") as a:
        [first, _] = a.names()
        with lib.writing("sel") as world:  # the world already holds the episode, scored again since
            a.copy([first], world)
            world.put_scores("eval", [{"ep_id": first, "scored_at": "2026-11-26T12:00:00.000000Z", "metrics": {"m": 9.0}}])
        assert lib.push(a)["scores"] == 1  # the second episode's row; the first's old row was there
        with lib.open("sel") as world:
            assert world.scores("eval")[first]["metrics"] == {"m": 9.0}
            assert world.db.execute("SELECT COUNT(*) FROM scores WHERE ep_id = ?", (first,)).fetchone() == (2,)


def test_a_push_takes_one_state_of_a_store_a_live_run_is_writing(lib, tmp_path, monkeypatch):
    """The names are read and then copied in one read transaction: a writer that commits in
    between waits for it, and what it wrote goes with the next push."""
    with played(lib, tmp_path / "a.db", "a") as a:
        copy, late = Store.copy, []

        def the_run_closes_another(self, names, into, scores=True):
            def close():
                with played(lib, tmp_path / "a.db", "later", n=1):
                    pass

            late.append(after(0.3, close))
            assert late[0].is_alive()  # its commit waits for this read
            return copy(self, names, into, scores)

        monkeypatch.setattr(Store, "copy", the_run_closes_another)
        assert lib.push(a)["episodes"] == 2
        late[0].join()
        monkeypatch.undo()
        assert len(a.names()) == 3 and lib.push(a)["episodes"] == 1


def test_a_checkout_takes_one_state_of_a_world_and_holds_no_push_up(lib, tmp_path, monkeypatch):
    """A push lands while the checkout is half-way through its read: it does not wait (the
    world is in write-ahead mode), and the checkout goes on reading the state it began with."""
    with played(lib, tmp_path / "a.db", "a") as a, played(lib, tmp_path / "b.db", "b") as b:
        lib.push(a)
        [first, second] = a.names()
        chain, seen = Store.copy_chain, []

        def a_push_lands(self, name, into):
            if name == second:  # the first chain has been read: the read transaction has begun
                assert lib.push(b)["episodes"] == 2
                seen.append(len(self.names()))  # the checkout's own reading of the world
            return chain(self, name, into)

        monkeypatch.setattr(Store, "copy_chain", a_push_lands)
        with Store(tmp_path / "worker.db") as worker:
            lib.checkout("sel", [first, second], worker)
            with lib.open("sel") as world:
                assert seen == [2] and len(world.names()) == 4
                assert [worker.load(world.chain(n)[-1]) for n in (first, second)] == [world.load(first), world.load(second)]


def test_a_snapshot_is_one_verified_state_and_the_two_newest_are_kept(lib, tmp_path, monkeypatch):
    """A push lands between the check of the live file and the copy: the copy is whatever
    one state the backup took, and it is verified before it gets its name."""
    with played(lib, tmp_path / "a.db", "a") as a, played(lib, tmp_path / "b.db", "b") as b:
        lib.push(a)
        verify, seen = library_module._verify, []

        def then_a_push_lands(path):
            verify(path)
            if not seen:
                seen.append(path)
                lib.push(b)

        monkeypatch.setattr(library_module, "_verify", then_a_push_lands)
        first = lib.snapshot("sel")
        monkeypatch.undo()
        library_module._verify(first)
        assert rows(first) == rows(lib.path("sel"))  # the push landed before the backup began
    taken = [first, lib.snapshot("sel"), lib.snapshot("sel"), lib.snapshot("sel")]
    assert lib.snapshots("sel") == taken[:1:-1] == sorted(lib.snapshots_dir.iterdir(), reverse=True)  # each one file, no other


@pytest.mark.parametrize("damage", ["header", "pages", "emptied"])
def test_a_damaged_world_fails_verify_is_never_snapshotted_and_is_restored_from_the_newest_good_snapshot(lib, tmp_path, damage):
    with played(lib, tmp_path / "a.db", "a") as a, played(lib, tmp_path / "b.db", "b") as b:
        lib.push(a)
        old = lib.snapshot("sel")
        lib.push(b)
        good = lib.snapshot("sel")
        kept = rows(good)
        with played(lib, tmp_path / "c.db", "c") as c:
            lib.push(c)  # pushed since the snapshot: lost by the restore, pushed again from the directory
            size = lib.path("sel").stat().st_size
            if damage == "header":
                overwrite(lib.path("sel"), 0, b"\0" * 100)
            elif damage == "pages":
                overwrite(lib.path("sel"), 4096, os.urandom(size - 4096))
            else:
                lib.path("sel").write_bytes(b"")
            with pytest.raises(Damaged):
                lib.verify("sel")
            with pytest.raises(Damaged):
                lib.snapshot("sel")
            assert lib.snapshots("sel") == [good, old]
            overwrite(good, 4096, os.urandom(8192))  # the newest snapshot is damaged too: the one before it is taken
            with pytest.raises(Damaged):
                library_module._verify(good)
            aside = lib.restore("sel")
            assert aside.name.startswith("sel.db.damaged-") and aside.stat().st_size == (0 if damage == "emptied" else size)
            lib.verify("sel")
            assert rows(lib.path("sel")) == rows(old) != kept
            assert lib.push(b)["episodes"] == 2 and lib.push(c)["episodes"] == 2
    with lib.open("sel") as world:
        assert {g: v["played"] for g, v in world.groups().items()} == {"a": True, "b": True, "c": True}


DIES_MID_WRITE = """
import os, sqlite3, sys
db = sqlite3.connect(sys.argv[1], isolation_level=None)
db.execute("PRAGMA cache_size = 5")
row = "INSERT INTO checkpoints (seg, n, kind, events, calls, body) VALUES (?, ?, 'turn', 0, 0, ?)"
db.execute(row, ("committed-by-the-dead-writer", 0, b"x"))      # one transaction that landed
db.execute("BEGIN IMMEDIATE")                                   # and one that never ended
db.executemany(row, [("dead-writer", n, os.urandom(4000)) for n in range(400)])
os.kill(os.getpid(), 9)
"""


def test_what_a_dead_writer_left_beside_a_world_is_not_applied_to_the_restored_file(lib, tmp_path):
    """A writer killed mid-transaction leaves a log beside the world file holding what it had
    committed and what it had not. Left there when the file is replaced, SQLite would apply
    it to the restored file, which would then mix the two and still pass the integrity check."""
    with played(lib, tmp_path / "a.db", "a") as a:
        lib.push(a)
        good = lib.snapshot("sel")
        with played(lib, tmp_path / "b.db", "b") as b:
            lib.push(b)
    assert side_files(lib.worlds_dir) == []
    assert subprocess.run([sys.executable, "-c", DIES_MID_WRITE, str(lib.path("sel"))]).returncode == -9
    log = lib.path("sel").with_name("sel.db-wal")
    assert log.stat().st_size > 400 * 4000
    with lib.open("sel") as world:  # readers see what was committed and nothing of the transaction that died
        assert world.db.execute("SELECT seg, COUNT(*) FROM checkpoints WHERE seg LIKE '%dead-writer' GROUP BY seg").fetchall() == [
            ("committed-by-the-dead-writer", 1)]
    aside = lib.restore("sel")
    assert rows(lib.path("sel")) == rows(good)
    lib.verify("sel")
    with Store(aside, readonly=True) as was:  # the file moved aside is the world with all that was committed to it
        assert set(was.groups()) == {"a", "b"} and was.db.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE seg LIKE '%dead-writer'").fetchone() == (1,)


def test_a_restore_waits_for_a_push_and_for_a_reader_and_nobody_opens_the_world_while_it_is_replaced(lib, tmp_path, monkeypatch):
    with played(lib, tmp_path / "a.db", "a") as a, played(lib, tmp_path / "b.db", "b") as b:
        lib.push(a)
        good = lib.snapshot("sel")
        copy, restoring = Store.copy, []

        def a_restore_starts(self, names, into, scores=True):
            restoring.append(after(0.3, lambda: lib.restore("sel")))
            assert restoring[0].is_alive()  # it waits for this push to let go of the world
            return copy(self, names, into, scores)

        monkeypatch.setattr(Store, "copy", a_restore_starts)
        assert lib.push(b)["episodes"] == 2
        restoring[0].join()
        monkeypatch.undo()
        assert rows(lib.path("sel")) == rows(good)
        assert lib.push(b)["episodes"] == 2

        # A reader that has the world open holds a restore off until it is done reading.
        with lib.open("sel") as reader:
            waiting = after(0.3, lambda: lib.restore("sel"))
            assert waiting.is_alive() and len(reader.names()) == 4
        waiting.join()

        # A reader arriving while the file is being replaced waits, and reads the restored one:
        # opened then, it would read the old file through side files the new one inherits.
        replace, arrived, seen = os.replace, [], []

        def readers_arrive(src, dst):
            def read():
                with lib.open("sel") as world:
                    seen.append(len(world.names()))

            arrived.extend([after(0.2, read), after(0.2, lambda: seen.append([w.name for w in lib.worlds()]))])
            assert all(thread.is_alive() for thread in arrived)
            replace(src, dst)

        lib.push(b)
        monkeypatch.setattr(library_module.os, "replace", readers_arrive)
        lib.restore("sel")
        for thread in arrived:
            thread.join()
        assert sorted(seen, key=str) == [2, ["sel"]]


def test_one_master_per_world(lib):
    with lib.master_lock("sel"):
        with pytest.raises(RuntimeError, match="another master is alive on this world"):
            with lib.master_lock("sel"):
                pass
    with lib.master_lock("sel"):
        pass
    with pytest.raises(LookupError, match="no world named 'nowhere'"):
        lib.verify("nowhere")
