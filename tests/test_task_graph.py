import json
import unittest
from pathlib import Path

from writing_agent.task_graph import (
    CheckpointV1,
    CommitV1,
    ContextRevisionV1,
    EnvironmentStateV1,
    EventV1,
    LineageRefV1,
    MessageV1,
    canonical_json,
    file_hash,
    load_canonical_json,
    tree_hash,
    validate_file_tree,
)

H = "0" * 64
FIXTURE = json.loads((Path(__file__).parent / "fixtures/task_graph_hashes.json").read_text())


class TaskGraphRecordsTest(unittest.TestCase):
    def state(self, files=None):
        files = {"a.txt": "hi"} if files is None else files
        return EnvironmentStateV1(
            instance_ref=H,
            position={"node_id": "n", "phase": "ready_writer"},
            files=files,
            tree_hash=tree_hash(files),
            history={"head": None, "seq": 0},
            context_ref=H,
            requirements_ref=H,
            decisions_ref=H,
            disclosures_ref=H,
            versions_ref=H,
            budgets_ref=H,
            rng_ref=H,
            external_inputs_ref=H,
            outcome_ref=H,
            provenance_ref=H,
        )

    def test_canonical_fixture_and_round_trip(self):
        self.assertEqual(canonical_json({"b": "é", "a": [1, True, None]}), FIXTURE["canonical"])
        message = MessageV1(content=("hello",), origin="a")
        context = ContextRevisionV1(messages=(message,), content_hash="1" * 64)
        state = self.state()
        checkpoint = CheckpointV1(state=state)
        event = EventV1(
            lineage_id="l",
            kind="writer_action",
            audience=("writer",),
            payload_ref=H,
            versions_ref=H,
            provenance_ref=H,
        )
        records = (
            (message, "message"),
            (context, "context"),
            (state, "state"),
            (checkpoint, "checkpoint"),
            (event, "event"),
        )
        for record, expected in records:
            self.assertEqual(record.identity(), FIXTURE[expected])
            self.assertEqual(type(record).from_json(record.to_json()), record)

    def test_duplicate_keys_and_noncanonical_input_rejected(self):
        with self.assertRaises(ValueError):
            load_canonical_json('{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            load_canonical_json('{"a": 1}')
        with self.assertRaises(ValueError):
            load_canonical_json('{"a":1.0}')

    def test_paths_utf8_and_mutation_sensitivity(self):
        validate_file_tree({"empty": "", "unicode/é.txt": "é\n"})
        for path in ("../x", "/x", "a//b", "a/./b", "a\\b", ""):
            with self.assertRaises(ValueError):
                validate_file_tree({path: "x"})
        with self.assertRaises(ValueError):
            validate_file_tree({"A": "x", "a": "x"})
        self.assertNotEqual(file_hash("x"), file_hash("x\n"))
        self.assertNotEqual(tree_hash({"a": "x"}), tree_hash({"a": "y"}))

    def test_state_and_checkpoint_invariants(self):
        state = self.state()
        with self.assertRaises(ValueError):
            EnvironmentStateV1(**{**state.to_dict(), "tree_hash": H})
        with self.assertRaises(ValueError):
            CheckpointV1(state=state, event_head="1" * 64)
        with self.assertRaises(ValueError):
            EnvironmentStateV1(**{**state.to_dict(), "in_flight_effects": ["call"]})

    def test_reference_records_and_mutation(self):
        commit = CommitV1(events=(H,), checkpoint=H)
        lineage = LineageRefV1(lineage_id="line", head_commit=commit.identity())
        self.assertEqual(CommitV1.from_json(commit.to_json()), commit)
        self.assertEqual(LineageRefV1.from_json(lineage.to_json()), lineage)
        self.assertNotEqual(
            commit.identity(), CommitV1(events=(H,), checkpoint="1" * 64).identity()
        )


if __name__ == "__main__":
    unittest.main()
