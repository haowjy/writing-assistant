"""Contracts owned by the canonical writer-call and tool-effect seam."""

from __future__ import annotations

import json
import random
import unittest

from writing_agent.task_graph import canonical_json
from writing_agent.task_graph_calls import (
    ToolQueueEntry,
    apply_effect,
    intake_message,
    parse_calls,
    tool_effect_contract,
    validate_intake_record,
)
from writing_agent.task_graph_errors import AdapterContractError, WriterRuntimeError
from writing_agent.task_graph_local import LocalTextToolProvider
from writing_agent.task_graph_ports import EnvironmentAction, EnvironmentSnapshot, EnvironmentSpec
from writing_agent.task_graph_writer import TransactionalWriterV1

ALLOWED = frozenset({"list_dir", "read_file", "search", "write_file", "patch_file", "ask_author"})


def _legacy_parse(batch, prior, scripted):
    queue, metadata = TransactionalWriterV1._parsed_calls(batch, "r", 0, ALLOWED, set(prior))
    if scripted:
        for call, meta in zip(queue, metadata, strict=True):
            if call["name"] == "ask_author" and meta["validation_error"] is None:
                try:
                    _ask_semantics(call["arguments"])
                except (TypeError, ValueError) as exc:
                    meta["validation_error"] = str(exc)
                    call["name"], call["arguments"] = "invalid_call", {}
    return queue, [item["validation_error"] for item in metadata]


def _ask_semantics(arguments):
    if arguments.get("question") == "q?":
        raise ValueError("decision is not public")


def _new_parse(batch, prior, scripted):
    record = intake_message({"content": "hello", "tool_calls": batch})
    encoded = canonical_json(record)
    decoded = json.loads(encoded)
    parsed = parse_calls(
        record,
        id_prefix="r:call:0",
        allowed=ALLOWED,
        prior_raw_ids=prior,
        ask_semantics=_ask_semantics if scripted else None,
    )
    round_trip = parse_calls(
        decoded,
        id_prefix="r:call:0",
        allowed=ALLOWED,
        prior_raw_ids=prior,
        ask_semantics=_ask_semantics if scripted else None,
    )
    if parsed != round_trip:
        raise AssertionError("parsed calls changed after canonical JSON round trip")
    queue = [
        {"call_id": entry.call_id, "name": entry.name, "arguments": entry.arguments}
        for entry in parsed
    ]
    return queue, [entry.rejection for entry in parsed]


def _diff_outcome(parser, batch, prior, scripted):
    try:
        return ("ok", parser(batch, prior, scripted))
    except WriterRuntimeError as exc:
        return ("error", type(exc), str(exc))


def _deep(depth):
    value = "x"
    for _ in range(depth):
        value = [value]
    return value


def _fuzz_generator(seed):
    """The judo generator, kept local so differential runs remain reproducible."""
    rng = random.Random(seed)
    ids = ["a", "b", "", " a", "a b", "x" * 300, 5, None, "\u200b", "tab\t", "\ud800", "é", "a\x01"]
    names = [
        "read_file",
        "write_file",
        "list_dir",
        "search",
        "patch_file",
        "ask_author",
        "shell",
        "",
        "read file",
        "x" * 300,
        None,
        3,
        "re\x01ad",
        "\ud800",
    ]
    paths = [
        "draft.txt",
        ".",
        "notes/",
        "../etc",
        "/abs",
        "a//b",
        "notes/source.md",
        "",
        "a\\b",
        "./a",
    ]

    def raw_id():
        return rng.choice(ids) if rng.random() < 0.3 else rng.choice(["a", "b", "c", "d", "é"])

    def arguments():
        kind = rng.randrange(14)
        if kind == 0:
            return json.dumps({"path": rng.choice(paths)})
        if kind == 13 and rng.random() < 0.5:
            return json.dumps(
                {
                    "decision_ids": ["d1"],
                    "question": rng.choice(["q?", "ok?"]),
                    "proposals": [],
                    "option_refs": [],
                }
            )
        if kind == 1:
            return json.dumps({"path": rng.choice(paths), "content": "hi"})
        if kind == 2:
            return "{not json"
        if kind == 3:
            return '{"a": NaN}'
        if kind == 4:
            return '{"a": "1", "a": "2"}'
        if kind == 5:
            return json.dumps([1, 2])
        if kind == 6:
            return {"path": "draft.txt"}
        if kind == 7:
            return json.dumps({"path": 5})
        if kind == 8:
            return json.dumps(
                {
                    "decision_ids": rng.choice([["d1"], ["d1", "d1"], []]),
                    "question": rng.choice(["q?", "ok?", " "]),
                    "proposals": rng.choice([[], [{"id": "p", "text": "t"}], [{"id": "p"}]]),
                    "option_refs": [],
                }
            )
        if kind == 9:
            return json.dumps({"decision_ids": "d1"})
        if kind == 10:
            return "x" * 70_000
        if kind == 11:
            return {str(index): "v" for index in range(70)}
        if kind == 12:
            return json.dumps({"path": "draft.txt", "k\ud800": "v"}, ensure_ascii=True)
        return json.dumps({})

    def raw_call():
        kind = rng.randrange(12)
        if kind == 0:
            return "not a dict"
        if kind == 1:
            return _deep(80)
        if kind == 2:
            return {"id": raw_id(), "type": "function"}
        if kind == 3:
            return {
                "id": raw_id(),
                "type": "fn",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        if kind == 4:
            return {
                "id": raw_id(),
                "type": "function",
                "function": {"name": rng.choice(names)},
            }
        if kind == 5:
            return {
                "id": raw_id(),
                "type": "function",
                "function": "read_file",
                "x": 1,
            }
        call = {
            "id": raw_id(),
            "type": "function",
            "function": {"name": rng.choice(names), "arguments": arguments()},
        }
        if kind == 6:
            call["function"]["extra"] = 1
        return call

    yield ({"x": 1}, (), False)
    yield ("s", (), False)
    yield ([], (), False)
    for _ in range(30_000):
        batch = [raw_call() for _ in range(rng.choice([1, 1, 1, 2, 3]))]
        prior = tuple(rng.sample(["a", "b", "é"], rng.randrange(3)))
        yield batch, prior, rng.random() < 0.5


def _good_call(arguments="{}", name="read_file", raw_id="z"):
    return {"id": raw_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _targeted_cases():
    deep = _deep(200)
    cyclic = {}
    cyclic["self"] = cyclic
    cases = [
        ("bytes value", [_good_call({"path": b"x"})], ()),
        ("float value", [_good_call({"path": 1.5})], ()),
        ("tuple arguments", [_good_call(("x",))], ()),
        ("int key", [_good_call({1: "v"})], ()),
        ("surrogate id", [_good_call("{}", raw_id="\ud800")], ()),
        ("surrogate argument key", [_good_call({"\ud800": "v"})], ()),
        ("200 deep", [_good_call({"x": deep})], ()),
        ("10MB arguments string", [_good_call("x" * 10_000_000)], ()),
        ("long id", [_good_call("{}", raw_id="x" * 300)], ()),
        ("duplicate within", [_good_call("{}", raw_id="d"), _good_call("{}", raw_id="d")], ()),
        (
            "ask mixed",
            [
                _good_call(
                    '{"decision_ids":[],"question":"?","proposals":[],"option_refs":[]}',
                    "ask_author",
                ),
                _good_call("{}", "read_file"),
            ],
            (),
        ),
        (
            "ask shape error",
            [
                _good_call(
                    '{"decision_ids":[],"question":"?","proposals":[],"option_refs":[]}',
                    "ask_author",
                )
            ],
            (),
        ),
        ("non-list tool_calls", "bad", ()),
        (
            "cyclic dict",
            [
                {
                    "id": "x",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": cyclic},
                }
            ],
            (),
        ),
        ("duplicate prior", [_good_call("{}", raw_id="prior")], ("prior",)),
    ]
    return cases


class IntakeAndParserTests(unittest.TestCase):
    def test_intake_fixed_point_and_marker_key_escape(self):
        message = {
            "content": {"$noncanonical": "float", "repr": "nan"},
            "tool_calls": [_good_call({"$noncanonical": "bytes", "hex": "00"})],
        }
        record = intake_message(message)
        self.assertEqual(record["content"]["$noncanonical"], "mapping")
        self.assertEqual(
            record["calls"][0]["value"]["function"]["arguments"]["$noncanonical"], "mapping"
        )
        encoded = canonical_json(record)
        restored = json.loads(encoded)
        parsed = parse_calls(record, id_prefix="r:call:0", allowed=ALLOWED)
        self.assertEqual(parsed, parse_calls(restored, id_prefix="r:call:0", allowed=ALLOWED))
        self.assertIsNone(parsed[0].rejection)
        self.assertEqual(parsed[0].arguments, {"$noncanonical": "bytes", "hex": "00"})

        forged = {
            "$sampled_message_v1": True,
            "content": 1.5,
            "tool_calls_was_list": True,
            "calls": [{"bounded": True, "value": (1,)}],
        }
        canonicalized = intake_message(forged)
        self.assertIsNot(canonicalized, forged)
        canonical_json(canonicalized)
        self.assertEqual(canonicalized["content"]["$noncanonical"], "float")

    def test_recorded_intake_rejects_malformed_markers(self):
        malformed_markers = (
            (
                "unhashable mapping key",
                {"$noncanonical": "mapping", "items": [[[1], 2]]},
            ),
            ("short mapping pair", {"$noncanonical": "mapping", "items": [[1]]}),
            ("invalid codepoint", {"$noncanonical": "surrogate-string", "codepoints": [-1]}),
            ("invalid float repr", {"$noncanonical": "float", "repr": "not-a-float"}),
            ("invalid bytes hex", {"$noncanonical": "bytes", "hex": "not-hex"}),
            (
                "unexpected marker key",
                {"$noncanonical": "float", "repr": "1.0", "extra": True},
            ),
        )
        for name, marker in malformed_markers:
            record = {
                "$sampled_message_v1": True,
                "content": None,
                "tool_calls_was_list": True,
                "calls": [{"bounded": True, "value": marker}],
            }
            with self.subTest(marker=name), self.assertRaises(ValueError):
                parse_calls(record, id_prefix="r:call:0", allowed=ALLOWED)

    def test_recorded_intake_validator_rejects_malformed_wrapper(self):
        with self.assertRaises(ValueError):
            validate_intake_record(
                {
                    "$sampled_message_v1": True,
                    "content": None,
                    "tool_calls_was_list": True,
                    "calls": [{"bounded": False, "value": {"not": "bounded-call"}}],
                }
            )

    def test_intake_encodes_non_json_values_and_non_array_calls(self):
        record = intake_message({"content": (b"x", 1.5), "tool_calls": "not calls"})
        self.assertEqual(record["content"]["$noncanonical"], "tuple")
        self.assertFalse(record["tool_calls_was_list"])
        self.assertEqual(record["calls"], "not calls")
        with self.assertRaises(WriterRuntimeError):
            parse_calls(record, id_prefix="r:call:0", allowed=ALLOWED)

    def test_full_seeded_differential_60006_batches(self):
        count = 0
        for seed in (7, 11):
            for batch, prior, scripted in _fuzz_generator(seed):
                count += 1
                self.assertEqual(
                    _diff_outcome(_legacy_parse, batch, prior, scripted),
                    _diff_outcome(_new_parse, batch, prior, scripted),
                    f"seed={seed}, batch={count}, case={batch!r}",
                )
        self.assertEqual(count, 60_006)

    def test_fifteen_targeted_adversarial_cases(self):
        self.assertEqual(len(_targeted_cases()), 15)
        for name, batch, prior in _targeted_cases():
            with self.subTest(case=name):
                self.assertEqual(
                    _diff_outcome(_legacy_parse, batch, prior, False),
                    _diff_outcome(_new_parse, batch, prior, False),
                )

    def test_queue_entry_wire_shape(self):
        entry = ToolQueueEntry("r:call:0:0", "read_file", {"path": "a.txt"})
        wire = entry.to_dict()
        self.assertEqual(
            wire,
            {
                "call_id": "r:call:0:0",
                "name": "read_file",
                "arguments": {"path": "a.txt"},
                "rejection": None,
            },
        )
        wire["arguments"]["path"] = "mutated.txt"
        self.assertEqual(entry.arguments, {"path": "a.txt"})


class ToolEffectTests(unittest.TestCase):
    spec = EnvironmentSpec(max_file_bytes=32, max_workspace_bytes=48)
    storage_limit = 64

    def _contract(self, name, arguments, before, after, ok=True, *, spec=None, storage=None):
        spec = self.spec if spec is None else spec
        tool_effect_contract(
            name,
            arguments,
            before,
            after,
            ok,
            max_file_bytes=spec.max_file_bytes,
            max_workspace_bytes=spec.max_workspace_bytes,
            storage_bytes_limit=self.storage_limit if storage is None else storage,
        )

    def _local_case(self, name, arguments, files):
        provider = LocalTextToolProvider()
        before = dict(files)
        result = provider.execute(
            self.spec,
            EnvironmentSnapshot.from_files(files),
            EnvironmentAction.from_arguments(name, arguments),
        )
        after = result.snapshot.files()
        self._contract(name, arguments, before, after, result.observation["ok"])
        return result, after

    def test_local_provider_effects_satisfy_contract(self):
        before = {"draft.txt": "alpha beta\n", "notes/source.txt": "body"}
        _, read_after = self._local_case("read_file", {"path": "draft.txt"}, before)
        _, search_after = self._local_case("search", {"query": "alpha", "path": "."}, before)
        _, list_after = self._local_case("list_dir", {"path": "."}, before)
        self.assertEqual(read_after, before)
        self.assertEqual(search_after, before)
        self.assertEqual(list_after, before)

        _, write_after = self._local_case(
            "write_file", {"path": "new.txt", "content": "written"}, before
        )
        self.assertEqual(write_after["new.txt"], "written")
        _, patch_after = self._local_case(
            "patch_file", {"path": "draft.txt", "old": "beta", "new": "gamma"}, before
        )
        self.assertEqual(patch_after["draft.txt"], "alpha gamma\n")

    def test_generated_local_write_patch_and_read_snapshots(self):
        rng = random.Random(5_120)
        for index in range(8):
            token = f"token-{index}-{rng.randrange(10_000)}"
            before = {"draft.txt": f"before {token} after", "notes/source.txt": "story note"}
            with self.subTest(index=index, operation="read_file"):
                self._local_case("read_file", {"path": "draft.txt"}, before)
            with self.subTest(index=index, operation="search"):
                self._local_case("search", {"query": token, "path": "."}, before)
            with self.subTest(index=index, operation="list_dir"):
                self._local_case("list_dir", {"path": "notes"}, before)
            with self.subTest(index=index, operation="write_file"):
                self._local_case(
                    "write_file",
                    {"path": "draft.txt", "content": f"rewrite-{rng.randrange(10_000)}"},
                    before,
                )
            with self.subTest(index=index, operation="patch_file"):
                self._local_case(
                    "patch_file",
                    {"path": "draft.txt", "old": token, "new": f"replacement-{index}"},
                    before,
                )

    def test_apply_effect_requires_matching_before_values(self):
        before = {"draft.txt": "alpha"}
        delta = {"draft.txt": {"before": "alpha", "after": "beta"}}
        self.assertEqual(apply_effect(before, delta), {"draft.txt": "beta"})
        with self.assertRaises(AdapterContractError):
            apply_effect(before, {"draft.txt": {"before": "wrong", "after": "beta"}})

    def test_contract_rejects_read_delete_and_write_forgery(self):
        before = {"draft.txt": "alpha"}
        with self.assertRaises(AdapterContractError):
            self._contract("read_file", {"path": "draft.txt"}, before, {})
        with self.assertRaises(AdapterContractError):
            self._contract(
                "write_file",
                {"path": "draft.txt", "content": "expected"},
                before,
                {"draft.txt": "different"},
            )

    def test_contract_rejects_ambiguous_or_empty_patch_target(self):
        before = {"draft.txt": "alpha beta beta"}
        with self.assertRaises(AdapterContractError):
            self._contract(
                "patch_file",
                {"path": "draft.txt", "old": "beta", "new": "gamma"},
                before,
                {"draft.txt": "alpha gamma beta"},
            )
        with self.assertRaises(AdapterContractError):
            self._contract(
                "patch_file",
                {"path": "draft.txt", "old": "", "new": "gamma"},
                before,
                before,
            )

    def test_contract_rejects_non_ok_effect_and_oversize_results(self):
        before = {"draft.txt": "alpha"}
        self._contract("write_file", {"path": "draft.txt", "content": "x"}, before, before, False)
        with self.assertRaises(AdapterContractError):
            self._contract(
                "write_file",
                {"path": "draft.txt", "content": "x"},
                before,
                {"draft.txt": "x"},
                False,
            )
        with self.assertRaises(AdapterContractError):
            self._contract(
                "write_file",
                {"path": "large.txt", "content": "x" * 33},
                before,
                {"draft.txt": "alpha", "large.txt": "x" * 33},
            )
        with self.assertRaises(AdapterContractError):
            self._contract(
                "write_file",
                {"path": "new.txt", "content": "1234567890"},
                before,
                {"draft.txt": "alpha", "new.txt": "1234567890"},
                spec=EnvironmentSpec(max_file_bytes=32, max_workspace_bytes=12),
            )
        with self.assertRaises(AdapterContractError):
            self._contract(
                "read_file",
                {"path": "draft.txt"},
                before,
                before,
                spec=EnvironmentSpec(max_file_bytes=32, max_workspace_bytes=48),
                storage=47,
            )


if __name__ == "__main__":
    unittest.main()
