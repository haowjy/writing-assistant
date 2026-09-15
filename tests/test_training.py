import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from writing_agent.data import read_records
from writing_agent.training import SFTSettings, encode_trajectory, prepare_sft, train_sft

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("transformers"), "optional training dependencies")
class TrainingPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.settings = SFTSettings()
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                cls.settings.model_id,
                revision=cls.settings.revision,
                local_files_only=True,
            )
        except OSError as exc:
            raise unittest.SkipTest("Pinned Gemma tokenizer is not cached") from exc
        cls.records = read_records(ROOT / "data/fixtures/sft-mask-probes.jsonl")

    def test_masks_exclude_context_and_observations_preserve_calls_and_endings(self):
        rows = [encode_trajectory(r, self.tokenizer, max_length=2048) for r in self.records]
        for record, row in zip(self.records, rows, strict=True):
            target = row["supervised_text"]
            self.assertNotIn("<|tool_response>", target)
            self.assertNotIn("<tool_response|>", target)
            self.assertNotIn("declaration:", target)
            self.assertNotIn("<|think|>", target)
            self.assertNotIn(record["messages"][0]["content"], target)
            self.assertTrue(target.endswith("<turn|>\n"))
            calls = sum(len(m.get("tool_calls", [])) for m in record["messages"])
            self.assertEqual(target.count("<|tool_call>"), calls)
            self.assertEqual(target.count("<tool_call|>"), calls)
        self.assertIn("call:patch_file", rows[1]["supervised_text"])
        self.assertNotIn("Nobody has measured its position", rows[4]["supervised_text"])

    def test_plain_text_tool_observation_is_context_only(self):
        record = copy.deepcopy(self.records[-1])
        record["messages"][2]["content"] = "OBSERVATION_SENTINEL"
        row = encode_trajectory(record, self.tokenizer, max_length=2048)
        self.assertIn("OBSERVATION_SENTINEL", row["rendered"])
        self.assertNotIn("OBSERVATION_SENTINEL", row["supervised_text"])

    def test_rejects_reasoning_overflow_and_evaluation_groups(self):
        record = copy.deepcopy(self.records[0])
        record["messages"][-1]["reasoning"] = "private trace"
        with self.assertRaisesRegex(ValueError, "Reasoning"):
            encode_trajectory(record, self.tokenizer, max_length=2048)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            encode_trajectory(self.records[0], self.tokenizer, max_length=2)
        record = copy.deepcopy(self.records[0])
        record["review_status"] = "accepted"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "evaluation source"):
                prepare_sft(
                    [record],
                    self.tokenizer,
                    Path(tmp) / "data",
                    settings=self.settings,
                    excluded_source_groups={"sft-mask-fixtures"},
                )
            with self.assertRaisesRegex(ValueError, "No accepted"):
                prepare_sft(
                    self.records,
                    self.tokenizer,
                    Path(tmp) / "data",
                    settings=self.settings,
                    excluded_source_groups=set(),
                )

    def test_preparation_inspection_and_tampering(self):
        records = copy.deepcopy(self.records)
        for record in records:
            record["review_status"] = (
                "accepted"  # Test-only acceptance, never writes production data.
            )
        with tempfile.TemporaryDirectory() as tmp:
            prepared, output = Path(tmp) / "data", Path(tmp) / "run"
            manifest = prepare_sft(
                records,
                self.tokenizer,
                prepared,
                settings=self.settings,
                excluded_source_groups=set(),
            )
            plan = train_sft(prepared, output, settings=self.settings)
            self.assertEqual(plan["preparation_hash"], manifest["preparation_hash"])
            self.assertFalse(output.exists())
            rows = json.loads((prepared / "encoded.json").read_text())
            rows[0]["labels"][0] = 42
            (prepared / "encoded.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "changed"):
                train_sft(prepared, output, settings=self.settings)

    @unittest.skipUnless(importlib.util.find_spec("trl"), "optional TRL dependency")
    def test_trl_batch_preserves_labels_without_optimization(self):
        from datasets import Dataset
        from transformers import DataCollatorForSeq2Seq, GPT2Config, GPT2LMHeadModel
        from trl import SFTConfig, SFTTrainer

        rows = [encode_trajectory(r, self.tokenizer, max_length=2048) for r in self.records[:2]]
        data = Dataset.from_list(
            [{k: r[k] for k in ("input_ids", "attention_mask", "labels")} for r in rows]
        )
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(self.tokenizer), n_embd=8, n_layer=1, n_head=1, n_positions=512
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SFTTrainer(
                model=model,
                processing_class=self.tokenizer,
                train_dataset=data,
                args=SFTConfig(
                    output_dir=tmp,
                    use_cpu=True,
                    bf16=False,
                    report_to="none",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    packing=False,
                    padding_free=False,
                    max_length=2048,
                    per_device_train_batch_size=2,
                ),
                data_collator=DataCollatorForSeq2Seq(self.tokenizer, label_pad_token_id=-100),
            )
            batch = next(iter(trainer.get_train_dataloader()))
            actual = {
                tuple(ids[mask.bool()].tolist()): labels[mask.bool()].tolist()
                for ids, labels, mask in zip(
                    batch["input_ids"], batch["labels"], batch["attention_mask"], strict=True
                )
            }
            for row in rows:
                self.assertEqual(actual[tuple(row["input_ids"])], row["labels"])
            self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
            self.assertEqual(trainer.state.global_step, 0)
