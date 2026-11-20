"""Real tiny ModernBERT: packed inputs preserve encoding and SAE training."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file
from transformers import dynamic_module_utils

from slideon import loading
from slideon.sae_tokens import PretokenizedDataset, write_pretokenized

import test_raw_sae_eval as raw_fixture
from test_sae_online import tiny_recipe, write_records
from test_sae_texts import sae


class PretokenizedEncoderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="slideon-pt-encoder-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base, _ = raw_fixture.RawSAEEvaluationTests.make_backbone(SimpleNamespace(root=self.root))
        self.recipe = tiny_recipe(d_model=16, max_steps=3, base_model="test/TinyDenseOn",
                                  base_model_revision="a" * 40, query_length=8, document_length=12)
        self.records = [
            dict(source="web" if i % 3 else "marco", id=str(i),
                 role="query" if i % 2 else "document", text="word other " * (i + 1))
            for i in range(6)
        ]
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(sae, "snapshot_download", return_value=str(self.base)).start()
        mock.patch.object(torch.cuda, "is_available", return_value=False).start()
        mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}).start()
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_tokenizer_only_export_and_exact_mixed_batch_embeddings(self):
        with mock.patch.object(sae.AutoModel, "from_pretrained", side_effect=AssertionError("Loaded backbone")):
            tokenizer = sae.FrozenTokens(self.recipe, torch.device("cpu"), load_encoder=False)
            identity = tokenizer.tokenization_metadata()
            path = self.root / "texts.pt"
            write_pretokenized(self.records, path, tokenizer, tokenize_batch_size=2, shard_tokens=15)
            self.assertEqual(identity, tokenizer.tokenization_metadata())

        encoder = sae.FrozenTokens(self.recipe, torch.device("cpu"))
        reader = PretokenizedDataset(path, expected_tokenization=encoder.tokenization_metadata())
        self.addCleanup(reader.close)
        saved = [reader.record(i) for i in range(len(reader))]
        mixed = [saved[i] if i % 2 else raw for i, raw in enumerate(self.records)]
        for side in ("left", "right"):
            with self.subTest(padding=side):
                encoder.tokenizer.padding_side = side
                expected = encoder._record_features(self.records)
                for records in (saved, mixed):
                    actual = encoder._record_features(records)
                    for key in ("input_ids", "attention_mask"):
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                calls = []
                hook = encoder.encoder.register_forward_pre_hook(
                    lambda module, args, kwargs: calls.append(kwargs["input_ids"].shape[0]), with_kwargs=True)
                try:
                    reference = encoder.encode_records(self.records)
                    for records in (saved, mixed):
                        actual = encoder.encode_records(records)
                        for left, right in zip(actual, reference):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                            self.assertFalse(left.requires_grad)
                    self.assertEqual(calls, [6, 6, 6])
                finally:
                    hook.remove()

    def test_role_limits_prompts_revision_and_tokenizer_mismatch_are_rejected(self):
        encoder = sae.FrozenTokens(self.recipe, torch.device("cpu"), load_encoder=False)
        path = self.root / "texts.pt"
        write_pretokenized(self.records, path, encoder)
        changes = [dict(query_length=9), dict(document_length=13), dict(query_prompt="other: "),
                   dict(base_model_revision="b" * 40)]
        for change in changes:
            with self.subTest(change=change):
                different = sae.FrozenTokens(replace(self.recipe, **change), torch.device("cpu"), load_encoder=False)
                with self.assertRaisesRegex(ValueError, "settings differ"):
                    PretokenizedDataset(path, expected_tokenization=different.tokenization_metadata())
        encoder.tokenizer.add_tokens(["brandnewtoken"])
        with self.assertRaisesRegex(ValueError, "settings differ"):
            PretokenizedDataset(path, expected_tokenization=encoder.tokenization_metadata())

    def test_export_rejects_blank_text_and_raw_text_takes_precedence_over_extra_fields(self):
        encoder = sae.FrozenTokens(self.recipe, torch.device("cpu"), load_encoder=False)
        for record in ({"source": "web", "text": " "}, {"source": "web"},
                       {"source": "web", "text": 12}):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, "text must be"):
                write_pretokenized([record], self.root / "invalid.pt", encoder)
        row = dict(self.records[0], input_ids=[65535])  # Irrelevant JSONL metadata.
        actual = encoder.tokenize_records([row])[0]
        expected = encoder.tokenize_records([self.records[0]])[0]
        torch.testing.assert_close(actual["input_ids"], expected["input_ids"], rtol=0, atol=0)

    def test_same_final_sae_for_jsonl_mixed_inputs_and_resumed_training(self):
        tokenizer = sae.FrozenTokens(self.recipe, torch.device("cpu"), load_encoder=False)
        raw = write_records(self.root / "all.jsonl", self.records)
        first = write_records(self.root / "first.jsonl", self.records[:3])
        packed = self.root / "second.pt"
        write_pretokenized(self.records[3:], packed, tokenizer, shard_tokens=15)
        common = dict(online=True, encode_batch_size=3, save_steps=100, log_steps=3)
        sae.train(self.recipe, None, self.root / "raw", texts_jsonl=[raw],
                  index_dir=self.root / "raw-index", **common)
        mixed = dict(texts_jsonl=[first, packed], index_dir=self.root / "mixed-index", **common)
        sae.train(self.recipe, None, self.root / "mixed", **mixed)
        sae.train(self.recipe, None, self.root / "resumed", stop_after=1, **mixed)
        sae.train(self.recipe, None, self.root / "resumed",
                  resume=self.root / "resumed/checkpoint-1/training_state.pt", **mixed)
        reference = load_file(str(self.root / "raw/final/sae.safetensors"))
        for name in ("mixed", "resumed"):
            actual = load_file(str(self.root / name / "final/sae.safetensors"))
            for key in reference:
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        final = self.root / "mixed/final"
        config = json.loads((final / "config.json").read_text())
        self.assertEqual(config["query_length"], 8)
        self.assertEqual(config["document_length"], 12)
        with mock.patch.object(loading, "snapshot_download", return_value=str(self.base)), \
             mock.patch.object(dynamic_module_utils, "HF_MODULES_CACHE", str(self.root / "custom-code")):
            model = loading.from_sae(str(final), device="cpu")
            torch.testing.assert_close(model[0].model.head.W_enc, reference["W_enc"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
