"""Offline JSONL-to-cache integration checks with a deterministic fake encoder.

The cache reader, token iterator and short CPU SAE training run are real. Only
DenseOn/tokenizer loading is replaced, so no model downloads or GPU are needed.
"""

from collections import Counter
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from safetensors.torch import load_file

from slideon.sae_texts import iter_jsonl, scan_jsonl


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slideon_sae_texts_test_target", ROOT / "scripts/sae-pretrain/train_slideon_sae.py"
)
sae = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sae
SPEC.loader.exec_module(sae)


class FakeTokenizer:
    def __call__(self, texts, *, truncation, max_length):
        assert truncation
        # Two special tokens and one token per whitespace-delimited word.
        return {"input_ids": [list(range(min(len(text.split()) + 2, max_length)))
                              for text in texts]}


class FakeFrozenTokens:
    def __init__(self, recipe, calls):
        self.recipe, self.calls = recipe, calls
        self.tokenizer = FakeTokenizer()

    def encode(self, texts, task):
        self.calls.append((task, list(texts)))
        return [self.expected(text, task, self.recipe) for text in texts]

    @staticmethod
    def expected(text, task, recipe):
        prompt = getattr(recipe, task + "_prompt")
        count = min(len((prompt + text).split()) + 2, getattr(recipe, task + "_length"))
        seed = sum(text.encode("utf-8")) + (10000 if task == "query" else 0)
        values = np.random.default_rng(seed).standard_normal((count, recipe.d_model))
        return (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float16)


class SAETextPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="slideon-sae-text-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache_path = self.root / "token cache"
        _, default_recipe = sae.parse_args(["--prepare", "--cache", str(self.cache_path)])
        self.recipe = replace(default_recipe, d_model=4, n_latents=8, k=2, aux_k=2,
                              batch_size=4, max_steps=2, bf16=False,
                              query_length=3, document_length=5, init_docs_per_split=1)
        self.calls = []
        self.encoder = mock.Mock(side_effect=lambda recipe, device: FakeFrozenTokens(recipe, self.calls))
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(sae, "FrozenTokens", self.encoder).start()
        mock.patch.object(torch.cuda, "is_available", return_value=False).start()
        # Every test fails immediately if a raw-text route attempts an HF dataset lookup.
        mock.patch.object(sae, "HfApi", side_effect=AssertionError("Unexpected Hub lookup")).start()
        mock.patch.object(sae, "load_dataset", side_effect=AssertionError("Unexpected dataset lookup")).start()

    def write_rows(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def fixture_paths(self):
        self.rows = [
            {"source": "marco", "id": "d1", "text": "one two"},
            {"source": "marco", "id": "q1", "role": "query", "text": "where"},
            {"source": "fineweb", "id": "f1", "role": "document", "text": "a b c d e f"},
            {"source": "marco", "id": "d2", "text": "three four five"},
            {"source": "fineweb", "text": "x", "extra_metadata": {"ignored": True}},
        ]
        return [self.write_rows("first.jsonl", self.rows[:3]),
                self.write_rows("second.jsonl", self.rows[3:])]

    def prepare(self, paths=None):
        paths = self.fixture_paths() if paths is None else paths
        sae.prepare_text_tokens(self.recipe, self.cache_path, paths, encode_batch_size=2)
        return sae.TokenCache(self.cache_path, self.recipe)

    def test_shards_feed_existing_cache_reader_with_roles_boundaries_and_counts(self):
        cache = self.prepare()
        self.assertTrue(cache.manifest["complete"])
        self.assertEqual(cache.manifest["dataset_revisions"], {})
        groups = {(source["split"], source["kind"]): (index, source)
                  for index, source in enumerate(cache.sources)}
        self.assertEqual(set(groups), {("marco", "documents"), ("marco", "queries"),
                                       ("fineweb", "documents")})
        self.assertEqual(len(cache.sources), 3)  # Input shards do not become new sources.
        for (source, kind), (part, metadata) in groups.items():
            role = "query" if kind == "queries" else "document"
            expected_rows = [row for row in self.rows
                             if row["source"] == source and row.get("role", "document") == role]
            self.assertEqual(metadata["rows"], len(expected_rows))
            self.assertEqual(cache.chars[part].tolist(), [len(row["text"]) for row in expected_rows])
            self.assertEqual(cache.arrays[part].dtype, np.dtype("float16"))
            for index, row in enumerate(expected_rows):
                np.testing.assert_array_equal(cache.row(part, index),
                                              FakeFrozenTokens.expected(row["text"], role, self.recipe))
        encoded = Counter((task, text) for task, texts in self.calls for text in texts)
        self.assertEqual(encoded, Counter((row.get("role", "document"), row["text"]) for row in self.rows))
        self.assertTrue(all(0 < len(texts) <= 2 for _, texts in self.calls))
        self.assertEqual(cache.scheduled_tokens, 20)  # Unprompted, role-truncated counts.
        self.assertEqual(cache.total_tokens, 22)      # Prompted vectors, as in the HF route.

    def test_reader_preserves_text_and_duplicate_examples_without_deduplication(self):
        rows = [{"source": "web", "id": "same", "text": " café\nline two "}] * 2
        path = self.write_rows("duplicates.jsonl", rows)
        parsed = list(iter_jsonl([path]))
        self.assertEqual(len(parsed), 2)
        self.assertEqual([row["text"] for row in parsed], [row["text"] for row in rows])
        self.assertTrue(all(row["role"] == "document" for row in parsed))
        cache = self.prepare([path])
        self.assertEqual(cache.sources[0]["rows"], 2)

    def test_complete_cache_reuse_does_not_reload_denseon(self):
        paths = self.fixture_paths()
        self.prepare(paths)
        self.encoder.reset_mock()
        self.calls.clear()
        self.prepare(paths)
        self.encoder.assert_not_called()
        self.assertEqual(self.calls, [])

    def test_interrupted_encoding_remains_unusable_then_rebuilds_cleanly(self):
        paths = self.fixture_paths()
        failed_encoder = FakeFrozenTokens(self.recipe, self.calls)
        failed_encoder.encode = mock.Mock(side_effect=RuntimeError("simulated interruption"))
        self.encoder.side_effect = lambda recipe, device: failed_encoder
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.prepare(paths)
        self.assertFalse(json.loads((self.cache_path / "manifest.json").read_text())["complete"])
        with self.assertRaises(ValueError):
            sae.TokenCache(self.cache_path, self.recipe)
        self.encoder.side_effect = lambda recipe, device: FakeFrozenTokens(recipe, self.calls)
        cache = self.prepare(paths)
        self.assertEqual(sum(source["rows"] for source in cache.sources), 5)
        self.assertEqual(cache.total_tokens, 22)
        self.assertEqual(list(self.cache_path.glob("*.partial")), [])

    def test_changed_input_contents_and_file_order_reject_cache_reuse(self):
        paths = self.fixture_paths()
        self.prepare(paths)
        with self.assertRaises(ValueError):
            self.prepare(list(reversed(paths)))
        original = paths[0].read_text(encoding="utf-8")
        # Same length and row count: content fingerprints, not file sizes, must catch this.
        paths[0].write_text(original.replace("one two", "one six"), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.prepare(paths)

    def test_input_digest_verification_detects_change_between_scan_and_encoding(self):
        path = self.write_rows("changing.jsonl", [{"source": "web", "text": "first"}])
        metadata = scan_jsonl([path])
        self.write_rows("changing.jsonl", [{"source": "web", "text": "other"}])
        with self.assertRaises(ValueError):
            list(iter_jsonl([path], expected_files=metadata["files"]))

    def test_invalid_records_report_file_and_line(self):
        bad_records = [
            "not json",
            json.dumps([]),
            json.dumps({"source": "web"}),
            json.dumps({"source": "web", "text": "   "}),
            json.dumps({"source": "web", "text": 42}),
            json.dumps({"text": "hello"}),
            json.dumps({"source": "", "text": "hello"}),
            json.dumps({"source": "web", "text": "hello", "role": "passage"}),
            json.dumps({"source": "web", "text": "hello", "id": 12}),
        ]
        for record in bad_records:
            with self.subTest(record=record):
                path = self.root / "invalid.jsonl"
                path.write_text('{"source":"web","text":"valid"}\n' + record + "\n", encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    scan_jsonl([path])
                self.assertIn("invalid.jsonl", str(caught.exception))
                self.assertRegex(str(caught.exception), r"(?:line\s*|:)2\b")
        self.encoder.assert_not_called()

    def test_query_only_input_fails_before_loading_encoder(self):
        path = self.write_rows("queries.jsonl", [{"source": "queries", "role": "query", "text": "hello"}])
        with self.assertRaisesRegex(ValueError, "document"):
            self.prepare([path])
        self.encoder.assert_not_called()

    def test_empty_input_fails_before_loading_encoder(self):
        path = self.write_rows("empty.jsonl", [])
        with self.assertRaises(ValueError):
            self.prepare([path])
        self.encoder.assert_not_called()

    def test_cache_identity_uses_encoding_settings_not_hf_selection_settings(self):
        self.prepare()
        # JSONL already specifies its texts; changing pair mining does not change them.
        sae.TokenCache(self.cache_path, replace(self.recipe, sampled_negatives=0))
        with self.assertRaises(ValueError):
            sae.TokenCache(self.cache_path, replace(self.recipe, document_length=6))

    def test_arbitrary_source_labels_cannot_escape_cache_directory(self):
        path = self.write_rows("labels.jsonl", [
            {"source": "../outside", "text": "hello"},
            {"source": "corpus/with spaces", "text": "world"},
        ])
        cache = self.prepare([path])
        for source in cache.sources:
            self.assertEqual(Path(source["name"]).name, source["name"])
            self.assertTrue((self.cache_path / (source["name"] + ".bin")).is_file())
        self.assertFalse((self.root / "outside.bin").exists())

    def test_cli_preserves_hf_default_and_restricts_jsonl_to_preparation(self):
        args, _ = sae.parse_args(["--prepare", "--cache", str(self.cache_path)])
        self.assertIsNone(args.texts_jsonl)
        args, _ = sae.parse_args(["--prepare", "--cache", str(self.cache_path),
                                  "--texts-jsonl", "one.jsonl", "two.jsonl"])
        self.assertEqual(args.texts_jsonl, ["one.jsonl", "two.jsonl"])
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            sae.parse_args(["--cache", str(self.cache_path), "--output", str(self.root / "output"),
                            "--texts-jsonl", "one.jsonl"])

    def test_cache_feeds_real_cpu_training_and_preserves_raw_checkpoint_contract(self):
        self.prepare()
        output = self.root / "trained"
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}):
            sae.train(self.recipe, self.cache_path, output, buffer_tokens=16, save_steps=100, log_steps=2)
        final = output / "final"
        config = json.loads((final / "config.json").read_text())
        self.assertEqual(config["base_model"], self.recipe.base_model)
        self.assertEqual(config["base_model_revision"], self.recipe.base_model_revision)
        self.assertTrue(config["normalized_inputs"])
        self.assertTrue(config["backbone_only"])
        self.assertEqual(config["n_latents"], 8)
        weights = load_file(str(final / "sae.safetensors"))
        self.assertEqual(weights["W_enc"].shape, (4, 8))
        self.assertTrue(all(torch.isfinite(value).all() for value in weights.values()))
        self.assertTrue((final / "training_state.pt").is_file())
        self.assertEqual(json.loads((output / "cache_manifest.json").read_text()),
                         json.loads((self.cache_path / "manifest.json").read_text()))


if __name__ == "__main__":
    unittest.main()
