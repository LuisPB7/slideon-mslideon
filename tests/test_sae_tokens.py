"""CPU checks for bounded, lossless pretokenized storage (no Hub or GPU)."""

import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sae_tokens_test_target", ROOT / "src/slideon/sae_tokens.py")
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


class Encoder:
    """Tokens are supplied by the fixture; production calls the real tokenizer."""

    def __init__(self):
        self.batches = []

    def tokenization_metadata(self):
        return {"tokenizer": "fixture", "revision": "fixed", "query_length": 256,
                "document_length": 256, "prompts": {"query": "q: ", "document": "d: "}}

    def tokenize_records(self, records):
        self.batches.append(len(records))
        return [{key: value for key, value in row.items() if key != "text"} for row in records]


def rows():
    return [
        {"text": "a", "source": "msmarco", "role": "document", "id": "café-文", "input_ids": [0, 65535, 33]},
        {"text": "q", "source": "generated", "role": "query", "input_ids": [3]},
        {"text": "b", "source": "fineweb", "role": "document", "id": "b", "input_ids": [7, 8]},
        {"text": "a", "source": "msmarco", "role": "document", "id": "café-文", "input_ids": [0, 65535, 33]},
    ]


class PretokenizedStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "tokens.pt"
        self.encoder = Encoder()

    def write(self, records=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return storage.write_pretokenized(iter(rows() if records is None else records), self.path,
                                             self.encoder, tokenize_batch_size=2, **kwargs)

    def load(self, **kwargs):
        dataset = storage.PretokenizedDataset(self.path, **kwargs)
        self.addCleanup(dataset.close)
        return dataset

    def rewrite_manifest(self, edit):
        manifest = torch.load(self.path, weights_only=True)
        edit(manifest)
        torch.save(manifest, self.path)

    def rewrite_shard(self, edit):
        manifest = torch.load(self.path, weights_only=True)
        path = self.path.parent / manifest["shards"][0]["file"]
        shard = torch.load(path, weights_only=True)
        edit(shard)
        torch.save(shard, path)
        manifest["shards"][0]["bytes"] = path.stat().st_size
        manifest["shards"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        torch.save(manifest, self.path)

    def test_roundtrip_preserves_order_duplicates_metadata_and_full_uint16_range(self):
        manifest = self.write(shard_tokens=4)
        dataset = self.load(expected_tokenization=self.encoder.tokenization_metadata())
        self.assertEqual((len(dataset), dataset.tokens), (4, 9))
        self.assertEqual(self.encoder.batches, [2, 2])
        self.assertEqual(manifest["max_token_id"], 65535)
        self.assertEqual(len(manifest["shards"]), 3)
        first_shard = torch.load(dataset.shards[0], weights_only=True, mmap=True)
        self.assertEqual(first_shard["input_id_bytes"].dtype, torch.uint8)
        self.assertEqual(first_shard["input_id_bytes"].tolist(), [0, 0, 255, 255, 33, 0, 3, 0])
        self.assertEqual(first_shard["input_id_bytes"].numel(), 2 * manifest["shards"][0]["tokens"])
        for index, expected in enumerate(rows()):
            actual = dataset.record(index)
            self.assertEqual(actual.pop("input_ids").tolist(), expected["input_ids"])
            self.assertEqual(actual, {k: v for k, v in expected.items() if k not in ("text", "input_ids")})
        self.assertEqual(list(dataset.iter_index()), [(0, "msmarco", "document", 3),
                         (1, "generated", "query", 1), (2, "fineweb", "document", 2),
                         (3, "msmarco", "document", 3)])
        full, cheap = dataset.fingerprint(), dataset.fingerprint(full=False)
        full.pop("sha256")
        for shard in full["shards"]:
            shard.pop("sha256")
        self.assertEqual(full, cheap)

    def test_descriptor_has_no_corpus_tensors_and_shards_are_mmap_weights_only(self):
        self.write(shard_tokens=3)
        with mock.patch.object(torch, "load", wraps=torch.load) as loader:
            dataset = self.load(max_open_shards=1)
            self.assertFalse(dataset._cache)
            for index in [0, 2, 3, 1, 0]:
                dataset.record(index)
                self.assertLessEqual(len(dataset._cache), 1)
            for call in loader.call_args_list[1:]:
                self.assertTrue(call.kwargs["weights_only"])
                self.assertTrue(call.kwargs["mmap"])
                self.assertEqual(call.kwargs["map_location"], "cpu")
        def walk(value):
            self.assertNotIsInstance(value, torch.Tensor)
            if isinstance(value, dict):
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(dataset.manifest)
        # Mutating returned batches cannot write into the mmap or the corpus.
        record = dataset.record(0)
        record["input_ids"].fill_(11)
        self.assertEqual(dataset.record(0)["input_ids"].tolist(), [0, 65535, 33])

    def test_limit_does_not_read_extra_rows_and_stream_counters_are_saved(self):
        provenance = {"read": 0}
        def stream():
            for row in rows():
                provenance["read"] += 1
                yield row
        self.write(stream(), limit=3, provenance=provenance)
        dataset = self.load()
        self.assertEqual(len(dataset), 3)
        self.assertEqual(self.encoder.batches, [2, 1])
        self.assertEqual(provenance, {"read": 3})
        self.assertEqual(dataset.manifest["provenance"], provenance)

    def test_many_short_sequences_cross_shards_without_padding_or_order_changes(self):
        records = ({"source": "queries", "role": "query", "input_ids": [index]}
                   for index in range(10000))
        with contextlib.redirect_stdout(io.StringIO()):
            storage.write_pretokenized(records, self.path, self.encoder,
                                      tokenize_batch_size=512, shard_tokens=2048)
        dataset = self.load()
        self.assertEqual((len(dataset), dataset.tokens), (10000, 10000))
        self.assertEqual([s["tokens"] for s in dataset.manifest["shards"]], [2048] * 4 + [1808])
        self.assertLessEqual(max(self.encoder.batches), 512)
        for index in [0, 2047, 2048, 4095, 4096, 8192, 9999]:
            self.assertEqual(dataset.record(index)["input_ids"].tolist(), [index])
        self.assertEqual(sum(row[3] for row in dataset.iter_index()), 10000)

    def test_token_ids_cannot_exceed_recorded_tokenizer_vocabulary(self):
        metadata = dict(self.encoder.tokenization_metadata(), max_token_id=100)
        with mock.patch.object(self.encoder, "tokenization_metadata", return_value=metadata):
            with self.assertRaisesRegex(ValueError, "tokenizer's recorded max_token_id"):
                self.write([{"source": "source", "input_ids": [101]}])
            self.assertEqual(list(self.path.parent.iterdir()), [])
            self.write([{"source": "source", "input_ids": [0, 100]}])
        self.assertEqual(self.load().record(0)["input_ids"].tolist(), [0, 100])
        self.rewrite_manifest(lambda m: m["tokenization"].update(max_token_id=99))
        with self.assertRaisesRegex(ValueError, "tokenizer's recorded max_token_id"):
            self.load()

    def test_default_role_missing_ids_and_single_long_document(self):
        self.write([{"source": "source", "input_ids": list(range(12))}], shard_tokens=4)
        dataset = self.load()
        self.assertEqual(dataset.record(0)["role"], "document")
        self.assertNotIn("id", dataset.record(0))
        self.assertEqual(dataset.manifest["shards"][0]["tokens"], 12)

    def test_changed_settings_and_incomplete_descriptors_are_rejected(self):
        self.write()
        metadata = self.encoder.tokenization_metadata()
        metadata["document_length"] = 128
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.load(expected_tokenization=metadata)
        self.rewrite_manifest(lambda m: m.update(complete=False))
        with self.assertRaisesRegex(ValueError, "completed"):
            self.load()

    def test_full_fingerprint_detects_changed_shard_even_at_same_size(self):
        self.write(shard_tokens=3)
        dataset = self.load()
        path = dataset.shards[-1]
        content = bytearray(path.read_bytes())
        content[100] ^= 1
        path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "differs"):
            dataset.fingerprint()

    def test_missing_or_truncated_shards_are_rejected_at_open(self):
        self.write()
        dataset = self.load()
        path = dataset.shards[0]
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "size changed"):
            self.load()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_path_escape_rejected(self):
        self.write()
        self.rewrite_manifest(lambda m: m["shards"][0].update(file="../elsewhere.pt"))
        with self.assertRaisesRegex(ValueError, "shard path"):
            self.load()

    def test_malformed_offsets_and_group_indices_rejected(self):
        for field, value in [("offsets", 0), ("group_ids", 999)]:
            with self.subTest(field=field):
                self.path = Path(self.temp.name) / f"{field}.pt"
                self.write()
                self.rewrite_shard(lambda data: data[field].__setitem__(1, value))
                dataset = self.load()
                with self.assertRaisesRegex(ValueError, "offsets or source groups"):
                    dataset.record(0)

    def test_invalid_tokens_cleanup_and_no_completed_descriptor(self):
        for values in [[-1], [65536], [], [1.5], [[1, 2]], [True]]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.write([{"source": "source", "input_ids": values}])
                self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_stream_failure_cleans_already_written_shards(self):
        def broken():
            yield from rows()
            raise RuntimeError("download interrupted")
        with self.assertRaisesRegex(RuntimeError, "download interrupted"):
            self.write(broken(), shard_tokens=1)
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_refuses_overwrite_and_empty_stream(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.write([])
        self.write()
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
