"""Pretokenized inputs preserve online token coverage, grouping and replay."""

from collections import Counter
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing

from slideon.sae_online import ENCODING_KEYS, OnlineTokenSource
from slideon.sae_tokens import write_pretokenized


def recipe():
    settings = dict.fromkeys(ENCODING_KEYS)
    settings.update(name="SlideOn", d_model=4, d_lift=0, bf16=False,
                    query_length=32, document_length=32)
    return SimpleNamespace(batch_size=4, epochs=2, seed=42, d_model=4,
                           init_docs_per_split=1, checkpoint_config=lambda: settings.copy())


class Encoder:
    """Unique token numbers make missing records or repeated tokens observable."""

    def __init__(self):
        self.counted = []
        self.encoded = []

    def tokenization_metadata(self):
        return dict(tokenizer="test", query_length=32, document_length=32)

    @staticmethod
    def ids(record):
        if "input_ids" in record:
            return record["input_ids"].long()
        identifier, count = map(int, record["text"].split("|"))
        return torch.arange(count, dtype=torch.long) + identifier * 20

    def tokenize_records(self, records):
        return [dict(source=r["source"], role=r.get("role", "document"),
                     id=r["id"], input_ids=self.ids(r)) for r in records]

    def count_tokens(self, records):
        self.counted.extend(records)
        return [len(self.ids(record)) for record in records]

    def encode_records(self, records):
        self.encoded.extend(records)
        return [self.ids(record).float().unsqueeze(1).expand(-1, 4).half() for record in records]


def records():
    return [dict(id="1", source="a", text="1|1"),
            dict(id="2", source="a", role="query", text="2|2"),
            dict(id="3", source="b", text="3|3"),
            dict(id="4", source="a", text="4|4"),
            dict(id="5", source="b", role="query", text="5|1")]


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(record) + "\n" for record in values))
    return path


def prepare_inputs(root):
    values = records()
    write_jsonl(root / "all.jsonl", values)
    write_jsonl(root / "first.jsonl", values[:2])
    write_pretokenized(values, root / "all.pt", Encoder(), shard_tokens=4)
    write_pretokenized(values[2:], root / "rest.pt", Encoder(), shard_tokens=4)


def source(root, inputs, name, encoder=None, rank=0, world=1):
    return OnlineTokenSource(recipe(), [root / path for path in inputs], root / name,
                             encoder or Encoder(), 2, "cpu", rank=rank, world=world)


def distributed_worker(rank, root_string):
    root = Path(root_string)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    current = None
    try:
        # Startup must compare stats, not rehash shards while peers wait in NCCL.
        with mock.patch("slideon.sae_tokens._fingerprint", wraps=__import__(
                "slideon.sae_tokens", fromlist=["_fingerprint"])._fingerprint) as fingerprint:
            current = source(root, ["first.jsonl", "rest.pt"], "distributed", rank=rank, world=2)
            assert all(call.args[1] is False for call in fingerprint.call_args_list)
        batches = current.batches()
        first = next(batches)
        state = batches.state
        tail = list(batches)
        replay = list(current.batches(state))
        for expected, actual in zip(tail, replay):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        assert len(tail) == len(replay) == current.steps - 1
        torch.save([first] + tail, root / f"rank-{rank}.pt")
    finally:
        if current is not None:
            current.close()
        dist.destroy_process_group()


class PretokenizedOnlineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        prepare_inputs(self.root)

    def open_source(self, inputs, name, encoder=None):
        current = source(self.root, inputs, name, encoder)
        self.addCleanup(current.close)
        return current

    def test_raw_pt_and_mixed_inputs_have_identical_batches_and_grouping(self):
        raw = self.open_source(["all.jsonl"], "raw")
        encoder = Encoder()
        pt = self.open_source(["all.pt"], "pt", encoder)
        self.assertEqual(encoder.counted, [])  # PT indexing never tokenizes again.
        mixed = self.open_source(["first.jsonl", "rest.pt"], "mixed")
        self.assertEqual(raw.groups, pt.groups)
        self.assertEqual(raw.groups, mixed.groups)
        self.assertEqual((raw.total_rows, raw.total_tokens, raw.steps), (5, 11, 5))
        expected = list(raw.batches())
        for current in (pt, mixed):
            actual = list(current.batches())
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(Counter(record["id"] for record in encoder.encoded),
                         Counter({str(i): 2 for i in range(1, 6)}))
        self.assertTrue(all(record["role"] == "document" for record in
                            [pt.record(i) for i in range(pt.total_rows)]
                            if record["id"] in ("1", "3", "4")))

    def test_mixed_pending_tokens_resume_exactly(self):
        current = self.open_source(["first.jsonl", "rest.pt"], "mixed")
        batches = current.batches()
        next(batches)
        state = batches.state
        self.assertTrue(state["segments"])
        expected = list(batches)
        current.close()
        restored = self.open_source(["first.jsonl", "rest.pt"], "mixed")
        actual = list(restored.batches(state))
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_initialization_selects_only_documents_per_source(self):
        encoder = Encoder()
        current = self.open_source(["first.jsonl", "rest.pt"], "mixed", encoder)
        current.initialization_tokens(random.Random(42))
        self.assertEqual(len(encoder.encoded), 2)
        self.assertEqual({record["source"] for record in encoder.encoded}, {"a", "b"})
        self.assertTrue(all(record["role"] == "document" for record in encoder.encoded))

    def test_pt_tokenization_mismatch_is_rejected(self):
        encoder = Encoder()
        encoder.tokenization_metadata = lambda: dict(tokenizer="other")
        with self.assertRaisesRegex(ValueError, "settings differ"):
            source(self.root, ["all.pt"], "mismatch", encoder)

    def test_changed_shard_invalidates_existing_index_even_with_old_mtime(self):
        current = self.open_source(["all.pt"], "pt")
        current.close()
        manifest = torch.load(self.root / "all.pt", weights_only=True)
        path = self.root / manifest["shards"][-1]["file"]
        before = path.stat()
        # Change bytes without changing file size or timestamp: full preflight
        # must validate every shard's checksum, including unused final shards.
        with path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            value = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([value[0] ^ 1]))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "shard differs"):
            source(self.root, ["all.pt"], "pt")

    def test_distributed_quick_check_rejects_changed_shard_metadata(self):
        current = self.open_source(["all.pt"], "pt")
        current.close()
        manifest = torch.load(self.root / "all.pt", weights_only=True)
        path = self.root / manifest["shards"][-1]["file"]
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000))
        # Exercise rank zero's pre-collective check without requiring a GPU or
        # opening sockets for this metadata validation case.
        with mock.patch("slideon.sae_online.dist.broadcast_object_list"):
            with self.assertRaisesRegex(RuntimeError, "input metadata changed"):
                source(self.root, ["all.pt"], "pt", rank=0, world=2)

    def test_jsonl_version_one_index_reused_without_tokenization(self):
        current = self.open_source(["all.jsonl"], "raw")
        self.assertEqual(current.manifest["version"], 1)
        current.close()
        encoder = Encoder()
        encoder.count_tokens = mock.Mock(side_effect=AssertionError("Unexpected tokenization"))
        restored = self.open_source(["all.jsonl"], "raw", encoder)
        self.assertEqual(restored.total_tokens, 11)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_two_cpu_ranks_preserve_coverage_and_resume(self):
        preflight = self.open_source(["first.jsonl", "rest.pt"], "distributed")
        preflight.close()
        multiprocessing.spawn(distributed_worker, args=(str(self.root),), nprocs=2, join=True)
        outputs = [torch.load(self.root / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
        baseline = self.open_source(["first.jsonl", "rest.pt"], "single")
        # DDP refills four texts instead of two, so compare complete token
        # multisets after accounting for the final incomplete two-token tail.
        self.assertEqual([len(output) for output in outputs], [baseline.steps, baseline.steps])
        observed = Counter(float(value) for rank in outputs for batch in rank for value in batch[:, 0])
        available = Counter(float(value) for record in records() for value in Encoder.ids(record))
        self.assertEqual(sum(observed.values()), 20)
        self.assertTrue(all(count <= available[value] * 2 for value, count in observed.items()))


if __name__ == "__main__":
    unittest.main()
