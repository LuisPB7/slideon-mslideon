"""Online SAE coverage, restart, distributed progress and checkpoint integration.

Small identifiable token vectors make dropped/duplicated records observable.
The final integration test uses an actual tiny ModernBERT and the raw SAE loader.
"""

from collections import Counter
import copy
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from safetensors.torch import load_file
from transformers import dynamic_module_utils

from slideon import loading
from slideon.sae_online import OnlineTokenSource

import test_raw_sae_eval as raw_fixture
import test_sae_texts as cache_fixture


sae = cache_fixture.sae


class IdentifiableEncoder:
    """Count and encode variable-length sequences without downloads or a GPU."""

    def __init__(self, recipe, calls=None):
        self.recipe = recipe
        self.calls = [] if calls is None else calls

    @staticmethod
    def count_tokens(records):
        return [int(record["text"].split("|")[1]) for record in records]

    @staticmethod
    def values(record, d_model=4):
        identifier, count = map(int, record["text"].split("|"))
        offset = identifier * 20 + (1000 if record.get("role", "document") == "query" else 0)
        tokens = torch.arange(count, dtype=torch.float32) + offset
        columns = [tokens, tokens.remainder(7) + 1, tokens.remainder(11) + 1, torch.ones_like(tokens)]
        return torch.stack(columns[:d_model], dim=1).half()

    def encode_records(self, records):
        self.calls.extend(dict(record) for record in records)
        return [self.values(record, self.recipe.d_model) for record in records]


def tiny_recipe(**overrides):
    _, recipe = sae.parse_args(["--prepare", "--cache", "unused-test-cache"])
    defaults = dict(d_model=4, n_latents=8, k=2, aux_k=2, batch_size=4,
                    max_steps=None, epochs=2, bf16=False, init_docs_per_split=1)
    defaults.update(overrides)
    return replace(recipe, **defaults)


def sample_records():
    # Eleven tokens/epoch: carrying across epochs yields five global batches and
    # only two leftovers. Dropping a tail at every epoch would lose another batch.
    return [
        {"id": "1", "source": "a", "text": "1|1"},
        {"id": "2", "source": "a", "role": "query", "text": "2|2"},
        {"id": "3", "source": "b", "role": "document", "text": "3|3"},
        {"id": "4", "source": "a", "role": "document", "text": "4|4"},
        {"id": "5", "source": "b", "role": "query", "text": "5|1"},
    ]


def write_records(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def drain(batches, expected_steps):
    values = [next(batches) for _ in range(expected_steps)]
    try:
        next(batches)
    except StopIteration:
        return values
    raise AssertionError("Online iterator exceeded its finite epoch budget")


def _distributed_worker(rank, root_string):
    """Spawn-safe two-rank probe: real collectives, DDP backward and replay."""
    root = Path(root_string)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=25))
    source = None
    try:
        recipe = tiny_recipe()
        encoder = IdentifiableEncoder(recipe)
        try:
            OnlineTokenSource(recipe, [root / "input.jsonl"], root / "missing-index",
                              encoder, 2, torch.device("cpu"), rank=rank, world=2)
        except RuntimeError as error:
            assert "index-only" in str(error), str(error)
        else:
            raise AssertionError("Distributed workers must require the tokenizer-only index first")
        source = OnlineTokenSource(recipe, [root / "input.jsonl"], root / "distributed-index",
                                   encoder, 2, torch.device("cpu"), rank=rank, world=2)
        batches = source.batches()
        prefix = [next(batches)]
        restart = copy.deepcopy(batches.state)
        suffix = drain(batches, source.steps - 1)
        coverage_calls = list(encoder.calls)
        replay = drain(source.batches(state=restart), source.steps - 1)
        for expected, actual in zip(suffix, replay):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.manual_seed(23)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(4, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)
        for values in prefix + suffix:
            assert values.shape == (2, 4), values.shape
            optimizer.zero_grad()
            model(values).square().mean().backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
            optimizer.step()
        torch.save(dict(batches=prefix + suffix, calls=coverage_calls,
                        weights=model.module.state_dict()), root / f"rank-{rank}.pt")
        source.close()
        source = None
        # Exercise the production SAE trainer's collectives and gathered restart
        # state as well as the iterator's independent replay contract above.
        training_recipe = replace(recipe, max_steps=3)
        with mock.patch.object(sae, "FrozenTokens", side_effect=lambda rec, dev: IdentifiableEncoder(rec)), \
             mock.patch.object(torch.cuda, "is_available", return_value=False), \
             mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank)}):
            options = dict(online=True, texts_jsonl=[root / "input.jsonl"],
                           index_dir=root / "distributed-index", encode_batch_size=2,
                           save_steps=100, log_steps=3)
            sae.train(training_recipe, None, root / "ddp-full", **options)
            sae.train(training_recipe, None, root / "ddp-resumed", stop_after=1, **options)
            sae.train(training_recipe, None, root / "ddp-resumed",
                      resume=root / "ddp-resumed/checkpoint-1/training_state.pt", **options)
    finally:
        if source is not None:
            source.close()
        dist.destroy_process_group()


class OnlineSAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="slideon-online-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = sample_records()
        self.path = write_records(self.root / "input.jsonl", self.records)
        self.recipe = tiny_recipe()

    def make_source(self, *, recipe=None, refill=2, index="index"):
        recipe = recipe or self.recipe
        encoder = IdentifiableEncoder(recipe)
        source = OnlineTokenSource(recipe, [self.path], self.root / index, encoder,
                                   refill, torch.device("cpu"))
        self.addCleanup(source.close)
        return source, encoder

    def assert_token_coverage(self, batches, recipe):
        expected = Counter(float(token[0]) for record in self.records
                           for token in IdentifiableEncoder.values(record))
        expected = Counter({key: count * recipe.epochs for key, count in expected.items()})
        actual = Counter(float(token[0]) for batch in batches for token in batch)
        self.assertFalse(actual - expected, "Unexpected or repeated token vectors")
        self.assertEqual(sum((expected - actual).values()), sum(expected.values()) % recipe.batch_size)

    def test_epochs_cover_every_text_once_and_carry_partial_batches(self):
        source, encoder = self.make_source()
        self.assertEqual(source.total_rows, 5)
        self.assertEqual(source.total_tokens, 11)
        self.assertEqual(source.steps, 5)
        values = drain(source.batches(), source.steps)
        self.assertTrue(all(batch.shape == (4, 4) for batch in values))
        self.assertTrue(all(batch.dtype == torch.float32 for batch in values))
        self.assertEqual(Counter(record["id"] for record in encoder.calls),
                         Counter({str(identifier): 2 for identifier in range(1, 6)}))
        for record in encoder.calls:
            if record["id"] == "1":
                self.assertEqual(record["role"], "document")
        self.assert_token_coverage(values, self.recipe)
        # Only indexes/provenance may persist; no token-embedding cache is produced.
        self.assertFalse(list(self.root.rglob("*.bin")))

    def test_replay_after_midbuffer_stop_matches_uninterrupted_batches(self):
        source, _ = self.make_source()
        batches = source.batches()
        next(batches)
        saved = copy.deepcopy(batches.state)
        expected = drain(batches, source.steps - 1)
        fresh_source, _ = self.make_source()
        actual = drain(fresh_source.batches(state=saved), source.steps - 1)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_new_run_with_same_seed_replays_text_and_token_order(self):
        first, first_encoder = self.make_source(index="first-index")
        second, second_encoder = self.make_source(index="second-index")
        left, right = drain(first.batches(), first.steps), drain(second.batches(), second.steps)
        self.assertEqual(first_encoder.calls, second_encoder.calls)
        for expected, actual in zip(left, right):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_changed_jsonl_is_rejected_when_reusing_an_index(self):
        self.make_source()
        self.path.write_text(self.path.read_text().replace("1|1", "6|1"))
        with self.assertRaises(ValueError):
            self.make_source()

    def test_initialization_uses_documents_from_each_source_only(self):
        source, encoder = self.make_source()
        values = source.initialization_tokens(random.Random(self.recipe.seed))
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(Counter(record["source"] for record in encoder.calls), Counter({"a": 1, "b": 1}))
        self.assertTrue(all(record["role"] == "document" for record in encoder.calls))
        self.assertTrue((values[:, 0] < 1000).all())

    def train(self, name, *, resume=None, stop_after=None, recipe=None):
        recipe = recipe or replace(self.recipe, max_steps=3)
        output = self.root / name
        with mock.patch.object(sae, "FrozenTokens", side_effect=lambda rec, dev: IdentifiableEncoder(rec)), \
             mock.patch.object(torch.cuda, "is_available", return_value=False), \
             mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}):
            sae.train(recipe, None, output, online=True, texts_jsonl=[self.path],
                      index_dir=self.root / "training-index", encode_batch_size=2,
                      resume=resume, stop_after=stop_after, save_steps=100, log_steps=1)
        return output

    def test_real_sae_training_resume_matches_full_run_exactly(self):
        full = self.train("full")
        resumed = self.train("resumed", stop_after=1)
        state_path = resumed / "checkpoint-1/training_state.pt"
        self.assertTrue(state_path.is_file())
        self.assertFalse((resumed / "final").exists())
        self.train("resumed", resume=state_path)
        expected = load_file(str(full / "final/sae.safetensors"))
        actual = load_file(str(resumed / "final/sae.safetensors"))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        full_metrics = [json.loads(line) for line in (full / "metrics.jsonl").read_text().splitlines()]
        resumed_metrics = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(full_metrics, resumed_metrics)

    def test_training_resume_rejects_modified_input(self):
        output = self.train("resumed", stop_after=1)
        self.path.write_text(self.path.read_text().replace("1|1", "6|1"))
        with self.assertRaises(ValueError):
            self.train("resumed", resume=output / "checkpoint-1/training_state.pt")

    def test_index_only_cli_needs_no_model_loading_or_denseon_forward(self):
        base, _ = raw_fixture.RawSAEEvaluationTests.make_backbone(SimpleNamespace(root=self.root))
        index = self.root / "tokenizer-only-index"
        with mock.patch.object(sae, "snapshot_download", return_value=str(base)), \
             mock.patch.object(sae.AutoModel, "from_pretrained", side_effect=AssertionError("Model loaded")), \
             mock.patch.object(sae.FrozenTokens, "encode_records", side_effect=AssertionError("DenseOn forward")), \
             mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            sae.main(["--online", "--index-only", "--texts-jsonl", str(self.path),
                      "--index-dir", str(index), "--query-length", "8", "--document-length", "12"])
        manifest = json.loads((index / "manifest.json").read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(sum(group["rows"] for group in manifest["groups"]), 5)
        self.assertFalse(list(index.glob("*.bin")))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Requires CPU Gloo support")
    def test_two_ranks_cover_uneven_sequences_replay_and_backpropagate(self):
        # Large-corpus indexing runs before workers enter process-group
        # collectives, so it cannot exhaust their communication timeout.
        indexed, _ = self.make_source(index="distributed-index")
        indexed.close()
        context = multiprocessing.start_processes(_distributed_worker, args=(str(self.root),),
                                                  nprocs=2, join=False, start_method="spawn")
        deadline = time.monotonic() + 60
        try:
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    self.fail("Distributed iterator did not finish within 60 seconds")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
        ranks = [torch.load(self.root / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
        self.assertEqual([len(result["batches"]) for result in ranks], [5, 5])
        self.assert_token_coverage([batch for result in ranks for batch in result["batches"]], self.recipe)
        self.assertEqual(Counter(record["id"] for result in ranks for record in result["calls"]),
                         Counter({str(identifier): 2 for identifier in range(1, 6)}))
        for key in ranks[0]["weights"]:
            torch.testing.assert_close(ranks[0]["weights"][key], ranks[1]["weights"][key], rtol=0, atol=0)
        full = load_file(str(self.root / "ddp-full/final/sae.safetensors"))
        resumed = load_file(str(self.root / "ddp-resumed/final/sae.safetensors"))
        for key in full:
            torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)

    def test_actual_frozen_encoder_mixed_roles_and_online_checkpoint_load(self):
        base, _ = raw_fixture.RawSAEEvaluationTests.make_backbone(SimpleNamespace(root=self.root))
        recipe = replace(self.recipe, d_model=16, max_steps=2, base_model="test/TinyDenseOn",
                         base_model_revision="a" * 40, query_length=8, document_length=12)
        records = [
            {"source": "web", "id": str(index), "role": "query" if index % 2 else "document",
             "text": "word other " * (index + 1)}
            for index in range(6)
        ]
        write_records(self.path, records)
        output = self.root / "actual"
        original_encoder = sae.FrozenTokens
        encoders = []

        def create_encoder(rec, device):
            encoder = original_encoder(rec, device)
            encoders.append(encoder)
            return encoder

        with mock.patch.object(sae, "snapshot_download", return_value=str(base)), \
             mock.patch.object(sae, "FrozenTokens", side_effect=create_encoder), \
             mock.patch.object(torch.cuda, "is_available", return_value=False), \
             mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}):
            encoder = original_encoder(recipe, torch.device("cpu"))
            forwards = []
            hook = encoder.encoder.register_forward_pre_hook(
                lambda module, args, kwargs: forwards.append(kwargs["input_ids"].shape[0]), with_kwargs=True)
            mixed = [records[index % len(records)] for index in range(512)]
            counts = encoder.count_tokens(mixed)
            self.assertEqual(forwards, [])
            arrays = encoder.encode_records(mixed)
            self.assertEqual(forwards, [512])
            self.assertEqual([len(array) for array in arrays], counts)
            self.assertTrue(all(array.dtype == torch.float16 for array in arrays))
            self.assertTrue(all(not array.requires_grad and not array.is_inference() for array in arrays))
            self.assertTrue(all(len(array) <= (8 if row["role"] == "query" else 12)
                                for array, row in zip(arrays, mixed)))
            # A downstream trainable operation must accept these frozen activations.
            head = torch.nn.Linear(16, 2)
            head(torch.cat(arrays).float()).square().mean().backward()
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            hook.remove()
            # Mixed-role batching must preserve the existing prompt/length rules.
            for index in (0, 1):
                reference = encoder.encode([records[index]["text"]], records[index]["role"])[0]
                torch.testing.assert_close(arrays[index].cpu(), torch.from_numpy(reference),
                                           rtol=2e-3, atol=1e-3)
            sae.train(recipe, None, output, online=True, texts_jsonl=[self.path],
                      index_dir=self.root / "actual-index", encode_batch_size=3,
                      save_steps=100, log_steps=2)
        self.assertTrue(encoders)
        for encoder in encoders:
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                for parameter in encoder.encoder.parameters()))
        final = output / "final"
        trained = load_file(str(final / "sae.safetensors"))
        with mock.patch.object(loading, "snapshot_download", return_value=str(base)), \
             mock.patch.object(dynamic_module_utils, "HF_MODULES_CACHE", str(self.root / "custom-code")):
            model = loading.from_sae(str(final), query_length=8, document_length=12, device="cpu")
            torch.testing.assert_close(model[0].model.head.W_enc, trained["W_enc"], rtol=0, atol=0)
            model.eval()
            for role in ("query", "document"):
                values = getattr(model, "encode_" + role)(["word other"], convert_to_tensor=True,
                                                          show_progress_bar=False)
                if values.layout != torch.strided:
                    values = values.to_dense()
                self.assertEqual(values.shape, (1, recipe.n_latents))
                self.assertTrue(torch.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
