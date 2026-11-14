"""CPU-only checks for opt-in sampling, sharding and checkpoint replay."""

import math
import pickle
import unittest
from collections import Counter
from types import SimpleNamespace

import torch
from accelerate import skip_first_batches
from accelerate.data_loader import prepare_data_loader
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler

from slideon.sampling import (
    DrawIndex,
    NegativeSamplingCollator,
    OccurrenceDataset,
    SamplingEpochCallback,
    SourceSamplingBatchSampler,
    configure_sampling,
)


class KeepRows:
    """Keep raw candidates visible; real tokenization is tested separately."""

    valid_label_columns = ["teacher_scores"]

    def __call__(self, rows):
        return rows


def examples(count=80, first_source_count=60, with_scores=True):
    rows = []
    for index in range(count):
        row = {
            "query": f"query-{index}",
            "positive": f"positive-{index}",
            "negatives": [f"negative-{index}-{candidate}" for candidate in range(10)],
            "source": "large" if index < first_source_count else "small",
        }
        if with_scores:
            row["teacher_scores"] = [1000 + index] + [100 * index + candidate for candidate in range(10)]
        rows.append(row)
    return Dataset.from_list(rows)


def loader_for(dataset, *, alpha=None, negatives=7, workers=0, world_size=1, rank=0, seed=42):
    dataset, collator, factory, _ = configure_sampling(
        dataset, KeepRows(), batch_size=4, alpha=alpha,
        num_negatives=negatives, seed=seed, world_size=world_size,
    )
    sampler = factory(dataset, batch_size=4, drop_last=True)
    extra = {"prefetch_factor": 3, "persistent_workers": True} if workers else {}
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator, num_workers=workers, **extra)
    return prepare_data_loader(
        loader, num_processes=world_size, process_index=rank,
        put_on_device=False, even_batches=False,
    )


class SamplingTests(unittest.TestCase):
    def test_both_options_unset_preserve_objects(self):
        dataset, collator = examples(), KeepRows()
        result = configure_sampling(dataset, collator, batch_size=4)
        self.assertIs(result[0], dataset)
        self.assertIs(result[1], collator)
        self.assertIsNone(result[2])
        self.assertEqual(result[3], {"enabled": False})

    def test_source_probabilities_and_no_duplicate_rows_inside_batches(self):
        dataset = examples(8000, 7200)
        sampler = SourceSamplingBatchSampler(dataset, 4, True, alpha=0.5, seed=23)
        expected = math.sqrt(7200) / (math.sqrt(7200) + math.sqrt(800))
        self.assertAlmostEqual(sampler.source_probabilities["large"], expected)
        counts = Counter()
        for batch in sampler:
            self.assertEqual(len(set(batch)), 4)
            sources = {"large" if index < 7200 else "small" for index in batch}
            self.assertEqual(len(sources), 1)
            counts[sources.pop()] += 1
        self.assertLess(abs(counts["large"] / len(sampler) - expected), 0.04)

    def test_alpha_zero_uniform_sources_alpha_one_natural_counts(self):
        dataset = examples()
        uniform = SourceSamplingBatchSampler(dataset, 4, True, alpha=0)
        natural = SourceSamplingBatchSampler(dataset, 4, True, alpha=1)
        self.assertEqual(uniform.source_probabilities, {"large": 0.5, "small": 0.5})
        self.assertAlmostEqual(natural.source_probabilities["large"], 0.75)

    def test_negative_only_first_epoch_matches_random_sampler(self):
        dataset = examples().remove_columns("source")
        sampler = SourceSamplingBatchSampler(dataset, 4, True, include_draw_ids=True, seed=42)
        actual = [draw.row for batch in sampler for draw in batch]
        expected = list(RandomSampler(dataset, generator=torch.Generator().manual_seed(42)))
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(set(actual)))

    def test_source_only_preserves_all_candidates_and_plain_dataset(self):
        original = examples()
        dataset, collator, factory, metadata = configure_sampling(
            original, KeepRows(), batch_size=4, alpha=0.5,
        )
        self.assertIs(dataset, original)
        self.assertIsInstance(collator, KeepRows)
        self.assertEqual(metadata["source_counts"], {"large": 60, "small": 20})
        self.assertEqual(metadata["global_epoch_draws"], 80)
        batch = next(iter(factory(dataset, batch_size=4, drop_last=True)))
        self.assertTrue(all(isinstance(index, int) for index in batch))
        self.assertEqual(len(dataset[batch[0]]["negatives"]), 10)

    def test_epoch_reconstruction_and_draw_identity(self):
        dataset = examples()
        sampler = SourceSamplingBatchSampler(dataset, 4, True, alpha=0.5, include_draw_ids=True)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        self.assertEqual([x.draw for batch in first for x in batch], list(range(80)))
        sampler.set_epoch(3)
        later = list(sampler)
        self.assertNotEqual(first, later)
        self.assertTrue(all(x.epoch == 3 for batch in later for x in batch))
        sampler.set_epoch(0)
        self.assertEqual(first, list(sampler))

    def test_uniform_subset_teacher_alignment_and_no_input_mutation(self):
        dataset = OccurrenceDataset(examples())
        collator = NegativeSamplingCollator(KeepRows(), 7, seed=11)
        original = dataset[DrawIndex(5, 2, 17)]
        untouched = pickle.loads(pickle.dumps(original))
        row = collator([original])[0]
        self.assertEqual(original, untouched)
        self.assertEqual(row["positive"], "positive-5")
        self.assertEqual(row["teacher_scores"][0], 1005)
        self.assertEqual(len(row["negatives"]), 7)
        positions = [int(text.rsplit("-", 1)[1]) for text in row["negatives"]]
        self.assertEqual(positions, sorted(set(positions)))
        self.assertEqual(row["teacher_scores"][1:], [500 + position for position in positions])
        self.assertEqual(collator([original])[0], row)
        self.assertNotIn("__slideon_sampling_draw__", row)
        visits = [collator([dataset[DrawIndex(5, 2, draw)]])[0]["negatives"] for draw in range(10)]
        self.assertGreater(len({tuple(visit) for visit in visits}), 1)

    def test_all_negatives_and_unsupervised_rows(self):
        dataset = OccurrenceDataset(examples(with_scores=False))
        row = dataset[DrawIndex(0, 0, 0)]
        result = NegativeSamplingCollator(KeepRows(), 10)([row])[0]
        self.assertEqual(result["negatives"], row["negatives"])
        self.assertNotIn("teacher_scores", result)

    def test_hf_dataset_view_preserves_indices_format_and_pickle(self):
        original = examples().select([9, 2, 4, 17]).with_format("python")
        dataset = OccurrenceDataset(original)
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(dataset._fingerprint, original._fingerprint)
        self.assertEqual(dataset.format, original.format)
        self.assertEqual(dataset[:], original[:])
        self.assertEqual(list(dataset["query"]), list(original["query"]))
        self.assertEqual(dataset[DrawIndex(1, 2, 3)]["query"], "query-2")
        restored = pickle.loads(pickle.dumps(dataset))
        fetched = restored.__getitems__([DrawIndex(1, 2, 3), DrawIndex(0, 2, 4)])
        self.assertEqual([row["query"] for row in fetched], ["query-2", "query-9"])

    def test_accelerate_shards_global_stream_once_and_forwards_epoch(self):
        dataset = examples(83, 60)
        rank_loaders = [loader_for(dataset, alpha=0.5, world_size=2, rank=rank) for rank in range(2)]
        dataset, collator, factory, metadata = configure_sampling(
            dataset, KeepRows(), batch_size=4, alpha=0.5, num_negatives=7, world_size=2,
        )
        sampler = factory(dataset, batch_size=4, drop_last=True)
        sampler.set_epoch(3)
        expected = list(DataLoader(dataset, batch_sampler=sampler, collate_fn=collator))
        self.assertEqual(metadata["global_epoch_batches"], 20)
        self.assertEqual(metadata["global_epoch_draws"], 80)
        for rank, loader in enumerate(rank_loaders):
            loader.set_epoch(3)
            self.assertEqual(list(loader), expected[rank::2])

    def test_accelerate_skip_replays_candidates_at_nonzero_epoch(self):
        dataset = examples()
        for alpha, negatives in ((None, 7), (0.5, None), (0.5, 7)):
            for world_size in (1, 2):
                with self.subTest(alpha=alpha, negatives=negatives, world_size=world_size):
                    full_loader = loader_for(dataset, alpha=alpha, negatives=negatives, world_size=world_size)
                    full_loader.set_epoch(3)
                    full = list(full_loader)
                    resumed_loader = loader_for(dataset, alpha=alpha, negatives=negatives, world_size=world_size)
                    # Trainer calls on_epoch_begin on its original loader, then
                    # adds SkipBatchSampler, THEN calls set_epoch. This ordering
                    # must work when a fresh resumed sampler still starts at 0.
                    SamplingEpochCallback().on_epoch_begin(
                        None, SimpleNamespace(epoch=3.3), None, train_dataloader=resumed_loader,
                    )
                    skipped = skip_first_batches(resumed_loader, 3)
                    skipped.set_epoch(3)
                    self.assertEqual(list(skipped), full[3:])

    def test_prefetch_workers_do_not_change_candidates_or_resume(self):
        dataset = examples()
        expected_loader = loader_for(dataset, alpha=0.5)
        expected_loader.set_epoch(2)
        expected = list(expected_loader)
        worker_loader = loader_for(dataset, alpha=0.5, workers=2)
        worker_loader.set_epoch(2)
        self.assertEqual(list(worker_loader), expected)
        worker_loader.set_epoch(2)
        self.assertEqual(list(skip_first_batches(worker_loader, 4)), expected[4:])

    def test_invalid_source_and_batch_shapes_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "source"):
            SourceSamplingBatchSampler(examples().remove_columns("source"), 4, True, alpha=0.5)
        with self.assertRaisesRegex(ValueError, "fewer than batch_size"):
            SourceSamplingBatchSampler(examples(8, 7), 4, True, alpha=0.5)
        with self.assertRaisesRegex(ValueError, "source string"):
            SourceSamplingBatchSampler(examples().add_column("bad", [0] * 80).remove_columns("source").rename_column("bad", "source"), 4, True, alpha=0.5)
        for alpha in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "alpha"):
                SourceSamplingBatchSampler(examples(), 4, True, alpha=alpha)

    def test_invalid_teacher_alignment_and_insufficient_negatives(self):
        dataset = OccurrenceDataset(examples())
        collator = NegativeSamplingCollator(KeepRows(), 7)
        row = dataset[DrawIndex(0, 0, 0)]
        row["teacher_scores"] = row["teacher_scores"][:-1]
        with self.assertRaisesRegex(ValueError, "one score per negative"):
            collator([row])
        row = dataset[DrawIndex(0, 0, 0)]
        row["negatives"] = row["negatives"][:6]
        with self.assertRaisesRegex(ValueError, "at least 7"):
            collator([row])
        with self.assertRaisesRegex(ValueError, "OccurrenceDataset"):
            collator([examples()[0]])

    def test_factory_captures_explicit_seed_and_validates_settings(self):
        dataset, _, factory, _ = configure_sampling(examples(), KeepRows(), batch_size=4, alpha=0.5, seed=17)
        sampler = factory(dataset, batch_size=4, drop_last=True, seed=0)
        self.assertEqual(sampler.seed, 17)
        with self.assertRaisesRegex(ValueError, "batch_size/drop_last"):
            factory(dataset, batch_size=8, drop_last=True)


if __name__ == "__main__":
    unittest.main()
