"""Optional training-time source mixing and aligned negative subsampling.

The ordinary training path does not use this module. ``configure_sampling``
connects these options through Sentence Transformers' public ``batch_sampler``
argument and the existing retrieval collator, without replacing the Trainer.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

import torch
from datasets import Dataset
from torch.utils.data import BatchSampler, Sampler
from transformers import TrainerCallback


# This field exists only in fetched rows, never in the saved Arrow/JSONL data.
# A sampler occurrence is distinct from a row: oversampling can visit a row
# several times in one epoch and each visit should have its own negative subset.
_DRAW_KEY = "__slideon_sampling_draw__"


class DrawIndex(NamedTuple):
    row: int
    epoch: int
    draw: int


class OccurrenceDataset(Dataset):
    """An HF Dataset view that accepts sampler indices carrying draw identity.

    A plain PyTorch Dataset wrapper would fail Sentence Transformers' dataset
    type checks. Shallow-copying the HF Dataset state preserves its Arrow table,
    selected-row indices, formatting and fingerprint without copying the data.
    Normal column/slice/integer access still has the standard HF behavior.
    """

    def __init__(self, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError("Sampling requires a map-style Hugging Face Dataset.")
        if _DRAW_KEY in dataset.column_names:
            raise ValueError(f"Input dataset uses reserved column {_DRAW_KEY!r}.")
        self.__dict__.update(dataset.__dict__)

    def __getitem__(self, key):
        if isinstance(key, DrawIndex):
            row = dict(super().__getitem__(key.row))
            row[_DRAW_KEY] = key
            return row
        return super().__getitem__(key)

    def __getitems__(self, keys):
        # PyTorch's batched fetcher calls this method, bypassing individual
        # __getitem__ calls. Fetch the Arrow rows together, then attach identity.
        if keys and isinstance(keys[0], DrawIndex):
            rows = super().__getitems__([key.row for key in keys])
            for row, key in zip(rows, keys):
                row[_DRAW_KEY] = key
            return rows
        return super().__getitems__(keys)


class _EpochRows(Sampler):
    """Expose set_epoch where Accelerate looks after wrapping batch samplers."""

    def __init__(self, size: int, seed: int):
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        # Reconstruct an epoch rather than advancing a persistent RNG: skipped
        # batches, DataLoader prefetch and checkpoint replay cannot move it ahead.
        generator = torch.Generator().manual_seed((self.seed + self.epoch) % (2**63))
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self):
        return self.size


def _group_sources(dataset: Dataset) -> dict[str, list[int]]:
    if "source" not in dataset.column_names:
        raise ValueError("Source sampling requires a non-empty 'source' string on every row.")
    groups = defaultdict(list)
    # Read only the source column; candidate texts stay in the Arrow dataset.
    for index, source in enumerate(dataset["source"]):
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Source sampling requires a non-empty source string; row {index} is invalid.")
        groups[source].append(index)
    return dict(sorted(groups.items()))


class SourceSamplingBatchSampler(BatchSampler):
    """Generate one reproducible global batch stream for Accelerate to shard.

    With alpha enabled, choose a source per microbatch with probability
    N_source**alpha / sum(N**alpha), then sample distinct rows uniformly inside
    that source. Sources/rows may recur across microbatches. This is a deliberate
    implementation of the reported source mixing, not recovered author code.

    With alpha=None, draw a normal no-replacement row permutation. This branch
    is needed only when runtime negatives require occurrence metadata; with both
    options disabled the original Trainer sampler remains untouched.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns=None,
        generator=None,
        seed: int = 42,
        *,
        alpha: float | None = None,
        world_size: int = 1,
        include_draw_ids: bool = False,
        source_indices: dict[str, list[int]] | None = None,
    ):
        super().__init__(_EpochRows(len(dataset), seed), batch_size, drop_last)
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
            raise ValueError("world_size must be a positive integer.")
        if not drop_last and world_size != 1:
            raise ValueError("Distributed optional sampling requires drop_last=True for equal-size batches.")
        if alpha is not None and (not math.isfinite(alpha) or alpha < 0):
            raise ValueError("source sampling alpha must be finite and non-negative.")
        self.seed = seed
        self.alpha = alpha
        self.world_size = world_size
        self.include_draw_ids = include_draw_ids
        self.valid_label_columns = valid_label_columns
        self.dataset_size = len(dataset)

        # Drop an incomplete group of rank-local microbatches. All processes see
        # this SAME stream; only Accelerate selects their rank's batches.
        batches = len(dataset) // batch_size if drop_last else math.ceil(len(dataset) / batch_size)
        self.num_batches = batches - batches % world_size
        if not self.num_batches:
            raise ValueError("Dataset is too small for one full batch on every process.")
        self.epoch_draws = min(len(dataset), self.num_batches * batch_size)
        self.source_indices = {}
        if alpha is not None:
            self.source_indices = source_indices if source_indices is not None else _group_sources(dataset)
        self.sources = list(self.source_indices)
        self.source_counts = {source: len(indices) for source, indices in self.source_indices.items()}
        self.source_probabilities = {}
        if alpha is not None:
            for source, count in self.source_counts.items():
                if count < batch_size:
                    raise ValueError(
                        f"Source {source!r} has {count} rows, fewer than batch_size={batch_size}; "
                        "source-homogeneous batches require distinct rows."
                    )
            # Log-space normalization also handles large, finite alpha values.
            logs = [math.log(count) for count in self.source_counts.values()]
            weights = [math.exp(alpha * (value - max(logs))) for value in logs]
            total = sum(weights)
            self.source_probabilities = dict(zip(self.sources, (weight / total for weight in weights)))

    @property
    def epoch(self):
        return self.sampler.epoch

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Capture the epoch for this iterator: changing a subsequent iterator's
        # epoch must not alter already-prefetched records from the old one.
        epoch = self.epoch
        if self.alpha is None:
            row_iterator = iter(self.sampler)
        else:
            rng = random.Random(self.seed + epoch)
            source_weights = list(self.source_probabilities.values())
        for batch_number in range(self.num_batches):
            size = min(self.batch_size, self.epoch_draws - batch_number * self.batch_size)
            if self.alpha is None:
                indices = [next(row_iterator) for _ in range(size)]
            else:
                source = rng.choices(self.sources, weights=source_weights, k=1)[0]
                indices = rng.sample(self.source_indices[source], size)
            if self.include_draw_ids:
                yield [
                    DrawIndex(index, epoch, batch_number * self.batch_size + offset)
                    for offset, index in enumerate(indices)
                ]
            else:
                yield indices


class NegativeSamplingCollator:
    """Select a uniform negative subset and its teacher scores, then tokenize.

    Randomness is a pure function of seed and sampler occurrence. Re-fetching
    a batch after checkpoint resume gives the same candidates even with worker
    prefetch; a later draw of the same row can receive a different subset.
    """

    def __init__(self, collator, num_negatives: int, seed: int = 42):
        if not isinstance(num_negatives, int) or isinstance(num_negatives, bool) or num_negatives < 1:
            raise ValueError("num_negatives must be a positive integer.")
        self.collator = collator
        self.num_negatives = num_negatives
        self.seed = seed
        self.valid_label_columns = collator.valid_label_columns

    def __call__(self, rows):
        selected = []
        for original in rows:
            row = dict(original)
            draw = row.pop(_DRAW_KEY, None)
            if not isinstance(draw, DrawIndex):
                raise ValueError("Runtime negative sampling requires OccurrenceDataset and its batch sampler.")
            negatives = row.get("negatives")
            if not isinstance(negatives, (list, tuple)) or len(negatives) < self.num_negatives:
                raise ValueError(f"Row {draw.row} needs at least {self.num_negatives} negatives.")
            scores = row.get("teacher_scores")
            if "teacher_scores" in row and (
                not isinstance(scores, (list, tuple)) or len(scores) != len(negatives) + 1
            ):
                raise ValueError(
                    f"Row {draw.row}: teacher_scores must contain positive score "
                    "followed by one score per negative."
                )

            # Python's process-randomized hash() is unsuitable for reproducible
            # seeds. A stable digest makes worker count and call order irrelevant.
            key = f"{self.seed}:{draw.epoch}:{draw.draw}:{draw.row}".encode("ascii")
            subseed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
            positions = sorted(random.Random(subseed).sample(range(len(negatives)), self.num_negatives))
            row["negatives"] = [negatives[index] for index in positions]
            if scores is not None:
                # Candidate zero is always the positive; negative j has score j+1.
                row["teacher_scores"] = [scores[0]] + [scores[index + 1] for index in positions]
            selected.append(row)
        return self.collator(selected)


class SamplingEpochCallback(TrainerCallback):
    """Set sampler epoch before Trainer adds its checkpoint-skip wrapper.

    On a distributed mid-epoch resume, Accelerate's fixed-depth forwarding can
    miss the sampler behind SkipBatchSampler -> BatchSamplerShard. The public
    callback receives the original training loader before that wrapper is added;
    walk its sampler chain explicitly so occurrence IDs retain the right epoch.
    Register this callback whenever configure_sampling returns a sampler factory.
    """

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        epoch = math.floor(state.epoch or 0)
        pending, visited = [train_dataloader], set()
        found = False
        while pending:
            node = pending.pop()
            if node is None or id(node) in visited:
                continue
            visited.add(id(node))
            if isinstance(node, SourceSamplingBatchSampler):
                node.set_epoch(epoch)
                found = True
                continue
            pending.extend(getattr(node, name, None) for name in ("batch_sampler", "sampler", "dataloader"))
        if not found:
            raise RuntimeError("SamplingEpochCallback could not find the optional sampling batch sampler.")
        return control


@dataclass
class _SamplingFactory:
    """Picklable adapter for the official Sentence Transformers sampler hook."""

    batch_size: int
    drop_last: bool
    configured_seed: int
    alpha: float | None
    world_size: int
    include_draw_ids: bool
    source_indices: dict[str, list[int]]

    def __call__(self, dataset, batch_size, drop_last, valid_label_columns=None, generator=None, seed=0):
        if batch_size != self.batch_size or drop_last != self.drop_last:
            raise ValueError("Trainer batch_size/drop_last differ from configure_sampling settings.")
        # ST 6.1's dataloader hook supplies seed=0 even when args.seed differs.
        # Capture the explicit training seed instead of accepting that default.
        return SourceSamplingBatchSampler(
            dataset, batch_size, drop_last, valid_label_columns=valid_label_columns,
            seed=self.configured_seed, alpha=self.alpha, world_size=self.world_size,
            include_draw_ids=self.include_draw_ids, source_indices=self.source_indices,
        )


def configure_sampling(
    dataset: Dataset,
    collator,
    *,
    batch_size: int,
    alpha: float | None = None,
    num_negatives: int | None = None,
    seed: int = 42,
    world_size: int = 1,
    drop_last: bool = True,
):
    """Return (dataset, collator, optional sampler factory, run metadata).

    Pass the returned factory as TrainingArguments(batch_sampler=factory) and
    register SamplingEpochCallback only when the factory is not None. Both
    options unset return the input objects unchanged.
    Reproduction requires the same data order, seed, batch size and world size;
    Trainer's standard resume path must retain its default ignore_data_skip=False.
    """
    if alpha is None and num_negatives is None:
        return dataset, collator, None, {"enabled": False}
    sampler = SourceSamplingBatchSampler(
        dataset, batch_size, drop_last, seed=seed, alpha=alpha, world_size=world_size,
        include_draw_ids=num_negatives is not None,
    )
    if num_negatives is not None:
        dataset = OccurrenceDataset(dataset)
        collator = NegativeSamplingCollator(collator, num_negatives, seed)
    factory = _SamplingFactory(
        batch_size, drop_last, seed, alpha, world_size, num_negatives is not None, sampler.source_indices,
    )
    metadata = {
        "enabled": True,
        "source_sampling_alpha": alpha,
        "sample_negatives": num_negatives,
        "seed": seed,
        "world_size": world_size,
        "batch_size": batch_size,
        "source_counts": sampler.source_counts,
        "source_probabilities": sampler.source_probabilities,
        "global_epoch_batches": len(sampler),
        "global_epoch_draws": sampler.epoch_draws,
        "source_sampling": (
            "source-homogeneous microbatches; distinct rows within each batch; replacement across batches"
            if alpha is not None else "shuffled rows without replacement"
        ),
        "negative_sampling": (
            "uniform subset without replacement per occurrence; original candidate order retained"
            if num_negatives is not None else None
        ),
        "epoch_seed": "training_seed + epoch",
        "distributed_sharding": "one global stream, sharded once by Accelerate",
    }
    return dataset, collator, factory, metadata
