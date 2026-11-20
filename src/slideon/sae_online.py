"""Online frozen-token batches backed by JSONL texts or pretokenized PT records.

The on-disk index contains file offsets and token counts, never embeddings.
Each refill encodes one text batch per worker. Its token vectors stay in memory
only until the fixed-size SAE batches have consumed them.
"""

from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .sae_texts import validate_record
from .sae_tokens import PretokenizedDataset


# Group identity supplies source/role, so each text needs only sixteen bytes.
RECORD_DTYPE = np.dtype([("file", "<u4"), ("offset", "<u8"), ("tokens", "<u4")])
ENCODING_KEYS = ("name", "base_model", "base_model_revision", "d_model", "d_lift", "bf16",
                 "query_length", "document_length", "query_prompt", "document_prompt")


def _write_manifest(path, manifest):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def _fingerprint(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"JSONL input changed while checking its fingerprint: {path}")
    return dict(path=str(path), bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                sha256=digest.hexdigest())


def _without_checksums(value):
    """Compare startup metadata without re-reading large shards under NCCL."""
    if isinstance(value, dict):
        return {key: _without_checksums(item) for key, item in value.items() if key != "sha256"}
    if isinstance(value, list):
        return [_without_checksums(item) for item in value]
    return value


class OnlineTokenSource:
    """Indexed texts, frozen encoder and the exact finite token-training budget.

    ``encoder`` provides count_tokens(records) and encode_records(records). Both
    use the same role-aware preprocessing; only the latter runs DenseOn. Encoding
    returns one FP16 tensor per text, after padding exclusion and normalization.
    PT inputs also require matching encoder.tokenization_metadata(), since their
    prompts, special tokens and truncation have already been applied on disk.
    """

    def __init__(self, recipe, texts_jsonl, index_path, encoder, encode_batch_size,
                 device, rank=0, world=1):
        if encode_batch_size < 1 or recipe.batch_size % world:
            raise ValueError("Encoding batch must be positive and global token batch divisible by world size")
        self.recipe, self.encoder = recipe, encoder
        self.encode_batch_size = encode_batch_size
        self.device, self.rank, self.world = torch.device(device), rank, world
        self.root = Path(index_path)
        self.paths = [Path(path).resolve() for path in texts_jsonl]
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("Supply one or more distinct JSONL or PT input files")
        if any(path.suffix.lower() not in (".jsonl", ".pt") for path in self.paths):
            raise ValueError("Online input files must end in .jsonl or .pt")
        self._streams = {}
        self._tokenized = {}
        self.arrays = []
        self.encoding = {key: recipe.checkpoint_config()[key] for key in ENCODING_KEYS}

        # Long indexing/checksum scans happen in a single-process --index-only
        # invocation BEFORE torchrun. Distributed startup performs only quick file
        # metadata checks, so workers cannot time out while waiting for a huge scan.
        # Broadcast rank zero's failure before peers try to open the index.
        error = None
        if rank == 0:
            try:
                self._prepare_index()
            except Exception as exc:
                if world == 1:
                    self.close()
                    raise
                error = f"{type(exc).__name__}: {exc}"
        if world > 1:
            message = [error]
            dist.broadcast_object_list(message, src=0)
            if message[0] is not None:
                self.close()
                raise RuntimeError(f"Online input indexing failed: {message[0]}")

        try:
            self.manifest = json.loads((self.root / "manifest.json").read_text())
            if not self.manifest.get("complete") or self.manifest.get("encoding") != self.encoding:
                raise ValueError("Online text index is incomplete or has different encoding settings")
            self.groups = self.manifest["groups"]
            self.ends = np.cumsum([group["rows"] for group in self.groups], dtype=np.int64)
            self.total_rows = int(self.ends[-1])
            self.total_tokens = sum(group["tokens"] for group in self.groups)
            self.steps = self.total_tokens * recipe.epochs // recipe.batch_size
            for group in self.groups:
                path = self.root / group["index"]
                if path.stat().st_size != group["rows"] * RECORD_DTYPE.itemsize:
                    raise ValueError(f"Invalid online index size: {path}")
                self.arrays.append(np.memmap(path, mode="r", dtype=RECORD_DTYPE, shape=(group["rows"],)))
        except Exception:
            self.close()
            raise

    def _tokenized_reader(self, file_index):
        if file_index not in self._tokenized:
            self._tokenized[file_index] = PretokenizedDataset(
                self.paths[file_index], expected_tokenization=self.encoder.tokenization_metadata())
        return self._tokenized[file_index]

    def _input_fingerprint(self, file_index, *, full):
        path = self.paths[file_index]
        if path.suffix.lower() == ".pt":
            return self._tokenized_reader(file_index).fingerprint(full=full)
        if full:
            return _fingerprint(path)
        stat = path.stat()
        return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

    def _prepare_index(self):
        manifest_path = self.root / "manifest.json"
        if self.world > 1 and not manifest_path.exists():
            raise ValueError("Online distributed training requires an index; run --online --index-only first")
        self.root.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("type") != "online-jsonl-index" or previous.get("version") != 1:
                raise ValueError("Index directory contains a different data format; choose a new --index-dir")
            if previous.get("complete"):
                if previous.get("encoding") != self.encoding:
                    raise ValueError("Online index encoding settings changed; choose a new --index-dir")
                if self.world > 1:
                    # The launcher just verified complete SHA256 fingerprints in
                    # its CPU-only stage. Check that files have not changed since
                    # that scan without doing another multi-hour read under NCCL.
                    files = [self._input_fingerprint(index, full=False) for index in range(len(self.paths))]
                    expected = _without_checksums(previous["files"])
                    if files != expected:
                        raise ValueError("Online input metadata changed; run --online --index-only first")
                    return
                print("Checking online input fingerprints...", flush=True)
                files = [self._input_fingerprint(index, full=True) for index in range(len(self.paths))]
                if previous["files"] != files:
                    raise ValueError("Online input differs from the index; choose a new --index-dir")
                return
            if self.world > 1:
                raise ValueError("Online index is incomplete; run --online --index-only first")

        manifest = dict(type="online-jsonl-index", version=1, encoding=self.encoding,
                        files=[], groups=[], complete=False)
        _write_manifest(manifest_path, manifest)
        print("Indexing input records and token counts (no DenseOn forward pass)...", flush=True)
        groups, pending = {}, []
        with ExitStack() as stack:
            def append_record(source, role, file_index, offset, count):
                count = int(count)
                if count < 1 or count > np.iinfo(np.uint32).max:
                    raise ValueError("Each indexed text must produce a positive uint32 token count")
                key = (source, role)
                if key not in groups:
                    name = f"records-{len(groups):05d}.idx"
                    stream = stack.enter_context((self.root / (name + ".partial")).open("wb"))
                    groups[key] = dict(source=key[0], role=key[1], index=name,
                                       rows=0, tokens=0, stream=stream)
                group = groups[key]
                item = np.array((file_index, offset, count), dtype=RECORD_DTYPE)
                group["stream"].write(item.tobytes())
                group["rows"] += 1
                group["tokens"] += count

            def flush():
                if not pending:
                    return
                counts = self.encoder.count_tokens([item[0] for item in pending])
                if len(counts) != len(pending):
                    raise ValueError("Tokenizer returned an incorrect number of token counts")
                for (record, file_index, offset), count in zip(pending, counts):
                    append_record(record["source"], record["role"], file_index, offset, count)
                pending.clear()

            for file_index, path in enumerate(self.paths):
                if path.suffix.lower() == ".pt":
                    # Preserve record/group order across mixed inputs. PT sequences
                    # already include prompts/special tokens and need no tokenizer.
                    flush()
                    reader = self._tokenized_reader(file_index)
                    identity = reader.fingerprint(full=True)
                    for row, source, role, count in reader.iter_index():
                        append_record(source, role, file_index, row, count)
                    if reader.fingerprint(full=False) != _without_checksums(identity):
                        raise ValueError(f"Pretokenized input changed during indexing: {path}")
                    manifest["files"].append(identity)
                    continue
                before = path.stat()
                digest, size = hashlib.sha256(), 0
                with path.open("rb") as stream:
                    for line_number, line in enumerate(stream, 1):
                        offset = size
                        digest.update(line)
                        size += len(line)
                        location = f"{path}:{line_number}"
                        try:
                            record = json.loads(line)
                        except (ValueError, UnicodeDecodeError) as exc:
                            raise ValueError(f"{location}: invalid JSON record") from exc
                        record = validate_record(record, location)
                        pending.append((record, file_index, offset))
                        if len(pending) == self.encode_batch_size:
                            flush()
                        if line_number % 100000 == 0:
                            print(f"  Indexed {path.name}: {line_number:,} texts", flush=True)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"JSONL input changed during indexing: {path}")
                manifest["files"].append(dict(path=str(path), bytes=size, mtime_ns=after.st_mtime_ns,
                                              sha256=digest.hexdigest()))
            flush()
            if not groups or not any(role == "document" for _, role in groups):
                raise ValueError("Online input needs at least one document for SAE initialization")

        for group in groups.values():
            (self.root / (group["index"] + ".partial")).replace(self.root / group["index"])
            manifest["groups"].append({key: value for key, value in group.items() if key != "stream"})
        manifest["complete"] = True
        _write_manifest(manifest_path, manifest)
        rows = sum(group["rows"] for group in manifest["groups"])
        tokens = sum(group["tokens"] for group in manifest["groups"])
        print(f"Indexed {rows:,} texts / {tokens:,} retained tokens; no embeddings saved", flush=True)

    def location(self, index):
        part = int(np.searchsorted(self.ends, index, side="right"))
        return part, int(index - (self.ends[part - 1] if part else 0))

    def token_count(self, index):
        part, row = self.location(index)
        return int(self.arrays[part][row]["tokens"])

    def record(self, index):
        part, row = self.location(index)
        item = self.arrays[part][row]
        file_index = int(item["file"])
        if self.paths[file_index].suffix.lower() == ".pt":
            # offset is a PT record number instead of a JSONL byte position.
            record = self._tokenized_reader(file_index).record(int(item["offset"]))
        else:
            if file_index not in self._streams:
                self._streams[file_index] = self.paths[file_index].open("rb")
            stream = self._streams[file_index]
            stream.seek(int(item["offset"]))
            record = validate_record(json.loads(stream.readline()), f"{self.paths[file_index]} byte {int(item['offset'])}")
        group = self.groups[part]
        if (record["source"], record["role"]) != (group["source"], group["role"]):
            raise ValueError("Input source/role changed after indexing")
        return record

    def encode(self, indices):
        if not indices:
            return torch.empty((0, self.recipe.d_model), dtype=torch.float16, device=self.device)
        arrays = self.encoder.encode_records([self.record(index) for index in indices])
        if len(arrays) != len(indices):
            raise ValueError("Frozen encoder returned an incorrect number of sequences")
        tensors = []
        for index, array in zip(indices, arrays):
            array = torch.as_tensor(array, device=self.device, dtype=torch.float16)
            if array.shape != (self.token_count(index), self.recipe.d_model) or not torch.isfinite(array).all():
                raise ValueError("Frozen embeddings differ from indexed token counts or contain nonfinite values")
            tensors.append(array.detach())
        # Concatenation outside encoder inference mode produces ordinary tensors
        # that are safe for the SAE's autograd operations.
        return torch.cat(tensors, dim=0)

    def initialization_tokens(self, rng):
        indices = []
        for part, group in enumerate(self.groups):
            if group["role"] == "document":
                start = int(self.ends[part - 1]) if part else 0
                indices.extend(start + row for row in rng.sample(
                    range(group["rows"]), min(self.recipe.init_docs_per_split, group["rows"])))
        arrays = [self.encode(indices[start:start + self.encode_batch_size]).cpu()
                  for start in range(0, len(indices), self.encode_batch_size)]
        return torch.cat(arrays, dim=0)

    def batches(self, state=None):
        return OnlineTokenBatches(self, state=state)

    def close(self):
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()
        for array in self.arrays:
            array._mmap.close()
        self.arrays.clear()
        for reader in self._tokenized.values():
            reader.close()
        self._tokenized.clear()


class OnlineTokenBatches:
    """Finite, fixed-token batches with no text replacement and exact replay.

Each segment records a refill's text references and how many shuffled vectors
were consumed. Resume recreates only these pending segments using frozen DenseOn.
"""

    def __init__(self, source, state=None):
        self.source = source
        self.recipe = source.recipe
        self.global_batch = self.recipe.batch_size
        self.local_batch = self.global_batch // source.world
        self._state = copy.deepcopy(state) if state is not None else dict(
            epoch=0, cursor=0, refill=0, segments=[])
        self._segments = []
        self._restored = state is None
        self._order_epoch, self._order = -1, None

    @property
    def state(self):
        return copy.deepcopy(self._state)

    def __iter__(self):
        return self

    def _encode_segment(self, segment):
        source = self.source
        refs = segment["refs"]
        start = source.rank * source.encode_batch_size
        selected = refs[start:start + source.encode_batch_size]
        error, local = None, None
        try:
            local = source.encode(selected)
        except Exception as exc:
            if source.world == 1:
                raise
            error = f"rank {source.rank}: {type(exc).__name__}: {exc}"
        if source.world > 1:
            errors = [None] * source.world
            dist.all_gather_object(errors, error)
            if any(errors):
                raise RuntimeError(f"Online DenseOn encoding failed: {errors}")
            sizes = [sum(source.token_count(index) for index in refs[
                rank * source.encode_batch_size:(rank + 1) * source.encode_batch_size])
                for rank in range(source.world)]
            padded = local.new_zeros((max(sizes), self.recipe.d_model))
            padded[:len(local)] = local
            gathered = [torch.empty_like(padded) for _ in range(source.world)]
            dist.all_gather(gathered, padded)
            tokens = torch.cat([array[:size] for array, size in zip(gathered, sizes)], dim=0)
        else:
            tokens = local
        order = torch.randperm(len(tokens), generator=torch.Generator().manual_seed(
            (self.recipe.seed + segment["refill"]) % (2**63))).to(source.device)
        return tokens[order]

    def _restore(self):
        if not self._restored:
            self._segments = [(segment, self._encode_segment(segment)) for segment in self._state["segments"]]
            self._restored = True

    def _refill(self):
        while self._state["epoch"] < self.recipe.epochs:
            epoch = self._state["epoch"]
            if self._order_epoch != epoch:
                # Release the previous epoch's record permutation before allocating
                # another one; the corpus can contain hundreds of millions of rows.
                self._order = None
                self._order = np.random.default_rng(self.recipe.seed + epoch).permutation(self.source.total_rows)
                self._order_epoch = epoch
            cursor = self._state["cursor"]
            if cursor == self.source.total_rows:
                self._state.update(epoch=epoch + 1, cursor=0)
                continue
            end = min(cursor + self.source.encode_batch_size * self.source.world, self.source.total_rows)
            segment = dict(refs=self._order[cursor:end].tolist(), refill=self._state["refill"], consumed=0)
            tokens = self._encode_segment(segment)
            self._state["segments"].append(segment)
            self._segments.append((segment, tokens))
            self._state.update(cursor=end, refill=self._state["refill"] + 1)
            return True
        return False

    def _available(self):
        return sum(len(tokens) - segment["consumed"] for segment, tokens in self._segments)

    def __next__(self):
        self._restore()
        while self._available() < self.global_batch:
            if not self._refill():
                raise StopIteration
        pieces, needed = [], self.global_batch
        while needed:
            segment, tokens = self._segments[0]
            start = segment["consumed"]
            count = min(needed, len(tokens) - start)
            pieces.append(tokens[start:start + count])
            segment["consumed"] += count
            needed -= count
            if segment["consumed"] == len(tokens):
                self._segments.pop(0)
                self._state["segments"].pop(0)
        # Each rank sees an equal slice of the SAME global token batch. Gathering
        # refill vectors avoids dropping tokens when ranks encode unequal lengths.
        batch = torch.cat(pieces, dim=0)
        start = self.source.rank * self.local_batch
        return batch[start:start + self.local_batch].float()

    def finish(self):
        """Visit the final texts and discard only the terminal incomplete batch."""
        self._restore()
        while self._refill():
            if self._available() >= self.global_batch:
                raise ValueError("Cannot finish while complete SAE token batches remain")
        remaining = self._available()
        if remaining >= self.global_batch:
            raise ValueError("Cannot finish while complete SAE token batches remain")
        self._segments.clear()
        self._state["segments"].clear()
        return remaining
