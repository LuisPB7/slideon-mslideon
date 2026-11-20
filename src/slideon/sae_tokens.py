"""Compact pretokenized SAE inputs, not a cache of DenseOn embeddings.

One small .pt descriptor references bounded tensor shards. Each shard packs
unpadded uint16 token IDs, document offsets, source/role groups and UTF-8 IDs.
Token IDs use little-endian byte pairs because torch.save(uint16) is not
supported by the oldest PyTorch version allowed by this project (2.5).
The reader maps tensor storage from disk and retains only a small shard LRU.
"""

from bisect import bisect_right
from collections import OrderedDict
from array import array
import hashlib
import json
from pathlib import Path
import shutil

import torch


FORMAT = "slideon-pretokenized"
VERSION = 1


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _fingerprint(path, full=True):
    before = path.stat()
    result = dict(path=str(path.resolve()), bytes=before.st_size, mtime_ns=before.st_mtime_ns)
    if full:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Pretokenized input changed while reading: {path}")
    return result


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _check_token_range(max_token_id, metadata):
    """Reject data outside the recorded tokenizer vocabulary before encoding."""
    if "max_token_id" in metadata:
        limit = _integer(metadata["max_token_id"], "tokenizer max_token_id")
        if max_token_id > limit:
            raise ValueError("Input token ID exceeds the tokenizer's recorded max_token_id")


def _array_tensor(values, dtype):
    # Python array uses native byte order, matching torch.frombuffer. The
    # fallback covers a platform whose C integer width differs from the dtype.
    if values.itemsize == torch.empty((), dtype=dtype).element_size():
        return torch.frombuffer(values, dtype=dtype)
    return torch.tensor(values, dtype=dtype)


def _record_metadata(record):
    if not isinstance(record, dict):
        raise ValueError("Each token record must be a dictionary")
    source, role = record.get("source"), record.get("role", "document")
    if not isinstance(source, str) or not source.strip() or role not in ("query", "document"):
        raise ValueError("Each token record needs a nonblank source and query/document role")
    identifier = record.get("id")
    if "id" in record and (not isinstance(identifier, str) or not identifier.strip()):
        raise ValueError("An optional record id must be a nonblank string")
    return source, role, identifier


def write_pretokenized(records, output, encoder, *, tokenize_batch_size=512,
                      shard_tokens=8_000_000, provenance=None, limit=None):
    """Tokenize an iterator once and publish a descriptor only after all shards.

    ``encoder`` supplies tokenization_metadata() and tokenize_records(records).
    The latter returns records with source/role/id plus unpadded input_ids. Texts
    and their tokenized copies are retained for at most one tokenization batch.
    Completed outputs are immutable; use another path for different settings.
    """
    _integer(tokenize_batch_size, "tokenize_batch_size", 1)
    _integer(shard_tokens, "shard_tokens", 1)
    if limit is not None:
        _integer(limit, "limit", 1)
    output = Path(output)
    if output.suffix.lower() != ".pt":
        raise ValueError("Pretokenized output must end in .pt")
    directory = output.with_suffix(".shards")
    staging = directory.with_name(directory.name + ".partial")
    partial = output.with_name(output.name + ".partial")
    for path in (output, directory, staging, partial):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose another output")
    metadata = _json_copy(encoder.tokenization_metadata())
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Encoder must supply nonempty tokenization metadata")
    manifest = dict(format=FORMAT, version=VERSION, complete=False,
                    tokenization=metadata, provenance=_json_copy(provenance or {}),
                    dtype="uint16-le", groups=[], shards=[], rows=0, tokens=0, max_token_id=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    groups = {}
    ids, offsets, group_ids = bytearray(), array("q", [0]), array("i")
    identifier_bytes, identifier_offsets = bytearray(), array("q", [0])
    pending, published_directory = [], False

    def flush_shard():
        nonlocal ids, offsets, group_ids, identifier_bytes, identifier_offsets
        if not group_ids:
            return
        name = f"part-{len(manifest['shards']):06d}.pt"
        path = staging / name
        # Only tensors enter large shards: no Python object per input document.
        payload = dict(input_id_bytes=torch.frombuffer(ids, dtype=torch.uint8),
                       offsets=_array_tensor(offsets, torch.int64),
                       group_ids=_array_tensor(group_ids, torch.int32),
                       id_bytes=(torch.frombuffer(identifier_bytes, dtype=torch.uint8)
                                 if identifier_bytes else torch.empty(0, dtype=torch.uint8)),
                       id_offsets=_array_tensor(identifier_offsets, torch.int64))
        torch.save(payload, path)
        identity = _fingerprint(path)
        manifest["shards"].append(dict(file=f"{directory.name}/{name}", rows=len(group_ids),
                                      tokens=offsets[-1], bytes=identity["bytes"], sha256=identity["sha256"]))
        print(f"Saved {name}: {len(group_ids):,} texts / {offsets[-1]:,} input tokens", flush=True)
        ids, offsets, group_ids = bytearray(), array("q", [0]), array("i")
        identifier_bytes, identifier_offsets = bytearray(), array("q", [0])

    def tokenize_pending():
        if not pending:
            return
        tokenized = encoder.tokenize_records(pending)
        if len(tokenized) != len(pending):
            raise ValueError("Tokenizer returned the wrong number of records")
        for raw, record in zip(pending, tokenized):
            source, role, identifier = _record_metadata(record)
            if (source, role, identifier) != _record_metadata(raw):
                raise ValueError("Tokenization changed source, role or document id")
            values = torch.as_tensor(record.get("input_ids"))
            if values.ndim != 1 or not values.numel() or values.dtype not in (
                    torch.uint8, torch.int8, torch.int16, torch.uint16, torch.int32, torch.int64):
                raise ValueError("input_ids must be a nonempty one-dimensional integer sequence")
            values = values.detach().to(device="cpu", dtype=torch.int64)
            low, high = int(values.min()), int(values.max())
            if low < 0 or high > 65535:
                raise ValueError("Token IDs do not fit uint16 (0..65535); this tokenizer needs a wider format")
            _check_token_range(high, metadata)
            # A single document may exceed shard_tokens; never split its context.
            if group_ids and offsets[-1] + len(values) > shard_tokens:
                flush_shard()
            key = (source, role)
            if key not in groups:
                groups[key] = len(groups)
                manifest["groups"].append(dict(source=source, role=role, rows=0, tokens=0))
            group = manifest["groups"][groups[key]]
            # One byte buffer avoids retaining millions of tiny tensors for a
            # shard of short queries. Explicit little-endian uint16 is portable;
            # torch.save sees only uint8, so PyTorch 2.5 can read the result.
            ids.extend(values.numpy().astype("<u2").tobytes())
            offsets.append(offsets[-1] + len(values))
            group_ids.append(groups[key])
            if identifier is not None:
                identifier_bytes.extend(identifier.encode("utf-8"))
            identifier_offsets.append(len(identifier_bytes))
            group["rows"] += 1
            group["tokens"] += len(values)
            manifest["rows"] += 1
            manifest["tokens"] += len(values)
            manifest["max_token_id"] = max(manifest["max_token_id"], high)
        pending.clear()

    try:
        for record in records:
            _record_metadata(record)
            pending.append(record)
            if len(pending) == tokenize_batch_size or (limit is not None and manifest["rows"] + len(pending) >= limit):
                tokenize_pending()
            if limit is not None and manifest["rows"] >= limit:
                break
        tokenize_pending()
        flush_shard()
        if not manifest["rows"]:
            raise ValueError("Cannot publish an empty pretokenized dataset")
        # Streaming producers may update read/skip counters while yielding rows.
        manifest["provenance"] = _json_copy(provenance or {})
        manifest["complete"] = True
        # The descriptor is the last completion marker, not a corpus tensor.
        torch.save(manifest, partial)
        if output.exists() or directory.exists():
            raise FileExistsError("Output appeared during preparation; refusing to replace it")
        staging.replace(directory)
        published_directory = True
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging)
        if published_directory and not output.exists():
            shutil.rmtree(directory)
        raise
    print(f"Prepared {output}: {manifest['rows']:,} texts; no embeddings or padding saved", flush=True)
    return manifest


class PretokenizedDataset:
    """Random-access records backed by memory-mapped, weights-only tensor shards."""

    def __init__(self, path, expected_tokenization=None, *, max_open_shards=2):
        _integer(max_open_shards, "max_open_shards", 1)
        self.path = Path(path).resolve()
        self.max_open_shards = max_open_shards
        self._cache = OrderedDict()
        self._validated = set()
        before = self.path.stat()
        self.manifest = torch.load(self.path, map_location="cpu", weights_only=True)
        after = self.path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Pretokenized descriptor changed while loading")
        self._descriptor_stat = (after.st_size, after.st_mtime_ns)
        m = self.manifest
        if not isinstance(m, dict) or m.get("format") != FORMAT or m.get("version") != VERSION or m.get("complete") is not True:
            raise ValueError("Expected a completed SlideOn pretokenized .pt descriptor")
        if expected_tokenization is not None and m.get("tokenization") != expected_tokenization:
            raise ValueError("Pretokenized tokenizer, prompts or length settings differ; rebuild inputs with matching settings")
        if m.get("dtype") != "uint16-le":
            raise ValueError("Unsupported pretokenized token dtype")
        if not isinstance(m.get("tokenization"), dict) or not m["tokenization"]:
            raise ValueError("Missing pretokenized tokenizer metadata")
        _integer(m.get("max_token_id"), "max_token_id")
        if m["max_token_id"] > 65535:
            raise ValueError("Pretokenized max_token_id does not fit uint16")
        _check_token_range(m["max_token_id"], m["tokenization"])
        self.groups = m.get("groups")
        if not isinstance(self.groups, list) or not self.groups:
            raise ValueError("Pretokenized descriptor has no source/role groups")
        for group in self.groups:
            _record_metadata(group)
            _integer(group.get("rows"), "group rows", 1)
            _integer(group.get("tokens"), "group tokens", 1)
        self.shards, self.ends = [], []
        rows, tokens = 0, 0
        seen = set()
        for shard in m.get("shards", []):
            name = shard.get("file")
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid pretokenized shard path")
            shard_path = (self.path.parent / name).resolve()
            if not shard_path.is_relative_to(self.path.parent) or shard_path in seen:
                raise ValueError("Pretokenized shard paths must be unique and inside the dataset directory")
            seen.add(shard_path)
            _integer(shard.get("rows"), "shard rows", 1)
            _integer(shard.get("tokens"), "shard tokens", 1)
            _integer(shard.get("bytes"), "shard bytes", 1)
            if not isinstance(shard.get("sha256"), str) or len(shard["sha256"]) != 64:
                raise ValueError("Missing pretokenized shard checksum")
            if shard_path.stat().st_size != shard["bytes"]:
                raise ValueError(f"Pretokenized shard size changed: {shard_path}")
            self.shards.append(shard_path)
            rows += shard["rows"]
            tokens += shard["tokens"]
            self.ends.append(rows)
        if (not self.shards or rows != m.get("rows") or tokens != m.get("tokens") or
                rows != sum(g["rows"] for g in self.groups) or tokens != sum(g["tokens"] for g in self.groups)):
            raise ValueError("Pretokenized descriptor counts are inconsistent")
        self.rows, self.tokens = rows, tokens

    def __len__(self):
        return self.rows

    def fingerprint(self, full=True):
        """Validate full checksums for preflight, or cheap metadata on DDP startup."""
        result = _fingerprint(self.path, full)
        if (result["bytes"], result["mtime_ns"]) != self._descriptor_stat:
            raise ValueError("Pretokenized descriptor changed after loading")
        result["shards"] = []
        for path, expected in zip(self.shards, self.manifest["shards"]):
            item = _fingerprint(path, full)
            if item["bytes"] != expected["bytes"] or (full and item["sha256"] != expected["sha256"]):
                raise ValueError(f"Pretokenized shard differs from its descriptor: {path}")
            result["shards"].append(item)
        return result

    def _shard(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        data = torch.load(self.shards[index], map_location="cpu", weights_only=True, mmap=True)
        if index not in self._validated:
            self._validate_shard(data, index)
            self._validated.add(index)
        self._cache[index] = data
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return data

    def _validate_shard(self, data, index):
        # Inspect offsets once, not on every mmap reopen during random sampling.
        expected = self.manifest["shards"][index]
        fields = {"input_id_bytes": torch.uint8, "offsets": torch.int64, "group_ids": torch.int32,
                  "id_bytes": torch.uint8, "id_offsets": torch.int64}
        if not isinstance(data, dict) or set(data) != set(fields):
            raise ValueError("Malformed pretokenized tensor shard")
        if any(not isinstance(data[k], torch.Tensor) or data[k].ndim != 1 or data[k].dtype != dtype
               for k, dtype in fields.items()):
            raise ValueError("Invalid pretokenized shard tensor dimensions or dtypes")
        offsets, id_offsets = data["offsets"], data["id_offsets"]
        if (len(data["input_id_bytes"]) != 2 * expected["tokens"] or len(data["group_ids"]) != expected["rows"] or
                len(offsets) != expected["rows"] + 1 or len(id_offsets) != expected["rows"] + 1 or
                int(offsets[0]) != 0 or int(offsets[-1]) != expected["tokens"] or
                bool((offsets[1:] <= offsets[:-1]).any()) or int(id_offsets[0]) != 0 or
                int(id_offsets[-1]) != len(data["id_bytes"]) or bool((id_offsets[1:] < id_offsets[:-1]).any()) or
                int(data["group_ids"].min()) < 0 or int(data["group_ids"].max()) >= len(self.groups)):
            raise ValueError("Invalid pretokenized document offsets or source groups")

    def record(self, index):
        _integer(index, "record index")
        if index >= self.rows:
            raise IndexError(index)
        shard_index = bisect_right(self.ends, index)
        row = index - (self.ends[shard_index - 1] if shard_index else 0)
        data = self._shard(shard_index)
        group = self.groups[int(data["group_ids"][row])]
        start, end = int(data["offsets"][row]), int(data["offsets"][row + 1])
        pairs = data["input_id_bytes"][2 * start:2 * end].reshape(-1, 2).long()
        token_ids = pairs[:, 0] | (pairs[:, 1] << 8)
        record = dict(source=group["source"], role=group["role"], input_ids=token_ids)
        start_id, end_id = int(data["id_offsets"][row]), int(data["id_offsets"][row + 1])
        if end_id > start_id:
            record["id"] = bytes(data["id_bytes"][start_id:end_id].tolist()).decode("utf-8")
        return record

    def iter_index(self):
        """Yield compact row metadata without tokenizing or reading raw texts."""
        base = 0
        for shard_index, expected in enumerate(self.manifest["shards"]):
            data = self._shard(shard_index)
            for row in range(expected["rows"]):
                group = self.groups[int(data["group_ids"][row])]
                count = int(data["offsets"][row + 1] - data["offsets"][row])
                yield base + row, group["source"], group["role"], count
            base += expected["rows"]

    def close(self):
        self._cache.clear()
